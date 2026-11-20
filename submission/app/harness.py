#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contest-harness 1.0.2 固定入口（v2 P0）。

平台推荐用法（本文件对应其中的 `/app/harness.py`）：

    python3 /app/harness.py describe
    python3 /app/harness.py train
    python3 /app/harness.py predict

约束（沿用 SDK 1.0.2 说明）：
- `/app` 只读，因此先关掉字节码写入再导入选手模块。
- stdout 只写协议 JSONL，其余一律 stderr。
- predict 逐行读、算完立刻写并 flush。
- predict 是**总输出**：每个非空输入行必须对应一行输出。解析不了的行不在 harness 拒答，
  交给选手模块走预注册的声明默认（平台缺一行就整批作废，报错停等于自己废掉整批）。
- 输出形状的 task 优先级是**上下文 > 行 > 产物**：上下文与行由平台生成且被 SDK 绑死，
  产物是我们自己挂的，只在两者都缺失时兜底（stderr 标注）。上下文与产物冲突不再整批
  fail-fast：该行按声明默认输出，其余行继续处理。
- predict 结束时的行数核对是**事后自检**，不是保护措施：一旦不等，整批已经作废，
  这里只把"总输出确实被破坏"留在 stderr，便于事后对账。

退出码：
    0  正常
    1  契约/内部错误（train 先 emit round_failed；stderr 留痕）
    2  未知命令（只给这一种情况）
    3  train 未实现（P0 只留接口）
"""

import sys

# `/app` 只读：必须在导入选手模块之前生效，否则会在 /app 下写 __pycache__。
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

try:
    import fact_probe  # 只写 stderr 的事实探针；任何异常都在模块内吞掉
except BaseException:  # noqa: BLE001 - 探针不可用不能影响协议
    fact_probe = None

PROTOCOL = "contest.harness.workspace.v2"
MODEL_ACCESS = "host_gateway_rpc"
ALLOWED_TASKS = ("classification", "regression")
DEFAULT_MODULE = "/app/contest_submission.py"

EXIT_OK = 0
EXIT_INTERNAL = 1   # 契约/内部错误（train 会先 emit round_failed）
EXIT_PROTOCOL = 2   # 只留给未知命令
EXIT_NOT_IMPLEMENTED = 3


class ProtocolError(Exception):
    """输入或环境不符合协议。harness 侧判定，转成退出码 2。"""


class _BadLineUtf8(ProtocolError):
    """某一行不是合法 UTF-8（逐行解码后的行内错误）。

    train 侧：只影响该帧，由 `cmd_train` 回成 `HOST_FRAME_INVALID` 的 round_failed，
    不会连带丢掉同一缓冲区里已解码成功的帧。
    predict 侧不在 harness 抛出：逐行捕获后构造 `Sample.invalid`，照常走声明默认。
    """

    def __init__(self, lineno, exc):
        detail = "第 %d 行不是合法 UTF-8：%s" % (lineno, exc)
        ProtocolError.__init__(self, detail)
        self.code = "HOST_FRAME_INVALID"
        self.detail = detail


class _ByteLineReader(object):
    """stdin 的字节行读取器，额外支持"下一行是否已可读"的非阻塞探测。

    读语义与逐行解码修正保持不变：坏字节只影响该行。额外的 `ready()` 只回答
    "写完上一行之后，下一行是否已经在手里"，不消费整行，供事实探针判断宿主
    是逐行请求-响应还是流水线预写。任何异常都由调用方降级，不影响 predict。
    """

    def __init__(self, binary):
        self._binary = binary
        self._buf = bytearray()
        self._eof = False
        self._fd = None
        try:
            self._fd = binary.fileno()
        except BaseException:  # noqa: BLE001 - 没有 fd（本地替身流）就走退路
            self._fd = None

    def readline(self):
        while True:
            index = self._buf.find(b"\n")
            if index >= 0:
                end = index + 1
                line = bytes(self._buf[:end])
                del self._buf[:end]
                return line
            if self._eof:
                if self._buf:
                    line = bytes(self._buf)
                    self._buf.clear()
                    return line
                return b""
            self._fill()

    def _fill(self):
        if self._fd is None:
            chunk = self._binary.readline()
            if not chunk:
                self._eof = True
                return
            self._buf.extend(chunk)
            return
        try:
            chunk = os.read(self._fd, 65536)
        except BlockingIOError:
            time.sleep(0.001)
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._eof = True
            return
        self._buf.extend(chunk)

    def ready(self):
        """非阻塞探测：下一行是否已可读。True/False；探不了就抛给调用方降级。"""
        if self._buf:
            return True
        if self._eof:
            return False
        if self._fd is None:
            raise RuntimeError("stdin 没有文件描述符")
        if os.name == "nt":
            return self._ready_windows()
        previous = os.get_blocking(self._fd)
        os.set_blocking(self._fd, False)
        try:
            chunk = os.read(self._fd, 65536)
        except BlockingIOError:
            return False
        finally:
            os.set_blocking(self._fd, previous)
        if not chunk:
            self._eof = True
            return False
        self._buf.extend(chunk)
        return True

    def _ready_windows(self):
        """Windows 管道用 PeekNamedPipe 探测：不消费整行，只把可用字节读进自有缓冲。"""
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(self._fd)
        available = ctypes.c_ulong(0)
        ok = ctypes.windll.kernel32.PeekNamedPipe(
            ctypes.c_void_p(handle), None, 0, None, ctypes.byref(available), None)
        if not ok:
            last = ctypes.windll.kernel32.GetLastError()
            if last in (109, 232):  # ERROR_BROKEN_PIPE / ERROR_NO_DATA：对端已关，按不可读处理
                self._eof = True
                return False
            raise OSError("PeekNamedPipe 探测失败（GetLastError=%d）" % last)
        if not available.value:
            return False
        chunk = os.read(self._fd, min(65536, int(available.value)))
        if not chunk:
            self._eof = True
            return False
        self._buf.extend(chunk)
        return True


_STDIN_READER = [None]


def _stdin_reader():
    if _STDIN_READER[0] is None:
        binary = getattr(sys.stdin, "buffer", None)
        if binary is None:
            return None
        _STDIN_READER[0] = _ByteLineReader(binary)
    return _STDIN_READER[0]


class _PerLineUtf8Stream(object):
    """把字节行读取器包成文本流外观：按字节读行、逐行解码。

    `TextIOWrapper.readline()` 是整块缓冲解码——块里混进一个坏字节，同块中
    已解码成功的行会一起消失。这里每行独立解码，坏行只让这一行的 `readline()`
    抛 `_BadLineUtf8`，前面的合法行已经交出去了。
    """

    def __init__(self, binary):
        self._binary = binary
        self._lineno = 0

    def readline(self, size=-1):
        raw = self._binary.readline()
        if not raw:
            return ""
        self._lineno += 1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _BadLineUtf8(self._lineno, exc)


def log(msg):
    sys.stderr.write("[harness] %s\n" % msg)
    sys.stderr.flush()


def _load_submission():
    path = os.environ.get("CONTEST_SUBMISSION") or DEFAULT_MODULE
    if not os.path.isfile(path):
        # 本地跑（非镜像）时回退到与本文件同目录的 contest_submission.py。
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contest_submission.py")
        if os.path.isfile(local):
            log("CONTEST_SUBMISSION=%s 不存在，回退到 %s" % (path, local))
            path = local
        else:
            raise ProtocolError("找不到选手模块：%s" % path)
    spec = importlib.util.spec_from_file_location("contest_submission", path)
    if spec is None or spec.loader is None:
        raise ProtocolError("无法从 %s 构造模块规格" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sample(object):
    """predict 的 stdin 行 + 平台上下文。字段名与 SDK 的 sample 对齐。

    `invalid` 非空表示这一行本身解析不了；不在这里拒答，交给选手模块走声明默认，
    保证"每一行都吐一行"（平台缺一行就整批作废）。
    """

    __slots__ = ("task", "inputs", "dataset_key", "artifact_id", "context", "raw_id",
                 "lineno", "invalid")

    def __init__(self, task, inputs, dataset_key=None, artifact_id=None, context=None,
                 raw_id=None, lineno=None, invalid=None):
        self.task = task
        self.inputs = inputs
        self.dataset_key = dataset_key
        self.artifact_id = artifact_id
        self.context = context
        self.raw_id = raw_id
        self.lineno = lineno
        self.invalid = invalid


def _read_prediction_context():
    """读取 CONTEST_PREDICTION_CONTEXT / CONTEST_DATASET_KEY / CONTEST_TASK。

    上下文缺失时返回空字典，调用方只用样本行里的 task（与 SDK 说明 L163 一致）。
    """
    ctx = {}
    raw = os.environ.get("CONTEST_PREDICTION_CONTEXT")
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ProtocolError("CONTEST_PREDICTION_CONTEXT 不是合法 JSON：%s" % exc)
        if not isinstance(parsed, dict):
            raise ProtocolError("CONTEST_PREDICTION_CONTEXT 必须是对象")
        ctx.update(parsed)
    for env_key, field in (("CONTEST_DATASET_KEY", "dataset_key"), ("CONTEST_TASK", "task")):
        val = os.environ.get(env_key)
        if val:
            if field in ctx and ctx[field] != val:
                raise ProtocolError("%s=%s 与 CONTEST_PREDICTION_CONTEXT.%s=%s 冲突"
                                    % (env_key, val, field, ctx[field]))
            ctx[field] = val
    if "task" in ctx and ctx["task"] not in ALLOWED_TASKS:
        raise ProtocolError("上下文 task=%r 不是 %s" % (ctx["task"], "/".join(ALLOWED_TASKS)))
    return ctx


def _invalid_sample(lineno, ctx, reason):
    """构造一行解析不了的 Sample：不在 harness 拒答，交给选手模块走声明默认。"""
    sample = Sample(
        task=None,
        inputs=None,
        dataset_key=ctx.get("dataset_key"),
        artifact_id=ctx.get("artifact_id"),
        context=ctx or None,
        lineno=lineno,
    )
    sample.invalid = reason
    return sample


def _iter_stdin_lines():
    """按字节读行、逐行解码，产出 `(lineno, text, decode_error, raw_len, arrived)`。

    坏字节只让**该行** text 为 None 并带 decode_error，既不抛到顶层，也不影响
    同一次底层读里已解码成功的其它行。空行照旧跳过且不占行号（平台行号口径）。
    本地测试若把 sys.stdin 换成没有 `.buffer` 的文本流，则退回逐行文本读取
    （那种流本身产不出坏字节）。
    """
    reader = _stdin_reader()
    lineno = 0
    while True:
        if reader is not None:
            raw = reader.readline()
            if not raw:
                return
            if not raw.strip():
                continue
            arrived = time.monotonic()
            lineno += 1
            try:
                yield lineno, raw.decode("utf-8"), None, len(raw), arrived
            except UnicodeDecodeError as exc:
                yield lineno, None, exc, len(raw), arrived
        else:
            line = sys.stdin.readline()
            if not line:
                return
            if not line.strip():
                continue
            lineno += 1
            yield lineno, line, None, len(line.encode("utf-8", "replace")), time.monotonic()


def _parse_sample_line(line, lineno, ctx):
    """把一行 stdin 解析成 Sample。行级问题只记 `invalid`，不在 harness 里拒答。"""
    sample = _invalid_sample(lineno, ctx, None)
    try:
        obj = json.loads(line)
    except ValueError as exc:
        sample.invalid = "第 %d 行不是合法 JSON：%s" % (lineno, exc)
        return sample
    if not isinstance(obj, dict):
        sample.invalid = "第 %d 行不是 JSON 对象" % lineno
        return sample

    sample.raw_id = obj.get("id")
    task = obj.get("task")
    if task is not None:
        if task not in ALLOWED_TASKS:
            sample.invalid = "第 %d 行 task=%r 不是 %s" % (lineno, task, "/".join(ALLOWED_TASKS))
        else:
            sample.task = task
    inputs = obj.get("inputs")
    if isinstance(inputs, dict):
        sample.inputs = inputs
    elif sample.invalid is None:
        sample.invalid = "第 %d 行 inputs 必须是对象" % lineno
    return sample


def _check_outputs(payload, batch_kind):
    """结构守卫。形状由选手模块按 上下文 > 行 > 产物 解析出的 task 决定，
    harness 只认形状本身，并要求同一批内形状一致。返回 (kind, {"outputs": payload})。
    """
    if isinstance(payload, dict):
        if set(payload.keys()) != {"value"}:
            raise ProtocolError("regression 输出必须形如 {'value': x}，实际为 %r" % (payload,))
        kind = "regression"
    elif isinstance(payload, list) and payload:
        labels = []
        for item in payload:
            if not isinstance(item, dict) or set(item.keys()) != {"label", "score"}:
                raise ProtocolError("分类条目必须形如 {'label','score'}，实际为 %r" % (item,))
            label = item["label"]
            if not isinstance(label, str) or not label:
                raise ProtocolError("label 必须是非空字符串")
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ProtocolError("label 出现重复：%r" % (labels,))
        kind = "classification"
    else:
        raise ProtocolError("predict 返回值既不是 {'value':x} 也不是非空 label 列表：%r" % (payload,))

    if batch_kind is not None and batch_kind != kind:
        raise ProtocolError("同一批输出形状不一致：前面是 %s，这一行是 %s" % (batch_kind, kind))
    return kind, {"outputs": payload}


def cmd_describe(argv):
    if argv:
        raise ProtocolError("describe 不接受参数")
    module = _load_submission()
    if not hasattr(module, "train"):
        raise ProtocolError("选手模块缺少 train")
    if not hasattr(module, "predict"):
        raise ProtocolError("选手模块缺少 predict")
    for name in ("describe", "PROTOCOL"):
        if not hasattr(module, name):
            raise ProtocolError("选手模块缺少 %s" % name)

    info = module.describe()
    if not isinstance(info, dict):
        raise ProtocolError("describe() 必须返回对象")
    if info.get("protocol") != PROTOCOL:
        raise ProtocolError("protocol=%r，期望 %s" % (info.get("protocol"), PROTOCOL))
    if not info.get("model_access"):
        raise ProtocolError("缺 model_access")
    tasks = info.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ProtocolError("tasks 必须是非空列表")
    unknown = [t for t in tasks if t not in ALLOWED_TASKS]
    if unknown:
        raise ProtocolError("tasks 含未支持的取值：%r" % (unknown,))
    if len(set(tasks)) != len(tasks):
        raise ProtocolError("tasks 有重复：%r" % (tasks,))

    out = {"protocol": PROTOCOL, "model_access": info["model_access"], "tasks": list(tasks)}
    if isinstance(info.get("classes"), dict):
        out["classes"] = info["classes"]
    _emit(out)
    return EXIT_OK


def cmd_train(argv):
    """真机入口：由 SDK 的 `Round.start` 从 stdin 读 start 帧，再把 rnd 交给选手模块。

    镜像外没有 SDK 时保持旧的 exit 3（本地单测用 rnd 替身驱动整轮）。
    """
    if argv:
        raise ProtocolError("train 不接受参数")
    module = _load_submission()
    try:
        from contest_harness.protocol import emit, isolate_stdio
        from contest_harness.round import Round
    except ImportError:
        log("train：本地 harness 没有 SDK 的 start 帧，无法构造 rnd；"
            "模块内 train(rnd) 已由 train_round.run 实现，真机由 SDK 宿主调用。"
            "本地验收见 tests/test_p2.py 的 rnd 替身。")
        return EXIT_NOT_IMPLEMENTED
    if not hasattr(module, "train"):
        raise ProtocolError("选手模块缺少 train(rnd)")

    # SDK 的 read_frame 只用文本流 readline()。整块缓冲解码会在坏字节处连带丢掉
    # 同缓冲区里已解码成功的帧；换成按字节读行、逐行解码，坏帧只抛 _BadLineUtf8。
    reader = _stdin_reader()
    if reader is not None:
        sys.stdin = _PerLineUtf8Stream(reader)

    # 协议帧走 stdout 的私有副本；选手模块的 print 被引到 stderr，不会污染协议。
    reader, writer = isolate_stdio()
    if fact_probe is not None:
        fact_probe.new_train_probe()
    try:
        rnd = Round.start(reader, writer)
        if fact_probe is not None:
            fact_probe.train_start(rnd)
        module.train(rnd)
        if not getattr(rnd, "_finished", False):
            raise ProtocolError("选手模块 train(rnd) 返回时本轮没有 finish")
    except _BadLineUtf8 as exc:
        # 坏帧按协议的 HOST_FRAME_INVALID 回，不再冒成未分类内部异常。
        log("train 帧解码失败（%s），按 %s 回 round_failed；此前已解码成功的帧不受影响"
            % (exc.detail, exc.code))
        try:
            emit({"type": "round_failed", "error": exc.code}, writer)
        except Exception:  # noqa: BLE001 - 宿主已断开时不再抬高
            pass
        return EXIT_INTERNAL
    except BaseException as exc:  # noqa: BLE001 - 任何异常都要回 round_failed，否则宿主等不到结束
        code = getattr(exc, "code", None) or "PARTICIPANT_EXCEPTION"
        detail = getattr(exc, "detail", None) or ("%s: %s" % (type(exc).__name__, exc))
        log("train 失败（%s）：%s" % (code, detail))
        traceback.print_exc(file=sys.stderr)
        try:
            emit({"type": "round_failed", "error": code}, writer)
        except Exception:  # noqa: BLE001 - 宿主已断开时不再抬高
            pass
        return EXIT_INTERNAL
    finally:
        if fact_probe is not None:
            fact_probe.train_summary()
    return EXIT_OK


def cmd_predict(argv):
    if argv:
        raise ProtocolError("predict 不接受参数")
    module = _load_submission()
    if not hasattr(module, "predict"):
        raise ProtocolError("选手模块缺少 predict")
    model_dir = os.environ.get("CONTEST_MODEL_DIR") or "/model"
    ctx = _read_prediction_context()
    log("predict 启动：model_dir=%s artifact=%s dataset_key=%s task=%s"
        % (model_dir, os.environ.get("CONTEST_ARTIFACT_PATH"), ctx.get("dataset_key"), ctx.get("task")))

    probe = fact_probe.new_predict_probe() if fact_probe is not None else None
    if probe is not None:
        probe.context(ctx)
    reader = _stdin_reader()
    prev_arrival = None

    lineno = 0
    written = 0
    batch_kind = None
    for lineno, line, decode_error, raw_len, arrived in _iter_stdin_lines():
        dt_ms = 0.0 if prev_arrival is None else (arrived - prev_arrival) * 1000.0
        prev_arrival = arrived
        if decode_error is not None:
            # 解码失败只影响本行：构造 invalid Sample 走声明默认，同块其它行照常出站。
            log("第 %d 行不是合法 UTF-8（%s），该行走声明默认，同块其它行不受影响"
                % (lineno, decode_error))
            sample = _invalid_sample(lineno, ctx, "第 %d 行不是合法 UTF-8：%s"
                                     % (lineno, decode_error))
        else:
            sample = _parse_sample_line(line, lineno, ctx)
        if probe is not None:
            if lineno == 1 and decode_error is None:
                probe.first_line(line, ctx)
            probe.line(lineno, raw_len, dt_ms, sample.inputs, decode_error)
        payload = module.predict(sample, model_dir)
        if probe is not None:
            probe.output(payload)
        batch_kind, wrapped = _check_outputs(payload, batch_kind)
        _emit(wrapped)
        written += 1
        if probe is not None:
            try:
                ready = reader.ready() if reader is not None else None
            except BaseException:  # noqa: BLE001 - 探测失败只降级为 unknown
                ready = None
            probe.after_write(ready)
    log("predict 结束：读入 %d 行，写出 %d 行" % (lineno, written))
    if probe is not None:
        probe.summary()
    if written != lineno:
        # 事后自检，不是保护措施：走到这里整批已经因为缺行作废了，
        # 这里只把"总输出确实被破坏"这件事留在 stderr，便于事后对账。
        raise ProtocolError("事后自检：读入 %d 行只写出 %d 行（整批已作废，这不是保护措施）"
                            % (lineno, written))
    return EXIT_OK


def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


COMMANDS = {"describe": cmd_describe, "train": cmd_train, "predict": cmd_predict}


def main(argv):
    if len(argv) < 2 or argv[1] not in COMMANDS:
        log("用法：harness.py {describe|train|predict}，实际收到 %r" % (argv[1:],))
        return EXIT_PROTOCOL
    try:
        return COMMANDS[argv[1]](argv[2:])
    except ProtocolError as exc:
        log("协议错误：%s" % exc)
        return EXIT_INTERNAL
    except Exception as exc:
        if type(exc).__name__ == "ContractError":
            # 检查点或样本不满足选手模块声明的契约，按协议错误处理，不吞成内部异常。
            log("契约错误：%s" % exc)
            return EXIT_INTERNAL
        log("内部异常：\n%s" % traceback.format_exc())
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main(sys.argv))

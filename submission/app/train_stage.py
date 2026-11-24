#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train 阶段的检测与选型入口（v2 P1）。

这是**唯一**会跑批级检测器（KL / 马氏 / BBSD）的地方：读 `/data/train.jsonl` 与
`/data/validation.jsonl`，在整批上算漂移，再由 selector 决定本轮造哪个候选。
predict 侧不导入本模块，也不导入 `detectors`。

P2 接 SDK 训练循环时，第一件事就是调 `run()`，把返回的 `candidate` 与 `selector_trace`
带进 `rnd.submit_candidate(...)` 的 `hypothesis` / `change_summary`；`selector_trace`
同时落盘，作为本轮选择的审计记录。
"""

import collections
import json
import math
import os
import sys

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import detectors  # noqa: E402
import selector  # noqa: E402
import detector_lib  # noqa: E402

DATA_DIR = "/data"
TRAIN_NAME = "train.jsonl"
VAL_NAME = "validation.jsonl"
TRACE_NAME = "selector_trace.json"
GROUP_KEYS = ("settings", "condition", "conditions", "group")
# 路径字符串只由结构决定（与数值无关），同一份数据每行都重复同一组路径。
# 缓存后每轮只需 json.dumps 特征数那么多次，不再对每个数值各做一次。
_PATH_TEXT_CACHE = {}


def _labels(rows):
    """分类标签：优先 `label`，其次 `answer`（A 线的类别就写在 answer 里）。

    只接受字符串标签：C/B 的 `answer` 是连续 RUL，不能当类别喂给聚类类检测器，
    否则会把每台机组的 RUL 当成一个新类。任一行缺失就整体返回 None，不拿部分标签硬算。
    """
    if not rows:
        return None
    values = []
    for row in rows:
        value = row.get("label")
        if value in (None, ""):
            value = row.get("answer")
        if isinstance(value, bool) or value in (None, ""):
            return None
        values.append(value)
    if not all(isinstance(value, str) for value in values):
        return None
    return values


def _row_field(row, key):
    """按 顶层 → meta → inputs 三级找字段；平台把 unit/cycle/settings 放在后两层。"""
    for scope in (row, row.get("meta"), row.get("inputs")):
        if isinstance(scope, dict):
            value = scope.get(key)
            if value not in (None, ""):
                return value
    return None


def _meta(rows):
    """可选的单元/快照/工况键；缺哪项就不提供哪项，检测器侧会显式 unavailable。"""
    meta = {}
    for key in ("unit", "snapshot"):
        values = []
        for row in rows:
            value = _row_field(row, key)
            if value is None and key == "snapshot":
                # FD 系列没有 snapshot 这个名字，时间序号写作 cycle。
                value = _row_field(row, "cycle")
            if value is None:
                values = []
                break
            values.append(value)
        if values:
            meta[key] = values
    for key in GROUP_KEYS:
        values = []
        for row in rows:
            value = _row_field(row, key)
            if value is None:
                values = []
                break
            values.append(str(value))
        if values:
            meta["cond_key"] = values
            break
    return meta or None



class TrainStageError(Exception):
    """train 阶段输入不满足要求。"""


def _log(msg):
    sys.stderr.write("[train_stage] %s\n" % msg)
    sys.stderr.flush()


_JSONL_CACHE = collections.OrderedDict()
_JSONL_CACHE_MAX = 4
_MATRIX_CACHE = collections.OrderedDict()
_MATRIX_CACHE_MAX = 4


def load_jsonl(path):
    """按文件顺序读 JSONL，返回 (行列表, 跳过行数)。坏行不猜，直接计数跳过。

    C 线性能：同一轮里 train_stage.run / _slice_card / train_mlp 会各要一次
    同一份 train/validation。这里按 (路径, mtime, size) 做进程内缓存，三处共用
    同一份行对象；读数与坏行计数逐位不变。行对象只读，调用方不得原地改写。
    """
    if not os.path.isfile(path):
        raise TrainStageError("找不到数据文件：%s" % path)
    key = file_key(path)
    hit = _JSONL_CACHE.get(key)
    if hit is not None:
        _JSONL_CACHE.move_to_end(key)
        return hit
    rows, skipped = [], 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), dict):
                skipped += 1
                continue
            rows.append(obj)
    out = (rows, skipped)
    _JSONL_CACHE[key] = out
    while len(_JSONL_CACHE) > _JSONL_CACHE_MAX:
        _JSONL_CACHE.popitem(last=False)
    return out


def file_key(path):
    """文件身份键：绝对路径 + 时间戳 + 大小。只用于进程内结果缓存，不改任何读数。"""
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def _flatten_inputs(inputs):
    """按官方 `data.flatten` 的规则展开嵌套 inputs：对象键字典序、数组保持顺序。

    平台真实数据的 inputs 不是"一组标量"，而是嵌套结构：
        A: {"features": [128 个数值]}
        B: {"features": {"horizontal": {...16...}, "vertical": {...16...}}}
        C: {"settings": [3 个数值], "sensors": [21 个数值]}
    旧实现只把顶层键当列、再对整值 `float()`，于是每一行都因 TypeError 被丢成无效行，
    三线第 1 轮全部在"有效行不足"处失败。这里与 predict 侧共用同一套展开口径。
    """
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("inputs 嵌套超过 16 层")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError("inputs 的键必须是字符串")
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [["index", index]], depth + 1)
        elif isinstance(value, bool):
            raise ValueError("bool 不算数值")
        elif isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("非有限数值")
            key = tuple((step[0], step[1]) for step in path)
            text = _PATH_TEXT_CACHE.get(key)
            if text is None:
                text = json.dumps(path, ensure_ascii=False, separators=(",", ":"))
                _PATH_TEXT_CACHE[key] = text
            paths.append(text)
            values.append(number)
        else:
            raise ValueError("inputs 只接受有限数值，实际为 %r" % (value,))

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("inputs 展开后应有 1..4096 个数值，实际 %d" % len(values))
    return paths, values


def _feature_order(train_rows, val_rows):
    """用第一条可展开的行钉住特征序；同一批数据的结构由平台保证一致。"""
    for row in train_rows + val_rows:
        try:
            paths, _ = _flatten_inputs(row["inputs"])
        except (TypeError, ValueError):
            continue
        return paths
    return []


def _collect_sig(value, out, depth=0, max_depth=None, numeric_only=False):
    """一次遍历同时产出结构签名与数值，替代 `_shape_of` + `_collect` 两遍走。

    签名口径与 `_flatten_inputs` 的路径口径一一对应（对象键字典序、数组记长度），
    所以"签名相同"与"paths 相同"等价；纯标量数组走整段 extend 快路径。
    - `max_depth`：矩阵路径保持 `_flatten_inputs` 的 16 层上限；
    - `numeric_only`：矩阵路径保持 `_flatten_inputs` 只认 int/float 的口径
      （旧 `_collect` 走 float() 会接受 "3" 这类字符串，两条旧口径本就不同，这里分开保留）。
    """
    if max_depth is not None and depth > max_depth:
        raise ValueError("inputs 嵌套超过 16 层")
    if isinstance(value, dict):
        signature = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("inputs 的键必须是字符串")
            signature.append((key, _collect_sig(value[key], out, depth + 1,
                                                max_depth, numeric_only)))
        return tuple(signature)
    if isinstance(value, list):
        count = len(value)
        scalar = True
        for child in value:
            kind = type(child)
            if kind is float:
                if not math.isfinite(child):
                    raise ValueError("非有限数值")
            elif kind is int:
                if not -1e308 < child < 1e308:
                    raise ValueError("数值超出 float 可表示范围")
            else:
                scalar = False
                break
        if scalar:
            out.extend(value)
        else:
            for child in value:
                _collect_sig(child, out, depth + 1, max_depth, numeric_only)
        return ("#", count)
    if isinstance(value, bool):
        raise ValueError("bool 不算数值")
    if numeric_only and not isinstance(value, (int, float)):
        raise ValueError("inputs 只接受数值，实际为 %r" % (value,))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("inputs 只接受有限数值，实际为 %r" % (value,))
    except OverflowError:
        raise ValueError("数值超出 float 可表示范围")
    if not math.isfinite(number):
        raise ValueError("非有限数值")
    out.append(number)
    return None


def _matrix(rows, order):
    """按统一特征序展开数值矩阵；结构不一致/非数值/非有限的行整行跳过并计数。

    实现口径与旧版 `_flatten_inputs(...) == order` 完全等价：结构签名相同 ⟺
    路径列表相同。第一条命中行仍用慢路径对齐 `order`（只做一次），其后逐行比签名，
    省掉每行构造路径字符串的开销（C 线实测 11.5s → 1.7s）。

    C 线性能：validation 矩阵在 train_stage.run 与 _slice_card 里各要一次；行对象
    由 `load_jsonl` 缓存共享，这里按 (行对象身份, 特征序) 复用同一份矩阵。返回数组
    只读，调用方不得原地改写。
    """
    key = (id(rows), len(rows), len(order), tuple(order))
    hit = _MATRIX_CACHE.get(key)
    if hit is not None and hit["rows"] is rows:
        _MATRIX_CACHE.move_to_end(key)
        return hit["matrix"], hit["skipped"]
    matrix, skipped = _matrix_uncached(rows, order)
    _MATRIX_CACHE[key] = {"rows": rows, "matrix": matrix, "skipped": skipped}
    while len(_MATRIX_CACHE) > _MATRIX_CACHE_MAX:
        _MATRIX_CACHE.popitem(last=False)
    return matrix, skipped


def _matrix_uncached(rows, order):
    import numpy as np

    kept, skipped = [], 0
    target = None
    for row in rows:
        values = []
        try:
            signature = _collect_sig(row["inputs"], values, 0, 16, True)
        except (TypeError, ValueError, OverflowError):
            skipped += 1
            continue
        if target is None:
            try:
                paths, _values = _flatten_inputs(row["inputs"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            if paths != order:
                skipped += 1
                continue
            target = signature
        elif signature != target or len(values) != len(order):
            skipped += 1
            continue
        kept.append(values)
    return np.asarray(kept, dtype=float).reshape(len(kept), len(order)), skipped


def _has_group(train_rows, val_rows):
    for row in train_rows + val_rows:
        if any(key in row["inputs"] for key in GROUP_KEYS):
            return True
    return False


def _trace_dir(out_dir):
    """trace 一定要落盘：显式目录 > CONTEST_TRACE_DIR > /workspace > 当前目录。"""
    if out_dir:
        return out_dir
    env = os.environ.get("CONTEST_TRACE_DIR")
    if env:
        return env
    if os.path.isdir("/workspace"):
        return "/workspace"
    return os.getcwd()


def run(data_dir=None, out_dir=None, task=None):
    """跑一轮检测 + 选择。返回 {candidate, action, detector_report, selector_trace, trace_path}。"""
    data_dir = data_dir or os.environ.get("CONTEST_DATA_DIR") or DATA_DIR
    train_rows, train_skipped = load_jsonl(os.path.join(data_dir, TRAIN_NAME))
    val_rows, val_skipped = load_jsonl(os.path.join(data_dir, VAL_NAME))

    order = _feature_order(train_rows, val_rows)
    if not order:
        raise TrainStageError("train/validation 里没有任何特征列")
    train_x, train_dropped = _matrix(train_rows, order)
    val_x, val_dropped = _matrix(val_rows, order)
    if train_x.shape[0] < 2 or val_x.shape[0] < 2:
        raise TrainStageError("有效行不足：train=%d val=%d（检测器至少各需 2 行）"
                              % (train_x.shape[0], val_x.shape[0]))

    report = detectors.run_all(train_x, val_x, stage="train")
    # v1 检测器库（19 条，三态）：只做观测与落账，不参与 select；
    # report["detectors"] 的三个数保持逐位不变。
    report["detector_registry"] = detector_lib.run(
        train_x, val_x, stage="train",
        labels_train=_labels(train_rows), labels_val=_labels(val_rows),
        meta_train=_meta(train_rows), meta_val=_meta(val_rows))
    # 账本只存压缩视图：configuration 有 8 KiB 预算，完整读数留在 detector_report 里。
    report["detector_registry_ledger"] = detector_lib.ledger_view(
        report["detector_registry"])
    chosen, trace = selector.select(report, has_group=_has_group(train_rows, val_rows))
    trace["inputs"].update({
        "data_dir": os.path.abspath(data_dir),
        "task": task,
        "feature_order": order,
        "skipped": {"train_bad_json": train_skipped, "val_bad_json": val_skipped,
                    "train_non_numeric": train_dropped, "val_non_numeric": val_dropped},
    })

    trace_path = os.path.join(_trace_dir(out_dir), TRACE_NAME)
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    _log("检测完成：kl=%.6g mahalanobis=%.6g bbsd=%.6g → 候选 %s（%s）"
         % (report["detectors"]["kl"]["value"],
            report["detectors"]["mahalanobis"]["value"],
            report["detectors"]["bbsd"]["value"],
            chosen, selector.CANDIDATES[chosen]["action"]))

    return {
        "candidate": chosen,
        "action": selector.CANDIDATES[chosen]["action"],
        "detector_report": report,
        "selector_trace": trace,
        "trace_path": trace_path,
    }

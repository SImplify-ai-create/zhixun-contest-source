#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：把 SDK 训练循环接上（一轮 = 一次 `train(rnd)`）。

固定顺序，任何一步失败都不提交半成品：

1. `rnd.parent.path(...)` 打开父检查点并**自己再算一遍 sha256** 与 `parent.checkpoint_sha256` 对账；
   不扫 `/app`、不改 `/previous`、不另造 baseline 信息。
2. 读 `/data/train.jsonl` + `/data/validation.jsonl`（或 `rnd.task["public_projection"]` 给出的路径），
   跑 P1 的批级检测与选型，得到候选建议与 `selector_trace`。
3. `rnd.history()` 作为经验库读取入口，把记录摘要进提案上下文，并算 `history_digest`。
4. 调 `rnd.model()` **一次**（失败时最多加一次修复调用），显式 `max_tokens`，不重复注入 `rnd.prompt`。
   提案先过 `llm_cage`：闭集臂、参数区间、引用、数字纪律、阶段门；再在 `/workspace` 沙盒构建候选，
   跑性质测试与红队五问，全过才允许 `submit_candidate`。失败把校验原文回灌一次；仍失败退回规则选择器。
5. `rnd.submit_candidate(...)`，读回 `artifact_id` 与平台验证摘要。
6. 复制 `/state` 到 `state_workspace` 后写入本轮状态（含回滚指针与各臂统计），`rnd.save_state(...)`。
7. `rnd.finish(selected_artifact_id=..., model_call_ids=rnd.call_ids, history_ids=...)`。

回合策略的悲观界（P2 加固）：

- 每个臂（PARENT / A / B / C）在 `round_state.json` 里记后验统计 `n / mean / m2`；
- 候选臂的悲观界 = `mean - k * sd`（sd 为样本标准差）。悲观界**低于父候选**的臂一律不采纳，
  直接退回父候选；父候选或该臂尚无观测时不做比较（不把没试过的臂提前判死）。
- 某臂提交失败一轮：`failures += 1`、`cooldown = 1`，本轮立即改用父候选重提，下一轮继续跳过它。
- `k` 是配置项（`PESSIMISM["k"]`，可用 `run(..., pessimism_k=...)` 或 `CONTEST_PESSIMISM_K` 覆盖），
  默认 1.0：这是 bandit 里“均值低一个后验标准差”的标准悲观估计，与分数尺度无关
  （accuracy / 负 RMSE 都适用）；k=0 退化为只比均值，无法体现“样本少、方差大”的臂更该被怀疑。

账本五字段与原有判据的对应：
    hypothesis       ← 本轮**实际提交候选**的漂移假设（PARENT 时为"沿用父检查点"）
    change_summary   ← 相对父模型实际改了什么（权重/动作）
    configuration    ← 检测器数值、选型规则、模型调号、history_digest、seed
    reason           ← **只**引用平台返回的数字（submit 回执里的验证摘要）
    state            ← 跨轮状态（上一轮 artifact_id 作为回滚指针）

可复现性：同一份 history + 同一份数据 + 同一个 `round_index`，提案与账本必须一致；代码里不写时间戳，
候选目录名由内容摘要决定，模型调用固定 `temperature=0`。
"""

import collections
import hashlib
import json
import math
import os
import shutil
import sys

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import selector  # noqa: E402
import train_stage  # noqa: E402
import train_mlp  # noqa: E402
import llm_cage  # noqa: E402
import bias_family  # noqa: E402
import failure_modes  # noqa: E402
import actions  # noqa: E402  输入动作（causal/transductive 门禁与逐行 step）
from v6alpha import alpha_lib as _alpha_lib  # noqa: E402  v6 纯函数库（原样复用）
from v6alpha import grid_serialize as _grid_serialize  # noqa: E402

try:
    import fact_probe  # 只写 stderr 的事实探针；异常都在模块内吞掉
except BaseException:  # noqa: BLE001 - 探针不可用不能影响训练
    fact_probe = None

PARENT = "PARENT"
CANDIDATE_IDS = ("A", "B", "C")
# 官方基线检查点的键。平台给的父产物是官方 MLP（无 task/coef），v2 动作臂套不上去。
OFFICIAL_KEYS = ("w1", "b1", "w2", "b2", "mean", "scale")
STATE_FILE = "round_state.json"
LEDGER_FILE = "ledger.json"
CHECKPOINT_NAME = "checkpoint.npz"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
MAX_MODEL_CALLS = 2  # 首提 + 一次修复；不展开成多轮对话

# 悲观界配置。k 不硬编码在比较逻辑里：优先级 run(pessimism_k=...) > 环境变量 > 默认值。
PESSIMISM = {
    "k": 1.0,
    "cooldown_rounds": 1,
    "env": "CONTEST_PESSIMISM_K",
    "rationale": "k=1 是“均值低一个后验标准差”的标准悲观估计（UCB 的对称面），与分数尺度无关；"
                 "k=0 退化为只比均值，看不出样本少、方差大的臂更该被怀疑。",
}

LOWER_BETTER = ("rmse", "mae", "mse", "loss", "error")


class TrainRoundError(Exception):
    """本轮无法安全推进。宁可整轮失败，也不提交半成品。"""


def _log(msg):
    sys.stderr.write("[train_round] %s\n" % msg)
    sys.stderr.flush()


# SDK 1.0.4：RPC 发出后若被信号/超时/取消打断，宿主是否已处理未知；本轮会话作废，
# 后续 RPC 必然报 RPC_OUTCOME_UNKNOWN。这类异常不能当普通失败重试。
_INTERRUPTION_TYPES = (KeyboardInterrupt, SystemExit, TimeoutError)
_INTERRUPTION_NAMES = ("cancellederror", "cancelled", "rpcinterrupted",
                       "interruptederror", "rpcinterruption")


def _is_interruption(exc):
    """判断是否属于“RPC 被中断、结果未知”这一类，而不是平台正常返回的错误。"""
    if isinstance(exc, _INTERRUPTION_TYPES):
        return True
    name = type(exc).__name__.lower()
    if name in _INTERRUPTION_NAMES or "outcome_unknown" in name:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.upper() == "RPC_OUTCOME_UNKNOWN":
        return True
    try:
        text = ("%s %r" % (exc, exc)).upper()
    except Exception:
        text = ""
    return "RPC_OUTCOME_UNKNOWN" in text


def _is_session_dead(exc):
    """本次 RPC 之后会话是否已不可用。

    `RPC_MISMATCH`（应答 id/类型不符）本身就是作废起点：SDK 在发请求前把
    `_outcome_unknown` 置 True，只有匹配应答才清回 False，失配后该标志保持 True。
    其余中断/超时按 `_is_interruption` 判。
    """
    if _is_interruption(exc):
        return True
    code = getattr(exc, "code", None)
    return isinstance(code, str) and code.upper() == "RPC_MISMATCH"


def _rpc_guard(label, call, *args, **kwargs):
    """所有 RPC 的统一出口：一旦命中“结果未知”，立即停手。

    SDK 1.0.4 的 _rpc 在发出请求前把 _outcome_unknown 置 True，只有收到**匹配**的应答
    才清回 False；应答失配（错 id/错类型）或读流中断时该标志保持 True，后续每个 RPC
    都会直接抛 RPC_OUTCOME_UNKNOWN。此时再往下走（存状态/结束本轮）只会把报错点挪到
    更晚的位置，因此这里就地判死并落一条明确日志。
    """
    try:
        return call(*args, **kwargs)
    except BaseException as exc:
        if _is_session_dead(exc):
            _log("RPC %s 应答失配/中断（%s: %s）：按 SDK 1.0.4 本轮会话已作废，"
                 "立即停手，不再发任何后续 RPC。"
                 % (label, type(exc).__name__, str(exc)[:200]))
            raise TrainRoundError(
                "RPC %s 之后会话结果未知，本轮立即中止（未再发后续 RPC）：%s: %s"
                % (label, type(exc).__name__, str(exc)[:200])) from exc
        raise


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(path, obj):
    """统一的落盘口径：UTF-8、LF、缩进 1、键排序、末尾换行。

    所有跨轮结构（state / ledger / best 元数据）都走这一对函数，回环测试才能
    盯住"写出去和读回来是同一个东西"，而不是逐字段各测一遍。
    """
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _digest(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fspath(value):
    if isinstance(value, str):
        return value
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    for attr in ("path", "dir", "directory"):
        inner = getattr(value, attr, None)
        if isinstance(inner, str):
            return inner
    return str(value)


def _digest_hex(value):
    """统一摘要口径：SDK 声明 `sha256:<hex>`，本地夹具用裸 hex。"""
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text


def _parent_checkpoint(rnd):
    parent = getattr(rnd, "parent", None)
    if parent is None:
        raise TrainRoundError("rnd.parent 不存在，无法核对父检查点")
    rels = getattr(parent, "checkpoint_paths", None)
    if not rels:
        descriptor = getattr(parent, "descriptor", None)
        if isinstance(descriptor, dict):
            rels = descriptor.get("checkpoint_paths")
    rels = rels or [CHECKPOINT_NAME]
    if isinstance(rels, str):
        rels = [rels]
    rel = str(rels[0])
    try:
        path = _fspath(parent.path(rel))
    except Exception as exc:  # 含 PARENT_DIGEST_MISMATCH：必须整轮失败，不能绕过
        raise TrainRoundError("父检查点不可用（%s）：%s" % (rel, exc))
    actual = _sha256_file(path)
    declared = getattr(parent, "checkpoint_sha256", None)
    if declared and _digest_hex(declared) != _digest_hex(actual):
        raise TrainRoundError("父检查点摘要不符：声明 %s，实际 %s" % (declared, actual))
    if not declared:
        _log("父产物未声明摘要，仅记录实际 sha256=%s" % actual[:16])
    return {"rel": rel, "path": path, "sha256": actual, "declared": declared}


def _data_dir(rnd):
    task = getattr(rnd, "task", None)
    projection = task.get("public_projection") if isinstance(task, dict) else None
    if isinstance(projection, dict):
        for key in ("train", "train_path", "training", "path"):
            value = projection.get(key)
            if isinstance(value, str) and value:
                return os.path.dirname(value) if value.endswith(".jsonl") else value
    elif isinstance(projection, str) and projection:
        return os.path.dirname(projection) if projection.endswith(".jsonl") else projection
    return os.environ.get("CONTEST_DATA_DIR") or "/data"


def _slice_card(data_dir, dataset_key, rnd=None):
    """当轮数据切片的事实卡片：只记录，不参与训练、不参与判优。

    目的：把"平台这一轮到底送进来哪份切片"变成回执里可见的指纹与统计。检测器
    （bbsd/mahalanobis）是这两份文件的确定性函数，所以逐轮数值变化可以直接归因到
    切片本身；文件 sha256 与行数把这一点钉成可对账的证据。
    """
    card = {"dir": os.path.abspath(data_dir)}
    try:
        files = []
        for name in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, name)
            if os.path.isfile(path):
                files.append([name, os.path.getsize(path)])
        card["files"] = files
        digests = {}
        for name in (train_stage.TRAIN_NAME, train_stage.VAL_NAME):
            path = os.path.join(data_dir, name)
            if not os.path.isfile(path):
                continue
            handle = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    handle.update(chunk)
            digests[name] = handle.hexdigest()[:16]
        card["sha256_16"] = digests
        # v4-slicelog：宿主投影的两个路径原样记账（只读不切换，读的仍是 data_dir）。
        view = getattr(rnd, "data", None)
        paths = {"train": getattr(view, "train_path", None),
                 "validation": getattr(view, "validation_path", None)}
        card["paths"] = paths
        card["sha"] = {}
        card["rows"] = {}
        for name, path in paths.items():
            card["sha"][name] = None
            card["rows"][name] = None
            if not isinstance(path, str) or not os.path.isfile(path):
                continue
            handle = hashlib.sha256()
            rows = 0
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    handle.update(chunk)
                    rows += chunk.count(b"\n")
            card["sha"][name] = handle.hexdigest()[:16]
            card["rows"][name] = rows
        train_rows, _ = train_stage.load_jsonl(
            os.path.join(data_dir, train_stage.TRAIN_NAME))
        val_rows, _ = train_stage.load_jsonl(
            os.path.join(data_dir, train_stage.VAL_NAME))
        card["validation_rows"] = len(val_rows)
        answers = [row.get("answer") for row in val_rows
                   if row.get("answer") is not None]
        key = str(dataset_key or "")
        if key == "A" or (answers and isinstance(answers[0], str)):
            counts = {}
            for answer in answers:
                label = str(answer)
                counts[label] = counts.get(label, 0) + 1
            card["class_counts"] = dict(sorted(counts.items()))
        else:
            import numpy as np

            values = np.asarray([float(item) for item in answers], dtype=float)
            if values.size:
                card["rul_quantiles"] = {
                    name: round(float(value), 3) for name, value in zip(
                        ("min", "p25", "p50", "p75", "max"),
                        np.percentile(values, [0, 25, 50, 75, 100]))}
            if key == "C":
                subsets = {}
                for row in val_rows:
                    name = str((row.get("meta") or {}).get("source_subset"))
                    subsets[name] = subsets.get(name, 0) + 1
                card["subset_counts"] = dict(sorted(subsets.items()))
        order = train_stage._feature_order(train_rows, val_rows)
        matrix, _ = train_stage._matrix(val_rows, order)
        if order:
            card["feature_mean"] = [round(float(value), 3)
                                    for value in matrix.mean(axis=0)]
        if len(json.dumps(card, ensure_ascii=False)) > 8192:
            card.pop("feature_mean", None)
            card["features_dropped"] = True
    except Exception as exc:  # noqa: BLE001 - 记录块不许影响本轮
        card["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    return card


def _work_dir(rnd):
    env = os.environ.get("CONTEST_WORKSPACE")
    if env:
        return env
    return "/workspace" if os.path.isdir("/workspace") else os.getcwd()


def _history(rnd):
    try:
        items = list(_rpc_guard("history", rnd.history) or [])
    except TrainRoundError:
        raise
    except Exception as exc:
        raise TrainRoundError("rnd.history() 读取失败：%s" % exc)
    records = []
    for item in items:
        if isinstance(item, dict):
            records.append({k: item.get(k) for k in
                            ("id", "round_index", "hypothesis", "change_summary", "artifact_id")
                            if k in item})
        else:
            records.append({"id": str(item)})
    ids = [str(rec["id"]) for rec in records if rec.get("id") is not None]
    return records, ids


def _signature(dataset_key, features):
    return "%s/%s" % (dataset_key, features)


def _review_context(round_index, arms, history_ids, previous_state, signature):
    """提案校验的上下文。签名表为空时按“未见”处理会误杀，故只在有表时判定。

    例外：状态降级（文件坏了）时无法确认签名历史，按"未见"处理——宁可退回父产物，
    也不能把坏掉的 state 当成"全新环境"来放宽闸门。
    """
    degraded = bool((previous_state or {}).get("_degraded"))
    signatures = previous_state.get("signatures") if isinstance(previous_state, dict) else None
    if degraded:
        signature_seen = False
    elif isinstance(signatures, list) and signatures:
        signature_seen = signature in [str(item) for item in signatures]
    else:
        signature_seen = True
    return {
        "arms": arms,
        "history_ids": history_ids,
        "phase": llm_cage.phase_of(round_index),
        "remaining_rounds": llm_cage.remaining_rounds(round_index),
        # 数字白名单只取平台帧给出的当轮轮次号，不取模型文本里的任何数字。
        "round_index": int(round_index),
        "signature_seen": signature_seen,
        "state_degraded": degraded,
        "failed_combos": list((previous_state or {}).get("failed_combos") or []),
    }


def _model_source(rnd, receipt):
    """这笔回复是"模型"还是"桩"：账本必须如实标注，不能一律写成模型。

    优先级：回执自带的 `response.source`（本地桩可显式写 stub）→ 宿主模型名，
    含 stub/fake/mock 等字样即判桩 → 其余一律按真实模型记。
    """
    response = getattr(receipt, "response", None)
    if isinstance(response, dict):
        tag = str(response.get("source") or "").strip().lower()
        if tag in ("stub", "model"):
            return tag
    name = str(getattr(rnd, "model_name", "") or "").lower()
    if any(hint in name for hint in ("stub", "fake", "mock", "dummy",
                                     "test-double", "local-fake")):
        return "stub"
    return "model"


MODEL_MAX_TOKENS = 4096
EMPTY_RETRY_MAX_TOKENS = 8192


def _call_model(rnd, messages):
    """真实模型调用。空回复时用更大预算重试一次，两次 call_id 都保留。

    预算 4096 而不是 512：Qwen3.8 的思维链可能吃光 512，content 为 null。
    v7 显式传 enable_thinking=False：SDK 1.0.4 的 `_MODEL_OPTIONS` 白名单已含该键
    （contest_harness/round.py），不会再 TypeError；不传则线上默认思维链开，
    实测同一批 36 轮通过率 2.8% vs 关闭时 44.4%。开关值同时写进账本 configuration.model。
    """
    cap = getattr(rnd, "max_output_tokens", None)
    cap = cap if isinstance(cap, int) and cap > 0 else None
    budget = MODEL_MAX_TOKENS if cap is None else min(MODEL_MAX_TOKENS, cap)
    rpc_calls = []

    def _once(limit):
        receipt = _rpc_guard("model", rnd.model, messages,
                             max_tokens=limit, temperature=0,
                             enable_thinking=False)
        if fact_probe is not None:
            fact_probe.train_receipt("model", receipt)
        text = getattr(receipt, "text", None)
        text = "" if text is None else str(text)
        call_id = getattr(receipt, "call_id", None)
        rpc_calls.append({"call_id": call_id, "max_tokens": limit,
                          "enable_thinking": False,
                          "empty": not text.strip(),
                          "response_sha256": llm_cage.sha256_text(text)})
        return receipt, text, call_id, limit

    receipt, text, call_id, used = _once(budget)
    empty_retry = None
    if not text.strip():
        retry_budget = (EMPTY_RETRY_MAX_TOKENS if cap is None
                        else min(EMPTY_RETRY_MAX_TOKENS, cap))
        empty_retry = {"first_call_id": call_id, "first_max_tokens": budget,
                       "retry_call_id": None, "retry_max_tokens": retry_budget}
        retry_receipt, retry_text, retry_call_id, retry_used = _once(retry_budget)
        empty_retry.update({"retry_call_id": retry_call_id,
                            "retry_empty": not retry_text.strip()})
        if retry_text.strip():
            receipt, text, call_id, used = (retry_receipt, retry_text,
                                            retry_call_id, retry_used)

    call_ids = []
    for record in rpc_calls:
        cid = record.get("call_id")
        if cid and cid not in call_ids:
            call_ids.append(cid)
    for cid in list(getattr(rnd, "call_ids", []) or []):
        if cid and cid not in call_ids:
            call_ids.append(cid)
    if not call_ids:
        raise TrainRoundError("本轮没有任何真实模型调用号，不能 finish")
    return {
        "call_id": call_id,
        "call_ids": call_ids,
        "text": text,
        "max_tokens": used,
        "enable_thinking": False,
        "source": _model_source(rnd, receipt),
        "response_sha256": llm_cage.sha256_text(text),
        "rpc_calls": rpc_calls,
        "empty_retry": empty_retry,
    }


def _rpc_entries(call):
    """把一次逻辑调用展开成真实 RPC 列表；空回复重试会有两条。"""
    records = call.get("rpc_calls") or []
    if records:
        return [{"call_id": record.get("call_id"),
                 "response_sha256": record.get("response_sha256"),
                 "max_tokens": record.get("max_tokens"),
                 "empty": bool(record.get("empty"))} for record in records]
    return [{"call_id": call.get("call_id"),
             "response_sha256": call.get("response_sha256")}]


def _all_call_ids(calls, rnd=None):
    """所有真实调用号，去重且保序；空回复重试的两次都在里面。"""
    ids = []
    for item in calls:
        for entry in _rpc_entries(item):
            cid = entry.get("call_id")
            if cid and cid not in ids:
                ids.append(cid)
    for cid in list(getattr(rnd, "call_ids", []) or []):
        if cid and cid not in ids:
            ids.append(cid)
    return ids


def _candidate_dir(work, round_index, history_digest, chosen, parent_sha):
    tag = _digest({"round": round_index, "history": history_digest,
                   "chosen": chosen, "parent": parent_sha})[:8]
    path = os.path.join(work, "round-%s-%s" % (round_index, tag), "candidate")
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def _build_candidate(parent, chosen, cand_dir, params=None, data_dir=None,
                     round_index=1, dataset_key=None, history=None):
    """PARENT = 逐字节复制；官方 MLP 父产物 = 同结构梯度续训；v2 父产物 = 只改写 action。

    官方格式（w1/w2/b1/b2/mean/scale，无 task/coef）的 predict 走官方前向，不读 action。
    往这种产物里写动作只会得到"官方键 + action"的假候选，因此这里改为**真训**：
    交给 train_mlp 在同一结构上续训，产出官方格式权重（见 train_mlp 模块说明）。
    v2 格式（含 coef）才沿用"保留数组、只改写 action"的老路径。
    """
    target = os.path.join(cand_dir, CHECKPOINT_NAME)
    if chosen == "C" and dataset_key == "C":
        # C 线性能：同一轮里 C 臂在四臂评分与接管 prepare 中各拟合一次（父产物、
        # 参数、轮次完全相同）。第二次直接复制第一次的字节相同产物，不再重复拟合。
        memo_key = (parent.get("sha256"),
                    json.dumps(params or {}, sort_keys=True), int(round_index))
        hit = _C_BUILD_MEMO.get(memo_key)
        if hit is not None:
            os.makedirs(cand_dir, exist_ok=True)
            for name in os.listdir(hit["dir"]):
                source = os.path.join(hit["dir"], name)
                if os.path.isfile(source):
                    shutil.copyfile(source, os.path.join(cand_dir, name))
            result = dict(hit["built"])
            result["sha256"] = _sha256_file(target)
            return result
    if chosen == PARENT:
        shutil.copyfile(parent["path"], target)
        # 本轮根本没训练：soup 不会尝试，必须把"为什么没进 soup"写进账本，
        # 否则回执只会看到 wc=0 / 同 sha，无法区分"训练后权重没变"与"直接复制父产物"。
        return {"weights_changed": False, "action": None,
                "sha256": _sha256_file(target),
                "soup_skip": {
                    "reason": "candidate_is_parent",
                    "parent": parent["sha256"][:16],
                    "history_before": [str(item.get("sha256") or "")[:16]
                                       for item in (history or [])],
                    "history_after": [],
                }}
    import numpy as np

    try:
        with np.load(parent["path"], allow_pickle=False) as archive:
            files = set(archive.files)
    except Exception as exc:
        raise TrainRoundError("父检查点不是可读的 npz，无法造候选：%s" % exc)

    if "task" not in files and all(name in files for name in OFFICIAL_KEYS):
        if not data_dir:
            raise TrainRoundError("官方 MLP 父产物需要训练数据目录才能续训")
        try:
            report = train_mlp.finetune(parent["path"], cand_dir, data_dir,
                                        arm=chosen, round_index=round_index,
                                        params=params, dataset_key=dataset_key,
                                        history_candidates=history)
        except train_mlp.TrainMlpError as exc:
            raise TrainRoundError("官方 MLP 续训失败：%s" % exc)
        result = {"weights_changed": report["weights_changed"], "action": None,
                  "params": {"lr": report["hyper"]["lr"],
                             "batch": report["hyper"]["batch"],
                             "epochs": report["hyper"]["epochs_run"],
                             "scope": report["scope"]},
                  "sha256": _sha256_file(target), "training": report}
        _memo_c_build(chosen, dataset_key, parent, params, round_index, cand_dir, result)
        return result
    if "coef" not in files:
        raise TrainRoundError("父检查点既不是官方格式，也没有 v2 的 coef；无法造动作候选")

    action = {"A": "streaming_ema_norm", "B": "per_cond_seq", "C": "identity"}[chosen]
    params = dict(params or {})
    with np.load(parent["path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["action"] = np.array(action)
    arrays["action_params"] = np.array(json.dumps(params, sort_keys=True))
    np.savez(target, **arrays)
    result = {"weights_changed": False, "action": action, "params": params,
              "sha256": _sha256_file(target)}
    _memo_c_build(chosen, dataset_key, parent, params, round_index, cand_dir, result)
    return result


def _memo_c_build(chosen, dataset_key, parent, params, round_index, cand_dir, result):
    """C 线性能：记住本轮 C 臂第一次拟合的产物，供接管 prepare 复制复用。"""
    if chosen == "C" and dataset_key == "C":
        _C_BUILD_MEMO[(parent.get("sha256"),
                       json.dumps(params or {}, sort_keys=True),
                       int(round_index))] = {"dir": cand_dir, "built": dict(result)}


def _weights_diff(parent_path, cand_path):
    """脚本候选与父产物的逐张量 rel_l2：脚本路径不能假定"权重未训练"。

    只比较两边都存在且能转成 float 的数组（字符串元数据跳过）；形状不一致记 None。
    比较失败由调用方兜底成"未核对"，绝不默认成"没改"。
    """
    import numpy as np

    diff = {}
    with np.load(parent_path, allow_pickle=False) as parent_npz:
        with np.load(cand_path, allow_pickle=False) as cand_npz:
            for name in sorted(set(parent_npz.files) & set(cand_npz.files)):
                try:
                    before = np.asarray(parent_npz[name], dtype=float)
                    after = np.asarray(cand_npz[name], dtype=float)
                except (TypeError, ValueError):
                    continue
                if before.shape != after.shape:
                    diff[name] = None
                    continue
                base = float(np.linalg.norm(before.reshape(-1)))
                delta = float(np.linalg.norm((after - before).reshape(-1)))
                if base > 0.0:
                    diff[name] = delta / base
                else:
                    diff[name] = 0.0 if delta == 0.0 else float("inf")
    changed = not diff
    for value in diff.values():
        if value is None or value > 1e-12:
            changed = True
    return {"diff": diff, "changed": bool(changed)}


SIZE_CAP = 64 * 1024 * 1024  # 平台公告：产物总量 ≤64 MiB，不用 SDK 的 512 MiB
CONFIG_BUDGET = 8 * 1024  # 账本 configuration 的内部预算：超了就按固定顺序删次要证据
_C_BUILD_MEMO = {}


def _candidate_cap(rnd):
    """候选上限 = min(64 MiB, 宿主限额)；宿主字段名待真机确认，读不到就回退 64 MiB。"""
    limits = getattr(rnd, "limits", None)
    if not isinstance(limits, dict):
        return SIZE_CAP
    for key in ("artifact_bytes", "candidate_bytes", "artifact_max_bytes",
                "candidate_max_bytes", "total_bytes"):
        value = limits.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return min(SIZE_CAP, int(value))
    return SIZE_CAP


def _config_size(cfg):
    return len(json.dumps(cfg, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def _shrink_config(cfg):
    """把 configuration 压到 8 KiB 内：只删重复/可回指的次要证据，并留 compacted 清单。

    顺序固定，绝不静默丢主证据：
      1. `data_slice.feature_mean`（指纹补充；sha256 与行数仍在）
      2. `cage.red_team[].citation`（q/ok 仍在，判据仍可读）
      3. `model.gate.repair.raw_excerpt`（status/arm 仍在，原文 sha256 仍在）
      4. `model.gate.raw_excerpt` 截到 120 字（原文 sha256 仍在）
    仍超预算就记 stderr，不伪造"已压缩"。
    """
    if _config_size(cfg) <= CONFIG_BUDGET:
        return cfg
    dropped = []
    card = cfg.get("data_slice") or {}
    if "feature_mean" in card:
        card.pop("feature_mean")
        dropped.append("data_slice.feature_mean")
    if _config_size(cfg) > CONFIG_BUDGET:
        trimmed = False
        for item in (cfg.get("cage") or {}).get("red_team") or []:
            if isinstance(item, dict) and "citation" in item:
                item.pop("citation")
                trimmed = True
        if trimmed:
            dropped.append("cage.red_team[].citation")
    if _config_size(cfg) > CONFIG_BUDGET:
        gate = (cfg.get("model") or {}).get("gate") or {}
        repair = gate.get("repair") or {}
        if repair.pop("raw_excerpt", None) is not None:
            dropped.append("model.gate.repair.raw_excerpt")
    if _config_size(cfg) > CONFIG_BUDGET:
        gate = (cfg.get("model") or {}).get("gate") or {}
        text = gate.get("raw_excerpt")
        if isinstance(text, str) and len(text) > 120:
            gate["raw_excerpt"] = text[:120]
            gate["raw_excerpt_truncated"] = True
            dropped.append("model.gate.raw_excerpt>120")
    if dropped:
        cfg["compacted"] = dropped
    # compacted 清单本身也占字节：把它算进去后再截一次，否则会出现
    # "压缩后 8202 字节"——清单把刚省下的空间又吃回去一部分。
    if _config_size(cfg) > CONFIG_BUDGET:
        gate = (cfg.get("model") or {}).get("gate") or {}
        text = gate.get("raw_excerpt")
        if isinstance(text, str) and len(text) > 60:
            gate["raw_excerpt"] = text[:60]
            gate["raw_excerpt_truncated"] = True
            cfg["compacted"] = list(dropped) + ["model.gate.raw_excerpt>60"]
    if _config_size(cfg) > CONFIG_BUDGET:
        _log("configuration 仍超过 8KiB：%d 字节（compacted=%s）"
             % (_config_size(cfg), ",".join(dropped) or "无"))
    return cfg


# P0-1 升级（2026-11-24）：SDK 的 submit_candidate / finish 都会用
# `json.dumps(configuration, allow_nan=False)` 生成声明，含 inf/nan 或不可 JSON 化的
# 对象时直接抛 ValueError/TypeError，已提交成功的产物会被平台判失败。这里把
# “整份 configuration 必须可序列化”变成提交前的硬自检，并给出降级路径。
CONFIG_NUM_ABS_LIMIT = 1e12  # 与 llm_cage.NUM_ABS_LIMIT 同一条上界
_DROP = object()  # _json_safe_copy 的“该字段已剔除”哨兵
_BOOKKEEP_FAILED = object()  # 提交后簿记步骤失败的哨兵（与正常返回 None 区分）


def _json_safe_copy(value, path="", dropped=None):
    """返回一份一定能过 `json.dumps(..., allow_nan=False)` 的副本。

    - 非有限 float（inf/nan）与 |x| > 1e12 的 float：剔除该字段并记路径；
    - dict/list/tuple 递归；非字符串键、集合/bytes/自定义对象：剔除并记路径；
    - 其余标量原样保留。合法输入的内容与序列化字节都不变。
    """
    if dropped is None:
        dropped = []
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > CONFIG_NUM_ABS_LIMIT:
            dropped.append(path or "<root>")
            return _DROP
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                dropped.append("%s[非字符串键 %r]" % (path, key))
                continue
            child = _json_safe_copy(item, ("%s.%s" % (path, key)) if path else key, dropped)
            if child is not _DROP:
                out[key] = child
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for index, item in enumerate(value):
            child = _json_safe_copy(item, "%s[%d]" % (path, index), dropped)
            if child is _DROP:
                continue
            out.append(child)
        return out
    dropped.append("%s(%s)" % (path or "<root>", type(value).__name__))
    return _DROP


def _config_selfcheck(cfg):
    """整份 configuration 的 allow_nan=False 自检：通过返回 None，否则返回错误文本。"""
    try:
        json.dumps(cfg, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        return "%s: %s" % (type(exc).__name__, str(exc)[:200])
    return None


def _minimal_configuration(round_index, note="configuration_degraded"):
    """降级用的最小合法配置：dict、短、全可序列化。"""
    return {"degraded": str(note), "seed": int(round_index)}


def _safe_configuration(cfg, round_index):
    """提交/收尾前把 configuration 收成一定能过 allow_nan=False 的形态。

    正常路径原样返回同一个对象：不改内容、不改字节，载荷与账本都不受影响。
    自检失败时先剔除可疑字段（而不是整轮放弃），仍不过才降级为最小配置。
    """
    problem = _config_selfcheck(cfg)
    if problem is None:
        return cfg
    dropped = []
    safe = _json_safe_copy(cfg, "", dropped)
    if safe is _DROP or not isinstance(safe, dict):
        safe = _minimal_configuration(round_index, "configuration_not_a_dict")
        dropped.append("<root>")
    elif dropped:
        safe["sanitized"] = list(dropped)[:8]
    remaining = _config_selfcheck(safe)
    if remaining is not None:
        _log("configuration 剔除可疑字段后仍不可序列化（%s）：降级为最小配置" % remaining)
        safe = _minimal_configuration(round_index, "configuration_not_serializable")
        dropped.append("<root>")
    if _config_size(safe) > CONFIG_BUDGET:
        _shrink_config(safe)
    _log("configuration 自检失败（%s）：已剔除 %d 个可疑字段（%s），按降级配置继续"
         % (problem, len(dropped), ", ".join(dropped[:8]) or "无"))
    return safe



def _budget(cand_dir, cap=SIZE_CAP):
    files, total = 0, 0
    for dirpath, _, filenames in os.walk(cand_dir):
        for name in filenames:
            files += 1
            total += os.path.getsize(os.path.join(dirpath, name))
    if files > 64 or total > cap:
        raise TrainRoundError("候选超限：%d 文件 / %d 字节（上限 %d）" % (files, total, cap))
    return {"files": files, "bytes": total, "cap": cap}


def _score_arms(parent, data_dir, round_index, dataset_key, work, chosen, prepared):
    """每轮把四个臂都构造出来，在**同一份**验证切片上各打一次分；本函数只评分。

    这是四臂实验的原始数据：模型选 / 规则选 / 实际最优必须来自同一把尺子，
    所以这里不读平台回执、不重跑第二轮，只做一次固定标尺评分。评分完成后的
    载荷接管在调用方（只对规则兜底的 B/C 回归轮、且须再过同一组闸门）。

    已构建的本轮候选直接复用，避免重复训练；其余臂在
    `round-<n>/arm-scores/<arm>/` 下单独构建。任一臂失败只记该臂 error，不拖垮整轮。
    """
    root = os.path.join(work, "round-%s" % round_index, "arm-scores")
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    try:
        z_val, y_val, task = train_mlp.load_yardstick(data_dir, parent["path"])
    except Exception as exc:  # noqa: BLE001 - 标尺不可用就如实记，不伪造分数
        return {"slice": {"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])},
                "arms": {}, "best": None, "chosen": chosen, "chosen_is_best": None,
                "best_candidates": [], "best_tied": None}
    metric = "macro_f1" if task == "classification" else "rmse"
    record = {"slice": {"rows": int(z_val.shape[0]), "metric": metric},
              "arms": {}, "best": None, "chosen": chosen, "chosen_is_best": None,
              "best_candidates": [], "best_tied": None}
    scores = {}
    for arm in (PARENT,) + CANDIDATE_IDS:
        if arm == chosen:
            path = os.path.join(prepared["dir"], CHECKPOINT_NAME)
            sha = (prepared.get("built") or {}).get("sha256")
        else:
            cand_dir = os.path.join(root, arm)
            os.makedirs(cand_dir, exist_ok=True)
            try:
                built = _build_candidate(parent, arm, cand_dir, {}, data_dir=data_dir,
                                         round_index=round_index, dataset_key=dataset_key)
            except Exception as exc:  # noqa: BLE001 - 单臂失败只记该臂
                record["arms"][arm] = {"status": "error",
                                       "error": "%s: %s" % (type(exc).__name__,
                                                            str(exc)[:120])}
                continue
            path = os.path.join(cand_dir, CHECKPOINT_NAME)
            sha = built.get("sha256")
        try:
            metrics = train_mlp.score_checkpoint(path, z_val, y_val)
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("评分结果缺少有限数值 %s" % metric)
            value = float(value)
        except Exception as exc:  # noqa: BLE001 - 单臂失败只记该臂
            record["arms"][arm] = {"status": "error",
                                   "error": "%s: %s" % (type(exc).__name__,
                                                        str(exc)[:120])}
            continue
        record["arms"][arm] = {
            "status": "ok",
            "value": round(value, 6),
            "sha256": str(sha or "")[:8],
            "changed": (False if arm == PARENT else
                        _digest_hex(sha) != _digest_hex(parent["sha256"])),
        }
        scores[arm] = value if metric == "macro_f1" else -value
    if scores:
        # 平局语义：等分不靠字典序定胜负。按账本落账精度（6 位小数）取最优集合；
        # 多于一个臂并列最优时记 "TIE"，避免 PARENT 仅凭名字顺序被记成严格最优。
        recorded = {arm: record["arms"][arm]["value"] for arm in scores}
        best_value = (max if metric == "macro_f1" else min)(recorded.values())
        best_candidates = sorted(arm for arm, v in recorded.items()
                                 if v == best_value)
        record["best_candidates"] = best_candidates
        record["best_tied"] = len(best_candidates) > 1
        record["best"] = "TIE" if record["best_tied"] else best_candidates[0]
        record["chosen_is_best"] = bool(not record["best_tied"]
                                        and record["best"] == chosen)
    return record


def _r10_value(task, metrics):
    """同一把尺子上的比较键：分类取 macro_f1（大者优），回归取 rmse（小者优）。"""
    if not isinstance(metrics, dict):
        return None
    key = "macro_f1" if task == "classification" else "rmse"
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _r10_better(task, value, current):
    if value is None:
        return False
    if current is None:
        return True
    return value > current if task == "classification" else value < current


def _r10_family(prepared, history, parent, data_dir, round_index, dataset_key,
                work, cap=SIZE_CAP, allow_apply=True):
    """R10 候选族同场测量：族内每份权重都在**当轮切片**上打分，再决定上传哪一份。

    同场 = 可比性的来源：所有候选共用 `load_yardstick` 的同一份 z_val/y_val。
    分线构成（06 号 J5）：
      - A（分类）：父、R1、R2 三份端点，不进插值（α 只有 1.0）；
      - B/C（回归）：父–当前 / 父–R1 / 父–R2 / R1–R2 四对 × {1.0, 0.5}，
        去重后 ≤8 份权重。
    双列（05 号）：`scores` 按提交口径（带自身 calibration），`scores_raw` 统一未校准。
    选中插值产物时写 `steps_semantics="schedule_anchor_not_trained_steps"`。
    """
    if int(round_index) != R10_ROUND:
        return None
    try:
        z_val, y_val, task = train_mlp.load_yardstick(data_dir, parent["path"])
    except Exception as exc:  # noqa: BLE001 - 标尺不可用就如实记，不选族
        return {"status": "error",
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
    slice_sha = None
    val_path = os.path.join(data_dir, train_stage.VAL_NAME)
    if os.path.isfile(val_path):
        slice_sha = _sha256_file(val_path)[:16]

    endpoints = [{"id": "P", "round": None, "path": parent["path"],
                  "sha256": _digest_hex(parent["sha256"]), "role": "parent"}]
    for entry in history or []:
        sha = _digest_hex(entry.get("sha256"))
        if not sha or any(_digest_hex(item["sha256"]) == sha for item in endpoints):
            continue
        if not os.path.isfile(entry.get("path") or ""):
            continue
        # 同一轮可能有两份端点（上传载荷 + 训练终点，sha 不同）；族内 id 必须唯一，
        # 否则 P-R3 会出现两对、scores 长度与 alphas 对不上，alpha_grid 直接抛错。
        ident = "R%s" % entry.get("round")
        if any(item["id"] == ident for item in endpoints):
            ident = "%s-%s" % (ident, sha[:4])
        endpoints.append({"id": ident,
                          "round": entry.get("round"), "path": entry["path"],
                          "sha256": sha, "role": "history"})
        if len(endpoints) >= 3:
            break

    payload_dir = prepared.get("payload_dir") or prepared["dir"]
    payload_path = os.path.join(payload_dir, CHECKPOINT_NAME)
    payload_sha = _digest_hex((prepared.get("built") or {}).get("sha256"))
    payload = {"id": "C", "round": round_index, "path": payload_path,
               "sha256": payload_sha, "role": "payload"}

    if task == "classification":
        alphas = list(R10_ALPHAS_ENDPOINTS)
        specs = [("P", endpoints[0], endpoints[0])]
        for item in endpoints[1:]:
            specs.append(("P-%s" % item["id"], endpoints[0], item))
        # 历史端点不足两份且本轮载荷是另一份权重时补进族（仍 ≤3 份、仍不插值）。
        if (len(specs) < 3 and payload_sha and os.path.isfile(payload_path)
                and all(payload_sha != item["sha256"] for item in endpoints)):
            specs.append(("P-C", endpoints[0], payload))
    else:
        alphas = list(R10_ALPHAS_SOUP)
        specs = [("P-C", endpoints[0], payload)]
        for item in endpoints[1:]:
            specs.append(("P-%s" % item["id"], endpoints[0], item))
        if len(endpoints) >= 3:
            specs.append(("%s-%s" % (endpoints[1]["id"], endpoints[2]["id"]),
                          endpoints[1], endpoints[2]))
        # 族 ≤8：4 对 × 2 α 去重后最多 8 份权重；超了按 03 号砍配（先丢端点位移
        # 最近、信息量最低的对）。
        while len(specs) * len(alphas) > R10_FAMILY_CAP and len(specs) > 1:
            specs.pop()
        # 两端点同一份权重（本轮没改动时 P-C 会退化成 parent–parent）不构成一对：
        # 只制造重复分数与假并列，直接剔除；parent 的分数已由各对端点覆盖。
        specs = [item for item in specs if item[1]["sha256"] != item[2]["sha256"]]

    # 权重与父逐字节相同、只有 metadata.calibration 不同的载荷：α<1 的插值会走
    # alpha_lib._merge_metadata，把 calibration 当作"换权重后失效"丢掉；但这里权重
    # 根本没换，丢掉的恰是唯一有效的校准（B 的校准载荷就是这种：权重=父、只有校准）。
    # 这类对只保留 α=1（端点本身参与同场打分），α<1 不生成，并在 ledger 的
    # soup_excluded 里记明原因。
    degenerate_ids = set()
    soup_excluded = []
    if task != "classification":
        import numpy as np  # noqa: E402 - 只在这一段比权重，避免顶层强依赖

        def _model_arrays(entry):
            try:
                return train_mlp._load_parent(entry["path"])["arrays"]
            except Exception:  # noqa: BLE001 - 读不出就按原路径处理
                return None

        for spec in specs:
            left, right = spec[1], spec[2]
            left_arrays, right_arrays = _model_arrays(left), _model_arrays(right)
            same_weights = bool(
                left_arrays is not None and right_arrays is not None
                and all(left_arrays[name].shape == right_arrays[name].shape
                        and np.array_equal(left_arrays[name], right_arrays[name])
                        for name in ("w1", "b1", "w2", "b2", "mean", "scale")
                        if name in left_arrays and name in right_arrays))
            if same_weights and left["sha256"] != right["sha256"]:
                degenerate_ids.add(spec[0])
                soup_excluded.append({
                    "pair": spec[0],
                    "reason": "weights_equal_parent_only_calibration_differs",
                    "a": left["sha256"][:16], "b": right["sha256"][:16]})

    interp_dir = os.path.join(work, "round-%s" % int(round_index), "family")
    cells = []
    for pair_id, left, right in specs:
        for alpha in alphas:
            if pair_id in degenerate_ids and 0.0 < float(alpha) < 1.0:
                # 见上：权重相同、只差校准的对不生成插值格，避免把校准静默抹掉。
                continue
            cell = {"pair": pair_id, "alpha": float(alpha),
                    "endpoints": [left["sha256"][:16], right["sha256"][:16]]}
            try:
                if alpha >= 1.0:
                    cell.update({"path": right["path"], "sha": right["sha256"],
                                 "interpolated": False,
                                 "source_round": right.get("round")})
                elif alpha <= 0.0:
                    cell.update({"path": left["path"], "sha": left["sha256"],
                                 "interpolated": False,
                                 "source_round": left.get("round")})
                else:
                    os.makedirs(interp_dir, exist_ok=True)
                    path = os.path.join(interp_dir, "%s-%.2f.npz"
                                        % (pair_id.replace("/", "_"), alpha))
                    if not os.path.isfile(path):
                        ckpt = _alpha_lib.interpolate_checkpoints(
                            left["path"], right["path"], float(alpha))
                        _alpha_lib.write_checkpoint(ckpt, path)
                    cell.update({"path": path, "sha": _sha256_file(path),
                                 "interpolated": True, "source_round": None})
            except Exception as exc:  # noqa: BLE001 - 单格失败只记该格
                cell.update({"error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
            cells.append(cell)

    # 同一份权重（同 sha）只打一次分：α=1 的格与端点格大量重合。
    seen = {}
    for cell in cells:
        if cell.get("error") or not cell.get("sha"):
            continue
        sha = cell["sha"]
        if sha not in seen:
            try:
                metrics = train_mlp.score_checkpoint(cell["path"], z_val, y_val)
                raw = train_mlp.score_checkpoint(cell["path"], z_val, y_val,
                                                 calibrated=False)
                value = _r10_value(task, metrics)
                if value is None:
                    raise ValueError("评分结果缺少有限数值")
                seen[sha] = {"value": value, "metrics": metrics,
                             "raw_value": _r10_value(task, raw),
                             "rel_l2": None}
            except Exception as exc:  # noqa: BLE001 - 单格失败只记该格
                seen[sha] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
        cell["value"] = seen[sha].get("value")
        cell["raw_value"] = seen[sha].get("raw_value")

    # 端点位移（供 03 号第 3 级砍配与账本对账）：与权重闸门同口径的拼接 rel_l2。
    try:
        parent_arrays = train_mlp._load_parent(parent["path"])["arrays"]
    except Exception:  # noqa: BLE001 - 父不可读时位移一律留 None
        parent_arrays = None
    for cell in cells:
        if cell.get("rel_l2") is None and cell.get("path") and parent_arrays:
            try:
                arrays = train_mlp._load_parent(cell["path"])["arrays"]
                cell["rel_l2"] = train_mlp.concat_rel_l2(parent_arrays, arrays)
            except Exception:  # noqa: BLE001 - 位移只作辅助，算不出就留 None
                cell["rel_l2"] = None

    pairs = []
    for pair_id, left, right in specs:
        rows = [cell for cell in cells if cell["pair"] == pair_id]
        scores = [None if row.get("value") is None else round(row["value"], 6)
                  for row in rows]
        raws = [None if row.get("raw_value") is None else round(row["raw_value"], 6)
                for row in rows]
        if any(value is None for value in scores):
            pairs.append({"id": pair_id, "a": left["sha256"][:16],
                          "b": right["sha256"][:16], "scores": scores,
                          "scores_raw": raws, "sel": None, "error": "存在未评分格"})
            continue
        best_at = (max(range(len(scores)), key=lambda i: scores[i])
                   if task == "classification"
                   else min(range(len(scores)), key=lambda i: scores[i]))
        rels = [row.get("rel_l2") for row in rows
                if isinstance(row.get("rel_l2"), (int, float))]
        pairs.append({"id": pair_id, "a": left["sha256"][:16],
                      "b": right["sha256"][:16], "scores": scores,
                      "scores_raw": raws, "sel": float(rows[best_at]["alpha"]),
                      "endpoints_rel": (round(max(rels), 6) if rels else None)})

    # 排名按**去重后的权重**算：同一份权重在多个格里出现（α=1 与端点格）不该
    # 把 margin 压成 0，也不该被当成"并列第二"。所选候选必须先过既有 0.1 闸门
    # （A 为计划累计预算），过不了的不参与选型——宁可不上传，也不提交注定被拒的权重。
    # v3 预登记（2026-11-22 c-push）：C 线的漂移守卫改成"预测位移"，判定点唯一在
    # train_mlp 的 soup（R10 族这里没有带标签切片，无法算位移）。所以 C 在族内只记录
    # rel_l2、不据此剔除；A/B 仍按 rel_l2 筛选。
    guard_enforced = str(dataset_key or "").strip().upper() != "C"
    try:
        weight_cap = float(train_mlp.weight_rel_cap(dataset_key))
    except Exception:  # noqa: BLE001 - 取不到闸门就退化为只记录不筛选
        weight_cap = None

    def _eligible(cell):
        if cell.get("value") is None or not cell.get("sha"):
            return False
        rel = cell.get("rel_l2")
        if not guard_enforced or weight_cap is None or rel is None:
            return True
        return float(rel) <= weight_cap

    unique = {}
    for cell in cells:
        if cell.get("value") is None or not cell.get("sha"):
            continue
        if cell["sha"] not in unique:
            unique[cell["sha"]] = cell
    eligible = [cell for cell in unique.values() if _eligible(cell)]
    ranked = sorted(eligible, key=lambda cell: cell["value"],
                    reverse=(task == "classification"))
    winner = ranked[0] if ranked else None
    tie = bool(len(ranked) > 1 and ranked[1]["value"] == ranked[0]["value"])
    margin = None
    if len(ranked) > 1:
        margin = round(abs(float(ranked[0]["value"]) - float(ranked[1]["value"])), 6)

    # 双列同生同灭：任一列有缺口（未评分格）的整对不进 alpha_grid，记进 grid_skipped。
    # 退化对（只保留 α=1）也在这里排除：它们的 scores 只有 1 列，放进网格会让
    # serialize_alpha_grid 的长度校验抛错。
    grid_pairs = [pair for pair in pairs
                  if pair["id"] not in degenerate_ids
                  and pair.get("sel") is not None
                  and all(isinstance(value, (int, float)) for value in pair["scores"])
                  and all(isinstance(value, (int, float))
                          for value in pair["scores_raw"])]
    grid_skipped = [pair["id"] for pair in pairs if pair not in grid_pairs]

    record = {
        "status": ("ok" if winner is not None
                   else ("no_eligible" if any(c.get("value") is not None for c in cells)
                         else "no_candidate")),
        "slice": {"sha256_16": slice_sha, "rows": int(z_val.shape[0]),
                  "metric": "macro_f1" if task == "classification" else "rmse"},
        "task": task,
        "cap": R10_FAMILY_CAP,
        "weight_cap": weight_cap,
        "weight_guard": ("pred_disp@soup" if not guard_enforced else "rel_l2"),
        "alphas": [float(a) for a in alphas],
        "tie": tie,
        "margin": margin,
        "pairs": [{"id": pair["id"], "sel": pair.get("sel")}
                  for pair in pairs],
        "grid_skipped": grid_skipped,
        "soup_excluded": soup_excluded,
        "candidates": [
            {"pair": cell["pair"], "alpha": cell["alpha"],
             "sha256": str(cell.get("sha") or "")[:16],
             "rel_l2": (None if cell.get("rel_l2") is None
                        else round(float(cell["rel_l2"]), 6)),
             "value": (None if cell.get("value") is None
                       else round(float(cell["value"]), 6)),
             "raw": (None if cell.get("raw_value") is None
                     else round(float(cell["raw_value"]), 6)),
             "interpolated": bool(cell.get("interpolated")),
             "eligible": _eligible(cell),
             "error": cell.get("error")}
            for cell in cells],
        "selected": None,
        "payload_changed": False,
        "grid": ({"slice": slice_sha,
                  "metric": "macro_f1" if task == "classification" else "rmse",
                  "alphas": [float(a) for a in alphas],
                  "margin": margin or 0.0,
                  "pairs": grid_pairs} if grid_pairs else None),
    }
    if winner is None:
        return record

    current_sha = payload_sha
    same_as_payload = bool(current_sha and winner["sha"] == current_sha)
    same_as_parent = bool(winner["sha"] == _digest_hex(parent["sha256"]))
    if same_as_parent:
        source = "parent"
    elif same_as_payload:
        source = "payload"
    else:
        source = "family"
    record["selected"] = {"pair": winner["pair"], "alpha": winner["alpha"],
                          "endpoints": winner["endpoints"],
                          "sha256": winner["sha"][:16],
                          "value": round(float(winner["value"]), 6),
                          "raw": (None if winner.get("raw_value") is None
                                  else round(float(winner["raw_value"]), 6)),
                          "interpolated": bool(winner.get("interpolated")),
                          "source_round": winner.get("source_round"),
                          "source": source,
                          # applied=是否真的改写了上传载荷；初值一律 False，
                          # 只有走到 :1080 真正替换成功才置 True。
                          "applied": False,
                          "skip_reason": (None if source == "family"
                                          else ("winner_is_parent" if source == "parent"
                                                else "winner_is_payload")),
                          "tie": tie}
    if same_as_payload or same_as_parent:
        # 族内最优就是现有载荷或父产物：不改动，只落账。
        return record
    if not allow_apply:
        # 本轮已判 PARENT（终局门/回退）：族照测、分数照记，但不越过门改上传物。
        record["selected"]["applied"] = False
        record["selected"]["blocked"] = "chosen_parent"
        record["selected"]["skip_reason"] = "final_round_gate"
        return record

    # 族内更优：把它做成上传载荷，仍要过体积上限与候选自检。
    # 不覆盖、不删已有载荷目录：另建 payload_r10，先整目录复制再替换 checkpoint，
    # 自检不过时原载荷原样可用（宁可少一次增益，不能丢可提交物）。
    source_dir = prepared.get("payload_dir") or prepared["dir"]
    payload_dir = os.path.join(os.path.dirname(source_dir), "payload_r10")
    shutil.rmtree(payload_dir, ignore_errors=True)
    os.makedirs(payload_dir)
    for name in sorted(os.listdir(source_dir)):
        src = os.path.join(source_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(payload_dir, name))
    target = os.path.join(payload_dir, CHECKPOINT_NAME)
    shutil.copyfile(winner["path"], target)
    record["selected"]["applied"] = False
    try:
        budget = _budget(payload_dir, cap)
    except TrainRoundError as exc:
        shutil.rmtree(payload_dir, ignore_errors=True)
        record["status"] = "over_budget"
        record["error"] = str(exc)[:200]
        return record
    check = llm_cage.check_candidate(payload_dir, parent_path=parent["path"],
                                     arm=dataset_key)
    if not check["ok"]:
        shutil.rmtree(payload_dir, ignore_errors=True)
        record["status"] = "candidate_rejected"
        record["error"] = "；".join(check["errors"])[:200]
        return record
    prepared["payload_dir"] = payload_dir
    prepared["payload_budget"] = budget
    prepared["payload_check"] = check
    prepared["built"]["trained_sha256"] = prepared["built"].get("sha256")
    prepared["built"]["sha256"] = _sha256_file(target)
    prepared["built"]["payload"] = "family"
    prepared["built"]["payload_round"] = winner.get("source_round")
    prepared["built"]["family_evidence"] = {
        "pair": winner["pair"], "alpha": winner["alpha"],
        "endpoints": winner["endpoints"],
        "steps_semantics": ("schedule_anchor_not_trained_steps"
                            if winner.get("interpolated") else None),
    }
    record["payload_changed"] = True
    record["selected"]["applied"] = True
    return record


def _platform_numbers(submitted):
    """只从平台回执里取数字，绝不混入我们自己算的检测值。"""
    found = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for key in sorted(node):
                walk(node[key], "%s.%s" % (prefix, key) if prefix else str(key))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if isinstance(node, float) and node != node:
                return
            found.append("%s=%s" % (prefix, node))

    for key in ("validation", "metrics", "measurement", "scores", "result"):
        if isinstance(submitted.get(key), dict):
            walk(submitted[key], key)
    if not found:
        for key in ("score", "value", "accuracy", "rmse", "mae", "r2"):
            value = submitted.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.append("%s=%s" % (key, value))
    return found


def _reason_text(round_index, chosen, submitted):
    numbers = _platform_numbers(submitted)
    head = "第%s轮选择候选 %s" % (round_index, chosen)
    if numbers:
        return head + "；平台返回：" + "，".join(numbers[:8])
    return head + "；平台未返回数值摘要"


def _pessimism_k(override=None):
    """解析悲观界系数 k，返回 (k, 来源)。配置项优先于默认值。"""
    if override is not None:
        value, source = float(override), "parameter"
    else:
        raw = os.environ.get(PESSIMISM["env"])
        if raw is None or not str(raw).strip():
            return float(PESSIMISM["k"]), "default"
        try:
            value, source = float(raw), "env:%s" % PESSIMISM["env"]
        except ValueError:
            raise TrainRoundError("%s 不是数字：%r" % (PESSIMISM["env"], raw))
    if not math.isfinite(value) or value < 0.0:
        raise TrainRoundError("悲观界系数 k 必须是非负有限数：%r" % (value,))
    return value, source


def _read_state(state_path):
    """读上一轮状态（只读副本）。

    文件不存在 = 首轮，返回空状态，正常。
    文件存在但读不出 = **状态降级**：必须显式标记，不能伪装成"首轮空状态"，
    否则签名表被当成空表、闸门被悄悄放宽，账本上看不出任何异常。
    """
    path = os.path.join(state_path, STATE_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        data = _load_json(path)
    except (ValueError, OSError) as exc:
        _log("状态文件不可解析（%s）：本轮按降级处理，闸门不放宽" % exc)
        return {"_degraded": True, "_error": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(data, dict):
        _log("状态文件不是对象（%r）：本轮按降级处理，闸门不放宽" % (type(data).__name__,))
        return {"_degraded": True, "_error": "state is %s" % type(data).__name__}
    return data


def _arm_bound(stats, k):
    """臂的后验悲观界 mean - k*sd。未观测的臂返回 None（不参与比较）。"""
    n = int((stats or {}).get("n", 0) or 0)
    if n <= 0:
        return None
    mean = float(stats.get("mean", 0.0))
    m2 = float(stats.get("m2", 0.0))
    sd = math.sqrt(m2 / (n - 1)) if n > 1 and m2 > 0.0 else 0.0
    bound = mean - k * sd
    return bound if math.isfinite(bound) else None


def _arm_update(stats, score):
    """Welford 累积一个臂的观测分数（已统一成“越大越好”）。"""
    entry = dict(stats or {})
    n = int(entry.get("n", 0) or 0) + 1
    mean = float(entry.get("mean", 0.0))
    m2 = float(entry.get("m2", 0.0))
    delta = score - mean
    mean += delta / n
    m2 += delta * (score - mean)
    return {"n": n, "mean": mean, "m2": max(0.0, m2),
            "failures": int(entry.get("failures", 0) or 0),
            "cooldown": int(entry.get("cooldown", 0) or 0),
            # 指标名随臂统计一起存：跨轮换了口径（macro_f1 ↔ 负 RMSE）时，
            # 只看 mean 无法发现量纲已经混了。
            "key": entry.get("key")}


def _arm_score(submitted):
    """从平台回执里取一个可比较的标量分数。取不到就不记，不猜。"""
    if not isinstance(submitted, dict):
        return None
    names = ("score", "accuracy", "f1", "macro_f1", "r2", "value",
             "rmse", "mae", "mse", "loss", "error")
    nodes = []
    for field in ("validation", "metrics", "measurement", "scores", "result"):
        if isinstance(submitted.get(field), dict):
            nodes.append((field, submitted[field]))
    nodes.append(("", submitted))
    for field, node in nodes:
        for name in names:
            value = node.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            key = "%s.%s" % (field, name) if field else name
            return {"key": key, "value": -value if name in LOWER_BETTER else value,
                    "raw": value}
    return None


def _next_arms(arms_in, chosen, score, failed_arm):
    """生成本轮要落盘的臂统计：cooldown 递减；失败臂记一次失败并进入冷却。"""
    arms = {arm: dict((arms_in or {}).get(arm) or {})
            for arm in (PARENT,) + CANDIDATE_IDS}
    for entry in arms.values():
        entry["cooldown"] = max(0, int(entry.get("cooldown", 0) or 0) - 1)
    if failed_arm and failed_arm != chosen:
        entry = arms.setdefault(failed_arm, {})
        entry["failures"] = int(entry.get("failures", 0) or 0) + 1
        entry["cooldown"] = int(PESSIMISM["cooldown_rounds"])
        # 收缩：失败一次就把该臂的后验均值向父候选拉一半（无尺度依赖）。
        # m2 必须跟着重算成"围绕新均值的离差平方和"：只改 mean 会让 (n, mean, m2)
        # 互不自洽——_arm_bound 用 m2/(n-1) 当 sd，账本上的方差就不再对应它记的均值。
        parent_mean = (arms.get(PARENT) or {}).get("mean")
        n_old = int(entry.get("n", 0) or 0)
        if parent_mean is not None and n_old > 0:
            mean_old = float(entry.get("mean", 0.0))
            mean_new = 0.5 * mean_old + 0.5 * float(parent_mean)
            m2_old = float(entry.get("m2", 0.0) or 0.0)
            # 观测集固定时把中心平移 d，离差平方和增加 n·d²：
            # sum(x-μ')² = sum(x-μ)² + n(μ-μ')²。
            entry["mean"] = mean_new
            entry["m2"] = max(0.0, m2_old + n_old * (mean_old - mean_new) ** 2)
    if score is not None:
        entry = _arm_update(arms.get(chosen), score["value"])
        entry["key"] = score.get("key")
        arms[chosen] = entry
    return arms


def _state_record(state_path, ledger, round_index, artifact_id, previous, arms, score,
                  signature=None, failed_combo=None, best=None,
                  bias_ema=None, bias_last=None):
    previous = previous or {}
    signatures = [str(item) for item in (previous.get("signatures") or [])][-7:]
    if signature and signature not in signatures:
        signatures.append(signature)
    combos = list(previous.get("failed_combos") or [])[-7:]
    if failed_combo:
        combos.append(failed_combo)
    rounds = list(previous.get("rounds") or [])[-9:]
    model = (ledger.get("configuration") or {}).get("model") or {}
    rounds.append({
        "round": round_index,
        "signature": signature or "—",
        "arm": ledger["configuration"].get("candidate"),
        "action": ledger["configuration"].get("action"),
        "delta": model.get("delta_vs_parent"),
        "verdict": ledger["configuration"].get("policy"),
        "ledger_id": "round-%s" % round_index,
    })
    record = {
        "schema": "contest.v2.round-state",
        "round_index": round_index,
        "selected_artifact_id": artifact_id,
        "previous_selected_artifact_id": previous.get("selected_artifact_id"),
        "previous_round_index": previous.get("round_index"),
        "history_digest": ledger["configuration"].get("history_digest"),
        "candidate": ledger["configuration"].get("candidate"),
        "action": ledger["configuration"].get("action"),
        "arm_score": score,
        "arms": arms,
        "signatures": signatures[-8:],
        "failed_combos": combos[-8:],
        "rounds": rounds[-10:],
        "best": best if best is not None else previous.get("best"),
        # v6：输出层偏置候选族的跨轮状态（EMA 必须落盘，否则跨轮回放不可复现）。
        "bias_ema": bias_ema if bias_ema is not None else previous.get("bias_ema"),
        "bias_last": bias_last if bias_last is not None else previous.get("bias_last"),
    }
    path = os.path.join(state_path, STATE_FILE)
    _dump_json(path, record)
    return record


def _bias_ema_state(previous_state, arm, new_entry):
    """本轮要落盘的 bias_ema：只更新本轮的臂，其它臂原样保留。

    形状必须按臂分开：C 是仿射校准，存 {"a","c","rounds"}；A 是逐类偏移，存
    {"offset","rounds"}。两者共用一个形状会让 C 的 a/c 在落盘时丢掉、下一轮
    `_prior` 静默退回默认 a=1,c=0（EMA 每轮重启，轨迹看起来却"很稳"）。
    """
    store = dict((previous_state or {}).get("bias_ema") or {})
    if isinstance(new_entry, dict) and new_entry:
        rounds = int(new_entry.get("rounds", 0) or 0)
        if "a" in new_entry or "c" in new_entry:
            store[str(arm)] = {"a": float(new_entry.get("a", 1.0)),
                               "c": float(new_entry.get("c", 0.0)),
                               "rounds": rounds}
        else:
            store[str(arm)] = {"offset": list(new_entry.get("offset") or []),
                               "rounds": rounds}
    return store


# ---- 本地历史最优副本（v4，2026-11-21）----
# 不再向平台读回任何历史产物：副本（权重 + 标尺分数 + 轮次）随 state 目录跨轮持久化，
# 每轮只在本轮 validation.jsonl 上给“本轮结果 / 副本”各打一次分，取更优者上传。
BEST_DIR = "best"
BEST_META = "best-meta.json"

# ---- 历史候选权重档（v7 J1/J5，2026-11-22）----
# 每轮把上传载荷原样另存进 state/history，供 B/C 的 soup 历史对（父–R1 / 父–R2 /
# R1–R2）与 R10 的候选族同场测量使用。只保留最近 2 个不同端点，文件数上限 +2。
HISTORY_DIR = "history"
HISTORY_INDEX = "index.json"
HISTORY_KEEP = 2
# soup 候选族在 configuration 里的字节预算（超了只留选中对 + 父–当前，其余计数量）。
SOUP_GRID_BUDGET = 1400

# ---- v7 J5：R10 候选族同场测量（06 号文档 J5 / 03 号砍配 / 05 号双列）----
# 族 ≤8：B/C = {父–当前, 父–R1, 父–R2, R1–R2} × {1.0, 0.5}，去重后最多 8 份权重；
# A = 仅端点（父、R1、R2），α 只有 1.0，不产生插值候选。
R10_ROUND = 10
R10_FAMILY_CAP = 8
R10_ALPHAS_SOUP = (0.5, 1.0)
R10_ALPHAS_ENDPOINTS = (1.0,)


def _best_meta_path(state_path):
    return os.path.join(state_path, BEST_DIR, BEST_META)


def _read_best(state_path, dataset_key):
    """读 state 里的历史最优副本；缺文件、坏 JSON 或数据集不符都当作“没有副本”。"""
    meta_path = _best_meta_path(state_path)
    ckpt_path = os.path.join(state_path, BEST_DIR, CHECKPOINT_NAME)
    if not (os.path.isfile(meta_path) and os.path.isfile(ckpt_path)):
        return None
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (ValueError, OSError) as exc:
        _log("历史最优副本元数据不可读（忽略副本）：%s" % exc)
        return None
    if not isinstance(meta, dict) or not isinstance(meta.get("metrics"), dict):
        return None
    if str(meta.get("dataset_key")) != str(dataset_key):
        _log("历史最优副本属于数据集 %r，本轮是 %r，忽略副本"
             % (meta.get("dataset_key"), dataset_key))
        return None
    declared = meta.get("sha256")
    if not declared:
        _log("历史最优副本元数据缺 sha256，无法与权重对账，忽略副本")
        return None
    actual = _sha256_file(ckpt_path)
    if _digest_hex(declared) != _digest_hex(actual):
        _log("历史最优副本摘要不符：元数据 %s，实际 %s，忽略副本"
             % (str(declared)[:16], actual[:16]))
        return None
    return {"path": ckpt_path, "round": meta.get("round"),
            "metrics": meta["metrics"], "sha256": actual}


def _history_dir(state_path):
    return os.path.join(state_path, HISTORY_DIR)


def _history_index_path(state_path):
    return os.path.join(_history_dir(state_path), HISTORY_INDEX)


def _read_history_index(state_path):
    """读原始历史清单（不做校验）；缺失/损坏一律当空。"""
    path = _history_index_path(state_path)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            items = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - 清单坏了就当没有历史
        _log("历史候选清单不可读（忽略）：%s" % exc)
        return []
    return items if isinstance(items, list) else []


def _read_history(state_path):
    """可用的历史候选权重（按 round 升序）：文件在、sha 对得上才算。

    只返回最近 HISTORY_KEEP 个不同端点——J1 的父–R1 / 父–R2 / R1–R2 只用到两个，
    再多的端点不会进候选族，只会白占 state 体积。
    """
    items = []
    for item in _read_history_index(state_path):
        name = item.get("file") or ""
        path = os.path.join(_history_dir(state_path), os.path.basename(name))
        if not name or not os.path.isfile(path):
            continue
        try:
            sha = _sha256_file(path)
        except Exception:  # noqa: BLE001
            continue
        if sha != item.get("sha256"):
            continue
        items.append({"round": int(item.get("round") or 0), "path": path, "sha256": sha,
                      "metrics": item.get("metrics")})
    items.sort(key=lambda entry: entry["round"])
    return items[-HISTORY_KEEP:]


def _archive_history(state_path, ckpt_path, round_index, metrics=None, tag=""):
    """把本轮上传载荷另存进 state/history（供 J1 历史对与 J5 的 R10 候选族）。

    只增不改；同一 sha 只留最早一条；文件数超过 HISTORY_KEEP + 2 时删最旧的。
    全部先写临时名再原子替换，state 里不会出现半截文件。
    """
    sha = _sha256_file(ckpt_path)
    items = _read_history_index(state_path)
    if any(item.get("sha256") == sha for item in items):
        return None
    os.makedirs(_history_dir(state_path), exist_ok=True)
    name = "round-%02d%s.npz" % (int(round_index), tag)
    target = os.path.join(_history_dir(state_path), name)
    temp = target + ".tmp"
    shutil.copyfile(ckpt_path, temp)
    os.replace(temp, target)
    entry = {"round": int(round_index), "file": name, "sha256": sha,
             "metrics": metrics}
    items.append(entry)
    items.sort(key=lambda item: int(item.get("round") or 0))
    keep = items[-(HISTORY_KEEP + 2):]
    for stale in items[:-len(keep)] if len(items) > len(keep) else []:
        try:
            os.remove(os.path.join(_history_dir(state_path), os.path.basename(stale.get("file") or "")))
        except OSError:
            pass
    payload = json.dumps(keep, ensure_ascii=False, sort_keys=True).encode("utf-8")
    temp_index = _history_index_path(state_path) + ".tmp"
    with open(temp_index, "wb") as fh:
        fh.write(payload)
    os.replace(temp_index, _history_index_path(state_path))
    return entry


def _write_best(state_path, ckpt_path, metrics, round_index, dataset_key):
    """把本轮权重写成 state 里的最优副本（先写临时名再原子替换）。"""
    best_dir = os.path.join(state_path, BEST_DIR)
    os.makedirs(best_dir, exist_ok=True)
    target = os.path.join(best_dir, CHECKPOINT_NAME)
    temp = target + ".tmp"
    shutil.copyfile(ckpt_path, temp)
    os.replace(temp, target)
    meta = {"round": round_index, "dataset_key": dataset_key,
            "metrics": metrics, "sha256": _sha256_file(target),
            "source": "local-train"}
    _dump_json(_best_meta_path(state_path), meta)
    return meta


def _bias_text(record):
    """把校准压成一行：C 是 a/c，A 是标量或逐类偏移。"""
    chosen = record.get("chosen") or {}
    if "a" in chosen:
        return "a=%.3f, c=%+.2f" % (float(chosen["a"]), float(chosen["c"]))
    delta = chosen.get("delta") or []
    if len(delta) == 1:
        return "%+.4g" % float(delta[0])
    return json.dumps({str(index): round(float(value), 4)
                       for index, value in enumerate(delta) if value},
                      ensure_ascii=False, sort_keys=True)


def _bias_label(record):
    return "输出侧仿射校准" if record.get("kind") == "affine" else "输出层偏置"


def _pick_best_item(dataset_key, best_metrics, current_metrics):
    """两件套按同一规则决胜，返回 "best" 或 "current"；并列时保留副本。

    v5：C 线关闭 nas 决胜，按纯 rmse 比较（含原 1% 带内）。
    依据：nas 旧口径（逐样本求和、不截断）在 B 上复评到 1e51→1e76，v5.1 已改均值 +
    残差截断 125；但 C 的计分指标仍无官方明文，且 R7/R8 的 soup 候选公开 RMSE
    51.6415 优于 round-01 的 51.6621，却被 nas 决胜换掉——指标确认前不拿它判优。
    """
    items = [("best", best_metrics), ("current", current_metrics)]
    task = current_metrics.get("task")
    if task == "classification":
        winner = max(items, key=lambda item: item[1].get("macro_f1", float("-inf")))
    else:
        # 回归口径缺 rmse 时不能硬取 KeyError（会把整轮 best 拖死）；
        # 判不出高下就保留已有副本，不拿残缺指标换掉它。
        for label, metrics in items:
            if not isinstance(metrics.get("rmse"), (int, float)):
                _log("判优缺 rmse（%s），保留现有副本" % label)
                return "best"
        winner = min(items, key=lambda item: item[1]["rmse"])
    return winner[0]


def _apply_local_best(rnd, prepared, best, state_path, data_dir, round_index,
                      dataset_key, parent_path, expected_features):
    """本轮结果 vs 本地副本各打一次分；更优者成为本轮上传载荷。

    返回写进 ledger["best"] 的记录。不发起任何 artifact_read / restore。
    """
    cand_path = os.path.join(prepared["dir"], CHECKPOINT_NAME)
    record = {"available": bool(best), "best_round": (best or {}).get("round"),
              "best_metrics": None, "current_metrics": None,
              "replaced": False, "source_round": round_index, "payload": "current"}
    z_val, y_val, _task = train_mlp.load_yardstick(data_dir, parent_path)
    current = train_mlp.score_checkpoint(cand_path, z_val, y_val)
    record["current_metrics"] = current
    if best is not None:
        try:
            best_metrics = train_mlp.score_checkpoint(best["path"], z_val, y_val)
        except Exception as exc:  # noqa: BLE001 - 副本不可用就当没有
            _log("历史最优副本无法在当前标尺上打分（忽略副本）：%s" % exc)
            best, best_metrics = None, None
        record["best_metrics"] = best_metrics
    if best is None:
        _write_best(state_path, cand_path, current, round_index, dataset_key)
        record.update({"replaced": True, "best_round": round_index,
                       "best_metrics": current, "source_round": round_index})
        return record
    if _pick_best_item(dataset_key, record["best_metrics"], current) == "current":
        _write_best(state_path, cand_path, current, round_index, dataset_key)
        record.update({"replaced": True, "best_round": round_index,
                       "best_metrics": current, "source_round": round_index})
        return record
    # 副本更优：把副本权重原样作为本轮候选（不读平台），仍要过 5b rel_l2 与体积上限。
    payload_dir = os.path.join(os.path.dirname(prepared["dir"]), "payload")
    if os.path.isdir(payload_dir):
        shutil.rmtree(payload_dir)
    os.makedirs(payload_dir)
    payload = os.path.join(payload_dir, CHECKPOINT_NAME)
    shutil.copyfile(best["path"], payload)
    budget = _budget(payload_dir, _candidate_cap(rnd))
    check = llm_cage.check_candidate(payload_dir,
                                     expected_features=expected_features,
                                     parent_path=parent_path, arm=dataset_key)
    if not check["ok"]:
        shutil.rmtree(payload_dir)
        _log("历史最优副本未过候选自检（%s）；本轮仍提交本轮结果"
             % "；".join(check["errors"])[:200])
        return record
    prepared["payload_dir"] = payload_dir
    prepared["payload_budget"] = budget
    prepared["payload_check"] = check
    prepared["built"]["trained_sha256"] = prepared["built"].get("sha256")
    prepared["built"]["sha256"] = _sha256_file(payload)
    # 不覆写 weights_changed：它描述"本轮训练结果"，而载荷来源另记 payload/payload_round。
    # 覆写会让 summary 同时出现"与父产物逐字节一致"和"上传第 N 轮副本"两条互斥陈述。
    prepared["built"]["payload"] = "best"
    prepared["built"]["payload_round"] = best["round"]
    prepared["built"]["adopted_from_round"] = best["round"]
    record.update({"replaced": False, "source_round": best["round"],
                   "payload": "best"})
    return record


# ---------------------------------------------------------------- 输入动作通道（④）
# official 产物的 metadata 可携带 causal 输入动作：predict 在 mean/scale 之前执行，
# 动作输出替代归一化结果。这里负责"逐线实测再决定开不开"——同一份权重、同一份
# validation 切片、同一把尺子；候选动作不优于 identity 就写 identity，并把原因与
# 数字记进账本。分组闸只挡实测退化的分组，分组键取 predict 可见的 raw_id。
ACTION_GRID = (0.5, 0.9, 0.99, 0.995, 0.999)
# 只在"predict 能稳定拿到分组键"的线上开分组闸：B 的行 id 形如 xjtu_Bearing1_4_s000001。
ACTION_GROUP = {"B": ("bearing", r"(Bearing\d+_\d+)")}
_ACTION_CACHE = collections.OrderedDict()
_ACTION_CACHE_MAX = 4


def _action_rows(data_dir, ref, dataset_key):
    """按产物 schema 读 validation 的**原始**特征与分组键（动作作用在归一化之前）。"""
    import numpy as np

    val_path = os.path.join(data_dir, train_mlp.VAL_NAME)
    if not os.path.isfile(val_path):
        raise TrainRoundError("找不到验证集：%s" % val_path)
    n_features, task = ref["n_features"], ref["task"]
    group_field = ACTION_GROUP.get(dataset_key, (None, None))[0]
    key = (train_stage.file_key(val_path), n_features, task, dataset_key)
    hit = _ACTION_CACHE.get(key)
    if hit is not None:
        _ACTION_CACHE.move_to_end(key)
        return hit

    rows, _bad = train_mlp._read_rows(val_path)
    shape = None
    for row in rows:
        try:
            signature = train_mlp._shape_of(row["inputs"])
            probe = []
            train_mlp._collect(row["inputs"], probe)
        except (TypeError, ValueError, OverflowError):
            continue
        if len(probe) == n_features:
            shape = signature
            break
    if shape is None:
        raise TrainRoundError("验证集没有与产物特征维 %d 匹配的行" % n_features)

    if task == "classification":
        index = {str(name): i for i, name in enumerate(ref["meta"].get("classes") or [])}

        def label_fn(row):
            answer = row.get("answer")
            return index.get(answer) if isinstance(answer, str) else None
    else:
        def label_fn(row):
            answer = row.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                return None
            value = float(answer)
            return value if math.isfinite(value) else None

    xs, ys, groups = [], [], []
    for row in rows:
        values = []
        try:
            signature = train_stage._collect_sig(row["inputs"], values)
        except (TypeError, ValueError, OverflowError):
            continue
        if signature != shape or len(values) != n_features:
            continue
        target = label_fn(row)
        if target is None:
            continue
        xs.append(values)
        ys.append(target)
        group_meta = row.get("meta")
        groups.append((group_meta.get(group_field) if isinstance(group_meta, dict) else None)
                      if group_field else None)
    if not xs:
        raise TrainRoundError("验证集没有可用行，动作选型无从测量")
    y = (np.asarray(ys, dtype=np.int64) if task == "classification"
         else np.asarray(ys, dtype=float))
    out = (np.asarray(xs, dtype=float), y, groups, task)
    _ACTION_CACHE[key] = out
    while len(_ACTION_CACHE) > _ACTION_CACHE_MAX:
        _ACTION_CACHE.popitem(last=False)
    return out


def _action_forward(arrays, meta, task, z):
    """与 predict 同口径：clip ±10σ → tanh 隐层 → 输出层 → target_scale/mean。"""
    import numpy as np

    z = np.clip(np.asarray(z, dtype=float), -10.0, 10.0)
    out = np.tanh(z @ arrays["w1"] + arrays["b1"]) @ arrays["w2"] + arrays["b2"]
    if task == "classification":
        return out
    value = (out[:, 0] * float(meta.get("target_scale", 1.0) or 1.0)
             + float(meta.get("target_mean", 0.0) or 0.0))
    return np.maximum(0.0, value)


def _action_metric(task, y, out):
    import numpy as np

    if task == "classification":
        return float(train_mlp._macro_f1(y, np.argmax(out, axis=1), int(out.shape[1])))
    return float(math.sqrt(float(np.mean((out - y) ** 2))))


def _action_run(name, params, x):
    """逐行跑动作，返回变换后的特征矩阵（行序即文件序，causal 状态只向前累积）。"""
    import numpy as np

    prepared = actions.prepare(name, params, stage="predict")
    state, rows = None, []
    for vector in x:
        vector, state = prepared["step"](vector, state, None)
        rows.append(np.asarray(vector, dtype=float))
    return np.vstack(rows)


def _action_run_batch(x):
    """C 线专用：一次行循环同时算出 5 个 EMA α 与 per_cond_seq 共 6 条路径。

    逐元素算术与 `actions._step_ema` / `_step_per_cond`（cond=None）逐步相同：
    第一行 EMA 输出全 0、状态 mean=x/var=0；per_cond_seq 走单组 Welford，
    eps 取注册表默认 1e-8。返回顺序与 `_select_feature_action` 的候选顺序一致。
    """
    import numpy as np

    alphas = [float(alpha) for alpha in ACTION_GRID]
    k = len(alphas)
    values = np.asarray(x, dtype=float)
    n, dim = values.shape
    eps = 1e-8  # actions.REGISTRY 里 streaming_ema_norm / per_cond_seq 的默认 eps
    alpha = np.asarray(alphas, dtype=float).reshape(k, 1)
    mean = np.zeros((k, dim))
    var = np.zeros((k, dim))
    ema_out = np.empty((k, n, dim))
    cond_out = np.empty((n, dim))
    cond_n = 0
    cond_mean = np.zeros(dim)
    cond_m2 = np.zeros(dim)
    for index in range(n):
        vector = values[index]
        if index == 0:
            ema_out[:, index, :] = 0.0
            mean[:] = vector
        else:
            delta = vector - mean
            mean = mean + alpha * delta
            var = (1.0 - alpha) * (var + alpha * delta * delta)
            sd = np.sqrt(np.maximum(var, 0.0))
            ema_out[:, index, :] = (vector - mean) / np.where(sd > 0.0, sd, eps)
        cond_n += 1
        gap = vector - cond_mean
        cond_mean = cond_mean + gap / cond_n
        cond_m2 = cond_m2 + gap * (vector - cond_mean)
        cond_sd = np.sqrt(np.maximum(cond_m2 / cond_n, 0.0))
        cond_out[index] = vector / np.where(cond_sd > 0.0, cond_sd, eps)
    return [ema_out[slot] for slot in range(k)] + [cond_out]


def _select_feature_action(dataset_key, checkpoint_path, data_dir):
    """逐线实测：有利就开、不利就挡。返回可直接落进 metadata / 账本的决定。"""
    import numpy as np

    ref = train_mlp._load_parent(checkpoint_path)
    x, y, groups, task = _action_rows(data_dir, ref, dataset_key)
    arrays = {name: np.asarray(ref["arrays"][name], dtype=float)
              for name in ("w1", "b1", "w2", "b2")}
    meta = ref["meta"]
    mean = np.asarray(ref["arrays"]["mean"], dtype=float).reshape(-1)
    scale = np.asarray(ref["arrays"]["scale"], dtype=float).reshape(-1)
    metric_name = "macro_f1" if task == "classification" else "rmse"

    def improved(value, base):
        if value is None:
            return False
        return value > base + 1e-12 if task == "classification" else value < base - 1e-12

    def score(z_values):
        return _action_metric(task, y, _action_forward(arrays, meta, task, z_values))

    z_identity = np.clip((x - mean) / scale, -10.0, 10.0)
    base = score(z_identity)
    scores = {"identity": base}
    best, best_value, best_z = ("identity", {}), base, z_identity
    any_improved = False
    candidates = [("streaming_ema_norm", {"alpha": alpha}) for alpha in ACTION_GRID]
    candidates.append(("per_cond_seq", {}))
    if dataset_key == "C":
        # C 线性能：6 条 causal 路径共用一次行循环；逐元素算术与 actions 的步进相同。
        measured = list(zip(candidates, _action_run_batch(x)))
    else:
        measured = []
        for name, params in candidates:
            try:
                measured.append(((name, params), _action_run(name, params, x)))
            except actions.ActionError as exc:
                measured.append(((name, params), exc))
    for (name, params), z_values in measured:
        label = name if name == "per_cond_seq" else "ema%.4g" % params["alpha"]
        if isinstance(z_values, actions.ActionError):
            scores[label] = None
            _log("动作 %s 不可用（%s），跳过" % (name, z_values))
            continue
        value = score(z_values)
        scores[label] = value
        if improved(value, base):
            any_improved = True
            if best[0] == "identity" or improved(value, best_value):
                best, best_value, best_z = (name, dict(params)), value, z_values

    # 交付口径：predict 实际会叠加输出侧校准，选型必须看到同一条曲线上的数字。
    cal_pair = train_mlp._calibration_of(meta, task) if task == "regression" else None

    def score_cal(z_values, target=None):
        if cal_pair is None:
            return None
        target = y if target is None else target
        raw = _action_forward(arrays, meta, task, z_values)
        return float(math.sqrt(float(np.mean(
            (np.maximum(0.0, cal_pair[0] * raw + cal_pair[1]) - target) ** 2))))

    delivered_identity = score_cal(z_identity)

    skip_groups, group_scores, group_pattern = [], {}, None
    gated = best_value
    z_final = best_z
    if best[0] != "identity" and dataset_key in ACTION_GROUP:
        _field, group_pattern = ACTION_GROUP[dataset_key]
        if groups and all(item is not None for item in groups):
            keep = np.ones(len(groups), dtype=bool)
            for group in sorted(set(groups)):
                mask = np.asarray([item == group for item in groups])
                before = _action_metric(task, y[mask],
                                        _action_forward(arrays, meta, task, z_identity[mask]))
                after = _action_metric(task, y[mask],
                                       _action_forward(arrays, meta, task, best_z[mask]))
                group_scores[str(group)] = {"identity": before, "action": after,
                                            "n": int(mask.sum())}
                if cal_pair is not None:
                    group_scores[str(group)]["delivered_identity"] = score_cal(
                        z_identity[mask], y[mask])
                    group_scores[str(group)]["delivered_action"] = score_cal(
                        best_z[mask], y[mask])
                if not improved(after, before):
                    skip_groups.append(str(group))
                    keep &= ~mask
            if len(skip_groups) == len(group_scores):
                best, best_value, gated = ("identity", {}), base, base
                z_final = z_identity
                skip_groups = []
            elif skip_groups:
                # 与 predict 同口径：被挡的行完全不进动作，状态只由动作行累积。
                z_gated = z_identity.copy()
                z_gated[keep] = _action_run(best[0], best[1], x[keep])
                gated = score(z_gated)
                z_final = z_gated
            group_pattern = group_pattern if skip_groups else None

    if best[0] == "identity":
        reason = ("候选动作在全部有效分组上退化" if any_improved
                  else "候选动作全部不优于 identity")
    else:
        reason = "逐线实测优于 identity"
        if skip_groups:
            reason += "；分组闸挡下 %s" % ",".join(skip_groups)
    return {"action": best[0], "params": dict(best[1]), "metric": metric_name,
            "identity": base, "selected": best_value, "gated": gated,
            "delivered": score_cal(z_final), "delivered_identity": delivered_identity,
            "skip_groups": skip_groups, "group_pattern": group_pattern,
            "groups": group_scores, "scores": scores, "reason": reason}


def _action_metadata_write(path, selection):
    """只增/删动作字段，权重与其他 metadata 字段原样保留；无变化则不重写。"""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw = arrays.pop("metadata", None)
    meta = {}
    if raw is not None:
        try:
            meta = json.loads(str(np.asarray(raw).tolist()))
        except (ValueError, TypeError):
            raise TrainRoundError("载荷 metadata 不是合法 JSON")
        if not isinstance(meta, dict):
            raise TrainRoundError("载荷 metadata 不是 JSON 对象")
    fields = ("feature_action", "feature_action_params",
              "feature_action_skip_groups", "feature_action_group_pattern")
    before = {name: meta.get(name) for name in fields}
    for name in fields:
        meta.pop(name, None)
    if selection["action"] != "identity":
        meta["feature_action"] = selection["action"]
        meta["feature_action_params"] = dict(selection["params"])
        if selection.get("skip_groups"):
            meta["feature_action_skip_groups"] = list(selection["skip_groups"])
            meta["feature_action_group_pattern"] = selection["group_pattern"]
    after = {name: meta.get(name) for name in fields}
    if before == after:
        return False
    payload = dict(arrays)
    payload["metadata"] = np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    tmp = path + ".action-tmp"
    try:
        with open(tmp, "wb") as handle:
            np.savez(handle, **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def _stamp_feature_action(dataset_key, prepared, data_dir):
    """提交/归档前把动作决定落到载荷，并记进 built 供账本对账。"""
    if not dataset_key:
        return None
    payload_dir = prepared.get("payload_dir") or prepared.get("dir")
    if not payload_dir:
        return None
    path = os.path.join(payload_dir, CHECKPOINT_NAME)
    if not os.path.isfile(path):
        return None
    built = prepared.setdefault("built", {})
    current = _sha256_file(path)
    if built.get("feature_action") and built.get("feature_action_sha256") == current:
        return built["feature_action"]
    try:
        selection = _select_feature_action(dataset_key, path, data_dir)
    except Exception as exc:  # noqa: BLE001 - 选型失败保持 identity，不拖垮提交
        _log("输入动作选型失败（保持 identity）：%s: %s" % (type(exc).__name__, exc))
        selection = {"action": "identity", "params": {}, "metric": None,
                     "identity": None, "selected": None, "gated": None,
                     "skip_groups": [], "group_pattern": None, "groups": {},
                     "scores": {}, "reason": "measurement_failed: %s" % type(exc).__name__}
    changed = _action_metadata_write(path, selection)
    selection = dict(selection)
    selection["metadata_written"] = bool(changed)
    built["feature_action"] = selection
    built["feature_action_sha256"] = _sha256_file(path)
    if changed:
        built["sha256"] = built["feature_action_sha256"]
        _log("输入动作写入载荷：action=%s params=%s metric=%s identity=%s selected=%s%s"
             % (selection["action"], selection["params"], selection["metric"],
                selection["identity"], selection["selected"],
                (" 分组闸=%s" % selection["skip_groups"]) if selection["skip_groups"] else ""))
    else:
        _log("输入动作保持 identity（载荷未改写）：%s" % selection["reason"])
    return selection


def run(rnd, data_dir=None, work_dir=None, pessimism_k=None):
    """执行一轮训练。返回 ledger（同时落盘）。异常即整轮不提交。

    `pessimism_k` 覆盖悲观界系数 k（优先级：本参数 > CONTEST_PESSIMISM_K > PESSIMISM["k"]）。
    """
    round_index = int(getattr(rnd, "round_index", 1) or 1)
    work = work_dir or _work_dir(rnd)
    os.makedirs(work, exist_ok=True)
    k, k_source = _pessimism_k(pessimism_k)

    parent = _parent_checkpoint(rnd)
    if fact_probe is not None:
        fact_probe.train_parent_metadata(parent["path"])
    _log("父检查点 %s sha256=%s" % (parent["rel"], parent["sha256"][:16]))

    # 状态副本必须在选臂之前读：上一轮各臂统计是悲观界的输入。
    state_path = _fspath(_rpc_guard(
        "state_workspace", rnd.state_workspace,
        os.path.join(work, "round-%s" % round_index, "state")))
    os.makedirs(state_path, exist_ok=True)
    previous_state = _read_state(state_path)
    arms_in = previous_state.get("arms") or {}
    weight_history = _read_history(state_path)  # v7 J1：state 里的历史候选权重

    data = data_dir or _data_dir(rnd)
    stage = train_stage.run(data_dir=data, out_dir=os.path.join(work, "round-%s" % round_index),
                            task=getattr(rnd, "dataset_key", None))
    detection = stage["detector_report"]
    recommendation = {
        "candidate": stage["candidate"],
        "action": stage["action"],
        "hypothesis": stage["selector_trace"]["chosen"]["hypothesis"],
        "rule": next((r["id"] for r in stage["selector_trace"]["rules"] if r["ok"]), None),
    }

    history, history_ids = _history(rnd)
    history_digest = _digest({"ids": history_ids, "records": history})

    data_summary = {
        "n_train": detection["inputs"]["n_train"],
        "n_validation": detection["inputs"]["n_val"],
        "features": detection["inputs"]["features"],
        "history_rows": len(history),
        "ledger_history": history[-3:],
    }
    # 失败模式两路：text 喂 prompt（不含数字），evidence 只落 configuration。
    # 数据只取本轮检测器读数与上一轮已记录的失败签名，不重算、不参与任何裁决。
    prior_failure = (previous_state or {}).get("failure_modes")
    prior_failure = prior_failure if isinstance(prior_failure, dict) else {}
    failure = failure_modes.build({
        "drift": detection.get("detectors"),
        "dead_classes": prior_failure.get("dead_classes") or [],
        "late_segment": prior_failure.get("late_segment") or [],
    })
    data_summary["failure_modes"] = failure["text"]
    signature = _signature(getattr(rnd, "dataset_key", None),
                           detection["inputs"]["features"])
    context = _review_context(round_index, arms_in, history_ids, previous_state, signature)
    phase = context["phase"]
    recent_rounds = list((previous_state or {}).get("rounds") or [])
    if not recent_rounds:
        recent_rounds = [{"signature": "—", "arm": item.get("candidate", "—"),
                          "delta": None, "verdict": "—", "ledger_id": item.get("id")}
                         for item in history[-3:]]
    # 状态卡的指标名必须是真实口径：写死字面量 "metric" 会让模型看到一个
    # 不存在的指标名，也没法把分类与回归的分数解释成同一个尺度。
    metric_name = ("macro_f1"
                   if detection["inputs"].get("task") == "classification" else "rmse")
    card = llm_cage.status_card(
        round_index, getattr(rnd, "dataset_key", None),
        metric_name, arms_in, recent_rounds, phase,
        context["signature_seen"])

    # 提案：首提一次；解析/校验失败把原文与错误回灌，只修复一次。
    calls = []
    review = None
    last_text = ""
    last_errors = []
    failed_combos = list(context.get("failed_combos") or [])
    for attempt_no in range(1, MAX_MODEL_CALLS + 1):
        if attempt_no == 1:
            messages = llm_cage.build_messages(
                card, data_summary, detection.get("detectors"), recommendation,
                round_index, failed_combos)
        else:
            messages = llm_cage.repair_messages(
                card, last_text, last_errors, data_summary, detection.get("detectors"),
                recommendation, round_index, failed_combos)
        call = _call_model(rnd, messages)
        calls.append(call)
        last_text = call["text"]
        # 数字白名单只认提示词里出现过的数值：模型复述检测器/账本读数属合法引用。
        review = llm_cage.review(
            last_text, dict(context, numeric_frame=llm_cage.frame_numbers(messages)))
        if review["status"] == "ok":
            break
        last_errors = review["errors"]
        _log("模型提案第%d次被判无效：%s" % (attempt_no, "；".join(last_errors)))

    proposal = review["proposal"] if review and review["status"] == "ok" else None
    proposal_arm = proposal["arm"] if proposal else None
    raw_review = review or {}
    offered_script = bool((raw_review.get("raw") or {}).get("build_script"))
    # 被接受时全文已在 model.proposal/hypothesis/kill_condition 里，不重复存整份；
    # 被拒绝时把整份提案与原文片段都留下——拒绝理由必须可复核。
    rejected = proposal is None
    gate = {
        "phase": phase,
        "signature": signature,
        "signature_seen": context["signature_seen"],
        "remaining_rounds": context["remaining_rounds"],
        "proposal": (raw_review.get("raw") if rejected else proposal_arm),
        "proposal_arm": proposal_arm,
        "raw_sha256": raw_review.get("raw_sha256"),
        "source": calls[-1]["source"] if calls else "model",
        "status": raw_review.get("status", "unusable"),
        "failure_kind": raw_review.get("failure_kind", "none"),
        "proposal_arm_raw": raw_review.get("proposal_arm_raw"),
        "errors": list(raw_review.get("errors") or []),
        "calls": [entry for item in calls for entry in _rpc_entries(item)],
        "allowed_digits": list(raw_review.get("allowed_digits") or []),
        "allowed_digits_source": raw_review.get("allowed_digits_source",
                                                "round_frame"),
        "state": None,
        # 第一道门（主动关闭）+ 阶段门（CLOSED_ROUNDS）都要在账本里留下代码事实。
        "runtime_code_gen": {
            "enabled": llm_cage.runtime_code_gen_enabled(),
            "attempted": offered_script,
            "rejected": bool(offered_script and not llm_cage.runtime_code_gen_enabled()),
        },
    }
    if rejected:
        gate["raw_excerpt"] = raw_review.get("raw_excerpt")
    retries = [item["empty_retry"] for item in calls if item.get("empty_retry")]
    if retries:
        gate["empty_retry"] = json.dumps(retries[0], ensure_ascii=False,
                                         sort_keys=True)

    if proposal_arm:
        chosen, policy = proposal_arm, "model_proposal"
    else:
        chosen, policy = recommendation["candidate"], "fallback_rule"
        _log("模型提案不可用：退回规则选择器 %s" % chosen)

    # 状态闸门：剩余 1 轮 / 签名未见 / 冷却 / 只有失败记录。
    if chosen != PARENT:
        allowed, gate_reason, gate_evidence = llm_cage.state_gate(chosen, context)
        gate["state"] = {"ok": allowed, "reason": gate_reason, "evidence": gate_evidence}
        if not allowed:
            _log("状态闸门否掉候选 %s（%s：%s）；本轮退回父候选"
                 % (chosen, gate_reason, gate_evidence))
            policy = "cage_%s" % gate_reason
            chosen = PARENT

    # 悲观界：候选臂的 mean - k*sd 低于父候选时一律不采纳；未观测的臂不参与比较。
    parent_bound = _arm_bound(arms_in.get(PARENT) or {}, k)
    rejection = None
    if chosen != PARENT:
        stats = arms_in.get(chosen) or {}
        bound = _arm_bound(stats, k)
        cooldown = int(stats.get("cooldown", 0) or 0)
        if cooldown > 0:
            rejection = {"candidate": chosen, "reason": "cooldown", "cooldown": cooldown,
                         "bound": bound, "parent_bound": parent_bound}
        elif bound is not None and parent_bound is not None and bound < parent_bound:
            rejection = {"candidate": chosen, "reason": "bound_below_parent",
                         "bound": bound, "parent_bound": parent_bound}
        if rejection is not None:
            _log("悲观界否掉候选 %s（%s）；本轮退回父候选" % (chosen, rejection["reason"]))
            policy = "pessimism_%s" % rejection["reason"]
            chosen = PARENT

    pessimism = {
        "k": k,
        "k_source": k_source,
        "parent_bound": parent_bound,
        "bounds": {arm: _arm_bound(arms_in.get(arm) or {}, k)
                   for arm in (PARENT,) + CANDIDATE_IDS},
        "cooldown": {arm: int((arms_in.get(arm) or {}).get("cooldown", 0) or 0)
                     for arm in CANDIDATE_IDS},
        "rejected": rejection,
        "fallback": None,
    }

    model_note = proposal["hypothesis"] if proposal else None
    params = dict((proposal or {}).get("params") or {})
    script = (proposal or {}).get("build_script")
    if proposal_arm != chosen:
        params, script = {}, None

    def prepare(arm, arm_params, build_script, tag):
        """在 /workspace 沙盒里造候选，跑性质测试与红队五问；全过才返回。"""
        if build_script and not llm_cage.runtime_code_gen_enabled():
            # 第一道门：运行时生成默认关闭。显式拒绝并落账，绝不静默跳过脚本。
            gate["runtime_code_gen"]["attempted"] = True
            gate["runtime_code_gen"]["rejected"] = True
            _log("运行时生成通道默认关闭（RUNTIME_CODE_GEN_ENABLED=False）："
                 "显式拒绝 build_script，不生成候选")
            raise TrainRoundError(
                "运行时生成已关闭：拒绝执行 build_script（RUNTIME_CODE_GEN_ENABLED=False）")
        cand_dir = _candidate_dir(work, round_index, history_digest, tag, parent["sha256"])
        scripted = bool(build_script)
        if scripted:
            result = llm_cage.run_build_script(
                build_script, cand_dir,
                env_extra={"CONTEST_PARENT_CHECKPOINT": parent["path"],
                           "CONTEST_CANDIDATE_DIR": cand_dir})
            if not result["ok"]:
                raise TrainRoundError("build_script 失败：%s；stderr=%s"
                                      % (result["error"], (result["stderr"] or "")[:300]))
            target = os.path.join(cand_dir, CHECKPOINT_NAME)
            if not os.path.isfile(target):
                raise TrainRoundError("build_script 未产出 %s" % CHECKPOINT_NAME)
            try:
                diff = _weights_diff(parent["path"], target)
            except Exception as exc:  # noqa: BLE001 - 比不了就如实标"未核对"
                _log("脚本候选与父产物的权重差异无法核对：%s" % exc)
                diff = {"diff": {}, "changed": None}
            built = {"weights_changed": diff["changed"], "action": None, "params": {},
                     "sha256": _sha256_file(target), "build": "script",
                     "diff": diff["diff"]}
        else:
            built = _build_candidate(parent, arm, cand_dir, arm_params,
                                     data_dir=data, round_index=round_index,
                                     dataset_key=getattr(rnd, "dataset_key", None),
                                     history=weight_history)
        budget = _budget(cand_dir, _candidate_cap(rnd))
        check = llm_cage.check_candidate(
            cand_dir, expected_features=detection["inputs"]["features"],
            parent_path=parent["path"],
            arm=getattr(rnd, "dataset_key", None))
        if not check["ok"]:
            raise TrainRoundError("候选性质测试未过：%s" % "；".join(check["errors"]))
        if scripted:
            # 脚本产出的候选以检查点自述为准：账本里的 action 必须是真实动作名，不能写 "script"。
            built["action"] = check["facts"].get("action")
            built["params"] = check["facts"].get("params") or {}
        questions = llm_cage.red_team(check["facts"],
                                      {"arm": arm, "params": arm_params}, context)
        failed = [item for item in questions if not item["ok"]]
        if failed:
            raise TrainRoundError("红队未过：%s" % "；".join(
                "%s（%s）" % (item["q"], item["citation"]) for item in failed))
        return {"dir": cand_dir, "built": built, "budget": budget,
                "check": check, "red_team": questions}

    prepared = None
    build_errors = []
    try:
        prepared = prepare(chosen, params, script, chosen)
    except Exception as exc:
        build_errors.append("%s: %s" % (type(exc).__name__, str(exc)[:400]))
        _log("候选 %s 构建/自检失败：%s" % (chosen, exc))

    # 构建失败也走同一次修复预算：把错误原文回灌，只修一次。
    if prepared is None and chosen != PARENT and len(calls) < MAX_MODEL_CALLS:
        _log("把构建错误回灌模型做一次修复")
        retry_messages = llm_cage.repair_messages(
            card, last_text, build_errors, data_summary,
            detection.get("detectors"), recommendation, round_index, failed_combos)
        call = _call_model(rnd, retry_messages)
        calls.append(call)
        gate["calls"].extend(_rpc_entries(call))
        retry = llm_cage.review(
            call["text"],
            dict(context, numeric_frame=llm_cage.frame_numbers(retry_messages)))
        if (retry.get("raw") or {}).get("build_script"):
            gate["runtime_code_gen"]["attempted"] = True
            if not llm_cage.runtime_code_gen_enabled():
                gate["runtime_code_gen"]["rejected"] = True
        if retry["proposal"]:
            arm2 = retry["proposal"]["arm"]
            allowed, gate_reason, gate_evidence = llm_cage.state_gate(arm2, context)
            if allowed:
                try:
                    prepared = prepare(arm2, retry["proposal"]["params"],
                                       retry["proposal"].get("build_script"),
                                       "%s-repair" % arm2)
                    chosen, policy = arm2, "model_proposal_repair"
                    proposal_arm = arm2
                    proposal = retry["proposal"]
                    model_note = proposal["hypothesis"]
                    gate["repair"] = {"arm": arm2, "status": retry["status"],
                                      "source": call["source"],
                                      "failure_kind": retry.get("failure_kind"),
                                      "raw_excerpt": retry.get("raw_excerpt")}
                except Exception as exc:
                    build_errors.append("%s: %s" % (type(exc).__name__, str(exc)[:400]))
                    _log("修复后的候选 %s 仍未过闸门：%s" % (arm2, exc))
            else:
                gate["repair"] = {"arm": arm2, "status": "state_gate_%s" % gate_reason,
                                  "source": call["source"],
                                  "failure_kind": "state_gate_%s" % gate_reason,
                                  "raw_excerpt": retry.get("raw_excerpt")}
        else:
            gate["repair"] = {"status": retry["status"], "errors": retry["errors"],
                              "failure_kind": retry.get("failure_kind"),
                              "source": call["source"],
                              "raw_excerpt": retry.get("raw_excerpt")}

    # 双调用：提案通过后，再让模型按五问清单做一次红队；无引用的结论不计入。
    if prepared is not None and chosen != PARENT and len(calls) < MAX_MODEL_CALLS:
        call = _call_model(rnd, llm_cage.red_team_messages(
            {"arm": chosen, "params": params, "hypothesis": model_note or ""},
            prepared["check"]["facts"], failed_combos))
        calls.append(call)
        gate["calls"].extend(_rpc_entries(call))
        model_review = llm_cage.parse_red_team(call["text"])
        decision = llm_cage.red_team_gate(model_review, prepared["red_team"])
        # 五问的确定性结果已在 cage.red_team 里完整保留；gate 只留判定与回指，
        # 避免同一份证据在 configuration 里存两遍（8KiB 预算）。
        decision = dict(decision)
        decision.pop("deterministic", None)
        decision["deterministic_ref"] = "cage.red_team"
        gate["red_team"] = decision
        prepared["red_team_gate"] = decision
        if not decision["ok"]:
            _log("红队否掉候选 %s：%s" % (chosen, "；".join(decision["reasons"])[:300]))
            prepared = None

    # 规则选择器兜底；规则候选也不过闸门就退回父候选。
    if prepared is None:
        rule_arm = recommendation["candidate"]
        allowed, gate_reason, gate_evidence = llm_cage.state_gate(rule_arm, context)
        if allowed and rule_arm != chosen:
            try:
                prepared = prepare(rule_arm, {}, None, rule_arm)
                chosen, policy = rule_arm, "fallback_rule"
                model_note = None
            except Exception as exc:
                _log("规则候选 %s 也未过闸门：%s" % (rule_arm, exc))
        if prepared is None:
            prepared = prepare(PARENT, {}, None, "PARENT-retry")
            chosen = PARENT
            policy = "fallback_parent_after_rule"
            model_note = None

    # 四臂同尺：每轮把 PARENT/A/B/C 都构造出来，在同一份验证切片上各打一次分。
    # 评分本身不改选臂；只有在模型提案不可用、规则臂兜底、且回归线实测存在
    # 严格更优臂时，才用**同一组闸门**（state_gate / 悲观界 / prepare 的候选自检）
    # 接管载荷。阈值与 rel_l2 上限不动。
    try:
        arm_scores = _score_arms(parent, data, round_index,
                                 getattr(rnd, "dataset_key", None),
                                 work, chosen, prepared)
    except Exception as exc:  # noqa: BLE001 - 评分失败不能拖垮本轮提交
        _log("四臂评分失败（忽略，不参与选臂）：%s: %s"
             % (type(exc).__name__, str(exc)[:200]))
        arm_scores = {"slice": {"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])},
                      "arms": {}, "best": None, "chosen": chosen,
                      "chosen_is_best": None, "best_candidates": [], "best_tied": None}

    if (policy == "fallback_rule" and chosen != PARENT
            and getattr(rnd, "dataset_key", None) in ("B", "C")):
        takeover = selector.best_scored_arm(arm_scores)
        if takeover:
            stats = arms_in.get(takeover) or {}
            cooldown = int(stats.get("cooldown", 0) or 0)
            bound = _arm_bound(stats, k)
            allowed, gate_reason, gate_evidence = llm_cage.state_gate(takeover, context)
            reason = None
            if cooldown > 0:
                reason = "cooldown"
            elif bound is not None and parent_bound is not None and bound < parent_bound:
                reason = "bound_below_parent"
            elif not allowed:
                reason = "state_gate_%s" % gate_reason
            delta = selector.best_arm_delta(arm_scores, takeover)
            if reason is None:
                try:
                    prepared_takeover = prepare(takeover, {}, None,
                                                "arm-best-%s" % takeover)
                    arm_scores["takeover"] = {
                        "from": chosen, "to": takeover, "applied": True,
                        "metric": (arm_scores.get("slice") or {}).get("metric"),
                        "delta": delta,
                        "gate": {"ok": allowed, "reason": gate_reason}}
                    _log("规则臂 %s 非四臂最优；按同一闸门接管 %s（改进 %.6f）"
                         % (chosen, takeover, delta or 0.0))
                    prepared = prepared_takeover
                    chosen = takeover
                    policy = "arm_score_takeover"
                    model_note = None
                except Exception as exc:  # noqa: BLE001 - 接管失败保持原臂
                    arm_scores["takeover"] = {
                        "from": chosen, "to": takeover, "applied": False,
                        "reason": "prepare_failed: %s" % str(exc)[:160],
                        "delta": delta,
                        "gate": {"ok": allowed, "reason": gate_reason}}
                    _log("接管臂 %s 未过闸门（%s）；保持 %s" % (takeover, exc, chosen))
            else:
                arm_scores["takeover"] = {
                    "from": chosen, "to": takeover, "applied": False,
                    "reason": reason, "delta": delta,
                    "gate": {"ok": allowed, "reason": gate_reason}}
                _log("接管臂 %s 被悲观界/状态闸门否掉（%s）；保持 %s"
                     % (takeover, reason, chosen))
        arm_scores["chosen"] = chosen
        arm_scores["chosen_is_best"] = bool(arm_scores.get("best") == chosen)

    # ---- v6：输出层偏置候选族（当轮切片测量 + EMA，全程无梯度）----
    # 候选生成 → 性质测试 → 用当轮切片逐个测量 → 选最优 → 提交 EMA 值。
    # 两级兜底：切片不可用 → 常数 δ；候选全被 cage 拒 → 退回父产物。
    # 放在四臂评分与接管之后：保证四臂在同一"未校准"口径上比较，偏置只作用于最终臂。
    bias_record = None
    bias_ema = None
    if chosen != PARENT:
        try:
            bias_record = bias_family.apply(
                getattr(rnd, "dataset_key", None), data, parent["path"], prepared,
                work, round_index, previous_state=previous_state,
                expected_features=detection["inputs"]["features"])
        except Exception as exc:  # noqa: BLE001 - 偏置是增益项，失败按原候选提交
            _log("偏置候选族失败（忽略，提交本轮候选）：%s: %s"
                 % (type(exc).__name__, str(exc)[:200]))
            bias_record = {"schema": "contest.v6.bias-family", "status": "error",
                           "applied": False,
                           "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
        if bias_record and bias_record.get("status") == "parent_fallback":
            _log("偏置候选族候选全被 cage 拒；本轮退回父产物")
            # 回退父产物时，本轮没有任何校准被写进载荷：applied/uploaded 必须同时归零，
            # 否则账本会声称"校准已生效"而上传的是未校准的父检查点。
            bias_record["applied"] = False
            bias_record["uploaded"] = False
            bias_record["upload_note"] = "候选全被 cage 拒，本轮上传父产物，校准未生效"
            prepared = prepare(PARENT, {}, None, "PARENT-bias-fallback")
            chosen = PARENT
            policy = "bias_family_parent_fallback"
            model_note = None
        else:
            bias_ema = _bias_ema_state(previous_state,
                                       (bias_record or {}).get("arm"),
                                       ((bias_record or {}).get("ema") or {}).get("state"))
    else:
        bias_record = {"schema": "contest.v6.bias-family",
                       "arm": str(getattr(rnd, "dataset_key", None) or ""),
                       "status": "skipped_parent", "applied": False}

    # 偏置族回退可能把最终臂改成 PARENT；四臂表里的 chosen/判定按最终臂对齐。
    if isinstance(arm_scores, dict) and arm_scores.get("arms"):
        arm_scores["chosen"] = chosen
        arm_scores["chosen_is_best"] = bool(arm_scores.get("best") == chosen)

    # 本地历史最优：副本在 state 目录里，不向平台读回任何历史产物（读取次数 0）。
    best_record = None
    best_state = None
    try:
        best_record = _apply_local_best(
            rnd, prepared, _read_best(state_path, getattr(rnd, "dataset_key", None)),
            state_path, data, round_index, getattr(rnd, "dataset_key", None),
            parent["path"], detection["inputs"]["features"])
        best_state = best_record
    except Exception as exc:  # noqa: BLE001 - 副本机制失败就按原样提交本轮候选
        _log("本地历史最优比较失败（忽略，提交本轮候选）：%s: %s"
             % (type(exc).__name__, exc))
        best_record = {"available": False,
                       "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
        # 错误记录只进本轮 ledger；state["best"] 保留上一轮，不能被错误覆盖。

    # v7 J5：R10 候选族同场测量。族内每份权重都在**当轮切片**上打分后再定上传物；
    # A 只用端点（父、R1、R2），B/C 加历史对插值（族 ≤8）。失败按 _apply_local_best 结果提交。
    r10_record = None
    if round_index == R10_ROUND:
        try:
            r10_record = _r10_family(
                prepared, weight_history, parent, data, round_index,
                getattr(rnd, "dataset_key", None), work,
                cap=_candidate_cap(rnd), allow_apply=(chosen != PARENT))
        except Exception as exc:  # noqa: BLE001 - 族是增益项，不拖垮本轮提交
            _log("R10 候选族测量失败（忽略，按本地最优结果提交）：%s: %s"
                 % (type(exc).__name__, str(exc)[:200]))
            r10_record = {"status": "error",
                          "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

    # v7 ④：载荷定稿后、归档与提交之前，逐线实测并写入输入动作（不利则保持 identity）。
    _stamp_feature_action(getattr(rnd, "dataset_key", None), prepared, data)

    # v7 J1：把本轮实际上传的载荷另存进 state/history，供下一轮历史对与 R10 候选族。
    # 失败不拖垮提交（候选族是增益项），但要在日志里留痕。
    try:
        archived = _archive_history(
            state_path,
            os.path.join(prepared.get("payload_dir") or prepared["dir"], CHECKPOINT_NAME),
            round_index,
            metrics=(best_record or {}).get("current_metrics"))
        if archived:
            _log("历史候选已归档：第%s轮 sha=%s" % (round_index, archived["sha256"][:16]))
        # v7 J1c：父权重被 keep 守卫留下时，把本轮"训练终点"也归档成独立端点。
        # 只归档上传载荷会让历史端点同 sha 去重后只剩父一条，插值族无方向可用。
        endpoint = ((prepared.get("built") or {}).get("training") or {}).get("trained_endpoint")
        if endpoint:
            archived_ep = _archive_history(
                state_path, endpoint["path"], round_index,
                metrics={"round": int(round_index), "rel_l2": endpoint.get("rel_l2"),
                         "role": "trained_endpoint"},
                tag="-endpoint")
            if archived_ep:
                _log("训练终点已归档：第%s轮 sha=%s rel_l2=%s"
                     % (round_index, archived_ep["sha256"][:16], endpoint.get("rel_l2")))
    except Exception as exc:  # noqa: BLE001
        _log("历史候选归档失败（忽略）：%s: %s" % (type(exc).__name__, str(exc)[:200]))

    # 偏置只作用于本轮训练结果；若最终上传的是历史最优副本，如实标注。
    if isinstance(bias_record, dict):
        if isinstance(best_record, dict) and best_record.get("payload") == "best":
            bias_record["uploaded"] = False
            bias_record["upload_note"] = "本轮上传历史最优副本；偏置只作用于本轮训练结果"
        else:
            bias_record["uploaded"] = bool(bias_record.get("applied"))

    def make_configuration(arm, prep, pol):
        if arm == PARENT:
            hyp = "沿用父检查点，不引入新的特征动作"
            if pol == "fallback_parent_after_rule":
                hyp += "（模型提案与规则候选均未通过闸门）"
            summary = "未改动父检查点：checkpoint.npz 与父产物逐字节一致"
        else:
            training = prep["built"].get("training") or {}
            if training:
                val = training["val"]
                hyp = ("在官方 MLP 结构上续训（%s，%d 步/%d 轮）：验证 %s %.6g→%.6g"
                       % (training["scope"], training["hyper"]["steps"],
                          training["hyper"]["epochs_run"], val["metric"],
                          val["parent"], val["best"]))
                if not prep["built"]["weights_changed"]:
                    hyp = "续训未优于父产物，按验证集保留父权重（%s）" % hyp
                if pol.startswith("model_proposal") and model_note:
                    hyp = "%s；模型理由：%s" % (hyp, model_note)
                if prep["built"]["weights_changed"]:
                    # 不能写死 w1：C 线是头重解，w1 逐位不变，写死会输出"w1 变化 0%"
                    # 而被读成回退。按实际 diff 报改动最大的张量，并交代 soup/NAS 的取舍。
                    diffs = {name: ((training.get("diff") or {}).get(name) or {}).get("rel_l2") or 0.0
                             for name in ("w1", "b1", "w2", "b2")}
                    name, rel = max(diffs.items(), key=lambda item: item[1])
                    touched = [key for key, value in diffs.items() if value > 1e-12]
                    bias = prep["built"].get("bias") or {}
                    if rel <= 1e-12 and bias.get("applied"):
                        # 续训原地未动权重，本轮的全部改动来自输出侧校准。
                        summary = ("只在输出侧做%s：%s（当轮切片 K=%s 测后取 "
                                   "EMA α=%.1f，来源 %s）；权重、mean/scale、结构均未改"
                                   % (_bias_label(bias), _bias_text(bias),
                                      bias.get("K"), bias.get("alpha", 0.3),
                                      (bias.get("chosen") or {}).get("source")))
                    else:
                        summary = ("只更新权重：%s 相对父产物 L2 变化 %.3g%%（改动张量 %s）；"
                                   "结构、mean/scale、动作均未改；选轮依据验证集（self-set）"
                                   % (name, rel * 100.0, "/".join(touched) or "无"))
                        soup = training.get("soup") or {}
                        pair_id = soup.get("pair")
                        if pair_id and pair_id != "parent-current":
                            # v7 J1：上传的是历史对插值，不是本轮训练端点——必须写明，
                            # 否则"续训了 N 步"会被读成上传权重的训练步数。
                            summary += ("；上传权重为历史对插值 %s（端点 %s，α=%.2f，"
                                        "rel_l2=%.3f，上限 %.2f），非本轮训练端点"
                                        % (pair_id, "/".join(soup.get("endpoints") or []),
                                           soup.get("alpha") or 0.0,
                                           soup.get("rel_l2") or 0.0,
                                           soup.get("cap") or 0.0))
                        elif soup.get("alpha") is not None and soup["alpha"] < 1.0:
                            summary += ("；soup α=%.2f（拼接 rel_l2=%.3f，上限 %.2f）："
                                        "公开验证增益按比例收缩，换取隐藏集迁移"
                                        % (soup["alpha"], soup.get("rel_l2") or 0.0,
                                           soup.get("cap") or 0.0))
                        nas = training.get("nas") or {}
                        if nas.get("delta_b2"):
                            summary += "；b2 偏置 δ=%.3g（NAS 对齐）" % nas["delta_b2"]
                        if bias.get("applied"):
                            summary += ("；另按当轮切片做%s %s（K=%s，EMA α=%.1f，来源 %s）"
                                        % (_bias_label(bias), _bias_text(bias),
                                           bias.get("K"), bias.get("alpha", 0.3),
                                           (bias.get("chosen") or {}).get("source")))
                else:
                    if prep["built"].get("calibrated"):
                        bias = prep["built"].get("bias") or {}
                        summary = ("续训未过验证集门槛，权重与父产物逐字节一致；"
                                   "另按当轮切片做%s %s（K=%s，EMA α=%.1f，来源 %s）"
                                   % (_bias_label(bias), _bias_text(bias),
                                      bias.get("K"), bias.get("alpha", 0.3),
                                      (bias.get("chosen") or {}).get("source")))
                    else:
                        summary = ("续训未过验证集门槛，候选检查点与父产物逐字节一致："
                                   "权重、mean/scale、结构均未改")
            else:
                hyp = selector.CANDIDATES[arm]["hypothesis"]
                if pol.startswith("model_proposal") and model_note:
                    hyp = "%s；模型理由：%s" % (hyp, model_note)
                if prep["built"].get("build") == "script":
                    # 构建脚本可能真训了权重：不能复用"权重未训练"的模板，按实际 diff 写。
                    diffs = prep["built"].get("diff") or {}
                    changed = prep["built"].get("weights_changed")
                    numeric = {name: value for name, value in diffs.items()
                               if isinstance(value, (int, float)) and math.isfinite(value)}
                    if changed is None:
                        hyp = "由模型构建脚本产出候选；权重是否改动未能核对"
                        summary = ("由构建脚本产出候选（sha=%s）；权重相对父产物的变化未能核对，"
                                   "动作=%s" % (prep["built"]["sha256"][:8],
                                                prep["built"]["action"]))
                    elif changed and numeric:
                        name, rel = max(numeric.items(), key=lambda item: item[1])
                        touched = [key for key, value in numeric.items() if value > 1e-12]
                        hyp = "由模型构建脚本产出候选（权重相对父产物有改动）"
                        summary = ("由构建脚本产出：权重相对父产物 L2 变化 %.3g%%（改动张量 %s）；"
                                   "动作=%s" % (rel * 100.0, "/".join(touched) or "无",
                                               prep["built"]["action"]))
                    elif changed:
                        hyp = "由模型构建脚本产出候选（权重形状或取值有改动）"
                        summary = ("由构建脚本产出：权重相对父产物有改动（无法按 rel_l2 量化）；"
                                   "动作=%s" % prep["built"]["action"])
                    else:
                        hyp = "由模型构建脚本产出候选（权重与父产物一致）"
                        summary = ("由构建脚本产出：各权重张量与父产物一致（rel_l2 均为 0）；"
                                   "动作=%s" % prep["built"]["action"])
                else:
                    summary = ("仅把特征动作改为 %s；权重未训练，仍为父检查点原值"
                               % prep["built"]["action"])
        # 平台报告页"本轮改动"一定显示 change_summary（声明四件套），而 configuration
        # 不展示：把闸门判定必需的三件套写在这里，B/C 块状的两种成因据此可分。
        payload = prep["built"].get("payload") or "current"
        payload_round = prep["built"].get("payload_round")
        bias = prep["built"].get("bias") or {}
        if bias.get("uploaded"):
            # 只有校准真进了载荷才写进 hypothesis；applied 只说明候选被改过。
            # 提交来源随臂不同：C 是 EMA，B 是 4 折折外最优（cv）。
            src = (bias.get("chosen") or {}).get("source") or "ema"
            hyp = "%s；输出侧按当轮切片选%s（%s）" % (hyp, _bias_label(bias), src)
        if payload_round is not None:
            # 载荷是历史最优副本：不再声称"与父产物逐字节一致"（那描述的是本轮候选，
            # 不是上传物），改为按 sha 对账后的载荷来源陈述。
            same = _digest_hex(prep["built"]["sha256"]) == _digest_hex(parent["sha256"])
            if arm == PARENT:
                summary = ("本轮候选为父产物（未训练）；上传第%s轮历史最优副本作为载荷"
                           "（sha=%s，%s父产物）"
                           % (payload_round, prep["built"]["sha256"][:8],
                              "等于" if same else "不等于"))
            else:
                summary = ("本轮上传第%s轮历史最优副本作为载荷（sha=%s，%s父产物）；"
                           "本轮训练结果在本轮验证集上未更优，未作为载荷"
                           % (payload_round, prep["built"]["sha256"][:8],
                              "等于" if same else "不等于"))
            hyp = "%s；本轮提交历史最优副本（第%s轮）" % (hyp, payload_round)
        # v7 J5：R10 上传物来自候选族时，summary 必须交代端点、α 与 steps 语义。
        # 只在真的换了上传物时写；族仅测量（已判 PARENT）时由 configuration.r10 交代。
        if payload == "family" and (r10_record.get("selected") or {}).get("applied"):
            sel = r10_record["selected"]
            summary = ("%s [R10 同场：族内最优 %s α=%.2f（端点 %s）；%s]"
                       % (summary, sel.get("pair"), sel.get("alpha") or 0.0,
                          "/".join(sel.get("endpoints") or []),
                          "产物为权重插值，steps 是排程锚点非训练步数"
                          if sel.get("interpolated") else "产物为端点权重"))
        wc = prep["built"]["weights_changed"]
        summary = "%s [wc=%s payload=%s p=%s c=%s]" % (
            summary, "?" if wc is None else str(1 if wc else 0),
            payload, parent["sha256"][:8], prep["built"]["sha256"][:8])
        cfg = {
            "round_index": round_index,
            "dataset_key": getattr(rnd, "dataset_key", None),
            "policy": pol,
            "candidate": arm,
            "action": prep["built"]["action"],
            "action_params": prep["built"].get("params") or {},
            "weights_changed": prep["built"]["weights_changed"],
            "payload": payload,
            "payload_round": payload_round,
            "parent_checkpoint_sha256": parent["sha256"],
            "candidate_checkpoint_sha256": prep["built"]["sha256"],
            "trained_checkpoint_sha256": prep["built"].get("trained_sha256"),
            "best": best_record,
            "history_digest": history_digest,
            "history_ids": history_ids,
            "model": {
                "call_id": calls[0]["call_id"],
                "call_ids": _all_call_ids(calls, rnd),
                "max_tokens": calls[0]["max_tokens"],
                "enable_thinking": calls[0].get("enable_thinking", False),
                "proposal": proposal_arm,
                "status": gate["status"],
                "hypothesis": (proposal or {}).get("hypothesis"),
                "kill_condition": (proposal or {}).get("kill_condition"),
                "confidence": (proposal or {}).get("confidence"),
                "note": model_note,
                "phase": phase,
                "gate": gate,
            },
            "cage": {
                "check": prep["check"],
                "red_team": prep["red_team"],
                "build_errors": build_errors,
            },
            "detectors": detection["detectors"],
            "detector_registry": detection.get("detector_registry_ledger"),
            "failure_modes": failure["evidence"],
            "data_slice": _slice_card(data, getattr(rnd, "dataset_key", None), rnd),
            # 四臂同尺的逐轮分数：每个臂都在这份验证切片上量过，失败原因逐臂记录。
            "arm_scores": arm_scores,
            # v6：每轮完整候选族网格结果（每个 δ 的当轮测量得分）+ EMA 前后值 + 兜底标记。
            "bias_family": bias_record,
            "selector": {"candidate": recommendation["candidate"],
                         "rule": recommendation["rule"],
                         "thresholds": stage["selector_trace"]["thresholds"]},
            "pessimism": pessimism,
            "state_degraded": bool(context.get("state_degraded")),
            "budget": prep.get("payload_budget") or prep["budget"],
            "seed": round_index,
        }
        # v7 ④：本轮输入动作决定（逐线实测的数字、分组闸与保持 identity 的原因）。
        selection = prep["built"].get("feature_action")
        if selection:
            cfg["feature_action"] = {
                "a": selection.get("action"),
                "p": selection.get("params") or {},
                "m": selection.get("metric"),
                "id": selection.get("identity"),
                "sel": selection.get("selected"),
                "g": selection.get("gated"),
                "skip": selection.get("skip_groups") or [],
                "grp": selection.get("groups") or {},
                "why": selection.get("reason"),
                "w": 1 if selection.get("metadata_written") else 0,
            }
            # 交付口径（校准后）只对带 calibration 的回归产物有意义；分类线与未带
            # 校准的产物省略这两个键（A 线账本因此与改动前逐字节一致）。
            if selection.get("delivered") is not None:
                cfg["feature_action"]["d"] = selection["delivered"]
                cfg["feature_action"]["di"] = selection.get("delivered_identity")
        # v7 J2：本轮训练的 Adam 续训装载口径（fresh 清单 + 与父产物声明的对账结果）。
        # 只有真训练过的候选才有这一项；PARENT 轮次不训练，不写。
        training = prep["built"].get("training") or {}
        if training.get("resume"):
            cfg["resume"] = training["resume"]
        # v7 J4：L2-SP 的 λ 标定 trace 全量进账本（未启用时也是 dry-run 留痕）。
        if training.get("l2sp"):
            cfg["l2sp"] = training["l2sp"]
        # 诊断：soup 没跑起来时也要留下原因（candidate_is_parent / weights_changed_false /
        # dedup_remaining_endpoints / lineage_gate / env_off / not_regression）。
        skip = prep["built"].get("soup_skip") or training.get("soup_skip")
        if skip:
            cfg["soup_skip"] = skip
        # v7 J1：soup 候选族（端点对 + 每档 α 的分数 + 选中的 pair），只对回归线存在。
        soup = training.get("soup")
        if soup:
            grid = [{"p": item.get("pair"), "a": item.get("alpha"),
                     "rl": (round(item["rel_l2"], 6)
                            if item.get("rel_l2") is not None else None),
                     "rm": (round(item["rmse"], 6)
                            if item.get("rmse") is not None else None),
                     "ok": item.get("ok")}
                    for item in (soup.get("grid") or [])]
            if len(json.dumps(grid, ensure_ascii=False)) > SOUP_GRID_BUDGET:
                # 超预算只留"选中对 + 父–当前"，其余档位以计数交代，不静默丢。
                keep = {soup.get("pair"), "parent-current"}
                kept = [row for row in grid if row["p"] in keep]
                cfg["soup_grid_omitted"] = len(grid) - len(kept)
                grid = kept
            cfg["soup"] = {
                "cap": soup.get("cap"),
                "metric": soup.get("metric"),
                "cap_value": soup.get("cap_value"),
                "pair": soup.get("pair"),
                "endpoints": soup.get("endpoints"),
                "alpha": soup.get("alpha"),
                "rel_l2": soup.get("rel_l2"),
                "pred_disp": soup.get("pred_disp"),
                "rmse": soup.get("rmse"),
                "pairs": soup.get("pairs"),
                "skipped": soup.get("skipped"),
                "skip_reason": soup.get("skip_reason"),
                "history": soup.get("history"),
                "n_variants": soup.get("n_variants"),
                "n_eligible": soup.get("n_eligible"),
                "grid": grid,
            }
        # v7 J5：R10 同场测量的紧凑留痕（候选 sha、切片标识、决策/诊断两列分数）。
        # 键一律用短名：这一块与 alpha_grid 同抢 8 KiB 预算。
        if r10_record:
            slice_info = r10_record.get("slice") or {}
            selected = r10_record.get("selected") or None
            cfg["r10"] = {
                "s": slice_info.get("sha256_16"),
                "m": slice_info.get("metric"),
                "rows": slice_info.get("rows"),
                "st": r10_record.get("status"),
                "tie": r10_record.get("tie"),
                "mg": r10_record.get("margin"),
                "chg": r10_record.get("payload_changed"),
                "skip": r10_record.get("grid_skipped"),
                "ex": r10_record.get("soup_excluded"),
                "err": r10_record.get("error"),
                "sel": (None if not selected else
                        {"p": selected.get("pair"), "a": selected.get("alpha"),
                         "ep": selected.get("endpoints"),
                         "sha": selected.get("sha256"), "v": selected.get("value"),
                         "raw": selected.get("raw"), "i": selected.get("interpolated"),
                         "src": selected.get("source"), "r": selected.get("source_round")}),
                "c": [[item.get("pair"), item.get("alpha"), item.get("sha256"),
                       item.get("value"), item.get("raw"),
                       (None if item.get("rel_l2") is None
                        else round(float(item["rel_l2"]), 4)),
                       1 if item.get("eligible") else 0]
                      for item in (r10_record.get("candidates") or [])
                      if item.get("value") is not None],
            }
        # v7 J3：alpha_grid 整块挂进 configuration（不摊平、不进 _shrink_config 级联，
        # 块内已有 03 号文档的四级砍配）。剩余额度 = CONFIG_BUDGET − 插入前 cfg 体积
        # − 键名开销，避免"刚满 8192 再加一层键名就溢出"。
        if r10_record and r10_record.get("grid"):
            reserve = _config_size({"alpha_grid": ""}) + 8
            remaining = CONFIG_BUDGET - _config_size(cfg) - reserve
            try:
                text, level = _grid_serialize.serialize_alpha_grid(
                    r10_record["grid"], budget=max(1, remaining))
                cfg["alpha_grid"] = json.loads(text)
                if level:
                    cfg["alpha_grid_level"] = int(level)
            except _grid_serialize.GridTooLargeError as exc:
                # 砍到第 3 级仍放不下：显式记账，不静默丢这个块。
                cfg["alpha_grid_error"] = str(exc)[:160]
        # r10 明细是增益证据，但预算优先：先砍候选列表，再整块让位，都显式标记。
        if _config_size(cfg) > CONFIG_BUDGET and "r10" in cfg:
            omitted = len(cfg["r10"].pop("c", []))
            cfg["r10"]["c_omitted"] = omitted
        if _config_size(cfg) > CONFIG_BUDGET and "r10" in cfg:
            cfg.pop("r10", None)
            cfg["r10_error"] = "over_budget"
        cfg = _shrink_config(cfg)
        encoded = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise TrainRoundError("configuration 超过 64KiB：%d 字节" % len(encoded))
        return hyp, summary, cfg

    hypothesis, change, configuration = make_configuration(chosen, prepared, policy)
    # P0-1 升级：submit 之前对整份 configuration 做 allow_nan=False 自检；
    # 过不了就剔除可疑字段（不整轮放弃），保证交给 SDK 的一定是可序列化的 dict。
    configuration = _safe_configuration(configuration, round_index)
    artifact_id = None
    submitted = None
    failed_arm = None
    failed_combo = None
    def _submit(payload_dir):
        return rnd.submit_candidate(
            payload_dir,
            hypothesis=hypothesis,
            change_summary=change,
            configuration=configuration,
            seed=round_index,
            checkpoint_paths=[CHECKPOINT_NAME],
            code_paths=[],
            predict_argv=list(PREDICT_ARGV),
        )

    def _payload(prep):
        return prep.get("payload_dir") or prep["dir"]

    def _raise_interruption(exc):
        _log("submit_candidate 中断型失败（应答失配/结果未知：%s: %s）：按 SDK 1.0.4，"
             "本轮会话已不可用，第二次 RPC 必然 RPC_OUTCOME_UNKNOWN，重试无效且可能撞出"
             "重复提交；直接判本轮失败，不再重试。"
             % (type(exc).__name__, str(exc)[:200]))
        if isinstance(exc, Exception):
            raise TrainRoundError(
                "submit_candidate 中断型失败，本轮不重试：%s: %s"
                % (type(exc).__name__, str(exc)[:200])) from exc
        raise exc

    try:
        submitted = _submit(_payload(prepared))
    except BaseException as exc:
        if _is_session_dead(exc):
            _raise_interruption(exc)
        if chosen == PARENT:
            raise
        _log("候选 %s 提交失败（%s）；记一次失败并立即退回父候选" % (chosen, exc))
        failed_arm = chosen
        failed_combo = {"arm": chosen, "action": prepared["built"]["action"],
                        "params": prepared["built"].get("params") or {},
                        "signature": signature, "round": round_index}
        policy = "pessimism_fallback_parent"
        pessimism["fallback"] = {"from": chosen, "to": PARENT,
                                 "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
        prepared = prepare(PARENT, {}, None, "PARENT-submit-retry")
        chosen = PARENT
        _stamp_feature_action(getattr(rnd, "dataset_key", None), prepared, data)
        hypothesis, change, configuration = make_configuration(PARENT, prepared, policy)
        configuration = _safe_configuration(configuration, round_index)
        try:
            submitted = _submit(_payload(prepared))
        except BaseException as exc2:
            if _is_session_dead(exc2):
                _raise_interruption(exc2)
            raise
    artifact_id = submitted.get("artifact_id") if isinstance(submitted, dict) else None
    if not artifact_id:
        raise TrainRoundError("submit_candidate 未返回 artifact_id")

    # ---- P0-2（2026-11-24）：submit 成功之后全是簿记。任何一步异常都只记 stderr，
    # 绝不允许冒泡成"train(rnd) 返回时没有 finish"——harness.py:457 会因此判
    # round_failed，把已提交成功的合法产物白白作废。finish 在最后无条件发一次，
    # selected_artifact_id 永远是上面已拿到的 artifact_id。
    finish_id = artifact_id
    ledger_path = os.path.join(work, "round-%s" % round_index, LEDGER_FILE)
    ledger = {
        "hypothesis": hypothesis,
        "change_summary": change,
        "configuration": configuration,
        "reason": "",
        "artifact_id": artifact_id,
        "finish_artifact_id": finish_id,
    }
    bookkeeping_errors = []

    def _bookkeep(label, call, *args, **kwargs):
        """跑一步簿记；失败只落 stderr 与 bookkeeping_errors，永不冒泡。"""
        try:
            return call(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 簿记失败不许冒泡到 finish
            detail = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            bookkeeping_errors.append("%s: %s" % (label, detail))
            _log("提交成功后的簿记步骤 %s 失败（artifact=%s，已忽略，不影响 finish）：%s"
                 % (label, artifact_id, detail))
            return _BOOKKEEP_FAILED

    def _model_call_ids():
        """finish 必须带非空 call_id 列表；对畸形的 calls 元素也做防御。"""
        ids = []
        for item in calls or []:
            call_id = item.get("call_id") if isinstance(item, dict) else None
            if call_id:
                ids.append(call_id)
        if not ids:
            ids = [str(item) for item in (getattr(rnd, "call_ids", None) or []) if item]
        return ids

    def _history_ids():
        return [str(item) for item in (history_ids or [])]

    if fact_probe is not None:
        _bookkeep("fact_probe.train_receipt", fact_probe.train_receipt,
                  "submit", submitted)

    # ---- 回执打分（只读平台返回的数字） ----
    score = _bookkeep("arm_score", _arm_score, submitted)
    if score is _BOOKKEEP_FAILED or not isinstance(score, dict):
        score = None

    # ---- 提交后的叙事收尾：老代码里紧跟在 submit 后的配置收尾窗口 ----
    def _guard_narrative():
        parent_mean = (arms_in.get(PARENT) or {}).get("mean")
        delta = None
        if score is not None and parent_mean is not None and chosen != PARENT:
            delta = float(score["value"]) - float(parent_mean)
        guarded, warning = llm_cage.guard_hypothesis(model_note, delta)
        # 只改本地账本里的叙事字段；平台已收到的 configuration 保持提交时的原样。
        configuration["model"]["hypothesis_guarded"] = guarded
        configuration["model"]["delta_vs_parent"] = delta
        configuration["model"]["guard_warning"] = warning
        # 这三个是提交后补的本地叙事字段：平台副本已在提交时定格，但本地账本
        # 会因此多出十几字节，可能重新越过 8 KiB。再压一次，保证账本也守预算。
        if _config_size(configuration) > CONFIG_BUDGET:
            _shrink_config(configuration)
        if warning:
            _log("叙事校验：%s" % warning)

    _bookkeep("guard_hypothesis", _guard_narrative)

    arms = _bookkeep("next_arms", _next_arms, arms_in, chosen, score, failed_arm)
    if arms is _BOOKKEEP_FAILED or not isinstance(arms, dict):
        arms = dict(arms_in)
    reason = _bookkeep("reason", _reason_text, round_index, chosen, submitted)
    if reason is _BOOKKEEP_FAILED:
        reason = ""

    # ---- 历史最优（v4，2026-11-21）：本地 state 副本，不再向平台读回产物 ----
    # 副本在上面的 _apply_local_best 里已经比过并把更优者做成了本轮上传载荷，
    # 因此 finish 只选本轮提交的产物：它本身就是"至今最优"，最后一轮交的即历史最优。
    def _log_best():
        _log("第%s轮最优：副本第%s轮 → 本轮上传第%s轮（替换=%s）"
             % (round_index, (best_record or {}).get("best_round"),
                (best_record or {}).get("source_round"),
                (best_record or {}).get("replaced")))

    _bookkeep("best_log", _log_best)

    # ---- state：_state_record 先落 state 文件，随后调用平台 save_state ----
    def _build_state():
        return _state_record(state_path, ledger, round_index, finish_id,
                             previous_state, arms, score, signature=signature,
                             failed_combo=failed_combo, best=best_state,
                             bias_ema=bias_ema,
                             bias_last=bias_family.compact(bias_record))

    state = _bookkeep("state_record", _build_state)
    if state is not _BOOKKEEP_FAILED and isinstance(state, dict):
        _bookkeep("save_state", _rpc_guard, "save_state", rnd.save_state, state_path)
        ledger["state"] = state

    ledger["best"] = best_record
    # 载荷事实：本轮真正上传的那份（payload_dir 优先）的 sha 与 metadata.calibration，
    # 让"校准是否进产物"能在账本里直接对账，不必回读工作目录。
    def _payload_fact():
        payload_path = os.path.join(_payload(prepared), CHECKPOINT_NAME)
        try:
            payload_meta = train_mlp._load_parent(payload_path)["meta"]
        except Exception:  # noqa: BLE001 - 读不出就留空，不影响本轮提交
            payload_meta = {}
        return {
            "path": payload_path,
            "sha256": (_sha256_file(payload_path) if os.path.isfile(payload_path) else None),
            "calibration": payload_meta.get("calibration"),
        }

    payload_fact = _bookkeep("payload_fact", _payload_fact)
    if payload_fact is not _BOOKKEEP_FAILED:
        ledger["payload"] = payload_fact
    if bookkeeping_errors:
        ledger["bookkeeping_errors"] = list(bookkeeping_errors)
    ledger["reason"] = reason
    _bookkeep("ledger_dump", _dump_json, ledger_path, ledger)
    # 落盘这一步自身失败时，文件里记不到它；返回的账本补上，便于外部对账。
    if bookkeeping_errors:
        ledger["bookkeeping_errors"] = list(bookkeeping_errors)

    def _probe_finish():
        fact_probe.train_finish(
            round_index=round_index,
            chosen=chosen,
            artifact_id=artifact_id,
            policy=policy,
            history_ids=_history_ids(),
            model_call_ids=_model_call_ids(),
            score=score,
        )

    if fact_probe is not None:
        _bookkeep("fact_probe.train_finish", _probe_finish)

    # ---- 无条件收尾：无论上面失败与否，都用已提交的 artifact_id 发最小合法 finish ----
    # finish 的 configuration 再过一次 P0-1 自检；过不了就换最小配置，artifact_id 不变。
    finish_configuration = _bookkeep(
        "safe_configuration", _safe_configuration, configuration, round_index)
    if finish_configuration is _BOOKKEEP_FAILED or not isinstance(finish_configuration, dict):
        finish_configuration = _minimal_configuration(round_index,
                                                      "finish_configuration_degraded")
    if not isinstance(reason, str) or not reason.strip():
        reason = "第%s轮已提交 artifact_id=%s；簿记降级收尾" % (round_index, artifact_id)
    _rpc_guard("finish", rnd.finish, selected_artifact_id=finish_id,
               hypothesis=hypothesis,
               change_summary=change,
               reason=reason,
               configuration=finish_configuration,
               seed=round_index,
               history_ids=_history_ids(),
               model_call_ids=_model_call_ids())
    _log("第%s轮完成：候选 %s，artifact=%s%s"
         % (round_index, chosen, artifact_id,
            ("（簿记降级 %d 处）" % len(bookkeeping_errors)) if bookkeeping_errors else ""))
    return ledger

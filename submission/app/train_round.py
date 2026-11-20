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


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _work_dir(rnd):
    env = os.environ.get("CONTEST_WORKSPACE")
    if env:
        return env
    return "/workspace" if os.path.isdir("/workspace") else os.getcwd()


def _history(rnd):
    try:
        items = list(rnd.history() or [])
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
    """提案校验的上下文。签名表为空时按“未见”处理会误杀，故只在有表时判定。"""
    signatures = previous_state.get("signatures") if isinstance(previous_state, dict) else None
    if isinstance(signatures, list) and signatures:
        signature_seen = signature in [str(item) for item in signatures]
    else:
        signature_seen = True
    return {
        "arms": arms,
        "history_ids": history_ids,
        "phase": llm_cage.phase_of(round_index),
        "remaining_rounds": llm_cage.remaining_rounds(round_index),
        "signature_seen": signature_seen,
        "failed_combos": list((previous_state or {}).get("failed_combos") or []),
    }


def _call_model(rnd, messages):
    cap = getattr(rnd, "max_output_tokens", None)
    budget = 512
    if isinstance(cap, int) and cap > 0:
        budget = min(budget, cap)
    receipt = rnd.model(messages, max_tokens=budget, temperature=0)
    if fact_probe is not None:
        fact_probe.train_receipt("model", receipt)
    text = getattr(receipt, "text", None)
    text = "" if text is None else str(text)
    call_id = getattr(receipt, "call_id", None)
    call_ids = list(getattr(rnd, "call_ids", []) or [])
    if not call_ids and call_id:
        call_ids = [call_id]
    if not call_ids:
        raise TrainRoundError("本轮没有任何真实模型调用号，不能 finish")
    return {
        "call_id": call_id,
        "call_ids": call_ids,
        "text": text,
        "max_tokens": budget,
        "response_sha256": llm_cage.sha256_text(text),
    }


def _candidate_dir(work, round_index, history_digest, chosen, parent_sha):
    tag = _digest({"round": round_index, "history": history_digest,
                   "chosen": chosen, "parent": parent_sha})[:8]
    path = os.path.join(work, "round-%s-%s" % (round_index, tag), "candidate")
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def _build_candidate(parent, chosen, cand_dir, params=None, data_dir=None,
                     round_index=1, dataset_key=None):
    """PARENT = 逐字节复制；官方 MLP 父产物 = 同结构梯度续训；v2 父产物 = 只改写 action。

    官方格式（w1/w2/b1/b2/mean/scale，无 task/coef）的 predict 走官方前向，不读 action。
    往这种产物里写动作只会得到"官方键 + action"的假候选，因此这里改为**真训**：
    交给 train_mlp 在同一结构上续训，产出官方格式权重（见 train_mlp 模块说明）。
    v2 格式（含 coef）才沿用"保留数组、只改写 action"的老路径。
    """
    target = os.path.join(cand_dir, CHECKPOINT_NAME)
    if chosen == PARENT:
        shutil.copyfile(parent["path"], target)
        return {"weights_changed": False, "action": None,
                "sha256": _sha256_file(target)}
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
                                        params=params, dataset_key=dataset_key)
        except train_mlp.TrainMlpError as exc:
            raise TrainRoundError("官方 MLP 续训失败：%s" % exc)
        return {"weights_changed": report["weights_changed"], "action": None,
                "params": {"lr": report["hyper"]["lr"],
                           "batch": report["hyper"]["batch"],
                           "epochs": report["hyper"]["epochs_run"],
                           "scope": report["scope"]},
                "sha256": _sha256_file(target), "training": report}
    if "coef" not in files:
        raise TrainRoundError("父检查点既不是官方格式，也没有 v2 的 coef；无法造动作候选")

    action = {"A": "streaming_ema_norm", "B": "per_cond_seq", "C": "identity"}[chosen]
    params = dict(params or {})
    with np.load(parent["path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["action"] = np.array(action)
    arrays["action_params"] = np.array(json.dumps(params, sort_keys=True))
    np.savez(target, **arrays)
    return {"weights_changed": False, "action": action, "params": params,
            "sha256": _sha256_file(target)}


SIZE_CAP = 64 * 1024 * 1024  # 平台公告：产物总量 ≤64 MiB，不用 SDK 的 512 MiB


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


def _budget(cand_dir, cap=SIZE_CAP):
    files, total = 0, 0
    for dirpath, _, filenames in os.walk(cand_dir):
        for name in filenames:
            files += 1
            total += os.path.getsize(os.path.join(dirpath, name))
    if files > 64 or total > cap:
        raise TrainRoundError("候选超限：%d 文件 / %d 字节（上限 %d）" % (files, total, cap))
    return {"files": files, "bytes": total, "cap": cap}


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
    """读上一轮状态（只读副本）；读不出就当作空状态，不阻断本轮。"""
    path = os.path.join(state_path, STATE_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


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
            "cooldown": int(entry.get("cooldown", 0) or 0)}


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
        # 收缩：失败一次就把该臂的后验均值向父候选拉一半（无尺度依赖），
        # 悲观界随之下降，下一轮更不容易再被选中。
        parent_mean = (arms.get(PARENT) or {}).get("mean")
        if parent_mean is not None and int(entry.get("n", 0) or 0) > 0:
            entry["mean"] = 0.5 * float(entry.get("mean", 0.0)) + 0.5 * float(parent_mean)
    if score is not None:
        arms[chosen] = _arm_update(arms.get(chosen), score["value"])
    return arms


def _state_record(state_path, ledger, round_index, artifact_id, previous, arms, score,
                  signature=None, failed_combo=None, artifacts=None):
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
        "artifacts": list(artifacts if artifacts is not None
                          else (previous.get("artifacts") or []))[-10:],
    }
    path = os.path.join(state_path, STATE_FILE)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    return record


REPICK_ENV = "CONTEST_REPICK"
# C 线回投决胜：rmse 在最优 1% 以内的产物按 NAS 取最小（TASK.md：RMSE 主、NAS 辅）。
REPICK_RMSE_TOL = 0.01


def _artifact_registry(previous_state):
    """历史产物登记册（存于 round_state.json）：每轮提交一个就追加一个。"""
    items = (previous_state or {}).get("artifacts") or []
    return [dict(item) for item in items
            if isinstance(item, dict) and item.get("artifact_id")]


def _resolve_artifact(rnd, entry, work, round_index, current_id, current_path):
    """把登记在册的产物解析成本地 checkpoint 路径；取不到返回 None（跳过该条）。"""
    artifact_id = entry["artifact_id"]
    if artifact_id == current_id and current_path and os.path.isfile(current_path):
        return current_path
    restore = getattr(rnd, "restore_artifact", None)
    if not callable(restore):
        return None
    dest = os.path.join(work, "round-%s" % round_index, "history",
                        str(artifact_id).replace("/", "_"))
    try:
        # SDK 的 restore_artifact 自己用 mkdir(exist_ok=False) 建目标目录：
        # 先 makedirs(dest) 会让每次回读都抛 STATE 冲突（实测于 10 轮链条 e2e）。
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        restore(artifact_id, dest)
    except Exception as exc:  # noqa: BLE001 - 单个产物取不回不影响其它产物
        _log("restore_artifact(%s) 失败：%s" % (artifact_id, exc))
        return None
    path = os.path.join(dest, CHECKPOINT_NAME)
    return path if os.path.isfile(path) else None


def _repick_best(rnd, registry, data_dir, work, round_index, dataset_key,
                 current_id, current_path):
    """在固定标尺（本轮 validation.jsonl）上给全部历史产物重新打分，返回最优登记项。

    选择规则：分类按 macro_f1 最大；C 线回归在 rmse ≤ 最优×(1+1%) 的产物里按
    NAS 最小（RMSE 主选、NAS 决胜）；其它回归按 rmse 最小。
    任何一步失败都返回 None——回投是增益，绝不能让整轮因此失败。
    """
    if os.environ.get(REPICK_ENV, "1") == "0" or not registry:
        return None
    z_val, y_val, _task = train_mlp.load_yardstick(data_dir, current_path)
    scored = []
    for entry in registry:
        path = _resolve_artifact(rnd, entry, work, round_index,
                                 current_id, current_path)
        if not path:
            continue
        try:
            metrics = train_mlp.score_checkpoint(path, z_val, y_val)
        except Exception as exc:  # noqa: BLE001
            _log("产物 %s 标尺打分失败：%s" % (entry["artifact_id"], exc))
            continue
        scored.append({"entry": entry, "metrics": metrics})
    if not scored:
        return None
    task = scored[0]["metrics"]["task"]
    if task == "classification":
        best = max(scored, key=lambda item: item["metrics"]["macro_f1"])
    elif str(dataset_key or "") == "C":
        floor = min(item["metrics"]["rmse"] for item in scored)
        pool = [item for item in scored
                if item["metrics"]["rmse"] <= floor * (1.0 + REPICK_RMSE_TOL)]
        best = min(pool, key=lambda item: item["metrics"]["nas"])
    else:
        best = min(scored, key=lambda item: item["metrics"]["rmse"])
    return {"artifact_id": best["entry"]["artifact_id"],
            "round": best["entry"].get("round"),
            "metrics": best["metrics"],
            "n_scored": len(scored), "n_registry": len(registry)}


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
    state_path = _fspath(rnd.state_workspace(os.path.join(work, "round-%s" % round_index, "state")))
    os.makedirs(state_path, exist_ok=True)
    previous_state = _read_state(state_path)
    arms_in = previous_state.get("arms") or {}

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
    signature = _signature(getattr(rnd, "dataset_key", None),
                           detection["inputs"]["features"])
    context = _review_context(round_index, arms_in, history_ids, previous_state, signature)
    phase = context["phase"]
    recent_rounds = list((previous_state or {}).get("rounds") or [])
    if not recent_rounds:
        recent_rounds = [{"signature": "—", "arm": item.get("candidate", "—"),
                          "delta": None, "verdict": "—", "ledger_id": item.get("id")}
                         for item in history[-3:]]
    card = llm_cage.status_card(
        round_index, getattr(rnd, "dataset_key", None),
        detection.get("metric") or "metric", arms_in, recent_rounds, phase,
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
        review = llm_cage.review(last_text, context)
        if review["status"] == "ok":
            break
        last_errors = review["errors"]
        _log("模型提案第%d次被判无效：%s" % (attempt_no, "；".join(last_errors)))

    proposal = review["proposal"] if review and review["status"] == "ok" else None
    proposal_arm = proposal["arm"] if proposal else None
    gate = {
        "phase": phase,
        "signature": signature,
        "signature_seen": context["signature_seen"],
        "remaining_rounds": context["remaining_rounds"],
        "proposal": proposal_arm,
        "status": review["status"] if review else "unusable",
        "errors": list((review or {}).get("errors") or []),
        "calls": [{"call_id": item["call_id"],
                   "response_sha256": item["response_sha256"]} for item in calls],
        "state": None,
    }

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
            built = {"weights_changed": None, "action": None, "params": {},
                     "sha256": _sha256_file(target)}
        else:
            built = _build_candidate(parent, arm, cand_dir, arm_params,
                                     data_dir=data, round_index=round_index,
                                     dataset_key=getattr(rnd, "dataset_key", None))
        budget = _budget(cand_dir, _candidate_cap(rnd))
        check = llm_cage.check_candidate(
            cand_dir, expected_features=detection["inputs"]["features"],
            parent_path=parent["path"])
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
        call = _call_model(rnd, llm_cage.repair_messages(
            card, last_text, build_errors, data_summary,
            detection.get("detectors"), recommendation, round_index, failed_combos))
        calls.append(call)
        gate["calls"].append({"call_id": call["call_id"],
                              "response_sha256": call["response_sha256"]})
        retry = llm_cage.review(call["text"], context)
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
                    gate["repair"] = {"arm": arm2, "status": retry["status"]}
                except Exception as exc:
                    build_errors.append("%s: %s" % (type(exc).__name__, str(exc)[:400]))
                    _log("修复后的候选 %s 仍未过闸门：%s" % (arm2, exc))
            else:
                gate["repair"] = {"arm": arm2, "status": "state_gate_%s" % gate_reason}
        else:
            gate["repair"] = {"status": retry["status"], "errors": retry["errors"]}

    # 双调用：提案通过后，再让模型按五问清单做一次红队；无引用的结论不计入。
    if prepared is not None and chosen != PARENT and len(calls) < MAX_MODEL_CALLS:
        call = _call_model(rnd, llm_cage.red_team_messages(
            {"arm": chosen, "params": params, "hypothesis": model_note or ""},
            prepared["check"]["facts"], failed_combos))
        calls.append(call)
        gate["calls"].append({"call_id": call["call_id"],
                              "response_sha256": call["response_sha256"]})
        model_review = llm_cage.parse_red_team(call["text"])
        decision = llm_cage.red_team_gate(model_review, prepared["red_team"])
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
                    summary = ("只更新权重：%s 相对父产物 L2 变化 %.3g%%（改动张量 %s）；"
                               "结构、mean/scale、动作均未改；选轮依据验证集（self-set）"
                               % (name, rel * 100.0, "/".join(touched) or "无"))
                    soup = training.get("soup") or {}
                    if soup.get("alpha") is not None and soup["alpha"] < 1.0:
                        summary += ("；soup α=%.2f（拼接 rel_l2=%.3f，上限 %.2f）："
                                    "公开验证增益按比例收缩，换取隐藏集迁移"
                                    % (soup["alpha"], soup.get("rel_l2") or 0.0,
                                       soup.get("cap") or 0.0))
                    nas = training.get("nas") or {}
                    if nas.get("delta_b2"):
                        summary += "；b2 偏置 δ=%.3g（NAS 对齐）" % nas["delta_b2"]
                else:
                    summary = ("续训未过验证集门槛，候选检查点与父产物逐字节一致："
                               "权重、mean/scale、结构均未改")
            else:
                hyp = selector.CANDIDATES[arm]["hypothesis"]
                if pol.startswith("model_proposal") and model_note:
                    hyp = "%s；模型理由：%s" % (hyp, model_note)
                summary = ("仅把特征动作改为 %s；权重未训练，仍为父检查点原值"
                           % prep["built"]["action"])
        # 平台报告页"本轮改动"一定显示 change_summary（声明四件套），而 configuration
        # 不展示：把闸门判定必需的三件套写在这里，B/C 块状的两种成因据此可分。
        summary = "%s [wc=%d p=%s c=%s]" % (
            summary, 1 if prep["built"]["weights_changed"] else 0,
            parent["sha256"][:8], prep["built"]["sha256"][:8])
        cfg = {
            "round_index": round_index,
            "dataset_key": getattr(rnd, "dataset_key", None),
            "policy": pol,
            "candidate": arm,
            "action": prep["built"]["action"],
            "action_params": prep["built"].get("params") or {},
            "weights_changed": prep["built"]["weights_changed"],
            "parent_checkpoint_sha256": parent["sha256"],
            "candidate_checkpoint_sha256": prep["built"]["sha256"],
            "history_digest": history_digest,
            "history_ids": history_ids,
            "model": {
                "call_id": calls[0]["call_id"],
                "call_ids": [item["call_id"] for item in calls],
                "max_tokens": calls[0]["max_tokens"],
                "response_sha256": [item["response_sha256"] for item in calls],
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
            "selector": {"candidate": recommendation["candidate"],
                         "rule": recommendation["rule"],
                         "thresholds": stage["selector_trace"]["thresholds"]},
            "pessimism": pessimism,
            "budget": prep["budget"],
            "seed": round_index,
        }
        encoded = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise TrainRoundError("configuration 超过 64KiB：%d 字节" % len(encoded))
        return hyp, summary, cfg

    hypothesis, change, configuration = make_configuration(chosen, prepared, policy)
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

    def _raise_interruption(exc):
        _log("submit_candidate 中断型失败（%s: %s）：按 SDK 1.0.4，本轮会话已不可用，"
             "第二次 RPC 必然 RPC_OUTCOME_UNKNOWN，重试无效且可能撞出重复提交；"
             "直接判本轮失败，不再重试。" % (type(exc).__name__, str(exc)[:200]))
        if isinstance(exc, Exception):
            raise TrainRoundError(
                "submit_candidate 中断型失败，本轮不重试：%s: %s"
                % (type(exc).__name__, str(exc)[:200])) from exc
        raise exc

    try:
        submitted = _submit(prepared["dir"])
    except BaseException as exc:
        if _is_interruption(exc):
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
        hypothesis, change, configuration = make_configuration(PARENT, prepared, policy)
        try:
            submitted = _submit(prepared["dir"])
        except BaseException as exc2:
            if _is_interruption(exc2):
                _raise_interruption(exc2)
            raise
    artifact_id = submitted.get("artifact_id") if isinstance(submitted, dict) else None
    if fact_probe is not None:
        fact_probe.train_receipt("submit", submitted)
    if not artifact_id:
        raise TrainRoundError("submit_candidate 未返回 artifact_id")

    score = _arm_score(submitted)
    parent_mean = (arms_in.get(PARENT) or {}).get("mean")
    delta = None
    if score is not None and parent_mean is not None and chosen != PARENT:
        delta = float(score["value"]) - float(parent_mean)
    guarded, warning = llm_cage.guard_hypothesis(model_note, delta)
    # 只改本地账本里的叙事字段；平台已收到的 configuration 保持提交时的原样。
    configuration["model"]["hypothesis_guarded"] = guarded
    configuration["model"]["delta_vs_parent"] = delta
    configuration["model"]["guard_warning"] = warning
    if warning:
        _log("叙事校验：%s" % warning)

    arms = _next_arms(arms_in, chosen, score, failed_arm)
    reason = _reason_text(round_index, chosen, submitted)

    # ---- 历史最优回投（P0，2026-11-20）----
    # finish 不再无条件交本轮产物。标尺 = 本轮 validation.jsonl：全部历史产物在同一
    # 集合上重新打分（平台逐轮分的跨轮不可比问题由此绕开）。平台"末轮无条件采纳"
    # 因此等价于"采纳固定标尺最优"；与第 10 轮状态闸门不冲突——回投是 finish 语义
    # （HARNESS_SDK.md:158 "选回更早的产物"），不是提交新候选。
    registry = _artifact_registry(previous_state)
    registry.append({"artifact_id": artifact_id,
                     "sha256": prepared["built"]["sha256"],
                     "round": round_index, "arm": chosen})
    repick = None
    try:
        repick = _repick_best(rnd, registry, data, work, round_index,
                              getattr(rnd, "dataset_key", None), artifact_id,
                              os.path.join(prepared["dir"], CHECKPOINT_NAME))
    except Exception as exc:  # noqa: BLE001 - 回投失败退回本轮产物
        _log("回投评分失败（忽略，采用本轮产物）：%s: %s" % (type(exc).__name__, exc))
    finish_id = artifact_id
    if repick and repick["artifact_id"] != artifact_id:
        finish_id = repick["artifact_id"]
        change = ("%s；回投：固定标尺复评中第%s轮产物更优（%s），finish 改选更早产物"
                  % (change, repick["round"],
                     json.dumps(repick["metrics"], ensure_ascii=False, sort_keys=True)))
        reason = "%s；回投第%s轮产物（固定标尺复评）" % (reason, repick["round"])
        _log("回投：本轮 %s → 第%s轮产物 %s"
             % (artifact_id, repick["round"], finish_id))

    ledger = {
        "hypothesis": hypothesis,
        "change_summary": change,
        "configuration": configuration,
        "reason": reason,
    }
    state = _state_record(state_path, ledger, round_index, finish_id,
                          previous_state, arms, score, signature=signature,
                          failed_combo=failed_combo, artifacts=registry)
    rnd.save_state(state_path)

    ledger["state"] = state
    ledger["artifact_id"] = artifact_id
    ledger["finish_artifact_id"] = finish_id
    ledger["repick"] = repick
    ledger_path = os.path.join(work, "round-%s" % round_index, LEDGER_FILE)
    with open(ledger_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")

    if fact_probe is not None:
        fact_probe.train_finish(
            round_index=round_index,
            chosen=chosen,
            artifact_id=artifact_id,
            policy=policy,
            history_ids=list(history_ids),
            model_call_ids=[item["call_id"] for item in calls if item["call_id"]],
            score=score,
        )
    rnd.finish(
        selected_artifact_id=finish_id,
        hypothesis=hypothesis,
        change_summary=change,
        reason=reason,
        configuration=configuration,
        seed=round_index,
        history_ids=history_ids,
        model_call_ids=[item["call_id"] for item in calls if item["call_id"]],
    )
    _log("第%s轮完成：候选 %s，artifact=%s" % (round_index, chosen, artifact_id))
    return ledger

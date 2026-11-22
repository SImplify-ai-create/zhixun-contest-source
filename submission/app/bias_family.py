#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6：输出侧校准候选族（运行时选择，全程不碰梯度）。

定位：规则原文允许"验证集用于测量、模型选择与决策"，但不允许把它并入梯度训练。
所以这里的动作不是训练，而是**在一个参数化候选族里做选择**：

    候选生成 → 性质测试（llm_cage） → 用当轮切片逐个测量 → 选最优 → 输出侧校准

三条线各自的族（本版只实现 A/C；B 保留关闭）：

    C（回归）  仿射校准 pred' = a·pred + c，a、c 各自小网格（K=25）。
               依据：δ 扫描显示最优方向随寿命段换号（早段要正、晚段要负），
               病根是预测被压缩向均值——该校的是斜率，不是截距；单常数 δ 有害。
               标量 δ 是 c 的特例，`ScalarDeltaArm` 留作子集。
    A（分类）  逐类偏移 b2[c] ← b2[c] + v，一次只动一类（默认关闭，退化为全零）。
    B          无校准臂（残差方向也随寿命段换号，量级 ±300 分钟），原样放行。

硬约束：
- **不改 `target_mean` / `target_scale`**（冻结项）。校准参数写进产物 metadata 的
  独立字段 `calibration`，由 `contest_submission._infer_official` 在 predict 时应用。
- 每轮重新选校准：训练侧写新产物时丢弃父产物的旧 `calibration`（旧斜率对新权重无效）。

跨轮平滑：a 与 c 分别走 EMA（α=0.3），提交用 EMA 值而不是当轮最优。EMA 状态由调用方
写进 `round_state.json` 的 `bias_ema` 并随账本落盘——跨轮回放必须能逐位复现。

两级兜底（顺序固定）：
1. 当轮切片不可用（validation 读不出/没有标签）→ 用 a=1, c=0（父产物行为，恒等校准），
   **不用任何历史段的最优常数**；
2. 候选全被 cage 拒（含最终 EMA 值不过检查）→ 返回 status=parent_fallback，退回父产物。

边界：臂接口（`Arm`）可插拔，但**不实现梯度臂**——"在当轮切片上小步续训"冻结。
"""

import json
import math
import math
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import llm_cage  # noqa: E402
import train_mlp  # noqa: E402

CHECKPOINT_NAME = "checkpoint.npz"
EMA_ALPHA = 0.3
# C 的仿射网格：斜率 × 截距。截距单位是预测单位（周期），覆盖"靠近 0 的一段"；
# 全量最优 ≈ −2，早段要正、晚段要负，所以两侧都留了对称的小步长。
A_GRID = (0.90, 0.95, 1.00, 1.05, 1.10)
C_GRID = (5.0, 2.0, 0.0, -2.0, -5.0)
# v7：C 线校准 a 的端点。依据是 T1/F1 两次回执里"网格最优 a=1.05，EMA 却走到 1.083877"：
# 同一份数据里网格自己给的边界就是 1.05，滑窗不该越过它。
# v7.1（2026-11-22 c-push）：截断同时作用于 **EMA 状态**（回写），否则上一轮钳制过的
# a 仍按未截断序列继续增长（C 线账本 R1→R9 的 a_before=1.015→1.0921 就是这个）。
C_A_BOUND = 0.05
# C 线校准截距 c 的端点，单位是预测单位（周期）。生产账本实测到过 −4.342066。
# 折内拟合把 c 打到 ±5 网格两端且逐折换号（+1.5/−1.5/−0.6/−1.5/−0.6），故上界取 ±2。
C_C_BOUND = 2.0
# 旧的标量 δ 网格（c 的特例），只留作子集与留痕，不进本版候选族。
DELTA_GRID = tuple(train_mlp.DELTA_GRID)
# A 的逐类偏移幅度（一次只动一类）。闸门未开时不生成这些候选。
A_OFFSET_GRID = (-0.05, -0.1, 0.05, 0.1)
# 三道闸门（离线事实，运行时不自动判定；env 仅供离线实验强制打开）。
A_GATE = {
    "enabled": False,
    "env": "CONTEST_BIAS_A",
    "requires": ["batch1 的 Acetaldehyde 计数",
                 "漂移尺子的坏模型灵敏度",
                 "A 的计分指标答复"],
}
SUPPORTED = ("A", "C")
# configuration 里最多保留多少条网格得分（K=25 时只记 top-N + 最优，避免撑大回执）。
GRID_RECORD_MAX = 20
GRID_KEEP = 12


def _log(msg):
    sys.stderr.write("[bias_family] %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------- 臂接口


class Arm(object):
    """候选族臂接口：只描述"输出侧怎么改"，不碰输入侧与隐层。

    梯度臂（在当轮切片上小步续训）留了这个位置但**不实现**：规则边界未获书面答复前冻结。
    """
    key = None
    kind = None

    def candidates(self, b2, gate_open=False):
        raise NotImplementedError


class AffineArm(Arm):
    """C：输出侧仿射校准 pred' = a·pred + c。"""
    key = "C"
    kind = "affine"

    def candidates(self, b2, gate_open=False):
        return [{"id": "a%.2f c%+.0f" % (a, c), "a": float(a), "c": float(c)}
                for a in A_GRID for c in C_GRID]


class ScalarDeltaArm(Arm):
    """标量 δ（b2 常数平移）——仿射族的 c 特例，保留作子集/对照，不进本版注册表。"""
    key = "C"
    kind = "scalar_delta"

    def candidates(self, b2, gate_open=False):
        return [{"id": "%+.3f" % delta,
                 "delta": [float(delta)] * int(b2.shape[0])}
                for delta in DELTA_GRID]


class PerClassOffsetArm(Arm):
    key = "A"
    kind = "per_class_offset"

    def candidates(self, b2, gate_open=False):
        zero = [0.0] * int(b2.shape[0])
        if not gate_open:
            return [{"id": "zero", "delta": zero}]
        out = [{"id": "zero", "delta": zero}]
        for index in range(int(b2.shape[0])):
            for value in A_OFFSET_GRID:
                delta = list(zero)
                delta[index] = float(value)
                out.append({"id": "c%d%+.2f" % (index, value), "delta": delta})
        return out


ARMS = {arm.key: arm for arm in (AffineArm(), PerClassOffsetArm())}


# ---------------------------------------------------------------- 工具


def _write_checkpoint(path, arrays, meta=None):
    """原子写出官方格式检查点；meta 给定时同步刷新 metadata 数组。"""
    import numpy as np

    payload = dict(arrays)
    if meta is not None:
        payload["metadata"] = np.array(
            json.dumps(meta, ensure_ascii=False, sort_keys=True))
    temp = path + ".tmp.npz"
    np.savez(temp, **payload)
    os.replace(temp, path)


def _measure(z_val, y_val, task, arrays, b2, target_mean, target_scale,
             calibration=None):
    """在当轮切片上给一个候选打分。返回 {score, macro_f1|rmse, nas}。"""
    pred = train_mlp._predict_calibrated(
        z_val, arrays["w1"], arrays["b1"], arrays["w2"], b2, task,
        target_mean, target_scale, calibration)
    if task == "classification":
        f1 = train_mlp._macro_f1(y_val, pred, int(arrays["w2"].shape[1]))
        return {"score": float(f1), "macro_f1": float(f1)}
    rmse = float(math.sqrt(float(((pred - y_val) ** 2).mean())))
    return {"score": -rmse, "rmse": rmse, "nas": train_mlp.nasa_score(y_val, pred)}


def _gate_open():
    return bool(A_GATE["enabled"]) or os.environ.get(A_GATE["env"]) == "1"


def _prior(previous_state, key):
    """读上一轮 EMA。C 是 (a, c)，A 是逐类偏移向量；缺失/形状不符按恒等起算。"""
    store = (previous_state or {}).get("bias_ema")
    entry = (store or {}).get(key) if isinstance(store, dict) else None
    if not isinstance(entry, dict):
        entry = {}
    if key == "C":
        try:
            return {"a": float(entry.get("a", 1.0)), "c": float(entry.get("c", 0.0))}
        except (TypeError, ValueError):
            return {"a": 1.0, "c": 0.0}
    vector = entry.get("offset")
    if isinstance(vector, list):
        try:
            return {"offset": [float(item) for item in vector]}
        except (TypeError, ValueError):
            pass
    return {"offset": []}


def _ema_blend(prior, best):
    """EMA 更新：ema = α·best + (1−α)·prior。a 与 c（或逐类偏移）各自平滑。"""
    alpha = EMA_ALPHA
    if "a" in prior:
        return {"a": alpha * float(best["a"]) + (1.0 - alpha) * float(prior["a"]),
                "c": alpha * float(best["c"]) + (1.0 - alpha) * float(prior["c"])}
    blend = []
    for index, value in enumerate(best.get("delta") or []):
        previous = prior["offset"][index] if index < len(prior["offset"]) else 0.0
        blend.append(alpha * float(value) + (1.0 - alpha) * previous)
    return {"offset": blend}


def _dir_bytes(path):
    files, total = 0, 0
    for _dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            files += 1
            total += os.path.getsize(os.path.join(path, name))
    return files, total


def _round(value, places=6):
    return round(float(value), places)


def _json_safe_number(value):
    """留痕用：有限值按 _round 记数，非有限值记 repr 字符串。

    非有限值不能原样写进 metadata/configuration：`json.dumps` 默认会产出 `NaN`/`Infinity`
    这类非标准 JSON，平台侧一旦严格解析就会整份拒绝。字符串保留了原值（`nan`/`inf`/`-inf`），
    又不会污染 JSON。
    """
    raw = float(value)
    if math.isfinite(raw):
        return _round(raw)
    return repr(raw)


def clamp_calibration_a(value):
    """C 线校准 a 的端点截断：|a − 1| ≤ C_A_BOUND。

    返回 (写入值, clamped, sanitized, 原始值)。两类异常分开记账：
    - `clamped` 只表示"有限值越出 [0.95, 1.05]，被压回端点"；
    - `sanitized` 表示"非有限值（NaN/±inf）不参与比较，替换成恒等值 1.0"。

    NaN 与 +inf 在旧实现里两个比较都为 False，会原样写进 metadata 并标 `clamped=false`，
    账本看着正常、predict 端出 NaN。这里先把非有限值截住，原始值由调用方留痕。
    """
    raw = float(value)
    if not math.isfinite(raw):
        return 1.0, False, True, raw
    high = 1.0 + C_A_BOUND
    low = 1.0 - C_A_BOUND
    if raw > high:
        return high, True, False, raw
    if raw < low:
        return low, True, False, raw
    return raw, False, False, raw


def sanitize_calibration_c(value):
    """C 线截距：非有限值归 0（恒等），返回 (写入值, sanitized)；原始值由调用方留痕。"""
    raw = float(value)
    if not math.isfinite(raw):
        return 0.0, True
    return raw, False


def clamp_calibration_c(value):
    """C 线校准截距 c 的端点截断：|c| ≤ C_C_BOUND（预测单位=周期）。

    v7.1（2026-11-22 c-push）：旧实现只挡非有限值、不设上界，生产账本实测到过
    c=−4.342066。C4 五折的折内拟合把 c 打到网格两端 ±5，逐折符号还随折翻转
    （+1.5/−1.5/−0.6/−1.5/−0.6），说明它是在拟合当轮切片的偏差而不是可迁移的偏移。
    这里把写入值和 EMA 状态都压到 ±C_C_BOUND，间隔仍是"预测量纲"而不是 a 的比例。

    返回 (写入值, clamped, sanitized, 原始值)。
    """
    raw = float(value)
    if not math.isfinite(raw):
        value, sanitized = sanitize_calibration_c(raw)
        return value, False, sanitized, raw
    if raw > C_C_BOUND:
        return C_C_BOUND, True, False, raw
    if raw < -C_C_BOUND:
        return -C_C_BOUND, True, False, raw
    return raw, False, False, raw


def _pack_grid(entries):
    """按分数保留 top-N（含最优），并标注被截断的总数。"""
    if len(entries) <= GRID_RECORD_MAX:
        return entries, False
    ranked = sorted(entries, key=lambda item: item.get("score", -1e18), reverse=True)
    return ranked[:GRID_KEEP], True


# ---------------------------------------------------------------- 主入口


def apply(dataset_key, data_dir, parent_path, prepared, work_dir, round_index,
          previous_state=None, expected_features=None):
    """对本轮候选跑一遍校准候选族。C 改 metadata，A 改 b2；原地改写候选目录。

    返回值直接进 `configuration["bias_family"]`；status == "parent_fallback" 时
    调用方必须退回父产物。EMA 状态在返回值 `ema.state` 里，由调用方写进 state。
    """
    started = time.monotonic()
    key = str(dataset_key or "")
    cand_path = os.path.join(prepared["dir"], CHECKPOINT_NAME)
    record = {
        "schema": "contest.v6.bias-family",
        "arm": key, "supported": key in SUPPORTED, "enabled": False,
        "alpha": EMA_ALPHA, "applied": False, "fallback": None,
        "gate": {"open": _gate_open(), "requires": list(A_GATE["requires"])},
        "grid": [], "best": None, "chosen": None,
        "ema": {"before": None, "after": None, "state": None},
        "source_sha256": None, "candidate_sha256": None,
        "status": "noop",
    }
    if key not in SUPPORTED:
        record["reason"] = "该线没有校准臂（本版只实现 A/C）"
        return _finish(record, started)
    if not os.path.isfile(cand_path):
        record["reason"] = "本轮候选检查点不存在"
        return _finish(record, started)

    arm = ARMS[key]
    try:
        ref = train_mlp._load_parent(cand_path)
    except Exception as exc:  # noqa: BLE001 - 非官方格式直接跳过
        record["reason"] = "候选不是可读的官方检查点：%s" % str(exc)[:120]
        return _finish(record, started)
    arrays = ref["arrays"]
    expected_task = "regression" if key == "C" else "classification"
    if ref["task"] != expected_task:
        record["reason"] = "候选任务 %r 与 %s 线不符" % (ref["task"], key)
        return _finish(record, started)
    if "b2" not in arrays:
        record["reason"] = "候选缺 b2，无法做输出侧校准"
        return _finish(record, started)

    record["enabled"] = True
    record["kind"] = arm.kind
    record["source_sha256"] = train_mlp._sha256_file(cand_path)
    b2 = arrays["b2"]
    meta = dict(ref["meta"])
    target_mean = float(meta.get("target_mean", 0.0) or 0.0)
    target_scale = float(meta.get("target_scale", 1.0) or 1.0)
    candidates = arm.candidates(b2, _gate_open())
    record["K"] = len(candidates)

    # ---- 当轮切片；不可用则走第一级兜底（恒等校准 a=1, c=0） ----
    z_val = y_val = None
    slice_error = None
    try:
        z_val, y_val, _task = train_mlp.load_yardstick(data_dir, parent_path)
    except Exception as exc:  # noqa: BLE001 - 切片不可用是已知情形
        slice_error = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    record["slice"] = {"rows": int(z_val.shape[0]) if z_val is not None else 0,
                       "error": slice_error}

    prior = _prior(previous_state, key)
    record["ema"]["before"] = ({name: _round(value) for name, value in prior.items()}
                               if key == "C"
                               else {"offset": [_round(value) for value in prior["offset"]]})

    if z_val is None:
        # 第一级兜底：恒等校准，不测量、不更新 EMA。
        record["fallback"] = "identity"
        chosen = {"a": 1.0, "c": 0.0} if key == "C" else {"delta": [0.0] * int(b2.shape[0])}
        record["chosen"] = chosen
        final = _apply_final(record, prepared, arrays, meta, chosen, parent_path,
                             expected_features, round_index, key, source="identity")
        if final is None:
            record["status"] = "parent_fallback"
            record["fallback"] = "identity_then_parent"
        else:
            record["status"] = "identity"
        return _finish(record, started)

    # ---- 候选生成 → 性质测试 → 逐个测量 ----
    import numpy as np

    scratch = os.path.join(work_dir, "round-%s" % round_index, "bias")
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    ok_items = []
    for index, cand in enumerate(candidates):
        if key == "C":
            calibration = (cand["a"], cand["c"])
            trial_b2 = b2
            trial_meta = dict(meta)
            trial_meta["calibration"] = {"a": cand["a"], "c": cand["c"],
                                         "source": "grid", "round": int(round_index)}
        else:
            calibration = None
            delta = [float(value) for value in cand["delta"]]
            trial_b2 = np.asarray(b2, dtype=float) + np.asarray(delta, dtype=float)
            trial_meta = meta
        path = os.path.join(scratch, "c%02d" % index, CHECKPOINT_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(arrays)
        payload["b2"] = trial_b2
        _write_checkpoint(path, payload, trial_meta)
        check = llm_cage.check_candidate(os.path.dirname(path),
                                         expected_features=expected_features,
                                         parent_path=parent_path, arm=key)
        entry = {"id": cand["id"], "ok": bool(check["ok"]),
                 "errors": list(check["errors"])[:2]}
        if key == "C":
            entry["a"] = cand["a"]
            entry["c"] = cand["c"]
        else:
            nonzero = [i for i, value in enumerate(cand["delta"]) if value]
            entry["class"] = nonzero[0] if nonzero else None
            entry["value"] = float(cand["delta"][nonzero[0]]) if nonzero else 0.0
        if check["ok"]:
            metrics = _measure(z_val, y_val, ref["task"], arrays, trial_b2,
                               target_mean, target_scale, calibration)
            entry.update({name: (_round(value) if isinstance(value, float) else value)
                          for name, value in metrics.items()})
            ok_items.append({"entry": entry, "cand": cand, "metrics": metrics})
        record["grid"].append(entry)

    if not ok_items:
        # 第二级兜底：候选全被 cage 拒 → 退回父产物。
        record["status"] = "parent_fallback"
        record["fallback"] = "cage_all_rejected"
        shutil.rmtree(scratch, ignore_errors=True)
        return _finish(record, started)

    best = max(ok_items, key=lambda item: item["metrics"]["score"])
    if key == "C":
        record["best"] = {"id": best["entry"]["id"],
                          "a": _round(best["cand"]["a"]),
                          "c": _round(best["cand"]["c"]),
                          "score": _round(best["metrics"]["score"])}
        best_delta = {"a": best["cand"]["a"], "c": best["cand"]["c"]}
    else:
        record["best"] = {"id": best["entry"]["id"],
                          "delta": [_round(value) for value in best["cand"]["delta"]],
                          "score": _round(best["metrics"]["score"])}
        best_delta = {"delta": list(best["cand"]["delta"])}

    ema_after = _ema_blend(prior, best_delta)
    record["ema"]["after"] = ({name: _round(value) for name, value in ema_after.items()}
                              if key == "C"
                              else {"offset": [_round(value) for value in ema_after["offset"]]})
    prev_entry = ((previous_state or {}).get("bias_ema") or {}).get(key)
    rounds = int(prev_entry.get("rounds", 0)) if isinstance(prev_entry, dict) else 0
    if key == "C":
        # 回写钳制后的值：否则"当轮应用 1.05、账本记 1.052995"，下一轮 prior 继续沿
        # 未截断序列爬升。state 必须等于实际生效的值，回放才能逐位复现。
        a_state, a_clamped_state, _, _ = clamp_calibration_a(ema_after["a"])
        c_state, c_clamped_state, _, _ = clamp_calibration_c(ema_after["c"])
        state = {"a": _round(a_state), "c": _round(c_state)}
        record["ema"]["state_clamped"] = bool(a_clamped_state or c_clamped_state)
        record["ema"]["state_raw"] = {"a": _round(ema_after["a"]),
                                      "c": _round(ema_after["c"])}
    else:
        state = {"offset": [_round(value) for value in ema_after["offset"]]}
    state["rounds"] = rounds + 1
    record["ema"]["state"] = state

    # 提交 EMA 值；它一般不在网格点上，所以单独过一次 cage。
    final = _apply_final(record, prepared, arrays, meta, ema_after, parent_path,
                         expected_features, round_index, key, source="ema")
    source = "ema"
    if final is None:
        # EMA 值被拒：退回当轮最优网格点（它已过 cage）。
        final = _apply_final(record, prepared, arrays, meta, best_delta,
                             parent_path, expected_features, round_index, key,
                             source="best_grid")
        source = "best_grid"
        if final is not None:
            record["fallback"] = "ema_rejected_best_grid"
    if final is None:
        record["status"] = "parent_fallback"
        record["fallback"] = "cage_rejected_final"
        shutil.rmtree(scratch, ignore_errors=True)
        return _finish(record, started)

    if key == "C":
        # chosen 记**实际写入**的值（已过端点），clamp 留痕单列，避免账本出现
        # "写的是 1.05、说的是 1.083877" 的错位。
        applied_c = record.get("chosen_plain") or (
            ema_after if source == "ema" else best_delta)
        record["chosen"] = {"a": _round(applied_c["a"]), "c": _round(applied_c["c"])}
    else:
        picked = ema_after["offset"] if source == "ema" else best_delta["delta"]
        record["chosen"] = {"delta": [_round(value) for value in picked]}
    record["chosen"]["source"] = source
    record["status"] = "selected"
    shutil.rmtree(scratch, ignore_errors=True)
    return _finish(record, started)


def _apply_final(record, prepared, arrays, meta, chosen, parent_path,
                 expected_features, round_index, key, source):
    """把最终校准写进本轮候选目录；不过 cage 时返回 None。

    C：只改 metadata 的 `calibration` 字段，权重与 target_mean/target_scale 不动。
    A：只改 b2；偏移全零时不动文件（逐字节保留训练产物）。
    """
    import numpy as np

    cand_dir = prepared["dir"]
    path = os.path.join(cand_dir, CHECKPOINT_NAME)
    payload = dict(arrays)
    stamp = dict(meta)
    if key == "C":
        # v7：a 走端点（|a−1| ≤ C_A_BOUND）。截断后的值才是写进 metadata 与账本的值；
        # 截断前原值只在这条留痕里（a_before_clamp），EMA 的 α 与网格都不动。
        # 非有限值（NaN/±inf）单独记 sanitized：它不参与区间比较，替换成恒等值 1.0，
        # 不能和"越界被压回"共用一个 clamped=true。
        a_value, a_clamped, a_sanitized, a_raw = clamp_calibration_a(chosen["a"])
        c_raw = float(chosen["c"])
        c_value, c_clamped, c_sanitized, _c_raw = clamp_calibration_c(c_raw)
        sanitized = bool(a_sanitized or c_sanitized)
        scale = float(meta.get("target_scale", 1.0) or 1.0)
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        payload["b2"] = np.asarray(arrays["b2"], dtype=float)
        stamp["calibration"] = {"a": a_value, "c": c_value, "source": source,
                                "alpha": EMA_ALPHA, "round": int(round_index),
                                "clamped": bool(a_clamped),
                                "c_clamped": bool(c_clamped),
                                "sanitized": sanitized,
                                "c_abs_over_scale": _round(abs(c_value) / abs(scale))}
        if a_clamped or a_sanitized:
            stamp["calibration"]["a_before_clamp"] = _json_safe_number(a_raw)
        if c_clamped or c_sanitized:
            stamp["calibration"]["c_before"] = _json_safe_number(c_raw)
        applied = abs(a_value - 1.0) > 1e-12 or abs(c_value) > 1e-12
        record["chosen_plain"] = {"a": a_value, "c": c_value}
        record["clamp"] = {"clamped": bool(a_clamped),
                           "c_clamped": bool(c_clamped),
                           "sanitized": sanitized,
                           "a_after": _round(a_value),
                           "a_before": _json_safe_number(a_raw),
                           "c_after": _round(c_value),
                           "c_before": _json_safe_number(c_raw),
                           "c_abs_over_scale": _round(abs(c_value) / abs(scale))}
    else:
        vector = np.asarray(chosen.get("delta", chosen.get("offset")) or [], dtype=float)
        if vector.size and float(np.max(np.abs(vector))) <= 0.0:
            record["applied"] = False
            record["chosen_note"] = "offset_zero"
            return {"ok": True, "errors": [], "facts": {}}
        payload["b2"] = np.asarray(arrays["b2"], dtype=float) + vector
        applied = True
        stamp["bias"] = {"kind": "per_class_offset",
                         "delta": [_round(value) for value in vector],
                         "source": source, "alpha": EMA_ALPHA,
                         "round": int(round_index)}

    _write_checkpoint(path, payload, stamp)
    check = llm_cage.check_candidate(cand_dir, expected_features=expected_features,
                                     parent_path=parent_path, arm=key)
    if not check["ok"]:
        record.setdefault("rejected", []).append(
            {"source": source, "errors": list(check["errors"])[:3]})
        return None
    # applied 只在候选真正通过 cage 之后才置真：被拒的尝试没有写进载荷，
    # 提前置真会让"回退父产物"的轮次在账本上仍声称校准已生效。
    record["applied"] = applied
    # C 的恒等校准也要落盘：训练侧已丢弃父产物的旧 calibration，这里补本轮的一份，
    # 保证 predict 行为与本轮测量口径一致（权重字节不变，变的是 metadata）。
    prepared["built"]["sha256"] = train_mlp._sha256_file(path)
    prepared["built"]["trained_sha256"] = record["source_sha256"]
    prepared["built"]["bias"] = record
    if key == "C":
        prepared["built"]["calibrated"] = True
        prepared["built"]["calibration"] = record["chosen_plain"]
    else:
        prepared["built"]["weights_changed"] = True
    prepared["check"] = check
    files, total = _dir_bytes(cand_dir)
    budget = dict(prepared.get("budget") or {})
    budget.update({"files": files, "bytes": total})
    prepared["budget"] = budget
    record["candidate_sha256"] = prepared["built"]["sha256"]
    record["final_check"] = {"ok": True,
                             "weight_rel_l2": check["facts"].get("weight_rel_l2")}
    return check


def _finish(record, started):
    # 网格留痕：K 变大时只留 top-N + 最优，避免把回执撑大；总数单独记。
    entries = record.get("grid")
    if isinstance(entries, list) and len(entries) > GRID_RECORD_MAX:
        record["grid_total"] = len(entries)
        record["grid"], _ = _pack_grid(entries)
        record["grid_truncated"] = True
    # 耗时只进 stderr，不进 configuration/state：账本要逐位可复现。
    _log("arm=%s status=%s K=%s applied=%s fallback=%s %.0fms"
         % (record.get("arm"), record.get("status"), record.get("K"),
            record.get("applied"), record.get("fallback"),
            (time.monotonic() - started) * 1000.0))
    return record


def compact(record):
    """写进 state 的摘要：去掉逐点网格，保留选择与 EMA 轨迹。"""
    if not isinstance(record, dict):
        return None
    return {name: record.get(name) for name in
            ("arm", "kind", "status", "applied", "uploaded", "upload_note",
             "fallback", "best", "chosen", "clamp",
             "ema", "K", "slice") if name in record}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在官方 CPU MLP 结构上做真正的梯度续训（纯 numpy，不引入新依赖）。

背景：平台父产物是官方格式（`w1/b1/w2/b2/mean/scale` + `metadata`），predict 走
`contest_submission._infer_official`，既不读 v2 的 `action`，也不会自己训练。
只往官方父产物里写 `action` 是假候选（装载器静默忽略）；要产生真实增益，只能在
同一结构上更新权重。本模块只做这件事：

    归一化：z = clip((x - mean) / scale, -10, 10)       # mean/scale 沿用父产物
    隐层：  h = tanh(z @ w1 + b1)
    输出：  o = h @ w2 + b2
    分类：  softmax + 交叉熵（输出维 = w2 列数 = len(classes)）
    回归：  MSE（目标先按 target_mean/target_scale 归一化；predict 侧再还原）

边界（对应 SDK 1.0.4 原文）：
- **结构不换**：只更新 w1/b1/w2/b2，形状与父产物一致；mean/scale 默认原样保留；
- **梯度只用训练集**：`train.jsonl`；`validation.jsonl` 只用来度量与选轮（早停、
  岭系数选择），每次决策都写进 `train_report.json` 和账本；
- **失败不提交**：任何非有限权重、维度不符都在这里抛错，由上层退回 PARENT。

三臂在本模块里表示"适应强度"，不是输入变换（官方 predict 没有动作位）：
    A（检测到漂移）      全网络续训，学习率 ×1.0，最多 3 轮
    B（分组/顺序漂移）   全网络续训，学习率 ×0.5，最多 2 轮
    C（未见稳定漂移）    只重解输出层（分类走 Adam 头、回归走岭回归闭式解）
"""

import collections
import json
import math
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_stage  # noqa: E402  复用 JSONL 读取与路径展开口径
from v6alpha import alpha_lib as _alpha_lib  # noqa: E402  v6 纯函数库（原样复用）
from v6alpha import l2sp as _l2sp  # noqa: E402

# v7 J4：L2-SP 的 d_target 预登记入口与时间保留量。
# d_target 未登记时锚定关闭，但 trace 照打（dry-run），供挑选可行区间。
ENV_L2SP_D_TARGET = "CONTEST_L2SP_D_TARGET"
# v7 预登记（2026-11-22 定案）：A=0.15（平台回执最坏一步 0.1332 < 0.15 < cap 0.1696）；
# B、C 不登记，按臂配置，不设全局值。
# 触发语义按实现：d0 > 1.1 × d_target（l2sp.TRIGGER_TOL=0.1）→ A 的触发点是 0.165，
# 保险区 (0.165, 0.1696] 与越 cap 轮；口径与 C4 尺子预登记见 app/preregistration.json。
L2SP_D_TARGET_PREREGISTERED = {"A": 0.15}
L2SP_RESERVE_SECONDS = 5.0
L2SP_WOULD_BE_RATIOS = (0.5, 0.7, 0.9)

OFFICIAL_KEYS = ("w1", "b1", "w2", "b2", "mean", "scale")
TRAIN_NAME = "train.jsonl"
VAL_NAME = "validation.jsonl"
REPORT_NAME = "train_report.json"
CHECKPOINT_NAME = "checkpoint.npz"
TASKS = ("classification", "regression")

# 三臂的适应强度。依据是 selector 的漂移假设：漂移越强，越敢动全网络。
ARM_PROFILES = {
    "A": {"scope": "all", "lr_multiplier": 1.0, "epochs": 3},
    "B": {"scope": "all", "lr_multiplier": 0.5, "epochs": 2},
    "C": {"scope": "head", "lr_multiplier": 1.0, "epochs": 1},
}
# --- 2026-11-21 v5：后段轮次步幅衰减表 ---
# 目标：后段每轮候选相对父产物的 rel_l2 收进 0.05–0.1，每轮都过 0.1 闸门、每轮都
# 产出一个可比较的真候选。0.1 阈值不动（放宽要等被拒候选复评结论）。
# 表项 = (首轮, 末轮, epochs, lr_multiplier)；lr_multiplier 是相对父产物 learning_rate
# 的绝对倍数（即覆盖 ARM_PROFILES 的值）。区间未覆盖的轮次沿用臂 profile。
# 实测（A 线公开切片，train_mlp 直接续训父产物，rel_l2 为提交权重 vs 父）：
#   lr×1.0 → 0.0160；×0.75 → 0.0125；×0.5 → 0.0087；×0.1 → 0.0061。
#   epochs 在 A 上不改变端点（best_epoch 恒为 1），所以衰减主轴是 lr，epochs 只兜底。
# C 线是头重解闭式（岭回归），lr/epochs 都不参与求解（实测 6 组设置逐位相同，
# 提交 rel_l2 恒为 0.0654、soup α=0.25），所以 C 表是恒等项，实际步幅由 soup 的
# α 在 0.1 闸门内选，不再重复加一层衰减。
STEP_DECAY = {
    "A": ((1, 2, 3, 1.0), (3, 4, 2, 0.5), (5, 6, 2, 0.35),
          (7, 8, 1, 0.35), (9, 99, 1, 0.25)),
    "B": ((1, 2, 2, 0.5), (3, 4, 2, 0.35), (5, 6, 1, 0.35),
          (7, 8, 1, 0.25), (9, 99, 1, 0.15)),
    "C": ((1, 99, 1, 1.0),),
}

DEFAULTS = {
    "patience": 1,
    "grad_clip": 5.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1e-8,
    "ridge_ratios": (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    "keep_parent_if_worse": True,
    "max_seconds": 600.0,
}
# 允许本地实验覆盖；线上路径不读环境变量之外的东西。
ENV_LR = "CONTEST_TRAIN_LR_MULT"
ENV_EPOCHS = "CONTEST_TRAIN_EPOCHS"
ENV_NO_GUARD = "CONTEST_TRAIN_KEEP_PARENT"
# 岭回归选系数的容忍带：取验证集上 RMSE 最小的那个 λ（容忍带归零）。
# v7.1（2026-11-22 c-push）：原值 0.015 表示"比最优差 1.5% 以内取最大 λ（最小权重范数）"。
# C4 五折（seed=c4-2026-11-22，496 台按 (source_subset,unit) 分组）实测：λ 曲线在
# 0→1e-2 上单调，内层验证与折外的 argmin 都落在 0~1e-5；容忍带把 λ 推到 1e-2，
# 纯岭头的折外 RMSE 从 54.2633（λ=1e-6）抬到 54.8391（λ=1e-2），5/5 折一致变差。
# 依据与逐折数字见 d:\智能比赛\2026-11-22\c-push\outputs\C线攻坚结论.md。
RIDGE_TOLERANCE = 0.0
# 固定标尺的最低有效行占比。低于此值说明切片大面积丢行（结构/标签不匹配），
# 此时任何两参数拟合都只是"有解"，不是"可评"，宁可让本轮退回父产物。
MIN_VALID_RATIO = 0.5

# --- 2026-11-20 加固：权重漂移上限 / soup 插值 / C 线 NAS 闸门与 δ 偏置 ---
# 权重相对父产物的拼接 L2 上限。check_candidate 与 soup 共用同一口径，两处必须同源。
ENV_REL_CAP = "CONTEST_WEIGHT_REL_CAP"
REL_CAP_DEFAULT = 0.1
# v3 预登记（2026-11-22 c-push）：C 线的守卫度量从"权重拼接 rel_l2"换成"预测位移"。
#   disp = RMS(pred_candidate − pred_parent) / |target_scale|，在当轮 validation 切片上
#   用未校准预测算。阈值 0.11 由 C4 五折敏感性表定（0.09→54.3430、0.10→54.2369、
#   0.11→54.2218、0.12→54.2218，带校准口径）；0.11 与 0.10 的差 0.0151 在折间噪声内，
#   按"同等就取更宽一档"取 0.11，且 0.11–0.12 是平台。定位是"位移记录 + 极端拦截"，
#   不宣称保护力（5 折 30 候选 AUC(bad>good)=0.167）。仅 C 线生效；A/B 仍用 rel_l2。
C_PRED_DISP_CAP = 0.11
# A 线的上限不再用单轮常数，而是"计划累计步幅预算"：把 STEP_DECAY 每轮的
# epochs × lr_multiplier 加起来，乘单轮实测锚点。锚点取 A 线公开切片上
# lr×1.0、best_epoch=1 的提交 rel_l2=0.0160（见 STEP_DECAY 注释）。
STEP_ANCHOR_REL_L2 = 0.0160
PLAN_ROUNDS = 10
# soup：提交权重 = 父 + α·(训练结果 − 父)。α 在验证集上选，且必须过 REL_CAP。
# 依据：三线实测迁移率与更新幅度单调反向（A 1.6% 最好，C 岭头 80.6% 为负）。
ENV_SOUP = "CONTEST_SOUP"
SOUP_GRID = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05)
# NAS 闸门只在 dataset_key == "C" 的回归线生效：C 的 TASK.md 写明主指标是 RMSE、
# NAS 是辅助，因此形态是"RMSE 主选 + NAS 闸门 + δ 偏置带 RMSE 约束"，不是整体替换。
ENV_NAS_GUARD = "CONTEST_NAS_GUARD"
ENV_NAS_TOL = "CONTEST_NAS_TOL"
NAS_TOL_DEFAULT = 0.05  # 候选 NAS 不得超过父产物的 (1+tol) 倍（同集比较才合法）
# δ 偏置：对 b2 加归一化单位的常数平移，在验证集上最小化 NAS，
# 约束 RMSE 升幅 ≤ DELTA_RMSE_TOL。NAS 对高估是指数罚，负 δ 收益是指数级的。
DELTA_GRID = (0.0, -0.01, -0.02, -0.05, -0.1, -0.2)
DELTA_RMSE_TOL = 0.01


class TrainMlpError(Exception):
    """训练输入或产物不满足要求；上层据此退回 PARENT。"""


def _log(msg):
    sys.stderr.write("[train_mlp] %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------- 数据


def _shape_of(value):
    """inputs 的结构签名：对象按键排序、数组记长度。只用于判断行是否同构。"""
    if isinstance(value, dict):
        return tuple((key, _shape_of(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return ("#", len(value))
    return None


def _collect(value, out):
    """按"对象键字典序、数组顺序"展开数值，与 predict 的 flatten 口径一致。"""
    if isinstance(value, dict):
        for key in sorted(value):
            _collect(value[key], out)
    elif isinstance(value, list):
        for child in value:
            _collect(child, out)
    elif isinstance(value, bool):
        raise ValueError("bool 不算数值")
    else:
        try:
            number = float(value)
        except OverflowError:
            raise ValueError("数值超出 float 可表示范围")
        if not math.isfinite(number):
            raise ValueError("非有限数值")
        out.append(number)


def _read_rows(path):
    if not os.path.isfile(path):
        raise TrainMlpError("找不到数据文件：%s" % path)
    # C 线性能：与 train_stage / _slice_card 共用同一份按文件缓存的解析结果，
    # 不再对同一份 train/validation 重复 json.loads。返回口径与旧实现一致。
    return train_stage.load_jsonl(path)


def _extract(rows, shape, n_features, label_fn):
    """按固定结构抽 X/y；结构不符或缺标签的行整行跳过并计数。

    v7.2 工程优化：用 `train_stage._collect_sig` 一次遍历同时校验结构与收集数值，
    替代原先 `_shape_of` + `_collect` 两遍走；坏行仍按"先结构、后数值"的旧口径归类。
    """
    import numpy as np

    kept_x, kept_y, stats = [], [], {"rows": len(rows), "bad_shape": 0,
                                     "bad_label": 0, "bad_value": 0}
    for row in rows:
        inputs = row["inputs"]
        values = []
        try:
            signature = train_stage._collect_sig(inputs, values)
        except (TypeError, ValueError, OverflowError):
            # 旧实现先算 _shape_of：结构不符计 bad_shape，结构对了但数值坏计 bad_value。
            try:
                mismatch = _shape_of(inputs) != shape
            except (TypeError, ValueError):
                mismatch = False
            stats["bad_shape" if mismatch else "bad_value"] += 1
            continue
        if signature != shape or len(values) != n_features:
            stats["bad_shape"] += 1
            continue
        target = label_fn(row)
        if target is None:
            stats["bad_label"] += 1
            continue
        kept_x.append(values)
        kept_y.append(target)
    x = np.asarray(kept_x, dtype=float).reshape(len(kept_x), n_features)
    return x, kept_y, stats


def _load_dataset_uncached(data_dir, paths, classes, task):
    """读 train/validation，返回 (x_train, y_train, x_val, y_val, stats)。"""
    n_features = len(paths)
    train_rows, train_bad = _read_rows(os.path.join(data_dir, TRAIN_NAME))
    val_rows, val_bad = _read_rows(os.path.join(data_dir, VAL_NAME))

    shape = None
    for row in train_rows:
        try:
            signature = _shape_of(row["inputs"])
            probe = []
            _collect(row["inputs"], probe)
        except (TypeError, ValueError):
            continue
        if len(probe) == n_features:
            shape = signature
            break
    if shape is None:
        raise TrainMlpError("训练集没有任何与父产物特征维 %d 匹配的行" % n_features)

    if task == "classification":
        index = {str(name): i for i, name in enumerate(classes)}

        def label_fn(row):
            answer = row.get("answer")
            if not isinstance(answer, str):
                return None
            return index.get(answer)
    else:
        def label_fn(row):
            answer = row.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                return None
            value = float(answer)
            return value if math.isfinite(value) else None

    x_train, y_train, s_train = _extract(train_rows, shape, n_features, label_fn)
    x_val, y_val, s_val = _extract(val_rows, shape, n_features, label_fn)
    stats = {
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "train_bad_json": train_bad, "val_bad_json": val_bad,
        "train": s_train, "val": s_val,
    }
    return x_train, y_train, x_val, y_val, stats


# v7.2 工程优化：同一轮内 finetune 会被调用多次（选中臂 + 四臂同尺的对照臂），
# 数据完全相同时没必要重复"解析 JSONL + 逐行展开"。键里带文件 mtime/ctime/size，
# 数据一换即失效；只缓存只读的 X/y（返回给调用方的 y 是 list，会被各调用方转成新数组）。
_LOAD_CACHE = collections.OrderedDict()
_LOAD_CACHE_MAX = 3


def _load_dataset(data_dir, paths, classes, task):
    """`_load_dataset_uncached` 的进程内缓存层：输入相同则复用同一份 X/y。"""
    train_path = os.path.join(data_dir, TRAIN_NAME)
    val_path = os.path.join(data_dir, VAL_NAME)
    key = None
    if os.path.isfile(train_path) and os.path.isfile(val_path):
        key = (train_stage.file_key(train_path), train_stage.file_key(val_path),
               tuple(paths), tuple(str(name) for name in classes), task)
        hit = _LOAD_CACHE.get(key)
        if hit is not None:
            _LOAD_CACHE.move_to_end(key)
            x_train, y_train, x_val, y_val, stats = hit
            return x_train, y_train, x_val, y_val, dict(stats)
    value = _load_dataset_uncached(data_dir, paths, classes, task)
    if key is not None:
        _LOAD_CACHE[key] = value
        while len(_LOAD_CACHE) > _LOAD_CACHE_MAX:
            _LOAD_CACHE.popitem(last=False)
    return value


# ---------------------------------------------------------------- 模型


def _load_parent(path):
    import numpy as np

    if not os.path.isfile(path):
        raise TrainMlpError("父检查点不存在：%s" % path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as exc:  # noqa: BLE001 - npz 读不出就是不可用
        raise TrainMlpError("父检查点不是可读的 npz：%s" % exc)
    missing = [name for name in OFFICIAL_KEYS if name not in arrays]
    if missing:
        raise TrainMlpError("父检查点缺官方键：%s" % ",".join(missing))
    meta = {}
    if "metadata" in arrays:
        try:
            meta = json.loads(str(np.asarray(arrays["metadata"]).tolist()))
        except (ValueError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
    w1 = np.asarray(arrays["w1"], dtype=float)
    if w1.ndim != 2:
        raise TrainMlpError("w1 必须是二维矩阵，实际 %r" % (w1.shape,))
    hidden = int(w1.shape[1])
    task = meta.get("task")
    if task not in TASKS:
        task = "classification" if int(np.asarray(arrays["w2"]).shape[1]) > 1 else "regression"
    paths = meta.get("paths")
    if not isinstance(paths, list) or len(paths) != int(w1.shape[0]):
        raise TrainMlpError("metadata.paths 缺失或与 w1 输入维 %d 不一致" % int(w1.shape[0]))
    return {"arrays": arrays, "meta": meta, "task": task, "hidden": hidden,
            "paths": [str(item) for item in paths], "n_features": int(w1.shape[0])}


def _softmax(logits):
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _forward(z, w1, b1, w2, b2):
    import numpy as np

    hidden = np.tanh(z @ w1 + b1)
    return hidden, hidden @ w2 + b2


def _loss_grad(z, y, w1, b1, w2, b2, task):
    """一个 minibatch 的损失与梯度。y：分类是 int 下标，回归是归一化目标。"""
    import numpy as np

    n = z.shape[0]
    hidden, out = _forward(z, w1, b1, w2, b2)
    if task == "classification":
        probs = _softmax(out)
        picked = probs[np.arange(n), y]
        loss = float(-np.mean(np.log(np.maximum(picked, 1e-12))))
        grad_out = probs
        grad_out[np.arange(n), y] -= 1.0
        grad_out /= n
    else:
        residual = out[:, 0] - y
        loss = float(np.mean(residual * residual))
        grad_out = (2.0 * residual / n).reshape(-1, 1)
    grad_w2 = hidden.T @ grad_out
    grad_b2 = grad_out.sum(axis=0)
    grad_hidden = (grad_out @ w2.T) * (1.0 - hidden * hidden)
    grad_w1 = z.T @ grad_hidden
    grad_b1 = grad_hidden.sum(axis=0)
    return loss, grad_w1, grad_b1, grad_w2, grad_b2


def _predict(z, w1, b1, w2, b2, task, target_mean, target_scale):
    """与 predict 官方前向完全同口径，返回标签下标或原始单位数值。"""
    return _predict_calibrated(z, w1, b1, w2, b2, task, target_mean, target_scale,
                               None)


def _predict_calibrated(z, w1, b1, w2, b2, task, target_mean, target_scale,
                        calibration=None):
    """官方前向 + 可选的输出侧仿射校准 pred' = a·pred + c。

    顺序与 `contest_submission._infer_official` 严格一致：先还原到原始单位，
    再仿射，最后截断到非负。calibration=None 时逐位等于旧 `_predict`。
    """
    import numpy as np

    _, out = _forward(z, w1, b1, w2, b2)
    if task == "classification":
        return out.argmax(axis=1)
    value = out[:, 0] * target_scale + target_mean
    if calibration is not None:
        value = float(calibration[0]) * value + float(calibration[1])
    return np.maximum(0.0, value)


def _macro_f1(y_true, y_pred, n_classes):
    """官方口径：在"真值 ∪ 预测"出现过的类别上做算术平均 F1。"""
    import numpy as np

    total = 0.0
    present = sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    for label in present:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total += (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return float(total / max(1, len(present)))


def _metric(task, z_val, y_val, weights, target_mean, target_scale):
    pred = _predict(z_val, weights["w1"], weights["b1"], weights["w2"],
                    weights["b2"], task, target_mean, target_scale)
    if task == "classification":
        return _macro_f1(y_val, pred, int(weights["w2"].shape[1]))
    return float(math.sqrt(sum((float(p) - float(t)) ** 2 for p, t in zip(pred, y_val))
                           / max(1, len(y_val))))


def nasa_score(y_true, y_pred):
    """NASA 非对称评分（C-MAPSS 口径，v5.1 截断 + 均值）。

    d = min(预测,125) − min(真实,125)；d≥0（高估剩余寿命）取 exp(d/10)−1，
    d<0 取 exp(−d/13)−1，逐样本**取均值**（旧版是逐样本求和）。
    截断 125 针对标签量程：C 标签到 434、B 到 1514，未截断时单行残差会主导指数惩罚。
    公式与常数已对平台回执锚定：父权重在 C 包 validation.jsonl 上本地复算（求和的旧口径）
    = 5,823,439,325.34，与回执显示值恰差 10³（转录错位，见研判第四份）。
    改成均值后跨样本集可直接横比（旧版求和口径下不可比）。
    """
    import numpy as np

    true = np.minimum(np.asarray(y_true, dtype=float), 125.0)
    pred = np.minimum(np.asarray(y_pred, dtype=float), 125.0)
    d = pred - true
    return float(np.mean(np.where(d >= 0.0, np.exp(d / 10.0) - 1.0,
                                  np.exp(-d / 13.0) - 1.0)))


def concat_rel_l2(current, parent_arrays):
    """w1/b1/w2/b2 拼接向量的相对 L2：‖Δ‖/‖父‖。check_candidate 与 soup 共用。"""
    import numpy as np

    num, den = 0.0, 0.0
    for name in ("w1", "b1", "w2", "b2"):
        base = np.asarray(parent_arrays[name], dtype=float)
        delta = np.asarray(current[name], dtype=float) - base
        num += float(np.sum(delta * delta))
        den += float(np.sum(base * base))
    if den <= 0.0:
        return None
    return math.sqrt(num) / math.sqrt(den)


def _pred_displacement(pred, pred_parent, target_scale):
    """C 线守卫度量（v3）：预测空间 RMS 位移，按 target_scale 归一。

    对闭式头，权重 rel_l2 会被参数化尺度放大（λ→1e-5 时岭头 rel 1.7–3.6，但折外
    预测未发散）；这里直接量函数空间位移。分母取 |target_scale|，所以阈值是无量纲的。
    """
    import numpy as np

    scale = abs(float(target_scale))
    if scale <= 0.0:
        return float("inf")
    diff = np.asarray(pred, dtype=float) - np.asarray(pred_parent, dtype=float)
    return float(math.sqrt(float(np.mean(diff * diff)))) / scale


def planned_cumulative_rel_budget(arm="A", rounds=PLAN_ROUNDS,
                                  anchor=STEP_ANCHOR_REL_L2):
    """计划累计步幅预算：Σ(每轮 epochs × lr_multiplier) × 单轮实测锚点。

    衰减表 STEP_DECAY 规定每轮跑几个 epoch、lr 乘多少；锚点是该臂在公开切片上
    lr×1.0、best_epoch=1 时实测的提交权重 rel_l2。A 线十轮：
        2×3×1.0 + 2×2×0.5 + 2×2×0.35 + 2×1×0.35 + 2×1×0.25 = 10.6
        10.6 × 0.0160 = 0.1696
    闸门按这个上限判，覆盖计划内的全部轮次；不写死 0.15，也不对权重做 clamp。
    """
    profile = ARM_PROFILES.get(arm) or ARM_PROFILES["A"]
    total = 0.0
    for index in range(1, int(rounds) + 1):
        plan = step_plan(arm, index) or {
            "epochs": int(profile["epochs"]),
            "lr_multiplier": float(profile["lr_multiplier"]),
        }
        total += (float(anchor) * float(plan["lr_multiplier"])
                  * float(plan["epochs"]))
    return total


def weight_rel_cap(arm=None):
    """权重拼接 rel_l2 上限。

    优先级：CONTEST_WEIGHT_REL_CAP > A 线的计划累计步幅预算 > 0.1。
    A 线按 STEP_DECAY 求和（0.1696@10 轮），其余线维持单轮口径 0.1。
    """
    raw = os.environ.get(ENV_REL_CAP)
    if raw is None or not str(raw).strip():
        if str(arm or "").strip().upper() == "A":
            return planned_cumulative_rel_budget("A")
        return REL_CAP_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        raise TrainMlpError("%s 不是数字：%r" % (ENV_REL_CAP, raw))
    if not math.isfinite(value) or value <= 0.0:
        raise TrainMlpError("权重 rel_l2 上限必须是正有限数：%r" % (value,))
    return value


def _load_yardstick_uncached(data_dir, reference_path):
    """按参照检查点的 schema 只读 validation.jsonl（历史最优回投的固定标尺）。

    返回 (z_val, y_val, task)。只读 validation 不读 train：C 线 train.jsonl 38 MB，
    回投每轮都跑，不能把训练集也拖进来。
    """
    import numpy as np

    ref = _load_parent(reference_path)
    n_features = ref["n_features"]
    task = ref["task"]
    rows, _bad = _read_rows(os.path.join(data_dir, VAL_NAME))
    shape = None
    for row in rows:
        try:
            signature = _shape_of(row["inputs"])
            probe = []
            _collect(row["inputs"], probe)
        except (TypeError, ValueError):
            continue
        if len(probe) == n_features:
            shape = signature
            break
    if shape is None:
        raise TrainMlpError("验证集没有任何与产物特征维 %d 匹配的行" % n_features)
    if task == "classification":
        classes = [str(item) for item in (ref["meta"].get("classes") or [])]
        index = {name: i for i, name in enumerate(classes)}

        def label_fn(row):
            answer = row.get("answer")
            if not isinstance(answer, str):
                return None
            return index.get(answer)
    else:
        def label_fn(row):
            answer = row.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                return None
            try:
                value = float(answer)
            except OverflowError:
                return None
            return value if math.isfinite(value) else None
    x_val, y_val, v_stats = _extract(rows, shape, n_features, label_fn)
    if x_val.shape[0] < 1:
        raise TrainMlpError("固定标尺为空：验证集没有可用行")
    ratio = float(x_val.shape[0]) / float(max(1, len(rows)))
    _log("标尺有效行 %d/%d（%.1f%%）：bad_shape=%d bad_label=%d bad_value=%d"
         % (x_val.shape[0], len(rows), ratio * 100.0,
            v_stats["bad_shape"], v_stats["bad_label"], v_stats["bad_value"]))
    if ratio < MIN_VALID_RATIO:
        # 标尺丢行是静默的：两参数仿射只要 2 个点就能"拟合成功"，
        # 所以必须在比例过低时直接判本轮不可评，而不是让选型基于残破切片。
        raise TrainMlpError("固定标尺有效行比例 %.1f%% < %.0f%%（%d/%d）："
                            "丢行过多，拒绝在此标尺上评分"
                            % (ratio * 100.0, MIN_VALID_RATIO * 100.0,
                               x_val.shape[0], len(rows)))
    mean = np.asarray(ref["arrays"]["mean"], dtype=float).reshape(-1)
    scale = np.asarray(ref["arrays"]["scale"], dtype=float).reshape(-1)
    z_val = np.clip((x_val - mean) / scale, -10.0, 10.0)
    if task == "classification":
        return z_val, np.asarray(y_val, dtype=np.int64), task
    return z_val, np.asarray(y_val, dtype=float), task


# v7.2 工程优化：一轮里 bias_family / 四臂同尺 / 局部最优回投都用同一父产物读同一份
# validation.jsonl，标尺完全一致。按 (validation 身份, 参照产物身份) 复用，数据换即失效。
_YARD_CACHE = collections.OrderedDict()
_YARD_CACHE_MAX = 2


def load_yardstick(data_dir, reference_path):
    """`_load_yardstick_uncached` 的进程内缓存层：同一标尺不重复解析。"""
    val_path = os.path.join(data_dir, VAL_NAME)
    key = None
    if os.path.isfile(val_path) and os.path.isfile(reference_path):
        key = (train_stage.file_key(val_path), train_stage.file_key(reference_path))
        hit = _YARD_CACHE.get(key)
        if hit is not None:
            _YARD_CACHE.move_to_end(key)
            return hit
    value = _load_yardstick_uncached(data_dir, reference_path)
    if key is not None:
        _YARD_CACHE[key] = value
        while len(_YARD_CACHE) > _YARD_CACHE_MAX:
            _YARD_CACHE.popitem(last=False)
    return value


def _calibration_of(meta, task):
    """读产物 metadata 里的输出侧仿射校准，校验口径与 predict 侧一致。

    返回 (a, c) 或 None（恒等）。非法直接报错，不静默降级：否则选型会在一个
    平台装载期就会拒绝的产物上打分。
    """
    raw = meta.get("calibration")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TrainMlpError("metadata.calibration 必须是 JSON 对象")
    if task != "regression":
        raise TrainMlpError("metadata.calibration 只用于回归产物")
    try:
        a_value = float(raw.get("a", 1.0))
        c_value = float(raw.get("c", 0.0))
    except (TypeError, ValueError):
        raise TrainMlpError("metadata.calibration 的 a/c 不是数值")
    if not math.isfinite(a_value) or not math.isfinite(c_value) or a_value <= 0.0:
        raise TrainMlpError("metadata.calibration 非法：a=%r c=%r" % (a_value, c_value))
    return (a_value, c_value)


def score_checkpoint(path, z_val, y_val, calibrated=True):
    """在固定标尺上给单个官方格式检查点打分（不回传预测，只回标量）。

    分类：{"task", "macro_f1"}；回归：{"task", "rmse", "nas"}。nas 一并算出，
    是否使用由调用方按 dataset_key 决定（TASK.md：NAS 是 C 的辅助指标）。

    `calibrated=False` 走统一未校准口径（v7 J5 的 `scores_raw` 诊断列，05 号文档）：
    决策列按提交口径（带 calibration 的带、不带的不带），诊断列只看权重本身。
    """
    import numpy as np

    ref = _load_parent(path)
    arrays = ref["arrays"]
    meta = ref["meta"]
    # 与 predict 同口径：产物自带的输出侧校准必须一起应用，否则"校准带来的增益"
    # 在选型里完全不可见（选出来的 best 不是平台实际会用的那个）。
    pred = _predict_calibrated(z_val,
                               np.asarray(arrays["w1"], dtype=float),
                               np.asarray(arrays["b1"], dtype=float),
                               np.asarray(arrays["w2"], dtype=float),
                               np.asarray(arrays["b2"], dtype=float),
                               ref["task"],
                               float(meta.get("target_mean", 0.0) or 0.0),
                               float(meta.get("target_scale", 1.0) or 1.0),
                               calibration=(_calibration_of(meta, ref["task"])
                                            if calibrated else None))
    if ref["task"] == "classification":
        return {"task": "classification",
                "macro_f1": _macro_f1(y_val, pred, int(np.asarray(arrays["w2"]).shape[1]))}
    rmse = float(math.sqrt(float(np.mean((pred - y_val) ** 2))))
    return {"task": "regression", "rmse": rmse, "nas": nasa_score(y_val, pred)}


# ---------------------------------------------------------------- 训练


def _load_moments(arrays, w):
    """从父产物读 Adam 一阶/二阶矩；缺失或形状不符时按零初始化。

    返回 `(moments, fresh)`：`fresh` 是缺失/形状不符、按零初始化（fresh Adam）
    的张量名清单。训练行为不因本清单改变，它只是把"静默清零"变成可对账的账本项。
    """
    import numpy as np

    out = {}
    fresh = []
    for name in ("w1", "b1", "w2", "b2"):
        first = arrays.get("adam_first_%s" % name)
        second = arrays.get("adam_second_%s" % name)
        if (isinstance(first, np.ndarray) and first.shape == w[name].shape
                and isinstance(second, np.ndarray) and second.shape == w[name].shape):
            out[name] = [np.asarray(first, dtype=float).copy(),
                         np.asarray(second, dtype=float).copy()]
        else:
            out[name] = [np.zeros_like(w[name]), np.zeros_like(w[name])]
            fresh.append(name)
    return out, fresh


def _adam(epochs, lr, grad_clip, parent, task, z, y, w, batch, seed,
          parent_epochs, parent_steps, patience, z_val, y_val,
          target_mean, target_scale, deadline, trainable=("w1", "b1", "w2", "b2"),
          l2sp=None):
    """全参数 Adam。返回 (最优轮权重, 末轮权重, history, 轮数, 步数, 最优指标, 最优轮, 末态矩, fresh)。

    Adam 状态与轮次种子都从父产物续上（与官方 model.fit 的续训口径一致）：
    偏差校正从 `metadata.steps` 继续，洗牌种子用 `[seed, metadata.epochs + epoch]`。
    父产物缺矩（例如插值产物，metadata 声明 `fresh_adam_on_missing_state`）时按零
    初始化，`fresh` 列出这些张量名供账本对账。

    `l2sp={"lam","start"}` 时给损失加 `(λ/2)‖W−W_start‖²`（λ=0 与不传等价）。
    """
    import numpy as np

    beta1, beta2, eps = DEFAULTS["beta1"], DEFAULTS["beta2"], DEFAULTS["adam_eps"]
    moments, fresh = _load_moments(parent, w)
    penalty_lam = float((l2sp or {}).get("lam") or 0.0)
    penalty_start = (l2sp or {}).get("start")
    if penalty_lam > 0.0 and not isinstance(penalty_start, dict):
        raise TrainMlpError("L2-SP 打开时缺 start 权重（锚点）")
    step = parent_steps
    history = []
    best = {name: w[name].copy() for name in ("w1", "b1", "w2", "b2")}
    best_metric = _metric(task, z_val, y_val, w, target_mean, target_scale)
    best_epoch = 0
    stale = 0
    epochs_run = 0
    n = z.shape[0]
    for epoch in range(1, epochs + 1):
        if time.monotonic() > deadline:
            _log("达到时间上限，提前结束训练（第 %d 轮）" % epoch)
            break
        order = np.random.default_rng(
            np.random.SeedSequence([seed, parent_epochs + epoch - 1])).permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            index = order[start:start + batch]
            zb = z[index]
            yb = y[index]
            loss, gw1, gb1, gw2, gb2 = _loss_grad(zb, yb, w["w1"], w["b1"],
                                                  w["w2"], w["b2"], task)
            grads = {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2}
            if penalty_lam > 0.0:
                # 软近端：损失加 (λ/2)‖W−W_start‖²，梯度加 λ(W−W_start)。
                # λ=0 时两者恒为零，与不传 l2sp 逐位等价。
                penalty = _l2sp.l2sp_penalty(w, penalty_start, penalty_lam)
                total += (loss + penalty) * len(index)
                anchor_grad = _l2sp.l2sp_gradient(w, penalty_start, penalty_lam)
                for name in grads:
                    grads[name] = grads[name] + anchor_grad[name]
            else:
                total += loss * len(index)
            norm = math.sqrt(float(sum(float(np.sum(g * g)) for g in grads.values())))
            scale = 1.0 if norm <= grad_clip else grad_clip / (norm + 1e-12)
            step += 1
            for name, grad in grads.items():
                if name not in moments or name not in trainable:
                    continue
                first, second = moments[name]
                first *= beta1
                first += (1.0 - beta1) * grad * scale
                second *= beta2
                second += (1.0 - beta2) * (grad * scale) ** 2
                m_hat = first / (1.0 - beta1 ** step)
                v_hat = second / (1.0 - beta2 ** step)
                w[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        epochs_run = epoch
        train_loss = total / max(1, n)
        metric = _metric(task, z_val, y_val, w, target_mean, target_scale)
        better = ((metric > best_metric + 1e-12) if task == "classification"
                  else (metric < best_metric - 1e-12))
        if better:
            best_metric = metric
            best_epoch = epoch
            best = {name: w[name].copy() for name in best}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val": metric,
                        "steps": step})
        final = {name: w[name].copy() for name in w}
        if stale > patience:
            break
    if epochs_run == 0:
        final = {name: w[name].copy() for name in w}
    final_moments = {name: [first.copy(), second.copy()]
                     for name, (first, second) in moments.items()}
    return best, final, history, epochs_run, step, best_metric, best_epoch, final_moments, fresh


def _ridge_head(z, y, w, w2_shape, ratios):
    """回归头闭式解：只重解 w2/b2，隐层冻结。

    隐层输出先按训练集的均值/标准差标准化再解，避免高相关特征把解推到病态
    （直接解会让 w2 放大十几倍，隐藏集一漂就崩）；解出后再折回原参数化。
    """
    import numpy as np

    hidden, _ = _forward(z, w["w1"], w["b1"], w["w2"], w["b2"])
    center = hidden.mean(axis=0)
    spread = hidden.std(axis=0)
    spread[spread < 1e-8] = 1.0
    scaled = (hidden - center) / spread
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1))], axis=1)
    gram = design.T @ design
    rhs = design.T @ y
    penalty = np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0  # 偏置不惩罚
    base = float(np.trace(gram)) / max(1, gram.shape[0])
    records = []
    solutions = []
    for ratio in ratios:
        scale = ratio * base if ratio else 0.0
        try:
            coef = np.linalg.solve(gram + scale * penalty, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram + scale * penalty, rhs, rcond=None)[0]
        raw = coef[:-1] / spread
        w2 = raw.reshape(w2_shape)
        b2 = np.asarray([coef[-1] - float(center @ raw)], dtype=float)
        residual = hidden @ raw + b2[0] - y
        records.append({"ridge_ratio": ratio, "train_mse": float(np.mean(residual ** 2))})
        solutions.append((w2, b2))
    return solutions, records


def step_plan(arm, round_index):
    """按轮次返回 {"epochs", "lr_multiplier"}；表外轮次返回 None（沿用臂 profile）。"""
    try:
        index = int(round_index)
    except (TypeError, ValueError):
        return None
    for low, high, epochs, multiplier in STEP_DECAY.get(arm, ()):
        if low <= index <= high:
            return {"epochs": int(epochs), "lr_multiplier": float(multiplier)}
    return None


def finetune(parent_path, out_dir, data_dir, arm="A", round_index=1, params=None,
             dataset_key=None, history_candidates=None):
    """在官方 MLP 结构上续训一轮，写出 checkpoint.npz 与 train_report.json。

    `dataset_key` 用于识别 C 线（"C"）：C 线回归启用 NAS 闸门与 δ 偏置
    （TASK.md：C 主指标 RMSE，NAS 辅助；形态 = RMSE 主选 + NAS 闸门）。
    `history_candidates` 是 state/history 里的历史候选（[{round, path, sha256}]），
    仅供 B/C 的 soup 扩对（v7 J1）；A 线是分类，`task == "regression"` 守门，不受影响。
    """
    import numpy as np

    started = time.monotonic()
    params = dict(params or {})
    profile = dict(ARM_PROFILES.get(arm) or ARM_PROFILES["A"])
    profile.update({key: params[key] for key in ("scope", "epochs", "lr_multiplier")
                    if key in params})
    # v5 步幅衰减表：后段轮次压 lr；B 线不再被表覆盖（epochs 与 lr 都保留臂 profile），
    # 避免"表里 2 轮/0.5 倍 < 臂 profile 3 轮/1.0 倍"把唯一有效的训练配置切短。
    # 环境变量（CONTEST_TRAIN_EPOCHS / CONTEST_TRAIN_LR_MULT）仍可覆盖，供离线实验。
    plan = step_plan(arm, round_index)
    if plan and str(dataset_key or "") != "B":
        profile["epochs"] = plan["epochs"]
        profile["lr_multiplier"] = plan["lr_multiplier"]
    if profile["scope"] not in ("all", "head"):
        raise TrainMlpError("scope=%r 不是 all/head" % (profile["scope"],))

    parent = _load_parent(parent_path)
    if parent["task"] == "classification":
        classes = parent["meta"].get("classes")
        if not isinstance(classes, list) or len(classes) != int(
                np.asarray(parent["arrays"]["w2"]).shape[1]):
            raise TrainMlpError("分类父产物的 metadata.classes 缺失或与 w2 列数不符")
        classes = [str(item) for item in classes]
    else:
        classes = []
    mean = np.asarray(parent["arrays"]["mean"], dtype=float).reshape(-1)
    scale = np.asarray(parent["arrays"]["scale"], dtype=float).reshape(-1)
    target_mean = float(parent["meta"].get("target_mean", 0.0))
    target_scale = float(parent["meta"].get("target_scale", 1.0))
    if parent["task"] == "regression" and (not math.isfinite(target_scale)
                                           or target_scale == 0.0):
        raise TrainMlpError("回归父产物的 target_scale 非法：%r" % target_scale)

    x_train, y_train, x_val, y_val, stats = _load_dataset(
        data_dir, parent["paths"], classes, parent["task"])
    if x_train.shape[0] < 2:
        raise TrainMlpError("训练集可用行不足（%d 行）：缺 answer 或结构不符"
                            % x_train.shape[0])
    if x_val.shape[0] < 1:
        raise TrainMlpError("验证集可用行不足（%d 行）" % x_val.shape[0])

    z_train = np.clip((x_train - mean) / scale, -10.0, 10.0)
    z_val = np.clip((x_val - mean) / scale, -10.0, 10.0)
    w = {name: np.asarray(parent["arrays"][name], dtype=float).copy()
         for name in ("w1", "b1", "w2", "b2")}
    if parent["task"] == "classification":
        y_train = np.asarray(y_train, dtype=np.int64)
        y_val = np.asarray(y_val, dtype=np.int64)
    else:
        # 训练目标是归一化后的量；验证指标用原始单位（predict 会反归一化）。
        y_val = np.asarray(y_val, dtype=float)
        y_train = (np.asarray(y_train, dtype=float) - target_mean) / target_scale

    parent_metric = _metric(parent["task"], z_val, y_val, w, target_mean, target_scale)
    metric_name = "macro_f1" if parent["task"] == "classification" else "rmse"
    base = parent["meta"]
    lr = float(base.get("learning_rate", 1e-3)) * float(profile["lr_multiplier"])
    batch = int(base.get("batch_size", 128) or 128)
    epochs = int(os.environ.get(ENV_EPOCHS) or profile["epochs"])
    lr *= float(os.environ.get(ENV_LR) or 1.0)
    seed = int(base.get("seed", 42) or 42)
    parent_epochs = int(base.get("epochs", 0) or 0)
    parent_steps = int(base.get("steps", 0) or 0)
    deadline = started + float(DEFAULTS["max_seconds"])

    _log("%s：%s，task=%s，train=%d val=%d，scope=%s lr=%g batch=%d epochs=%d"
         % (arm, metric_name, parent["task"], x_train.shape[0], x_val.shape[0],
            profile["scope"], lr, batch, epochs))

    history, records = [], []
    hist_candidates = list(history_candidates or [])  # v7 J1：历史候选端点（非训练曲线）
    best_metric, best_epoch, epochs_run, steps_run = parent_metric, 0, 0, 0
    final = {name: w[name].copy() for name in w}
    final_moments, fresh_adam = _load_moments(parent["arrays"], w)
    l2sp_report = {"status": "no_training", "lambda": 0.0, "d_target": None,
                   "triggered": False, "unconverged": False, "probe_calls": 0,
                   "trace": [], "would_be": []}
    ridge_choice = None
    if profile["scope"] == "head" and parent["task"] == "regression" and epochs > 0:
        l2sp_report = {"status": "not_applicable_head", "lambda": 0.0,
                       "reason": "岭头是闭式解，没有梯度训练，L2-SP 无从注入",
                       "triggered": False, "unconverged": False, "probe_calls": 0,
                       "trace": [], "would_be": []}
        solutions, records = _ridge_head(z_train, y_train, w, w["w2"].shape,
                                         DEFAULTS["ridge_ratios"])
        scored = []
        for (w2, b2), record in zip(solutions, records):
            trial = dict(w)
            trial["w2"], trial["b2"] = w2, b2
            metric = _metric(parent["task"], z_val, y_val, trial, target_mean, target_scale)
            record["val_rmse"] = metric
            record["w2_norm"] = float(np.linalg.norm(w2))
            scored.append((float(record["ridge_ratio"]), metric, trial))
            final = {name: trial[name].copy() for name in trial}
        # 验证最优就在 λ 的下边界（0~1e-5）时容忍带已归零，这里退化成 argmin。
        lowest = min(item[1] for item in scored)
        ceiling = lowest * (1.0 + RIDGE_TOLERANCE)
        ratio, metric, trial = max((item for item in scored if item[1] <= ceiling),
                                   key=lambda item: item[0])
        ridge_choice = {"ratio": ratio, "val": metric,
                        "best_val": lowest, "tolerance": RIDGE_TOLERANCE}
        if metric < parent_metric - 1e-12:
            best_metric, best_epoch = metric, 1
            w = trial
        epochs_run, steps_run = (1 if best_epoch else 0), 0
        history = [{"epoch": 1, "ridge": True, "ridge_ratio": ratio,
                    "train_mse": records[-1]["train_mse"] if records else None,
                    "val": best_metric}]
    elif epochs > 0:
        steps_per_epoch = int(math.ceil(float(z_train.shape[0]) / float(max(1, batch))))
        planned_steps = max(1, steps_per_epoch * epochs)
        arm_cap = weight_rel_cap(arm)
        d_target, d_target_source = _l2sp_d_target(params, arm)
        start = {name: w[name].copy() for name in w}
        trainable = (("w2", "b2") if profile["scope"] == "head"
                     else ("w1", "b1", "w2", "b2"))

        def _train_once(lam):
            trial_w = {name: start[name].copy() for name in start}
            return _adam(
                epochs, lr, DEFAULTS["grad_clip"], parent["arrays"], parent["task"],
                z_train, y_train, trial_w, batch, seed, parent_epochs, parent_steps,
                DEFAULTS["patience"], z_val, y_val, target_mean, target_scale, deadline,
                trainable=trainable,
                l2sp=({"lam": float(lam), "start": start} if lam > 0.0 else None))

        class _CalibrationAbort(Exception):
            pass

        cache, probes = {}, []

        def _measure(lam_key):
            key = float(lam_key)
            if key in cache:
                return cache[key][1]
            if probes and time.monotonic() > deadline - L2SP_RESERVE_SECONDS:
                raise _CalibrationAbort()
            result = _train_once(key)
            rel = _endpoint_rel_l2(result[0], start)
            cache[key] = (result, rel)
            probes.append({"lambda": key, "rel_l2": rel})
            return rel

        l2sp_report = {
            "d_target": d_target, "d_target_source": d_target_source, "cap": arm_cap,
            "eta": lr, "steps": planned_steps,
            "lambda": 0.0, "triggered": False, "unconverged": False,
            "d0": None, "d0_tensors": {}, "probe_calls": 0, "trace": [], "would_be": [],
        }
        if d_target is None:
            # 预登记值未定：锚定关闭（λ=0），但 trace 照打——本轮端点位移 d0 就是
            # 选 d_target 的依据（必须落在 (d0, cap] 之间），"如果启用"的 λ 量级
            # 用模块的解析初值给出，不额外训练。
            d0 = _measure(0.0)
            l2sp_report.update({
                "status": "not_configured",
                "reason": "d_target 未预登记，L2-SP 关闭（dry-run 只留痕）",
                "d0": d0,
                "d0_tensors": {name: (round(item["rel_l2"], 6) if item["rel_l2"] is not None
                                      else None)
                               for name, item in _diff_stats(cache[0.0][0][0], start).items()},
                "would_be": _l2sp_would_be(d0, lr, planned_steps),
            })
            chosen_lambda = 0.0
        else:
            calibration = None
            try:
                calibration = _l2sp.calibrate_lambda(_measure, d_target, lr, planned_steps)
            except _CalibrationAbort:
                calibration = None
            if calibration is None:
                status, reason, chosen_lambda = "aborted_deadline", "标定探针撞上时间上限，回退 λ=0", 0.0
            else:
                chosen_lambda = float(calibration["lambda"])
                # 只有真的启用锚定才算 active；未触发/退化回退都记 not_triggered，
                # 否则账本会把"λ=0 没介入"读成"锚在工作"。
                status = "active" if calibration["triggered"] else "not_triggered"
                reason = calibration.get("reason")
                l2sp_report.update({
                    "triggered": bool(calibration["triggered"]),
                    "unconverged": bool(calibration["unconverged"]),
                    "d0": calibration.get("d0"),
                })
            if chosen_lambda > 0.0:
                try:
                    _measure(chosen_lambda)
                except _CalibrationAbort:
                    chosen_lambda = 0.0
                    status = "aborted_deadline"
                    reason = "选定 λ 的复训撞上时间上限，回退 λ=0"
            l2sp_report.update({"status": status, "reason": reason,
                                "lambda": chosen_lambda})
        l2sp_report["trace"] = list(probes)
        l2sp_report["probe_calls"] = max(0, len(cache) - 1)
        if l2sp_report.get("d0") is None and probes:
            l2sp_report["d0"] = probes[0]["rel_l2"]
        if not l2sp_report.get("d0_tensors") and 0.0 in cache:
            l2sp_report["d0_tensors"] = {
                name: (round(item["rel_l2"], 6) if item["rel_l2"] is not None else None)
                for name, item in _diff_stats(cache[0.0][0][0], start).items()}
        (w, final, history, epochs_run, steps_run, best_metric, best_epoch,
         final_moments, fresh_adam) = cache[chosen_lambda][0]
        _log("%s：L2-SP %s，d0=%s cap=%.4f λ=%g（探针 %d 次）"
             % (arm, l2sp_report["status"], l2sp_report["d0"], arm_cap,
                chosen_lambda, l2sp_report["probe_calls"]))

    if not all(np.all(np.isfinite(array)) for array in list(w.values()) + list(final.values())):
        raise TrainMlpError("训练后出现非有限权重，拒绝产出候选")

    trained = {name: w[name].copy() for name in w}
    keep_guard = params.get("keep_parent_if_worse", DEFAULTS["keep_parent_if_worse"])
    if os.environ.get(ENV_NO_GUARD) == "0":
        keep_guard = False
    improved = ((best_metric > parent_metric + 1e-12) if parent["task"] == "classification"
                else (best_metric < parent_metric - 1e-12))
    selection = "best_epoch" if improved else ("parent_kept" if keep_guard else "last_epoch")
    if improved:
        submitted = {name: w[name].copy() for name in w}
    elif keep_guard:
        submitted = {name: np.asarray(parent["arrays"][name], dtype=float).copy()
                     for name in w}
    else:
        submitted = {name: final[name].copy() for name in w}
    weights_changed = any(
        not np.array_equal(submitted[name], np.asarray(parent["arrays"][name], dtype=float))
        for name in submitted)

    # ---- 回归线加固：soup 插值（过 REL_CAP 与 NAS 闸门）+ C 线 δ 偏置 ----
    # 依据（2026-11-20 研判）：三线公开→隐藏迁移率与更新幅度单调反向；C 岭头
    # w2 相对变化 80.6%，拼接 rel_l2≈0.4，必须插值收缩；C 的 NAS 由尾部少数
    # 大误差样本支配，RMSE 主选会放过 NAS 灾难候选（R7→R9 +134%）。
    # v7.1（2026-11-22 c-push）：C 的 NASA 不计分，NAS 闸门与 δ 偏置默认关闭
    # （CONTEST_NAS_GUARD 缺省值由 "1" 改为 "0"，需要旧行为时显式置 1）。
    # C4 五折实测：闸门 5/5 折从不 binding（开/关逐折同值）；真正花掉 RMSE 的是
    # δ 偏置——真实 app 路径 5 折均值 −0.2987（55.136673 → 54.838013），3/5 折变差、
    # fold4 单项 +2.32（b2 下移 −0.2）。
    soup_info = None
    nas_info = None
    # 诊断用：soup 没跑时也要能回答"为什么没跑"（weights_changed / env_off /
    # dedup_remaining_endpoints / lineage_gate / final_round_gate）。只落账，不参与决策。
    parent_sha16_hint = _sha256_file(parent_path)[:16]
    history_before = [str(item.get("sha256") or "")[:16]
                      for item in (hist_candidates or [])]
    history_after = []
    for sha16 in history_before:
        if sha16 and sha16 != parent_sha16_hint and sha16 not in history_after:
            history_after.append(sha16)
    soup_skip = None
    # v7 J1b：keep 守卫把父权重留下时，soup 仍以"本轮训练终点"为端点。
    # 否则 submitted 与父逐字节相同，插值退化，历史对也永远看不到训练方向。
    soup_endpoint = submitted if weights_changed else final
    endpoint_source = "submitted" if weights_changed else "final"
    soup_endpoint_moved = any(
        not np.array_equal(np.asarray(soup_endpoint[name], dtype=float),
                           np.asarray(parent["arrays"][name], dtype=float))
        for name in ("w1", "b1", "w2", "b2"))
    if (parent["task"] == "regression" and soup_endpoint_moved
            and os.environ.get(ENV_SOUP, "1") != "0"):
        # 资格线必须与最终候选自检同源：check_candidate 用的是 dataset_key（线），
        # 不是本轮选中的适应臂。此前用 arm，B/C 线回退到 A 臂时会把上限放宽成 A 的
        # 累计步幅预算（10 轮 0.1696），随后又被最终自检按 0.1 拒掉，白跑一轮。
        cap = weight_rel_cap(dataset_key or arm)
        nas_on = (str(dataset_key or "") == "C"
                  and os.environ.get(ENV_NAS_GUARD, "0") != "0")
        raw_tol = os.environ.get(ENV_NAS_TOL)
        nas_tol = float(raw_tol) if raw_tol else NAS_TOL_DEFAULT
        parent_arrays = {name: np.asarray(parent["arrays"][name], dtype=float)
                         for name in ("w1", "b1", "w2", "b2")}
        pred_parent = _predict(z_val, parent_arrays["w1"], parent_arrays["b1"],
                               parent_arrays["w2"], parent_arrays["b2"],
                               "regression", target_mean, target_scale)
        nas_parent = nasa_score(y_val, pred_parent) if nas_on else None
        is_c = (str(dataset_key or "") == "C")
        variants = []
        parent_sha16 = _sha256_file(parent_path)[:16]
        trained_sha16 = _sha256_arrays(soup_endpoint)
        for alpha in SOUP_GRID:
            variant = {name: parent_arrays[name] + alpha * (soup_endpoint[name] - parent_arrays[name])
                       for name in ("w1", "b1", "w2", "b2")}
            rel = concat_rel_l2(variant, parent["arrays"])
            pred_v = _predict(z_val, variant["w1"], variant["b1"], variant["w2"],
                              variant["b2"], "regression", target_mean, target_scale)
            rmse_v = float(math.sqrt(float(np.mean((pred_v - y_val) ** 2))))
            nas_v = nasa_score(y_val, pred_v) if nas_on else None
            disp = _pred_displacement(pred_v, pred_parent, target_scale) if is_c else None
            ok = ((disp <= C_PRED_DISP_CAP + 1e-12) if is_c
                  else (rel is None or rel <= cap + 1e-12))
            if nas_on and nas_v > nas_parent * (1.0 + nas_tol):
                ok = False
            variants.append({"alpha": alpha, "rel_l2": rel, "rmse": rmse_v,
                             "nas": nas_v, "ok": bool(ok), "weights": variant,
                             "pred_disp": disp,
                             "pair": "parent-current",
                             "endpoints": [parent_sha16, trained_sha16]})
        # ---- v7 J1：历史对变体（父–R1 / 父–R2 / R1–R2），复用同一插值式 ----
        # 只对 B/C（本块由 task == "regression" 守门）。端点按 sha 去重；血缘
        # （mean/scale/形状）不符或读取失败的端点整对跳过，原因记进 skipped。
        pair_specs, pairs_skipped = _history_pair_specs(parent, parent_path, hist_candidates)
        for spec in pair_specs:
            for alpha in SOUP_GRID:
                variant = {name: spec["a"][name] + alpha * (spec["b"][name] - spec["a"][name])
                           for name in ("w1", "b1", "w2", "b2")}
                rel = concat_rel_l2(variant, parent["arrays"])
                pred_v = _predict(z_val, variant["w1"], variant["b1"], variant["w2"],
                                  variant["b2"], "regression", target_mean, target_scale)
                rmse_v = float(math.sqrt(float(np.mean((pred_v - y_val) ** 2))))
                nas_v = nasa_score(y_val, pred_v) if nas_on else None
                disp = _pred_displacement(pred_v, pred_parent, target_scale) if is_c else None
                ok = ((disp <= C_PRED_DISP_CAP + 1e-12) if is_c
                      else (rel is None or rel <= cap + 1e-12))
                if nas_on and nas_v > nas_parent * (1.0 + nas_tol):
                    ok = False
                variants.append({"alpha": alpha, "rel_l2": rel, "rmse": rmse_v,
                                 "nas": nas_v, "ok": bool(ok), "weights": variant,
                                 "pred_disp": disp,
                                 "pair": spec["id"], "endpoints": list(spec["endpoints"])})
        eligible = [item for item in variants if item["ok"]]
        chosen_v = min(eligible, key=lambda item: item["rmse"]) if eligible else None
        if chosen_v is not None and chosen_v["rmse"] < parent_metric - 1e-12:
            submitted = chosen_v["weights"]
            weights_changed = True
        else:
            # 没有任何过闸门的插值比父产物好：逐字节退回父权重（与 keep_guard 同口径）
            submitted = {name: parent_arrays[name].copy() for name in parent_arrays}
            weights_changed = False
            selection = "parent_kept"
            chosen_v = None
        soup_info = {"cap": cap,
                     "metric": ("pred_disp" if is_c else "rel_l2"),
                     "cap_value": (C_PRED_DISP_CAP if is_c else cap),
                     "grid": [{k: v for k, v in item.items() if k != "weights"}
                                          for item in variants],
                     "alpha": chosen_v["alpha"] if chosen_v else None,
                     "pair": chosen_v.get("pair") if chosen_v else None,
                     "endpoints": chosen_v.get("endpoints") if chosen_v else None,
                     "rel_l2": chosen_v["rel_l2"] if chosen_v else None,
                     "pred_disp": chosen_v.get("pred_disp") if chosen_v else None,
                     "rmse": chosen_v["rmse"] if chosen_v else None,
                     "pairs": [{"pair": spec["id"], "endpoints": list(spec["endpoints"])}
                               for spec in pair_specs],
                     "skipped": pairs_skipped,
                     "history": {"parent": parent_sha16_hint,
                                 "before": history_before,
                                 "after": history_after,
                                 "n_cross_pairs": len(pair_specs)},
                     "skip_reason": (None if pair_specs
                                     else ("dedup_remaining_endpoints" if not pairs_skipped
                                           else "lineage_gate")),
                     "n_variants": len(variants),
                     "n_eligible": len(eligible),
                     "endpoint": endpoint_source,
                     "endpoint_rel_l2": concat_rel_l2(soup_endpoint, parent["arrays"]),
                     "note": " Adam 矩与步数对应训练端点；α<1 时权重为插值" }
        # δ 偏置：只在 C 线、候选仍保留时。约束 RMSE 升幅 ≤1% 下最小化 NAS。
        if nas_on and weights_changed and chosen_v is not None:
            rmse0, nas0 = chosen_v["rmse"], chosen_v["nas"]
            best_delta, best_nas_d, best_rmse_d = 0.0, nas0, rmse0
            for delta in DELTA_GRID[1:]:
                trial = dict(submitted)
                trial["b2"] = trial["b2"] + delta
                pred_d = _predict(z_val, trial["w1"], trial["b1"], trial["w2"],
                                  trial["b2"], "regression", target_mean, target_scale)
                rmse_d = float(math.sqrt(float(np.mean((pred_d - y_val) ** 2))))
                if rmse_d > rmse0 * (1.0 + DELTA_RMSE_TOL):
                    continue
                nas_d = nasa_score(y_val, pred_d)
                if nas_d < best_nas_d - 1e-9:
                    best_delta, best_nas_d, best_rmse_d = delta, nas_d, rmse_d
            if best_delta:
                submitted = dict(submitted)
                submitted["b2"] = submitted["b2"] + best_delta
            nas_info = {"parent": nas_parent, "before_delta": nas0,
                        "after": best_nas_d, "delta_b2": best_delta,
                        "rmse": best_rmse_d, "rmse_tol": DELTA_RMSE_TOL,
                        "nas_tol": nas_tol}
        elif nas_on:
            nas_info = {"parent": nas_parent, "note": "候选未过闸门，未做 δ 偏置",
                        "nas_tol": nas_tol}
    if soup_info is None:
        if parent["task"] != "regression":
            reason = "not_regression"
        elif not soup_endpoint_moved:
            reason = "endpoint_equal_parent"
        elif not weights_changed:
            reason = "weights_changed_false"
        elif os.environ.get(ENV_SOUP, "1") == "0":
            reason = "env_off"
        else:
            reason = "unknown"
        soup_skip = {"reason": reason, "parent": parent_sha16_hint,
                     "history_before": history_before,
                     "history_after": history_after,
                     "weights_changed": bool(weights_changed)}

    # ---- v7 J2：fresh Adam 续训装载留痕（行为不变，只把"静默清零"变成可对账项）----
    # 父产物若是权重空间插值，metadata 会声明 resume_policy=fresh_adam_on_missing_state；
    # 这里用同一份清单反向对账：声明了就必须四张量全 fresh，没声明就不该有 fresh。
    interpolation = parent["meta"].get("interpolation")
    resume_declared = (interpolation.get("resume_policy")
                       if isinstance(interpolation, dict) else None)
    adam_ran = bool(epochs_run and profile["scope"] != "head")
    if not fresh_adam:
        resume_consistent = True     # 无 fresh，谈不上矛盾
    elif resume_declared == "fresh_adam_on_missing_state":
        resume_consistent = set(fresh_adam) == {"w1", "b1", "w2", "b2"}
    elif resume_declared is None:
        resume_consistent = None     # 父产物无声明（官方父产物即如此），无从对账
    else:
        resume_consistent = False
    if resume_consistent is False:
        _log("续训装载与父产物声明不符：declared=%r fresh_adam=%s"
             % (resume_declared, ",".join(fresh_adam) or "无"))
    resume = {
        "fresh_adam": list(fresh_adam),
        "declared": resume_declared,
        "consistent": resume_consistent,
        "policy": ("fresh_adam_on_missing_state" if fresh_adam
                   else "resume_from_parent_moments"),
        "adam_ran": adam_ran,
        "parent_steps": int(parent_steps),
        "parent_epochs": int(parent_epochs),
    }

    diff = _diff_stats(submitted, parent["arrays"])
    trained_diff = _diff_stats(trained, parent["arrays"])
    report = {
        "schema": "contest.v2.train-report",
        "arm": arm, "task": parent["task"], "scope": profile["scope"],
        "parent_sha256": _sha256_file(parent_path),
        "seed": seed,
        "data": {"feature_dim": parent["n_features"], "n_train": int(x_train.shape[0]),
                 "n_val": int(x_val.shape[0]), **stats},
        "hyper": {"lr": lr, "batch": batch, "epochs_planned": epochs,
                  "epochs_run": epochs_run, "steps": steps_run,
                  "patience": DEFAULTS["patience"], "grad_clip": DEFAULTS["grad_clip"]},
        "step_plan": dict(plan or {}, round_index=round_index) if plan else None,
        "loss": {"start": history[0].get("train_loss") if history else None,
                 "end": history[-1].get("train_loss") if history else None,
                 "by_epoch": history},
        "val": {"metric": metric_name, "parent": parent_metric, "best": best_metric,
                "best_epoch": best_epoch, "final": history[-1].get("val") if history else None,
                "ridge": records},
        "selection": selection,
        "resume": resume,
        "l2sp": l2sp_report,
        "weights_changed": bool(weights_changed),
        "diff": diff,
        "trained_diff": trained_diff,
        "soup": soup_info,
        "soup_skip": soup_skip,
        "nas": nas_info,
        "seconds": round(time.monotonic() - started, 3),
    }

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, CHECKPOINT_NAME)
    if not weights_changed:
        # 权重与父产物完全一致时逐字节复制父文件：不写新 metadata（避免"说训过、
        # 权重却回退"），也不留下与 steps 错配的 Adam 状态。
        shutil.copyfile(parent_path, checkpoint_path)
        report["artifact"] = "parent_copy"
    else:
        arrays = {name: parent["arrays"][name] for name in parent["arrays"]}
        for name in ("w1", "b1", "w2", "b2"):
            arrays[name] = np.ascontiguousarray(submitted[name], dtype=float)
        # Adam 矩与步数是续训状态的一部分：必须写回训练末态，而不是沿用父产物的旧矩。
        for name, (first, second) in final_moments.items():
            arrays["adam_first_%s" % name] = np.ascontiguousarray(first, dtype=float)
            arrays["adam_second_%s" % name] = np.ascontiguousarray(second, dtype=float)
        meta = dict(parent["meta"])
        # 父产物的输出侧校准是上一轮斜率的产物，对新权重已失效；本轮由校准候选族重新选。
        meta.pop("calibration", None)
        meta["steps"] = int(meta.get("steps", 0) or 0) + int(steps_run)
        meta["epochs"] = int(meta.get("epochs", 0) or 0) + int(epochs_run)
        meta["finetune"] = {
            "arm": arm, "scope": profile["scope"], "selection": selection,
            "steps": int(steps_run), "epochs": int(epochs_run), "lr": lr,
            "batch": batch, "seed": seed, "weights_changed": bool(weights_changed),
            "val": {"metric": metric_name, "parent": parent_metric, "after": best_metric},
            "parent_sha256": report["parent_sha256"],
            # 本轮装载时的 Adam 续训口径：下一轮据此对账（fresh 清单非空即为 fresh Adam）。
            "adam_resume": {"fresh": list(fresh_adam), "declared": resume_declared,
                            "policy": resume["policy"]},
            # L2-SP：λ 与标定状态。未登记 d_target 时恒为 λ=0（状态 not_configured）。
            "l2sp": {"status": l2sp_report.get("status"), "lambda": l2sp_report.get("lambda", 0.0),
                     "d_target": l2sp_report.get("d_target"),
                     "triggered": l2sp_report.get("triggered", False)},
        }
        if soup_info and soup_info.get("alpha") is not None:
            meta["finetune"]["soup_alpha"] = soup_info["alpha"]
            meta["finetune"]["soup_metric"] = soup_info.get("metric")
            meta["finetune"]["soup_cap_value"] = soup_info.get("cap_value")
            meta["finetune"]["soup_pred_disp"] = soup_info.get("pred_disp")
            meta["finetune"]["soup_rel_l2"] = soup_info["rel_l2"]
            meta["finetune"]["soup_pair"] = soup_info.get("pair")
            meta["finetune"]["soup_endpoints"] = soup_info.get("endpoints")
            if soup_info.get("pair") and soup_info["pair"] != "parent-current":
                # 上传权重是历史对插值，不是本轮训练端点：steps/epochs 只作排程锚。
                meta["finetune"]["steps_semantics"] = "schedule_anchor_not_trained_steps"
        if nas_info and nas_info.get("delta_b2"):
            meta["finetune"]["delta_b2"] = nas_info["delta_b2"]
        arrays["metadata"] = np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True))
        np.savez(checkpoint_path, **arrays)
        # 断言：下一轮 `_load_parent` 读到的矩/步数必须等于本轮末状态。
        _assert_saved_state(checkpoint_path, submitted, final_moments, meta["steps"])
        report["artifact"] = "retrained"
        report["adam"] = {"steps": int(steps_run), "saved": True}
    # v7 J1c：父权重被 keep 守卫留下时，把本轮训练终点另存为历史端点候选，
    # 避免"每轮上传都是父副本 → 历史端点同 sha 去重只剩一条"。该文件不上传。
    if parent["task"] == "regression" and not weights_changed and soup_endpoint_moved:
        endpoint_path = os.path.join(out_dir, "trained-endpoint.npz")
        arrays = {name: parent["arrays"][name] for name in parent["arrays"]}
        for name in ("w1", "b1", "w2", "b2"):
            arrays[name] = np.ascontiguousarray(soup_endpoint[name], dtype=float)
        endpoint_meta = dict(parent["meta"])
        endpoint_meta.pop("calibration", None)
        endpoint_meta["endpoint"] = {
            "round": int(round_index), "selection": selection,
            "val": {"metric": metric_name, "parent": parent_metric,
                    "final": (history[-1].get("val") if history else None)},
            "rel_l2": (soup_info or {}).get("endpoint_rel_l2"),
            "note": "trained endpoint not submitted; history candidate",
        }
        arrays["metadata"] = np.array(json.dumps(endpoint_meta, ensure_ascii=False,
                                                 sort_keys=True))
        np.savez(endpoint_path, **arrays)
        report["trained_endpoint"] = {
            "path": endpoint_path, "sha256": _sha256_file(endpoint_path),
            "rel_l2": (soup_info or {}).get("endpoint_rel_l2"),
        }
    with open(os.path.join(out_dir, REPORT_NAME), "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    _log("完成：%s，%s %s→%.6g，%s，%.2fs"
         % (arm, metric_name, parent_metric, best_metric, selection, report["seconds"]))
    return report


# ---------------------------------------------------------------- 工具


def _assert_saved_state(path, weights, moments, steps):
    """断言落盘状态可被下一轮原样读回：权重、Adam 矩与累计步数逐位一致。"""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        for name in ("w1", "b1", "w2", "b2"):
            assert np.array_equal(archive[name], weights[name]), \
                "写入的 %s 与内存权重不一致" % name
            first, second = moments[name]
            assert np.array_equal(archive["adam_first_%s" % name], first), \
                "写入的 adam_first_%s 与训练末态不一致" % name
            assert np.array_equal(archive["adam_second_%s" % name], second), \
                "写入的 adam_second_%s 与训练末态不一致" % name
        meta = json.loads(str(np.asarray(archive["metadata"]).tolist()))
        assert int(meta.get("steps", -1)) == int(steps), \
            "写入的 steps=%r 与训练末态 %r 不一致" % (meta.get("steps"), steps)


def _sha256_arrays(weights):
    """四张量的确定性摘要（用于给"训练端点"一个可对账的 sha16）。"""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for name in ("w1", "b1", "w2", "b2"):
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(np.asarray(weights[name], dtype=float)).tobytes())
    return digest.hexdigest()


def _history_pair_specs(parent, parent_path, history):
    """把 state/history 的历史候选配成端点对：父–R1 / 父–R2 / R1–R2。

    端点按 sha 去重（与父产物同 sha 的直接跳过）；血缘闸门只做本地可判的三件事：
    四张量形状与父产物一致、mean/scale 逐位一致、文件可读。任一不符整对不进
    候选族，原因写进 `skipped`。返回 (specs, skipped)。
    """
    import numpy as np

    parent_arrays = {name: np.asarray(parent["arrays"][name], dtype=float)
                     for name in ("w1", "b1", "w2", "b2")}
    parent_mean = np.asarray(parent["arrays"]["mean"], dtype=float)
    parent_scale = np.asarray(parent["arrays"]["scale"], dtype=float)
    endpoints = [{"id": "parent", "sha256": _sha256_file(parent_path),
                  "arrays": parent_arrays}]
    skipped = []
    for entry in history or []:
        if len(endpoints) >= 3:  # 父 + 两个历史端点，足够组三对
            break
        sha = str(entry.get("sha256") or "")
        path = entry.get("path") or ""
        label = "R%s" % entry.get("round")
        if not sha or sha == endpoints[0]["sha256"] or any(e["sha256"] == sha for e in endpoints):
            continue
        try:
            with np.load(path, allow_pickle=False) as store:
                arrays = {name: np.asarray(store[name], dtype=float)
                          for name in ("w1", "b1", "w2", "b2")}
                mean = np.asarray(store["mean"], dtype=float)
                scale = np.asarray(store["scale"], dtype=float)
        except Exception as exc:  # noqa: BLE001 - 读不了就不进族
            skipped.append({"endpoint": label, "reason": "读取失败：%s" % str(exc)[:80]})
            continue
        bad = [name for name in ("w1", "b1", "w2", "b2")
               if arrays[name].shape != parent_arrays[name].shape]
        if bad:
            skipped.append({"endpoint": label,
                            "reason": "形状不符：%s" % "/".join(bad)})
            continue
        if not (np.array_equal(mean, parent_mean) and np.array_equal(scale, parent_scale)):
            skipped.append({"endpoint": label, "reason": "mean/scale 与父产物不一致"})
            continue
        endpoints.append({"id": label, "sha256": sha, "arrays": arrays})
    specs = []
    for first in range(len(endpoints)):
        for second in range(first + 1, len(endpoints)):
            left, right = endpoints[first], endpoints[second]
            specs.append({
                "id": "%s--%s" % (left["id"], right["id"]),
                "endpoints": [left["sha256"][:16], right["sha256"][:16]],
                "a": left["arrays"], "b": right["arrays"],
            })
    return specs, skipped


def _endpoint_rel_l2(weights, start):
    """本轮训练端点相对起点的位移：与权重闸门同口径的拼接 rel_l2。

    2026-11-22 修正：原实现是"逐张量 rel_l2 取最大"，注释虽写"与权重闸门同口径"
    却不成立——闸门与账本用的是 concat_rel_l2（拼接向量 ‖Δ‖/‖父‖），逐张量 max
    恒 ≥ 拼接值。d_target=0.15 是按闸门的拼接读数（最坏 0.1332）登记的，两把尺子
    混用会让 L2-SP 触发判定落在目标带从未标定过的区间里（b2 单张量 0.275 就能把
    d0 顶到 0.27，而闸门看到的是 0.084）。
    """
    rel = concat_rel_l2(weights, start)
    return 0.0 if rel is None else float(rel)


def _l2sp_d_target(params, arm):
    """d_target 的预登记值：显式参数 > 环境变量 > 按臂预登记表；都没有 = 未登记。

    v7 预登记（2026-11-22 定案）：A = 0.15。依据是平台回执的逐轮 weight_rel_l2
    （最坏一步 0.1332），0.1332 < 0.15 < cap 0.1696，留 12.4% 余量。B/C 未登记。
    未登记就不启用锚定（λ=0），也不拿 cap 或 d0 顶上：1.1×cap 会被闸门先拒、
    1.0×d0 会常态触发，两者都把可移动判据当成了判据。
    """
    raw = params.get("l2sp_d_target") if isinstance(params, dict) else None
    source = "params"
    if raw is None:
        raw = os.environ.get(ENV_L2SP_D_TARGET)
        source = "env"
    if raw is None:
        raw = L2SP_D_TARGET_PREREGISTERED.get(str(arm or "").upper())
        source = "preregistered"
    if raw is None or not str(raw).strip():
        return None, "unregistered"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TrainMlpError("L2-SP d_target 不是数字：%r（来源 %s）" % (raw, source))
    if not math.isfinite(value) or value <= 0.0:
        raise TrainMlpError("L2-SP d_target 必须是正有限数：%r" % (value,))
    return value, source


def _l2sp_would_be(d0, eta, steps):
    """dry-run：若把 d_target 钉在 0.5/0.7/0.9×d0，解析初值会给到多大 λ。

    只用模块的 `lambda_init` / `expected_rel_l2`，不训练；"expected_rel_l2" 是
    常梯度近似，仅供看量级，不作判据。
    """
    out = []
    for ratio in L2SP_WOULD_BE_RATIOS:
        try:
            lam = _alpha_lib.lambda_init(eta, steps, ratio)
            out.append({"d_target_over_d0": ratio, "lambda_init": lam,
                        "expected_rel_l2": _l2sp.expected_rel_l2(d0, lam, eta, steps)})
        except _alpha_lib.InterpolationError:
            continue
    return out


def _diff_stats(weights, parent_arrays):
    import numpy as np

    out = {}
    for name in ("w1", "b1", "w2", "b2"):
        current = np.asarray(weights[name], dtype=float)
        base = np.asarray(parent_arrays[name], dtype=float)
        delta = current - base
        base_norm = float(np.linalg.norm(base))
        out[name] = {
            "shape": list(current.shape),
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "l2": float(np.linalg.norm(delta)),
            "rel_l2": float(np.linalg.norm(delta) / base_norm) if base_norm else None,
        }
    return out


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

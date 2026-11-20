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
DEFAULTS = {
    "patience": 1,
    "grad_clip": 5.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1e-8,
    "ridge_ratios": (0.0, 1e-6, 1e-4, 1e-3, 1e-2),
    "keep_parent_if_worse": True,
    "max_seconds": 600.0,
}
# 允许本地实验覆盖；线上路径不读环境变量之外的东西。
ENV_LR = "CONTEST_TRAIN_LR_MULT"
ENV_EPOCHS = "CONTEST_TRAIN_EPOCHS"
ENV_NO_GUARD = "CONTEST_TRAIN_KEEP_PARENT"
# 岭回归选系数的容忍带：比验证最优差 1.5% 以内时取最大的 λ（最小权重范数）。
RIDGE_TOLERANCE = 0.015

# --- 2026-11-20 加固：权重漂移上限 / soup 插值 / C 线 NAS 闸门与 δ 偏置 ---
# 权重相对父产物的拼接 L2 上限。check_candidate 与 soup 共用同一口径，两处必须同源。
ENV_REL_CAP = "CONTEST_WEIGHT_REL_CAP"
REL_CAP_DEFAULT = 0.1
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
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("非有限数值")
        out.append(number)


def _read_rows(path):
    if not os.path.isfile(path):
        raise TrainMlpError("找不到数据文件：%s" % path)
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), dict):
                bad += 1
                continue
            rows.append(obj)
    return rows, bad


def _extract(rows, shape, n_features, label_fn):
    """按固定结构抽 X/y；结构不符或缺标签的行整行跳过并计数。"""
    import numpy as np

    kept_x, kept_y, stats = [], [], {"rows": len(rows), "bad_shape": 0,
                                     "bad_label": 0, "bad_value": 0}
    for row in rows:
        inputs = row["inputs"]
        try:
            if _shape_of(inputs) != shape:
                stats["bad_shape"] += 1
                continue
            values = []
            _collect(inputs, values)
        except (TypeError, ValueError):
            stats["bad_value"] += 1
            continue
        if len(values) != n_features:
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


def _load_dataset(data_dir, paths, classes, task):
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
    import numpy as np

    _, out = _forward(z, w1, b1, w2, b2)
    if task == "classification":
        return out.argmax(axis=1)
    return np.maximum(0.0, out[:, 0] * target_scale + target_mean)


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
    """NASA 非对称评分（C-MAPSS 口径）。

    d = 预测 − 真实；d≥0（高估剩余寿命）取 exp(d/10)−1，d<0 取 exp(−d/13)−1，
    逐样本求和。公式与常数已对平台回执锚定：父权重在 C 包 validation.jsonl 上
    本地复算 = 5,823,439,325.34，与回执显示值恰差 10³（转录错位，见研判第四份）。
    逐样本求和 ⇒ 跨样本集（公开 23,049 / 隐藏 9,687）不可直接横比（TASK.md:23）。
    """
    import numpy as np

    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d >= 0.0, np.exp(d / 10.0) - 1.0,
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


def weight_rel_cap():
    """权重拼接 rel_l2 上限：默认 0.1，CONTEST_WEIGHT_REL_CAP 覆盖。"""
    raw = os.environ.get(ENV_REL_CAP)
    if raw is None or not str(raw).strip():
        return REL_CAP_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        raise TrainMlpError("%s 不是数字：%r" % (ENV_REL_CAP, raw))
    if not math.isfinite(value) or value <= 0.0:
        raise TrainMlpError("权重 rel_l2 上限必须是正有限数：%r" % (value,))
    return value


def load_yardstick(data_dir, reference_path):
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
            value = float(answer)
            return value if math.isfinite(value) else None
    x_val, y_val, _stats = _extract(rows, shape, n_features, label_fn)
    if x_val.shape[0] < 1:
        raise TrainMlpError("固定标尺为空：验证集没有可用行")
    mean = np.asarray(ref["arrays"]["mean"], dtype=float).reshape(-1)
    scale = np.asarray(ref["arrays"]["scale"], dtype=float).reshape(-1)
    z_val = np.clip((x_val - mean) / scale, -10.0, 10.0)
    if task == "classification":
        return z_val, np.asarray(y_val, dtype=np.int64), task
    return z_val, np.asarray(y_val, dtype=float), task


def score_checkpoint(path, z_val, y_val):
    """在固定标尺上给单个官方格式检查点打分（不回传预测，只回标量）。

    分类：{"task", "macro_f1"}；回归：{"task", "rmse", "nas"}。nas 一并算出，
    是否使用由调用方按 dataset_key 决定（TASK.md：NAS 是 C 的辅助指标）。
    """
    import numpy as np

    ref = _load_parent(path)
    arrays = ref["arrays"]
    meta = ref["meta"]
    pred = _predict(z_val,
                    np.asarray(arrays["w1"], dtype=float),
                    np.asarray(arrays["b1"], dtype=float),
                    np.asarray(arrays["w2"], dtype=float),
                    np.asarray(arrays["b2"], dtype=float),
                    ref["task"],
                    float(meta.get("target_mean", 0.0) or 0.0),
                    float(meta.get("target_scale", 1.0) or 1.0))
    if ref["task"] == "classification":
        return {"task": "classification",
                "macro_f1": _macro_f1(y_val, pred, int(np.asarray(arrays["w2"]).shape[1]))}
    rmse = float(math.sqrt(float(np.mean((pred - y_val) ** 2))))
    return {"task": "regression", "rmse": rmse, "nas": nasa_score(y_val, pred)}


# ---------------------------------------------------------------- 训练


def _load_moments(arrays, w):
    """从父产物读 Adam 一阶/二阶矩；缺失或形状不符时按零初始化。"""
    import numpy as np

    out = {}
    for name in ("w1", "b1", "w2", "b2"):
        first = arrays.get("adam_first_%s" % name)
        second = arrays.get("adam_second_%s" % name)
        if (isinstance(first, np.ndarray) and first.shape == w[name].shape
                and isinstance(second, np.ndarray) and second.shape == w[name].shape):
            out[name] = [np.asarray(first, dtype=float).copy(),
                         np.asarray(second, dtype=float).copy()]
        else:
            out[name] = [np.zeros_like(w[name]), np.zeros_like(w[name])]
    return out


def _adam(epochs, lr, grad_clip, parent, task, z, y, w, batch, seed,
          parent_epochs, parent_steps, patience, z_val, y_val,
          target_mean, target_scale, deadline, trainable=("w1", "b1", "w2", "b2")):
    """全参数 Adam。返回 (最优轮权重, 末轮权重, history, 轮数, 步数, 最优指标, 最优轮, 末态矩)。

    Adam 状态与轮次种子都从父产物续上（与官方 model.fit 的续训口径一致）：
    偏差校正从 `metadata.steps` 继续，洗牌种子用 `[seed, metadata.epochs + epoch]`。
    """
    import numpy as np

    beta1, beta2, eps = DEFAULTS["beta1"], DEFAULTS["beta2"], DEFAULTS["adam_eps"]
    moments = _load_moments(parent, w)
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
            total += loss * len(index)
            grads = {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2}
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
    return best, final, history, epochs_run, step, best_metric, best_epoch, final_moments


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


def finetune(parent_path, out_dir, data_dir, arm="A", round_index=1, params=None,
             dataset_key=None):
    """在官方 MLP 结构上续训一轮，写出 checkpoint.npz 与 train_report.json。

    `dataset_key` 用于识别 C 线（"C"）：C 线回归启用 NAS 闸门与 δ 偏置
    （TASK.md：C 主指标 RMSE，NAS 辅助；形态 = RMSE 主选 + NAS 闸门）。
    """
    import numpy as np

    started = time.monotonic()
    params = dict(params or {})
    profile = dict(ARM_PROFILES.get(arm) or ARM_PROFILES["A"])
    profile.update({key: params[key] for key in ("scope", "epochs", "lr_multiplier")
                    if key in params})
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
    best_metric, best_epoch, epochs_run, steps_run = parent_metric, 0, 0, 0
    final = {name: w[name].copy() for name in w}
    final_moments = _load_moments(parent["arrays"], w)
    ridge_choice = None
    if profile["scope"] == "head" and parent["task"] == "regression" and epochs > 0:
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
        # 验证最优往往落在 λ→0 的边界解上，权重会被放大十几倍；在"比最优差 1.5%
        # 以内"的解里取最大的 λ（最小权重范数），换隐藏集上的稳健性。
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
        (w, final, history, epochs_run, steps_run, best_metric, best_epoch,
         final_moments) = _adam(
            epochs, lr, DEFAULTS["grad_clip"], parent["arrays"], parent["task"],
            z_train, y_train, w, batch, seed, parent_epochs, parent_steps,
            DEFAULTS["patience"], z_val, y_val, target_mean, target_scale, deadline,
            trainable=(("w2", "b2") if profile["scope"] == "head"
                       else ("w1", "b1", "w2", "b2")))

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
    soup_info = None
    nas_info = None
    if (parent["task"] == "regression" and weights_changed
            and os.environ.get(ENV_SOUP, "1") != "0"):
        cap = weight_rel_cap()
        nas_on = (str(dataset_key or "") == "C"
                  and os.environ.get(ENV_NAS_GUARD, "1") != "0")
        raw_tol = os.environ.get(ENV_NAS_TOL)
        nas_tol = float(raw_tol) if raw_tol else NAS_TOL_DEFAULT
        parent_arrays = {name: np.asarray(parent["arrays"][name], dtype=float)
                         for name in ("w1", "b1", "w2", "b2")}
        pred_parent = _predict(z_val, parent_arrays["w1"], parent_arrays["b1"],
                               parent_arrays["w2"], parent_arrays["b2"],
                               "regression", target_mean, target_scale)
        nas_parent = nasa_score(y_val, pred_parent) if nas_on else None
        variants = []
        for alpha in SOUP_GRID:
            variant = {name: parent_arrays[name] + alpha * (submitted[name] - parent_arrays[name])
                       for name in ("w1", "b1", "w2", "b2")}
            rel = concat_rel_l2(variant, parent["arrays"])
            pred_v = _predict(z_val, variant["w1"], variant["b1"], variant["w2"],
                              variant["b2"], "regression", target_mean, target_scale)
            rmse_v = float(math.sqrt(float(np.mean((pred_v - y_val) ** 2))))
            nas_v = nasa_score(y_val, pred_v) if nas_on else None
            ok = (rel is None or rel <= cap + 1e-12)
            if nas_on and nas_v > nas_parent * (1.0 + nas_tol):
                ok = False
            variants.append({"alpha": alpha, "rel_l2": rel, "rmse": rmse_v,
                             "nas": nas_v, "ok": bool(ok), "weights": variant})
        eligible = [item for item in variants if item["ok"]]
        chosen_v = min(eligible, key=lambda item: item["rmse"]) if eligible else None
        if chosen_v is not None and chosen_v["rmse"] < parent_metric - 1e-12:
            submitted = chosen_v["weights"]
        else:
            # 没有任何过闸门的插值比父产物好：逐字节退回父权重（与 keep_guard 同口径）
            submitted = {name: parent_arrays[name].copy() for name in parent_arrays}
            weights_changed = False
            selection = "parent_kept"
            chosen_v = None
        soup_info = {"cap": cap, "grid": [{k: v for k, v in item.items() if k != "weights"}
                                          for item in variants],
                     "alpha": chosen_v["alpha"] if chosen_v else None,
                     "rel_l2": chosen_v["rel_l2"] if chosen_v else None,
                     "rmse": chosen_v["rmse"] if chosen_v else None,
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
        "loss": {"start": history[0].get("train_loss") if history else None,
                 "end": history[-1].get("train_loss") if history else None,
                 "by_epoch": history},
        "val": {"metric": metric_name, "parent": parent_metric, "best": best_metric,
                "best_epoch": best_epoch, "final": history[-1].get("val") if history else None,
                "ridge": records},
        "selection": selection,
        "weights_changed": bool(weights_changed),
        "diff": diff,
        "trained_diff": trained_diff,
        "soup": soup_info,
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
        meta["steps"] = int(meta.get("steps", 0) or 0) + int(steps_run)
        meta["epochs"] = int(meta.get("epochs", 0) or 0) + int(epochs_run)
        meta["finetune"] = {
            "arm": arm, "scope": profile["scope"], "selection": selection,
            "steps": int(steps_run), "epochs": int(epochs_run), "lr": lr,
            "batch": batch, "seed": seed, "weights_changed": bool(weights_changed),
            "val": {"metric": metric_name, "parent": parent_metric, "after": best_metric},
            "parent_sha256": report["parent_sha256"],
        }
        if soup_info and soup_info.get("alpha") is not None:
            meta["finetune"]["soup_alpha"] = soup_info["alpha"]
            meta["finetune"]["soup_rel_l2"] = soup_info["rel_l2"]
        if nas_info and nas_info.get("delta_b2"):
            meta["finetune"]["delta_b2"] = nas_info["delta_b2"]
        arrays["metadata"] = np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True))
        np.savez(checkpoint_path, **arrays)
        # 断言：下一轮 `_load_parent` 读到的矩/步数必须等于本轮末状态。
        _assert_saved_state(checkpoint_path, submitted, final_moments, meta["steps"])
        report["artifact"] = "retrained"
        report["adam"] = {"steps": int(steps_run), "saved": True}
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

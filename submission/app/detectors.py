#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批级漂移检测器（v2 P1）：只允许在 train 阶段运行。

三个检测器都在**整批**上计算，天然是 transductive 的：

    kl            逐特征直方图 KL（对称化），看分布形状变化
    mahalanobis   用 train 的均值/协方差打分 validation 每一行，取均值，看位置与尺度变化
    bbsd          两样本 Hotelling T²（v1 的 BBSD 核心统计量）

v1 的 BBSD 建在 6 维 softmax 上、且只适用于分类；train 阶段拿不到可靠的后验概率时，
这里退化为**特征空间**的 Hotelling T²，并在报告里用 `bbsd_domain` 标明。
这是降级而不是等价替换，读报告时必须看这一项。

`stage` 是硬门禁：只要不是 "train" 就抛 `DetectorNotAllowed`。predict 侧不导入本模块。
"""

import math

STAGE = "train"


class DetectorNotAllowed(Exception):
    """检测器只允许在 train 阶段运行。"""


def _matrix(name, value):
    import numpy as np

    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("%s 必须是二维矩阵，实际形状 %r" % (name, arr.shape))
    if arr.shape[0] < 2:
        raise ValueError("%s 至少需要 2 行，实际 %d" % (name, arr.shape[0]))
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s 含非有限值" % name)
    return arr


def _check_stage(stage):
    if stage != STAGE:
        raise DetectorNotAllowed("批级检测器只允许在 %s 阶段运行，收到 stage=%r" % (STAGE, stage))


def kl_divergence(train, val, bins=10):
    """逐特征直方图 KL 的对称均值。恒定列（无分辨率）跳过，不插补。"""
    import numpy as np

    a = _matrix("train", train)
    b = _matrix("val", val)
    if a.shape[1] != b.shape[1]:
        raise ValueError("train/val 特征维不一致：%d vs %d" % (a.shape[1], b.shape[1]))

    total, used = 0.0, 0
    for j in range(a.shape[1]):
        lo = float(min(a[:, j].min(), b[:, j].min()))
        hi = float(max(a[:, j].max(), b[:, j].max()))
        if not (hi > lo) or not math.isfinite(hi - lo):
            continue  # 两边都是常数：没有可比较的形状
        edges = np.linspace(lo, hi, int(bins) + 1)
        pa, _ = np.histogram(a[:, j], bins=edges)
        pb, _ = np.histogram(b[:, j], bins=edges)
        pa = pa.astype(float) + 1e-9
        pb = pb.astype(float) + 1e-9
        pa /= pa.sum()
        pb /= pb.sum()
        total += float(np.sum(pa * np.log(pa / pb)) + np.sum(pb * np.log(pb / pa))) / 2.0
        used += 1
    return total / used if used else 0.0


def mahalanobis(train, val, ridge=1e-6):
    """validation 各行到 train 高斯（均值+协方差）的平均平方马氏距离。"""
    import numpy as np

    a = _matrix("train", train)
    b = _matrix("val", val)
    if a.shape[1] != b.shape[1]:
        raise ValueError("train/val 特征维不一致：%d vs %d" % (a.shape[1], b.shape[1]))
    d = a.shape[1]
    mean = a.mean(axis=0)
    cov = np.cov(a, rowvar=False) if d > 1 else np.asarray([[a[:, 0].var()]])
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov)) / d if d else 0.0
    cov = cov + np.eye(d) * max(ridge * max(scale, 1.0), 1e-12)
    inv = np.linalg.pinv(cov)
    diff = b - mean
    dist = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return float(np.maximum(dist, 0.0).mean())


def hotelling_t2(train, val, ridge=1e-6):
    """两样本 Hotelling T²（BBSD 的核心统计量）。"""
    import numpy as np

    a = _matrix("train", train)
    b = _matrix("val", val)
    if a.shape[1] != b.shape[1]:
        raise ValueError("train/val 特征维不一致：%d vs %d" % (a.shape[1], b.shape[1]))
    n1, n2 = a.shape[0], b.shape[0]
    d = a.shape[1]
    m1, m2 = a.mean(axis=0), b.mean(axis=0)
    s1 = np.atleast_2d(np.cov(a, rowvar=False)) if d > 1 else np.asarray([[a[:, 0].var()]])
    s2 = np.atleast_2d(np.cov(b, rowvar=False)) if d > 1 else np.asarray([[b[:, 0].var()]])
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1)
    scale = float(np.trace(pooled)) / d if d else 0.0
    pooled = pooled + np.eye(d) * max(ridge * max(scale, 1.0), 1e-12)
    diff = m1 - m2
    t2 = (n1 * n2 / float(n1 + n2)) * float(diff @ np.linalg.pinv(pooled) @ diff)
    return max(t2, 0.0)


def bbsd(train, val, domain="features"):
    """BBSD。domain=features 时就是两样本 Hotelling T²（见模块说明的降级说明）。"""
    if domain not in ("features", "softmax"):
        raise ValueError("bbsd_domain 只能是 features/softmax：%r" % (domain,))
    spec = {"statistic": "hotelling_t2", "domain": domain}
    if domain == "softmax":
        raise DetectorNotAllowed("train 阶段没有可靠后验概率，softmax 域 BBSD 不可用")
    return hotelling_t2(train, val), spec


def run_all(train, val, stage=STAGE, bins=10):
    """跑齐三个检测器，返回可直接写进 selector_trace 的报告。"""
    _check_stage(stage)
    a = _matrix("train", train)
    b = _matrix("val", val)
    t2, spec = bbsd(a, b)
    return {
        "stage": STAGE,
        "inputs": {"n_train": int(a.shape[0]), "n_val": int(b.shape[0]),
                   "features": int(a.shape[1])},
        "detectors": {
            "kl": {"value": kl_divergence(a, b, bins=bins), "unit": "nats/feature"},
            "mahalanobis": {"value": mahalanobis(a, b), "unit": "mean squared distance"},
            "bbsd": {"value": t2, "unit": "hotelling_t2", "note": spec},
        },
    }

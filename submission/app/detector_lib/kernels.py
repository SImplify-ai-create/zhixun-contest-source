# -*- coding: utf-8 -*-
"""19 条检测器的 numpy 实现（无第三方依赖）。

口径与 v1 ``detlib`` 对齐；v7 只拿得到 train/validation 的特征矩阵，
拿不到的层（输出/倒数第二层/logits）一律显式 unavailable，不造数。
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def _matrix(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("%s 必须是二维矩阵，实际形状 %r" % (name, arr.shape))
    if arr.shape[0] < 2:
        raise ValueError("%s 至少需要 2 行，实际 %d" % (name, arr.shape[0]))
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s 含非有限值" % name)
    return arr


def _subsample(arr, cap=400, seed=0):
    """大样本下确定性抽稀，控制 MMD/C2ST 的 O(n²) 成本。"""
    if arr.shape[0] <= cap:
        return arr
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(arr.shape[0], cap, replace=False))
    return arr[idx]


def _cov(arr, ridge=1e-6):
    d = arr.shape[1]
    cov = np.atleast_2d(np.cov(arr, rowvar=False)) if d > 1 else np.asarray([[arr[:, 0].var()]])
    scale = float(np.trace(cov)) / d if d else 0.0
    return cov + np.eye(d) * max(ridge * max(scale, 1.0), 1e-12)


def hist_kl(train, val, bins=10):
    """逐特征直方图对称 KL 均值（与 v7 ``detectors.kl_divergence`` 同口径）。"""
    a, b = _matrix("train", train), _matrix("val", val)
    total, used = 0.0, 0
    for j in range(a.shape[1]):
        lo = float(min(a[:, j].min(), b[:, j].min()))
        hi = float(max(a[:, j].max(), b[:, j].max()))
        if not (hi > lo) or not math.isfinite(hi - lo):
            continue
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


def mahalanobis_mean(train, val):
    """validation 各行到 train 高斯的平均平方马氏距离。"""
    a, b = _matrix("train", train), _matrix("val", val)
    inv = np.linalg.pinv(_cov(a))
    diff = b - a.mean(axis=0)
    return float(np.maximum(np.einsum("ij,jk,ik->i", diff, inv, diff), 0.0).mean())


def hotelling_t2(train, val):
    """两样本 Hotelling T²（BBSD 的核心统计量）。"""
    a, b = _matrix("train", train), _matrix("val", val)
    n1, n2 = a.shape[0], b.shape[0]
    pooled = ((n1 - 1) * _cov(a) + (n2 - 1) * _cov(b)) / max(n1 + n2 - 2, 1)
    diff = a.mean(axis=0) - b.mean(axis=0)
    t2 = (n1 * n2 / float(n1 + n2)) * float(diff @ np.linalg.pinv(pooled) @ diff)
    return max(t2, 0.0)


def probs_to_shares(probs):
    """逐行概率 → 批内类份额（argmax 分布口径）。"""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs 必须是二维矩阵")
    return p.mean(axis=0)


def output_share_kl(shares_train, shares_val, prior=None):
    """argmax 类份额 vs 先验的对称 KL（分类专用）；无先验时用 train 份额。"""
    a = np.asarray(shares_train, dtype=float).ravel()
    b = np.asarray(shares_val, dtype=float).ravel()
    q = np.asarray(prior if prior is not None else a, dtype=float).ravel()
    a, b, q = (np.maximum(x, 1e-9) / max(float(np.maximum(x, 1e-9).sum()), EPS)
               for x in (a, b, q))
    return float(np.sum(a * np.log(a / q)) + np.sum(b * np.log(b / q))) / 2.0


def entropy_mean(probs):
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-(p * np.log(p)).sum(axis=1).mean())


def energy_score(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / float(temperature)
    top = z.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(z - top).sum(axis=1))
    return float((-float(temperature) * lse).mean())


def mmd_rbf(train, val):
    """有偏 RBF-MMD；核宽取 train 两两距离中位数。"""
    a, b = _subsample(_matrix("train", train)), _subsample(_matrix("val", val), seed=1)
    z = np.vstack([a, b])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
    pos = d2[d2 > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    gamma = 1.0 / max(med, EPS)
    kaa = np.exp(-gamma * ((a[:, None, :] - a[None, :, :]) ** 2).sum(axis=2))
    kbb = np.exp(-gamma * ((b[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    kab = np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(max(kaa.mean() + kbb.mean() - 2.0 * kab.mean(), 0.0))


def c2st_lda(train, val):
    """LDA 判别方向上的平衡准确率（0.5 = 不可分）。"""
    a, b = _subsample(_matrix("train", train)), _subsample(_matrix("val", val), seed=1)
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    sw = ((a.shape[0] - 1) * _cov(a) + (b.shape[0] - 1) * _cov(b)) / max(a.shape[0] + b.shape[0] - 2, 1)
    w = np.linalg.pinv(sw) @ (mb - ma)
    cut = float((ma + mb) @ w) / 2.0
    ta = a @ w >= cut
    tb = b @ w < cut
    return float(0.5 * (ta.mean() + tb.mean()))


def _kmeans2(x, iters=25, seed=0):
    rng = np.random.RandomState(seed)
    if x.shape[0] < 2:
        return x.copy(), x.copy()
    c = x[rng.choice(x.shape[0], 2, replace=False)].copy()
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for k in (0, 1):
            if np.any(lab == k):
                c[k] = x[lab == k].mean(axis=0)
    return c, lab


def _sep_1d(a, b):
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    w = b.mean(axis=0) - a.mean(axis=0)
    n = float(np.linalg.norm(w))
    if n < EPS:
        return 0.0
    w = w / n
    pa, pb = a @ w, b @ w
    return float(abs(pa.mean() - pb.mean()) / (pa.std() + pb.std() + 1e-9))


def _cluster_basics(x, y):
    classes = [c for c in np.unique(y) if int((y == c).sum()) >= 2]
    stats = {}
    for c in classes:
        z = x[y == c]
        stats[c] = (z.mean(axis=0), _cov(z), _kmeans2(z)[1])
    return classes, stats


def cluster_separability(train, val, labels_train=None, labels_val=None):
    """sw/(sw+sb) 的批值；无标签时 unavailable。"""
    if labels_train is None or labels_val is None:
        return None
    xt = _matrix("val", val)
    yt = np.asarray(labels_val)
    classes, stats = _cluster_basics(xt, yt)
    if len(classes) < 2:
        return None
    means = np.vstack([stats[c][0] for c in classes])
    within = np.mean([np.trace(stats[c][1]) for c in classes])
    between = float(np.trace(np.cov(means, rowvar=False))) if len(classes) > 2 else float(
        np.sum((means - means.mean(axis=0)) ** 2))
    return float(within / max(within + between, EPS))


def cluster_mean_shift(train, val, labels_train=None, labels_val=None):
    if labels_train is None or labels_val is None:
        return None
    xt, yt = _matrix("train", train), np.asarray(labels_train)
    xv, yv = _matrix("val", val), np.asarray(labels_val)
    _, src = _cluster_basics(xt, yt)
    _, tgt = _cluster_basics(xv, yv)
    shared = sorted(set(src) & set(tgt))
    if not shared:
        return None
    d = xt.shape[1]
    vals = []
    for c in shared:
        inv = np.linalg.pinv(src[c][1])
        diff = tgt[c][0] - src[c][0]
        vals.append(float(np.sqrt(max(diff @ inv @ diff, 0.0)) / math.sqrt(d)))
    return float(np.mean(vals))


def cluster_cov_shift(train, val, labels_train=None, labels_val=None):
    if labels_train is None or labels_val is None:
        return None
    xt, yt = _matrix("train", train), np.asarray(labels_train)
    xv, yv = _matrix("val", val), np.asarray(labels_val)
    _, src = _cluster_basics(xt, yt)
    _, tgt = _cluster_basics(xv, yv)
    shared = sorted(set(src) & set(tgt))
    if not shared:
        return None
    return float(np.mean([abs(np.trace(tgt[c][1]) - np.trace(src[c][1]))
                          / max(np.trace(src[c][1]), EPS) for c in shared]))


def cluster_merge(train, val, labels_train=None, labels_val=None):
    if labels_train is None or labels_val is None:
        return None
    xt, yt = _matrix("train", train), np.asarray(labels_train)
    xv, yv = _matrix("val", val), np.asarray(labels_val)
    _, src = _cluster_basics(xt, yt)
    _, tgt = _cluster_basics(xv, yv)
    shared = sorted(set(src) & set(tgt))
    if len(shared) < 2:
        return None
    vals = []
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = shared[i], shared[j]
            s0 = _sep_1d(xt[yt == a], xt[yt == b])
            s1 = _sep_1d(xv[yv == a], xv[yv == b])
            if s0 > EPS:
                vals.append(max(0.0, 1.0 - s1 / s0))
    return float(np.mean(vals)) if vals else None


def cluster_split(train, val, labels_train=None, labels_val=None):
    if labels_train is None or labels_val is None:
        return None
    xt, yt = _matrix("train", train), np.asarray(labels_train)
    xv, yv = _matrix("val", val), np.asarray(labels_val)
    shared = sorted(set(np.unique(yt)) & set(np.unique(yv)))
    vals = []
    for c in shared:
        a, b = xt[yt == c], xv[yv == c]
        if a.shape[0] < 4 or b.shape[0] < 4:
            continue
        _, la = _kmeans2(a)
        _, lb = _kmeans2(b)
        if len(set(la.tolist())) < 2 or len(set(lb.tolist())) < 2:
            continue
        vals.append(max(0.0, _sep_1d(b[lb == 0], b[lb == 1]) - _sep_1d(a[la == 0], a[la == 1])))
    return float(np.mean(vals)) if vals else None


def mahalanobis_ps(train, val):
    a, b = _matrix("train", train), _matrix("val", val)
    inv = np.linalg.pinv(_cov(a))
    diff = b - a.mean(axis=0)
    return float(np.maximum(np.einsum("ij,jk,ik->i", diff, inv, diff), 0.0).mean())


def prototype_ps(train, val):
    a, b = _matrix("train", train), _matrix("val", val)
    centers, _ = _kmeans2(a)
    d2 = ((b[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).mean())


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > EPS else 0.0


def unit_monotone(train, val, meta_train=None, meta_val=None):
    """逐单元 Spearman(快照序号, 健康指标) 的均值；需 unit/snapshot。"""
    if not meta_train or not meta_val:
        return None
    if meta_train.get("unit") is None or meta_train.get("snapshot") is None:
        return None
    if meta_val.get("unit") is None or meta_val.get("snapshot") is None:
        return None
    xt = _matrix("train", train)
    direction = np.linalg.svd(xt - xt.mean(axis=0), full_matrices=False)[2][0]
    units = np.asarray(meta_val.get("unit"))
    snaps = np.asarray(meta_val.get("snapshot"), dtype=float)
    health = _matrix("val", val) @ direction
    vals = []
    for u in np.unique(units):
        m = units == u
        if m.sum() >= 3 and len(np.unique(snaps[m])) >= 3:
            vals.append(_spearman(snaps[m], health[m]))
    return float(np.mean(vals)) if vals else None


def trend_residual_drift(train, val, meta_train=None, meta_val=None):
    """源趋势残差的分布距离；需 unit/snapshot。"""
    if not meta_train or not meta_val:
        return None
    if meta_train.get("snapshot") is None or meta_val.get("snapshot") is None:
        return None
    xt, xv = _matrix("train", train), _matrix("val", val)
    st = np.asarray(meta_train.get("snapshot"), dtype=float)
    sv = np.asarray(meta_val.get("snapshot"), dtype=float)
    direction = np.linalg.svd(xt - xt.mean(axis=0), full_matrices=False)[2][0]
    ht, hv = xt @ direction, xv @ direction
    coef = np.polyfit(st, ht, 1) if len(np.unique(st)) >= 2 else np.asarray([0.0, float(ht.mean())])
    rt = (ht - np.polyval(coef, st)).reshape(-1, 1)
    rv = (hv - np.polyval(coef, sv)).reshape(-1, 1)
    return hist_kl(rt, rv)


def cond_feature_drift(train, val, meta_train=None, meta_val=None):
    """按工况键分组的标准化均值差，取最差键。"""
    if not meta_train or not meta_val:
        return None
    if meta_train.get("cond_key") is None or meta_val.get("cond_key") is None:
        return None
    kt = np.asarray(meta_train.get("cond_key"))
    kv = np.asarray(meta_val.get("cond_key"))
    xt, xv = _matrix("train", train), _matrix("val", val)
    sd = xt.std(axis=0) + 1e-9
    vals = []
    for k in np.unique(kv):
        if k in set(kt.tolist()):
            vals.append(float(np.max(np.abs((xv[kv == k].mean(axis=0) - xt[kt == k].mean(axis=0)) / sd))))
    return float(np.max(vals)) if vals else None


def cond_key_coverage(train, val, meta_train=None, meta_val=None):
    if not meta_train or not meta_val:
        return None
    if meta_train.get("cond_key") is None or meta_val.get("cond_key") is None:
        return None
    kt = set(np.asarray(meta_train.get("cond_key")).tolist())
    kv = np.asarray(meta_val.get("cond_key")).tolist()
    return float(sum(k in kt for k in kv) / max(len(kv), 1))

"""S2.1 AutoHPO-Lite：CPU 轻量自动超参数优化。

设计约束（任务书）：
- 最多 8 组候选：2 隐藏层 × 2 学习率 × 2 正则化，一轮搜索、不二次调参；
- 只在开发数据上搜索，时间隔离验证——train 段全部早于 val 段，
  禁止随机打乱造成未来泄漏；
- 每次候选拟合计为一次训练消耗，如实记录 CPU/墙钟成本，
  不把多次试训伪装成一次训练；
- 与正式数据流解耦：在线管线仍用验证过的单一配置训练，
  未经官方规则确认不做在线多候选试训。
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .model import BaselineRegressor

# 搜索空间：2×2×2 = 8 组（任务书上限，不扩展）
DEFAULT_GRID = {
    "hidden_layer_sizes": [(32, 16), (64, 32)],
    "learning_rate_init": [0.001, 0.01],
    "alpha": [0.0001, 0.001],
}


def candidate_grid(grid=DEFAULT_GRID) -> list[dict]:
    """展开候选组合（上限 8 组，超出直接报错——防范围漂移）。"""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    if len(combos) > 8:
        raise ValueError("候选组合超过任务书上限 8 组")
    return combos


def time_split(X, y, val_ratio=0.3):
    """时间隔离切分：train = 前 (1-val_ratio)，val = 尾部 val_ratio。

    保证 max(train_index) < min(val_index)——模型选择的验证集
    永远在训练数据之后，与正式流的协议一致。
    """
    n_val = max(30, int(len(y) * val_ratio))
    cut = len(y) - n_val
    return X[:cut], y[:cut], X[cut:], y[cut:]


def evaluate_candidates(streams, model_cls=BaselineRegressor,
                        grid=DEFAULT_GRID, n_hpo=1200, val_ratio=0.3):
    """在开发流上评估全部候选。

    streams: [(name, X, y)]——只取每流前 n_hpo 个样本做 HPO，
    避免用到流后段（相当于只用"开发初期"数据选参）。
    返回 (ranking, rows)：ranking 按跨场景均值验证 RMSE 升序，
    rows 是逐 run 明细（候选×流），含 CPU/墙钟成本。
    """
    cands = candidate_grid(grid)
    rows = []
    for ci, params in enumerate(cands):
        for name, X, y in streams:
            Xd, yd = X[:n_hpo], y[:n_hpo]
            Xtr, ytr, Xva, yva = time_split(Xd, yd, val_ratio)
            t0, c0 = time.perf_counter(), time.process_time()
            m = model_cls(**params).fit(Xtr, ytr)
            wall = time.perf_counter() - t0
            cpu = time.process_time() - c0
            rmse = float(m.evaluate(Xva, yva)["rmse"])
            rows.append({"candidate": ci, "stream": name,
                         "val_rmse": rmse, "n_train": len(ytr),
                         "n_val": len(yva), "fit_wall_s": round(wall, 4),
                         "fit_cpu_s": round(cpu, 4),
                         "n_train_runs": 1,   # 每次拟合计一次训练
                         **{f"p_{k}": str(v) for k, v in params.items()}})
    # 按候选聚合：跨流均值验证 RMSE 升序排名
    agg = {}
    for ci, params in enumerate(cands):
        g = [r for r in rows if r["candidate"] == ci]
        agg[ci] = {"candidate": ci,
                   "mean_val_rmse": float(np.mean([r["val_rmse"] for r in g])),
                   "std_val_rmse": float(np.std([r["val_rmse"] for r in g])),
                   "total_cpu_s": round(sum(r["fit_cpu_s"] for r in g), 3),
                   "total_wall_s": round(sum(r["fit_wall_s"] for r in g), 3),
                   "train_runs": len(g),
                   **params}
    ranking = sorted(agg.values(), key=lambda d: d["mean_val_rmse"])
    for i, d in enumerate(ranking):
        d["rank"] = i + 1
    return ranking, rows

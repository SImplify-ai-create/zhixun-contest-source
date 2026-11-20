"""数据流生成与加载。

校赛阶段官方数据未开放（预计 11 月初随评测平台发布），
先用合成漂移数据自验证闭环；官方数据开放后实现 OfficialStreamLoader 即可无缝切换。
"""
from __future__ import annotations

import numpy as np


class SyntheticDriftStream:
    """合成漂移数据流：同时在输入分布（协变量漂移）和输入-输出关系（概念漂移）上注入漂移。

    y = X @ w + noise，在漂移点处：
    - 部分特征的均值发生平移（PSI 可检出）
    - 真实系数 w 改变（残差恶化可检出）
    """

    def __init__(self, n_samples=6000, n_features=10, drift_points=(2000, 4000),
                 drift_type="sudden", mean_shift=2.0, noise=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.drift_points = list(drift_points)
        bounds = [0, *drift_points, n_samples]

        X = rng.normal(0, 1, size=(n_samples, n_features))
        coefs = [rng.normal(0, 1, n_features) for _ in range(len(bounds) - 1)]
        y = np.zeros(n_samples)

        for seg in range(len(coefs)):
            lo, hi = bounds[seg], bounds[seg + 1]
            if seg > 0:
                # 协变量漂移：前 3 个特征均值平移
                X[lo:hi, :3] += mean_shift * seg
                if drift_type == "gradual":
                    t = np.linspace(0, 1, hi - lo)[:, None]
                    w = coefs[seg - 1] * (1 - t) + coefs[seg] * t
                    y[lo:hi] = np.sum(X[lo:hi] * w, axis=1)
                    continue
            y[lo:hi] = X[lo:hi] @ coefs[seg]

        self.X = X
        self.y = y + rng.normal(0, noise, n_samples)

    def batches(self, batch_size=200):
        for i in range(0, len(self.y), batch_size):
            yield i, self.X[i:i + batch_size], self.y[i:i + batch_size]


class TieredDriftStream:
    """多档漂移生成器：用于验证创新点 2 的严重度分档是否真的可分。

    三个场景只改"什么变了"，接口与 SyntheticDriftStream 完全一致：
      - domain ：只换 X 的边际分布（P(X) 变、P(Y|X) 不变）→ 期望 observe / incremental
      - shallow：P(Y|X) 轻微改（系数在原有 w 附近小扰动）→ 期望 incremental
      - deep   ：P(Y|X) 整体换（每个漂移点换成全新随机系数）→ 期望 full

    这样可以把"域偏移 / 浅概念漂移 / 深概念漂移"三类漂移各自跑一遍，
    统计 classify() 的档位命中分布，再决定阈值——而不是拿单一突变流反复调参数。
    """

    def __init__(self, scenario="domain", n_samples=6000, n_features=10,
                 drift_points=(2000, 4000), mean_shift=1.0, coef_perturb=0.08,
                 noise=0.1, seed=42):
        if scenario not in ("domain", "shallow", "deep"):
            raise ValueError(f"未知场景：{scenario}（可选 domain / shallow / deep）")
        rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.n_features = n_features
        self.drift_points = list(drift_points)
        bounds = [0, *drift_points, n_samples]

        X = rng.normal(0, 1, size=(n_samples, n_features))
        w0 = rng.normal(0, 1, n_features)
        y = np.zeros(n_samples)
        w_segs = []

        for seg in range(len(bounds) - 1):
            lo, hi = bounds[seg], bounds[seg + 1]
            if seg == 0:
                w_seg = w0
            elif scenario == "domain":
                # 只动 X：前 3 个特征按段数累加均值平移，w 全程不变
                X[lo:hi, :3] += mean_shift * seg
                w_seg = w0
            elif scenario == "shallow":
                # 只动映射：w 在 w0 附近小幅扰动（不换方向，量级小）
                w_seg = w0 + coef_perturb * rng.normal(0, 1, n_features)
            else:  # deep：映射整体更换
                w_seg = rng.normal(0, 1, n_features)
            w_segs.append(w_seg)
            y[lo:hi] = X[lo:hi] @ w_seg

        # 记录生成过程的真实参数，供 sample_segment() 独立复采样
        self.w_segs = w_segs
        self.bounds = bounds
        self.mean_shift = mean_shift
        self.coef_perturb = coef_perturb
        self.noise = noise
        self.X = X
        self.y = y + rng.normal(0, noise, n_samples)
        self.name = f"synthetic-tier-{scenario}"

    def batches(self, batch_size=200):
        for i in range(0, len(self.y), batch_size):
            yield i, self.X[i:i + batch_size], self.y[i:i + batch_size]

    def sample_segment(self, seg: int, n: int, seed: int):
        """从同一生成过程独立采样第 seg 段的数据（用于构造"独立金标准"）。

        采样噪声走另一条随机源，但沿用该段的真实系数与 X 偏移，
        因此与主数据流**同分布、不同样本**——这就是"数据独立"的可操作定义。
        """
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, size=(n, self.n_features))
        if self.scenario == "domain" and seg > 0:
            X[:, :3] += self.mean_shift * seg
        y = X @ self.w_segs[seg]
        return X, y + rng.normal(0, self.noise, n)


class OfficialStreamLoader:
    """官方数据包加载器（占位）。

    官方公开集（训练集/验证集）预计 11 月初随评测平台开放。
    实现时保持与 SyntheticDriftStream 相同的接口：提供 batches(batch_size) 迭代器，
    管线代码无需任何改动即可切换。
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        raise NotImplementedError("官方数据包预计 11 月初开放，届时在此实现加载逻辑")

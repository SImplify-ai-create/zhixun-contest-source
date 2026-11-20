"""漂移监测与归因模块（创新点 1 + 创新点 2 前置）。

创新点 1：解耦式双路漂移监测
  - 不用工作模型的错误率做监控（会被模型自我适应欺骗 → 盲点悖论，arXiv:2610.05853）。
  - 用【冻结参考编码器】（一次训练、永不更新）把原始特征映射到稳定表征空间，双路判据：
      结构路：RBF 核 MMD²（表征的【联合分布】是否整体变化）
      边际路：KS 检验（单变量边际是否变化，Bonferroni 校正）
  - 双路联合判据：两路同时报警才确认漂移，抑制单路噪声误报。

  为什么结构路不用 CKA：CKA 衡量的是"同一批样本的两种表征"之间的对齐度
  （Kornblith 等 2019），要求两组表征的样本行一一对应。本模块要做的是
  "参考批 vs 当前批"的【两样本】分布比较，两批样本互不对应，CKA 的分子
  （交叉协方差）在独立样本下期望为 0——实测同分布数据自比也只得 0.049，
  结构路因此恒报警、双路判据退化成单路 KS。两样本场景必须用两样本检验，
  故选 MMD²（Gretton 等 2012）；其阈值由初始参考数据自行校准（无标签、免拍脑袋）。

创新点 2 前置：漂移归因
  - 检测到漂移后，区分"域偏移（P(X) 变，P(Y|X) 不变）"与"概念漂移（P(Y|X) 变）"。
  - 依据：若输入分布（MMD/PSI）显著漂移但模型残差未恶化 → 域偏移 → 增量微调即可；
          若模型残差显著恶化 → 概念漂移 → 需要全量重训练。

数学要点（答辩可讲）：
  MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]，k 为 RBF 核；
  无偏估计在 P=Q 时期望为 0，用参考数据内部对半切分的零分布定阈值（3σ 线），
  使报警阈值随数据尺度自适应，不依赖固定常数。
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


# ------------------------- 统计量 -------------------------

def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index（单变量分布偏移）。"""
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_ref = np.clip(np.histogram(reference, bins=bins)[0].astype(float), 1e-6, None)
    p_cur = np.clip(np.histogram(current, bins=bins)[0].astype(float), 1e-6, None)
    p_ref /= p_ref.sum()
    p_cur /= p_cur.sum()
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def cka(reference: np.ndarray, current: np.ndarray) -> float:
    """中心化核对齐（Centered Kernel Alignment）。

    ⚠️ 仅适用于【同一批样本】的两种表征（第 i 行必须是同一个样本），
    不能用来比较两批独立样本的分布——独立样本的交叉协方差期望为 0，
    结果会恒接近 0。两样本分布比较请用 mmd2_rbf()。
    """
    n_ref, n_cur = len(reference), len(current)
    if n_ref != n_cur:
        n = min(n_ref, n_cur)
        rng = np.random.RandomState(42)
        if n_ref > n:
            reference = reference[rng.choice(n_ref, n, replace=False)]
        else:
            current = current[rng.choice(n_cur, n, replace=False)]

    F_r = reference - reference.mean(axis=0, keepdims=True)
    F_c = current - current.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(F_r.T @ F_c, ord="fro") ** 2
    norm_r = np.linalg.norm(F_r.T @ F_r, ord="fro")
    norm_c = np.linalg.norm(F_c.T @ F_c, ord="fro")
    denom = norm_r * norm_c
    if denom < 1e-12:
        return 1.0
    return float(cross / denom)


def mmd2_rbf(reference: np.ndarray, current: np.ndarray,
             gamma: float | None = None) -> dict:
    """RBF 核 MMD²（无偏估计），用于两批样本的分布比较。

    gamma 给定时沿用同一核尺度（参考分布上标定的带宽），保证不同批次可比。
    未给定时用中位数启发式（对合并样本两两距离的中位数取 1/median）。
    """
    if gamma is None:
        z = np.vstack([reference, current])
        if len(z) > 400:                       # 只为核心估计采样，避免 O(n²) 过大
            idx = np.random.RandomState(42).choice(len(z), 400, replace=False)
            z = z[idx]
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)

    def sq_dists(a, b):
        return (np.sum(a * a, 1)[:, None] + np.sum(b * b, 1)[None, :]
                - 2.0 * a @ b.T)

    k_rr = np.exp(-gamma * np.maximum(sq_dists(reference, reference), 0))
    k_cc = np.exp(-gamma * np.maximum(sq_dists(current, current), 0))
    k_rc = np.exp(-gamma * np.maximum(sq_dists(reference, current), 0))
    m, n = len(reference), len(current)
    # 无偏估计：去掉对角项
    np.fill_diagonal(k_rr, 0.0)
    np.fill_diagonal(k_cc, 0.0)
    val = (k_rr.sum() / (m * (m - 1)) + k_cc.sum() / (n * (n - 1))
           - 2.0 * k_rc.mean())
    return {"mmd2": float(val), "gamma": float(gamma)}


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """两样本 Kolmogorov-Smirnov 检验，衡量边际分布差异。

    多维时对每个维度各做一次 KS，取最大统计量；p 值做 Bonferroni 校正
    （min p * 维度数）。未校正时 8 维取最小值会把零假设下的误报率抬到
    1-(1-0.05)^8≈34%，实测空流 23%~33%，这正是"单路"误报偏高的来源。
    """
    if reference.ndim == 1:
        ref, cur = reference, current
        stat, p = ks_2samp(ref, cur)
        return {"statistic": float(stat), "p_value": float(p),
                "p_value_raw": float(p)}
    stats, pvals = [], []
    for j in range(reference.shape[1]):
        s, p = ks_2samp(reference[:, j], current[:, j])
        stats.append(s)
        pvals.append(p)
    p_raw = float(np.min(pvals))
    p_adj = float(min(1.0, p_raw * reference.shape[1]))
    return {"statistic": float(np.max(stats)), "p_value": p_adj,
            "p_value_raw": p_raw}


# ------------------------- 冻结参考编码器 -------------------------

class FrozenReferenceEncoder:
    """一次训练、永不更新的特征编码器（创新点 1 核心）。

    用初始稳定段数据训练 StandardScaler + PCA，此后永远冻结，
    提供一个不随工作模型更新而漂移的稳定表征空间。
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        n = min(self.n_components, Xs.shape[1], Xs.shape[0] - 1)
        self.pca = PCA(n_components=n, random_state=42)
        self.pca.fit(Xs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("先调用 fit 训练编码器")
        return self.pca.transform(self.scaler.transform(X))


# ------------------------- 双路漂移检测器 -------------------------

class DriftDetector:
    """解耦式双路漂移检测器 + 漂移归因。

    检测通道（创新点 1）：
      - 结构路：表征空间 MMD² 的 z 分数（相对参考数据的零分布）> mmd_z_threshold
      - 边际路：KS 检验（Bonferroni 校正后）p_value < ks_alpha
      - 双路同时报警 → drift_detected=True（抑制单路噪声误报）
    传统 PSI 保留作为参考指标输出。

    阈值不再用固定常数：MMD 的零分布由参考数据内部对半切分标定（calib_splits 次），
    报警线 = 零分布均值 + mmd_z_threshold 倍标准差，随数据尺度自适应。

    归因通道（创新点 2 前置）：
      classify(...) 把漂移严重度当作连续谱，输出 observe / incremental / full 三档。
    """

    def __init__(self, n_bins=10, mmd_z_threshold=3.0, ks_alpha=0.05,
                 psi_drift_threshold=0.25, min_drift_features=2,
                 calib_splits=32, calib_size=200, tier_bounds=(1.0, 3.0),
                 psi_top_k=0, ks_effect_min=0.1):
        self.n_bins = n_bins
        # 零分布按典型批大小标定：MMD² 估计量的方差随样本数变化，
        # 若用 300 vs 300 标定、却用 200 vs 200 打分，同一分布也会被误判
        self.calib_size = calib_size
        self.mmd_z_threshold = mmd_z_threshold
        self.ks_alpha = ks_alpha
        # KS 效应量门控：p 值是显著性判据，n 大时任何微小差异都显著（空流常报）。
        # D 统计量本身是效应量（0~1），报警须同时满足 p<alpha 且 D>=ks_effect_min。
        # 200 条批上真实漂移 D~0.3+，噪声 D~0.05-0.09，0.1 对现有结果无影响；
        # 现场批大小涨到几千时，这道门是检测器不被显著性爆炸淹没的关键。
        self.ks_effect_min = ks_effect_min
        self.psi_drift_threshold = psi_drift_threshold
        self.min_drift_features = min_drift_features
        self.calib_splits = calib_splits
        self.tier_low, self.tier_high = tier_bounds
        # 备用选项（默认关）：psi_top_k>0 时严重度改用 top-k 高敏感特征的平均 PSI。
        # 10 维合成特征上实测无收益（top3 与等权结果几乎相同），仅当官方预提取
        # 特征为高维（几十维以上）时再开启。
        self.psi_top_k = psi_top_k
        self.encoder: FrozenReferenceEncoder | None = None
        self.repr_ref: np.ndarray | None = None   # 参考分布在冻结表征空间的样本
        self.raw_ref: np.ndarray | None = None     # 原始参考数据（用于 PSI）
        self.mmd_gamma: float | None = None        # 参考数据上标定的 RBF 带宽
        self.mmd_null_mean = 0.0
        self.mmd_null_std = 1.0

    def fit_reference(self, X: np.ndarray):
        """用初始稳定段数据建立参考分布 + 训练冻结编码器 + 标定 MMD 零分布。"""
        self.encoder = FrozenReferenceEncoder().fit(X)
        self.repr_ref = self.encoder.transform(X)
        self.raw_ref = X.copy()
        self._calibrate_mmd()

    def _calibrate_mmd(self):
        """用参考数据内部对半切分估计 MMD² 的零分布（无标签、免拍阈值）。"""
        z = self.repr_ref
        if len(z) > 400:
            idx = np.random.RandomState(42).choice(len(z), 400, replace=False)
            z = z[idx]
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        self.mmd_gamma = 1.0 / max(med, 1e-12)

        n = len(self.repr_ref)
        half = min(n // 2, self.calib_size)
        rs = np.random.RandomState(0)
        vals = []
        for _ in range(self.calib_splits):
            perm = rs.permutation(n)
            a, b = perm[:half], perm[half:2 * half]
            if len(b) == 0:
                continue
            vals.append(mmd2_rbf(self.repr_ref[a], self.repr_ref[b],
                                 gamma=self.mmd_gamma)["mmd2"])
        self.mmd_null_mean = float(np.mean(vals))
        self.mmd_null_std = float(max(np.std(vals), 1e-9))

    def score(self, X: np.ndarray) -> dict:
        """双路检测 + 传统 PSI 参考。"""
        if self.encoder is None:
            raise RuntimeError("先调用 fit_reference 建立参考分布")

        repr_cur = self.encoder.transform(X)

        # 结构路：MMD²。参考侧下采样到与当前批同规模，保持与零分布相近的方差
        m = len(repr_cur)
        if len(self.repr_ref) >= m:
            idx = np.random.RandomState(0).choice(len(self.repr_ref), m,
                                                  replace=False)
            ref_sub = self.repr_ref[idx]
        else:
            ref_sub = self.repr_ref
        mmd_res = mmd2_rbf(ref_sub, repr_cur, gamma=self.mmd_gamma)
        mmd_z = (mmd_res["mmd2"] - self.mmd_null_mean) / self.mmd_null_std
        structural_alarm = mmd_z > self.mmd_z_threshold

        # 边际路：KS 检验（Bonferroni 校正）
        ks_res = ks_test(self.repr_ref, repr_cur)
        marginal_alarm = (ks_res["p_value"] < self.ks_alpha
                          and ks_res["statistic"] >= self.ks_effect_min)

        # 传统 PSI（参考指标，不参与触发）
        per_feature = [
            {"feature": j,
             "psi": psi(self.raw_ref[:, j], X[:, j], self.n_bins)}
            for j in range(X.shape[1])
        ]
        mean_psi = float(np.mean([f["psi"] for f in per_feature]))
        n_drifted_psi = sum(1 for f in per_feature
                            if f["psi"] > self.psi_drift_threshold)

        dual_alarm = structural_alarm and marginal_alarm

        return {
            "mmd2": mmd_res["mmd2"],
            "mmd_z": float(mmd_z),
            "ks_statistic": ks_res["statistic"],
            "ks_p_value": ks_res["p_value"],
            "structural_alarm": structural_alarm,
            "marginal_alarm": marginal_alarm,
            "drift_detected": dual_alarm,   # 双路联合判据
            "mean_psi": mean_psi,
            "n_drifted_features": n_drifted_psi,
            "top_drifted": sorted(per_feature, key=lambda d: -d["psi"])[:3],
        }

    def classify(self, score: dict, rmse: float, baseline_rmse: float,
                 residual_factor: float = 3.0) -> dict:
        """漂移归因：输出严重度与动作档位（取代已退化的二分类归因）。

        失效背景：旧 attribute_drift 用"残差倍数 > 2.0"先判概念漂移，而合成流
        突变点后残差倍数普遍 >2，判据恒命中 → 恒返回 "concept"。实测紧/松预算下
        "含归因"与"去归因"两组数值完全相同，创新点 2 零区分度。

        严重度只由两项构成（1.0 = 刚好压线）：
            sev = max( (rmse/baseline) / residual_factor,        # 残差线 = factor 倍基线
                       mean_psi / psi_drift_threshold )          # 边际线 = PSI 阈值
        为什么不用 MMD 的 z 当严重度：z 是检验统计量，随样本量线性放大——实测纯域
        偏移(mean_shift=1) 的 mmd_z≈67，任何输入偏移都会顶到上限，失去分档能力。
        PSI 恰恰是为"稳定性分带"设计的（Siddiqi 2006: <0.10 / 0.10~0.25 / >0.25），
        所以 MMD/KS 只负责【检测】（报警），PSI 负责【严重度】（分档）。这是
        检测统计量与效应量的分工，不是两套判据互相打架。
        残差项必须除以 residual_factor，否则量纲随数据漂移——未归一化时残差倍数
        直接是 20~90，任何阈值都恒判 full（另一个方向的退化）。

        输入侧证据封顶 INPUT_CAP=2.6（低于 full 的 3.0 线）：
        输入分布偏移只说明 P(X) 变了，P(Y|X) 是否变只有残差能回答；
        若允许输入证据独自判 full，纯域偏移（w 不变）就会被当成"旧标签全废"。
        故 full 只能由残差恶化触发，输入证据最多给到 incremental。

        档位映射（阈值 1.0 / 3.0）：
            sev < 1.0         → observe      两路都没过线（报警已回落）→ 不重训
            1.0 <= sev < 3.0  → incremental  轻度退化/仅输入偏移 → 增量微调
            sev >= 3.0        → full         残差严重恶化（>9 倍基线）→ 清洗后全量
        1.0 作为下界依据："至少一路刚压线"才值得动模型；实测把下界设在 1.5
        会让浅概念漂移（残差 3~4 倍基线）落进 observe，末段 0.328 vs 微调 0.191。
        边界不写死：tier_bounds 可被离线标定反解（公开集上按"精度 + 标注成本"
        双目标搜索），思路迁移自装调技术的逆向优化法。

        返回 {"tier", "severity", "residual_ratio", "mmd_term", "psi_term"}。
        """
        baseline = max(baseline_rmse, 1e-6)
        residual_ratio = rmse / baseline
        residual_term = residual_ratio / max(residual_factor, 1e-6)
        input_cap = 2.6
        # 备用开关：psi_top_k>0 时用 top-k 敏感特征的 PSI 均值（高维特征场景）
        if self.psi_top_k and score.get("top_drifted"):
            top = [d["psi"] for d in score["top_drifted"][:self.psi_top_k]]
            psi_val = float(np.mean(top))
        else:
            psi_val = score["mean_psi"]
        psi_term = min(psi_val / max(self.psi_drift_threshold, 1e-6),
                       input_cap)
        severity = float(max(residual_term, psi_term))

        if severity < self.tier_low:
            tier = "observe"
        elif severity < self.tier_high:
            tier = "incremental"
        else:
            tier = "full"
        return {"tier": tier, "severity": severity,
                "residual_ratio": float(residual_ratio),
                "mmd_z": float(score["mmd_z"]),
                "mmd_term": float(score["mmd_z"] / max(self.mmd_z_threshold, 1e-6)),
                "psi_term": float(psi_term)}

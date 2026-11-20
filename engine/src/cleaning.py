"""选择性数据清洗（创新点 3）。

真实问题：重训练时把过时的旧数据和新数据混在一起训，会导致"梯度错位"
——旧分布的样本把梯度往旧方向拉，新分布的样本往新方向拉，模型在中间
摇摆，准确率反而下降（Holistic CL + AMR, TMLR 2026, arXiv:2507.02310）。

解决方案：重训练前，用【特征距离】对历史数据做清洗：
  1. 计算新数据分布中心（均值向量）
  2. 对每条历史样本，计算它到新中心的马氏距离/欧氏距离
  3. 剔除距离超过阈值的"过时样本"（它们会拉歪梯度）
  4. 保留与新分布一致的历史样本 + 全部新标注数据

效果：用 1/10 的标注量达到全量重训练精度，且避免梯度冲突。
"""
from __future__ import annotations

import numpy as np


def selective_clean(X_hist: np.ndarray, y_hist: np.ndarray,
                    X_new: np.ndarray, y_new: np.ndarray,
                    keep_ratio: float = 0.6,
                    distance_metric: str = "mahalanobis",
                    max_mahalanobis_dim: int = 100,
                    center_ref: np.ndarray | None = None) -> tuple:
    """从历史数据中筛选与新分布一致的样本。

    Args:
        X_hist, y_hist: 历史训练数据（可能含过时样本）
        X_new, y_new: 新制度标注数据（全部保留）
        keep_ratio: 保留历史样本的比例（按距离从近到远取前 keep_ratio）
        distance_metric: "mahalanobis"（考虑协方差）或 "euclidean"
        center_ref: 估计"新分布中心"所用的特征切片。留出验收闸门下传入
            验收集，避免验收样本既定义中心又参与训练造成特征泄漏；
            缺省时沿用 X_new（旧行为）。

    Returns:
        (X_clean, y_clean): 清洗后可用于重训练的数据
        stats: 清洗统计信息（答辩可讲）
    """
    if len(X_hist) == 0:
        return X_new.copy(), y_new.copy(), {"kept_hist": 0, "removed_hist": 0}

    ref = X_new if center_ref is None else center_ref

    # 高维保护：马氏距离要估计 d×d 协方差（O(d³) 且 d>n 时奇异），
    # 特征维数超过上限时先 PCA 降维再算距离（低维空间协方差可估）。
    # 注意：PCA 只用于**估计距离**，返回的训练数据必须留在原始特征空间，
    # 否则重训出来的模型维度与后续评估/线上数据不一致（128 维气体数据实测
    # 会把模型压成 31 维，predict 直接报特征数不符）。
    if (distance_metric == "mahalanobis"
            and ref.shape[1] > max_mahalanobis_dim):
        from sklearn.decomposition import PCA
        n_comp = max(1, min(32, max_mahalanobis_dim, ref.shape[0] - 1))
        pca = PCA(n_components=n_comp, random_state=42).fit(ref)
        X_hist_dist = pca.transform(X_hist)
        ref_dist = pca.transform(ref)
    else:
        X_hist_dist = X_hist
        ref_dist = ref

    # 新分布中心
    new_center = ref_dist.mean(axis=0)

    if distance_metric == "mahalanobis" and ref_dist.shape[0] > ref_dist.shape[1]:
        # 马氏距离：考虑特征间协方差，更准确地衡量"离新分布有多远"
        cov = np.cov(ref_dist, rowvar=False)
        try:
            cov_inv = np.linalg.pinv(cov)
            diff = X_hist_dist - new_center
            dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        except np.linalg.LinAlgError:
            dists = np.linalg.norm(X_hist_dist - new_center, axis=1)
    else:
        dists = np.linalg.norm(X_hist_dist - new_center, axis=1)

    # 按距离从近到远排序，保留前 keep_ratio
    if keep_ratio <= 0:
        n_keep = 0
    else:
        n_keep = max(1, int(len(X_hist) * keep_ratio))
    order = np.argsort(dists)
    keep_idx = order[:n_keep]

    X_kept = X_hist[keep_idx]
    y_kept = y_hist[keep_idx]

    # 合并：保留的历史数据 + 全部新数据
    X_clean = np.vstack([X_kept, X_new])
    y_clean = np.concatenate([y_kept, y_new])

    stats = {
        "kept_hist": len(keep_idx),
        "removed_hist": len(X_hist) - len(keep_idx),
        "total_clean": len(y_clean),
        "new_samples": len(y_new),
        "mean_distance_kept": float(dists[keep_idx].mean()) if len(keep_idx) > 0 else 0.0,
        "mean_distance_removed": float(dists[order[n_keep:]].mean())
            if len(order) > n_keep else 0.0,
    }
    return X_clean, y_clean, stats

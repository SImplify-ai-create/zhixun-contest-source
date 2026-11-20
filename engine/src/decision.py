"""v2.0 改进版：代价敏感动作决策器（备用切换，默认不启用）。

思路迁移自光学装调技术（docs/装调技术1.docx 摘要）：
  - 灵敏度矩阵法：把"证据向量 → 各动作的收益"建成显式映射（在线一次查表，O(k)）；
  - 逆向优化法：分档边界 / 动作选择不再手工设定，而是离线在标定数据上
    按「精度 + 标注成本」目标反解。

与 v1.0（阈值分档）的关系：
  v1.0 回答"漂移有多严重"→ 按严重度映射动作；
  v2.0 回答"做哪个动作划算"→ argmin_a [ 预测RMSE(a) + cost_label * 标注(a) ]。
  标定表缺失或查询点离表太远时，自动回退 v1.0 阈值规则（安全兜底，
  保证评测中任何情况下都有可用决策）。

标定表由 scripts/calibrate_actions.py 生成（官方公开集到手后重跑一次）。
每行记录：决策点证据 (residual_term, psi_term, mmd_term) + 强制动作 + 该动作的
实测结果（末段 RMSE、标注消耗）。在线决策时对每个动作取 k 近邻平均作为
反事实收益预测——即"动作反事实标定"。

已知边界（答辩如实说明）：
  - 标定行的结果标注是"整条流"的消耗，不是单次决策的精确归因（合成 dry-run 的近似）；
  - 代价权重不含剩余槽位/预算的实时状态，槽位感知决策留给 v2.1；
  - 合成数据上的 dry-run 只验证机制可运行，真实收益要等官方公开集标定。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .drift import DriftDetector

FEATURES = ("residual_term", "psi_term", "mmd_term")
ACTIONS = ("observe", "incremental", "full")


class CostSensitiveDetector(DriftDetector):
    """继承 v1.0 检测器，只改"决策"一步：阈值规则 → 查表 + 代价比较。

    用法：
        det = CostSensitiveDetector(calib_path="runs/action_calibration.json")
        # 未加载标定表时行为与 DriftDetector 完全一致
    """

    def __init__(self, *args, calib_path: str | None = None,
                 cost_label: float = 1e-4, knn_k: int = 3,
                 max_query_dist: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_label = cost_label
        self.knn_k = knn_k
        self.max_query_dist = max_query_dist
        self.table: list[dict] | None = None
        self.table_meta: dict = {}
        # 证据项量纲差异大（mmd_term 可达 20+，psi_term 封顶 2.6），
        # 查表前按标定表的均值/标准差做 z 归一，避免距离被单一维度主导
        self.feat_mean = np.zeros(len(FEATURES))
        self.feat_std = np.ones(len(FEATURES))
        if calib_path:
            self.load_calibration(calib_path)

    def load_calibration(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"标定表不存在：{p}\n先运行 scripts/calibrate_actions.py 生成。")
        payload = json.loads(p.read_text(encoding="utf-8"))
        self.table = payload["rows"]
        self.table_meta = {k: payload[k] for k in payload if k != "rows"}
        xs = np.array([[r[f] for f in FEATURES] for r in self.table])
        self.feat_mean = xs.mean(axis=0)
        self.feat_std = np.maximum(xs.std(axis=0), 1e-6)
        return self

    # ---- 内部工具 ----

    def _evidence(self, score: dict, rmse: float, baseline_rmse: float,
                  residual_factor: float) -> dict:
        baseline = max(baseline_rmse, 1e-6)
        residual_ratio = rmse / baseline
        return {
            "residual_term": residual_ratio / max(residual_factor, 1e-6),
            "psi_term": min(score["mean_psi"] / max(self.psi_drift_threshold, 1e-6), 2.6),
            "mmd_term": score["mmd_z"] / max(self.mmd_z_threshold, 1e-6),
        }

    def _knn_predict(self, action: str, x: np.ndarray) -> tuple[float, float] | None:
        """返回该动作下 k 近邻的 (预测末段RMSE, 平均标注消耗)；无该动作样本返回 None。"""
        assert self.table is not None
        rows = [r for r in self.table if r["action"] == action]
        if not rows:
            return None
        xs = np.array([[r[f] for f in FEATURES] for r in rows])
        xs = (xs - self.feat_mean) / self.feat_std
        x = (x - self.feat_mean) / self.feat_std
        d = np.linalg.norm(xs - x, axis=1)
        k = min(self.knn_k, len(rows))
        idx = np.argsort(d)[:k]
        if float(np.mean(d[idx])) > self.max_query_dist:
            return None                     # 离表太远 → 交回阈值规则
        pred = float(np.mean([rows[i]["final_rmse"] for i in idx]))
        labels = float(np.mean([rows[i]["labels"] for i in idx]))
        return pred, labels

    # ---- 决策 ----

    def classify(self, score: dict, rmse: float, baseline_rmse: float,
                 residual_factor: float = 3.0) -> dict:
        base = super().classify(score, rmse, baseline_rmse, residual_factor)
        if self.table is None:
            return dict(base, decision="threshold_fallback")

        ev = self._evidence(score, rmse, baseline_rmse, residual_factor)
        x = np.array([ev[f] for f in FEATURES])
        costs, detail = {}, {}
        for a in ACTIONS:
            r = self._knn_predict(a, x)
            if r is None:
                detail[a] = None
            else:
                pred, labels = r
                costs[a] = pred + self.cost_label * labels
                detail[a] = {"pred_rmse": round(pred, 4),
                             "labels": round(labels, 1),
                             "cost": round(costs[a], 5)}
        if not costs:                       # 三个动作都查不到 → 兜底
            return dict(base, decision="threshold_fallback", evidence=ev)

        action = min(costs, key=costs.get)
        return dict(base, tier=action, decision="cost_sensitive",
                    evidence={k: round(v, 3) for k, v in ev.items()},
                    action_costs=detail)

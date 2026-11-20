"""新机制（创新点 4 + 5）：只改"重训时取多少数据、取哪一段"，其余保持现有管线。

为什么这样实现：第一版把主循环整个重写，结果全量重训一次都没触发（分级动作的判断被我写坏了），
四组配置数值完全相同——说明改动引入了新缺陷而不是新能力。因此改为最小侵入：
沿用 SelfEvolutionPipeline 的检测、冷却、渐变模式、分级重训，
只覆盖取数钩子 _select_batch 一个方法。

失效 A —— 预算饥饿（紧预算 sudden 末段 3.709，seed=2024 高达 7.565）：
    旧逻辑按"窗口有 600 条就要 600 条标注"，第一次重训吃光 600，第二次只剩 300，
    晚到的漂移没钱修。
    → 创新点 4 预算节流。实测"按槽位均分"在紧预算下反而饿死第一次（450 < 600 所需），
      故默认改为压力触发：只有剩余槽位 <=1 时才限额，其余时候放开。

失效 B —— 窗口滞后（渐变末段 1.254 追不上 0.175）：
    不限预算实验：窗口 1/2/3/5 批 → 末段 0.706 / 1.015 / 1.309 / 2.537，窗口长度是主因。
    固定 600（3 批）在漂移期等于拿 30% 过渡期的混合数据去拟合当前目标。
    → 创新点 5 速率自适应窗口：残差变化速率快 → 1 批；中 → 2 批；稳 → 3 批。
"""
from __future__ import annotations

from .pipeline import SelfEvolutionPipeline


class PacedAdaptivePipeline(SelfEvolutionPipeline):
    def __init__(self, *args, batch_size=200, use_budget_pacing=True,
                 use_rate_window=True, pacing_trigger="pressure",
                 # ---- 软门配额（v1.1 简化版）----
                 # 只缩 incremental 档的训练切片配额，full 档永远全量。
                 # 降级为固定比例（旧 sigmoid 连续门难以解释、且与验收集耦合
                 # 出过负例）；quota 只影响训练切片，验收集从完整窗口取，
                 # 不影响升级判定/回滚/静态锚（见 pipeline._accept_split）。
                 soft_quota=False, soft_quota_incremental_ratio=0.6,
                 soft_quota_floor=120, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.use_budget_pacing = use_budget_pacing
        self.use_rate_window = use_rate_window
        self.pacing_trigger = pacing_trigger  # "pressure"（槽位压力）或 "even"（均分）
        self.soft_quota = bool(soft_quota)
        self.soft_quota_incremental_ratio = float(soft_quota_incremental_ratio)
        self.soft_quota_floor = int(soft_quota_floor)

    # ---------------- 创新点 5：自适应窗口 ----------------

    def _window_batches(self, baseline_rmse: float) -> int:
        if not self.use_rate_window:
            return max(1, self.window_size // self.batch_size)
        hist = self._rmse_history
        if len(hist) < 3:
            return 3
        d1 = hist[-1] - hist[-2]
        d2 = hist[-2] - hist[-3]
        # 只有残差连续两步上升才判定"漂移仍在进行中"。
        # 实测：只用最后一步差值会把模型已经收敛后的微小抖动（基线 RMSE≈0.08，
        # 抖动 0.004 就是 5%）误判成快漂移，把突变恢复期的长窗砍成 1 批，
        # 松预算 sudden 末段反而从 0.151 退化到 0.198。
        if d1 <= 0 or d2 <= 0:
            return 3
        rate = max(d1, d2) / max(baseline_rmse, 1e-6)
        if rate > 0.05:
            return 1        # 快漂移：只用最新一批，避免过渡期混合数据
        if rate > 0.01:
            return 2
        return 3            # 漂移已稳定：样本更足

    # ---------------- 创新点 4：预算节流 ----------------

    def _paced_quota(self, want: int) -> int:
        remaining = self.label_budget - self.labels_used
        if not self.use_budget_pacing:
            return max(0, min(want, remaining))
        slots = max(1, self.max_retrains - self.n_retrains)
        if self.pacing_trigger == "pressure" and slots > 1:
            return max(0, min(want, remaining))   # 槽位不紧张 → 不设额外上限
        return max(0, min(want, remaining // slots))

    # ---------------- 覆盖取数钩子 ----------------

    def _select_batch(self, X_new, y_new, baseline_rmse,
                      severity=None, tier=None):
        # 创新点 5：按速率取最近 w 批
        w = self._window_batches(baseline_rmse)
        n_want = min(len(y_new), w * self.batch_size)
        # 创新点 4：按剩余预算/槽位节流
        quota = self._paced_quota(n_want)
        # 软门（v1.1 简化）：只对 incremental 档缩训练配额，full 档永远全量。
        # 固定比例 + 下限，可解释、可消融；验收集从完整窗口取，不受 quota 影响。
        if self.soft_quota and tier == "incremental" and quota > 0:
            quota = max(self.soft_quota_floor,
                        int(quota * self.soft_quota_incremental_ratio))
            quota = min(quota, n_want)  # 不超过想取的量
        start = len(y_new) - quota
        return (X_new[start:], y_new[start:], quota,
                {"window_batches": w, "quota": quota, "tier": tier})

"""S2 阶段 4：证据驱动的条件模型选择（Ridge → MLP 单次切换）。

设计约束（任务书）：
- 只用当时可获得的在线证据（classify 的 tier/severity），不用未来信息；
- 切换是单向的：被判 full/fast_full 时才换 MLP 候选，验收通过即永久
  切换，被拒绝则保持 Ridge——避免来回震荡；
- 切换不额外消耗槽位/标签：候选仍走同一验收闸门与同一账本。
"""
from __future__ import annotations

from .advanced import PacedAdaptivePipeline
from .model import BaselineRegressor, RidgeRegressor


class ConditionalPipeline(PacedAdaptivePipeline):
    """以 Ridge 起步；严重漂移被确认后改用 MLP 候选。

    近似线性且稳定 → Ridge 精度相当且拟合便宜；深度概念漂移 →
    MLP 的非线性容量与全量重训恢复力更优。切换判据只看 classify
    给出的 tier（证据当时已可得），符合"不许用未来信息"约束。
    """

    def __init__(self, *args, switch_cls=BaselineRegressor, **kwargs):
        kwargs["model_cls"] = RidgeRegressor
        super().__init__(*args, **kwargs)
        self._switch_cls = switch_cls
        self._switched = False

    def _retrain_attributed(self, X_new, y_new, ids_new, old_rmse,
                            baseline_rmse, tier, severity=1.0):
        if not self._switched and tier in ("full", "fast_full"):
            self.model_cls = self._switch_cls          # 候选用 MLP
            updated, new_rmse, mode = super()._retrain_attributed(
                X_new, y_new, ids_new, old_rmse, baseline_rmse,
                tier, severity)
            if updated:
                self._switched = True
                self.logger.log({"event": "model_switch",
                                 "to": self._switch_cls.__name__,
                                 "tier": tier})
            else:
                self.model_cls = RidgeRegressor        # 被拒则退回 Ridge
            return updated, new_rmse, mode
        return super()._retrain_attributed(
            X_new, y_new, ids_new, old_rmse, baseline_rmse,
            tier, severity)

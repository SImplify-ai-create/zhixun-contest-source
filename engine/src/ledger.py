"""统一标注预算账本（LabelLedger）。

S1 整改核心：所有模型/所有更新路径共用同一套标签计量规则。

口径定义（写入每条实验记录，供评审核对）：

  acquired  新获取的真实标签条数——用于参数更新（fit/partial_fit/track）或
            验收决策（holdout gate）。样本按流内 offset 编号，同一样本
            重复用于训练不再计费（见 reused）。
  reused    已付费标签被再次投入训练的条数（清洗保留的历史样本、窗口重叠）。
            不产生新费用，但如实记录复用量。
  eval      仅用于评分/监控的标签条数。残差报警、静态基线对照、OOF 估计
            都要先看到 y 才能算 RMSE——按"预测→评分→标签到达"的流式协议
            它们是事后评分标签，默认不计入预算（eval_free=True），
            但条数完整记录；若官方口径改为"评分也计费"，改一个开关即可。
  超支事件   想要获取但因预算耗尽被拒的次数/条数（over_budget_events/labels）。

预算断言：acquired 永远 <= budget，acquired 永远 == len(paid_ids)，
即"账本里的钱"与"实际拿到手的样本"逐条对应，无暗账。
"""
from __future__ import annotations

from collections import Counter


class LabelLedger:
    """样本 ID 级标签账本。

    每条数据以流内 offset 为唯一 ID。acquire_ids 只对新 ID 计费，
    已付费 ID 自动走复用通道，保证"重复标签不重复扣费"。
    """

    def __init__(self, budget: int, eval_free: bool = True):
        self.budget = int(budget)
        # eval_free=True：评分/监控用标签只记录不扣费（默认口径，见模块 docstring）
        self.eval_free = bool(eval_free)
        self.acquired = 0            # 已付费新标签总数
        self.reused = 0              # 已付费标签复用总数
        self.eval_only = 0           # 仅评分标签总数
        self.by_purpose: Counter = Counter()
        self.over_budget_events = 0  # 想要但买不起的次数
        self.over_budget_labels = 0  # 想买但买不起的条数
        self._paid: set[int] = set()  # 已付费样本 ID 全集

    # ------------------------- 查询 -------------------------

    def remaining(self) -> int:
        return max(0, self.budget - self.acquired)

    def is_paid(self, sample_id: int) -> bool:
        return sample_id in self._paid

    # ------------------------- 记账 -------------------------

    def acquire_ids(self, ids, purpose: str) -> set[int]:
        """为一组样本 ID 购买标签。

        已付费 ID 自动计入 reused（不产生新费用）。新 ID 按剩余预算
        尽量购买，不够时优先买**最新**的（对新近分布最有价值）。
        返回实际获准使用的 ID 集合（已付费 ∪ 新购得）。
        """
        ids = {int(i) for i in ids}
        if not ids:
            return set()
        already = ids & self._paid
        new_ids = ids - self._paid
        if already:
            self.reused += len(already)
        grant = self.remaining()
        if len(new_ids) > grant:
            self.over_budget_events += 1
            self.over_budget_labels += len(new_ids) - grant
        # 预算不够时优先保留最新样本（ID 越大越新）
        granted_new = set(sorted(new_ids)[-grant:]) if grant else set()
        self.acquired += len(granted_new)
        self.by_purpose[purpose] += len(granted_new)
        self._paid |= granted_new
        assert self.acquired <= self.budget, "标签预算被击穿"
        assert self.acquired == len(self._paid), "账本与付费样本集不一致"
        return already | granted_new

    def acquire(self, n: int, purpose: str) -> int:
        """无 ID 场景的退化接口（按条数计费）。返回实际获准条数。"""
        n = int(n)
        grant = min(n, self.remaining())
        if grant < n:
            self.over_budget_events += 1
            self.over_budget_labels += n - grant
        self.acquired += grant
        self.by_purpose[purpose] += grant
        assert self.acquired <= self.budget, "标签预算被击穿"
        return grant

    def record_eval(self, ids) -> None:
        """仅评分用标签：默认只记录不扣费；严格口径下同样计入 acquired。

        入参为样本 ID 集合。eval_free=False（严格口径）时对这些 ID
        调用 acquire_ids —— 已付费样本自动走复用、未付费样本进入
        _paid 集合，后续任何训练路径不会再对它们重复计费。
        修复 S1 遗留缺陷：旧实现只按条数 acquire(n)，不把 ID 放进
        付费集合，会被后续 acquire_ids 重复计费。
        """
        if isinstance(ids, int):
            # 计数兼容路径（无 ID 场景）：严格口径下按条数计费，
            # 无法去重——生产路径一律传 ID 集合。
            self.eval_only += ids
            if not self.eval_free:
                self.acquire(ids, "eval")
            return
        ids = {int(i) for i in ids}
        unpaid = ids - self._paid
        self.eval_only += len(unpaid)
        if not self.eval_free and unpaid:
            self.acquire_ids(unpaid, "eval")

    def record_reuse(self, n: int) -> None:
        """已付费标签被再次投入训练（清洗保留样本等），记录复用量。"""
        self.reused += int(n)

    # ------------------------- 汇总 -------------------------

    def summary(self) -> dict:
        return {
            "labels_acquired": self.acquired,
            "labels_reused": self.reused,
            "labels_eval_only": self.eval_only,
            "labels_by_purpose": dict(self.by_purpose),
            "label_budget": self.budget,
            "labels_remaining": self.remaining(),
            "over_budget_events": self.over_budget_events,
            "over_budget_labels": self.over_budget_labels,
            "eval_free": self.eval_free,
        }

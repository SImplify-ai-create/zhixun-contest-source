# -*- coding: utf-8 -*-
"""v1 检测器库注册表（19 条 DetectorSpec）迁入 v7 镜像。

来源：``2026-11-16/detector-lib/detlib/registry.py``（只读迁移，字段按 v7
需要裁剪，语义不变）。注册表只登记「能算什么量」与「默认三态」，
不登记「该怎么处置」；处置仍由 ``app/selector.py`` 独占。

三态（与 v1 一致）：
    enabled      已标定，默认开；每轮都会尝试出数
    conditional  默认关；显式激活后仍必须有一次离线标定记录才允许动作
    alarm_only   只报警；永不进入动作决策（CUSUM：k/h 未标定，保持 None）

本模块纯 numpy，无第三方依赖；只在 train 阶段被 ``app/train_stage.py`` 导入。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

ENABLED = "enabled"
CONDITIONAL = "conditional"
ALARM_ONLY = "alarm_only"
STATUSES = (ENABLED, CONDITIONAL, ALARM_ONLY)


@dataclass(frozen=True)
class DetectorSpec:
    id: str
    name: str
    status: str
    family: str
    min_batch_n: int
    calibration_required: str      # none / offline_200 / never
    needs: tuple[str, ...]         # 运行时需要的输入：features / labels / probs / logits / meta
    note: str

    @property
    def is_default_on(self) -> bool:
        return self.status == ENABLED

    @property
    def is_alarm_only(self) -> bool:
        return self.status == ALARM_ONLY


_SPECS: tuple[DetectorSpec, ...] = (
    # ---- 3 条默认开（v1 已标定，无需额外标定）
    DetectorSpec("argmax_kl", "argmax 预测分布 KL", ENABLED, "distribution", 1, "none",
                 ("probs",), "预测类份额 vs 池先验的 KL；分类专用。v7 无输出层时显式 unavailable。"),
    DetectorSpec("mahal_pen", "倒数第二层马氏距离", ENABLED, "geometry", 1, "none",
                 ("features",), "v7 无 checkpoint 中间层，按特征空间代理出数并标 space=features。"),
    DetectorSpec("bbsd", "BBSD（两样本 Hotelling T²）", ENABLED, "classifier_two_sample", 1, "none",
                 ("features",), "v1 在 softmax 上；v7 降级为特征空间 T²，标 space=features。"),
    # ---- 15 条默认关、可显式激活（激活后仍需离线标定才允许动作）
    DetectorSpec("entropy_mean", "平均预测熵", CONDITIONAL, "confidence", 20, "offline_200",
                 ("probs",), "置信度塌缩代理量；无 probs 时 unavailable。"),
    DetectorSpec("mmd_rbf", "RBF-MMD（标准化原始空间）", CONDITIONAL, "distribution", 20, "offline_200",
                 ("features",), "有偏估计，核宽取 train 中位距离。"),
    DetectorSpec("c2st_lda", "C2ST（LDA 平衡准确率）", CONDITIONAL, "classifier_two_sample", 20,
                 "offline_200", ("features",), "train vs 批 的 LDA 平衡准确率。"),
    DetectorSpec("cluster_separability", "簇间可分性 sw/(sw+sb)", CONDITIONAL, "geometry", 60,
                 "offline_200", ("features", "labels"), "越小越可分；只作汇总读数，不单独告警。"),
    DetectorSpec("cluster_mean_shift", "簇均值漂移", CONDITIONAL, "geometry", 60,
                 "offline_200", ("features", "labels"), "每类质心到源质心的马氏距离 / sqrt(d)。"),
    DetectorSpec("cluster_cov_shift", "簇协方差漂移", CONDITIONAL, "geometry", 60,
                 "offline_200", ("features", "labels"), "类内协方差的迹比型差异。"),
    DetectorSpec("cluster_merge", "簇合并", CONDITIONAL, "geometry", 60,
                 "offline_200", ("features", "labels"), "类间一维分离度相对源值的塌缩。"),
    DetectorSpec("cluster_split", "簇分裂", CONDITIONAL, "geometry", 60,
                 "offline_200", ("features", "labels"), "类内 2-means 分离度相对源值的增量。"),
    DetectorSpec("mahalanobis_ps", "逐样本马氏距离", CONDITIONAL, "geometry", 1, "offline_200",
                 ("features",), "逐样本分位口径；返回均值，标 granularity=per_sample。"),
    DetectorSpec("energy_ps", "逐样本能量得分", CONDITIONAL, "confidence", 1, "offline_200",
                 ("logits",), "-T*logsumexp(logits/T)；无 logits 时 unavailable。"),
    DetectorSpec("prototype_ps", "逐样本原型距离", CONDITIONAL, "geometry", 1, "offline_200",
                 ("features",), "到 train 2-means 原型的平均距离。"),
    DetectorSpec("unit_monotone", "单元内退化单调性", CONDITIONAL, "regression", 20, "offline_200",
                 ("features", "meta"), "逐单元 Spearman(快照序号, 健康指标) 均值；需 unit/snapshot。"),
    DetectorSpec("trend_residual_drift", "趋势残差分布漂移", CONDITIONAL, "regression", 20,
                 "offline_200", ("features", "meta"), "源趋势残差的分布距离；需 unit/snapshot。"),
    DetectorSpec("cond_feature_drift", "逐工况特征分布漂移", CONDITIONAL, "condition", 20,
                 "offline_200", ("features", "meta"), "按工况键分组，取最差键的标准化均值差。"),
    DetectorSpec("cond_key_coverage", "工况键覆盖率", CONDITIONAL, "condition", 1,
                 "offline_200", ("features", "meta"), "1 - 未见工况键占比；缺 settings 时 unavailable。"),
    # ---- 1 条只报警
    DetectorSpec("cusum_composite", "CUSUM（合成分数沿 C2ST 方向）", ALARM_ONLY, "sequential", 20,
                 "never", ("features",), "k/h 未标定保持 None；只报警，永不动作。"),
)

REGISTRY: MappingProxyType = MappingProxyType({s.id: s for s in _SPECS})
ALL_IDS: tuple[str, ...] = tuple(s.id for s in _SPECS)
ENABLED_IDS: tuple[str, ...] = tuple(s.id for s in _SPECS if s.status == ENABLED)
CONDITIONAL_IDS: tuple[str, ...] = tuple(s.id for s in _SPECS if s.status == CONDITIONAL)
ALARM_ONLY_IDS: tuple[str, ...] = tuple(s.id for s in _SPECS if s.status == ALARM_ONLY)
CALIBRATABLE_IDS: tuple[str, ...] = tuple(s.id for s in _SPECS
                                          if s.calibration_required == "offline_200")


def get(detector_id: str) -> DetectorSpec:
    try:
        return REGISTRY[detector_id]
    except KeyError:
        raise KeyError("未登记的检测器 %r（已登记 %d 条）" % (detector_id, len(REGISTRY))) from None


def table() -> list[dict]:
    """整张注册表，供运行时查询与账本落盘。"""
    return [{"id": s.id, "name": s.name, "status": s.status, "family": s.family,
             "min_batch_n": s.min_batch_n, "calibration_required": s.calibration_required,
             "needs": list(s.needs), "note": s.note} for s in _SPECS]

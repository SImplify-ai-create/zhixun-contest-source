"""能力开关注册表（影子模式：只按证据判定并留痕，不接管任何开关）。

设计原文：`能力开关自动化设计_2026-11-20.md`。本模块把"默认关闭的能力"写成可审计的行：
`默认态 / 触发开判据 / kill_if 关判据 / 标定状态 / 可判时点 / 判定函数`。

三条红线（`tests/test_capability_registry.py` 会锁定）：
1. 影子模式：本模块不被 pipeline / 训练循环消费；导入不产生副作用。判定结果只写
   `selector_trace` 留痕，**不修改任何默认值，也不切换任何实际开关**。
2. 纯函数：判定函数只读传入的 evidence 映射（已落盘的测量值），不读时间、随机数、
   文件、环境变量或模型自评。同一份 evidence 必然给出同一份判定。
3. 不自造阈值：判据里出现的数字必须能在预注册文档或盘上报告里找到；找不到就返回
   `indeterminate` 并保持默认关闭，不猜、不凑。

措辞约定：判定结论是"按证据放行 / 维持默认关闭 / 无法判定"，不是"自动激活能力"。
即使结论为 `allow_on`，也只表示"证据允许提请放开"，影子模式下不会执行。
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHADOW_MODE = "shadow"
DEFAULT_OFF = "off"

# "可判时点"只有两个合法取值：轮内可判 / 仅离线可判。
EVAL_IN_ROUND = "轮内可判"
EVAL_OFFLINE = "仅离线可判"
EVAL_POINTS = (EVAL_IN_ROUND, EVAL_OFFLINE)

# 判定结论（不是动作指令）
ALLOW_ON = "allow_on"              # 证据满足触发开判据：允许提请放开（影子模式不执行）
KEEP_DEFAULT = "keep_default_off"  # kill_if 触发或触发判据未满足：维持默认关闭
INDETERMINATE = "indeterminate"    # 证据缺失/判据未预注册：不判，维持默认关闭

VERDICT_VALUES = (ALLOW_ON, KEEP_DEFAULT, INDETERMINATE)


@dataclass(frozen=True)
class Verdict:
    """一次判定的留痕内容。"""

    decision: str
    criterion: str
    observed: str
    reason: str


@dataclass(frozen=True)
class Switch:
    """注册表的一行。judge 必须是纯函数：evidence -> Verdict。"""

    name: str
    default_state: str
    open_criterion: str
    kill_if: str
    calibration_status: str
    evaluable_at: str
    evidence_ref: str
    judge: Callable[[Mapping[str, Any]], Verdict]


# ---------------------------------------------------------------- 纯函数工具

def _number(value: Any) -> float | None:
    """把证据里的数值转成 float；bool / 非数值 / NaN 一律视为缺失。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if value != value else value


def _fmt(value: float) -> str:
    """稳定、可回读的数值文本（不用时间/本地化，保证同输入同输出）。"""
    return repr(float(value))


def _indeterminate(criterion: str, observed: str) -> Verdict:
    return Verdict(INDETERMINATE, criterion, observed, "证据不足或判据未预注册：不判，保持默认关闭")


# ---------------------------------------------------------------- 判定函数

_REF_DETECTOR_AUC = (
    "2026-11-19/product-brief/outputs/作品全景说明.md#L91"
    "（跨工况 AUC 0.8093→0.6848→0.5808；旧检查器 0.5323；原始报告目录已不在盘上）"
)
_REF_KALMAN = "2026-11-19/product-brief/outputs/作品全景说明.md#L680-L681（Shapley +1.533595；15/106）"
_REF_ACTION_CALIB = (
    "2026-11-19/product-brief/outputs/作品全景说明.md#L697（LOO 56% vs 恒定基线 48%）"
    "；判据预注册于 能力开关自动化设计_2026-11-20.md#L46"
    "（argmin ≥ 基线 +15pp、表外回退 <10%）"
)
_REF_UNCERTAINTY = "2026-11-19/product-brief/outputs/作品全景说明.md#L687-L689（ΔIS -0.255，CI [-0.599,+0.165]）"
_REF_CUSUM = "2026-11-16/detector-lib/detlib/config.py#L51-L52（cusum_k / cusum_h = None）"
_REF_ROUNDS = "2026-11-17/contest-v2/app/llm_cage.py#L41-L42（CLOSED_ROUNDS=6，TOTAL_ROUNDS=10）"
_REF_DESIGN_L16 = "能力开关自动化设计_2026-11-20.md#L16（仅定性：kill_if 未过 / 未做）"


def judge_detector_migration(evidence: Mapping[str, Any]) -> Verdict:
    """16 个未启用检测器的族级判据：跨工况迁移是否不损失。

    只用符号比较（未见工况 AUC ≥ 可见工况 AUC），不引入新的数值门槛。
    """
    criterion = "auc_unseen_worst >= auc_visible（迁移不损失；符号判据，不引入新阈值）"
    block = evidence.get("detector_migration")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, "证据缺失：detector_migration 未落盘")
    visible = _number(block.get("auc_visible"))
    worst = _number(block.get("auc_unseen_worst"))
    if visible is None or worst is None:
        return _indeterminate(criterion, "证据字段缺失：auc_visible / auc_unseen_worst")
    observed = f"auc_visible={_fmt(visible)}; auc_unseen_worst={_fmt(worst)}"
    if worst >= visible:
        return Verdict(ALLOW_ON, criterion, observed, "未见工况未低于可见工况；仅表示证据允许提请放开")
    return Verdict(KEEP_DEFAULT, criterion, observed, "跨工况 AUC 下降：维持默认关闭")


def judge_kalman_ablation(evidence: Mapping[str, Any]) -> Verdict:
    """Kalman：Shapley(ΔRMSE) ≤ 0 且支持单元数过半，才允许提请放开。

    ΔRMSE 为正表示更差（与盘上报告同号），因此这里不引入额外阈值。
    """
    criterion = "shapley_rmse_delta <= 0 且 better_units > worse_units（符号判据）"
    block = evidence.get("kalman_ablation")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, "证据缺失：kalman_ablation 未落盘")
    shapley = _number(block.get("shapley_rmse_delta"))
    better = _number(block.get("better_units"))
    worse = _number(block.get("worse_units"))
    if shapley is None or better is None or worse is None:
        return _indeterminate(criterion, "证据字段缺失：shapley_rmse_delta / better_units / worse_units")
    observed = (f"shapley_rmse_delta={_fmt(shapley)}; "
                f"better_units={int(better)}; worse_units={int(worse)}")
    if shapley <= 0.0 and better > worse:
        return Verdict(ALLOW_ON, criterion, observed, "配对消融转正；仅表示证据允许提请放开")
    return Verdict(KEEP_DEFAULT, criterion, observed, "Shapley 为正或支持单元不过半：维持默认关闭")


def _judge_recorded_kill_if(block_name: str) -> Callable[[Mapping[str, Any]], Verdict]:
    """消费"已在离线侧执行过的预注册 kill_if"结论；协议/数值未落盘时不判。"""

    def judge(evidence: Mapping[str, Any]) -> Verdict:
        criterion = f"{block_name}: 预注册 kill_if 已执行且未触发"
        block = evidence.get(block_name)
        if not isinstance(block, Mapping):
            return _indeterminate(criterion, f"证据缺失：{block_name} 的 kill_if 结论未落盘")
        passed = block.get("kill_if_passed")
        if passed is True:
            return Verdict(ALLOW_ON, criterion, "kill_if_passed=True", "仅表示证据允许提请放开")
        if passed is False:
            return Verdict(KEEP_DEFAULT, criterion, "kill_if_passed=False", "kill_if 触发：维持默认关闭")
        return _indeterminate(criterion, "kill_if_passed 缺失或不是布尔值")

    return judge


judge_gpr = _judge_recorded_kill_if("gpr_degradation")
judge_meeker = _judge_recorded_kill_if("meeker_degradation")


def judge_action_calibration_v2(evidence: Mapping[str, Any]) -> Verdict:
    """动作反事实标定 v2.0：两步预注册判据（设计文档 §4，数字写死）。"""
    criterion = "loo_hit - constant_baseline_hit >= 0.15 且 out_of_table_fallback < 0.10"
    block = evidence.get("action_calibration_v2")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, "证据缺失：action_calibration_v2 未落盘")
    loo = _number(block.get("loo_hit"))
    base = _number(block.get("constant_baseline_hit"))
    if loo is None or base is None:
        return _indeterminate(criterion, "证据字段缺失：loo_hit / constant_baseline_hit")
    gain = round(loo - base, 9)  # 去掉二进制浮点尾差，判据与留痕同一口径
    observed = (f"loo_hit={_fmt(loo)}; constant_baseline_hit={_fmt(base)}; "
                f"gain={_fmt(gain)}")
    if gain < 0.15:
        return Verdict(KEEP_DEFAULT, criterion, observed,
                       "LOO 增益未达 +15pp：维持默认关闭（第二判据无需再判）")
    fallback = _number(block.get("out_of_table_fallback"))
    if fallback is None:
        return _indeterminate(criterion, observed + "; out_of_table_fallback 未落盘")
    observed += f"; out_of_table_fallback={_fmt(fallback)}"
    if fallback < 0.10:
        return Verdict(ALLOW_ON, criterion, observed, "两步判据均满足；仅表示证据允许提请切换 mode")
    return Verdict(KEEP_DEFAULT, criterion, observed, "表外回退 ≥10%：维持默认关闭")


def judge_online_learning(evidence: Mapping[str, Any]) -> Verdict:
    """在线学习：先判轮数上限（结构性判据），再看跨轮增量收益区间。"""
    budget = evidence.get("run_budget")
    rounds = _number(budget.get("total_rounds")) if isinstance(budget, Mapping) else None
    if rounds is None:
        return _indeterminate("total_rounds <= 10 即维持关闭", "证据缺失：run_budget.total_rounds")
    if rounds <= 10:
        criterion = "total_rounds <= 10 即维持关闭（10 轮不足以标定在线学习器）"
        return Verdict(KEEP_DEFAULT, criterion, f"total_rounds={int(rounds)}",
                       "轮数上限内无法标定：维持默认关闭")
    criterion = "跨轮增量收益的区间下界 > 0"
    block = evidence.get("online_learning")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, f"total_rounds={int(rounds)}; 无预注册的跨轮收益判据")
    low = _number(block.get("gain_ci_low"))
    if low is None:
        return _indeterminate(criterion, "证据字段缺失：gain_ci_low")
    if low > 0.0:
        return Verdict(ALLOW_ON, criterion, f"gain_ci_low={_fmt(low)}",
                       "仅表示证据允许提请放开")
    return Verdict(KEEP_DEFAULT, criterion, f"gain_ci_low={_fmt(low)}",
                   "区间下界未转正：维持默认关闭")


def judge_uncertainty_abstention(evidence: Mapping[str, Any]) -> Verdict:
    """不确定性/弃权：ΔIS 的 95% 区间下界是否为正。"""
    criterion = "delta_is 的 95% 区间下界 > 0（区间跨零即维持关闭）"
    block = evidence.get("uncertainty_abstention")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, "证据缺失：uncertainty_abstention 未落盘")
    delta = _number(block.get("delta_is"))
    low = _number(block.get("ci_low"))
    high = _number(block.get("ci_high"))
    if delta is None or low is None or high is None:
        return _indeterminate(criterion, "证据字段缺失：delta_is / ci_low / ci_high")
    observed = f"delta_is={_fmt(delta)}; ci=[{_fmt(low)}, {_fmt(high)}]"
    if low > 0.0:
        return Verdict(ALLOW_ON, criterion, observed, "仅表示证据允许提请放开")
    if high < 0.0:
        return Verdict(KEEP_DEFAULT, criterion, observed, "区间整体为负：维持默认关闭")
    return Verdict(KEEP_DEFAULT, criterion, observed, "区间跨零，无法证明正收益：维持默认关闭")


def judge_cusum_sequential(evidence: Mapping[str, Any]) -> Verdict:
    """CUSUM 顺序通道：k/h 是配置层前置条件，轮内即可查。"""
    criterion = "cusum_k 与 cusum_h 均已预注册且非空（否则不得出判决）"
    block = evidence.get("sequential_cusum")
    if not isinstance(block, Mapping):
        return _indeterminate(criterion, "证据缺失：sequential_cusum")
    k = _number(block.get("cusum_k"))
    h = _number(block.get("cusum_h"))
    observed = f"cusum_k={block.get('cusum_k')!r}; cusum_h={block.get('cusum_h')!r}"
    if k is None or h is None:
        return Verdict(KEEP_DEFAULT, criterion, observed,
                       "k/h 留空：顺序通道不出判决，维持默认关闭")
    return _indeterminate(criterion, observed,
                          "k/h 已存在，但顺序通道功效门未预注册，无法判定")


# ---------------------------------------------------------------- 注册表

# 19 个检测器中未启用的 16 个（`detlib/config.py` 只启用 argmax_kl / mahal_pen / bbsd）。
DISABLED_DETECTORS = (
    "entropy_mean",
    "mmd_rbf",
    "c2st_lda",
    "cluster_separability",
    "cluster_mean_shift",
    "cluster_cov_shift",
    "cluster_merge",
    "cluster_split",
    "mahalanobis_ps",
    "energy_ps",
    "prototype_ps",
    "unit_monotone",
    "trend_residual_drift",
    "cond_feature_drift",
    "cond_key_coverage",
    "cusum_composite",
)


def _detector_switch(detector_id: str) -> Switch:
    """除 CUSUM 外的 15 个：共用族级跨工况迁移判据。"""
    return Switch(
        name=f"detector:{detector_id}",
        default_state=DEFAULT_OFF,
        open_criterion="预注册的跨工况迁移复测：未见工况 AUC 不低于可见工况",
        kill_if="跨工况 AUC 随迁移下降（族级已判负：0.8093→0.6848→0.5808）",
        calibration_status="已判负（族级：跨工况不可迁移）",
        evaluable_at=EVAL_OFFLINE,
        evidence_ref=_REF_DETECTOR_AUC,
        judge=judge_detector_migration,
    )


REGISTRY: tuple[Switch, ...] = tuple(
    [_detector_switch(det) for det in DISABLED_DETECTORS if det != "cusum_composite"]
    + [
        Switch(
            name="detector:cusum_composite",
            default_state=DEFAULT_OFF,
            open_criterion="cusum_k / cusum_h 已预注册且非空",
            kill_if="k/h 任一为空即不得出判决（当前两者均为 None）",
            calibration_status="未标定（k/h 显式留空）",
            evaluable_at=EVAL_IN_ROUND,
            evidence_ref=_REF_CUSUM,
            judge=judge_cusum_sequential,
        ),
        Switch(
            name="kalman",
            default_state=DEFAULT_OFF,
            open_criterion="配对消融重测：Shapley(ΔRMSE) ≤ 0 且支持单元数过半",
            kill_if="Shapley(ΔRMSE) > 0 或支持单元不过半（已判负：+1.5336，15/106）",
            calibration_status="已判负（两次：5/5 更差；2×2 Shapley +1.53）",
            evaluable_at=EVAL_OFFLINE,
            evidence_ref=_REF_KALMAN,
            judge=judge_kalman_ablation,
        ),
        Switch(
            name="gpr",
            default_state=DEFAULT_OFF,
            open_criterion="预注册 kill_if 复跑后仍未触发",
            kill_if="kill_if 未过（结论与协议需落盘，当前仅有定性记录）",
            calibration_status="已判负（kill_if 未过）",
            evaluable_at=EVAL_OFFLINE,
            evidence_ref=_REF_DESIGN_L16,
            judge=judge_gpr,
        ),
        Switch(
            name="meeker",
            default_state=DEFAULT_OFF,
            open_criterion="先补做预注册 kill_if，且复跑未触发",
            kill_if="未做即维持默认关闭",
            calibration_status="未做",
            evaluable_at=EVAL_OFFLINE,
            evidence_ref=_REF_DESIGN_L16,
            judge=judge_meeker,
        ),
        Switch(
            name="action_calibration_v2",
            default_state=DEFAULT_OFF,
            open_criterion="LOO 命中 − 恒定基线 ≥ 0.15 且表外回退 < 0.10",
            kill_if="LOO 增益 < 0.15 或表外回退 ≥ 0.10（当前 +8pp）",
            calibration_status="未标定（合成表 +8pp，门槛 +15pp 未达）",
            evaluable_at=EVAL_OFFLINE,
            evidence_ref=_REF_ACTION_CALIB,
            judge=judge_action_calibration_v2,
        ),
        Switch(
            name="online_learning",
            default_state=DEFAULT_OFF,
            open_criterion="总轮数 > 10 且跨轮增量收益的区间下界 > 0",
            kill_if="总轮数 ≤ 10 即维持关闭（10 轮养不出在线学习器）",
            calibration_status="未标定",
            evaluable_at=EVAL_IN_ROUND,
            evidence_ref=_REF_ROUNDS,
            judge=judge_online_learning,
        ),
        Switch(
            name="uncertainty_abstention",
            default_state=DEFAULT_OFF,
            open_criterion="ΔIS 的 95% 区间下界 > 0",
            kill_if="区间跨零或整体为负（当前 -0.255，CI [-0.599, +0.165]）",
            calibration_status="判负（K4 未触发，NMI 0.0817）",
            evaluable_at=EVAL_OFFLINE,
            evidence_ref=_REF_UNCERTAINTY,
            judge=judge_uncertainty_abstention,
        ),
    ]
)

BY_NAME: dict[str, Switch] = {switch.name: switch for switch in REGISTRY}


# ---------------------------------------------------------------- 留痕

def selector_trace_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """跑一遍注册表，返回可直接写入 selector_trace 的行（顺序固定、无时间戳）。"""
    rows: list[dict[str, Any]] = []
    for switch in REGISTRY:
        verdict = switch.judge(evidence)
        rows.append({
            "event": "selector_trace",
            "mode": SHADOW_MODE,
            "switch": switch.name,
            "default_state": switch.default_state,
            "verdict": verdict.decision,
            "criterion": verdict.criterion,
            "observed": verdict.observed,
            "reason": verdict.reason,
            "evidence_ref": switch.evidence_ref,
            "evaluable_at": switch.evaluable_at,
            "calibration_status": switch.calibration_status,
        })
    return rows


def write_selector_trace(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """把 trace 行写成 JSONL；键排序固定，内容与写入时间无关。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return out


def emit_selector_trace(logger, evidence: Mapping[str, Any], step: int | None = None) -> int:
    """把判定结果逐行写进现有 ExperimentLogger（同样只留痕，不消费判定）。"""
    rows = selector_trace_rows(evidence)
    for row in rows:
        logger.log(row, step=step)
    return len(rows)

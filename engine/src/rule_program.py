"""规则程序：把可审计的数值 if-then 规则编译成与 RuleController 同接口的策略。

为什么单独一层：策略控制器（src/strategy.py）只约定 decide(evidence) 的输入输出，
把"策略内容"写死在代码里，外部提案（LLM / 离线启发式）就无法在不改管线的前提下
被验证。这里把策略内容外置为 JSON 规则程序，编译后仍是同一个 controller 插槽，
管线其他部分零改动。

规则程序的约束（不是建议，是校验器强制的）：
  - 条件树深度 <= 2：根只能是 all/any，叶子只能是单特征比较；
  - 特征必须在白名单内（对应管线真实能提供的证据字段）；
  - 档位必须是 observe/incremental/full；
  - 规则按顺序首条命中生效，无命中走 default_tier。

与 RuleController 的一致性：
  - 同样带"剩余标注不足 600 条时 full 降级 incremental"的预算护栏，
    默认开启；因此规则之间的差异只来自规则本身，不来自护栏缺失。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TIERS = ("observe", "incremental", "full")

# 管线 controller 证据里真实存在、且提案方被允许使用的字段。
FEATURES = (
    "severity", "residual_ratio", "psi", "psi_term", "mmd_z", "mean_psi",
    "ks_p_value", "labels_left", "retrains_left", "batch_index",
    "residual_alarm", "input_alarm", "task_alarm",
)

OPS = ("<", "<=", ">", ">=", "==", "!=")

_BOOL_FEATURES = ("residual_alarm", "input_alarm", "task_alarm")

DEFAULT_TIER_SENTINEL = "follow_severity"

# 与 RuleController 完全一致的预算护栏阈值（一次全量重训约需 600 条）
FULL_RETRAIN_COST = 600


class RuleProgramError(ValueError):
    """规则程序不合法（提案阶段即拒绝，不进入验收）。"""


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_program(obj: dict) -> list[str]:
    """返回错误列表；空列表表示合法。校验是严格的，宁拒绝不猜测。"""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON 对象"]

    rules = obj.get("rules")
    if not isinstance(rules, list):
        errs.append("rules 必须是数组（允许为空：纯默认档透传程序）")
    else:
        seen_ids = set()
        for i, r in enumerate(rules):
            where = f"rules[{i}]"
            if not isinstance(r, dict):
                errs.append(f"{where} 必须是对象")
                continue
            rid = r.get("id")
            if not isinstance(rid, str) or not rid:
                errs.append(f"{where}.id 必须是非空字符串")
            elif rid in seen_ids:
                errs.append(f"{where}.id 重复：{rid}")
            else:
                seen_ids.add(rid)
            cond = r.get("if")
            if not isinstance(cond, dict):
                errs.append(f"{where}.if 必须是对象")
            else:
                errs.extend(_validate_cond(cond, f"{where}.if", depth=1))
            then = r.get("then")
            if not isinstance(then, dict) or then.get("tier") not in TIERS:
                errs.append(f"{where}.then.tier 必须是 {TIERS} 之一")

    default = obj.get("default_tier", DEFAULT_TIER_SENTINEL)
    if default not in TIERS and default != DEFAULT_TIER_SENTINEL:
        errs.append(f"default_tier 必须是 {TIERS} 或 {DEFAULT_TIER_SENTINEL}")
    return errs


def _validate_cond(cond: dict, where: str, depth: int) -> list[str]:
    errs: list[str] = []
    if depth > 2:
        return [f"{where} 条件深度超过 2"]
    if "feature" in cond:
        f = cond.get("feature")
        op = cond.get("op")
        v = cond.get("value")
        if f not in FEATURES:
            errs.append(f"{where}.feature 不在白名单：{f!r}")
        if op not in OPS:
            errs.append(f"{where}.op 非法：{op!r}")
        if f in _BOOL_FEATURES:
            # 布尔特征接受 true/false 与 0/1（LLM 常见写法）；语义不变
            if not isinstance(v, bool) and v not in (0, 1):
                errs.append(f"{where}.value 对布尔特征必须是 true/false（或 0/1）")
        elif not _is_number(v):
            errs.append(f"{where}.value 必须是数值")
        extra = set(cond) - {"feature", "op", "value"}
        if extra:
            errs.append(f"{where} 含多余键：{sorted(extra)}")
        return errs

    modes = [k for k in ("all", "any") if k in cond]
    if len(modes) != 1:
        return [f"{where} 必须是 all/any 之一（或单特征比较）"]
    if set(cond) - {modes[0]}:
        errs.append(f"{where} 含多余键：{sorted(set(cond) - {modes[0]})}")
    children = cond[modes[0]]
    if not isinstance(children, list) or not children:
        return [f"{where}.{modes[0]} 必须是非空数组"]
    for j, child in enumerate(children):
        if not isinstance(child, dict):
            errs.append(f"{where}.{modes[0]}[{j}] 必须是对象")
            continue
        errs.extend(_validate_cond(child, f"{where}.{modes[0]}[{j}]", depth + 1))
    return errs


def _cmp(op: str, a, b) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise RuleProgramError(f"未知算子 {op!r}")


def _eval_cond(cond: dict, evidence: dict) -> bool:
    if "feature" in cond:
        f = cond["feature"]
        if f not in evidence:
            # 证据缺字段时该条视为不命中，而不是默认放行（保守）
            return False
        v = evidence[f]
        if v is None:
            return False
        try:
            return _cmp(cond["op"], v, cond["value"])
        except TypeError:
            return False
    if "all" in cond:
        return all(_eval_cond(c, evidence) for c in cond["all"])
    return any(_eval_cond(c, evidence) for c in cond["any"])


@dataclass
class CompiledProgram:
    program_id: str
    rules: list[dict]
    default_tier: str = DEFAULT_TIER_SENTINEL
    source: dict = field(default_factory=dict)

    def evaluate(self, evidence: dict) -> tuple[str | None, str | None]:
        """返回 (tier, matched_rule_id)；无命中时 tier=None 表示走默认档。"""
        for r in self.rules:
            if _eval_cond(r["if"], evidence):
                return r["then"]["tier"], r.get("id")
        return None, None

    def to_dict(self) -> dict:
        return {"program_id": self.program_id, "rules": self.rules,
                "default_tier": self.default_tier, "source": self.source}


def compile_program(obj: dict) -> CompiledProgram:
    errs = validate_program(obj)
    if errs:
        raise RuleProgramError("; ".join(errs))
    return CompiledProgram(
        program_id=str(obj.get("program_id") or "unnamed"),
        rules=list(obj["rules"]),
        default_tier=obj.get("default_tier", DEFAULT_TIER_SENTINEL),
        source=dict(obj.get("source") or {}),
    )


def load_program(path: str | Path) -> CompiledProgram:
    return compile_program(json.loads(Path(path).read_text(encoding="utf-8")))


class RuleProgramController:
    """与 RuleController 同接口的规则程序控制器。

    name 字段会写进 controller_decision 事件，便于在实验记录里区分策略来源。
    """

    name = "rule_program"

    def __init__(self, program: CompiledProgram | dict, budget_guard: bool = True):
        self.program = (program if isinstance(program, CompiledProgram)
                        else compile_program(program))
        self.budget_guard = bool(budget_guard)
        self.name = f"rule_program:{self.program.program_id}"

    def decide(self, evidence: dict) -> dict:
        tier, rule_id = self.program.evaluate(evidence)
        if tier is None:
            if self.program.default_tier == DEFAULT_TIER_SENTINEL:
                tier = evidence.get("tier", "observe")
                why = "默认沿用严重度分档"
            else:
                tier = self.program.default_tier
                why = f"默认档 {tier}"
        else:
            why = f"命中规则 {rule_id}"

        if self.budget_guard and tier == "full":
            labels_left = int(evidence.get("labels_left", 10 ** 9))
            if labels_left < FULL_RETRAIN_COST:
                return {"tier": "incremental",
                        "rationale": (f"{why}；剩余标注 {labels_left} 条不足以支撑"
                                      f"全量重训，降级增量")}
        return {"tier": tier, "rationale": why}

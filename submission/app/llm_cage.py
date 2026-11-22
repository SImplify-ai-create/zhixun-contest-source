#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3：把 LLM 提案关进“错了也伤不到回合”的笼子。

判断前提：10 轮预算下 LLM 的价值上限由“解析率 × 约束合规率”决定，不由推理能力决定。
因此本模块只做四件事，且每件都可核对：

1. 提案模板：照 `LLM使用建议_2026-11-18.md` §3 原文，动作只从已注册闭集里选，
   `params` 给区间并在本侧校验；硬约束与输出格式放首尾。
2. 状态卡：臂后验（均值/标准差/次数）由代码算，模型只读不写；附最近 3 轮账本摘要、
   当前轮次、冻结约束。记忆用数据，不用叙事。
3. 外部验证器：候选先在 `/workspace` 沙盒里构建，跑性质测试（因果、恒等、体积、契约、维度）
   与红队五问，全过才允许提交；失败原文回灌一次修复重试，仍失败退回规则选择器。
4. 红队五问：照 §3 原文的第二段模板；每问必须带引用，无引用的结论不计入决策。

不做的三件（与原文一致）：自一致采样；让模型写“经验总结”进策略库；为拉分反复换提示词版本。
`temperature=0`，记 `call_id` 与响应 sha256，但不假设输出逐位可复现——可复现性由闸门保证。
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import actions  # noqa: E402

ARMS = ("PARENT", "A", "B", "C")
ARM_ACTIONS = {"PARENT": None, "A": "streaming_ema_norm", "B": "per_cond_seq", "C": "identity"}
PARAM_DEFAULTS = {"alpha": 0.5, "eps": 1e-8}
PARAM_RANGES = {"alpha": (0.05, 0.95), "eps": (1e-12, 1e-3)}
CONFIDENCE = ("low", "medium", "high")
CLOSED_ROUNDS = 6
TOTAL_ROUNDS = 10
# 运行时生成（build_script）的第一道门：代码默认关闭。阶段门（CLOSED_ROUNDS）
# 只是第二道；答辩口径是"通道已实现、默认关闭、二次门限"，三层都要有代码对应。
RUNTIME_CODE_GEN_ENABLED = False
RUNTIME_CODE_GEN_ENV = "CONTEST_RUNTIME_CODE_GEN"
LIMIT_FILES = 64
LIMIT_BYTES = 64 * 1024 * 1024
SCRIPT_MAX_CHARS = 4000
SCRIPT_TIMEOUT = 20.0
SCRIPT_OUTPUT_CAP = 64 * 1024
HYPOTHESIS_MAX = 200
KILL_MAX = 200
DIGITS = re.compile(r"[0-9]")
# 数字白名单按"完整数字 token"比较，不做子串剥离：允许的只是平台帧给定的轮次号，
# 模型自产的 0.02 / 3.5 / 123 仍是越界 token。
NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")
LEDGER_REF = re.compile(r"(?:round|ledger|h|H)[-_]?\d+|[A-Za-z]+-\d+")
NO_HISTORY_PHRASES = (
    "账本为空",
    "无历史",
    "没有历史",
    "尚无历史",
    "首个回合",
    "首次回合",
    "首轮",
    "无可引用",
    "无账本",
    "账本尚无记录",
)
BETTER_WORDS = ("提升", "更好", "改善", "优于", "增益", "变好")
WORSE_WORDS = ("变差", "回退", "下降", "劣于", "恶化", "退化")

# §3 提案调用模板（原文照录，只替换中文占位符；JSON 行原样保留）。
PROPOSAL_TEMPLATE = """你是本轮候选选择器。只输出一个 JSON 对象，不要任何解释或前后缀。

硬约束（违反即作废，逐条自查后再输出）：
1) 不得使用任何序号大于当前行的信息；
2) 不得选择 transductive 动作：{冻结清单}；
3) 参数必须与你所选臂的契约一致：{区间表}；多余键会被直接拒收。
4) 不得重复账本中已判失败的"签名 × 臂"组合：{失败清单}。

可选臂（只能从中选一个）：{枚举}

状态
- 臂后验（均值/标准差/次数）：{表}
- 最近 3 轮：{最近三轮}
- 本轮：第 {r} 轮 / 共 {N} 轮

输出格式
{"arm":"...","params":{...},"hypothesis":"≤200字，须点名引用账本行","kill_condition":"...","confidence":"low|medium|high"}"""

# §3 红队调用模板（原文照录）。
RED_TEAM_TEMPLATE = """逐条回答下面五问，每问必须给出具体字段或行号作为引用；无法引用就写"无法判定"。
1) 提案是否使用了未来行信息？
2) 是否触及 transductive 动作（对照冻结清单）？
3) 候选体积/耗时是否越界？
4) 参数是否越出注册区间？
5) 是否重复了账本中的失败组合？
只输出 {"q1":"...","q2":"...","q3":"...","q4":"...","q5":"...","verdict":"pass|reject","reason":"≤100字"}"""


def runtime_code_gen_enabled():
    """运行时生成是否被显式打开。默认值由代码常量保证，不靠注释或配置约定。"""
    raw = os.environ.get(RUNTIME_CODE_GEN_ENV)
    if raw is None or not str(raw).strip():
        return bool(RUNTIME_CODE_GEN_ENABLED)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class CageError(Exception):
    """提案不满足闭集、区间、引用或阶段约束。原文回灌给模型做一次修复。"""


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def phase_of(round_index):
    """前 6 轮 closed（只允许闭集臂菜单）；后 4 轮 script（允许模型写构建脚本）。"""
    index = int(round_index or 1)
    return "closed" if index <= CLOSED_ROUNDS else "script"


def remaining_rounds(round_index):
    return max(0, TOTAL_ROUNDS - int(round_index or 1) + 1)


def _sd(stats):
    n = int((stats or {}).get("n", 0) or 0)
    m2 = float((stats or {}).get("m2", 0.0) or 0.0)
    if n <= 1 or m2 <= 0.0:
        return 0.0
    return math.sqrt(m2 / (n - 1))


def arm_table(arms):
    """状态卡的臂后验表。完全由代码算，模型不得改写。"""
    rows = []
    for arm in ARMS:
        stats = (arms or {}).get(arm) or {}
        n = int(stats.get("n", 0) or 0)
        if n <= 0:
            row = "%s: 次数=0（未观测）" % arm
            failures = int(stats.get("failures", 0) or 0)
            cooldown = int(stats.get("cooldown", 0) or 0)
            if failures or cooldown:
                row += "；失败=%d；冷却=%d" % (failures, cooldown)
        else:
            row = "%s: 均值=%.4f；标准差=%.4f；次数=%d" % (
                arm, float(stats.get("mean", 0.0)), _sd(stats), n)
            failures = int(stats.get("failures", 0) or 0)
            cooldown = int(stats.get("cooldown", 0) or 0)
            if failures or cooldown:
                row += "；失败=%d；冷却=%d" % (failures, cooldown)
        rows.append(row)
    return "\n  ".join(rows)


def status_card(round_index, dataset_key, metric, arms, recent_rounds, phase,
                signature_seen, total_rounds=TOTAL_ROUNDS):
    """状态卡：臂后验 + 最近 3 轮 + 轮次 + 冻结约束。总长控制在数百 token。

    `recent_rounds` 每项形如 {signature, arm, delta, verdict, ledger_id}，全部由账本提供。
    """
    lines = [
        "[状态卡] dataset=%r；指标=%s；分数已统一为“越大越好”；阶段=%s" % (dataset_key, metric, phase),
        "- 臂后验（均值/标准差/次数）：",
        "  " + arm_table(arms),
        "- 最近 3 轮：签名 | 臂 | Δ | 判定",
    ]
    recent = list(recent_rounds or [])[-3:]
    if not recent:
        lines.append("  （空：这是首个有记录的回合）")
    for item in recent:
        lines.append("  %s | %s | %s | %s | id=%s" % (
            item.get("signature", "—"), item.get("arm", "—"),
            "—" if item.get("delta") is None else "%+.4f" % float(item["delta"]),
            item.get("verdict") or "—", item.get("ledger_id") or "—"))
    lines.append("- 本轮：第 %d 轮 / 共 %d 轮；剩余 %d 轮"
                 % (int(round_index), int(total_rounds), remaining_rounds(round_index)))
    lines.append("- 签名：%s" % ("已见过" if signature_seen else "未见（本轮只允许 PARENT）"))
    lines.append("- 冻结约束：候选 <= %d 文件 / %d MiB；predict 只允许 causal；不得用未来行"
                 % (LIMIT_FILES, LIMIT_BYTES // (1024 * 1024)))
    if phase != "script":
        lines.append("- 阶段约束：前 %d 轮只允许闭集臂菜单，不接受 build_script" % CLOSED_ROUNDS)
    else:
        lines.append("- 阶段约束：允许 build_script，但必须过沙盒、性质测试与体积/时限")
    return "\n".join(lines)


def _transductive_list():
    return ", ".join(actions.transductive_actions())


def _range(name):
    low, high = PARAM_RANGES[name]
    return "[%g, %g]" % (low, high)


def arm_contract(arm):
    """单个臂的参数契约。与 _check_params 的 allowed 表同源，只换表述。"""
    arm = str(arm or "PARENT").strip().upper()
    if arm == "A":
        return "A 接受 alpha%s 与 eps%s" % (_range("alpha"), _range("eps"))
    if arm == "B":
        return "B 只接受 eps%s" % _range("eps")
    if arm == "C":
        return "C 不接受任何参数"
    return "PARENT 不接受任何参数"


def _ranges_text():
    """按臂给契约，不再输出与臂无关的全局区间表。"""
    return "；".join(arm_contract(arm) for arm in ARMS)


def _failed_text(failed):
    if not failed:
        return "（空）"
    items = []
    for entry in list(failed)[-5:]:
        items.append("%s×%s×%s" % (entry.get("signature", "—"), entry.get("arm", "—"),
                                   json.dumps(entry.get("params") or {}, sort_keys=True)))
    return "; ".join(items)


def build_messages(card, data_summary, detectors, rule_recommendation,
                   round_index, failed_combos):
    text = PROPOSAL_TEMPLATE
    for key, value in (
        ("冻结清单", _transductive_list()),
        ("区间表", _ranges_text()),
        ("失败清单", _failed_text(failed_combos)),
        ("枚举", " / ".join(ARMS)),
        ("表", ""),
        ("最近三轮", ""),
        ("{r}", str(int(round_index))),
        ("{N}", str(TOTAL_ROUNDS)),
    ):
        text = text.replace("{%s}" % key if not key.startswith("{") else key, value)
    # 状态卡替换掉模板里的三行状态占位
    text = text.replace(
        "- 臂后验（均值/标准差/次数）：\n- 最近 3 轮：\n- 本轮：第 %d 轮 / 共 %d 轮"
        % (int(round_index), TOTAL_ROUNDS), card)
    payload = {
        "data": data_summary,
        "detectors": detectors,
        "rule_recommendation": rule_recommendation,
    }
    # 推荐某臂时，把该臂自己的参数契约一并给出，避免“推荐 B 却按 A 的参数表填”。
    if isinstance(rule_recommendation, dict):
        arm = str(rule_recommendation.get("candidate") or "PARENT").strip().upper()
        payload["rule_recommendation"] = dict(rule_recommendation,
                                              param_contract=arm_contract(arm))
    return [
        {"role": "system", "content": text},
        {"role": "user",
         "content": "<untrusted_data>\n%s\n</untrusted_data>\n只输出上文规定格式的 JSON，不要解释。"
                    % json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def repair_messages(card, previous_text, errors, data_summary, detectors,
                    rule_recommendation, round_index, failed_combos):
    messages = build_messages(card, data_summary, detectors, rule_recommendation,
                              round_index, failed_combos)
    payload = {"previous_reply": str(previous_text or "")[:1200],
               "validator_errors": list(errors or [])}
    messages.append({"role": "user",
                     "content": "上一次输出被判无效，按错误清单修正，只输出同一 schema 的 JSON：\n"
                                + json.dumps(payload, ensure_ascii=False, sort_keys=True)})
    return messages


def red_team_messages(proposal, facts, failed_combos):
    payload = {
        "proposal": proposal,
        "frozen": {
            "transductive": list(actions.transductive_actions()),
            "causal": list(actions.causal_actions()),
            "param_ranges": {key: list(value) for key, value in PARAM_RANGES.items()},
            "limits": {"files": LIMIT_FILES, "bytes": LIMIT_BYTES},
            "failed_combos": list(failed_combos or []),
        },
        "candidate_facts": facts,
    }
    return [
        {"role": "system", "content": RED_TEAM_TEMPLATE},
        {"role": "user",
         "content": "<untrusted_data>\n%s\n</untrusted_data>"
                    % json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _iter_balanced_objects(raw):
    """从头扫描配平的 {...} 片段，跳过字符串里的花括号与转义引号。

    只做切片，不判断合法性：合法性交给 json.loads。
    前一对象不合法时继续从下一个 `{` 开始，避免思维链里的草稿对象连带整段作废。
    """
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield raw[start:index + 1]
                    start = raw.find("{", index + 1)
                    break
        else:
            # 这个起点没配平（截断或残缺），换下一个起点
            start = raw.find("{", start + 1)


def _extract_json(text):
    raw = str(text or "").strip()
    if not raw:
        return None, "empty"
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        raw = fenced.group(1)
    for candidate in _iter_balanced_objects(raw):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj, "ok"
    return None, "unusable"


def parse_proposal(text):
    """只做语法解析。返回 (proposal|None, status, error)。"""
    obj, status = _extract_json(text)
    if obj is None:
        return None, status, ("回复为空" if status == "empty" else "没有可解析的 JSON 对象")
    arm = obj.get("arm", obj.get("candidate"))
    if arm is None:
        return None, "unusable", "缺 arm 字段（旧格式 candidate 可兼容）"
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        return None, "unusable", "params 必须是对象"
    hypothesis = obj.get("hypothesis", obj.get("reason", ""))
    kill_condition = obj.get("kill_condition", "")
    confidence = str(obj.get("confidence", "medium") or "medium").strip().lower()
    script = obj.get("build_script")
    if script is not None and not isinstance(script, str):
        return None, "unusable", "build_script 必须是字符串或 null"
    return {
        "arm": str(arm).strip().upper(),
        "params": dict(params),
        "hypothesis": str(hypothesis or "").strip(),
        "kill_condition": str(kill_condition or "").strip(),
        "confidence": confidence,
        "build_script": script,
    }, "ok", None


def _check_params(arm, params):
    if arm == "PARENT":
        if params:
            raise CageError("PARENT 不接受 params：%r" % (params,))
        return {}
    # 参数表必须与 actions.REGISTRY 一致：identity 不接受任何参数（写入 eps 会让候选在装载期被拒）。
    # 白名单只放行 alpha/eps（B 只有 eps）。新增键位前须重新评估：像 l2sp_d_target 这类
    # 训练侧参数一旦从这里放进来，就成了绕过 pre-registration 的侧信道。
    allowed = {"A": ("alpha", "eps"), "B": ("eps",), "C": ()}[arm]
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise CageError("arm=%s 不接受参数 %r（允许：%s）"
                        % (arm, unknown, ", ".join(allowed) or "无"))
    merged = {}
    for name in allowed:
        value = params.get(name, PARAM_DEFAULTS.get(name))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise CageError("参数 %s 必须是数值：%r" % (name, value))
        low, high = PARAM_RANGES[name]
        if not math.isfinite(value) or value < low or value > high:
            raise CageError("参数 %s=%r 越界，必须在 [%g, %g]" % (name, value, low, high))
        merged[name] = value
    return merged


def _cites_ledger(hypothesis, history_ids):
    # 空账本：语义判据是"承认了没有历史可引"，不是猜中某个词；闭集短语统一在这里。
    text = str(hypothesis or "")
    if not history_ids:
        return any(phrase in text for phrase in NO_HISTORY_PHRASES)
    if any(str(item) in text for item in history_ids):
        return True
    return bool(LEDGER_REF.search(text))


def classify_failure(errors):
    """把校验错误串归类成机器可读的失败分类，供账本与审计聚合。"""
    text = "；".join(str(e) for e in errors or [])
    if not text:
        return "none"
    if "回复为空" in text:
        return "empty_reply"
    if "没有可解析的 JSON" in text or "缺 arm" in text or "params 必须是对象" in text:
        return "parse_error"
    if "不得出现数字" in text:
        return "digits_in_hypothesis"
    if "未点名引用账本行" in text:
        return "missing_ledger_ref"
    if "不在闭集" in text:
        return "arm_not_in_closed_set"
    if "越界" in text or "必须是数值" in text or "不接受参数" in text or "不接受 params" in text:
        return "params_out_of_range"
    if "RUNTIME_CODE_GEN_ENABLED" in text:
        return "runtime_code_gen_off"
    if "build_script" in text or "只允许闭集臂菜单" in text:
        return "phase_violation"
    if "hypothesis 不能为空" in text or "kill_condition 不能为空" in text:
        return "missing_field"
    return "other"


def _strip_refs(text, history_ids):
    """引用账本行必然带编号；这类编号是引用而非测量值，扫数字前先剥掉。"""
    out = str(text or "")
    for item in history_ids or []:
        out = out.replace(str(item), "")
    return LEDGER_REF.sub("", out)


def _allowed_digits(context):
    """有界白名单：只允许平台帧给出的当轮轮次号，绝不从模型文本里取。"""
    try:
        round_index = int((context or {}).get("round_index"))
    except (TypeError, ValueError):
        return set()
    return {str(round_index)}


def validate_proposal(proposal, context):
    """把“解析成功”推进到“可执行”。任何一条不满足都抛 CageError。"""
    if proposal is None:
        raise CageError("没有提案对象")
    arm = proposal.get("arm")
    if arm not in ARMS:
        raise CageError("arm=%r 不在闭集 %s 里；transductive 动作不可提案" % (arm, "/".join(ARMS)))

    params = _check_params(arm, dict(proposal.get("params") or {}))

    hypothesis = proposal.get("hypothesis") or ""
    if not hypothesis:
        raise CageError("hypothesis 不能为空，且必须点名引用账本行")
    if len(hypothesis) > HYPOTHESIS_MAX:
        raise CageError("hypothesis 超过 %d 字：%d" % (HYPOTHESIS_MAX, len(hypothesis)))
    allowed = _allowed_digits(context)
    tokens = NUMBER_TOKEN.findall(_strip_refs(hypothesis,
                                              list(context.get("history_ids") or [])))
    offenders = [token for token in tokens if token not in allowed]
    if offenders:
        raise CageError("hypothesis 不得出现数字（数字只能来自平台回执）；"
                        "本轮白名单=%s，越界=%s：%r"
                        % (sorted(allowed) or "空", offenders, hypothesis))
    if not _cites_ledger(hypothesis, list(context.get("history_ids") or [])):
        raise CageError("hypothesis 未点名引用账本行：%r" % hypothesis)

    kill_condition = proposal.get("kill_condition") or ""
    if not kill_condition:
        raise CageError("kill_condition 不能为空")
    if len(kill_condition) > KILL_MAX:
        raise CageError("kill_condition 超过 %d 字：%d" % (KILL_MAX, len(kill_condition)))

    confidence = proposal.get("confidence")
    if confidence not in CONFIDENCE:
        raise CageError("confidence=%r 不在 low/medium/high 里" % (confidence,))

    phase = context.get("phase") or "closed"
    script = proposal.get("build_script")
    if script:
        if not runtime_code_gen_enabled():
            # 第一道门：主动关闭。显式拒绝并说明开关名，绝不静默忽略脚本。
            raise CageError(
                "运行时生成通道默认关闭（RUNTIME_CODE_GEN_ENABLED=False），"
                "本版不接受 build_script；如需开启须显式设置 %s=1"
                % RUNTIME_CODE_GEN_ENV)
        if phase != "script":
            raise CageError("前 %d 轮只允许闭集臂菜单，不接受 build_script" % CLOSED_ROUNDS)
        if len(script) > SCRIPT_MAX_CHARS:
            raise CageError("build_script 超过 %d 字符" % SCRIPT_MAX_CHARS)
        if "```" in script:
            raise CageError("build_script 不要带 Markdown 围栏")

    return {
        "arm": arm,
        "params": params,
        "hypothesis": hypothesis,
        "kill_condition": kill_condition,
        "confidence": confidence,
        "build_script": script,
        "phase": phase,
    }


def review(text, context):
    """解析 + 校验一次。返回结构化结果，不抛异常。"""
    proposal, status, error = parse_proposal(text)
    errors = []
    if error:
        errors.append(error)
    accepted = None
    if proposal is not None:
        try:
            accepted = validate_proposal(proposal, context)
        except CageError as exc:
            errors.append(str(exc))
    return {
        "status": ("ok" if accepted is not None
                   else ("invalid" if status == "ok" else status)),
        "parse_status": status,
        "proposal": accepted,
        "raw": proposal,
        "errors": errors,
        "raw_sha256": sha256_text(text),
        "raw_excerpt": str(text or "")[:400],
        # 审计字段：被拒提案的臂与机器可读的失败分类，别让审计去猜字符串。
        "proposal_arm_raw": (proposal or {}).get("arm"),
        "failure_kind": ("none" if accepted is not None else classify_failure(errors)),
        "auditable": bool((text or "").strip()),
        # 数字白名单的来源与取值：来源固定是平台帧，取值只含当轮轮次号。
        "allowed_digits": sorted(_allowed_digits(context)),
        "allowed_digits_source": "round_frame",
    }


def parse_red_team(text):
    """解析五问清单。没有引用的结论标记为 uncited，不计入决策。"""
    obj, status = _extract_json(text)
    if obj is None:
        return {"status": status, "error": "红队回复不可解析", "answers": [],
                "verdict": None, "reason": "", "uncited": []}
    answers = []
    uncited = []
    for index in range(1, 6):
        answer = obj.get("q%d" % index)
        answer = "" if answer is None else str(answer)
        answers.append({"q": index, "answer": answer,
                        "cited": _has_citation(answer)})
        if not _has_citation(answer):
            uncited.append(index)
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "reject"):
        verdict = None
    return {"status": "ok", "error": None, "answers": answers,
            "verdict": verdict, "reason": str(obj.get("reason", "") or "")[:300],
            "uncited": uncited}


def _has_citation(answer):
    """有没有可核对的具体引用。只写“无法判定”不算引用，不计入决策。"""
    text = str(answer or "")
    if not text:
        return False
    tokens = ("action", "params", "files", "bytes", "ledger", "round", "arm",
              "signature", "checkpoint", "hypothesis", "第", "行", "字段", "账本")
    return any(token in text for token in tokens)


def red_team_gate(model_review, deterministic):
    """合并“模型红队”与“本地确定性红队”。本地判定优先；无引用的模型结论不计入。"""
    reasons = ["%s: %s" % (item["q"], item["citation"])
               for item in deterministic if not item["ok"]]
    if model_review and model_review.get("status") == "ok":
        if model_review.get("verdict") == "reject" and len(model_review.get("uncited", [])) < 5:
            reasons.append("模型红队 reject：%s" % model_review.get("reason", "")[:120])
    return {"ok": not reasons, "reasons": reasons,
            "model": model_review, "deterministic": deterministic}


def state_gate(arm, context):
    """状态型闸门：这里只拦“剩余 1 轮”与“签名未见”。

    冷却与失败组合由悲观界 / 红队处理：它们需要读臂统计，和分数界在同一处判断，
    避免两套规则给出两个不同的 policy 名。
    """
    if arm == "PARENT":
        return True, None, "PARENT 永远允许"
    remaining = int(context.get("remaining_rounds", TOTAL_ROUNDS))
    if remaining <= 1:
        return False, "final_round", "remaining_rounds=%d" % remaining
    if not context.get("signature_seen", True):
        return False, "unseen_signature", "数据签名未出现在已见签名表里"
    return True, None, "ok"


def guard_hypothesis(hypothesis, delta):
    """确定性校验：定性判断与平台差值方向相反时替换并告警。"""
    text = str(hypothesis or "")
    if delta is None:
        return text, None
    better = any(word in text for word in BETTER_WORDS)
    worse = any(word in text for word in WORSE_WORDS)
    if better and delta < 0:
        return ("模型称提升，但平台差值 Δ=%+.6g<0，已按纪律置为中性" % delta), \
               "hypothesis 声称提升，实际 Δ=%+.6g<0，已替换" % delta
    if worse and delta > 0:
        return ("模型称变差，但平台差值 Δ=%+.6g>0，已按纪律置为中性" % delta), \
               "hypothesis 声称变差，实际 Δ=%+.6g>0，已替换" % delta
    return text, None


def _walk_budget(cand_dir):
    files, total = 0, 0
    names = []
    for dirpath, _, filenames in os.walk(cand_dir):
        for name in filenames:
            files += 1
            total += os.path.getsize(os.path.join(dirpath, name))
            names.append(os.path.relpath(os.path.join(dirpath, name), cand_dir))
    return {"files": files, "bytes": total, "names": sorted(names)}


OFFICIAL_KEYS = ("w1", "b1", "w2", "b2", "mean", "scale")


def _check_official_checkpoint(archive, errors, facts, expected_features,
                               parent_path=None, arm=None):
    """官方格式父检查点：缺 task/coef 不算错（父检查点是平台给的，不是我们编的）。

    仍然执行本函数负责的其他校验：形状、有限性、标准化常量、任务推断与特征维。
    体积/文件数已在调用方校验过。给定 parent_path 时追加第 5b 项：权重相对
    父产物的拼接 rel_l2 不得超过 train_mlp.weight_rel_cap(arm)——A 线用
    STEP_DECAY 求和得到的计划累计步幅预算，其余线维持 0.1；官方格式没有
    coef，原第 5 项恒等校验整段跳过，本项补上这个空档。
    """
    import numpy as np

    facts["format"] = "official"
    facts["action"] = "identity"
    facts["params"] = {}
    try:
        facts["action_kind"] = actions.prepare("identity", {})["kind"]
    except actions.ActionError as exc:
        errors.append("动作门禁失败：%s" % exc)
        facts["action_kind"] = "invalid"

    w1 = np.asarray(archive["w1"], dtype=float)
    w2 = np.asarray(archive["w2"], dtype=float)
    mean = np.asarray(archive["mean"], dtype=float).reshape(-1)
    scale = np.asarray(archive["scale"], dtype=float).reshape(-1)
    b1 = np.asarray(archive["b1"], dtype=float).reshape(-1)
    b2 = np.asarray(archive["b2"], dtype=float).reshape(-1)

    if w1.ndim != 2 or w2.ndim != 2:
        errors.append("官方 w1/w2 必须是二维矩阵，实际 %r/%r" % (w1.shape, w2.shape))
        facts["feature_dim"] = int(w1.shape[0]) if w1.ndim == 2 else None
        return {"ok": not errors, "errors": errors, "facts": facts}

    facts["feature_dim"] = int(w1.shape[0])
    if w1.shape[1] != w2.shape[0]:
        errors.append("官方 w1 输出维 %d 与 w2 输入维 %d 不一致"
                      % (w1.shape[1], w2.shape[0]))
    if b1.shape != (w1.shape[1],):
        errors.append("官方 b1 形状应为 (%d,)，实际 %r" % (w1.shape[1], b1.shape))
    if b2.shape != (w2.shape[1],):
        errors.append("官方 b2 形状应为 (%d,)，实际 %r" % (w2.shape[1], b2.shape))
    if mean.shape != (w1.shape[0],) or scale.shape != (w1.shape[0],):
        errors.append("官方 mean/scale 维度应为 (%d,)，实际 %r/%r"
                      % (w1.shape[0], mean.shape, scale.shape))
    elif np.any(scale == 0.0):
        errors.append("官方 scale 含 0")
    for name, array in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                        ("mean", mean), ("scale", scale)):
        if not np.all(np.isfinite(array)):
            errors.append("官方 %s 含非有限值" % name)

    task = "classification" if int(w2.shape[1]) > 1 else "regression"
    meta = {}
    if "metadata" in archive.files:
        try:
            meta = json.loads(str(np.asarray(archive["metadata"]).tolist()))
        except (ValueError, TypeError) as exc:
            errors.append("官方 metadata 不是合法 JSON：%s" % exc)
            meta = {}
        if not isinstance(meta, dict):
            errors.append("官方 metadata 必须是 JSON 对象")
            meta = {}
    declared_task = meta.get("task")
    if declared_task is not None:
        if declared_task not in ("classification", "regression"):
            errors.append("官方 metadata.task=%r 不是 classification/regression" % (declared_task,))
        else:
            task = declared_task
    facts["task"] = task

    # classes 只在分类下有意义：运行装载器 `contest_submission._load_official` 也只在
    # task == "classification" 时读 metadata.classes，回归分支完全不看它、只要求
    # w2 输出维为 1。校验层按同一分支口径，否则回归父产物声明的 `classes: []`
    # 会被算成“0 项 ≠ w2 列数 1”而误拒——同一份产物运行层放行、门禁层拒绝，是门禁比运行更严的 bug。
    # 回归父产物若仍给了非空 classes，与运行层一致地忽略（它不参与装载与输出），不额外报错。
    declared = meta.get("classes")
    if task == "classification":
        if isinstance(declared, list):
            facts["classes_source"] = "metadata.classes"
            if len(declared) != int(w2.shape[1]):
                errors.append("官方 metadata.classes 有 %d 项，与 w2 列数 %d 不一致"
                              % (len(declared), int(w2.shape[1])))
            if len(set(map(str, declared))) != len(declared):
                errors.append("官方 metadata.classes 有重复")
        else:
            facts["classes_source"] = "train.jsonl.sorted"
    else:
        facts["classes_source"] = "not_applicable"
        if int(w2.shape[1]) != 1:
            errors.append("官方回归 w2 输出维应为 1，实际 %d" % int(w2.shape[1]))

    paths = meta.get("paths")
    if isinstance(paths, list) and len(paths) != int(w1.shape[0]):
        errors.append("官方 metadata.paths 有 %d 项，与 w1 输入维 %d 不一致"
                      % (len(paths), int(w1.shape[0])))

    if expected_features is not None and facts.get("feature_dim") != int(expected_features):
        errors.append("特征维 %s 与数据 %s 不一致" % (facts.get("feature_dim"), expected_features))

    # 第 5b 项：权重漂移幅度上限（与 soup 同源口径；PARENT 逐字节副本 rel=0 必过）。
    # v3 预登记（2026-11-22 c-push）：C 线改用"预测位移"守卫，判定点前移到 train_mlp
    # 的 soup（那里有带标签的当轮切片，cage 不读数据拿不到标签）。所以 C 在这里只
    # **记录** rel_l2，不再据此报错；A/B 仍按 rel_l2 报错。
    is_c = str(arm or "").strip().upper() == "C"
    if parent_path and os.path.isfile(parent_path):
        try:
            import train_mlp
            with np.load(parent_path, allow_pickle=False) as par:
                if all(name in par.files for name in OFFICIAL_KEYS):
                    current = {name: np.asarray(archive[name], dtype=float)
                               for name in ("w1", "b1", "w2", "b2")}
                    base = {name: np.asarray(par[name], dtype=float)
                            for name in ("w1", "b1", "w2", "b2")}
                    rel = train_mlp.concat_rel_l2(current, base)
                    cap = train_mlp.weight_rel_cap(arm)
                    facts["weight_rel_l2"] = rel
                    facts["weight_rel_cap"] = cap
                    facts["weight_rel_cap_arm"] = str(arm or "")
                    facts["weight_rel_cap_enforced"] = not is_c
                    facts["weight_rel_cap_source"] = (
                        "planned_cumulative_step_budget"
                        if str(arm or "").strip().upper() == "A"
                        else ("pred_disp_at_soup" if is_c else "default_or_env"))
                    if (not is_c) and rel is not None and rel > cap:
                        errors.append("权重相对父产物拼接 L2=%.4f 超上限 %.4f"
                                      "（%s，%s）"
                                      % (rel, cap, facts["weight_rel_cap_arm"] or "—",
                                         facts["weight_rel_cap_source"]))
        except Exception as exc:  # noqa: BLE001 - 复核失败视同不过
            errors.append("父检查点权重距离复核失败：%s" % exc)
    return {"ok": not errors, "errors": errors, "facts": facts}


def check_candidate(cand_dir, expected_features=None, checkpoint_name="checkpoint.npz",
                    parent_path=None, arm=None):
    """候选性质测试：体积、契约、因果、恒等、维度。不读数据、不跑平台。"""
    errors = []
    facts = {}
    budget = _walk_budget(cand_dir)
    facts.update({"files": budget["files"], "bytes": budget["bytes"]})
    if budget["files"] > LIMIT_FILES:
        errors.append("文件数 %d 超过 %d" % (budget["files"], LIMIT_FILES))
    if budget["bytes"] > LIMIT_BYTES:
        errors.append("体积 %d 超过 %d" % (budget["bytes"], LIMIT_BYTES))

    path = os.path.join(cand_dir, checkpoint_name)
    if not os.path.isfile(path):
        errors.append("缺少检查点 %s" % checkpoint_name)
        return {"ok": False, "errors": errors, "facts": facts}

    import numpy as np

    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        errors.append("检查点无法按 npz 读取：%s" % exc)
        return {"ok": False, "errors": errors, "facts": facts}

    with archive:
        if "task" not in archive.files:
            if all(name in archive.files for name in OFFICIAL_KEYS):
                return _check_official_checkpoint(archive, errors, facts,
                                                  expected_features, parent_path, arm)
            errors.append("检查点既缺 task，也不是官方格式（缺 %s）"
                          % "/".join(name for name in OFFICIAL_KEYS
                                     if name not in archive.files))
            return {"ok": False, "errors": errors, "facts": facts}
        task = str(np.asarray(archive["task"]).tolist())
        facts["task"] = task
        if "coef" not in archive.files:
            errors.append("检查点缺 coef（v2 契约）")
            return {"ok": False, "errors": errors, "facts": facts}
        coef = np.asarray(archive["coef"], dtype=float)
        facts["feature_dim"] = int(coef.shape[-1])
        if not np.all(np.isfinite(coef)):
            errors.append("coef 含非有限值")
        action = "identity"
        params = {}
        if "action" in archive.files:
            action = str(np.asarray(archive["action"]).tolist())
        if "action_params" in archive.files:
            try:
                params = json.loads(str(np.asarray(archive["action_params"]).tolist()))
            except ValueError as exc:
                errors.append("action_params 不是合法 JSON：%s" % exc)
                params = {}
    facts["action"] = action
    facts["params"] = params

    try:
        prepared = actions.prepare(action, params, stage="predict")
        facts["action_kind"] = prepared["kind"]
    except actions.ActionError as exc:
        errors.append("动作门禁失败：%s" % exc)
        facts["action_kind"] = "invalid"

    if expected_features is not None and facts.get("feature_dim") != int(expected_features):
        errors.append("特征维 %s 与数据 %s 不一致" % (facts.get("feature_dim"), expected_features))

    # 恒等：候选只改动作、不改权重；coef 必须与父检查点逐位相同。
    if parent_path and os.path.isfile(parent_path):
        try:
            with np.load(parent_path, allow_pickle=False) as parent:
                if "coef" in parent.files:
                    same = bool(np.array_equal(np.asarray(parent["coef"], dtype=float), coef))
                    facts["coef_identical_to_parent"] = same
                    if not same:
                        errors.append("coef 与父检查点不一致（疑似权重更新或转置）")
        except Exception as exc:
            errors.append("父检查点无法复核：%s" % exc)

    return {"ok": not errors, "errors": errors, "facts": facts}


def red_team(facts, proposal, context):
    """本地确定性红队五问。每问必须带引用；任何一项 ok=False 即不得提交。"""
    params = proposal.get("params") or {}
    failed = list(context.get("failed_combos") or [])
    combo = {"arm": proposal.get("arm"), "action": facts.get("action"), "params": params}
    repeated = [item for item in failed if item.get("arm") == combo["arm"]
                and item.get("action") == combo["action"]
                and (item.get("params") or {}) == combo["params"]]
    return [
        {
            "q": "是否用未来行",
            "ok": facts.get("action_kind") == "causal",
            "citation": "checkpoint.action=%s kind=%s（predict 侧只用当前与已见行）"
                        % (facts.get("action"), facts.get("action_kind")),
        },
        {
            "q": "是否触及 transductive",
            "ok": facts.get("action") in actions.causal_actions(),
            "citation": "checkpoint.action=%s；causal=%s；transductive=%s"
                        % (facts.get("action"),
                           ",".join(actions.causal_actions()),
                           ",".join(actions.transductive_actions())),
        },
        {
            "q": "体积是否越界",
            "ok": facts.get("files", 0) <= LIMIT_FILES and facts.get("bytes", 0) <= LIMIT_BYTES,
            "citation": "files=%s<=%d；bytes=%s<=%d"
                        % (facts.get("files"), LIMIT_FILES, facts.get("bytes"), LIMIT_BYTES),
        },
        {
            "q": "参数是否越界",
            "ok": all(PARAM_RANGES[name][0] <= float(value) <= PARAM_RANGES[name][1]
                      for name, value in params.items()),
            "citation": "params=%s；区间=%s"
                        % (json.dumps(params, sort_keys=True),
                           json.dumps({k: list(v) for k, v in PARAM_RANGES.items()}, sort_keys=True)),
        },
        {
            "q": "是否重复失败组合",
            "ok": not repeated,
            "citation": ("无相同 (arm, action, params) 失败记录" if not repeated
                         else "重复失败：%s" % json.dumps(repeated[:2], ensure_ascii=False)),
        },
    ]


def run_build_script(script, sandbox, timeout=SCRIPT_TIMEOUT, env_extra=None):
    """进程级沙盒：脚本只能写 sandbox，stdout/stderr 截断，超时即杀。

    注意：这不是安全沙盒，只保证“失败不伤回合”。脚本写出的候选仍要过 `check_candidate`。
    """
    os.makedirs(sandbox, exist_ok=True)
    script_path = os.path.join(sandbox, "build.py")
    with open(script_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(script)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1",
           "CONTEST_SANDBOX_DIR": os.path.abspath(sandbox)}
    for key, value in (env_extra or {}).items():
        env[str(key)] = str(value)
    try:
        proc = subprocess.run(
            [sys.executable, "-B", script_path],
            cwd=sandbox,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "脚本超时（%.0fs）" % timeout, "stdout": "", "stderr": ""}
    out = proc.stdout.decode("utf-8", "replace")[:SCRIPT_OUTPUT_CAP]
    err = proc.stderr.decode("utf-8", "replace")[:SCRIPT_OUTPUT_CAP]
    if proc.returncode != 0:
        return {"ok": False, "error": "脚本退出码 %d" % proc.returncode,
                "stdout": out, "stderr": err}
    return {"ok": True, "error": None, "stdout": out, "stderr": err}

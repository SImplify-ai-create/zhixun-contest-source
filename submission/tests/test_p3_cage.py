# -*- coding: utf-8 -*-
"""v2 P3 验收：LLM 提案笼子（`llm_cage.py`）与每轮提案通道的接线。

对应七件事，逐件留证据：

1. 提案模板：动作只从闭集选，params 区间在本侧校验，硬约束与输出格式放首尾。
2. 状态卡：臂后验由代码算，两次生成逐字节一致，不含模型叙事。
3. 外部验证器：候选先过性质测试（体积/契约/因果/恒等/维度）才允许提交。
4. 红队五问：每问带引用；无引用的模型结论不计入决策。
5. 数字与叙事分离：定性句与 Δ 方向冲突时替换并告警。
6. 分阶段：前 6 轮拒绝 build_script，后 4 轮放行且必须过沙盒 + 性质测试。
7. 离线评测：见 `2026-11-18/llm-cage/outputs/eval_offline.py`（本文件只验代码侧）。
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(_HERE), "app")
for _path in (APP, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import actions  # noqa: E402
import llm_cage  # noqa: E402
import train_round  # noqa: E402
from test_p2 import FEATURES, FakeRnd, _run, _write_data, _write_parent, _write_v2_parent  # noqa: E402


# ---------------------------------------------------------------- 夹具

def _context(**overrides):
    ctx = {
        "arms": {},
        "history_ids": ["h1"],
        "phase": "closed",
        "remaining_rounds": 8,
        "signature_seen": True,
        "failed_combos": [],
    }
    ctx.update(overrides)
    return ctx


def _proposal(**overrides):
    body = {
        "arm": "C",
        "params": {},
        "hypothesis": "账本 h1 显示漂移未复现，先验证不归一化的线性头",
        "kill_condition": "若平台分低于父候选则回退",
        "confidence": "medium",
    }
    body.update(overrides)
    return body


def _text(**overrides):
    return json.dumps(_proposal(**overrides), ensure_ascii=False)


def _write_checkpoint(root, coef=None, task="regression", action=None, params=None):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "checkpoint.npz")
    fields = {
        "task": np.array(task),
        "coef": np.linspace(0.1, 1.0, len(FEATURES)) if coef is None else np.asarray(coef, dtype=float),
        "feature_order": np.array(FEATURES),
    }
    if action:
        fields["action"] = np.array(action)
    if params is not None:
        fields["action_params"] = np.array(json.dumps(params, sort_keys=True))
    np.savez(path, **fields)
    return path


def _card(arms=None):
    return llm_cage.status_card(
        3, "ds-1", "accuracy", arms or {"A": {"n": 2, "mean": 0.61, "m2": 0.02}},
        [{"signature": "ds-1/4", "arm": "A", "delta": 0.01,
          "verdict": "model_proposal", "ledger_id": "round-2"}],
        "closed", True)


# ---------------------------------------------------------------- 1. 模板

class TemplateTest(unittest.TestCase):
    def test_proposal_template_head_and_tail(self):
        text = llm_cage.PROPOSAL_TEMPLATE
        head, tail = text.split("状态", 1)
        self.assertIn("硬约束", head, "硬约束必须在首部")
        self.assertIn("transductive", head)
        self.assertIn("参数必须落在给定区间", head)
        self.assertIn("输出格式", tail, "输出格式必须在尾部")
        self.assertIn('"arm":"..."', tail)
        for line in ("不得使用任何序号大于当前行的信息", "逐条自查后再输出"):
            self.assertIn(line, text, "§3 原文句子不得改写：%s" % line)

    def test_red_team_template_five_questions_verbatim(self):
        text = llm_cage.RED_TEAM_TEMPLATE
        for index, keyword in enumerate(("未来行", "transductive", "体积", "参数", "失败组合"), 1):
            self.assertIn("%d) " % index, text)
            self.assertIn(keyword, text)
        self.assertIn('"verdict":"pass|reject"', text)
        self.assertIn("无法引用就写", text)

    def test_build_messages_substitutes_frozen_and_ranges(self):
        messages = llm_cage.build_messages(
            _card(), {"n_train": 120}, {"kl": 0.3}, {"candidate": "A"},
            3, [{"signature": "ds-1/4", "arm": "B", "params": {"eps": 1e-8}}])
        system = messages[0]["content"]
        for name in actions.transductive_actions():
            self.assertIn(name, system)
        self.assertIn("alpha", system)
        self.assertIn("PARENT / A / B / C", system)
        self.assertIn("第 3 轮 / 共 10 轮", system)
        self.assertIn("ds-1/4×B×", system, "失败组合必须带签名")
        self.assertNotIn("{冻结清单}", system)
        self.assertNotIn("{区间表}", system)
        self.assertIn("<untrusted_data>", messages[1]["content"])

    def test_repair_message_carries_original_and_errors(self):
        messages = llm_cage.repair_messages(
            _card(), "原始回复", ["arm=D 不在闭集"], {}, {}, {}, 3, [])
        blob = json.dumps(messages, ensure_ascii=False)
        self.assertIn("原始回复", blob)
        self.assertIn("arm=D 不在闭集", blob)


# ---------------------------------------------------------------- 2. 闭集 / 区间 / 引用

class ProposalGateTest(unittest.TestCase):
    def test_unknown_arm_rejected(self):
        with self.assertRaises(llm_cage.CageError):
            llm_cage.validate_proposal(
                llm_cage.parse_proposal(_text(arm="D"))[0], _context())

    def test_transductive_action_name_rejected(self):
        result = llm_cage.review(_text(arm="per_cond_mono_norm"), _context())
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("闭集" in item for item in result["errors"]))

    def test_alpha_out_of_range_rejected(self):
        result = llm_cage.review(_text(arm="A", params={"alpha": 2.0}), _context())
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("越界" in item for item in result["errors"]))
        self.assertEqual(result["parse_status"], "ok", "语法合法但取值越界：解析状态仍应为 ok")
        self.assertEqual(result["status"], "invalid", "取值越界必须与解析失败分开统计")

    def test_unknown_param_rejected(self):
        result = llm_cage.review(_text(arm="C", params={"alpha": 0.4}), _context())
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("不接受参数" in item for item in result["errors"]))

    def test_identity_takes_no_params(self):
        accepted = llm_cage.validate_proposal(llm_cage.parse_proposal(_text(arm="C"))[0],
                                              _context())
        self.assertEqual(accepted["params"], {})
        head = actions.prepare("identity", accepted["params"], stage="predict")
        self.assertEqual(head["params"], {}, "笼子放行的参数必须能被 actions 接受")

    def test_defaults_fill_when_omitted(self):
        accepted = llm_cage.validate_proposal(
            llm_cage.parse_proposal(_text(arm="A"))[0], _context())
        self.assertEqual(accepted["params"], {"alpha": 0.5, "eps": 1e-8})

    def test_hypothesis_digits_rejected(self):
        result = llm_cage.review(
            _text(hypothesis="账本 h1 的分数提升 0.02，故选 C"), _context())
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("数字" in item for item in result["errors"]))

    def test_hypothesis_must_cite_ledger(self):
        result = llm_cage.review(
            _text(hypothesis="凭直觉选不归一化的线性头"), _context())
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("引用" in item for item in result["errors"]))
        empty = llm_cage.review(
            _text(hypothesis="账本为空，先验证线性头"), _context(history_ids=[]))
        self.assertIsNotNone(empty["proposal"], "账本为空时允许用“账本为空”自证")

    def test_kill_condition_and_confidence_required(self):
        self.assertIsNone(llm_cage.review(_text(kill_condition=""), _context())["proposal"])
        self.assertIsNone(llm_cage.review(_text(confidence="very-high"), _context())["proposal"])

    def test_empty_and_unusable_status(self):
        self.assertEqual(llm_cage.review("", _context())["status"], "empty")
        self.assertEqual(llm_cage.review("我觉得可以", _context())["status"], "unusable")

    def test_build_script_blocked_before_round_seven(self):
        result = llm_cage.review(_text(build_script="print(1)"), _context(phase="closed"))
        self.assertIsNone(result["proposal"])
        self.assertTrue(any("build_script" in item for item in result["errors"]))

    def test_build_script_allowed_in_script_phase(self):
        accepted = llm_cage.validate_proposal(
            llm_cage.parse_proposal(_text(build_script="print(1)"))[0],
            _context(phase="script"))
        self.assertEqual(accepted["build_script"], "print(1)")
        self.assertIsNone(llm_cage.review(
            _text(build_script="```python\nprint(1)\n```"), _context(phase="script"))["proposal"])


# ---------------------------------------------------------------- 3. 状态闸门

class StateGateTest(unittest.TestCase):
    def test_final_round_blocks(self):
        allowed, reason, _ = llm_cage.state_gate("A", _context(remaining_rounds=1))
        self.assertFalse(allowed)
        self.assertEqual(reason, "final_round")

    def test_unseen_signature_blocks(self):
        allowed, reason, _ = llm_cage.state_gate("B", _context(signature_seen=False))
        self.assertFalse(allowed)
        self.assertEqual(reason, "unseen_signature")

    def test_parent_always_allowed(self):
        allowed, _, _ = llm_cage.state_gate(
            "PARENT", _context(remaining_rounds=1, signature_seen=False))
        self.assertTrue(allowed)


# ---------------------------------------------------------------- 4. 状态卡

class StatusCardTest(unittest.TestCase):
    def test_card_is_deterministic_and_data_only(self):
        arms = {"PARENT": {"n": 3, "mean": 0.5, "m2": 0.02},
                "A": {"n": 2, "mean": 0.61, "m2": 0.01, "cooldown": 1}}
        first = _card(arms)
        second = _card(arms)
        self.assertEqual(first, second, "同一输入两次生成必须逐字节一致")
        self.assertIn("均值=0.6100", first)
        self.assertIn("标准差=0.1000", first)
        self.assertIn("次数=2", first)
        self.assertIn("冷却=1", first)
        self.assertIn("剩余 8 轮", first)
        for word in ("经验", "学到了", "总结"):
            self.assertNotIn(word, first, "状态卡只放数据，不放叙事")
        self.assertLess(len(first), 1200, "状态卡必须控制在数百 token")

    def test_card_ignores_unobserved_arm_narrative(self):
        card = _card({"A": {}})
        self.assertIn("A: 次数=0（未观测）", card)


# ---------------------------------------------------------------- 5. 叙事校验

class NarrativeGuardTest(unittest.TestCase):
    def test_overclaim_replaced(self):
        text, warning = llm_cage.guard_hypothesis("该动作能提升准确率", -0.02)
        self.assertIn("Δ=-0.02", text)
        self.assertIsNotNone(warning)
        self.assertIn("已替换", warning)

    def test_consistent_claim_kept(self):
        text, warning = llm_cage.guard_hypothesis("该动作能提升准确率", 0.02)
        self.assertEqual(text, "该动作能提升准确率")
        self.assertIsNone(warning)

    def test_no_delta_is_neutral(self):
        text, warning = llm_cage.guard_hypothesis("该动作能提升准确率", None)
        self.assertEqual(text, "该动作能提升准确率")
        self.assertIsNone(warning)

    def test_underclaim_replaced(self):
        text, warning = llm_cage.guard_hypothesis("该动作会变差", 0.03)
        self.assertIn("Δ=+0.03", text)
        self.assertIsNotNone(warning)


# ---------------------------------------------------------------- 6. 候选性质测试

class CandidateCheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_identity_candidate_passes(self):
        cand = os.path.join(self.tmp, "cand")
        path = _write_checkpoint(cand)
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES),
                                          parent_path=path)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["files"], 1)
        self.assertEqual(result["facts"]["action_kind"], "causal")
        self.assertTrue(result["facts"]["coef_identical_to_parent"])
        self.assertNotIn("dir", result["facts"], "facts 不得含绝对路径（确定性要求）")

    def test_transductive_candidate_rejected(self):
        cand = os.path.join(self.tmp, "cand")
        _write_checkpoint(cand, action="batch_zscore", params={"eps": 1e-8})
        result = llm_cage.check_candidate(cand)
        self.assertFalse(result["ok"])
        self.assertTrue(any("动作门禁" in item for item in result["errors"]))

    def test_coef_mismatch_rejected(self):
        parent = os.path.join(self.tmp, "parent")
        parent_path = _write_checkpoint(parent)
        cand = os.path.join(self.tmp, "cand")
        _write_checkpoint(cand, coef=np.zeros(len(FEATURES)))
        result = llm_cage.check_candidate(cand, parent_path=parent_path)
        self.assertFalse(result["ok"])
        self.assertTrue(any("coef" in item for item in result["errors"]))

    def test_missing_checkpoint_rejected(self):
        cand = os.path.join(self.tmp, "empty")
        os.makedirs(cand)
        result = llm_cage.check_candidate(cand)
        self.assertFalse(result["ok"])
        self.assertTrue(any("缺少检查点" in item for item in result["errors"]))

    def test_file_budget_enforced(self):
        cand = os.path.join(self.tmp, "many")
        os.makedirs(cand)
        for index in range(65):
            with open(os.path.join(cand, "f%02d.bin" % index), "wb") as handle:
                handle.write(b"x")
        result = llm_cage.check_candidate(cand)
        self.assertFalse(result["ok"])
        self.assertTrue(any("文件数" in item for item in result["errors"]))

    # --- 官方格式父检查点：缺 task/coef 不算错，但其他校验项照旧 ---

    def _official(self, root, **overrides):
        os.makedirs(root, exist_ok=True)
        n = len(FEATURES)
        fields = {
            "w1": np.zeros((n, 3)),
            "b1": np.zeros(3),
            "w2": np.zeros((3, 1)),
            "b2": np.zeros(1),
            "mean": np.zeros(n),
            "scale": np.ones(n),
            "metadata": np.array(json.dumps({"task": "regression"})),
        }
        fields.update(overrides)
        path = os.path.join(root, "checkpoint.npz")
        np.savez(path, **fields)
        return path

    def test_official_parent_checkpoint_passes(self):
        cand = os.path.join(self.tmp, "official")
        path = self._official(cand)
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES),
                                          parent_path=path)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["format"], "official")
        self.assertEqual(result["facts"]["action"], "identity")
        self.assertEqual(result["facts"]["action_kind"], "causal")
        self.assertEqual(result["facts"]["feature_dim"], len(FEATURES))
        self.assertNotIn("coef_identical_to_parent", result["facts"],
                         "官方格式没有 coef，恒等检查不适用")

    def test_official_checkpoint_still_checks_shapes(self):
        cand = os.path.join(self.tmp, "official-bad")
        self._official(cand, mean=np.zeros(len(FEATURES) + 1))
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES))
        self.assertFalse(result["ok"])
        self.assertTrue(any("mean/scale" in item for item in result["errors"]))

    def test_official_checkpoint_checks_expected_features(self):
        cand = os.path.join(self.tmp, "official-dim")
        self._official(cand)
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES) + 1)
        self.assertFalse(result["ok"])
        self.assertTrue(any("特征维" in item for item in result["errors"]))

    def test_official_classes_must_match_w2_columns(self):
        cand = os.path.join(self.tmp, "official-classes")
        meta = {"task": "classification", "classes": ["a", "b"]}
        self._official(cand, w2=np.zeros((3, 3)), b2=np.zeros(3),
                       metadata=np.array(json.dumps(meta)))
        result = llm_cage.check_candidate(cand)
        self.assertFalse(result["ok"])
        self.assertTrue(any("classes" in item for item in result["errors"]))

    def test_official_regression_with_empty_classes_passes(self):
        """回归父产物声明 classes=[]：只校验 w2 输出维为 1，不按分类口径比长度。"""
        cand = os.path.join(self.tmp, "official-regression")
        meta = {"task": "regression", "classes": [],
                "target_mean": 10.0, "target_scale": 2.0}
        self._official(cand, metadata=np.array(json.dumps(meta)))
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES))
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["task"], "regression")
        self.assertEqual(result["facts"]["classes_source"], "not_applicable")

    def test_official_classification_with_classes_still_passes(self):
        """对照：分类父产物带正确 classes 仍按原口径通过，长度校验没有被放松。"""
        cand = os.path.join(self.tmp, "official-classification")
        meta = {"task": "classification", "classes": ["x", "y", "z"]}
        self._official(cand, w2=np.zeros((3, 3)), b2=np.zeros(3),
                       metadata=np.array(json.dumps(meta)))
        result = llm_cage.check_candidate(cand, expected_features=len(FEATURES))
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["classes_source"], "metadata.classes")

    def test_missing_both_formats_still_rejected(self):
        cand = os.path.join(self.tmp, "neither")
        os.makedirs(cand)
        np.savez(os.path.join(cand, "checkpoint.npz"), w1=np.zeros((4, 2)))
        result = llm_cage.check_candidate(cand)
        self.assertFalse(result["ok"])
        self.assertTrue(any("既缺 task" in item for item in result["errors"]))


# ---------------------------------------------------------------- 7. 红队五问

class RedTeamTest(unittest.TestCase):
    def _facts(self, **overrides):
        facts = {"action": "streaming_ema_norm", "action_kind": "causal",
                 "params": {"alpha": 0.5, "eps": 1e-8},
                 "files": 1, "bytes": 1024}
        facts.update(overrides)
        return facts

    def test_each_question_has_citation(self):
        questions = llm_cage.red_team(
            self._facts(), {"arm": "A", "params": {"alpha": 0.5, "eps": 1e-8}},
            _context())
        self.assertEqual(len(questions), 5)
        for item in questions:
            self.assertTrue(item["ok"])
            self.assertTrue(item["citation"])
            self.assertTrue(llm_cage._has_citation(item["citation"]))

    def test_transductive_candidate_flagged(self):
        questions = llm_cage.red_team(
            self._facts(action="per_cond_mono_norm", action_kind="transductive"),
            {"arm": "A", "params": {}}, _context())
        self.assertFalse(questions[0]["ok"])
        self.assertFalse(questions[1]["ok"])

    def test_repeated_failure_combo_flagged(self):
        failed = [{"signature": "ds-1/4", "arm": "B",
                   "action": "per_cond_seq", "params": {"eps": 1e-8}}]
        questions = llm_cage.red_team(
            self._facts(action="per_cond_seq", params={"eps": 1e-8}),
            {"arm": "B", "params": {"eps": 1e-8}}, _context(failed_combos=failed))
        self.assertFalse(questions[4]["ok"])
        self.assertIn("重复失败", questions[4]["citation"])

    def test_uncited_model_verdict_ignored(self):
        model_review = {"status": "ok", "verdict": "reject", "reason": "感觉不对",
                        "uncited": [1, 2, 3, 4, 5]}
        deterministic = llm_cage.red_team(
            self._facts(), {"arm": "A", "params": {"alpha": 0.5, "eps": 1e-8}},
            _context())
        decision = llm_cage.red_team_gate(model_review, deterministic)
        self.assertTrue(decision["ok"], "五问全无引用时，模型 reject 不计入决策")

    def test_cited_model_reject_blocks(self):
        model_review = {"status": "ok", "verdict": "reject", "reason": "第 3 问越界",
                        "uncited": [2]}
        deterministic = llm_cage.red_team(
            self._facts(), {"arm": "A", "params": {"alpha": 0.5, "eps": 1e-8}},
            _context())
        decision = llm_cage.red_team_gate(model_review, deterministic)
        self.assertFalse(decision["ok"])

    def test_deterministic_failure_always_blocks(self):
        deterministic = [{"q": "体积是否越界", "ok": False, "citation": "files=65"}]
        self.assertFalse(llm_cage.red_team_gate(None, deterministic)["ok"])

    def test_parse_red_team_marks_citations(self):
        review = llm_cage.parse_red_team(json.dumps({
            "q1": "checkpoint.action=causal", "q2": "对照 action 动作库",
            "q3": "files=1", "q4": "params.alpha=0.5", "q5": "账本无记录",
            "verdict": "pass", "reason": "无风险"}))
        self.assertEqual(review["status"], "ok")
        self.assertEqual(review["uncited"], [])
        self.assertEqual(review["verdict"], "pass")

    def test_cannot_judge_is_not_a_citation(self):
        review = llm_cage.parse_red_team(json.dumps({
            "q1": "无法判定", "q2": "无法判定", "q3": "无法判定",
            "q4": "无法判定", "q5": "无法判定",
            "verdict": "reject", "reason": "依据不足"}))
        self.assertEqual(review["uncited"], [1, 2, 3, 4, 5])
        deterministic = llm_cage.red_team(
            self._facts(), {"arm": "A", "params": {"alpha": 0.5, "eps": 1e-8}},
            _context())
        self.assertTrue(llm_cage.red_team_gate(review, deterministic)["ok"])


# ---------------------------------------------------------------- 8. 沙盒

class SandboxTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_script_runs_and_writes_sandbox(self):
        result = llm_cage.run_build_script(
            "import os\nopen(os.path.join(os.environ['CONTEST_SANDBOX_DIR'], 'x.txt'), 'w')"
            ".write('ok')\nprint('done')", self.tmp)
        self.assertTrue(result["ok"], result)
        self.assertIn("done", result["stdout"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "x.txt")))

    def test_nonzero_exit_fails(self):
        result = llm_cage.run_build_script("raise SystemExit(3)", self.tmp)
        self.assertFalse(result["ok"])
        self.assertIn("3", result["error"])

    def test_timeout_fails(self):
        result = llm_cage.run_build_script("import time\ntime.sleep(5)", self.tmp, timeout=0.3)
        self.assertFalse(result["ok"])
        self.assertIn("超时", result["error"])


# ---------------------------------------------------------------- 9. 每轮接线

class SeqRnd(FakeRnd):
    """按顺序吐多条回复的 rnd 替身（提案 → 修复 → 红队）。"""

    def __init__(self, *args, replies=(), **kwargs):
        super(SeqRnd, self).__init__(*args, **kwargs)
        self.replies = list(replies)

    def model(self, messages, max_tokens=None, temperature=None):
        if self.replies:
            self.reply = self.replies.pop(0)
        return super(SeqRnd, self).model(messages, max_tokens, temperature)


class InterruptRnd(FakeRnd):
    """首次 submit_candidate 抛指定异常，之后恢复正常（用来区分中断型与普通错误）。"""

    def __init__(self, *args, **kwargs):
        self.submit_error = kwargs.pop("submit_error", None)
        super(InterruptRnd, self).__init__(*args, **kwargs)

    def submit_candidate(self, candidate_dir, **kwargs):
        if self.submit_error is not None:
            exc = self.submit_error
            self.submit_error = None
            work = os.path.dirname(os.path.dirname(candidate_dir))
            self.rounds_at_submit = sorted(
                d for d in os.listdir(work) if d.startswith("round-"))
            self.submits.append({"candidate_dir": candidate_dir, "kwargs": kwargs})
            raise exc
        return super(InterruptRnd, self).submit_candidate(candidate_dir, **kwargs)


class RPCOutcomeUnknown(Exception):
    """按名字/消息模拟 SDK 的中断型错误。"""


def _run_capture(rnd, **kwargs):
    """跑一轮；即使整轮抛错，也把 stderr 与异常一起收回来。"""
    kwargs.setdefault("work_dir", rnd.work_root)
    buffer = io.StringIO()
    ledger = None
    error = None
    with contextlib.redirect_stderr(buffer):
        try:
            ledger = train_round.run(rnd, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 测试要看清失败类型
            error = exc
    return ledger, buffer.getvalue(), error


class RoundWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.parent_root = os.path.join(cls.tmp, "parent")
        # 本类验收"提案 → 动作臂 → 提交"的接线，父产物必须是 v2 格式：
        # 官方 MLP 父产物没有 coef，动作臂会被 _build_candidate 明确拒绝并退回 PARENT。
        cls.parent_npz = _write_v2_parent(cls.parent_root)
        cls.data = os.path.join(cls.tmp, "data")
        _write_data(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _case(self, name, round_index=2, **kwargs):
        work = os.path.join(self.tmp, "work", name)
        os.makedirs(work, exist_ok=True)
        kwargs.setdefault("reply", _text(arm="C"))
        return SeqRnd(self.parent_root, work, data_dir=self.data,
                      round_index=round_index, **kwargs), work

    def test_out_of_range_params_never_reach_submit(self):
        rnd, _ = self._case("range", replies=[
            _text(arm="A", params={"alpha": 2.0}),
            _text(arm="A", params={"alpha": 2.0}),
        ])
        ledger, err = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["policy"], "fallback_rule", "越界提案必须退回规则选择器")
        self.assertIn("越界", err)
        self.assertEqual(len(rnd.submits), 1)
        cand = os.path.join(rnd.submits[0]["candidate_dir"], "checkpoint.npz")
        with np.load(cand, allow_pickle=False) as archive:
            self.assertEqual(json.loads(str(archive["action_params"])), {},
                             "提交的候选不得带模型给的越界参数")
    def test_once_repair_then_proposal_accepted(self):
        rnd, _ = self._case("repair", replies=[
            _text(arm="D"),
            _text(arm="C"),
        ])
        ledger, err = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "C")
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(len(rnd.calls), 2, "首提 + 一次修复，不得展开成多轮")
        self.assertIn("模型提案第1次被判无效", err)
        self.assertEqual(config["model"]["gate"]["calls"][0]["call_id"], "call-1")

    def test_build_script_blocked_in_closed_phase(self):
        script = "print('noop')"
        rnd, _ = self._case("closed-script", round_index=3, replies=[
            json.dumps(_proposal(arm="C", build_script=script), ensure_ascii=False),
            json.dumps(_proposal(arm="C", build_script=script), ensure_ascii=False),
        ])
        ledger, err = _run(rnd)
        self.assertEqual(ledger["configuration"]["policy"], "fallback_rule")
        self.assertIn("build_script", err)
        self.assertEqual(ledger["configuration"]["candidate"], "A")

    def test_build_script_allowed_in_script_phase(self):
        script = (
            "import os, shutil\n"
            "shutil.copyfile(os.environ['CONTEST_PARENT_CHECKPOINT'],\n"
            "                os.path.join(os.environ['CONTEST_CANDIDATE_DIR'], 'checkpoint.npz'))\n"
        )
        rnd, _ = self._case("script-phase", round_index=7, replies=[
            json.dumps(_proposal(arm="C", build_script=script), ensure_ascii=False),
            json.dumps({"q1": "checkpoint.action=identity", "q2": "对照动作库",
                        "q3": "files=1", "q4": "params 空", "q5": "账本无记录",
                        "verdict": "pass", "reason": "无风险"}, ensure_ascii=False),
        ])
        ledger, err = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(config["action"], "identity", "账本 action 必须取自检查点自述")
        self.assertEqual(config["model"]["phase"], "script")
        self.assertEqual(len(rnd.calls), 2, "提案 + 红队各一次")
        self.assertIn("red_team", config["model"]["gate"])
        self.assertTrue(config["model"]["gate"]["red_team"]["ok"])

    def test_narrative_guarded_after_receipt(self):
        # 平台回执 score=0.6123，父臂无观测 → delta 为 None，不得误判；这里只验字段落盘。
        rnd, _ = self._case("narrative")
        ledger, _ = _run(rnd)
        model = ledger["configuration"]["model"]
        self.assertIn("hypothesis_guarded", model)
        self.assertIn("delta_vs_parent", model)
        self.assertIsNone(model["guard_warning"])

    def test_round1_adopts_proposal_and_records_it(self):
        rnd, _ = self._case("r1", round_index=1)
        ledger, err = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "C", "第 1 轮不再强制父候选")
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(config["model"]["proposal"], "C")
        self.assertNotIn("R1 固定续训父检查点", err, "旧的自设规则必须已移除")
        self.assertIn("phase", config["model"]["gate"])

    def _interrupt_case(self, name, submit_error, round_index=2):
        work = os.path.join(self.tmp, "work", name)
        os.makedirs(work, exist_ok=True)
        rnd = InterruptRnd(self.parent_root, work, data_dir=self.data,
                           round_index=round_index, reply=_text(arm="C"),
                           submit_error=submit_error)
        return rnd, work

    def test_interruption_submits_once_and_does_not_retry(self):
        rnd, work = self._interrupt_case("interrupt", TimeoutError("rpc timed out"))
        ledger, err, exc = _run_capture(rnd)
        self.assertIsNone(ledger, "中断型必须整轮失败，不能产出账本")
        self.assertIsInstance(exc, train_round.TrainRoundError)
        self.assertEqual(len(rnd.submits), 1, "中断型只能调用一次 submit_candidate")
        rounds_after = sorted(d for d in os.listdir(work) if d.startswith("round-"))
        self.assertEqual(rounds_after, rnd.rounds_at_submit, "中断型不得再构造第二个候选")
        self.assertIn("中断型", err)
        self.assertIn("RPC_OUTCOME_UNKNOWN", err)
        self.assertIn("不再重试", err)
        self.assertEqual(rnd.finishes, [], "整轮失败后不得调用 finish")

    def test_rpc_outcome_unknown_named_error_is_interruption(self):
        rnd, work = self._interrupt_case(
            "outcome-unknown", RPCOutcomeUnknown("RPC_OUTCOME_UNKNOWN: interrupted"))
        ledger, err, exc = _run_capture(rnd)
        self.assertIsNone(ledger)
        self.assertIsInstance(exc, train_round.TrainRoundError)
        self.assertEqual(len(rnd.submits), 1, "名为 RPC_OUTCOME_UNKNOWN 的错误也不得重试")
        self.assertIn("中断型", err)

    def test_normal_submit_error_still_falls_back_to_parent(self):
        rnd, work = self._interrupt_case(
            "normal-fail", RuntimeError("platform rejected candidate"))
        ledger, err, exc = _run_capture(rnd)
        self.assertIsNone(exc, "普通平台错误不应整轮失败，应退回 PARENT 重提")
        self.assertEqual(len(rnd.submits), 2, "普通错误仍走一次 PARENT 重试")
        self.assertEqual(ledger["configuration"]["candidate"], "PARENT")
        self.assertEqual(ledger["configuration"]["policy"], "pessimism_fallback_parent")
        self.assertIn("退回父候选", err)
        self.assertNotIn("中断型", err)


if __name__ == "__main__":
    unittest.main()

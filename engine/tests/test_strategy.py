"""strategy.py 控制器工厂与规则控制器的单元测试。"""
import unittest

from src.strategy import RuleController, LLMController, build_controller


class TestBuildController(unittest.TestCase):
    def test_rule_by_name(self):
        self.assertIsInstance(build_controller("rule"), RuleController)

    def test_unknown_falls_back_to_rule(self):
        """未知名称默认回退规则控制器，主流程绝不因配置错误中断。"""
        self.assertIsInstance(build_controller("anything-else"), RuleController)

    def test_llm_by_name(self):
        self.assertIsInstance(build_controller("llm"), LLMController)

    def test_llm_case_insensitive(self):
        self.assertIsInstance(build_controller("LLM"), LLMController)


class TestRuleController(unittest.TestCase):
    def test_passthrough_when_budget_ok(self):
        out = RuleController().decide({"tier": "full", "labels_left": 2000})
        self.assertEqual(out["tier"], "full")

    def test_full_downgraded_when_labels_low(self):
        """预算护栏：剩余标注不足一次全量重训时 full 降级为 incremental。"""
        out = RuleController().decide({"tier": "full", "labels_left": 100})
        self.assertEqual(out["tier"], "incremental")
        self.assertIn("降级", out["rationale"])

    def test_missing_tier_defaults_observe(self):
        out = RuleController().decide({})
        self.assertEqual(out["tier"], "observe")

    def test_output_shape(self):
        """两条控制器输出契约一致：{tier, rationale}。"""
        out = RuleController().decide({"tier": "incremental"})
        self.assertEqual(set(out), {"tier", "rationale"})


if __name__ == "__main__":
    unittest.main()

"""能力开关注册表的结构、纯函数与留痕行为测试。

锁定三件事：
1. 覆盖齐全：16 个未启用检测器 + Kalman / GPR / Meeker / 动作标定 v2.0 / 在线学习 / 不确定性；
   每行五列 + "可判时点"（只允许两个取值），默认态一律 off。
2. 判定是纯函数：同输入同输出，不读时间/随机数，不改 evidence。
3. 缺证据时不硬判：返回 indeterminate 并保持默认关闭；trace 字段齐全。
"""
import copy
import json
import random
import tempfile
import time
import unittest
from pathlib import Path

from src import capability_registry as reg

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / "docs" / "capability_evidence.json").read_text(encoding="utf-8"))


def _verdicts(evidence):
    return {row["switch"]: row["verdict"] for row in reg.selector_trace_rows(evidence)}


class TestRegistryShape(unittest.TestCase):
    def test_covers_16_disabled_detectors(self):
        self.assertEqual(len(reg.DISABLED_DETECTORS), 16)
        names = {switch.name for switch in reg.REGISTRY}
        for detector in reg.DISABLED_DETECTORS:
            self.assertIn(f"detector:{detector}", names)

    def test_covers_named_capabilities(self):
        names = {switch.name for switch in reg.REGISTRY}
        for name in ("kalman", "gpr", "meeker", "action_calibration_v2",
                     "online_learning", "uncertainty_abstention"):
            self.assertIn(name, names)
        self.assertEqual(len(reg.REGISTRY), 22)

    def test_evaluable_at_vocabulary(self):
        for switch in reg.REGISTRY:
            self.assertIn(switch.evaluable_at, reg.EVAL_POINTS)

    def test_all_rows_default_off_and_complete(self):
        for switch in reg.REGISTRY:
            self.assertEqual(switch.default_state, reg.DEFAULT_OFF)
            self.assertTrue(switch.open_criterion)
            self.assertTrue(switch.kill_if)
            self.assertTrue(switch.calibration_status)
            self.assertTrue(switch.evidence_ref)
            self.assertTrue(callable(switch.judge))


class TestPurity(unittest.TestCase):
    def test_same_input_same_output(self):
        self.assertEqual(reg.selector_trace_rows(EVIDENCE), reg.selector_trace_rows(EVIDENCE))

    def test_no_time_or_random(self):
        original_time, original_random = time.time, random.random

        def _boom(*_args, **_kwargs):
            raise AssertionError("判定函数读取了时间或随机数")

        time.time = _boom
        random.random = _boom
        try:
            reg.selector_trace_rows(EVIDENCE)
        finally:
            time.time, random.random = original_time, original_random

    def test_evidence_not_mutated(self):
        before = copy.deepcopy(EVIDENCE)
        reg.selector_trace_rows(EVIDENCE)
        self.assertEqual(EVIDENCE, before)


class TestVerdicts(unittest.TestCase):
    def test_current_evidence(self):
        verdicts = _verdicts(EVIDENCE)
        for detector in reg.DISABLED_DETECTORS:
            self.assertEqual(verdicts[f"detector:{detector}"], reg.KEEP_DEFAULT)
        for name in ("kalman", "action_calibration_v2", "online_learning",
                     "uncertainty_abstention"):
            self.assertEqual(verdicts[name], reg.KEEP_DEFAULT)
        # 判据/数值未落盘：不硬判，维持默认关闭
        for name in ("gpr", "meeker"):
            self.assertEqual(verdicts[name], reg.INDETERMINATE)

    def test_empty_evidence_is_indeterminate(self):
        rows = reg.selector_trace_rows({})
        self.assertTrue(all(row["verdict"] == reg.INDETERMINATE for row in rows))

    def test_action_calibration_two_step_criterion(self):
        judge = reg.BY_NAME["action_calibration_v2"].judge
        evidence = copy.deepcopy(EVIDENCE)
        block = evidence["action_calibration_v2"]
        block["loo_hit"] = 0.65          # +17pp，第一步过
        block["out_of_table_fallback"] = 0.05
        self.assertEqual(judge(evidence).decision, reg.ALLOW_ON)
        block["out_of_table_fallback"] = 0.12
        self.assertEqual(judge(evidence).decision, reg.KEEP_DEFAULT)
        block["out_of_table_fallback"] = None
        self.assertEqual(judge(evidence).decision, reg.INDETERMINATE)

    def test_missing_second_block_does_not_fabricate(self):
        # 只给第一层证据：检测器族判据可判，Kalman 不可判
        evidence = {"detector_migration": {"auc_visible": 0.8093, "auc_unseen_worst": 0.8093}}
        verdicts = _verdicts(evidence)
        self.assertEqual(verdicts["detector:entropy_mean"], reg.ALLOW_ON)
        self.assertEqual(verdicts["kalman"], reg.INDETERMINATE)


class TestSelectorTrace(unittest.TestCase):
    def test_required_fields(self):
        required = {"event", "mode", "switch", "default_state", "verdict",
                    "criterion", "observed", "evidence_ref", "evaluable_at",
                    "calibration_status"}
        for row in reg.selector_trace_rows(EVIDENCE):
            self.assertTrue(required <= set(row))
            self.assertEqual(row["event"], "selector_trace")
            self.assertEqual(row["mode"], reg.SHADOW_MODE)
            self.assertIn(row["verdict"], reg.VERDICT_VALUES)

    def test_write_is_deterministic(self):
        rows = reg.selector_trace_rows(EVIDENCE)
        with tempfile.TemporaryDirectory() as tmp:
            first = reg.write_selector_trace(Path(tmp) / "a.jsonl", rows)
            second = reg.write_selector_trace(Path(tmp) / "b.jsonl", rows)
            self.assertEqual(first.read_text(encoding="utf-8"),
                             second.read_text(encoding="utf-8"))
            lines = first.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(reg.REGISTRY))
            self.assertEqual(json.loads(lines[0])["event"], "selector_trace")


if __name__ == "__main__":
    unittest.main()

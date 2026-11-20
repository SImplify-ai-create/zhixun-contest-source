"""交付合规测试：逐批预测落盘契约 + 离线镜像禁止外部模型调用。

红线：
  1. 评委离线评测时必须能从产物目录直接拿到"逐批预测结果文件"；
  2. 交付镜像内不得存在任何可被触发的外部模型调用开关（ZHIXUN_OFFLINE=1）。
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from src.advanced import PacedAdaptivePipeline
from src.drift import DriftDetector
from src.strategy import LLMController, RuleController, build_controller

_ROOT = Path(__file__).resolve().parent.parent


def _load_run_pipeline():
    spec = importlib.util.spec_from_file_location(
        "run_pipeline_mod", _ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPredictionDump(unittest.TestCase):
    def test_header_and_rows(self):
        mod = _load_run_pipeline()
        rows = [(3, 600, 5.545155, 5.157502), (3, 601, 4.6, 4.604757)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "predictions-x.csv"
            mod.write_predictions(rows, out)
            lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "batch_step,sample_index,y_true,y_pred")
        self.assertEqual(lines[1], "3,600,5.545155,5.157502")
        self.assertEqual(len(lines), 3)

    def test_non_numeric_label_kept_as_text(self):
        """分类任务的类别名不能因为格式化丢成数字。"""
        mod = _load_run_pipeline()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "predictions-y.csv"
            mod.write_predictions([(0, 0, "Ammonia", "Toluene")], out)
            body = out.read_text(encoding="utf-8").splitlines()[1]
        self.assertEqual(body, "0,0,Ammonia,Toluene")

    def test_pipeline_dump_default_off(self):
        """默认不落盘（不改存量行为），只有调用方显式置 [] 才收集。"""
        pipe = PacedAdaptivePipeline(DriftDetector(), None)
        self.assertIsNone(pipe.pred_dump)


class TestOfflineGuard(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("ZHIXUN_OFFLINE")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("ZHIXUN_OFFLINE", None)
        else:
            os.environ["ZHIXUN_OFFLINE"] = self._old

    def test_llm_blocked_in_offline_image(self):
        os.environ["ZHIXUN_OFFLINE"] = "1"
        self.assertIsInstance(build_controller("llm"), RuleController)

    def test_llm_available_outside_image(self):
        os.environ.pop("ZHIXUN_OFFLINE", None)
        self.assertIsInstance(build_controller("llm"), LLMController)


if __name__ == "__main__":
    unittest.main()

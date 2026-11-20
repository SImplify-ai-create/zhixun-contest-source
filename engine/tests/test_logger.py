"""logger.py 的官方列名映射（OFFICIAL_COLUMN_MAP）行为测试。

官方《实验记录规范》格式 11/9 前下发，届时只需改 logger.py 顶部的映射表。
本测试锁定映射机制本身的正确性：不改映射时列名原样，改映射后重命名/常量/表达式生效。
"""
import csv
import tempfile
import unittest
from pathlib import Path

import src.logger as logger_mod
from src.logger import ExperimentLogger


def _read_rows(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class TestOfficialColumnMap(unittest.TestCase):
    def tearDown(self):
        logger_mod.OFFICIAL_COLUMN_MAP = None  # 每个用例后还原，避免污染

    def test_no_map_passthrough(self):
        """映射为 None（默认）时列名与行内容原样输出。"""
        with tempfile.TemporaryDirectory() as d:
            lg = ExperimentLogger(out_dir=d, use_swanlab=False, experiment_name="t")
            lg.log({"eval_rmse": 0.19, "batch": 3})
            lg.close()
            rows = _read_rows(lg.csv_path)
            self.assertEqual(len(rows), 1)
            self.assertIn("eval_rmse", rows[0])
            self.assertIn("batch", rows[0])

    def test_rename_and_const(self):
        """字符串映射重命名，{"const"} 注入常量列。"""
        logger_mod.OFFICIAL_COLUMN_MAP = {
            "final_rmse": "eval_rmse",
            "run_id": {"const": "audit-001"},
        }
        with tempfile.TemporaryDirectory() as d:
            lg = ExperimentLogger(out_dir=d, use_swanlab=False, experiment_name="t")
            lg.log({"eval_rmse": 0.19})
            lg.close()
            rows = _read_rows(lg.csv_path)
            self.assertEqual(rows[0]["final_rmse"], "0.19")
            self.assertEqual(rows[0]["run_id"], "audit-001")
            # 原始列也保留（信息不丢）
            self.assertIn("eval_rmse", rows[0])

    def test_eval_expression(self):
        """{"eval"} 表达式可引用 row 与 metrics。"""
        logger_mod.OFFICIAL_COLUMN_MAP = {
            "gain_vs_static": {"eval": "round(1 - row.get('eval_rmse', 0) / 0.2, 3)"},
        }
        with tempfile.TemporaryDirectory() as d:
            lg = ExperimentLogger(out_dir=d, use_swanlab=False, experiment_name="t")
            lg.log({"eval_rmse": 0.1})
            lg.close()
            rows = _read_rows(lg.csv_path)
            self.assertEqual(rows[0]["gain_vs_static"], "0.5")

    def test_missing_source_column_is_blank(self):
        """映射指向不存在的列时留空而不抛异常。"""
        logger_mod.OFFICIAL_COLUMN_MAP = {"nonexistent_target": "no_such_col"}
        with tempfile.TemporaryDirectory() as d:
            lg = ExperimentLogger(out_dir=d, use_swanlab=False, experiment_name="t")
            lg.log({"eval_rmse": 0.19})
            lg.close()
            rows = _read_rows(lg.csv_path)
            self.assertEqual(rows[0]["nonexistent_target"], "")


if __name__ == "__main__":
    unittest.main()

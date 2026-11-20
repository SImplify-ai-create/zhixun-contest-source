"""策略层自进化链（五段）的结构、纯函数与默认关闭测试。

锁定五件事：
1. 五段齐全且判据可核对：`PREREGISTERED_GATES` 的 doc_ref 必须能在盘上核到锚点原文；
   启动门槛与 pef 两处常量不得漂移。
2. 两条候选源走同一接口：确定性枚举 96 条、LLM 账本去重后 7 条，字段一致。
3. 样本不足只能是 `insufficient_evidence`，且与 `accept` 不共用出口（passed 为 None）。
4. 采纳账本字段固定、不含时间戳；同一输入两次运行行内容逐字一致。
5. 默认不启用：模块不被生产入口引用，链不移动版本指针。
"""
import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from src import strategy_evolution as se

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ROW_KEYS = {"round", "candidate_id", "source", "criterion", "verdict",
                     "evidence_ref", "preregistration", "sample", "measured",
                     "version_pointer"}


@lru_cache(maxsize=1)
def _det():
    return se.deterministic_candidates()


@lru_cache(maxsize=1)
def _llm():
    return se.llm_candidates()


def _sig(cands):
    return [(c.candidate_id, round(float(c.proxy_score or 0.0), 12)) for c in cands]


@lru_cache(maxsize=1)
def _chain_result():
    return se.run_chain(round_no=1)


class TestPreregistration(unittest.TestCase):
    def test_doc_refs_resolve_to_disk(self):
        rows = se.check_doc_refs()
        self.assertGreaterEqual(len(rows), 4)
        bad = [row for row in rows if not row["ok"]]
        self.assertEqual(bad, [], f"预注册判据核不上盘上原文：{bad}")

    def test_thresholds_match_pef(self):
        learning = se._load_pef_learning(se.DEFAULT_PEF_DIR)
        self.assertEqual(
            (learning.MIN_DECISIONS, learning.MIN_STREAMS, learning.MIN_REPEATS),
            (se.MIN_DECISIONS, se.MIN_STREAMS, se.MIN_REPEATS))
        self.assertEqual(se.PROTOCOL_ID, "X-RUN-SIGN-1")

    def test_every_gate_has_criterion_and_ref(self):
        for spec in se.PREREGISTERED_GATES:
            self.assertTrue(spec.criterion)
            self.assertTrue(spec.doc_refs)
            self.assertEqual(len(spec.doc_refs), len(spec.doc_anchors))


class TestCandidateSources(unittest.TestCase):
    def test_deterministic_space_is_96_and_complete(self):
        cands, meta = _det()
        self.assertTrue(meta["switch"]["enabled"])
        self.assertEqual(len(cands), 96)
        self.assertEqual(len({c.candidate_id for c in cands}), 96)
        self.assertEqual(meta["corpus"]["n_decisions"], 1443)
        self.assertEqual(meta["corpus"]["n_streams"], 8)
        self.assertEqual(meta["corpus"]["min_repeats"], 3)
        for cand in cands:
            self.assertEqual(cand.source, se.SOURCE_DETERMINISTIC)
            self.assertIsNotNone(cand.proxy_score)
            self.assertTrue(cand.candidate_id.startswith("v1.2-learned-"))

    def test_deterministic_enumeration_is_stable(self):
        first = _sig(_det()[0])
        second = _sig(se.deterministic_candidates()[0])
        self.assertEqual(first, second)

    def test_llm_ledger_reads_unique_proposals_with_line_refs(self):
        cands, meta = _llm()
        self.assertEqual(meta["n_lines"], 7)
        self.assertEqual(meta["n_unique"], 7)
        self.assertEqual(len(cands), 7)
        lines = (ROOT / "docs" / "llm_rule_ledger.jsonl").read_text(
            encoding="utf-8").splitlines()
        for cand in cands:
            self.assertEqual(cand.source, se.SOURCE_LLM)
            ref = cand.provenance["ledger"]
            line_no = int(ref.rsplit("#L", 1)[1])
            entry = json.loads(lines[line_no - 1])
            self.assertEqual(entry["program_id"], cand.candidate_id)

    def test_unified_interface_fields(self):
        det, _ = _det()
        llm, _ = _llm()
        for cand in (det[0], llm[0]):
            self.assertTrue(cand.candidate_id)
            self.assertIn(cand.source, (se.SOURCE_DETERMINISTIC, se.SOURCE_LLM))
            self.assertIsInstance(dict(cand.body), dict)
            self.assertIsInstance(dict(cand.provenance), dict)


class TestAcceptance(unittest.TestCase):
    SAMPLE_OK = {"n_decisions": 1443, "n_streams": 8, "n_repeats": 3}

    def _perfect_cells(self):
        return [{"stream": "FD001", "seed": 42, "incumbent": 30.0,
                 "candidate": 30.0, "lower_is_better": True} for _ in range(16)]

    def test_insufficient_sample_never_passes(self):
        # 16 格全部非劣，但样本只有 10 条决策：必须走 insufficient，且 passed 为 None。
        out = se.evaluate_acceptance(
            self._perfect_cells(),
            {"n_decisions": 10, "n_streams": 8, "n_repeats": 3})
        self.assertEqual(out["verdict"], se.INSUFFICIENT)
        self.assertIsNone(out["passed"])
        self.assertIsNone(out["measured"])

    def test_insufficient_when_no_cells(self):
        out = se.evaluate_acceptance([], self.SAMPLE_OK)
        self.assertEqual(out["verdict"], se.INSUFFICIENT)
        self.assertIsNone(out["passed"])

    def test_accept_path_is_reachable(self):
        out = se.evaluate_acceptance(self._perfect_cells(), self.SAMPLE_OK)
        self.assertEqual(out["verdict"], se.ACCEPT)
        self.assertTrue(out["passed"])
        self.assertEqual(out["measured"]["non_worse"], 16)

    def test_reject_when_single_cell_beyond_cap(self):
        cells = self._perfect_cells()
        cells[3] = {"stream": "FD002", "seed": 42, "incumbent": 28.268838,
                    "candidate": 33.271701, "lower_is_better": True}
        out = se.evaluate_acceptance(cells, self.SAMPLE_OK)
        self.assertEqual(out["verdict"], se.REJECT)
        self.assertFalse(out["passed"])
        self.assertIn("FD002", out["reason"])

    def test_acceptance_is_pure(self):
        cells = self._perfect_cells()
        before = json.dumps(cells, sort_keys=True)
        a = se.evaluate_acceptance(cells, self.SAMPLE_OK)
        b = se.evaluate_acceptance(cells, self.SAMPLE_OK)
        self.assertEqual(a, b)
        self.assertEqual(before, json.dumps(cells, sort_keys=True))

    def test_recorded_cells_only_attach_to_chosen_candidate(self):
        report = json.loads((se.DEFAULT_PEF_DIR / "outputs" / "learning"
                             / "learning_report.json").read_text(encoding="utf-8"))
        chosen = report["chosen_candidate"]["version"]
        cells = se.recorded_cells(report, chosen)
        self.assertEqual(len(cells), 16)
        self.assertEqual(se.recorded_cells(report, "v1.2-learned-c1cd0p4r0"), [])


class TestChain(unittest.TestCase):
    def _result(self):
        return _chain_result()

    def test_preregistration_output_schema_and_default_off(self):
        self.assertFalse(se.ENABLED_BY_DEFAULT)
        self.assertFalse(se.CHAIN_MOVES_POINTER)
        for name in ("src/pipeline.py", "src/advanced.py",
                     "scripts/run_pipeline.py", "src/strategy.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("strategy_evolution", text,
                             f"{name} 不应引用策略进化链（默认不启用）")

    def test_chain_counts_and_pointer(self):
        result = self._result()
        self.assertEqual(result["preregistration_counts"], {"pass": 103})
        self.assertEqual(result["counts"].get(se.ACCEPT, 0), 0)
        self.assertEqual(result["counts"].get(se.REJECT, 0), 1)
        self.assertEqual(result["counts"].get(se.INSUFFICIENT, 0), 102)
        self.assertEqual(result["counts"].get(se.REJECT_PREREG, 0), 0)
        self.assertTrue(result["pointer_unchanged"])
        self.assertEqual(result["pointer"]["active_version"],
                         result["pointer_after"]["active_version"])

    def test_rejected_row_is_the_recorded_negative_result(self):
        rows = self._result()["rows"]
        rejected = [r for r in rows if r["verdict"] == se.REJECT]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["source"], se.SOURCE_DETERMINISTIC)
        self.assertIn("FD002", rejected[0]["reason"])
        self.assertIn("learning_report.json", rejected[0]["evidence_ref"])

    def test_rows_have_fixed_schema_and_no_timestamps(self):
        for row in self._result()["rows"]:
            self.assertTrue(REQUIRED_ROW_KEYS <= set(row))
            self.assertEqual(row["version_pointer"]["chain_moves_pointer"], False)
            blob = json.dumps(row, ensure_ascii=False, sort_keys=True)
            for token in ("created_at", "timestamp", "elapsed", "run_id"):
                self.assertNotIn(token, blob)

    def test_llm_rows_are_insufficient_not_accept(self):
        rows = [r for r in self._result()["rows"] if r["source"] == se.SOURCE_LLM]
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertEqual(row["verdict"], se.INSUFFICIENT)
            self.assertIsNone(row["passed"])
            self.assertIn("每流重复 1 < 2", row["reason"])
            self.assertIn("llm_rule_ledger.jsonl#L", row["evidence_ref"])

    def test_rows_are_byte_stable_across_runs(self):
        a = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in self._result()["rows"]]
        b = [json.dumps(r, ensure_ascii=False, sort_keys=True)
             for r in se.run_chain(round_no=1)["rows"]]
        self.assertEqual(a, b)

    def test_append_rows_increments_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adoption_ledger.jsonl"
            rows = self._result()["rows"][:2]
            se.append_rows(path, rows)
            before = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(before["round"], 1)
            # 追加后 _next_round 必须递增（每轮一条，可比对）。
            self.assertEqual(se._next_round(path), 2)

    def test_out_of_space_candidate_is_rejected_at_precheck(self):
        bad = se.Candidate(candidate_id="v1.2-learned-c9cd0p4r0",
                           source=se.SOURCE_DETERMINISTIC,
                           body={"params": {"trigger_confirm": 9,
                                            "trigger_cooldown": 0,
                                            "periodic_period": 4,
                                            "reserve": False}},
                           provenance={}, proxy_score=0.0)
        gate = se.preregistration_rows([bad])[0]
        self.assertFalse(gate["passed"])
        self.assertIn("P1_candidate_space", gate["failed"])
        rows = se.build_adoption_rows([bad], [gate], round_no=1,
                                      pointer={"active_version": "x", "sha256": "y"},
                                      samples={se.SOURCE_DETERMINISTIC: {
                                          "n_decisions": 1443, "n_streams": 8,
                                          "n_repeats": 3}},
                                      cells_by_id={})
        self.assertEqual(rows[0]["verdict"], se.REJECT_PREREG)


if __name__ == "__main__":
    unittest.main()

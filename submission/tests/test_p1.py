# -*- coding: utf-8 -*-
"""v2 P1 验收：批级检测下沉到 train，predict 侧只保留因果动作。

覆盖：
1. 动作库按 causal / transductive 分开，predict 侧显式拒绝 transductive。
2. causal 动作在 predict 里逐行累积状态，兜底分支不推进状态。
3. train_stage 对 train/validation 跑 KL / 马氏 / BBSD，产出可复现的 selector_trace。
4. 检测器与 selector 拒绝非 train 阶段。
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True  # 测试自身不得往 app/ 落 pyc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_host  # noqa: E402

APP_DIR = os.path.join(fake_host.APP_DIR, "app")
sys.path.insert(0, APP_DIR)
import actions  # noqa: E402
import detectors  # noqa: E402
import selector  # noqa: E402
import train_stage  # noqa: E402

CLASSES = ["c0", "c1", "c2", "c3", "c4", "c5"]
FEATURES = ["f0", "f1", "f2", "f3"]


def _build_npz(path, action=None, params=None):
    coef = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    extra = {}
    if action:
        extra["action"] = np.array(action)
    if params:
        extra["action_params"] = np.array(json.dumps(params))
    np.savez(path, task=np.array("classification"), classes=np.array(CLASSES),
             feature_order=np.array(FEATURES), coef=coef,
             intercept=np.zeros(len(CLASSES), dtype=float),
             prior_majority_class=np.array("c4"), **extra)


def _row(values, rid="r"):
    return json.dumps({"task": "classification",
                       "inputs": dict(zip(FEATURES, values)), "id": rid})


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class P1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.plain = os.path.join(cls.tmp, "plain.npz")
        cls.ema = os.path.join(cls.tmp, "ema.npz")
        cls.transductive = os.path.join(cls.tmp, "transductive.npz")
        cls.unknown = os.path.join(cls.tmp, "unknown.npz")
        _build_npz(cls.plain)
        _build_npz(cls.ema, action="streaming_ema_norm", params={"alpha": 0.5})
        _build_npz(cls.transductive, action="batch_zscore")
        _build_npz(cls.unknown, action="no_such_action")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _predict(self, rows, npz):
        return fake_host.run("predict", "\n".join(rows) + "\n",
                             env={"CONTEST_ARTIFACT_PATH": npz}, cwd=self.tmp)

    # ---------------------------------------------------------------- 动作门禁

    def test_action_registry_split(self):
        self.assertEqual(actions.causal_actions(),
                         ["identity", "per_cond_seq", "streaming_ema_norm"])
        self.assertEqual(actions.transductive_actions(),
                         ["batch_zscore", "per_cond_mono_norm"])

    def test_predict_rejects_transductive_action(self):
        rc, out, err = self._predict([_row([1, 0, 0, 0])], self.transductive)
        self.assertEqual(rc, 1, err)
        self.assertEqual(out, [], "transductive 动作必须在启动期拒绝，一行不写")
        self.assertIn("必须显式拒绝", err)

    def test_predict_rejects_unknown_action(self):
        rc, out, err = self._predict([_row([1, 0, 0, 0])], self.unknown)
        self.assertEqual(rc, 1, err)
        self.assertEqual(out, [])

    def test_check_stage_allows_causal_in_predict(self):
        for name in actions.causal_actions():
            actions.check_stage(name, "predict")
        for name in actions.transductive_actions():
            with self.assertRaises(actions.ActionError):
                actions.check_stage(name, "predict")
            actions.check_stage(name, "train")

    # ---------------------------------------------------------------- 因果状态

    def test_causal_action_advances_stream_state(self):
        rows = [_row([0, 0, 0, 0], "z"), _row([1, 1, 1, 1], "o1"), _row([1, 1, 1, 1], "o2")]
        rc, out, err = self._predict(rows, self.ema)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 3)
        self.assertNotEqual(out[1], out[2], "同一输入第二、三次出现应因历史不同而不同")

    def test_fallback_does_not_advance_stream_state(self):
        base = [_row([0, 0, 0, 0], "z"), _row([1, 1, 1, 1], "o")]
        with_bad = [base[0], json.dumps({"task": "classification",
                                         "inputs": {"f0": "boom", "f1": 0, "f2": 0, "f3": 0}}),
                    base[1]]
        rc1, out1, err1 = self._predict(with_bad, self.ema)
        rc2, out2, err2 = self._predict(base, self.ema)
        self.assertEqual(rc1, 0, err1)
        self.assertEqual(rc2, 0, err2)
        self.assertEqual(out1[2], out2[1], "坏行兜底不得推进流式状态")
        self.assertIn("声明默认", err1)

    def test_identity_action_keeps_old_behaviour(self):
        rc, out, err = self._predict([_row([1, 0, 0, 0])], self.plain)
        self.assertEqual(rc, 0, err)
        outputs = json.loads(out[0])["outputs"]
        scores = [o["score"] for o in outputs]
        self.assertEqual(outputs[scores.index(max(scores))]["label"], "c0")

    # ---------------------------------------------------------------- 检测只进 train

    def test_detectors_refuse_non_train_stage(self):
        a = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(detectors.DetectorNotAllowed):
            detectors.run_all(a, a, stage="predict")

    def test_selector_refuses_non_train_report(self):
        report = {"stage": "predict", "inputs": {"n_train": 2, "n_val": 2, "features": 2},
                  "detectors": {"kl": {"value": 0.0}, "mahalanobis": {"value": 0.0},
                                "bbsd": {"value": 0.0}}}
        with self.assertRaises(ValueError):
            selector.select(report)

    def test_predict_module_does_not_import_detectors(self):
        code = ("import sys; sys.path.insert(0, %r); import contest_submission; "
                "print('detectors' in sys.modules, 'selector' in sys.modules)" % APP_DIR)
        proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=self.tmp,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        self.assertEqual(proc.stdout.decode("utf-8").strip(), "False False",
                         "predict 侧不得加载批级检测器")

    # ---------------------------------------------------------------- train_stage

    def _data_dir(self, shift=3.0, n=200):
        data = os.path.join(self.tmp, "data_%s" % shift)
        os.makedirs(data, exist_ok=True)
        rng = np.random.default_rng(7)
        train = [{"id": "t%d" % i,
                  "inputs": {"f0": float(v), "f1": float(i), "f2": 1.0, "f3": 0.0}}
                 for i, v in enumerate(rng.normal(0.0, 1.0, n))]
        val = [{"id": "v%d" % i,
                "inputs": {"f0": float(v + shift), "f1": float(i), "f2": 1.0, "f3": 0.0}}
               for i, v in enumerate(rng.normal(0.0, 1.0, n))]
        _write_jsonl(os.path.join(data, "train.jsonl"), train)
        _write_jsonl(os.path.join(data, "validation.jsonl"), val)
        return data

    def test_train_stage_writes_selector_trace(self):
        data = self._data_dir()
        out_dir = os.path.join(self.tmp, "out_trace")
        result = train_stage.run(data_dir=data, out_dir=out_dir)
        self.assertIn(result["candidate"], ("A", "B", "C"))
        self.assertIn(result["action"], actions.causal_actions(),
                      "候选只能挂 causal 动作")
        trace = result["selector_trace"]
        self.assertEqual(trace["schema"], "contest.v2.selector-trace")
        self.assertEqual(trace["stage"], "train")
        for name in ("kl", "mahalanobis", "bbsd"):
            self.assertIn(name, trace["detectors"])
            self.assertTrue(np.isfinite(trace["detectors"][name]["value"]))
        self.assertEqual(trace["chosen"]["candidate"], result["candidate"])
        self.assertEqual(len(trace["rejected"]), 2)
        self.assertEqual([r["id"] for r in trace["rules"]], ["R1", "R2", "R3"])

        path = os.path.join(out_dir, train_stage.TRACE_NAME)
        self.assertTrue(os.path.isfile(path), "train 侧必须留 selector_trace")
        with open(path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertEqual(on_disk, trace)

    def test_train_stage_selector_is_deterministic(self):
        data = self._data_dir()
        out_dir = os.path.join(self.tmp, "out_det")
        first = train_stage.run(data_dir=data, out_dir=out_dir)
        second = train_stage.run(data_dir=data, out_dir=out_dir)
        self.assertEqual(first["candidate"], second["candidate"])
        self.assertEqual(first["selector_trace"], second["selector_trace"])

    def test_drift_pushes_candidate_away_from_identity(self):
        out_dir = os.path.join(self.tmp, "out_drift")
        shifted = train_stage.run(data_dir=self._data_dir(shift=30.0), out_dir=out_dir)
        self.assertNotEqual(shifted["candidate"], "C",
                            "明显漂移时不应选 identity")
        flat = train_stage.run(data_dir=self._data_dir(shift=0.0), out_dir=out_dir)
        self.assertEqual(flat["candidate"], "C", "无漂移时应选 identity")

    def test_train_stage_missing_data_is_error(self):
        with self.assertRaises(train_stage.TrainStageError):
            train_stage.run(data_dir=os.path.join(self.tmp, "no_such_dir"))

    def test_selector_candidates_are_causal(self):
        for cid, spec in selector.CANDIDATES.items():
            self.assertEqual(actions.kind_of(spec["action"]), actions.CAUSAL,
                             "候选 %s 不能挂非 causal 动作" % cid)


if __name__ == "__main__":
    unittest.main(verbosity=2)

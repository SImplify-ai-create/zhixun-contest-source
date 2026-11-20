# -*- coding: utf-8 -*-
"""v2 P2 加固验收：确定性审计 / 载入期常量预计算 / 回合策略悲观界。

对应三件加固（H-D3 / 资源效率 / H-D2）：

1. 确定性：同一批样本连续 predict 两遍，stdout 逐字节（sha256）一致才算过；
   覆盖分类、回归、声明默认兜底行、以及带流式动作的分类批次。
2. 载入期预计算：feature_order 钉成 tuple、动作参数一次校验、系数转连续数组；
   行路径不得再排序、求均值或读文件（用计数器与「禁止调用」补丁证明）。
3. 悲观界：候选臂 mean - k*sd 低于父候选一律不采纳；提交失败一轮即回退父候选；
   k 是配置项（参数 > CONTEST_PESSIMISM_K > 默认 1.0）。
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(_HERE), "app")
for _path in (APP, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import contest_submission  # noqa: E402
import fake_host  # noqa: E402
import harness  # noqa: E402
import train_round  # noqa: E402
from test_p2 import FEATURES, FakeRnd, _run, _write_data, _write_parent, _write_v2_parent  # noqa: E402


# ---------------------------------------------------------------- 夹具

def _write_classification(root, feature_order=tuple(FEATURES), classes=("a", "b", "c"),
                          action=None, action_params=None, prior="a"):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "checkpoint.npz")
    n_features = len(feature_order or FEATURES)
    coef = np.round(
        np.arange(len(classes) * n_features, dtype=float).reshape(len(classes), n_features) / 7.0 - 1.0,
        6)
    fields = {
        "task": np.array("classification"),
        "coef": coef,
        "intercept": np.zeros(len(classes), dtype=float),
        "classes": np.array(classes),
        "prior_majority_class": np.array(prior),
    }
    if feature_order is not None:
        fields["feature_order"] = np.array(feature_order)
    if action:
        fields["action"] = np.array(action)
    if action_params is not None:
        fields["action_params"] = np.array(json.dumps(action_params))
    np.savez(path, **fields)
    return path


def _inputs(i, task="classification"):
    return {"task": task,
            "id": "r%d" % i,
            "inputs": {name: round(0.17 * i + 0.31 * j - 0.4, 6)
                       for j, name in enumerate(FEATURES)}}


def _line(i, task="classification"):
    return json.dumps(_inputs(i, task), ensure_ascii=False)


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- 1. 确定性审计

class DeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.cls = _write_classification(os.path.join(cls.tmp, "cls"))
        cls.ema = _write_classification(os.path.join(cls.tmp, "ema"),
                                        action="streaming_ema_norm",
                                        action_params={"alpha": 0.5, "eps": 1e-8})
        cls.reg = _write_parent(os.path.join(cls.tmp, "reg"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _twice(self, artifact, lines):
        env = {"CONTEST_ARTIFACT_PATH": artifact}
        text = "\n".join(lines) + "\n"
        rc1, out1, err1 = fake_host.run_bytes("predict", text, env=env, cwd=self.tmp)
        rc2, out2, err2 = fake_host.run_bytes("predict", text, env=env, cwd=self.tmp)
        self.assertEqual(rc1, 0, err1)
        self.assertEqual(rc2, 0, err2)
        self.assertEqual(_sha(out1), _sha(out2), "同一批样本两次 predict 的 stdout 哈希必须一致")
        self.assertEqual(out1, out2, "逐字节比对必须完全一致，不只是语义一致")
        return out1, err1

    def test_classification_byte_identical(self):
        out, _ = self._twice(self.cls, [_line(i) for i in range(6)])
        rows = [json.loads(line) for line in out.decode("utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 6)
        self.assertIsInstance(rows[0]["outputs"], list)

    def test_regression_byte_identical(self):
        out, _ = self._twice(self.reg, [_line(i, task="regression") for i in range(6)])
        rows = [json.loads(line) for line in out.decode("utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 6)
        self.assertIn("value", rows[0]["outputs"])

    def test_declared_default_rows_byte_identical(self):
        lines = [
            _line(0),
            "{这不是 JSON",                                              # 行级解析失败
            _line(1, task="regression"),                                 # 行 task 与产物不符
            json.dumps({"id": "x", "task": "classification", "inputs": {"f0": 1.0}}),  # 缺特征
            json.dumps({"id": "y", "task": "classification",
                        "inputs": {name: "坏值" for name in FEATURES}}),  # 非数值
        ]
        out, err = self._twice(self.cls, lines)
        rows = [json.loads(line) for line in out.decode("utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 5, "兜底行也必须一行吐一行")
        for row in rows[1:]:
            payload = row["outputs"]
            self.assertEqual(payload[0], {"label": "a", "score": 1.0}, "分类兜底回先验多数类")
            self.assertTrue(all(item["score"] == 0.0 for item in payload[1:]))
        self.assertIn("声明默认", err)

    def test_streaming_action_byte_identical(self):
        out, _ = self._twice(self.ema, [_line(i) for i in range(8)])
        rows = [json.loads(line) for line in out.decode("utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 8)

    def test_same_process_repeat_is_identical(self):
        """同一进程里把同一批喂两遍：无状态动作下，两段输出必须逐字节相同。"""
        lines = [_line(i) for i in range(3)]
        text = "\n".join(lines + lines) + "\n"
        rc, out, err = fake_host.run_bytes(
            "predict", text, env={"CONTEST_ARTIFACT_PATH": self.cls}, cwd=self.tmp)
        self.assertEqual(rc, 0, err)
        rows = [line for line in out.splitlines() if line.strip()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[:3], rows[3:], "同一进程内重复同一批，输出必须一致")


# ---------------------------------------------------------------- 2. 载入期预计算

class PrecomputeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        contest_submission._CACHE.clear()
        contest_submission._STREAM_STATE.clear()
        for key in contest_submission._STATS:
            contest_submission._STATS[key] = 0
        self._patch = mock.patch.dict(os.environ, {}, clear=False)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _head(self, artifact):
        os.environ["CONTEST_ARTIFACT_PATH"] = artifact
        return contest_submission._load_head("/model")

    def _sample(self, i, task="classification"):
        return harness.Sample(task=None, inputs=_inputs(i)["inputs"],
                              dataset_key="ds-1", context={"task": task},
                              lineno=i + 1)

    def test_load_precomputes_order_action_and_constants(self):
        head = self._head(_write_classification(os.path.join(self.tmp, "cls")))
        self.assertEqual(head.order, tuple(FEATURES), "feature_order 必须在装载期钉成 tuple")
        self.assertEqual(head.order_source, "feature_order")
        self.assertEqual(head.n_features, len(FEATURES))
        self.assertEqual(head.act["action"], "identity")
        self.assertEqual(head.act["kind"], "causal")
        self.assertTrue(head.coef.flags["C_CONTIGUOUS"], "coef 必须转成连续数组")
        self.assertEqual(head.act["params"], {}, "动作参数应在装载期解析完")
        self.assertEqual(contest_submission._STATS["loads"], 1)

    def test_second_load_hits_cache(self):
        artifact = _write_classification(os.path.join(self.tmp, "cls"))
        self._head(artifact)
        self._head(artifact)
        self.assertEqual(contest_submission._STATS["loads"], 1, "同进程只允许读一次检查点")

    def test_ema_params_are_validated_once(self):
        artifact = _write_classification(os.path.join(self.tmp, "ema"),
                                         action="streaming_ema_norm",
                                         action_params={"alpha": 0.25, "eps": 1e-6})
        head = self._head(artifact)
        self.assertEqual(head.act["params"], {"alpha": 0.25, "eps": 1e-6})
        self.assertIsInstance(head.act["params"]["alpha"], float)

    def test_row_path_has_no_sort_mean_or_reread(self):
        """40 行里：读文件恰好 1 次、排序 0 次、求均值 0 次。"""
        artifact = _write_classification(os.path.join(self.tmp, "cls"))
        os.environ["CONTEST_ARTIFACT_PATH"] = artifact
        counts = {"load": 0}
        real_load = np.load

        def counting_load(*args, **kwargs):
            counts["load"] += 1
            return real_load(*args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("行路径不得求均值")

        with mock.patch.object(np, "load", counting_load), \
                mock.patch.object(np, "mean", forbidden), \
                mock.patch.object(np, "average", forbidden):
            for i in range(40):
                payload = contest_submission.predict(self._sample(i), "/model")
                self.assertIsInstance(payload, list)

        self.assertEqual(counts["load"], 1, "检查点只允许在装载期读一次")
        self.assertEqual(contest_submission._STATS["sorts"], 0, "行路径不得重复排序 feature_order")
        self.assertEqual(contest_submission._STATS["rows"], 40)

    def test_missing_feature_order_sorts_once_then_pins(self):
        declared = _write_classification(os.path.join(self.tmp, "declared"))
        undeclared = _write_classification(os.path.join(self.tmp, "undeclared"),
                                           feature_order=None)
        first = self._head(declared)
        texts = [contest_submission.predict(self._sample(i), "/model") for i in range(3)]

        contest_submission._CACHE.clear()
        contest_submission._STATS["sorts"] = 0
        second = self._head(undeclared)
        self.assertIsNone(second.order, "未声明 feature_order 时装载期无法定序")
        got = [contest_submission.predict(self._sample(i), "/model") for i in range(3)]
        self.assertEqual(contest_submission._STATS["sorts"], 1, "只允许首行排序一次")
        self.assertEqual(second.order, tuple(sorted(second.order)))
        self.assertEqual([item["score"] for row in got for item in row],
                         [item["score"] for row in texts for item in row],
                         "首行钉住的顺序必须与显式 feature_order 等价")
        self.assertNotEqual(first.order_source, second.order_source)


# ---------------------------------------------------------------- 3. 悲观界

def _arm(n, mean, sd=0.0, cooldown=0, failures=0):
    return {"n": n, "mean": mean, "m2": (sd * sd * (n - 1)) if n > 1 else 0.0,
            "cooldown": cooldown, "failures": failures}


class PessimismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.parent_root = os.path.join(cls.tmp, "parent")
        # 悲观界比较的是动作臂统计，父产物必须是 v2 格式（官方 MLP 父产物上动作臂无处安放）。
        _write_v2_parent(cls.parent_root)
        cls.data = os.path.join(cls.tmp, "data")
        _write_data(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _state(self, name, arms):
        path = os.path.join(self.tmp, "state", name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "round_state.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": "contest.v2.round-state", "round_index": 1, "arms": arms},
                      handle, ensure_ascii=False, indent=1)
        return path

    def _rnd(self, name, reply=('{"arm":"A","params":{},"hypothesis":"账本为空，按漂移检测选标准化臂",'
                                '"kill_condition":"若平台分低于父候选则回退","confidence":"medium"}'),
             **kwargs):
        work = os.path.join(self.tmp, "work", name)
        os.makedirs(work, exist_ok=True)
        kwargs.setdefault("round_index", 2)
        return FakeRnd(self.parent_root, work, data_dir=self.data, reply=reply, **kwargs)

    def test_rejects_arm_below_parent_bound(self):
        rnd = self._rnd("bound", state_src=self._state("bound", {
            "PARENT": _arm(20, 0.90, 0.01),
            "A": _arm(20, 0.50, 0.01),
        }))
        ledger, err = _run(rnd)
        self.assertEqual(ledger["configuration"]["candidate"], "PARENT")
        self.assertEqual(ledger["configuration"]["policy"], "pessimism_bound_below_parent")
        rejected = ledger["configuration"]["pessimism"]["rejected"]
        self.assertEqual(rejected["candidate"], "A")
        self.assertLess(rejected["bound"], rejected["parent_bound"])
        self.assertIn("悲观界否掉候选 A", err)
        self.assertEqual(len(rnd.submits), 1)

    def test_unobserved_arm_is_not_rejected(self):
        rnd = self._rnd("unseen", state_src=self._state("unseen", {"PARENT": _arm(20, 0.90, 0.01)}))
        ledger, _ = _run(rnd)
        self.assertEqual(ledger["configuration"]["candidate"], "A")
        self.assertEqual(ledger["configuration"]["policy"], "model_proposal")
        self.assertIsNone(ledger["configuration"]["pessimism"]["bounds"]["A"])

    def test_k_controls_rejection(self):
        arms = {"PARENT": _arm(10, 0.90, 0.0), "A": _arm(10, 0.95, 0.10)}
        loose = self._rnd("k0", state_src=self._state("k", arms))
        ledger_loose, _ = _run(loose, pessimism_k=0.0)
        self.assertEqual(ledger_loose["configuration"]["candidate"], "A")
        self.assertEqual(ledger_loose["configuration"]["pessimism"]["k"], 0.0)
        self.assertEqual(ledger_loose["configuration"]["pessimism"]["k_source"], "parameter")

        strict = self._rnd("k1", state_src=self._state("k", arms))
        ledger_strict, _ = _run(strict, pessimism_k=1.0)
        self.assertEqual(ledger_strict["configuration"]["candidate"], "PARENT")

        with mock.patch.dict(os.environ, {"CONTEST_PESSIMISM_K": "3"}):
            from_env = self._rnd("kenv", state_src=self._state("k", arms))
            ledger_env, _ = _run(from_env)
        self.assertEqual(ledger_env["configuration"]["pessimism"]["k"], 3.0)
        self.assertTrue(ledger_env["configuration"]["pessimism"]["k_source"].startswith("env:"))
        self.assertEqual(ledger_env["configuration"]["candidate"], "PARENT")

    def test_default_k_has_rationale(self):
        self.assertEqual(train_round.PESSIMISM["k"], 1.0)
        self.assertIn("标准差", train_round.PESSIMISM["rationale"])
        with mock.patch.dict(os.environ, {"CONTEST_PESSIMISM_K": ""}):
            rnd = self._rnd("kdefault")
            ledger, _ = _run(rnd)
        self.assertEqual(ledger["configuration"]["pessimism"]["k"], 1.0)
        self.assertEqual(ledger["configuration"]["pessimism"]["k_source"], "default")

    def test_failed_submit_falls_back_to_parent_in_same_round(self):
        rnd = self._rnd("fallback", state_src=self._state("fallback", {
            "PARENT": _arm(10, 0.80, 0.01),
            "A": _arm(10, 0.95, 0.01),
        }))
        real_submit = rnd.submit_candidate
        seen = []

        def flaky(candidate_dir, **kwargs):
            seen.append(kwargs["configuration"]["candidate"])
            if len(seen) == 1:
                raise RuntimeError("平台 503")
            return real_submit(candidate_dir, **kwargs)

        rnd.submit_candidate = flaky
        ledger, err = _run(rnd)
        self.assertEqual(seen, ["A", "PARENT"], "失败一轮必须立即改用父候选重提")
        self.assertEqual(ledger["configuration"]["candidate"], "PARENT")
        self.assertEqual(ledger["configuration"]["policy"], "pessimism_fallback_parent")
        self.assertEqual(ledger["configuration"]["pessimism"]["fallback"]["from"], "A")
        self.assertEqual(len(rnd.submits), 1, "只有成功的父候选进入平台回执")
        self.assertEqual(len(rnd.finishes), 1)
        arms = ledger["state"]["arms"]
        self.assertEqual(arms["A"]["failures"], 1)
        self.assertEqual(arms["A"]["cooldown"], 1)
        self.assertAlmostEqual(arms["A"]["mean"], 0.875, msg="失败后均值应向父候选收缩一半")
        self.assertIn("退回父候选", err)

    def test_cooldown_blocks_next_round(self):
        arms = {"A": _arm(10, 0.99, 0.0, cooldown=1)}
        rnd = self._rnd("cooldown", state_src=self._state("cooldown", arms))
        ledger, _ = _run(rnd)
        self.assertEqual(ledger["configuration"]["candidate"], "PARENT")
        self.assertEqual(ledger["configuration"]["policy"], "pessimism_cooldown")
        self.assertEqual(ledger["configuration"]["pessimism"]["rejected"]["cooldown"], 1)
        self.assertEqual(ledger["state"]["arms"]["A"]["cooldown"], 0, "冷却一回合后应解除")

    def test_successful_round_records_arm_stats(self):
        rnd = self._rnd("stats")
        ledger, _ = _run(rnd)
        arms = ledger["state"]["arms"]
        self.assertEqual(arms["A"]["n"], 1)
        self.assertAlmostEqual(arms["A"]["mean"], 0.6123)
        self.assertEqual(ledger["state"]["arm_score"]["key"], "validation.score")

    def test_round2_state_feeds_round3(self):
        """第 2 轮写下的臂统计必须能被第 3 轮原样读回并继续累积。"""
        second = self._rnd("chain2")
        _run(second)
        state_dir = second.saves[-1]

        third = self._rnd("chain3", state_src=state_dir)
        ledger, _ = _run(third)
        self.assertEqual(ledger["configuration"]["candidate"], "A")
        self.assertEqual(ledger["state"]["arms"]["A"]["n"], 2)
        self.assertAlmostEqual(ledger["state"]["arms"]["A"]["mean"], 0.6123)


if __name__ == "__main__":
    unittest.main()

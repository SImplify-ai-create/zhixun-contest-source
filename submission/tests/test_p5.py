# -*- coding: utf-8 -*-
"""v2 P5 验收（2026-11-20 加固四条）：

1. NAS 非对称评分：C-MAPSS 口径（d≥0 高估 exp(d/10)−1；d<0 exp(−d/13)−1，逐样本求和），
   已对平台回执锚定（父权重 C 包公开验证集 = 5,823,439,325.34，回执显示差 10³）。
2. 权重拼接 rel_l2 上限：check_candidate 第 5b 项，补官方格式无 coef 时第 5 项跳过的空档。
3. soup 插值：回归提交权重 = 父 + α·(训练结果 − 父)，α 必须过 REL_CAP（C 岭头 rel≈0.26 会被拦）。
4. 历史最优回投：finish 在固定标尺（本轮 validation.jsonl）上选历史最优产物，
   无 restore_artifact 时退回本轮产物（HARNESS_SDK.md:158 允许"选回更早的产物"）。
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(_HERE), "app")
if APP not in sys.path:
    sys.path.insert(0, APP)

import llm_cage  # noqa: E402
import train_mlp  # noqa: E402
import train_round  # noqa: E402
from test_p2 import FakeRnd, _write_data, _write_parent, _run  # noqa: E402


class _EnvGuard(object):
    """环境变量临时改写；退出时恢复原样，避免用例间串味。"""

    def __init__(self, **mapping):
        self.mapping = mapping
        self.saved = {}

    def __enter__(self):
        for key, value in self.mapping.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _write_official(root, w1, b1, w2, b2, mean, scale, paths, task="regression",
                    target_mean=0.0, target_scale=1.0, classes=None):
    os.makedirs(root, exist_ok=True)
    meta = {"task": task, "paths": paths, "hidden": int(w1.shape[1]),
            "target_mean": float(target_mean), "target_scale": float(target_scale),
            "classes": list(classes or [])}
    np.savez(os.path.join(root, "checkpoint.npz"),
             w1=w1, b1=b1, w2=w2, b2=b2, mean=mean, scale=scale,
             metadata=np.array(json.dumps(meta)))
    return os.path.join(root, "checkpoint.npz")


class RestoringRnd(FakeRnd):
    """FakeRnd + restore_artifact：把登记的 artifact_id 映射到本地 checkpoint。"""

    def __init__(self, *args, restores=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.restores = {str(k): v for k, v in (restores or {}).items()}
        self.restored = []

    def restore_artifact(self, artifact_id, dest):
        self.restored.append(str(artifact_id))
        os.makedirs(dest, exist_ok=True)
        shutil.copyfile(self.restores[str(artifact_id)],
                        os.path.join(dest, "checkpoint.npz"))


class NasaScoreTest(unittest.TestCase):
    def test_zero_error_is_zero(self):
        y = np.array([3.0, 10.0, 0.0])
        self.assertEqual(train_mlp.nasa_score(y, y.copy()), 0.0)

    def test_symmetric_at_10_and_13(self):
        # d=+10 与 d=−13 的单样本贡献都等于 e−1：常数 10/13 的方向不能写反。
        plus = train_mlp.nasa_score(np.array([0.0]), np.array([10.0]))
        minus = train_mlp.nasa_score(np.array([0.0]), np.array([-13.0]))
        self.assertAlmostEqual(plus, np.e - 1.0, places=9)
        self.assertAlmostEqual(minus, np.e - 1.0, places=9)

    def test_overestimate_penalized_more(self):
        # 同样 |d|=10：高估（d>0）必须比低估罚得重（TASK.md：高估剩余寿命惩罚更大）。
        over = train_mlp.nasa_score(np.array([0.0]), np.array([10.0]))
        under = train_mlp.nasa_score(np.array([0.0]), np.array([-10.0]))
        self.assertGreater(over, under)

    def test_sum_over_samples(self):
        y = np.array([0.0, 0.0])
        total = train_mlp.nasa_score(y, np.array([10.0, -13.0]))
        self.assertAlmostEqual(total, 2.0 * (np.e - 1.0), places=9)


class RelCapTest(unittest.TestCase):
    def test_identical_is_zero(self):
        base = {"w1": np.ones((3, 2)), "b1": np.zeros(2),
                "w2": np.ones((2, 1)), "b2": np.zeros(1)}
        self.assertEqual(train_mlp.concat_rel_l2(base, base), 0.0)

    def test_known_ratio(self):
        base = {"w1": np.full((2, 2), 2.0), "b1": np.zeros(2),
                "w2": np.zeros((2, 1)), "b2": np.zeros(1)}
        cur = {k: v.copy() for k, v in base.items()}
        cur["w1"] = base["w1"] * 1.5  # Δ = 0.5×w1 ⇒ rel = 0.5×‖w1‖/‖w1‖ = 0.5
        self.assertAlmostEqual(train_mlp.concat_rel_l2(cur, base), 0.5, places=9)

    def test_cap_default_and_override(self):
        with _EnvGuard(CONTEST_WEIGHT_REL_CAP=None):
            self.assertEqual(train_mlp.weight_rel_cap(), 0.1)
        with _EnvGuard(CONTEST_WEIGHT_REL_CAP="0.25"):
            self.assertEqual(train_mlp.weight_rel_cap(), 0.25)
        with _EnvGuard(CONTEST_WEIGHT_REL_CAP="abc"):
            self.assertRaises(train_mlp.TrainMlpError, train_mlp.weight_rel_cap)


class CandidateRelCheckTest(unittest.TestCase):
    """check_candidate 第 5b 项：官方格式候选与父产物的拼接 rel_l2 上限。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        rng = np.random.RandomState(0)
        cls.paths = [json.dumps([["key", "f%d" % i]]) for i in range(4)]
        cls.w1 = rng.normal(size=(4, 5))
        cls.b1 = rng.normal(size=5)
        cls.w2 = rng.normal(size=(5, 1))
        cls.b2 = rng.normal(size=1)
        cls.parent = _write_official(
            os.path.join(cls.tmp, "parent"), cls.w1, cls.b1, cls.w2, cls.b2,
            np.zeros(4), np.ones(4), cls.paths)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _candidate(self, scale):
        return _write_official(
            os.path.join(self.tmp, "cand-%.2f" % scale),
            self.w1 * (1.0 + scale), self.b1, self.w2, self.b2,
            np.zeros(4), np.ones(4), self.paths)

    def test_small_drift_passes(self):
        cand = self._candidate(0.02)  # rel ≈ 0.02×‖w1‖/‖all‖ < 0.1
        result = llm_cage.check_candidate(os.path.dirname(cand), parent_path=self.parent)
        self.assertTrue(result["ok"], result["errors"])
        self.assertLess(result["facts"]["weight_rel_l2"], 0.1)

    def test_large_drift_rejected(self):
        cand = self._candidate(1.0)  # w1 翻倍：rel 远超 0.1
        result = llm_cage.check_candidate(os.path.dirname(cand), parent_path=self.parent)
        self.assertFalse(result["ok"])
        self.assertTrue(any("拼接 L2" in msg for msg in result["errors"]),
                        result["errors"])

    def test_parent_copy_passes(self):
        cand_dir = os.path.join(self.tmp, "cand-copy")
        os.makedirs(cand_dir, exist_ok=True)
        shutil.copyfile(self.parent, os.path.join(cand_dir, "checkpoint.npz"))
        result = llm_cage.check_candidate(cand_dir, parent_path=self.parent)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["facts"]["weight_rel_l2"], 0.0)


class SoupTest(unittest.TestCase):
    """回归 soup：上限收紧时提交权重必须是插值，且仍过上限、仍有记录。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.parent = _write_parent(os.path.join(cls.tmp, "parent"))
        cls.data = os.path.join(cls.tmp, "data")
        _write_data(cls.data, labels=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tight_cap_forces_interpolation(self):
        # 先按原路径（soup 关）测出本夹具全量更新的真实 rel，再把 cap 钉在它的一半：
        # 夹具的更新幅度不做先验假设，用例对"更新大小"自洽。
        plain_out = os.path.join(self.tmp, "out-plain-first")
        with _EnvGuard(CONTEST_SOUP="0"):
            train_mlp.finetune(self.parent, plain_out, self.data, arm="A",
                               dataset_key="B")
        with np.load(self.parent, allow_pickle=False) as par, \
                np.load(os.path.join(plain_out, "checkpoint.npz"),
                        allow_pickle=False) as cur:
            rel_full = train_mlp.concat_rel_l2(
                {k: cur[k] for k in ("w1", "b1", "w2", "b2")},
                {k: par[k] for k in ("w1", "b1", "w2", "b2")})
        self.assertGreater(rel_full, 0.0, "夹具必须真的产生权重更新")
        cap = rel_full / 2.0

        out = os.path.join(self.tmp, "out-soup")
        with _EnvGuard(CONTEST_WEIGHT_REL_CAP=repr(cap), CONTEST_SOUP=None):
            report = train_mlp.finetune(self.parent, out, self.data, arm="A",
                                        dataset_key="B")
        soup = report["soup"]
        self.assertIsNotNone(soup, "回归线默认开 soup")
        self.assertIsNotNone(soup["alpha"])
        self.assertLess(soup["alpha"], 1.0, "cap=rel/2 必须拦住全量更新")
        self.assertLessEqual(soup["rel_l2"], cap + 1e-12)
        self.assertTrue(any(not item["ok"] for item in soup["grid"]),
                        "网格里应出现被上限拦掉的 α")
        if report["weights_changed"]:
            with np.load(os.path.join(out, "checkpoint.npz"),
                         allow_pickle=False) as archive, \
                    np.load(self.parent, allow_pickle=False) as par:
                rel = train_mlp.concat_rel_l2(
                    {k: archive[k] for k in ("w1", "b1", "w2", "b2")},
                    {k: par[k] for k in ("w1", "b1", "w2", "b2")})
            self.assertLessEqual(rel, cap + 1e-12)

    def test_soup_disabled_restores_plain_path(self):
        out = os.path.join(self.tmp, "out-plain")
        with _EnvGuard(CONTEST_SOUP="0"):
            report = train_mlp.finetune(self.parent, out, self.data, arm="A",
                                        dataset_key="B")
        self.assertIsNone(report["soup"])
        self.assertIsNone(report["nas"])


class RepickRuleTest(unittest.TestCase):
    """_repick_best 的选择规则：分类 argmax f1；B 式 argmin rmse；C 式 NAS 决胜。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.paths = [json.dumps([["key", "f%d" % i]]) for i in range(2)]
        # 两个回归产物：对 y=10 的标尺分别恒预测 11（高估 1）与 9（低估 1），
        # rmse 同为 1.0；NAS 不同（高估罚得重）。
        zero_w1 = np.zeros((2, 2))
        zero_b1 = np.zeros(2)
        zero_w2 = np.zeros((2, 1))
        cls.over = _write_official(os.path.join(cls.tmp, "over"),
                                   zero_w1, zero_b1, zero_w2, np.array([11.0]),
                                   np.zeros(2), np.ones(2), cls.paths)
        cls.under = _write_official(os.path.join(cls.tmp, "under"),
                                    zero_w1, zero_b1, zero_w2, np.array([9.0]),
                                    np.zeros(2), np.ones(2), cls.paths)
        data = os.path.join(cls.tmp, "data")
        os.makedirs(data)
        for name in ("train.jsonl", "validation.jsonl"):
            with open(os.path.join(data, name), "w", encoding="utf-8") as handle:
                for i in range(8):
                    handle.write(json.dumps(
                        {"id": "v-%d" % i, "inputs": {"f0": 0.1 * i, "f1": 0.0},
                         "answer": 10.0}) + "\n")
        cls.data = data

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _rnd(self):
        work = os.path.join(self.tmp, "work-repick")
        os.makedirs(work, exist_ok=True)
        return RestoringRnd(self.tmp, work, data_dir=self.data,
                            restores={"art-over": self.over, "art-under": self.under}), work

    def _registry(self):
        return [{"artifact_id": "art-over", "round": 1, "arm": "A"},
                {"artifact_id": "art-under", "round": 2, "arm": "A"}]

    def test_c_line_picks_lower_nas_on_rmse_tie(self):
        rnd, work = self._rnd()
        pick = train_round._repick_best(rnd, self._registry(), self.data, work, 3,
                                        "C", "art-current", self.over)
        self.assertEqual(pick["artifact_id"], "art-under",
                         "rmse 并列时 C 线必须按 NAS 取低估侧")

    def test_other_regression_keeps_rmse_order(self):
        rnd, work = self._rnd()
        pick = train_round._repick_best(rnd, self._registry(), self.data, work, 3,
                                        "B", "art-current", self.over)
        self.assertEqual(pick["artifact_id"], "art-over",
                         "rmse 并列时非 C 线保持登记册顺序（先到先得）")


class RepickRunTest(unittest.TestCase):
    """整轮回投：历史产物在固定标尺上更优时，finish 改选它；并核对 wc/p/c 标签。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.parent = _write_parent(os.path.join(cls.tmp, "parent"))
        cls.data = os.path.join(cls.tmp, "data")
        _write_data(cls.data, labels=True)
        # art-old：用父的隐层 + 在验证集上闭式解头（in-sample 拟合），
        # 保证它在标尺上优于任何从父产物出发的当轮训练结果。
        with np.load(cls.parent, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        meta = json.loads(str(np.asarray(arrays["metadata"]).tolist()))
        import train_mlp as tm
        z, y, _task = tm.load_yardstick(cls.data, cls.parent)
        hidden = np.tanh(z @ arrays["w1"] + arrays["b1"])
        target = (y - float(meta["target_mean"])) / float(meta["target_scale"])
        design = np.concatenate([hidden, np.ones((hidden.shape[0], 1))], axis=1)
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        arrays["w2"] = coef[:-1].reshape(arrays["w2"].shape)
        arrays["b2"] = np.asarray([coef[-1]])
        cls.old = os.path.join(cls.tmp, "old", "checkpoint.npz")
        os.makedirs(os.path.dirname(cls.old), exist_ok=True)
        np.savez(cls.old, **arrays)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_finish_repicks_better_historical_artifact(self):
        work = os.path.join(self.tmp, "work-run")
        state_src = os.path.join(self.tmp, "state-src")
        os.makedirs(state_src)
        with open(os.path.join(state_src, "round_state.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"artifacts": [{"artifact_id": "art-old", "round": 1,
                                      "arm": "A"}]}, handle)
        rnd = RestoringRnd(os.path.dirname(self.parent), work,
                           data_dir=self.data, state_src=state_src,
                           round_index=2, artifact_id="art-new",
                           restores={"art-old": self.old})
        with _EnvGuard(CONTEST_WEIGHT_REL_CAP="100"):  # 让当轮训练不被 soup 拦住
            ledger, _err = _run(rnd)
        self.assertEqual(len(rnd.finishes), 1)
        finish = rnd.finishes[0]
        self.assertEqual(finish["selected_artifact_id"], "art-old",
                         "固定标尺上 art-old 更优，finish 必须回投")
        self.assertEqual(ledger["artifact_id"], "art-new", "提交仍记本轮产物")
        self.assertEqual(ledger["finish_artifact_id"], "art-old")
        self.assertEqual(ledger["repick"]["artifact_id"], "art-old")
        self.assertEqual(ledger["state"]["selected_artifact_id"], "art-old")
        self.assertEqual([a["artifact_id"] for a in ledger["state"]["artifacts"]],
                         ["art-old", "art-new"], "登记册要跨轮累积")
        self.assertIn("art-old", rnd.restored)
        summary = rnd.submits[0]["kwargs"]["change_summary"]
        self.assertIn("[wc=", summary)
        self.assertIn(" p=", summary)
        self.assertIn(" c=", summary)

    def test_no_restore_falls_back_to_current(self):
        work = os.path.join(self.tmp, "work-plain")
        state_src = os.path.join(self.tmp, "state-plain")
        os.makedirs(state_src)
        with open(os.path.join(state_src, "round_state.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"artifacts": [{"artifact_id": "art-old", "round": 1,
                                      "arm": "A"}]}, handle)
        rnd = FakeRnd(os.path.dirname(self.parent), work,
                      data_dir=self.data, state_src=state_src,
                      round_index=2, artifact_id="art-new")
        ledger, _err = _run(rnd)
        self.assertEqual(rnd.finishes[0]["selected_artifact_id"], "art-new",
                         "没有 restore_artifact 时必须采用本轮产物")


if __name__ == "__main__":
    unittest.main()

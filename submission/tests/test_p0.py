# -*- coding: utf-8 -*-
"""v2 P0 验收：本地假宿主把 describe / train / predict 三入口跑通。

重点覆盖本轮修正：
1. predict 是总输出 + 声明默认（每行必出一行，处理不了走预注册兜底，不 fail-fast）。
2. describe 报两个 task，predict 按 CONTEST_TASK 泛型分派；形状归属 上下文 > 行 > 产物。
3. 导入不往 /app 落 pyc。
"""

import json
import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True  # 测试自身不得往 app/ 落 pyc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_host  # noqa: E402

PROTOCOL = "contest.harness.workspace.v2"
CLASSES = ["c0", "c1", "c2", "c3", "c4", "c5"]
FEATURES = ["f0", "f1", "f2", "f3"]
PRIOR_CLASS = "c4"
PRIOR_MEAN = 1.5


def _build_classification(path):
    coef = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    np.savez(
        path,
        task=np.array("classification"),
        classes=np.array(CLASSES),
        feature_order=np.array(FEATURES),
        coef=coef,
        intercept=np.zeros(len(CLASSES), dtype=float),
        prior_majority_class=np.array(PRIOR_CLASS),
    )


def _build_regression(path):
    np.savez(
        path,
        task=np.array("regression"),
        feature_order=np.array(["x", "y"]),
        coef=np.array([1.0, 0.0], dtype=float),
        intercept=np.array(-2.0),
        prior_mean=np.array(PRIOR_MEAN),
    )


def _cls_row(f0, f1, f2, f3, rid):
    return json.dumps({"task": "classification",
                       "inputs": {"f0": f0, "f1": f1, "f2": f2, "f3": f3},
                       "id": rid}, ensure_ascii=False)


def _snapshot(root):
    files = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            files[os.path.relpath(full, root)] = os.path.getsize(full)
    return files


APP = os.path.join(fake_host.APP_DIR, "app")


class P0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.cls_npz = os.path.join(cls.tmp, "cls.npz")
        cls.reg_npz = os.path.join(cls.tmp, "reg.npz")
        cls.bad_npz = os.path.join(cls.tmp, "bad.npz")
        _build_classification(cls.cls_npz)
        _build_regression(cls.reg_npz)
        np.savez(cls.bad_npz, task=np.array("classification"),
                 classes=np.array(["a"]), coef=np.zeros((3, 2), dtype=float))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _predict(self, rows, npz, extra_env=None, cwd=None):
        env = {"CONTEST_ARTIFACT_PATH": npz, "CONTEST_MODEL_DIR": "/model"}
        if extra_env:
            env.update(extra_env)
        return fake_host.run("predict", "\n".join(rows) + "\n", env=env, cwd=cwd or self.tmp)

    # ---------------------------------------------------------- describe

    def test_describe_shape(self):
        rc, out, err = fake_host.run("describe")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1, out)
        info = json.loads(out[0])
        self.assertEqual(set(info.keys()), {"protocol", "model_access", "tasks", "classes"})
        self.assertEqual(info["protocol"], PROTOCOL)
        self.assertEqual(info["model_access"], "host_gateway_rpc")
        self.assertEqual(info["tasks"], ["classification", "regression"])
        # classes 顺序的落点：describe 从 manifest.json 读，不另存第二份。
        manifest_path = os.path.join(os.path.dirname(APP), "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            block = json.load(handle)["classes_contract"]
        self.assertEqual(info["classes"], block)
        self.assertEqual(info["classes"]["labels"][:2], ["Ethanol", "Ethylene"])
        self.assertEqual(len(info["classes"]["labels"]), 6)
        self.assertIn("字典序", info["classes"]["rule"])
        self.assertIn("metadata.classes", info["classes"]["rule_short"])

    def test_describe_import_has_no_side_effects(self):
        rc, out, err = fake_host.run("describe", env={"CONTEST_ARTIFACT_PATH": "/nonexistent/x.npz"})
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out[0])["protocol"], PROTOCOL)

    def test_describe_rejects_args(self):
        # fake_host 把整串当一个 argv，因此这里实际是"未知命令"，按契约保留 exit 2。
        rc, out, err = fake_host.run("describe extra")
        self.assertEqual(rc, 2, err)
        self.assertEqual(out, [])

    def test_no_bytecode_written_into_app_dir(self):
        before = _snapshot(APP)
        fake_host.run("describe")
        fake_host.run("predict", _cls_row(1, 0, 0, 0, "p") + "\n",
                      env={"CONTEST_ARTIFACT_PATH": self.cls_npz}, cwd=self.tmp)
        after = _snapshot(APP)
        self.assertEqual(before, after, "导入后 /app 不应出现新文件")
        stray = [p for p in after if p.endswith(".pyc")]
        self.assertEqual(stray, [], "不应往 /app 落任何 .pyc：%r" % (stray,))
        stray = [p for p in after if p.endswith(".pyc")]
        self.assertEqual(stray, [], "不应往 /app 落任何 .pyc：%r" % (stray,))

    # ---------------------------------------------------------- train

    def test_train_is_stub(self):
        # 本地 harness 没有 SDK 的 start 帧，构造不出 rnd；真机由宿主调用 train(rnd)，
        # 一轮流程在 train_round.run 里，P2 用 tests/test_p2.py 的 rnd 替身验收。
        rc, out, err = fake_host.run("train")
        self.assertEqual(rc, 3, err)
        self.assertEqual(out, [], "train 不应往 stdout 写任何东西")
        self.assertIn("train_round.run", err)

    # ---------------------------------------------------------- predict：正常路径

    def test_predict_classification_rows(self):
        rows = [
            _cls_row(1, 0, 0, 0, "r0"),
            _cls_row(0, 1, 0, 0, "r1"),
            _cls_row(0, 0, 1, 0, "r2"),
            _cls_row(0, 0, 0, 1, "r3"),
        ]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), len(rows), err)

        for line, want in zip(out, ["c0", "c1", "c2", "c3"]):
            payload = json.loads(line)
            self.assertEqual(set(payload.keys()), {"outputs"})
            outputs = payload["outputs"]
            self.assertEqual([o["label"] for o in outputs], CLASSES)
            scores = [o["score"] for o in outputs]
            for s in scores:
                self.assertTrue(0.0 <= s <= 1.0, s)
            top = max(scores)
            self.assertEqual(scores.count(top), 1, "最高分必须唯一")
            self.assertEqual(outputs[scores.index(top)]["label"], want)

    def test_predict_stdout_is_pure_jsonl(self):
        rows = [_cls_row(1, 0, 0, 0, "a"), _cls_row(0, 0, 0, 1, "b")]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 2)
        for line in out:
            self.assertEqual(list(json.loads(line).keys()), ["outputs"])

    def test_predict_flushes_per_line(self):
        rows = [_cls_row(1, 0, 0, 0, "s0"), _cls_row(0, 1, 0, 0, "s1")]
        rc, replies, err = fake_host.stream_predict(
            rows, env={"CONTEST_ARTIFACT_PATH": self.cls_npz}, cwd=self.tmp)
        self.assertEqual(rc, 0, err)
        for reply in replies:
            self.assertTrue(reply, "每一行都必须在读下一行之前拿到响应")
            self.assertIn("outputs", json.loads(reply))

    def test_predict_regression_clips_negative(self):
        rows = [
            json.dumps({"task": "regression", "inputs": {"x": 5.0, "y": 9.0}}),
            json.dumps({"task": "regression", "inputs": {"x": 0.0, "y": 9.0}}),
        ]
        rc, out, err = self._predict(rows, self.reg_npz)
        self.assertEqual(rc, 0, err)
        values = [json.loads(line)["outputs"]["value"] for line in out]
        self.assertAlmostEqual(values[0], 3.0)
        self.assertEqual(values[1], 0.0)

    def test_predict_tie_break_is_deterministic(self):
        rows = [_cls_row(0, 0, 0, 0, "z")]
        rc1, out1, err1 = self._predict(rows, self.cls_npz)
        rc2, out2, err2 = self._predict(rows, self.cls_npz)
        self.assertEqual(rc1, 0, err1)
        self.assertEqual(rc2, 0, err2)
        self.assertEqual(out1, out2, "同样的输入必须给出同样的输出")
        outputs = json.loads(out1[0])["outputs"]
        scores = [o["score"] for o in outputs]
        self.assertEqual(scores.count(max(scores)), 1)
        self.assertEqual(outputs[scores.index(max(scores))]["label"], "c0")

    # ---------------------------------------------------------- predict：总输出

    def test_missing_task_uses_artifact_task(self):
        rows = [json.dumps({"inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}, "id": "n"})]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1)
        outputs = json.loads(out[0])["outputs"]
        scores = [o["score"] for o in outputs]
        self.assertEqual(outputs[scores.index(max(scores))]["label"], "c0")
        self.assertIn("上下文与行内 task 都缺失", err, "两者都缺时必须标注用了产物 task")

    def test_context_task_wins_over_row_task(self):
        # 上下文 classification、行 task regression：上下文优先，按分类正常推断，不走兜底。
        rows = [json.dumps({"task": "regression",
                            "inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}, "id": "ctx"})]
        rc, out, err = self._predict(rows, self.cls_npz,
                                     extra_env={"CONTEST_TASK": "classification"})
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1)
        outputs = json.loads(out[0])["outputs"]
        scores = [o["score"] for o in outputs]
        self.assertEqual(outputs[scores.index(max(scores))]["label"], "c0")
        self.assertNotIn("声明默认", err, "上下文优先时不应触发兜底")

    def test_fallback_does_not_disturb_later_rows(self):
        # 兜底分支不得更新流式状态：中间夹一个坏行，不能改变后面同一行的输出。
        good = _cls_row(1, 0, 0, 0, "same")
        rc, out, err = self._predict([good, "{oops", good], self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], out[2], "坏行的兜底不得影响后续行的归一化")
        self.assertIn("声明默认", err)

    def test_malformed_json_still_emits_one_line(self):
        rc, out, err = self._predict(["{oops"], self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1, "坏行也不能缺行")
        outputs = json.loads(out[0])["outputs"]
        self.assertEqual(outputs[CLASSES.index(PRIOR_CLASS)]["score"], 1.0)
        self.assertIn("声明默认", err)

    def test_unknown_task_value_falls_back(self):
        rows = [json.dumps({"task": "clustering", "inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}})]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1)
        self.assertIn("声明默认", err)

    def test_row_task_mismatch_falls_back_not_aborts(self):
        rows = [json.dumps({"task": "regression", "inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}})]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(json.loads(out[0])["outputs"]), len(CLASSES))
        self.assertIn("声明默认", err)

    def test_classification_declared_default_shape(self):
        rows = [_cls_row("not-a-number", 0, 0, 0, "d1")]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        outputs = json.loads(out[0])["outputs"]
        self.assertEqual([o["label"] for o in outputs], CLASSES)
        by_label = {o["label"]: o["score"] for o in outputs}
        self.assertEqual(by_label[PRIOR_CLASS], 1.0, "兜底回先验多数类且 score=1")
        self.assertEqual(sum(1 for s in by_label.values() if s == 1.0), 1)
        for label, score in by_label.items():
            if label != PRIOR_CLASS:
                self.assertEqual(score, 0.0)

    def test_regression_declared_default_uses_prior_mean(self):
        rows = [json.dumps({"task": "regression", "inputs": {"x": "boom", "y": 1.0}})]
        rc, out, err = self._predict(rows, self.reg_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(out[0])["outputs"]["value"], PRIOR_MEAN)
        self.assertIn("声明默认", err)

    def test_declared_default_is_deterministic(self):
        rows = [json.dumps({"task": "regression", "inputs": {"x": "boom", "y": 1.0}})]
        rc1, out1, _ = self._predict(rows, self.reg_npz)
        rc2, out2, _ = self._predict(rows, self.reg_npz)
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertEqual(out1, out2)

    def test_total_output_one_line_per_row(self):
        rows = [
            _cls_row(1, 0, 0, 0, "ok-1"),
            "{oops",
            json.dumps({"inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}, "id": "no-task"}),
            json.dumps({"task": "regression", "inputs": {"f0": 1, "f1": 0, "f2": 0, "f3": 0}}),
            _cls_row(0, 0, 1, 0, "ok-2"),
            json.dumps({"task": "classification", "inputs": {"f0": 1}, "id": "short"}),
        ]
        rc, out, err = self._predict(rows, self.cls_npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), len(rows), "每个非空输入行必须对应一行输出")
        for line in out:
            self.assertEqual(list(json.loads(line).keys()), ["outputs"])

    # ---------------------------------------------------------- predict：逐行兜底，不再批级退出

    def test_context_conflict_falls_back_per_row(self):
        rc, out, err = self._predict([_cls_row(1, 0, 0, 0, "c")], self.cls_npz,
                                     extra_env={"CONTEST_TASK": "regression"})
        self.assertEqual(rc, 0, "上下文冲突不再整批退出：%s" % err)
        self.assertEqual(len(out), 1, "冲突行也必须有一行输出")
        payload = json.loads(out[0])["outputs"]
        self.assertEqual([item["label"] for item in payload], list(CLASSES))
        self.assertEqual(sum(item["score"] for item in payload), 1.0)
        self.assertEqual(max(item["score"] for item in payload), 1.0)
        self.assertIn("声明默认", err)

    def test_mixed_row_tasks_only_conflicting_rows_fall_back(self):
        base = [_cls_row(1, 0, 0, 0, "a"), _cls_row(0, 1, 0, 0, "b"),
                _cls_row(0, 0, 1, 0, "c")]
        plain_rc, plain_out, plain_err = self._predict(base, self.cls_npz)
        self.assertEqual(plain_rc, 0, plain_err)
        mixed = [json.loads(line) for line in base]
        mixed[1]["task"] = "regression"
        rc, out, err = self._predict([json.dumps(row) for row in mixed], self.cls_npz)
        self.assertEqual(rc, 0, "混行批次不得整批退出：%s" % err)
        self.assertEqual(len(out), 3, "冲突行不得吞行")
        self.assertEqual(out[0], plain_out[0], "正常行输出不受冲突行影响")
        self.assertEqual(out[2], plain_out[2], "冲突行之后的正常行继续处理")
        fallback = json.loads(out[1])["outputs"]
        self.assertEqual([item["label"] for item in fallback], list(CLASSES))
        self.assertEqual(sum(item["score"] for item in fallback), 1.0)
        self.assertEqual(sum(1 for item in fallback if item["score"] == 1.0), 1,
                         "声明默认必须只有一个最高分")
        self.assertIn("第 2 行", err, "兜底必须能定位到具体行")

    def test_predict_without_artifact_fails_fast(self):
        rc, out, err = fake_host.run("predict", _cls_row(1, 0, 0, 0, "n") + "\n", cwd=self.tmp)
        self.assertEqual(rc, 1, err)
        self.assertEqual(out, [])
        self.assertIn("CONTEST_ARTIFACT_PATH", err)

    def test_predict_bad_checkpoint_shape_is_rejected(self):
        rc, out, err = self._predict([_cls_row(1, 0, 0, 0, "x")], self.bad_npz)
        self.assertEqual(rc, 1, err)
        self.assertEqual(out, [])

    def test_predict_skips_blank_lines(self):
        text = _cls_row(1, 0, 0, 0, "b0") + "\n\n   \n" + _cls_row(0, 1, 0, 0, "b1") + "\n"
        rc, out, err = fake_host.run("predict", text,
                                     env={"CONTEST_ARTIFACT_PATH": self.cls_npz}, cwd=self.tmp)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 2)

    # ---------------------------------------------------------- 候选目录自检

    def test_candidate_dir_budget(self):
        limit_files, limit_bytes = 64, 64 * 1024 * 1024
        count, total = 0, 0
        for dirpath, _, filenames in os.walk(APP):
            for name in filenames:
                count += 1
                total += os.path.getsize(os.path.join(dirpath, name))
        self.assertLessEqual(count, limit_files, "候选目录文件数 %d 超过 64" % count)
        self.assertLessEqual(total, limit_bytes, "候选目录 %d 字节超过 64 MiB" % total)


def _official_cls(path, classes=CLASSES, *, mean=None, scale=None, w1=None, b1=None,
                  w2=None, b2=None, include_classes=True, paths=True):
    """官方基线格式（w1/w2/b1/b2/mean/scale + metadata）。"""
    n, hidden = len(FEATURES), 2
    meta = {"task": "classification", "hidden": hidden,
            "paths": [json.dumps([["key", name]], separators=(",", ":")) for name in FEATURES]}
    if include_classes:
        meta["classes"] = list(classes)
    if not paths:
        meta.pop("paths")
    np.savez(
        path,
        w1=np.asarray(w1 if w1 is not None else np.zeros((n, hidden)), dtype=float),
        b1=np.asarray(b1 if b1 is not None else np.zeros(hidden), dtype=float),
        w2=np.asarray(w2 if w2 is not None else np.zeros((hidden, len(classes))), dtype=float),
        b2=np.asarray(b2 if b2 is not None else np.zeros(len(classes)), dtype=float),
        mean=np.asarray(mean if mean is not None else np.zeros(n), dtype=float),
        scale=np.asarray(scale if scale is not None else np.ones(n), dtype=float),
        metadata=np.array(json.dumps(meta, sort_keys=True)),
    )


class OfficialCheckpointTest(unittest.TestCase):
    """官方基线检查点：`/model` 下没有 task/coef，装载器必须走官方前向。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _predict(self, rows, npz, extra_env=None):
        env = {"CONTEST_ARTIFACT_PATH": npz, "CONTEST_MODEL_DIR": "/model"}
        if extra_env:
            env.update(extra_env)
        return fake_host.run("predict", "\n".join(rows) + "\n", env=env, cwd=self.tmp)

    def _cls_npz(self, name, **kwargs):
        path = os.path.join(self.tmp, name)
        _official_cls(path, **kwargs)
        return path

    def test_official_forward_matches_formula(self):
        # mean/scale 在输入侧、tanh 在隐层：两者任一错位都会让下面的数值对不上。
        w1 = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        w2 = [[1.0, 0.0], [0.0, 1.0]]
        npz = self._cls_npz("formula.npz", classes=["c0", "c1"],
                            mean=[10.0, 0, 0, 0], scale=[2.0, 1, 1, 1],
                            w1=w1, b1=[2.0, -1.0], w2=w2, b2=[0.0, 0.0])
        rows = [_cls_row(10, 0, 0, 0, "a"), _cls_row(12, 0, 0, 0, "b")]
        rc, out, err = self._predict(rows, npz)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(out), 2, err)

        def expect(hidden_input, second):
            a = np.tanh(hidden_input)
            b = np.tanh(second)
            e0, e1 = np.exp(a), np.exp(b)
            return float(e0 / (e0 + e1))

        got = json.loads(out[0])["outputs"]
        self.assertEqual([item["label"] for item in got], ["c0", "c1"])
        self.assertAlmostEqual(got[0]["score"], expect(0.0 + 2.0, -1.0), places=12)
        self.assertAlmostEqual(got[1]["score"], 1.0 - expect(0.0 + 2.0, -1.0), places=12)
        # (12-10)/2 = 1，隐层输入 1+2=3
        self.assertAlmostEqual(json.loads(out[1])["outputs"][0]["score"],
                               expect(3.0, -1.0), places=12)

    def test_classes_order_comes_from_metadata(self):
        # metadata 的顺序就是 w2 的列序；字典序会错位，必须原样使用。
        classes = ["z-class", "a-class", "m-class"]
        w2 = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        npz = self._cls_npz("order.npz", classes=classes, w1=np.zeros((4, 2)),
                            b1=[1.0, 0.0], w2=w2, b2=[0.0, 0.0, 0.0])
        rc, out, err = self._predict([_cls_row(0, 0, 0, 0, "o")], npz)
        self.assertEqual(rc, 0, err)
        outputs = json.loads(out[0])["outputs"]
        self.assertEqual([item["label"] for item in outputs], classes)
        self.assertEqual(outputs[0]["score"], max(item["score"] for item in outputs))

    def test_missing_metadata_classes_uses_sorted_train_labels(self):
        data = os.path.join(self.tmp, "data-fallback")
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, "train.jsonl"), "w", encoding="utf-8") as handle:
            for i, answer in enumerate(["zzz", "aaa", "mmm", "zzz"]):
                handle.write(json.dumps({"id": "t%d" % i,
                                         "inputs": {"f0": 0, "f1": 0, "f2": 0, "f3": 0},
                                         "answer": answer}) + "\n")
        npz = self._cls_npz("fallback.npz", classes=["zzz", "aaa", "mmm"],
                            include_classes=False, w1=np.zeros((4, 2)),
                            b1=[1.0, 0.0], w2=np.zeros((2, 3)), b2=[0.0, 0.0, 0.0])
        rc, out, err = self._predict([_cls_row(0, 0, 0, 0, "f")], npz,
                                     extra_env={"CONTEST_DATA_DIR": data})
        self.assertEqual(rc, 0, err)
        # 首次出现序是 zzz/aaa/mmm；兜底写死为字典序 aaa/mmm/zzz。
        self.assertEqual([item["label"] for item in json.loads(out[0])["outputs"]],
                         ["aaa", "mmm", "zzz"])
        self.assertIn("字典序兜底", err)
        self.assertIn("可能与 w2 列序错位", err)

    def test_feature_paths_must_match_metadata(self):
        npz = self._cls_npz("paths.npz", w1=np.zeros((4, 2)),
                            b1=[1.0, 0.0], w2=np.zeros((2, 6)), b2=np.zeros(6))
        row = json.dumps({"task": "classification",
                          "inputs": {"zzz": 1, "f1": 0, "f2": 0, "f3": 0}, "id": "p"})
        rc, out, err = self._predict([row], npz)
        self.assertEqual(rc, 0, "行级 schema 不符要走声明默认，不吞行")
        self.assertEqual(len(out), 1)
        self.assertIn("声明默认", err)

    def test_official_regression_uses_target_scale_and_clip(self):
        path = os.path.join(self.tmp, "reg-official.npz")
        meta = {"task": "regression", "hidden": 1,
                "paths": [json.dumps([["key", name]], separators=(",", ":")) for name in FEATURES],
                "target_mean": -1.0, "target_scale": 2.0}
        np.savez(path,
                 w1=np.array([[1.0], [0.0], [0.0], [0.0]]), b1=np.zeros(1),
                 w2=np.ones((1, 1)), b2=np.zeros(1),
                 mean=np.zeros(4), scale=np.ones(4),
                 metadata=np.array(json.dumps(meta, sort_keys=True)))
        rows = [json.dumps({"task": "regression",
                            "inputs": {"f0": 1.5, "f1": 0, "f2": 0, "f3": 0}, "id": "r"}),
                json.dumps({"task": "regression",
                            "inputs": {"f0": -1.5, "f1": 0, "f2": 0, "f3": 0}, "id": "n"})]
        rc, out, err = self._predict(rows, path)
        self.assertEqual(rc, 0, err)
        self.assertAlmostEqual(json.loads(out[0])["outputs"]["value"],
                               -1.0 + 2.0 * float(np.tanh(1.5)), places=9)
        self.assertEqual(json.loads(out[1])["outputs"]["value"], 0.0, "负值必须 clip 到 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-
"""v2 P2 验收：SDK 训练循环（一轮 = 一次 train(rnd)），用 rnd 替身驱动。

覆盖任务书里的三条硬约束与账本落地：
1. 每轮恰好一次真实 `rnd.model()`，显式 `max_tokens`，消息里不重复注入 `rnd.prompt`。
2. `receipt.text` 为空 / 不可解析 → 退回"续训父检查点"，并记 stderr。
3. `finish(model_call_ids=...)` 只来自真实调用（`rnd.call_ids`）。
4. 账本五字段 hypothesis / change_summary / configuration / reason / state 落盘。
5. `reason` 只引平台返回的数字，不混入我们自己算的检测值。
6. 第 1 轮不再强制父候选：模型提案直接生效并真训（旧的自设 "R1 一律交父权重" 已移除）。
7. 同一份 history 跑两次，提案与账本一致（可复现性检验）。
8. 父检查点只走 `rnd.parent.path()` 并自核 sha256；摘要不符整轮失败。
"""

import contextlib
import io
import json
import os
import pathlib
import random
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True  # 测试自身不得往 app/ 落 pyc

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(_HERE), "app")
if APP not in sys.path:
    sys.path.insert(0, APP)

import train_round  # noqa: E402

PROMPT_SENTINEL = "PROMPT-SENTINEL-9f3a71c0"
FEATURES = ["f0", "f1", "f2", "f3"]


# ---------------------------------------------------------------- 替身

class FakeReceipt(object):
    def __init__(self, text, call_id):
        self.text = text
        self.call_id = call_id
        self.response = {"text": text}


class FakeParent(object):
    """父产物：只允许通过 path(rel) 拿文件，并像 SDK 一样核对摘要。"""

    def __init__(self, root, declared=None, rel="checkpoint.npz"):
        self.root = root
        self.checkpoint_sha256 = declared
        self.checkpoint_paths = [rel]
        self.calls = []

    def path(self, rel=None):
        rel = rel or self.checkpoint_paths[0]
        self.calls.append(rel)
        full = os.path.join(self.root, rel)
        if not os.path.isfile(full):
            raise FileNotFoundError("父产物里没有 %s" % rel)
        if self.checkpoint_sha256 is not None:
            import hashlib
            h = hashlib.sha256()
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != self.checkpoint_sha256:
                raise RuntimeError("PARENT_DIGEST_MISMATCH")
        return pathlib.Path(full)


class FakeRnd(object):
    """SDK `rnd` 的最小替身：只实现本轮流程真正用到的那几个方法。"""

    def __init__(self, parent_root, work_root, data_dir=None, state_src=None,
                 round_index=1, history=None, reply="", prompt=PROMPT_SENTINEL,
                 max_output_tokens=1024, declared_digest=None, call_id=None,
                 artifact_id=None):
        self.parent = FakeParent(parent_root, declared=declared_digest)
        self.work_root = work_root
        self.data_dir = data_dir
        self.state_src = state_src
        self.round_index = round_index
        self.history_records = list(history or [])
        self.reply = reply
        self.prompt = prompt
        self.prompt_version = "industrial-workspace-flexible-v3"
        self.max_output_tokens = max_output_tokens
        self.dataset_key = "ds-1"
        self.task = ({"public_projection": {"train": os.path.join(data_dir, "train.jsonl")}}
                     if data_dir else {})
        self.call_ids = []
        self.calls = []
        self.history_calls = 0
        self.submits = []
        self.finishes = []
        self.saves = []
        self.state_calls = []
        self._call_seq = 0
        self._emit_call_id = call_id is not False
        self.artifact_id = artifact_id or ("art-r%d" % round_index)

    # --- SDK 接口 ---
    def history(self):
        self.history_calls += 1
        return [dict(record) for record in self.history_records]

    def model(self, messages, max_tokens=None, temperature=None):
        self._call_seq += 1
        call_id = None
        if self._emit_call_id:
            call_id = "call-%d" % self._call_seq
            self.call_ids.append(call_id)
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature})
        return FakeReceipt(self.reply, call_id)

    def submit_candidate(self, candidate_dir, **kwargs):
        self.submits.append({"candidate_dir": candidate_dir, "kwargs": kwargs})
        return {"artifact_id": self.artifact_id,
                "validation": {"score": 0.6123, "n": 87}}

    def state_workspace(self, rel):
        dest = os.path.join(self.work_root, "state-copy-%d" % (len(self.state_calls) + 1))
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        if self.state_src and os.path.isdir(self.state_src):
            shutil.copytree(self.state_src, dest)
        else:
            os.makedirs(dest)
        self.state_calls.append({"rel": rel, "path": dest})
        return dest

    def save_state(self, path):
        self.saves.append(path)

    def finish(self, **kwargs):
        self.finishes.append(kwargs)


# ---------------------------------------------------------------- 夹具

def _write_parent(root, n_features=len(FEATURES), hidden=5):
    """父检查点夹具：写成**官方基线格式**（w1/w2/b1/b2/mean/scale/metadata），
    而不是 v2 自有格式（task/coef）。平台给 A 的父产物就是官方 MLP。"""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "checkpoint.npz")
    rng = np.random.RandomState(0)
    features = FEATURES[:n_features]
    paths = [json.dumps([["key", name]]) for name in features]
    metadata = json.dumps({
        "task": "regression",
        "paths": paths,
        "hidden": hidden,
        "target_mean": 0.5,
        "target_scale": 1.0,
    })
    np.savez(
        path,
        w1=rng.normal(size=(n_features, hidden)),
        b1=rng.normal(size=hidden),
        w2=rng.normal(size=(hidden, 1)),
        b2=rng.normal(size=1),
        mean=np.zeros(n_features),
        scale=np.ones(n_features),
        metadata=np.array(metadata),
    )
    return path


def _write_v2_parent(root, n_features=len(FEATURES)):
    """v2 自有格式父检查点（task/coef/intercept/feature_order）。

    只有这种父产物能被动作臂改写：官方 MLP 父产物的 predict 走官方前向、不读 action，
    把动作写进去等于造假候选（`_build_candidate` 现在会明确拒绝）。
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "checkpoint.npz")
    rng = np.random.RandomState(1)
    np.savez(
        path,
        task=np.array("regression"),
        coef=rng.normal(scale=0.1, size=n_features),
        intercept=np.array(0.5),
        feature_order=np.array(FEATURES[:n_features]),
        prior_mean=np.array(0.5),
    )
    return path


def _write_data(dirpath, shift=0.0, n=120, labels=False):
    os.makedirs(dirpath, exist_ok=True)
    rng = random.Random(7)
    for name, offset in (("train.jsonl", 0.0), ("validation.jsonl", shift)):
        with open(os.path.join(dirpath, name), "w", encoding="utf-8", newline="\n") as handle:
            for i in range(n):
                inputs = {"f%d" % k: round(rng.gauss(offset, 1.0), 6) for k in range(4)}
                row = {"id": "%s-%d" % (name, i), "inputs": inputs}
                if labels:
                    # 回归标签：可学的线性信号 + 小噪声，供续训用例使用
                    row["answer"] = round(3.0 * inputs["f0"] - 2.0 * inputs["f1"] + 0.5, 6)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(rnd, **kwargs):
    """跑一轮并把 stderr 收回来（验证"记 stderr"用）。

    `work_dir` 显式指向本次用例目录：真机上由 `/workspace` 兜底，测试不污染仓库根目录。
    """
    kwargs.setdefault("work_dir", rnd.work_root)
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        ledger = train_round.run(rnd, **kwargs)
    return ledger, buffer.getvalue()


class P2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.parent_root = os.path.join(cls.tmp, "parent")
        cls.parent_npz = _write_parent(cls.parent_root)
        cls.v2_parent_root = os.path.join(cls.tmp, "parent-v2")
        cls.v2_parent_npz = _write_v2_parent(cls.v2_parent_root)
        cls.data = os.path.join(cls.tmp, "data")
        _write_data(cls.data)
        cls.labelled = os.path.join(cls.tmp, "data-labelled")
        _write_data(cls.labelled, labels=True)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _case(self, name, **kwargs):
        work = os.path.join(self.tmp, "work", name)
        os.makedirs(work, exist_ok=True)
        parent_root = kwargs.pop("parent_root", self.parent_root)
        kwargs.setdefault("data_dir", self.data)
        kwargs.setdefault("reply",
                          '{"arm":"A","params":{},"hypothesis":"账本为空，按漂移检测选标准化臂",'
                          '"kill_condition":"若平台分低于父候选则回退","confidence":"medium"}')
        return FakeRnd(parent_root, work, **kwargs), work

    # ---------------------------------------------------------- R1 主流程

    def test_r1_adopts_proposal_and_trains(self):
        rnd, work = self._case("r1", data_dir=self.labelled)
        ledger, err = _run(rnd)

        self.assertEqual(len(rnd.submits), 1, "应只提交一个候选")
        self.assertEqual(len(rnd.finishes), 1)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "A", "第 1 轮不再强制父候选")
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(config["weights_changed"], True, "第 1 轮必须真的改权重")
        self.assertIsNone(config["action"])
        self.assertIn("续训", ledger["hypothesis"])
        self.assertNotIn("R1 固定续训父检查点", err, "旧的强制父候选规则必须已移除")

        cand = os.path.join(rnd.submits[0]["candidate_dir"], "checkpoint.npz")
        with open(cand, "rb") as a, open(self.parent_npz, "rb") as b:
            self.assertNotEqual(a.read(), b.read(), "第 1 轮候选必须与父产物不同")
        self.assertEqual(ledger["configuration"]["weights_changed"], True)

        kwargs = rnd.submits[0]["kwargs"]
        self.assertEqual(kwargs["checkpoint_paths"], ["checkpoint.npz"])
        self.assertEqual(kwargs["code_paths"], [])
        self.assertEqual(kwargs["predict_argv"], ["python3", "/app/harness.py", "predict"])
        self.assertLessEqual(len(kwargs["hypothesis"].encode("utf-8")), 4000)
        self.assertLessEqual(len(kwargs["change_summary"].encode("utf-8")), 4000)
        self.assertLessEqual(
            len(json.dumps(kwargs["configuration"], ensure_ascii=False).encode("utf-8")),
            64 * 1024)

        budget = ledger["configuration"]["budget"]
        self.assertEqual(budget["files"], 2, "续训候选含 checkpoint.npz 与 train_report.json")
        self.assertLessEqual(budget["bytes"], 64 * 1024 * 1024)

    def test_model_calls_use_explicit_budget(self):
        rnd, _ = self._case("once")
        _run(rnd)
        # 第 1 轮真训后会有红队/修复的第二问，但仍受修复预算约束。
        self.assertGreaterEqual(len(rnd.calls), 1)
        self.assertLessEqual(len(rnd.calls), 2, "模型调用不得超过修复预算")
        for call in rnd.calls:
            self.assertIsInstance(call["max_tokens"], int)
            self.assertGreater(call["max_tokens"], 0)
            self.assertEqual(call["temperature"], 0)
            blob = json.dumps(call["messages"], ensure_ascii=False)
            self.assertNotIn(PROMPT_SENTINEL, blob, "不得重复注入 rnd.prompt")
        self.assertEqual(rnd.history_calls, 1, "rnd.history() 是经验库读取入口")

    def test_max_tokens_respects_round_cap(self):
        rnd, _ = self._case("cap", max_output_tokens=128)
        _run(rnd)
        self.assertEqual(rnd.calls[0]["max_tokens"], 128)

    def test_model_proposal_recorded_when_training_falls_back(self):
        rnd, _ = self._case("r1-proposal")
        ledger, _ = _run(rnd)
        model = ledger["configuration"]["model"]
        self.assertEqual(model["proposal"], "A")
        self.assertEqual(model["status"], "ok")
        self.assertEqual(model["call_id"], "call-1")
        self.assertEqual(ledger["configuration"]["candidate"], "PARENT")
        self.assertEqual(ledger["configuration"]["policy"], "fallback_parent_after_rule")

    # ---------------------------------------------------------- 硬约束：空回复

    def test_empty_receipt_falls_back_to_rule(self):
        rnd, _ = self._case("empty", round_index=2, reply="",
                            parent_root=self.v2_parent_root)
        ledger, err = _run(rnd)
        self.assertIn("退回规则选择器", err)
        self.assertEqual(ledger["configuration"]["candidate"], "A")
        self.assertEqual(ledger["configuration"]["policy"], "fallback_rule")
        self.assertEqual(ledger["configuration"]["model"]["status"], "empty")

    def test_unparseable_receipt_falls_back_to_rule(self):
        rnd, _ = self._case("unusable", round_index=3, reply="我也不确定该怎么选。",
                            parent_root=self.v2_parent_root)
        ledger, err = _run(rnd)
        self.assertIn("模型提案不可用", err)
        self.assertEqual(ledger["configuration"]["candidate"], "A")
        self.assertEqual(ledger["configuration"]["policy"], "fallback_rule")

    # ---------------------------------------------------------- 模型真提案

    def test_round2_uses_model_proposal_over_detector_rule(self):
        # 动作臂只能改写 v2 父产物：官方 MLP 父产物上动作无处安放（见下一条用例）。
        rnd, _ = self._case(
            "round2", round_index=2, parent_root=self.v2_parent_root,
            reply='{"arm":"C","params":{},"hypothesis":"账本为空，先验证不归一化的线性头",'
                  '"kill_condition":"若平台分低于父候选则回退","confidence":"medium"}')
        ledger, _ = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "C")
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(config["action"], "identity")
        self.assertIn("线性头", ledger["hypothesis"], "hypothesis 必须描述实际提交的候选")
        self.assertEqual(config["model"]["hypothesis"], "账本为空，先验证不归一化的线性头")
        self.assertEqual(len(rnd.calls), 2, "提案 + 红队各一次")
        self.assertNotEqual(config["selector"]["candidate"], "C",
                            "检测器建议与模型提案不同，最终采用模型提案才算真提案")

        cand = os.path.join(rnd.submits[0]["candidate_dir"], "checkpoint.npz")
        with np.load(cand, allow_pickle=False) as archive:
            self.assertEqual(str(archive["action"]), "identity")
            # identity 在 actions.REGISTRY 里不接受任何参数，写 eps 会让候选在装载期被拒
            self.assertEqual(json.loads(str(archive["action_params"])), {})
            with np.load(self.v2_parent_npz, allow_pickle=False) as parent:
                np.testing.assert_array_equal(archive["coef"], parent["coef"])

    def test_official_parent_without_labels_falls_back_to_parent(self):
        """官方父产物要真训；数据缺 answer 时训练无法成立，必须干净退回 PARENT，
        不能产出"官方键 + action"的混合候选（装载器会静默忽略 action）。"""
        rnd, _ = self._case(
            "official-arm", round_index=2, reply='{"arm":"C","params":{},"hypothesis":"换线性头",'
                                                 '"kill_condition":"低于父候选则回退","confidence":"low"}')
        ledger, err = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "PARENT")
        self.assertEqual(config["policy"], "fallback_parent_after_rule")
        self.assertIsNone(config["action"])
        self.assertTrue(any("续训失败" in item for item in config["cage"]["build_errors"]),
                        "构建错误必须写清是训练失败，不能静默降级")
        self.assertIn("续训失败", err)

        cand = os.path.join(rnd.submits[0]["candidate_dir"], "checkpoint.npz")
        with open(cand, "rb") as a, open(self.parent_npz, "rb") as b:
            self.assertEqual(a.read(), b.read(), "退回 PARENT 后必须是父产物原文")
        with np.load(cand, allow_pickle=False) as archive:
            self.assertIn("w1", archive.files)
            self.assertNotIn("action", archive.files, "官方父产物里绝不能混入 action")

    def test_official_parent_trains_and_keeps_official_format(self):
        """有标签时，官方父产物上的臂必须真训：权重变化、格式不变、不写 action。"""
        data = os.path.join(self.tmp, "data-label")
        _write_data(data, labels=True)
        rnd, _ = self._case(
            "official-train", round_index=2, data_dir=data,
            reply='{"arm":"C","params":{},"hypothesis":"账本为空，冻结隐层重解输出头",'
                  '"kill_condition":"低于父候选则回退","confidence":"medium"}')
        ledger, _ = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["candidate"], "C")
        self.assertEqual(config["policy"], "model_proposal")
        self.assertEqual(config["cage"]["build_errors"], [])
        self.assertIsNone(config["action"], "官方候选没有动作位")
        self.assertIn("续训", ledger["hypothesis"])

        cand = os.path.join(rnd.submits[0]["candidate_dir"], "checkpoint.npz")
        with np.load(cand, allow_pickle=False) as archive:
            self.assertNotIn("action", archive.files)
            self.assertEqual(archive.files.count("metadata"), 1)
            with np.load(self.parent_npz, allow_pickle=False) as parent:
                self.assertTrue(config["weights_changed"], "head 臂应解出新的输出层")
                np.testing.assert_array_equal(archive["w1"], parent["w1"],
                                              "head 臂冻结隐层，w1 必须逐位不变")
                self.assertFalse(np.array_equal(archive["w2"], parent["w2"]))
                self.assertEqual(archive["w1"].shape, parent["w1"].shape)

    # ---------------------------------------------------------- 账本

    def test_ledger_five_fields_and_reason_platform_only(self):
        rnd, work = self._case("ledger")
        ledger, _ = _run(rnd)

        for field in ("hypothesis", "change_summary", "configuration", "reason", "state"):
            self.assertIn(field, ledger, "账本缺字段 %s" % field)
        self.assertIn("validation.score=0.6123", ledger["reason"])
        self.assertIn("validation.n=87", ledger["reason"])
        self.assertNotIn("kl", ledger["reason"].lower())
        self.assertNotIn("mahalanobis", ledger["reason"].lower())
        self.assertNotIn("bbsd", ledger["reason"].lower())
        kl = ledger["configuration"]["detectors"]["kl"]["value"]
        self.assertNotIn(str(kl), ledger["reason"], "reason 不得混入我们自算的检测值")

        path = os.path.join(work, "round-1", "ledger.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertEqual(on_disk["reason"], ledger["reason"])

    def test_finish_uses_real_call_ids_and_selected_artifact(self):
        rnd, _ = self._case("finish")
        ledger, _ = _run(rnd)
        finish = rnd.finishes[0]
        self.assertEqual(finish["model_call_ids"][0], "call-1")
        self.assertEqual(finish["model_call_ids"], rnd.call_ids)
        self.assertEqual(finish["selected_artifact_id"], rnd.artifact_id)
        self.assertEqual(finish["history_ids"], [])
        self.assertEqual(finish["reason"], ledger["reason"])
        self.assertEqual(finish["configuration"], ledger["configuration"])
        self.assertEqual(finish["hypothesis"], ledger["hypothesis"])

    def test_history_ids_flow_into_finish(self):
        history = [{"id": "h1", "round_index": 0, "hypothesis": "旧假设",
                    "change_summary": "旧改动", "artifact_id": "art-0"}]
        rnd, _ = self._case(
            "history", round_index=2, history=history,
            reply='{"arm":"PARENT","params":{},"hypothesis":"账本 h1 显示旧假设未复现，沿用父候选",'
                  '"kill_condition":"若平台分低于父候选则回退","confidence":"low"}')
        ledger, _ = _run(rnd)
        self.assertEqual(rnd.finishes[0]["history_ids"], ["h1"])
        self.assertEqual(ledger["configuration"]["history_ids"], ["h1"])
        blob = json.dumps(rnd.calls[0]["messages"], ensure_ascii=False)
        self.assertIn("旧假设", blob, "history 必须真的进入提案上下文")

    # ---------------------------------------------------------- 状态与回滚

    def test_state_carries_previous_artifact_for_rollback(self):
        state_src = os.path.join(self.tmp, "state-src")
        os.makedirs(state_src, exist_ok=True)
        with open(os.path.join(state_src, "round_state.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": "contest.v2.round-state", "round_index": 1,
                       "selected_artifact_id": "art-prev"}, handle)

        rnd, _ = self._case("state", round_index=2, state_src=state_src,
                            reply='{"candidate":"PARENT"}')
        ledger, _ = _run(rnd)

        state = ledger["state"]
        self.assertEqual(state["selected_artifact_id"], "art-r2")
        self.assertEqual(state["previous_selected_artifact_id"], "art-prev")
        self.assertEqual(state["previous_round_index"], 1)
        self.assertEqual(rnd.saves, [rnd.state_calls[0]["path"]])
        with open(os.path.join(rnd.saves[0], "round_state.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["selected_artifact_id"], "art-r2")

    # ---------------------------------------------------------- 父检查点

    def test_parent_digest_mismatch_fails_whole_round(self):
        rnd, _ = self._case("digest", declared_digest="0" * 64)
        with self.assertRaises(train_round.TrainRoundError):
            _run(rnd)
        self.assertEqual(rnd.submits, [], "父检查点对不上时不得提交任何候选")
        self.assertEqual(rnd.finishes, [])

    def test_parent_read_via_sdk_path_not_scan(self):
        rnd, _ = self._case("parent-path")
        _run(rnd)
        self.assertEqual(rnd.parent.calls, ["checkpoint.npz"])

    def test_missing_parent_fails(self):
        work = os.path.join(self.tmp, "work", "no-parent")
        os.makedirs(work, exist_ok=True)
        rnd = FakeRnd(os.path.join(self.tmp, "absent"), work, data_dir=self.data)
        with self.assertRaises(train_round.TrainRoundError):
            _run(rnd)

    # ---------------------------------------------------------- 真实调用号

    def test_finish_forbidden_without_real_call_id(self):
        rnd, _ = self._case("no-call-id", call_id=False)
        with self.assertRaises(train_round.TrainRoundError):
            _run(rnd)
        self.assertEqual(rnd.submits, [])
        self.assertEqual(rnd.finishes, [])

    # ---------------------------------------------------------- 可复现性

    def test_same_history_twice_gives_same_proposal_and_ledger(self):
        history = [{"id": "h1", "round_index": 0, "hypothesis": "旧假设"}]
        reply = ('{"arm":"B","params":{},"hypothesis":"账本 h1 的旧假设需要换臂复核",'
                 '"kill_condition":"若平台分低于父候选则回退","confidence":"medium"}')
        first_rnd, _ = self._case("det-1", round_index=2, history=history, reply=reply,
                                  parent_root=self.v2_parent_root)
        second_rnd, _ = self._case("det-2", round_index=2, history=history, reply=reply,
                                   parent_root=self.v2_parent_root)
        first, _ = _run(first_rnd)
        second, _ = _run(second_rnd)

        self.assertEqual(first["configuration"]["model"]["proposal"],
                         second["configuration"]["model"]["proposal"])
        self.assertEqual(first["configuration"], second["configuration"])
        self.assertEqual(first["hypothesis"], second["hypothesis"])
        self.assertEqual(first["change_summary"], second["change_summary"])
        self.assertEqual(first["reason"], second["reason"])
        self.assertEqual(os.path.basename(first_rnd.submits[0]["candidate_dir"]),
                         os.path.basename(second_rnd.submits[0]["candidate_dir"]),
                         "候选目录名由内容摘要决定，两次必须一致")

    # ---------------------------------------------------------- 数据入口

    def test_data_dir_resolved_from_public_projection(self):
        rnd, _ = self._case("projection")
        rnd.task = {"public_projection": {"train": os.path.join(self.data, "train.jsonl")}}
        ledger, _ = _run(rnd)
        config = ledger["configuration"]
        self.assertEqual(config["dataset_key"], "ds-1")
        self.assertEqual(sorted(config["detectors"]), ["bbsd", "kl", "mahalanobis"])
        self.assertEqual(config["selector"]["rule"] in ("R1", "R2", "R3"), True)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train 阶段的检测与选型入口（v2 P1）。

这是**唯一**会跑批级检测器（KL / 马氏 / BBSD）的地方：读 `/data/train.jsonl` 与
`/data/validation.jsonl`，在整批上算漂移，再由 selector 决定本轮造哪个候选。
predict 侧不导入本模块，也不导入 `detectors`。

P2 接 SDK 训练循环时，第一件事就是调 `run()`，把返回的 `candidate` 与 `selector_trace`
带进 `rnd.submit_candidate(...)` 的 `hypothesis` / `change_summary`；`selector_trace`
同时落盘，作为本轮选择的审计记录。
"""

import json
import math
import os
import sys

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import detectors  # noqa: E402
import selector  # noqa: E402

DATA_DIR = "/data"
TRAIN_NAME = "train.jsonl"
VAL_NAME = "validation.jsonl"
TRACE_NAME = "selector_trace.json"
GROUP_KEYS = ("settings", "condition", "conditions", "group")


class TrainStageError(Exception):
    """train 阶段输入不满足要求。"""


def _log(msg):
    sys.stderr.write("[train_stage] %s\n" % msg)
    sys.stderr.flush()


def load_jsonl(path):
    """按文件顺序读 JSONL，返回 (行列表, 跳过行数)。坏行不猜，直接计数跳过。"""
    if not os.path.isfile(path):
        raise TrainStageError("找不到数据文件：%s" % path)
    rows, skipped = [], 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), dict):
                skipped += 1
                continue
            rows.append(obj)
    return rows, skipped


def _flatten_inputs(inputs):
    """按官方 `data.flatten` 的规则展开嵌套 inputs：对象键字典序、数组保持顺序。

    平台真实数据的 inputs 不是"一组标量"，而是嵌套结构：
        A: {"features": [128 个数值]}
        B: {"features": {"horizontal": {...16...}, "vertical": {...16...}}}
        C: {"settings": [3 个数值], "sensors": [21 个数值]}
    旧实现只把顶层键当列、再对整值 `float()`，于是每一行都因 TypeError 被丢成无效行，
    三线第 1 轮全部在"有效行不足"处失败。这里与 predict 侧共用同一套展开口径。
    """
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("inputs 嵌套超过 16 层")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError("inputs 的键必须是字符串")
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [["index", index]], depth + 1)
        elif isinstance(value, bool):
            raise ValueError("bool 不算数值")
        elif isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("非有限数值")
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(number)
        else:
            raise ValueError("inputs 只接受有限数值，实际为 %r" % (value,))

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("inputs 展开后应有 1..4096 个数值，实际 %d" % len(values))
    return paths, values


def _feature_order(train_rows, val_rows):
    """用第一条可展开的行钉住特征序；同一批数据的结构由平台保证一致。"""
    for row in train_rows + val_rows:
        try:
            paths, _ = _flatten_inputs(row["inputs"])
        except (TypeError, ValueError):
            continue
        return paths
    return []


def _matrix(rows, order):
    """按统一特征序展开数值矩阵；结构不一致/非数值/非有限的行整行跳过并计数。"""
    import numpy as np

    kept, skipped = [], 0
    for row in rows:
        try:
            paths, values = _flatten_inputs(row["inputs"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if paths != order:
            skipped += 1
            continue
        kept.append(values)
    return np.asarray(kept, dtype=float).reshape(len(kept), len(order)), skipped


def _has_group(train_rows, val_rows):
    for row in train_rows + val_rows:
        if any(key in row["inputs"] for key in GROUP_KEYS):
            return True
    return False


def _trace_dir(out_dir):
    """trace 一定要落盘：显式目录 > CONTEST_TRACE_DIR > /workspace > 当前目录。"""
    if out_dir:
        return out_dir
    env = os.environ.get("CONTEST_TRACE_DIR")
    if env:
        return env
    if os.path.isdir("/workspace"):
        return "/workspace"
    return os.getcwd()


def run(data_dir=None, out_dir=None, task=None):
    """跑一轮检测 + 选择。返回 {candidate, action, detector_report, selector_trace, trace_path}。"""
    data_dir = data_dir or os.environ.get("CONTEST_DATA_DIR") or DATA_DIR
    train_rows, train_skipped = load_jsonl(os.path.join(data_dir, TRAIN_NAME))
    val_rows, val_skipped = load_jsonl(os.path.join(data_dir, VAL_NAME))

    order = _feature_order(train_rows, val_rows)
    if not order:
        raise TrainStageError("train/validation 里没有任何特征列")
    train_x, train_dropped = _matrix(train_rows, order)
    val_x, val_dropped = _matrix(val_rows, order)
    if train_x.shape[0] < 2 or val_x.shape[0] < 2:
        raise TrainStageError("有效行不足：train=%d val=%d（检测器至少各需 2 行）"
                              % (train_x.shape[0], val_x.shape[0]))

    report = detectors.run_all(train_x, val_x, stage="train")
    chosen, trace = selector.select(report, has_group=_has_group(train_rows, val_rows))
    trace["inputs"].update({
        "data_dir": os.path.abspath(data_dir),
        "task": task,
        "feature_order": order,
        "skipped": {"train_bad_json": train_skipped, "val_bad_json": val_skipped,
                    "train_non_numeric": train_dropped, "val_non_numeric": val_dropped},
    })

    trace_path = os.path.join(_trace_dir(out_dir), TRACE_NAME)
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    _log("检测完成：kl=%.6g mahalanobis=%.6g bbsd=%.6g → 候选 %s（%s）"
         % (report["detectors"]["kl"]["value"],
            report["detectors"]["mahalanobis"]["value"],
            report["detectors"]["bbsd"]["value"],
            chosen, selector.CANDIDATES[chosen]["action"]))

    return {
        "candidate": chosen,
        "action": selector.CANDIDATES[chosen]["action"],
        "detector_report": report,
        "selector_trace": trace,
        "trace_path": trace_path,
    }

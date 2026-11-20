# -*- coding: utf-8 -*-
"""只写 stderr 的事实探针。

设计约束（与任务书一致）：
- stdout 一个字节都不碰；所有输出都是 `[probe]` 前缀的 stderr 单行。
- 每个入口都吞掉自身异常，最多把该项标成"不可用"，绝不让观测把一轮跑挂。
- 日志有硬预算：逐行日志 ≤ 2000 行、重字段 ≤ 前 500 行、总字节 ≤ 512 KiB。
- 不打印完整特征向量，只打印维度、短哈希和分位；不打印整行 payload，
  只打印键集合、类型和嵌套深度。

事实覆盖：
1. 帧结构：pid、payload 键集合/类型/深度、context 键、id/meta/unit/seq 类字段探测。
2. 时序与流式性：逐行到达间隔、原始字节长度、行号；结束给 min/median/max/total；
   每次写完输出后非阻塞探测下一行是否已可读（pipelined / blocked）。
3. 单元/工况指纹：inputs 数值叶子数、sha1 短哈希、绝对值分位、连续同指纹游程、
   列级 min/max（用于判断某列在一段内是否恒等）。
4. train：parent 描述符键、checkpoint_paths、parent npz metadata 键、
   model / submit / finish 回执关键字段。
5. 汇总：predict / train 结束各打一段 stderr 摘要。
"""

import hashlib
import json
import os
import struct
import sys
import time

MAX_ROW_LINES = 2000
MAX_HEAVY_ROWS = 500
MAX_LOG_BYTES = 512 * 1024
MAX_COLUMNS = 4096
ID_HINTS = ("id", "meta", "unit", "seq", "group", "trial", "case", "sample", "window", "batch")

_state = {
    "logged": 0,
    "degraded": [],
    "train": None,
    "started": time.monotonic(),
}


def _write(msg):
    """带预算的 stderr 单行输出；自身永不抛。"""
    try:
        if _state["logged"] >= MAX_LOG_BYTES:
            return
        line = "[probe] %s\n" % msg
        raw = line.encode("utf-8", "replace")
        if _state["logged"] + len(raw) > MAX_LOG_BYTES:
            if _state["logged"] < MAX_LOG_BYTES:
                sys.stderr.write("[probe] 日志预算 %d 字节已用尽，后续观测只保留汇总\n"
                                 % MAX_LOG_BYTES)
                sys.stderr.flush()
                _state["logged"] = MAX_LOG_BYTES
            return
        sys.stderr.write(line)
        sys.stderr.flush()
        _state["logged"] += len(raw)
    except BaseException:
        pass


def _degrade(item, exc):
    try:
        if item in _state["degraded"]:
            return
        _state["degraded"].append(item)
        _write("观测项 %s 不可用：%s: %s" % (item, type(exc).__name__, str(exc)[:160]))
    except BaseException:
        pass


def _safe(name):
    """把任意探针方法降级成"记录一次不可用，然后返回 None"。"""
    def decorate(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - 观测绝不允许外抛
                _degrade(name, exc)
                return None
        return wrapper
    return decorate


def _hint_keys(keys):
    hits = []
    for key in keys:
        low = str(key).lower()
        if any(hint in low for hint in ID_HINTS):
            hits.append(key)
    return hits


def _depth(value, limit=6):
    if limit <= 0:
        return limit
    if isinstance(value, dict):
        if not value:
            return 1
        return 1 + max(_depth(v, limit - 1) for v in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return 1 + max(_depth(v, limit - 1) for v in value)
    return 0


def _shape(value, depth=0):
    """类型 + 嵌套深度摘要，不携带实际内容。"""
    if depth >= 4:
        return "..."
    if isinstance(value, dict):
        items = list(value.items())[:12]
        return {"{%d keys}" % len(value): {str(k): _shape(v, depth + 1) for k, v in items}}
    if isinstance(value, list):
        return ["[%d]" % len(value), _shape(value[0], depth + 1) if value else None]
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def _numbers(value, out, depth=0):
    """按确定顺序展开数值叶子；只收有限值。"""
    if depth > 16:
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _numbers(value[key], out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _numbers(item, out, depth + 1)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        number = float(value)
        if number == number and number not in (float("inf"), float("-inf")):
            out.append(number)


def _fingerprint(values):
    if not values:
        return "empty"
    payload = bytearray()
    for number in values[:MAX_COLUMNS]:
        payload.extend(struct.pack("<d", number))
    return hashlib.sha1(bytes(payload)).hexdigest()[:12]


def _quantiles(values):
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    ordered = sorted(abs(number) for number in values[:MAX_COLUMNS])
    last = len(ordered) - 1
    return (ordered[int(last * 0.25)], ordered[int(last * 0.5)],
            ordered[int(last * 0.75)], ordered[last])


def _fmt4(value):
    try:
        return "%.4g" % float(value)
    except BaseException:
        return "?"


class PredictProbe(object):
    def __init__(self):
        self.t0 = time.monotonic()
        self.rows = 0
        self.raw_bytes = 0
        self.dts = []
        self.written = 0
        self.kind = None
        self.next_counts = {"pipelined": 0, "blocked": 0, "unknown": 0}
        self.fps = []
        self.runs = 0
        self.longest_run = 0
        self.cols_min = []
        self.cols_max = []
        self.col_overflow = False
        self.unit_hint = None
        self.payload_keys = None
        _write("阶段 predict 启动 pid=%d t=%.1f" % (os.getpid(), self.t0 - _state["started"]))
        limits(None)  # predict 侧拿不到 rnd，按契约写 limits=absent

    @_safe("predict.context")
    def context(self, ctx):
        keys = sorted(ctx or {})
        _write("context 键=%s 类型=%s id类键=%s"
               % (keys, {k: type(ctx[k]).__name__ for k in keys}, _hint_keys(keys)))

    @_safe("predict.first_line")
    def first_line(self, raw_line, ctx):
        try:
            obj = json.loads(raw_line)
        except ValueError as exc:
            _write("首行不是合法 JSON（%s），帧结构观测跳过" % exc)
            return
        if not isinstance(obj, dict):
            _write("首行不是 JSON 对象：%s" % type(obj).__name__)
            return
        self.payload_keys = sorted(obj)
        _write("首行 payload 键=%s 深度=%s 形状=%s id类键=%s"
               % (self.payload_keys,
                  {k: _depth(obj[k]) for k in self.payload_keys},
                  _shape(obj),
                  _hint_keys(self.payload_keys)))
        inputs = obj.get("inputs")
        if isinstance(inputs, dict):
            input_keys = sorted(inputs)
            _write("首行 inputs 键=%s 形状=%s id类键=%s"
                   % (input_keys, _shape(inputs), _hint_keys(input_keys)))
        ctx_hits = _hint_keys(sorted(ctx or {}))
        self.unit_hint = bool(_hint_keys(self.payload_keys) or
                              (isinstance(inputs, dict) and _hint_keys(sorted(inputs))) or
                              ctx_hits)
        _write("单元标识线索：payload/inputs/context 中 id类键=%s；context=%s"
               % (self.unit_hint, ctx_hits))

    def _track_columns(self, values):
        if len(values) > MAX_COLUMNS:
            self.col_overflow = True
        width = len(values) if len(values) <= MAX_COLUMNS else MAX_COLUMNS
        while len(self.cols_min) < width:
            self.cols_min.append(float("inf"))
            self.cols_max.append(float("-inf"))
        for index in range(width):
            number = values[index]
            if number < self.cols_min[index]:
                self.cols_min[index] = number
            if number > self.cols_max[index]:
                self.cols_max[index] = number

    @_safe("predict.line")
    def line(self, lineno, raw_len, dt_ms, inputs, decode_error):
        self.rows += 1
        self.raw_bytes += int(raw_len)
        self.dts.append(float(dt_ms))
        values = []
        if decode_error is None and inputs is not None:
            _numbers(inputs, values)
        fp = _fingerprint(values)
        if fp == "empty" and decode_error is not None:
            fp = "decode-error"
        if self.fps and self.fps[-1] == fp:
            self.runs += 1
        else:
            self.runs = 1
        if self.runs > self.longest_run:
            self.longest_run = self.runs
        self.fps.append(fp)
        if len(self.fps) > MAX_ROW_LINES:
            self.fps = self.fps[-MAX_ROW_LINES:]
        self._track_columns(values)
        if self.rows <= MAX_ROW_LINES:
            parts = ["row", "n=%d" % lineno, "bytes=%d" % int(raw_len),
                     "dt_ms=%.2f" % float(dt_ms), "dim=%d" % len(values), "fp=%s" % fp]
            if decode_error is not None:
                parts.append("decode=error")
            if self.rows <= MAX_HEAVY_ROWS:
                q25, q50, q75, qmax = _quantiles(values)
                parts.append("abs_q=%s/%s/%s/%s"
                             % (_fmt4(q25), _fmt4(q50), _fmt4(q75), _fmt4(qmax)))
            _write(" ".join(parts))

    @_safe("predict.next_ready")
    def after_write(self, ready):
        if ready is True:
            self.next_counts["pipelined"] += 1
        elif ready is False:
            self.next_counts["blocked"] += 1
        else:
            self.next_counts["unknown"] += 1

    @_safe("predict.output")
    def output(self, payload):
        if isinstance(payload, dict):
            self.kind = "regression"
        elif isinstance(payload, list):
            self.kind = "classification/%d" % len(payload)
        self.written += 1

    @_safe("predict.summary")
    def summary(self):
        ordered = sorted(self.dts)
        if ordered:
            total = sum(ordered)
            middle = ordered[len(ordered) // 2]
            timing = ("min=%.2fms median=%.2fms max=%.2fms total=%.1fms"
                      % (ordered[0], middle, ordered[-1], total))
        else:
            timing = "无行"
        distinct = len(set(self.fps))
        constant = 0
        constant_idx = []
        for index, (low, high) in enumerate(zip(self.cols_min, self.cols_max)):
            if low == high:
                constant += 1
                if len(constant_idx) < 12:
                    constant_idx.append(index)
        columns = len(self.cols_min)
        _write("SUMMARY predict pid=%d rows=%d written=%d bytes=%d kind=%s elapsed=%.1fms"
               % (os.getpid(), self.rows, self.written, self.raw_bytes, self.kind,
                  (time.monotonic() - self.t0) * 1000.0))
        _write("SUMMARY predict 行间隔 %s" % timing)
        _write("SUMMARY predict 流式性 pipelined=%d blocked=%d unknown=%d"
               % (self.next_counts["pipelined"], self.next_counts["blocked"],
                  self.next_counts["unknown"]))
        _write("SUMMARY predict 指纹 distinct=%d longest_same_run=%d fps_head=%s"
               % (distinct, self.longest_run, self.fps[:8]))
        _write("SUMMARY predict 列恒定 constant=%d/%d idx_head=%s overflow=%s"
               % (constant, columns, constant_idx, self.col_overflow))
        _write("SUMMARY predict 单元线索 unit_key_present=%s payload_keys=%s"
               % (self.unit_hint, self.payload_keys))
        _write("SUMMARY predict 观测不可用项=%s"
               % (_state["degraded"] or "无"))


class TrainProbe(object):
    def __init__(self):
        self.t0 = time.monotonic()
        self.receipts = []
        self.parent = None
        _write("阶段 train 启动 pid=%d t=%.1f" % (os.getpid(), self.t0 - _state["started"]))

    @_safe("train.start")
    def start(self, rnd):
        limits(rnd)
        parent = getattr(rnd, "parent", None)
        descriptor = getattr(parent, "descriptor", None) if parent is not None else None
        descriptor = descriptor if isinstance(descriptor, dict) else {}
        task = getattr(rnd, "task", None)
        _write("train 阶段 pid=%d round=%s stage=%s dataset_key=%s"
               % (os.getpid(), getattr(rnd, "round_index", None),
                  getattr(rnd, "stage_id", None), getattr(rnd, "dataset_key", None)))
        _write("train task_contract 键=%s public_projection=%s"
               % (sorted(task) if isinstance(task, dict) else type(task).__name__,
                  (task or {}).get("public_projection") if isinstance(task, dict) else None))
        _write("train parent 描述符键=%s checkpoint_paths=%s code_paths=%s"
               % (sorted(descriptor),
                  descriptor.get("checkpoint_paths"),
                  descriptor.get("code_paths")))
        self._write_receipt("history", {"ids": list(getattr(rnd, "history_ids", []) or [])})

    @_safe("train.parent_metadata")
    def parent_metadata(self, path):
        import numpy as np

        info = {"path": os.path.basename(str(path))}
        with np.load(str(path), allow_pickle=False) as archive:
            info["npz_keys"] = list(archive.files)
            for key in ("w1", "mean", "scale"):
                if key in archive.files:
                    info[key + "_shape"] = list(np.asarray(archive[key]).shape)
            meta = {}
            if "metadata" in archive.files:
                raw = archive["metadata"]
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                meta = json.loads(str(raw))
        if not isinstance(meta, dict):
            meta = {}
        classes = meta.get("classes")
        paths = meta.get("paths")
        parent = {
            "metadata_keys": sorted(meta),
            "task": meta.get("task"),
            "classes_len": len(classes) if isinstance(classes, list) else None,
            "classes_empty": (isinstance(classes, list) and not classes),
            "paths_len": len(paths) if isinstance(paths, list) else None,
            "target_mean": meta.get("target_mean"),
            "target_scale": meta.get("target_scale"),
            "npz": info,
        }
        self.parent = parent
        _write("train parent metadata 键=%s task=%s classes_len=%s classes_empty=%s "
               "paths_len=%s target_mean=%s target_scale=%s"
               % (parent["metadata_keys"], parent["task"], parent["classes_len"],
                  parent["classes_empty"], parent["paths_len"],
                  parent["target_mean"], parent["target_scale"]))
        _write("train parent npz keys=%s shapes=%s"
               % (info.get("npz_keys"), {k: v for k, v in info.items()
                                         if k.endswith("_shape")}))

    def _write_receipt(self, kind, value):
        fields = {"kind": kind}
        if isinstance(value, dict):
            fields["keys"] = sorted(value)
            for key in ("call_id", "artifact_id", "validation", "score", "selected_artifact_id"):
                if key in value:
                    item = value[key]
                    if isinstance(item, (str, int, float, bool)) or item is None:
                        fields[key] = item
                    elif isinstance(item, dict):
                        fields[key] = {"keys": sorted(item)}
                    elif isinstance(item, list):
                        fields[key] = {"len": len(item)}
        else:
            for key in ("call_id", "text", "response"):
                item = getattr(value, key, None)
                if key == "text" and item is not None:
                    fields["text_len"] = len(str(item))
                elif key == "response" and isinstance(item, dict):
                    fields["response_keys"] = sorted(item)
                elif item is not None:
                    fields[key] = item if isinstance(item, (str, int, float, bool)) else type(item).__name__
        self.receipts.append(fields)
        _write("train 回执 %s" % json.dumps(fields, ensure_ascii=False, sort_keys=True,
                                            default=str))

    @_safe("train.receipt")
    def receipt(self, kind, value):
        self._write_receipt(kind, value)

    @_safe("train.finish")
    def finish(self, **fields):
        self._write_receipt("finish", fields)

    @_safe("train.summary")
    def summary(self):
        _write("SUMMARY train pid=%d receipts=%d elapsed=%.1fms"
               % (os.getpid(), len(self.receipts),
                  (time.monotonic() - self.t0) * 1000.0))
        for item in self.receipts:
            _write("SUMMARY train 回执 %s" % json.dumps(item, ensure_ascii=False,
                                                        sort_keys=True, default=str))
        _write("SUMMARY train 观测不可用项=%s" % (_state["degraded"] or "无"))


# ------------------------------------------------------------------ 模块级入口

@_safe("limits")
def limits(rnd=None):
    """宿主限额一条：拿得到 rnd.limits 就压成一行打，拿不到写 limits=absent。"""
    value = getattr(rnd, "limits", None) if rnd is not None else None
    if value is None:
        _write("limits=absent")
        return
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except BaseException:
        text = repr(value)
    if len(text) > 512:
        text = text[:512] + "...truncated"
    _write("limits=%s" % text)


@_safe("predict.probe_new")
def new_predict_probe():
    return PredictProbe()


@_safe("train.probe_new")
def new_train_probe():
    probe = TrainProbe()
    _state["train"] = probe
    return probe


@_safe("train.start_module")
def train_start(rnd):
    probe = _state.get("train")
    if probe is not None:
        probe.start(rnd)


@_safe("train.parent_module")
def train_parent_metadata(path):
    probe = _state.get("train")
    if probe is not None:
        probe.parent_metadata(path)


@_safe("train.receipt_module")
def train_receipt(kind, value):
    probe = _state.get("train")
    if probe is not None:
        probe.receipt(kind, value)


@_safe("train.finish_module")
def train_finish(**fields):
    probe = _state.get("train")
    if probe is not None:
        probe.finish(**fields)


@_safe("train.summary_module")
def train_summary():
    probe = _state.get("train")
    if probe is not None:
        probe.summary()

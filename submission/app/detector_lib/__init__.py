# -*- coding: utf-8 -*-
"""v1 检测器库在 v7 镜像里的运行时（19 条，三态 + 标定门禁）。

用法（只在 train 阶段）::

    import detector_lib
    report = detector_lib.run(train_x, val_x, stage="train",
                              labels_train=ytr, labels_val=yva)
    ledger["configuration"]["detector_registry"] = report

激活（默认关的 15 条）分两步：``activate(id)`` 只让它出数；
``activate(id, {标定记录})`` 才允许进入动作决策。没有标定记录时即使被打开，
``allows_action`` 仍为 False。标定记录必须含 method/n_reps/frozen_at/space/null_p95，
且 method="offline_200"、n_reps>=200。

    reg = detector_lib.default_registry()
    reg.activate("mmd_rbf", {"method": "offline_200", "n_reps": 200,
                             "frozen_at": "2026-11-22T00:00:00Z",
                             "space": "features", "null_p95": 0.031})

也可用环境变量 ``ZHIXUN_DETECTOR_ACTIVATIONS`` 指向上述 JSON 数组。
本模块不进入 predict；``stage != "train"`` 直接抛 ``StageNotAllowed``。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

from . import kernels
from .registry import (ALARM_ONLY, ALL_IDS, CONDITIONAL, CONDITIONAL_IDS, ENABLED,
                       ENABLED_IDS, REGISTRY, STATUSES, DetectorSpec, get, table)

__all__ = ["DetectorRegistry", "StageNotAllowed", "CalibrationRequired", "default_registry",
           "reset_default_registry", "run", "table", "table_digest", "ledger_view",
           "REGISTRY", "STATUSES",
           "ENABLED", "CONDITIONAL", "ALARM_ONLY", "ALL_IDS", "ENABLED_IDS",
           "CONDITIONAL_IDS", "DetectorSpec", "get"]

_ENV_PATH = "ZHIXUN_DETECTOR_ACTIVATIONS"
_CALIBRATION_KEYS = ("method", "n_reps", "frozen_at", "space", "null_p95")


class StageNotAllowed(RuntimeError):
    """非 train 阶段调用检测器库。"""


class CalibrationRequired(ValueError):
    """激活请求缺少或伪造了离线标定记录。"""


def _check_stage(stage):
    if stage != "train":
        raise StageNotAllowed(
            "detector_lib 只在 train 阶段运行；当前 stage=%r。predict 侧不得导入本模块。" % (stage,))


def _validate_calibration(det_id, calibration):
    if not isinstance(calibration, dict):
        raise CalibrationRequired("%s 激活需要一份 dict 标定记录，实际 %r" % (det_id, type(calibration)))
    missing = [k for k in _CALIBRATION_KEYS if k not in calibration]
    if missing:
        raise CalibrationRequired("%s 标定记录缺字段：%s" % (det_id, ",".join(missing)))
    if calibration["method"] != "offline_200":
        raise CalibrationRequired("%s 标定方法必须是 offline_200，实际 %r"
                                  % (det_id, calibration["method"]))
    try:
        n_reps = int(calibration["n_reps"])
    except (TypeError, ValueError):
        raise CalibrationRequired("%s 的 n_reps 不是整数：%r" % (det_id, calibration["n_reps"])) from None
    if n_reps < 200:
        raise CalibrationRequired("%s 的标定重复数 %d < 200，不满足一次性离线标定" % (det_id, n_reps))
    if not str(calibration["space"]).strip():
        raise CalibrationRequired("%s 的标定未声明 space" % det_id)
    return {"method": "offline_200", "n_reps": n_reps,
            "frozen_at": str(calibration["frozen_at"]), "space": str(calibration["space"]),
            "null_p95": float(calibration["null_p95"])}


class DetectorRegistry:
    """三态注册表：enabled / conditional / alarm_only。"""

    def __init__(self):
        self._state = {
            det_id: {"activated": REGISTRY[det_id].status == ENABLED,
                     "calibration": None, "cusum_s": 0.0}
            for det_id in ALL_IDS
        }

    # ---- 查询
    def status(self, det_id):
        return get(det_id).status

    def is_activated(self, det_id):
        get(det_id)
        return bool(self._state[det_id]["activated"])

    def is_calibrated(self, det_id):
        get(det_id)
        return self._state[det_id]["calibration"] is not None

    def allows_action(self, det_id):
        spec = get(det_id)
        if spec.status == ALARM_ONLY:
            return False
        if spec.status == ENABLED:
            return True
        return self.is_calibrated(det_id)

    def snapshot(self):
        return {det_id: {"status": get(det_id).status,
                         "activated": self._state[det_id]["activated"],
                         "calibrated": self._state[det_id]["calibration"] is not None,
                         "allows_action": self.allows_action(det_id),
                         "calibration": deepcopy(self._state[det_id]["calibration"])}
                for det_id in ALL_IDS}

    # ---- 变更（唯一的开闸口）
    def activate(self, det_id, calibration=None):
        """打开一条默认关的检测器。

        不传 ``calibration`` 只「出数」（allows_action=False）；传了完整标定记录
        才允许进入动作决策。标定记录不全/伪造直接抛 ``CalibrationRequired``。
        """
        spec = get(det_id)
        if spec.status == ALARM_ONLY:
            raise CalibrationRequired("%s 是 alarm_only：只能报警，不能激活为动作检测器" % det_id)
        if spec.status == ENABLED:
            self._state[det_id]["activated"] = True
            return
        if calibration is None:
            self._state[det_id]["activated"] = True
            self._state[det_id]["calibration"] = None
            return
        record = _validate_calibration(det_id, calibration)
        self._state[det_id]["activated"] = True
        self._state[det_id]["calibration"] = record

    def deactivate(self, det_id):
        spec = get(det_id)
        if spec.status == ENABLED:
            raise CalibrationRequired("%s 默认开，不能关闭（否则与 v1 三态不一致）" % det_id)
        self._state[det_id]["activated"] = False
        self._state[det_id]["calibration"] = None

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise CalibrationRequired("%s 必须是 JSON 数组" % path)
        for item in payload:
            if not isinstance(item, dict) or "id" not in item:
                raise CalibrationRequired("%s 的条目必须是 {id, calibration}" % path)
            self.activate(item["id"], item.get("calibration"))

    # ---- 执行
    def run(self, train, val, stage="train", probs_train=None, probs_val=None,
            logits_train=None, logits_val=None, labels_train=None, labels_val=None,
            meta_train=None, meta_val=None):
        _check_stage(stage)
        n_train, n_val = len(train), len(val)
        inputs = {"probs": probs_val is not None and probs_train is not None,
                  "logits": logits_val is not None and logits_train is not None,
                  "labels": labels_val is not None and labels_train is not None,
                  "meta": bool(meta_val) and bool(meta_train),
                  "features": True}
        results, computed = {}, 0
        for det_id in ALL_IDS:
            spec = REGISTRY[det_id]
            entry = {"status": spec.status, "activated": self.is_activated(det_id),
                     "calibrated": spec.status == ENABLED or self.is_calibrated(det_id),
                     "allows_action": self.allows_action(det_id),
                     "value": None, "space": "features", "proxy": False, "reason": None}
            if spec.status == CONDITIONAL and not entry["activated"]:
                entry["reason"] = "dormant_default"
                results[det_id] = entry
                continue
            if min(n_train, n_val) < spec.min_batch_n:
                entry["reason"] = "insufficient_n(min=%d,train=%d,val=%d)" % (
                    spec.min_batch_n, n_train, n_val)
                results[det_id] = entry
                continue
            missing = [need for need in spec.needs if not inputs.get(need, False)]
            if missing:
                entry["reason"] = "unavailable:" + ",".join(missing)
                results[det_id] = entry
                continue
            try:
                value, extra = self._compute(det_id, train, val, probs_train, probs_val,
                                             logits_train, logits_val, labels_train,
                                             labels_val, meta_train, meta_val)
            except Exception as exc:  # 单条检测器失败不拖垮整轮
                entry["reason"] = "error:%s:%s" % (type(exc).__name__, exc)
                results[det_id] = entry
                continue
            entry.update(extra or {})
            entry["value"] = value
            entry["reason"] = None if value is not None else "not_computable"
            if extra and extra.get("reason"):
                # 例如 cusum 的“k/h 未标定”：有数也要把只报警的原因写进账本
                entry["reason"] = extra["reason"]
            if value is not None:
                computed += 1
            results[det_id] = entry
        return {
            "stage": stage,
            "wired_to_selector": False,   # 决策仍只由 app/selector.py 读 report["detectors"]
            "counts": {"total": len(ALL_IDS), "enabled": len(ENABLED_IDS),
                       "conditional": len(CONDITIONAL_IDS),
                       "alarm_only": len(ALL_IDS) - len(ENABLED_IDS) - len(CONDITIONAL_IDS),
                       "activated": sum(self.is_activated(i) for i in ALL_IDS),
                       "calibrated": sum(1 for i in ALL_IDS
                                         if REGISTRY[i].status == ENABLED or self.is_calibrated(i)),
                       "computed": computed},
            "detectors": results,
        }

    def _compute(self, det_id, train, val, p_tr, p_va, l_tr, l_va, y_tr, y_va, m_tr, m_va):
        if det_id == "argmax_kl":
            shares_tr, shares_va = kernels.probs_to_shares(p_tr), kernels.probs_to_shares(p_va)
            return kernels.output_share_kl(shares_tr, shares_va), {"space": "output", "proxy": False}
        if det_id == "mahal_pen":
            return kernels.mahalanobis_mean(train, val), {"space": "features", "proxy": True}
        if det_id == "bbsd":
            return kernels.hotelling_t2(train, val), {"space": "features", "proxy": True,
                                                      "unit": "hotelling_t2"}
        if det_id == "entropy_mean":
            return kernels.entropy_mean(p_va), {"space": "output"}
        if det_id == "mmd_rbf":
            return kernels.mmd_rbf(train, val), {}
        if det_id == "c2st_lda":
            return kernels.c2st_lda(train, val), {}
        if det_id == "cluster_separability":
            return kernels.cluster_separability(train, val, y_tr, y_va), {}
        if det_id == "cluster_mean_shift":
            return kernels.cluster_mean_shift(train, val, y_tr, y_va), {}
        if det_id == "cluster_cov_shift":
            return kernels.cluster_cov_shift(train, val, y_tr, y_va), {}
        if det_id == "cluster_merge":
            return kernels.cluster_merge(train, val, y_tr, y_va), {}
        if det_id == "cluster_split":
            return kernels.cluster_split(train, val, y_tr, y_va), {}
        if det_id == "mahalanobis_ps":
            return kernels.mahalanobis_ps(train, val), {"granularity": "per_sample"}
        if det_id == "energy_ps":
            return kernels.energy_score(l_va), {"space": "output", "granularity": "per_sample"}
        if det_id == "prototype_ps":
            return kernels.prototype_ps(train, val), {"granularity": "per_sample"}
        if det_id == "unit_monotone":
            return kernels.unit_monotone(train, val, m_tr, m_va), {}
        if det_id == "trend_residual_drift":
            return kernels.trend_residual_drift(train, val, m_tr, m_va), {}
        if det_id == "cond_feature_drift":
            return kernels.cond_feature_drift(train, val, m_tr, m_va), {}
        if det_id == "cond_key_coverage":
            return kernels.cond_key_coverage(train, val, m_tr, m_va), {}
        if det_id == "cusum_composite":
            score = kernels.c2st_lda(train, val)
            s = max(0.0, self._state[det_id]["cusum_s"] + (score - 0.5))
            self._state[det_id]["cusum_s"] = s
            # v1 的 k/h 未标定，保持 None：只报警，永不动作。
            return s, {"alarm": False, "k": None, "h": None,
                       "reason": "k/h 未标定：只报警、不动作"}
        raise KeyError(det_id)


_DEFAULT = None


def table_digest():
    """19 条规格表的稳定摘要，账本里用它对账"这一轮用的是哪张表"。"""
    payload = json.dumps(table(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ledger_view(report):
    """把 run() 的完整报告压成账本口径（configuration 有 8 KiB 预算）。

    默认关的 15 条压成 id 列表；有读数的条目只留值、空间、原因等必要字段。
    完整逐条读数仍在 train_stage 返回的 report["detector_registry"] 里；
    规格可用 table() 重建，table_sha256 用于对账。
    """
    det = report["detectors"]
    dormant = [i for i in ALL_IDS if det[i].get("reason") == "dormant_default"]
    enabled, activated, alarm = {}, {}, {}
    for det_id in ALL_IDS:
        entry = det[det_id]
        if entry.get("reason") == "dormant_default":
            continue
        kept = {}
        for key in ("activated", "calibrated", "allows_action", "value", "space",
                    "proxy", "reason", "alarm", "k", "h", "calibration"):
            value = entry.get(key)
            if value is None or value is False or value == "features":
                continue
            kept[key] = value
        if entry["status"] == ENABLED:
            enabled[det_id] = kept
        elif entry["status"] == CONDITIONAL:
            activated[det_id] = kept
        else:
            alarm[det_id] = kept
    return {"stage": report.get("stage"),
            "wired_to_selector": report.get("wired_to_selector"),
            "counts": report.get("counts"), "enabled": enabled,
            "activated": activated, "alarm_only": alarm, "dormant": dormant,
            "table_sha256": table_digest()}


def default_registry():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DetectorRegistry()
        path = os.environ.get(_ENV_PATH)
        if path:
            _DEFAULT.load(path)
    return _DEFAULT


def reset_default_registry():
    global _DEFAULT
    _DEFAULT = None


def run(train, val, stage="train", **kwargs):
    return default_registry().run(train, val, stage=stage, **kwargs)

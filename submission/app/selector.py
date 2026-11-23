#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选选择（v2 P1）：只吃 train 侧的批级检测报告，产出本轮造哪个候选 + 可审计的 selector_trace。

三个候选都只挂 causal 动作——predict 侧不允许 transductive，候选在打包时就不能违反这条：

    A  streaming_ema_norm   在线均值/尺度漂移
    B  per_cond_seq         条件分组的顺序漂移（仅当数据带分组字段）
    C  identity             没有稳定漂移，保留线性头

阈值是**量级参考**，不是调参结果：取 v1 drift-benchmark 里三个检测器的噪声地板与信号量级，
写进 trace 里是为了让每次选择都能被复核，不是为了声称最优。
"""

from actions import CAUSAL, REGISTRY, kind_of

import math

THRESHOLDS = {
    # 直方图 KL 在小样本下有正的估计偏差（n=200、10 分箱时噪声地板约 0.06），
    # 阈值取 0.15 是为了不被这个偏差顶穿；明显漂移（如均值平移 30σ）在几十仍远高于它。
    "kl": 0.15,
    "mahalanobis": 6.0,
    "bbsd": 25.0,
}

CANDIDATES = {
    "A": {
        "action": "streaming_ema_norm",
        "hypothesis": "validation 相对 train 出现均值/尺度漂移，用逐行因果 EMA 标准化吸收",
    },
    "B": {
        "action": "per_cond_seq",
        "hypothesis": "数据带工况/设置分组且组内顺序漂移，用分组扩张尺度归一化",
    },
    "C": {
        "action": "identity",
        "hypothesis": "检测器均在阈值内，没有稳定漂移，保留线性头最省",
    },
}


def _value(report, name):
    return float(report["detectors"][name]["value"])


def _check_candidates():
    for cid, spec in CANDIDATES.items():
        if kind_of(spec["action"]) != CAUSAL:
            raise AssertionError("候选 %s 挂了非 causal 动作 %s" % (cid, spec["action"]))
        if spec["action"] not in REGISTRY:
            raise AssertionError("候选 %s 的动作 %s 不在动作库" % (cid, spec["action"]))


def select(report, has_group=False):
    """返回 (候选代号, selector_trace)。同样的输入必须给出同样的 trace。"""
    _check_candidates()
    if report.get("stage") != "train":
        raise ValueError("候选选择只接受 train 阶段的检测报告，收到 %r" % (report.get("stage"),))

    kl = _value(report, "kl")
    mahal = _value(report, "mahalanobis")
    t2 = _value(report, "bbsd")
    drift = (kl >= THRESHOLDS["kl"] or mahal >= THRESHOLDS["mahalanobis"]
             or t2 >= THRESHOLDS["bbsd"])

    rules = [
        {
            "id": "R1",
            "candidate": "B",
            "condition": "has_group and (kl>=%.4g or bbsd>=%.4g)" % (THRESHOLDS["kl"], THRESHOLDS["bbsd"]),
            "ok": bool(has_group and (kl >= THRESHOLDS["kl"] or t2 >= THRESHOLDS["bbsd"])),
        },
        {
            "id": "R2",
            "candidate": "A",
            "condition": "(kl>=%.4g or mahalanobis>=%.4g or bbsd>=%.4g)"
                         % (THRESHOLDS["kl"], THRESHOLDS["mahalanobis"], THRESHOLDS["bbsd"]),
            "ok": bool(drift),
        },
        {"id": "R3", "candidate": "C", "condition": "no rule above fired", "ok": not drift},
    ]

    chosen = next(rule["candidate"] for rule in rules if rule["ok"])
    rejected = []
    for cid in sorted(CANDIDATES):
        if cid == chosen:
            continue
        if cid == "B":
            why = ("数据没有分组字段" if not has_group
                   else "分组存在但 kl=%.4g / bbsd=%.4g 未越阈值" % (kl, t2))
        elif cid == "A":
            why = ("检测器全部在阈值内：kl=%.4g mahalanobis=%.4g bbsd=%.4g" % (kl, mahal, t2)
                   if not drift else "已由更高优先级规则选中其它候选")
        else:
            why = "存在越阈值检测器：kl=%.4g mahalanobis=%.4g bbsd=%.4g" % (kl, mahal, t2)
        rejected.append({"candidate": cid, "reason": why})

    trace = {
        "schema": "contest.v2.selector-trace",
        "stage": "train",
        "inputs": {
            "n_train": report["inputs"]["n_train"],
            "n_val": report["inputs"]["n_val"],
            "features": report["inputs"]["features"],
            "has_group": bool(has_group),
        },
        "detectors": report["detectors"],
        "thresholds": dict(THRESHOLDS),
        "rules": rules,
        "chosen": {
            "candidate": chosen,
            "action": CANDIDATES[chosen]["action"],
            "hypothesis": CANDIDATES[chosen]["hypothesis"],
        },
        "rejected": rejected,
        "threshold_note": "阈值量级取自 v1 drift-benchmark 的噪声地板/信号量级，未经本轮调参",
    }
    return chosen, trace


def best_scored_arm(record):
    """四臂同尺记录里严格优于当前臂的最优臂；没有则 None。

    只读 `_score_arms` 的落账结构，不重跑、不读平台回执。与 `select()` 的分工：
    `select()` 是训练前的预登记规则选臂（验证盲）；本函数只用于候选已造好、
    同一把尺子已量过之后的载荷接管判断。平局不接管，避免只为名字顺序换臂。
    """
    arms = (record or {}).get("arms") or {}
    metric = ((record or {}).get("slice") or {}).get("metric")
    if metric not in ("rmse", "macro_f1"):
        return None
    rows = []
    for name, item in arms.items():
        if name == "PARENT" or not isinstance(item, dict) or item.get("status") != "ok":
            continue
        # changed=false 等价于父副本（sha 与 PARENT 相同），不是真实新候选；
        # 它即使分数更好也只是"退回父"，由 _apply_local_best 负责，不在这里接管。
        if item.get("changed") is False:
            continue
        value = item.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        rows.append((float(value), name))
    if not rows:
        return None
    value, name = (min(rows) if metric == "rmse" else max(rows))
    chosen = (record or {}).get("chosen")
    if chosen == "PARENT":
        return None
    current = (arms.get(chosen) or {}).get("value")
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        return name
    current = float(current)
    if metric == "rmse":
        return name if value < current - 1e-12 else None
    return name if value > current + 1e-12 else None


def best_arm_delta(record, arm):
    """接管臂相对当前臂的分数改进量（rmse 为旧−新，macro_f1 为新−旧）；不可算时 None。"""
    arms = (record or {}).get("arms") or {}
    metric = ((record or {}).get("slice") or {}).get("metric")
    chosen = (record or {}).get("chosen")
    new = (arms.get(arm) or {}).get("value")
    old = (arms.get(chosen) or {}).get("value")
    if metric not in ("rmse", "macro_f1"):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in (new, old)):
        return None
    return (float(old) - float(new)) if metric == "rmse" else (float(new) - float(old))

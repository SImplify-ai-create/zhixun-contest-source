#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把复评失败签名格式化成两条互不混用的路。

- `text`：只写模式与建议动作，**不含任何数字**，用于喂 prompt；
- `evidence`：带数值与来源，只落 `configuration.failure_modes`，供审计。

纯函数：不读环境、不写文件、不依赖时间/随机；同一输入逐位同一输出。
已知失败签名：A 的死类（召回接近零）、C 的晚段多工况超基线、漂移检测值。
"""

import re

DIGITS = re.compile(r"[0-9]")

_DRIFT_NAMES = {
    "kl": "KL 偏离",
    "mahalanobis": "马氏距离",
    "bbsd": "BBSD 距离",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _drift(detectors):
    values = {}
    present = False
    for key in ("kl", "mahalanobis", "bbsd"):
        item = (detectors or {}).get(key)
        value = (item or {}).get("value") if isinstance(item, dict) else item
        number = _number(value)
        if number is None:
            continue
        values[key] = number
        present = True
    return values, present


def _dead_classes(items):
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        f1 = _number(item.get("f1"))
        recall = _number(item.get("recall"))
        if f1 is None and recall is None:
            continue
        rows.append({
            "name": str(item.get("name") or "未命名类别"),
            "f1": f1,
            "recall": recall,
            "source": str(item.get("source") or "recheck"),
        })
    rows.sort(key=lambda row: (row["name"], row["source"]))
    return rows


def _late_segment(items):
    rows = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        rmse = _number(item.get("rmse"))
        baseline = _number(item.get("baseline"))
        if rmse is None or baseline is None:
            continue
        rows.append({
            "name": str(item.get("name") or "未命名工况"),
            "rmse": rmse,
            "baseline": baseline,
            "delta": rmse - baseline,
            "source": str(item.get("source") or "recheck"),
        })
    rows.sort(key=lambda row: (row["name"], row["source"]))
    return rows


def build(metrics=None):
    """返回 {"text": 无数字描述, "evidence": 带数字与来源}。"""
    metrics = metrics or {}
    drift, has_drift = _drift(metrics.get("drift"))
    dead = _dead_classes(metrics.get("dead_classes"))
    late = _late_segment(metrics.get("late_segment"))

    lines = []
    if has_drift:
        lines.append("- 漂移：本轮检测器读数已记录，候选可在适应强度与近端约束之间取舍。")
    if dead:
        lines.append("- 分类死类：存在召回接近零的类别，优先在训练集内做类平衡，"
                     "或事后做对数偏置调整。")
    if late:
        lines.append("- 回归晚段：多工况后段误差高于基线，优先做条件校准，"
                     "并约束误差升幅。")
    if not lines:
        lines.append("- 本轮复评没有可归因的失败签名。")

    text = ("失败模式（只描述模式，数值与来源见 configuration.failure_modes）：\n"
            + "\n".join(lines))
    evidence = {
        "source": "recheck+train_stage.detectors",
        "drift": dict(sorted(drift.items())),
        "dead_classes": dead,
        "late_segment": late,
    }
    return {"text": text, "evidence": evidence}

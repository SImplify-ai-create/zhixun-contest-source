#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动作库与阶段门禁（v2 P1）。

只有两类动作：

- **causal（因果）**：逐行可算，只用当前行与已经见过的行，不允许看未来。
  `identity`、`streaming_ema_norm`、`per_cond_seq`。
- **transductive（直推）**：统计量取自整批，包含还没到的行。
  `per_cond_mono_norm`、`batch_zscore`。

门禁规则（P1 的硬约束）：

- predict 侧只允许 causal。transductive 动作在 predict 侧**必须显式拒绝**，
  不允许"先跑再说"或静默降级成 identity。
- 批级检测（KL / 马氏 / BBSD）只在 train 侧对 train+validation 跑，不在这里。

状态提交约定：`apply()` 只计算并返回 `(新向量, 新状态)`，不写回任何全局量。
调用方必须在产出通过出站断言之后才提交新状态；声明默认分支一律不提交，
这样坏行不会把后续行的归一化带偏。
"""

import math

CAUSAL = "causal"
TRANSDUCTIVE = "transductive"
STAGES = ("predict", "train")

REGISTRY = {
    "identity": {
        "kind": CAUSAL,
        "params": {},
        "note": "不做处理；用于'检测到无稳定漂移'的候选",
    },
    "streaming_ema_norm": {
        "kind": CAUSAL,
        "params": {"alpha": 0.5, "eps": 1e-8},
        "note": "逐行因果 EMA 标准化，mean/var 只由当前与历史行更新",
    },
    "per_cond_seq": {
        "kind": CAUSAL,
        "params": {"eps": 1e-8},
        "note": "按条件分组的扩张标准差归一化，只用当前与历史行",
    },
    "per_cond_mono_norm": {
        "kind": TRANSDUCTIVE,
        "params": {"eps": 1e-8},
        "note": "统计量取自整批（含未来行），predict 侧不可用",
    },
    "batch_zscore": {
        "kind": TRANSDUCTIVE,
        "params": {"eps": 1e-8},
        "note": "整批均值方差，predict 侧不可用",
    },
}


class ActionError(Exception):
    """动作未知，或在当前阶段不被允许。"""


def kind_of(name):
    spec = REGISTRY.get(name)
    if spec is None:
        raise ActionError("未知 action：%r（可用：%s）" % (name, ", ".join(sorted(REGISTRY))))
    return spec["kind"]


def causal_actions():
    return sorted(n for n, s in REGISTRY.items() if s["kind"] == CAUSAL)


def transductive_actions():
    return sorted(n for n, s in REGISTRY.items() if s["kind"] == TRANSDUCTIVE)


def check_stage(name, stage):
    """动作在指定阶段是否允许。predict 侧遇到 transductive 一律显式拒绝。"""
    if stage not in STAGES:
        raise ActionError("未知阶段：%r（应为 %s）" % (stage, "/".join(STAGES)))
    kind = kind_of(name)
    if stage == "predict" and kind != CAUSAL:
        raise ActionError("action=%s 是 %s 动作，predict 侧必须显式拒绝" % (name, kind))
    return REGISTRY[name]


def _params(name, overrides):
    merged = dict(REGISTRY[name]["params"])
    for key, value in (overrides or {}).items():
        if key not in merged:
            raise ActionError("action=%s 不认识参数 %r" % (name, key))
        merged[key] = value
    return merged


def _positive(name, key, value, upper=None):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ActionError("action=%s 的 %s 必须是正有限数：%r" % (name, key, value))
    if upper is not None and value > upper:
        raise ActionError("action=%s 的 %s 必须 <= %r：%r" % (name, key, upper, value))
    return value


def validate(name, overrides=None, stage="predict"):
    """装载期校验：动作是否存在、能否用于该阶段、参数是否合法。不碰数据。"""
    spec = check_stage(name, stage)
    params = _params(name, overrides)
    if name == "streaming_ema_norm":
        _positive(name, "alpha", params["alpha"], upper=1.0)
        _positive(name, "eps", params["eps"])
    elif "eps" in params:
        _positive(name, "eps", params["eps"])
    return {"action": name, "kind": spec["kind"], "params": params}


def _step_identity(name, params, vector, state, cond):
    import numpy as np

    return np.asarray(vector, dtype=float), None


def _step_ema(name, params, vector, state, cond):
    import numpy as np

    alpha = params["alpha"]
    eps = params["eps"]
    x = np.asarray(vector, dtype=float)
    state = state or {}
    mean = state.get("mean")
    var = state.get("var")
    if mean is None:
        # 第一行没有历史：均值就是它自己，残差 0，方差 0，输出全 0。
        return np.zeros_like(x), {"mean": x.copy(), "var": np.zeros_like(x)}
    dm = x - mean
    mean = mean + alpha * dm
    var = (1.0 - alpha) * (var + alpha * dm * dm)
    sd = np.sqrt(np.maximum(var, 0.0))
    out = (x - mean) / np.where(sd > 0.0, sd, eps)
    return out, {"mean": mean, "var": var}


def _step_per_cond(name, params, vector, state, cond):
    import numpy as np

    eps = params["eps"]
    key = "__all__" if cond is None else str(cond)
    x = np.asarray(vector, dtype=float)
    # 每条件独立的 Welford：n / mean / m2 都只由已见行累积。
    whole = dict(state or {})
    sub = dict(whole.get(key) or {})
    n = int(sub.get("n", 0)) + 1
    mean = sub.get("mean")
    m2 = sub.get("m2")
    if mean is None:
        mean = np.zeros_like(x)
        m2 = np.zeros_like(x)
    delta = x - mean
    mean = mean + delta / n
    m2 = m2 + delta * (x - mean)
    whole[key] = {"n": n, "mean": mean, "m2": m2}
    sd = np.sqrt(np.maximum(m2 / n, 0.0))
    return x / np.where(sd > 0.0, sd, eps), whole


_STEPS = {
    "identity": _step_identity,
    "streaming_ema_norm": _step_ema,
    "per_cond_seq": _step_per_cond,
}


def prepare(name, overrides=None, stage="predict"):
    """装载期把门禁、参数校验与实现选择一次算好，返回可逐行复用的 step。

    predict 每行只应做「取数 → step → 矩阵乘」，不应重复解析参数或查表；
    因此 `_load_head` 在装载时调用一次本函数，把返回的 `step` 挂在 head 上。
    """
    spec = validate(name, overrides, stage=stage)
    step = _STEPS.get(name)
    if step is None:
        # transductive 动作在 train 侧也还没有实现；宁可显式报错，不静默顶替。
        raise ActionError("action=%s 尚未提供实现（P1 只交付 causal 执行）" % name)
    params = dict(spec["params"])

    def run(vector, state, cond=None):
        return step(name, params, vector, state, cond)

    return {"action": name, "kind": spec["kind"], "params": params, "step": run}


def apply(name, vector, state, cond=None, stage="predict", **overrides):
    """按动作处理一行特征，返回 (新向量, 新状态)。

    `state` 是该动作上一行留下的状态（可为 None）；`新状态` 由调用方在成功出站后提交。
    `stage` 默认 predict，门禁与执行绑定，避免绕过。
    """
    prepared = prepare(name, overrides, stage=stage)
    return prepared["step"](vector, state, cond)

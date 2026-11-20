"""SHAP 误差归因：漂移确认后回答"哪些特征把模型带偏了"。

定位（与答辩口径一致）：
- 只读分析，不参与任何决策——产出审计/答辩证据，不改变闭环行为。
- 触发频率 = 重训事件次数（全程通常 <10 次），KernelSHAP 对小 MLP
  是几百次前向，CPU 秒级，不进入效率账的敏感区。
- 现场一次性跑通约束：全部样本数有上限；任何异常由调用方捕获，
  归因失败绝不允许中断主流程。

口径：对漂移窗口与参考窗口各取等量样本算 KernelSHAP，
"贡献增量 delta = 漂移窗口 mean|SHAP| − 参考窗口 mean|SHAP|"，
delta 最大的特征即"本次漂移的主要驱动维度"。
"""
from __future__ import annotations

import numpy as np


def attribute_error(model, X_ref, X_drift, max_bg=50, max_eval=150,
                    nsamples=128, seed=42, top_k=5) -> dict:
    """对 model 在漂移窗口上的预测做 KernelSHAP，并与参考窗口对照。

    model 只需暴露 predict(X)。返回按 delta 降序的特征排行。
    """
    import shap  # 延迟导入：归因是可选产物，不拖累无 shap 的最小环境

    rng = np.random.default_rng(seed)
    bg = X_ref[rng.choice(len(X_ref), min(max_bg, len(X_ref)), replace=False)]
    explainer = shap.KernelExplainer(model.predict, bg)

    def mean_abs_shap(X):
        Xe = X[rng.choice(len(X), min(max_eval, len(X)), replace=False)]
        sv = np.asarray(explainer.shap_values(Xe, nsamples=nsamples,
                                              silent=True))
        return np.abs(sv).mean(axis=0)

    drift_abs = mean_abs_shap(X_drift)
    ref_abs = mean_abs_shap(X_ref)
    delta = drift_abs - ref_abs
    order = np.argsort(-delta)
    return {
        "n_features": int(X_ref.shape[1]),
        "nsamples": nsamples,
        "mean_abs_shap_drift": drift_abs.tolist(),
        "mean_abs_shap_ref": ref_abs.tolist(),
        "delta": delta.tolist(),
        "top_features": [
            {"feature": f"f{int(i)}", "index": int(i),
             "delta": float(delta[i]),
             "drift_abs": float(drift_abs[i]),
             "ref_abs": float(ref_abs[i])}
            for i in order[:top_k]],
    }

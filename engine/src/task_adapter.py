"""任务适配器：把"回归 RMSE / 分类 macro-F1"统一成"越小越好的标量损失"。

设计动机（详见 ``real_data/真实数据迁移规划.md`` §3）：

管线的全部决策——漂移严重度分档、留出验收、静态基线锚、回滚、最小增益门——
比较的都是一个标量。只要把任务差异收敛到这个标量上，主循环就不需要为
分类数据集再写第二套逻辑，也避免"两套逻辑各自演进后口径漂移"。

约定：
  - ``loss(model, X, y) -> float``，越小越好；回归=RMSE，分类=1-macro_F1。
  - ``metrics(model, X, y) -> dict[str, float]``，只允许标量（会被写进 CSV 行）。
  - ``evidence`` 可选：分类专属的漂移旁证（置信度 / 预测熵 / 错误率）。
    回归适配器不提供该方法，管线侧用 hasattr 判断，保持零影响。

注意 macro-F1 的类别口径：``labels=np.unique(y)`` 只对**该批真实出现过的类别**
求平均。UCI 气体数据 batch3/4/5 本身没有类别 6，若按全类别平均会把"该批不存在
的类"算成 0 分，属于把数据缺口记到模型头上，故显式限定 labels。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class TaskAdapter:
    """任务适配基类。"""

    name = "generic"

    def loss(self, model, X, y) -> float:  # pragma: no cover - 抽象
        raise NotImplementedError

    def metrics(self, model, X, y) -> dict:  # pragma: no cover - 抽象
        raise NotImplementedError

    def format_pred(self, pred):
        return pred


class RegressionAdapter(TaskAdapter):
    """回归：损失 = RMSE（与改造前 ``evaluate()["rmse"]`` 逐位一致）。"""

    name = "regression"

    def loss(self, model, X, y) -> float:
        return float(model.evaluate(X, y)["rmse"])

    def metrics(self, model, X, y) -> dict:
        return model.evaluate(X, y)


class ClassificationAdapter(TaskAdapter):
    """分类：损失 = 1 - macro_F1；漂移旁证 = 置信度 / 预测熵 / 错误率。

    阈值（``entropy_thr`` / ``error_thr`` / ``conf_thr``）由
    :func:`calibrate_classification_thresholds` 在**初始稳定批**上 5 折留出统计后
    给出（均值 ± k·标准差），不手写常数。三个阈值任一越界即判"分类侧报警"；
    None 表示该路不参与（例如无法拿到标签时错误率不可算）。
    """

    name = "classification"

    def __init__(self, average="macro", entropy_thr=None, error_thr=None,
                 conf_thr=None):
        self.average = average
        self.entropy_thr = entropy_thr
        self.error_thr = error_thr
        self.conf_thr = conf_thr

    # ---------------- 统一损失 ----------------

    def loss(self, model, X, y) -> float:
        pred = np.asarray(model.predict(X)).ravel()
        y = np.asarray(y).ravel()
        f1 = f1_score(y, pred, average=self.average,
                      labels=np.unique(y), zero_division=0)
        return float(1.0 - f1)

    def metrics(self, model, X, y) -> dict:
        pred = np.asarray(model.predict(X)).ravel()
        y = np.asarray(y).ravel()
        labels = np.unique(y)
        return {
            "macro_f1": float(f1_score(y, pred, average=self.average,
                                       labels=labels, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            # 该批真实出现的类别数：batch3/4/5 只有 5 类，需在报表里显式标注
            "n_classes": int(len(labels)),
        }

    # ---------------- 漂移旁证 ----------------

    def evidence(self, model, X, y=None) -> dict:
        """置信度分布 / 预测熵 / 错误率，与输入侧 MMD²·KS 做融合判据。"""
        proba = np.asarray(model.predict_proba(X), dtype=float)
        proba = np.clip(proba, 1e-12, 1.0)
        conf = float(proba.max(axis=1).mean())
        ent = float(-(proba * np.log(proba)).sum(axis=1).mean())
        out = {"confidence": round(conf, 4), "entropy": round(ent, 4)}
        reasons = []
        if self.entropy_thr is not None and ent > self.entropy_thr:
            reasons.append("entropy")
        if self.conf_thr is not None and conf < self.conf_thr:
            reasons.append("confidence")
        if y is not None:
            pred = np.asarray(model.predict(X)).ravel()
            err = float((pred != np.asarray(y).ravel()).mean())
            out["error_rate"] = round(err, 4)
            if self.error_thr is not None and err > self.error_thr:
                reasons.append("error_rate")
        out["alarm"] = bool(reasons)
        out["reasons"] = ",".join(reasons)
        return out


def _probe(model, X, y) -> dict:
    """在给定模型上算三项证据的原值（不含阈值判定）。"""
    proba = np.clip(np.asarray(model.predict_proba(X), dtype=float), 1e-12, 1.0)
    pred = np.asarray(model.predict(X)).ravel()
    y = np.asarray(y).ravel()
    return {
        "confidence": float(proba.max(axis=1).mean()),
        "entropy": float(-(proba * np.log(proba)).sum(axis=1).mean()),
        "error_rate": float((pred != y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro",
                                   labels=np.unique(y), zero_division=0)),
    }


def calibrate_classification_thresholds(X, y, model_factory, folds=5,
                                        k_sigma=2.0, seed=42) -> dict:
    """在初始稳定批上做分层 k 折留出，用"均值 ± k·标准差"给阈值。

    为什么这样做：阈值必须来自"同分布留出"的经验波动，而不是拍脑袋常数。
    留出折上的错误率/熵本来就应该很小，其波动上界就是"正常抖动"的合理边界，
    超过它才可能是分布变化。

    返回：{"entropy": {...}, "error": {...}, "confidence": {...}, "folds": n}
    """
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    folds = max(2, int(min(folds, counts.min())))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for tr, te in skf.split(X, y):
        m = model_factory().fit(X[tr], y[tr])
        rows.append(_probe(m, X[te], y[te]))

    def _stat(key):
        v = np.asarray([r[key] for r in rows], dtype=float)
        return {"mu": float(v.mean()), "sd": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    ent, err, conf = _stat("entropy"), _stat("error_rate"), _stat("confidence")
    return {
        "folds": folds,
        "k_sigma": float(k_sigma),
        "entropy": {**ent, "thr": ent["mu"] + k_sigma * ent["sd"]},
        "error": {**err, "thr": err["mu"] + k_sigma * err["sd"]},
        # 置信度是"越高越好"，所以异常方向是下界
        "confidence": {**conf, "thr": conf["mu"] - k_sigma * conf["sd"]},
        "macro_f1": _stat("macro_f1"),
    }


def adapter_from_calibration(calib: dict) -> ClassificationAdapter:
    """把标定结果装进适配器。"""
    return ClassificationAdapter(
        entropy_thr=calib["entropy"]["thr"],
        error_thr=calib["error"]["thr"],
        conf_thr=calib["confidence"]["thr"],
    )

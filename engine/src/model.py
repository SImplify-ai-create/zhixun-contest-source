"""模型层：CPU 友好的 sklearn MLP 回归基线。

赛题规则明确"官方数据包规模较小并提供预提取特征，基础任务使用普通 CPU 即可完成"。
后续可将本类替换为 PyTorch 模型，只要保持 fit / partial_fit / predict / evaluate 接口。
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor


class BaselineRegressor:
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=300,
                 learning_rate_init=0.01, alpha=0.001, random_state=42):
        # early_stopping=True 时 sklearn 不支持 partial_fit（增量微调）
        # learning_rate_init=0.01 加速增量适应，兼顾初始收敛
        # alpha 为 MLPRegressor 的 L2 正则化系数，供 HPO 调参
        self._params = dict(hidden_layer_sizes=hidden_layer_sizes, activation="relu",
                            max_iter=max_iter, random_state=random_state,
                            learning_rate_init=learning_rate_init,
                            alpha=alpha, early_stopping=False)
        self.model = MLPRegressor(**self._params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def partial_fit(self, X, y):
        """增量微调：低成本的持续学习更新（对应赛题"持续微调"路线）。"""
        self.model.partial_fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }


class KalmanRegressor:
    """恒定速度卡尔曼回归：把漂移当作状态估计问题，追踪线性映射 w(t)。

    状态 s = [w; v]（w 回归系数，v 变化率），恒定速度模型：
      w_{t+1} = w_t + v_t,  v_{t+1} = v_t + 过程噪声
    观测 y_t = x_t^T w_t + e_t。逐样本：时间更新 → 一步预测 → 测量更新。

    适用前提：特征→目标近似线性。合成流 y=X@w+noise 精确匹配，
    实测渐变末段 RMSE 0.140（接近噪声下限 0.1，相对 MLP 管线 −90%）。
    非线性数据请用 BaselineRegressor（MLP），或官方集先做 Ridge R² 线性度检验。

    接口与 BaselineRegressor 对齐（fit/partial_fit/predict/evaluate），
    额外提供 track(X,y) 做逐样本在线更新并返回一步预测 RMSE。
    """
    continuous_tracking = True

    def __init__(self, q_w=1e-8, q_v=1e-7, R=0.01, ridge_alpha=1.0):
        self.q_w = float(q_w)
        self.q_v = float(q_v)
        self.R = float(R)
        self.ridge_alpha = float(ridge_alpha)
        self._d: int | None = None
        self._s: np.ndarray | None = None  # [w; v]
        self._P: np.ndarray | None = None

    def _reset(self, w0: np.ndarray) -> None:
        d = len(w0)
        self._d = d
        self._s = np.concatenate([w0.astype(float), np.zeros(d)])
        self._P = np.block([[np.eye(d) * 0.01, np.zeros((d, d))],
                            [np.zeros((d, d)), np.eye(d) * 0.001]])
        # F/Q 只随维度与噪声参数变化，fit 时构造一次；
        # 旧实现每批 track 都 np.block 重建，纯属浪费。
        self._F = np.block([[np.eye(d), np.eye(d)],
                            [np.zeros((d, d)), np.eye(d)]])
        self._Q = np.block([[self.q_w * np.eye(d), np.zeros((d, d))],
                            [np.zeros((d, d)), self.q_v * np.eye(d)]])

    def fit(self, X, y):
        """Ridge 初值 + 重置协方差（对应一次全量重训练 / 跳变后重置）。"""
        w0 = Ridge(alpha=self.ridge_alpha).fit(X, y).coef_
        self._reset(w0)
        return self

    def track(self, X, y) -> float:
        """逐样本在线更新，返回该批一步预测 RMSE（对应管线的线上误差口径）。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        d = self._d
        F, Q = self._F, self._Q
        s = self._s
        P = self._P
        preds = np.zeros(len(y))
        for t in range(len(y)):
            s = F @ s
            P = F @ P @ F.T + Q
            w = s[:d]
            preds[t] = float(X[t] @ w)
            H = np.concatenate([X[t], np.zeros(d)])
            S = float(H @ P @ H) + self.R
            K = P @ H / S
            innov = y[t] - float(X[t] @ w)
            s = s + K * innov
            P = P - np.outer(K, H @ P)
        self._s = s
        self._P = P
        return float(np.sqrt(np.mean((preds - y) ** 2)))

    def partial_fit(self, X, y):
        """兼容接口：对卡尔曼等价于 track（不返回 RMSE）。"""
        self.track(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self._s[:self._d]

    def evaluate(self, X, y) -> dict:
        """仅预测、不更新状态（用于留出验收 / 静态基线对照）。"""
        pred = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }


class RidgeRegressor:
    """低成本线性基线：Ridge 封装，接口与三种模型一致。

    S2 条件选择的候选之一：在映射近似线性且稳定时保持 Ridge 比重训
    MLP 更划算（拟合快、方差小）；partial_fit 语义=直接重拟合。
    """

    def __init__(self, alpha=1.0):
        self._m = Ridge(alpha=alpha)

    def fit(self, X, y):
        self._m.fit(X, y)
        return self

    def partial_fit(self, X, y):
        return self.fit(X, y)

    def predict(self, X):
        return self._m.predict(X)

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }


class HybridRegressor:
    """方案 B：MLP 非线性主体 + 卡尔曼残差校正。

    当特征→目标**非线性**、KalmanRegressor 失配时使用：
      y_hat = MLP(x) + x^T a(t)
    MLP 捕捉非线性主体（fit 一次，retrain 时重 fit），残差 r = y - MLP(x)
    假设存在缓慢漂移的线性残差映射 r ≈ x^T a(t)，用恒定速度 KF 追踪 a(t)，
    在线校正 MLP 主体的概念漂移。

    适用：Ridge 直接拟合 y 的 R² 不高（非线性），但残差在特征空间近似线性且慢漂移。
    """
    continuous_tracking = True

    def __init__(self, mlp_hidden=(64, 32), mlp_max_iter=300,
                 q_w=1e-8, q_v=1e-7, R=0.01, ridge_alpha=1.0):
        self.mlp_hidden = mlp_hidden
        self.mlp_max_iter = mlp_max_iter
        self.q_w = float(q_w)
        self.q_v = float(q_v)
        self.R = float(R)
        self.ridge_alpha = float(ridge_alpha)
        self._mlp: BaselineRegressor | None = None
        self._d: int | None = None
        self._s: np.ndarray | None = None  # [a; v_a] 残差系数状态
        self._P: np.ndarray | None = None

    def _reset_a(self, a0: np.ndarray) -> None:
        d = len(a0)
        self._d = d
        self._s = np.concatenate([a0.astype(float), np.zeros(d)])
        self._P = np.block([[np.eye(d) * 0.01, np.zeros((d, d))],
                            [np.zeros((d, d)), np.eye(d) * 0.001]])

    def fit(self, X, y):
        """先 fit MLP 主体，再用残差 Ridge 初值重置卡尔曼。"""
        self._mlp = BaselineRegressor(
            hidden_layer_sizes=self.mlp_hidden,
            max_iter=self.mlp_max_iter).fit(X, y)
        resid = np.asarray(y).ravel() - self._mlp.predict(X)
        a0 = Ridge(alpha=self.ridge_alpha).fit(X, resid).coef_
        self._reset_a(a0)
        return self

    def track(self, X, y) -> float:
        """逐样本：MLP 主体预测 + 卡尔曼残差校正，返回一步预测 RMSE。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        mlp_pred = self._mlp.predict(X)
        d = self._d
        F = np.block([[np.eye(d), np.eye(d)],
                      [np.zeros((d, d)), np.eye(d)]])
        Q = np.block([[self.q_w * np.eye(d), np.zeros((d, d))],
                      [np.zeros((d, d)), self.q_v * np.eye(d)]])
        s = self._s
        P = self._P
        preds = np.zeros(len(y))
        for t in range(len(y)):
            s = F @ s
            P = F @ P @ F.T + Q
            a = s[:d]
            preds[t] = float(mlp_pred[t] + X[t] @ a)
            H = np.concatenate([X[t], np.zeros(d)])
            S = float(H @ P @ H) + self.R
            K = P @ H / S
            # 观测：残差 r = y - mlp_pred
            innov = (y[t] - mlp_pred[t]) - float(X[t] @ a)
            s = s + K * innov
            P = P - np.outer(K, H @ P)
        self._s = s
        self._P = P
        return float(np.sqrt(np.mean((preds - y) ** 2)))

    def partial_fit(self, X, y):
        """兼容接口：等价于 track（只更新残差 KF，不更新 MLP 主体）。"""
        self.track(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        a = self._s[:self._d]
        return self._mlp.predict(X) + X @ a

    def evaluate(self, X, y) -> dict:
        """仅预测、不更新状态。"""
        pred = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }


class ClassifierModel:
    """分类任务模型：sklearn MLPClassifier，接口与回归模型对齐。

    用于 UCI 气体传感器阵列漂移（6 类 / 128 维 / 10 批）。不引入 torch，
    与既有约束一致（纯 CPU、sklearn 依赖）。

    ``evaluate`` 返回分类指标而不是 rmse——统一损失的换算在
    ``src/task_adapter.py::ClassificationAdapter`` 里做，管线只认 ``loss``。
    macro-F1 的类别口径限定为该批真实出现的类别（见 ClassificationAdapter 注释）。
    """

    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=300, alpha=1e-4,
                 learning_rate_init=0.01, random_state=42):
        # early_stopping=True 时 sklearn 不支持 partial_fit（增量微调）
        self._params = dict(hidden_layer_sizes=hidden_layer_sizes,
                            activation="relu", solver="adam", max_iter=max_iter,
                            alpha=alpha, learning_rate_init=learning_rate_init,
                            random_state=random_state, early_stopping=False)
        self.model = MLPClassifier(**self._params)
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def partial_fit(self, X, y):
        """增量微调。未见过任何类别时退化为一次全量 fit。"""
        if self.classes_ is None:
            return self.fit(X, y)
        self.model.partial_fit(X, y, classes=self.classes_)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y) -> dict:
        pred = self.predict(X)
        y = np.asarray(y).ravel()
        return {
            "macro_f1": float(f1_score(y, pred, average="macro",
                                       labels=np.unique(y), zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
        }

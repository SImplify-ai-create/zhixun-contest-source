"""自进化管线：三大创新点集成。

创新点 1：解耦式双路漂移监测
  - 用冻结参考编码器（PCA 表征）上的 MMD² + KS 双路检测，而非模型错误率
    （规避盲点悖论）：结构路 MMD² 的 z 分数、边际路 KS 检验（Bonferroni
    校正 + 效应量下限），见 drift.py::DriftDetector.score()。
  - drift.py 里的 cka() 为历史遗留实现，当前无调用点，不参与检测。

创新点 2：漂移归因驱动的分级重训练
  - 检测到漂移后先归因：域偏移 → 增量微调（省 90% 算力）；
    概念漂移 → 全量重训练（保证恢复质量）。
  - 把预算花在真正需要全量重训练的概念漂移上。

创新点 3：选择性数据清洗
  - 重训练前剔除过时样本，避免新旧数据梯度错位。
  - 用更少标注量达到全量重训练精度。

通用评分点：
  - 预算约束（重训练次数 + 标注配额）
  - 评估上线 + 回滚（新模型更差则回滚）

v1.1 整改（比赛门槛优先，详见 docs/方案版本说明.md）：
  - P0 留出验收闸门：旧实现用"重训后在同一批数据上自评"，新模型天然占优，
    验收形同虚设，且回滚判据拿"训练集自评"与"线上外推"比，量纲不可比。
    现在按时间顺序切出 20% 最近样本作 holdout，训练/验收/回滚全部在这份
    留出集上比较，并在同一切片上与冻结静态基线做锚定（门槛③）。
  - 静态基线锚：初始模型冻结一份 static_model，全程每批并行打点，
    vs_static_ratio > 1 记 static_baseline_deficit 事件；重训候选若在留出集
    上劣于静态基线 + margin 直接拒绝，守住"综合性能不低于静态基线"。
  - APD 闸门（从实验臂 A6 移植）：持续报警但残差平坦 ⇒ 基准过期/虚拟漂移，
    抑制重训且不消耗预算，避免把标注花在噪声上。
  - 渐变窗口默认缩到 1 批：实测 3 批过渡期混合数据把末段 RMSE 拖到 1.395，
    1 批为 0.706（旧标签混入是主因，不是学习器本身）。
"""
from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

import numpy as np

from .drift import DriftDetector
from .model import BaselineRegressor, KalmanRegressor
from .cleaning import selective_clean
from .attribution import attribute_error


class SelfEvolutionPipeline:
    def __init__(self, detector: DriftDetector, logger, window_size=600,
                 residual_factor=1.5, max_retrains=10, label_budget=2000,
                 clean_keep_ratio_covariate=0.7,
                 clean_keep_ratio_concept=0.0, incremental_epochs=20,
                 attribution=True, attribution_nsamples=128,
                 attribution_every=1,
                 # 最小相对增益门：候选在留出集上相对旧模型改善低于该比例则放弃上线
                 # 渐变 seed42 step25 单次增益仅 0.061（5.7%），属于边际无效槽位
                 min_relative_gain=0.03,
                 hist_max=5000, max_mahalanobis_dim=100,
                 # ---- P0：留出验收闸门 + 静态基线锚 ----
                 gate_mode="holdout", accept_ratio=0.2, accept_min=30,
                 static_margin=0.05,
                 # ---- P1：APD 闸门 / 渐变窗口 ----
                 apd_enabled=False, apd_k=8, apd_max_suppress=6,
                 apd_stable_z=2.0, gradual_window=1, gradual_label_ratio=0.5,
                 retention_rollback=False, controller=None,
                 # ---- S1：统一标签账本口径 ----
                 eval_labels_free=True,
                 # ---- S2：快速恢复 off|a1|a2 ----
                 fast_recovery="off",
                 # ---- 模型选择 ----
                 model_cls=BaselineRegressor,
                 # ---- 任务适配（回归 / 分类）----
                 # None=回归 RMSE（旧行为，逐位不变）；classification 传
                 # src/task_adapter.py::ClassificationAdapter，主循环只比较
                 # 统一损失 loss（分类 = 1 - macro_F1），不区分任务。
                 task=None,
                 # 分类旁证与输入侧的融合方式（D5 已拍板，见
                 # real_data/真实数据迁移规划.md §13.6）：
                 #   or        = 输入侧或性能侧任一路报警即重训（**默认**）
                 #   perf_only = 只看性能侧（残差 + 分类旁证），输入侧仅用于
                 #               漂移类型归因；**多工况数据的推荐模式**
                 #   and       = 两路同时越界才报警；**已评估，不推荐**
                 #               （四个场景无一胜出，气体数据上退化为性能侧，
                 #                保留代码仅供复现对照，不进入默认配置）
                 # 实测：单工况 FD001 `or` 28.174 优于 perf_only 33.091；
                 # 六工况 FD002 三者同精度 35.130，perf_only 省 840 标签。
                 task_fuse="or",
                 # ---- D6-1：概念漂移档的历史类别缓冲（新增标注 0）----
                 # full 档 keep_ratio=0 会把历史全丢，小批次缺类时重训后该类别
                 # 直接消失（UCI batch4 缺类 6 → 下一批 macro-F1 崩到 0.229）。
                 # 改为从**已有标签**里按类各留 per_class 条最近样本当缓冲。
                 # 仅分类任务生效（回归的 y 是连续值，"每类"无意义）。
                 class_buffer_per_class=20,
                 # ---- D6-2：验收有效性闸门 ----
                 # 验收集过小 / 类别缺失 / 小验收集上出现 0.000 这类异常完美值
                 # 时，验收指标已无判别力，判定"本次验收不可信"→ 不升级。
                 accept_min_class_coverage=0.6,
                 accept_min_per_class=3,
                 accept_perfect_eps=1e-9):
        self.detector = detector
        self.logger = logger
        self.window_size = window_size
        self.residual_factor = residual_factor
        self.max_retrains = max_retrains
        self.label_budget = label_budget
        # 创新点 3：按漂移类型差异化清洗比例
        #   - 域偏移(covariate)：P(Y|X) 不变，旧标签仍有效 → 多保留
        #   - 概念漂移(concept)：P(Y|X) 变了，旧标签全部失效 → 几乎不保留，
        #     只留极少量作为正则化，避免梯度错位
        self.clean_keep_ratio_covariate = clean_keep_ratio_covariate
        self.clean_keep_ratio_concept = clean_keep_ratio_concept
        self.incremental_epochs = incremental_epochs
        # SHAP 归因：只读审计产物（默认开，失败自动降级为记录日志）
        self.attribution = attribution
        self.attribution_nsamples = attribution_nsamples
        # 归因降频：SHAP 是纯算力开销，按每 N 次重训跑一次（1=每次都跑）
        self.attribution_every = max(1, int(attribution_every))
        self._attr_attempts = 0
        self._attr_skipped = 0
        # 最小相对增益门：候选在留出集上改善幅度低于该比例则放弃上线，
        # 避免"花一个槽位+几百标注只换 RMSE 0.06"的边际无效消耗
        self.min_relative_gain = max(0.0, float(min_relative_gain))
        self.n_low_gain = 0
        # 算力/时长证据：按阶段累计 wall clock 与进程 CPU 时间（秒）
        self._wall = defaultdict(float)
        self._cpu = defaultdict(float)
        self._X_ref: np.ndarray | None = None  # 初始参考窗口（SHAP 背景）
        self._y_ref: np.ndarray | None = None  # 初始参考标签（抗遗忘度量）
        # 规模保险（现场大数据场景）：历史缓冲封顶保留最近 hist_max 条，
        # 避免长流下 X_hist 无界增长拖垮清洗与内存；清洗高维时先 PCA 降维。
        self.hist_max = hist_max
        self.max_mahalanobis_dim = max_mahalanobis_dim
        self.labels_used = 0
        self.n_retrains = 0
        self.model: BaselineRegressor | None = None
        self.static_model: BaselineRegressor | None = None  # P0：冻结静态基线
        self.X_hist: np.ndarray | None = None  # 历史训练集（用于清洗后复用）
        self.y_hist: np.ndarray | None = None
        # 渐变漂移检测：跟踪 CKA 与 RMSE 历史
        self._cka_history: list[float] = []
        self._rmse_history: list[float] = []
        self._in_gradual_mode = False  # 渐变模式下每批做 partial_fit

        # P0 闸门
        self.gate_mode = gate_mode            # holdout | in_sample（旧行为，仅对照）
        self.accept_ratio = accept_ratio
        self.accept_min = accept_min
        self.static_margin = static_margin
        self.n_gated = 0                      # 走留出验收的次数
        self.n_accept = 0                     # 通过验收的次数
        self.n_rollback = 0                   # 因劣于旧模型回滚的次数
        self.n_static_reject = 0              # 因劣于静态基线被拒的次数
        self.retention_rollback = retention_rollback

        # P1 APD 闸门
        self.apd_enabled = apd_enabled
        self.apd_k = apd_k
        self.apd_max_suppress = apd_max_suppress
        self.apd_stable_z = apd_stable_z
        self.apd_suppress_streak = 0
        self.alarm_streak = 0
        self.gradual_window = max(1, int(gradual_window))
        # 渐变期标注分级：相邻批同分布、样本冗余，每批只需一部分新标签即可
        # 跟踪；突变/概念漂移才需要整批。实测整批付费会在第二次漂移前耗尽 4000
        # 预算，模型冻结在漂移中段（末段 1.205 → 半量付费可覆盖全流）。
        self.gradual_label_ratio = float(gradual_label_ratio)
        self._recent_batches: list[tuple] = []  # 最近批（APD 跨窗内聚判据）
        self._static_ratios: list[float] = []   # 每批 rmse/static_rmse（门槛③）
        self.n_static_deficit = 0

        # P1 策略控制器（RuleController / LLMController），None 时沿用 classify
        self.controller = controller
        # 交付合规：逐批预测落盘缓冲。None=关闭（默认）；调用方可先置为 []，
        # 主循环会把 (batch_step, sample_index, y_true, y_pred) 逐样本追加进去。
        self.pred_dump: list[tuple] | None = None
        self.model_cls = model_cls
        # 任务适配器：None=回归（RMSE），分类传 ClassificationAdapter
        self.task = task
        self.task_fuse = str(task_fuse)
        if self.task_fuse not in ("or", "and", "perf_only"):
            raise ValueError(
                f"task_fuse must be or/and/perf_only, got {self.task_fuse!r}")
        # D6-1 历史类别缓冲（新增标注 0）
        self.class_buffer_per_class = int(class_buffer_per_class)
        self.n_buffer_added = 0
        # D6-2 验收有效性闸门
        self.accept_min_class_coverage = float(accept_min_class_coverage)
        self.accept_min_per_class = int(accept_min_per_class)
        self.accept_perfect_eps = float(accept_perfect_eps)
        self.n_accept_invalid = 0

        # S1：评分/监控标签是否免费（LabelLedger 口径，True=只记录不扣费）
        self.eval_labels_free = bool(eval_labels_free)
        # S2：高置信漂移快速恢复模式——off/a1/a2
        self.fast_recovery = str(fast_recovery)
        if self.fast_recovery not in ("off", "a1", "a2"):
            raise ValueError(f"fast_recovery must be off/a1/a2, got {self.fast_recovery!r}")

    # ------------------------- 算力/时长证据 -------------------------

    @staticmethod
    def _clock():
        return time.perf_counter(), time.process_time()

    def _lap(self, phase: str, t0: float, c0: float) -> tuple[float, float]:
        """累计一段阶段耗时，返回 (wall_ms, cpu_ms)；wall 与 cpu 分别求和。"""
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        self._wall[phase] += wall
        self._cpu[phase] += cpu
        return round(wall * 1000, 1), round(cpu * 1000, 1)

    # ------------------------- P0：留出验收切片 -------------------------

    def _accept_split(self, X_train, y_train, X_full, y_full):
        """训练/验收切片解耦。

        训练切片是软门缩过的 X_train/y_train；验收集从**完整窗口**里切，
        因此软门缩 quota 不影响验收集大小、升级判定、静态锚、回滚——根治
        报告里 "quota 缩小→训练+验收集一起缩小→扰动 incremental→full 升级"
        的耦合 bug。

        验收集的取法按任务区分：
          - 回归：时序最近 accept_ratio（外推口径，与 C-MAPSS 退化推进一致）；
          - 分类：**分层随机**留出（固定种子）。不能取时序尾部——UCI 的 .dat
            文件按类别排序，尾部往往是单一类别，验收会退化成"全对/全错"
            （batch4 实测 32 条同类样本 loss=0.000），D6-2 闸门会把它判为
            不可信。分层留出不新增任何标注，只是"留哪几条"的选择。

        返回 (X_tr, y_tr, X_ac, y_ac) 或 None（holdout 模式下没有足够样本）。
        """
        if self.gate_mode != "holdout":
            return None
        accept_size = max(self.accept_min,
                          int(round(len(y_full) * self.accept_ratio)))
        train_size = len(y_train)
        if train_size < 30:
            return None
        if accept_size > len(y_full):
            return None
        if self.task is None:
            X_ac = X_full[-accept_size:]
            y_ac = y_full[-accept_size:]
        else:
            try:
                from sklearn.model_selection import train_test_split
                _, X_ac, _, y_ac = train_test_split(
                    X_full, y_full, test_size=accept_size,
                    stratify=y_full, random_state=42)
            except ValueError:
                # 某类只有 1 条，无法分层 → 退回时序尾部，由 D6-2 闸门判定是否可信
                X_ac = X_full[-accept_size:]
                y_ac = y_full[-accept_size:]
        return X_train, y_train, X_ac, y_ac

    # ------------------------- 任务无关的统一损失 -------------------------

    def _loss(self, model, X, y) -> float:
        """统一损失（越小越好）：回归=RMSE，分类=1-macro_F1。

        主循环里所有"拿误差比大小"的地方都改成走这里（漂移严重度、留出验收、
        静态锚、回滚、最小增益门），因此分类数据集不需要第二套 run()。
        task=None 时与改造前逐位一致。
        """
        if self.task is None:
            return float(model.evaluate(X, y)["rmse"])
        return float(self.task.loss(model, X, y))

    def _oof_loss(self, X, y, folds=5):
        """初始基线的折外估计：训练集自评（in-sample）系统性偏低 20%~40%，
        直接当 baseline 会把 severity 抬高、把改进行为的收益算虚。

        分类任务改用分层折：batch1 类别不均衡（最少一类只有 30 条），
        普通 KFold 会出现整折缺类，折外损失不可比。
        """
        try:
            if self.task is not None:
                from sklearn.model_selection import StratifiedKFold
                yy = np.asarray(y).ravel()
                _, counts = np.unique(yy, return_counts=True)
                k = max(2, int(min(folds, counts.min())))
                skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                vals, ws = [], []
                for tr, te in skf.split(X, yy):
                    m = self.model_cls().fit(X[tr], yy[tr])
                    vals.append(self.task.loss(m, X[te], yy[te]))
                    ws.append(len(te))
                return float(np.average(vals, weights=ws))
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import KFold
            n = len(y)
            k = min(folds, n)
            pred = np.zeros(n)
            kf = KFold(n_splits=k, shuffle=False)
            use_ridge = self.model_cls is KalmanRegressor
            for tr, te in kf.split(X):
                if use_ridge:
                    pred[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
                else:
                    pred[te] = BaselineRegressor().fit(X[tr], y[tr]).predict(X[te])
            return float(np.sqrt(np.mean((pred - y) ** 2)))
        except Exception:  # noqa: BLE001 - 折外估计失败退回自评，不阻塞主流程
            return self._loss(self.model, X, y)

    def _static_anchor_eval(self, X_ac, y_ac):
        """候选模型在留出集上是否劣于冻结静态基线（门槛③锚定）。"""
        if self.static_model is None:
            return None, False
        anchor = self._loss(self.static_model, X_ac, y_ac)
        return anchor, anchor * (1.0 + self.static_margin)

    def _retention_eval(self, snapshot):
        """抗遗忘度量：新旧模型在初始参考分布上的损失之比（>1 表示未遗忘）。"""
        if self._X_ref is None or self._y_ref is None:
            return None, None
        old = self._loss(snapshot, self._X_ref, self._y_ref)
        new = self._loss(self.model, self._X_ref, self._y_ref)
        ratio = old / max(new, 1e-9)
        return float(ratio), float(old)

    # ------------------------- 创新点 3：清洗钩子 -------------------------

    def _clean_data(self, X_tr, y_tr, keep_ratio, center_ref=None):
        """重训前的数据组装（创新点 3）。子类可覆盖以做消融（如"不清洗"）。"""
        if self.X_hist is not None and len(self.X_hist) > 0:
            X_clean, y_clean, stats = selective_clean(
                self.X_hist, self.y_hist, X_tr, y_tr, keep_ratio=keep_ratio,
                max_mahalanobis_dim=self.max_mahalanobis_dim,
                center_ref=center_ref)
            stats["keep_ratio_used"] = keep_ratio
            self.logger.log({"event": "data_cleaning", **stats})
            return X_clean, y_clean
        return X_tr, y_tr

    # ------------------------- D6-1：历史类别缓冲（新增标注 0） -------------------------

    def _class_buffer(self):
        """从**已有标签**里按类各取最近 N 条，作为概念漂移档的正则化缓冲。

        动机：full 档 keep_ratio=0 会把历史全部丢掉。小批次且缺类时（UCI
        batch4 只有 161 条、无类别 6），重训后该类别直接从模型里消失，
        下一批 macro-F1 崩到 0.229（低于从不重训的 0.268）。

        约束（按拍板结果）：
          - 不新增任何标注，全部取自 self.X_hist/self.y_hist，不计 labels_used；
          - 规模由 class_buffer_per_class 控制（默认每类 20 条）；
          - 仅分类任务生效（回归的 y 是连续值，"每类"无意义）。

        返回 (X_buf, y_buf) 或 (None, None)。
        """
        if (self.task is None or self.class_buffer_per_class <= 0
                or self.X_hist is None or self.y_hist is None
                or len(self.y_hist) == 0):
            return None, None
        Xs, ys = [], []
        classes = np.unique(self.y_hist)
        for c in classes:
            idx = np.flatnonzero(self.y_hist == c)[-self.class_buffer_per_class:]
            if len(idx) == 0:
                continue
            Xs.append(self.X_hist[idx])
            ys.append(self.y_hist[idx])
        if not Xs:
            return None, None
        X_buf = np.vstack(Xs)
        y_buf = np.concatenate(ys)
        self.n_buffer_added += len(y_buf)
        self.logger.log({"event": "class_buffer", "n_buffer": int(len(y_buf)),
                         "n_classes": int(len(classes)),
                         "per_class_cap": self.class_buffer_per_class,
                         "new_labels": 0})
        return X_buf, y_buf

    # ------------------------- D6-2：验收有效性闸门 -------------------------

    def _acceptance_valid(self, y_ac, y_tr, new_loss) -> tuple[bool, str]:
        """判定本次留出验收是否还有判别力；不可信则拒绝升级（宁可不动）。

        退化为无判别力的典型形态：
          - 验收集太小（几几十条，随机波动就能"全对"）；
          - 验收集类别缺失/某类样本过少（macro-F1 只对出现过的类别平均，
            缺类时数值不代表真实恢复水平）；
          - 小且缺类的验收集上出现 0.000 这类异常完美值（UCI batch4 实测：
            32 条留出全是同一类 → loss=0.000，旧闸门据此判"升级成功"）。

        返回 (ok, reason)；reason 非空时写入实验日志，便于事后审计。
        """
        n = int(len(y_ac))
        if n < int(self.accept_min):
            return False, f"holdout_too_small(n={n}<{self.accept_min})"
        # 回归任务没有类别概念：只查样本量下限
        if self.task is None:
            return True, ""
        ac = np.unique(y_ac)
        tr = np.unique(y_tr)
        if len(ac) <= 1:
            return False, f"holdout_single_class(n={n},n_classes=1)"
        cov = len(ac) / max(len(tr), 1)
        if cov < self.accept_min_class_coverage:
            return False, (f"class_coverage_low({len(ac)}/{len(tr)}="
                           f"{cov:.2f}<{self.accept_min_class_coverage})")
        _, counts = np.unique(y_ac, return_counts=True)
        min_per = int(counts.min()) if len(counts) else 0
        if min_per < self.accept_min_per_class:
            return False, (f"holdout_class_too_few(min={min_per}<"
                           f"{self.accept_min_per_class})")
        if (new_loss <= self.accept_perfect_eps
                and (n < 3 * int(self.accept_min) or len(ac) < len(tr))):
            return False, (f"perfect_on_unreliable_holdout(loss={new_loss:.6f},"
                           f"n={n},n_classes={len(ac)}/{len(tr)})")
        return True, ""

    # ------------------------- 创新点 2：归因驱动的分级重训练 -------------------------

    def _select_batch(self, X_new, y_new, baseline_rmse,
                      severity=None, tier=None):
        """挑选本次重训可用的标注切片（子类可覆盖以改变取数策略）。

        返回 (X_use, y_use, quota, extra_log_fields)。默认：在剩余标注预算内
        取窗口最早的一段（FIFO），这是原始行为。
        severity / tier 供子类的软门配额使用（只缩训练切片，不影响验收集）。
        """
        quota = max(0, min(len(y_new), self.label_budget - self.labels_used))
        return X_new[:quota], y_new[:quota], quota, {}

    def _retrain_attributed(self, X_new, y_new, old_rmse, baseline_rmse,
                            tier: str, severity: float = 1.0):
        """按漂移严重度档位选择动作（创新点 2）。

        - observe：噪声级抖动，不消耗标注/算力，直接跳过
        - incremental（sev 1.3~3）：增量微调 8 轮；旧标签保留 70% 作正则
          （映射只是部分改变），恢复不足再升级为全量
        - full（sev >= 3）：旧标签基本失效，清洗（keep_ratio 0）后全量重训练

        v1.1：训练/验收/回滚使用不同切片。X_use 的最早 80% 训练，
        最近 20% 留出验收；回滚判据与静态基线锚都在留出集上算。
        """
        if tier == "observe":
            self.logger.log({"event": "retrain_skipped", "reason": "observe_tier"})
            return False, old_rmse, "observe"

        if self.n_retrains >= self.max_retrains:
            self.logger.log({"event": "retrain_skipped", "reason": "budget"})
            return False, old_rmse, tier

        X_use, y_use, quota, extra = self._select_batch(
            X_new, y_new, baseline_rmse, severity=severity, tier=tier)
        if quota < 30:
            self.logger.log({"event": "retrain_skipped", "reason": "label_budget"})
            return False, old_rmse, tier

        t_r0, c_r0 = self._clock()
        split = self._accept_split(X_use, y_use, X_new, y_new)
        if split is None and self.gate_mode == "holdout":
            # 没有可靠留出集就不验收上线：宁可等下一批，也不拿自评冒充验收
            self.logger.log({"event": "retrain_skipped",
                             "reason": "no_holdout", "quota": quota,
                             "full_n": len(y_new)})
            return False, old_rmse, tier
        if split is None:
            X_tr, y_tr = X_use, y_use
            X_ac, y_ac = X_use, y_use
        else:
            X_tr, y_tr, X_ac, y_ac = split
            self.n_gated += 1

        snapshot = copy.deepcopy(self.model)
        # 留出集只用于验收，不参与清洗中心估计（避免特征泄漏进"新分布中心"）
        center_ref = X_ac if split is not None else None
        gate_n = len(y_ac)

        if tier == "incremental":
            keep_ratio = self.clean_keep_ratio_covariate
            X_clean, y_clean = self._clean_data(
                X_tr, y_tr, keep_ratio, center_ref=center_ref)
            mode = "incremental"
            for _ in range(self.incremental_epochs):
                self.model.partial_fit(X_clean, y_clean)
            new_rmse = self._loss(self.model, X_ac, y_ac)
            # 增量微调恢复不足 → 升级为全量重训练（判据同样看留出集）
            if new_rmse > max(baseline_rmse * 2.0, 0.5):
                mode = "incremental_upgraded_full"
                self.model = self.model_cls().fit(X_clean, y_clean)
                new_rmse = self._loss(self.model, X_ac, y_ac)

        else:  # full：映射确实变了，旧标签基本失效
            keep_ratio = self.clean_keep_ratio_concept
            X_clean, y_clean = self._clean_data(
                X_tr, y_tr, keep_ratio, center_ref=center_ref)
            # D6-1：追加历史类别缓冲（每类 N 条、新增标注 0）。keep_ratio=0
            # 时历史全丢，小批次缺类会让该类别从模型里消失，故按类补少量
            # **已有标签**样本做正则，避免下一批该类别的 F1 直接归零。
            X_buf, y_buf = self._class_buffer()
            if X_buf is not None:
                X_clean = np.vstack([X_clean, X_buf])
                y_clean = np.concatenate([y_clean, y_buf])
                extra = dict(extra, class_buffer=int(len(y_buf)))
            mode = "full_with_cleaning"
            self.model = self.model_cls().fit(X_clean, y_clean)
            new_rmse = self._loss(self.model, X_ac, y_ac)

        # 对照：旧模型在同一留出集上的表现（量纲可比）
        old_rmse_gate = self._loss(snapshot, X_ac, y_ac)

        # D6-2：验收有效性闸门——验收集本身没有判别力时不升级（宁可不动）。
        # 放在增益门之前：数据不可信时任何"改善/恶化"判断都不可信。
        valid, why = self._acceptance_valid(y_ac, y_tr, new_rmse)
        if not valid:
            self.model = snapshot
            self.n_accept_invalid += 1
            t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
            self.logger.log({"event": "accept_invalid", "tier": tier,
                             "mode": mode, "reason": why,
                             "new_rmse_untrusted": new_rmse,
                             "old_rmse_accept": old_rmse_gate,
                             "accept_n": gate_n,
                             "holdout_classes": int(len(np.unique(y_ac))),
                             "train_classes": int(len(np.unique(y_tr))),
                             "quota_not_consumed": quota, **extra,
                             "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms})
            return False, old_rmse, "holdout_untrusted"

        # 最小相对增益门：候选改善幅度不够就放弃，不消耗槽位/标注
        # 训练已做完（算力消耗无法回滚），但不再计入重训次数与标注预算
        rel_gain = ((old_rmse_gate - new_rmse)
                    / max(old_rmse_gate, 1e-9))
        if (self.min_relative_gain > 0
                and rel_gain < self.min_relative_gain
                and new_rmse <= old_rmse_gate):
            self.model = snapshot
            self.n_low_gain += 1
            t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
            self.logger.log({
                "event": "retrain_low_gain", "tier": tier, "mode": mode,
                "old_rmse": old_rmse_gate, "new_rmse": new_rmse,
                "rel_gain": round(rel_gain, 4),
                "quota_not_consumed": quota, "accept_n": gate_n,
                "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms})
            return False, old_rmse, "low_gain"

        # 只有通过增益门后才计入预算（之前是先记账再验，被拒绝也白算）
        self.labels_used += quota
        self.n_retrains += 1

        anchor, anchor_limit = self._static_anchor_eval(X_ac, y_ac)
        retention_ratio, ref_old = self._retention_eval(snapshot)

        # 门槛③守卫：候选劣于冻结静态基线（含 margin）直接拒绝
        if anchor_limit is not None and new_rmse > anchor_limit:
            self.model = snapshot
            self.n_static_reject += 1
            t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
            self.logger.log({"event": "static_anchor_reject", "tier": tier,
                             "mode": mode, "new_rmse": new_rmse,
                             "static_rmse": anchor,
                             "limit": anchor_limit, "accept_n": gate_n,
                             "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms})
            return False, old_rmse, "static_reject"

        # 回滚保护（同一切片上的新旧模型对比）
        if new_rmse > old_rmse_gate * 1.2:
            self.model = snapshot
            self.n_rollback += 1
            t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
            self.logger.log({"event": "rollback", "tier": tier, "mode": mode,
                             "old_rmse": old_rmse_gate, "new_rmse": new_rmse,
                             "accept_n": gate_n,
                             "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms})
            return False, old_rmse, "rollback"

        # 抗遗忘（可选上线）：新模型在初始分布上退化过多则回滚
        if (self.retention_rollback and retention_ratio is not None
                and retention_ratio < 0.5):
            self.model = snapshot
            self.n_rollback += 1
            t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
            self.logger.log({"event": "retention_rollback", "tier": tier,
                             "retention_ratio": retention_ratio,
                             "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms})
            return False, old_rmse, "retention_rollback"

        self.n_accept += 1
        # 审计口径：同一候选在训练切片上的自评 vs 留出集上的真评（旧闸门用的就是前者）
        diag = {}
        if split is not None:
            ins = self._loss(self.model, X_tr, y_tr)
            diag = {"new_rmse_in_sample": ins,
                    "in_sample_optimism": new_rmse / max(ins, 1e-9)}
        # 更新历史训练集（用于下次清洗）——只存训练切片，验收切片不进历史
        self.X_hist = (np.vstack([self.X_hist, X_tr])
                       if self.X_hist is not None else X_tr.copy())
        self.y_hist = (np.concatenate([self.y_hist, y_tr])
                       if self.y_hist is not None else y_tr.copy())
        # 历史缓冲封顶：漂移场景下近期样本最有价值，保留最近 hist_max 条
        if len(self.y_hist) > self.hist_max:
            self.X_hist = self.X_hist[-self.hist_max:]
            self.y_hist = self.y_hist[-self.hist_max:]

        t_ms, c_ms = self._lap("retrain", t_r0, c_r0)
        self.logger.log({"event": "retrain_mode", "mode": mode, "tier": tier,
                         "accept_n": gate_n,
                         "old_rmse_accept": old_rmse_gate,
                         "new_rmse_accept": new_rmse,
                         "static_rmse_accept": anchor,
                         "retention_ratio": retention_ratio,
                         "t_retrain_ms": t_ms, "cpu_retrain_ms": c_ms,
                         **diag, **extra})
        return True, new_rmse, mode

    # ------------------------- SHAP 归因（只读审计产物） -------------------------

    def _log_attribution(self, step: int, tier: str, Xw) -> None:
        """漂移确认、重训之前，对当前模型在漂移窗口的误差做特征归因。

        铁律：归因失败绝不中断主流程（现场一次性跑通约束）。
        产物：SwanLab/CSV 事件（top 特征）+ runs/attribution-stepN.json（全量）。
        """
        if not self.attribution or self._X_ref is None:
            return
        # 降频：SHAP 只作审计产物，按每 N 次重训跑一次即可（1=每次都跑）
        if self.attribution_every > 1:
            self._attr_attempts += 1
            if (self._attr_attempts - 1) % self.attribution_every != 0:
                self._attr_skipped += 1
                self.logger.log({"event": "attribution_skipped", "step": step,
                                 "tier": tier, "every": self.attribution_every})
                return
        t0, c0 = self._clock()
        try:
            rep = attribute_error(self.model, self._X_ref, Xw,
                                  nsamples=self.attribution_nsamples)
            t_attr_ms, cpu_attr_ms = self._lap("attrib", t0, c0)
            rep["step"] = step
            rep["tier"] = tier
            self.logger.log({"event": "shap_attribution", "step": step,
                             "tier": tier,
                             "t_attrib_ms": t_attr_ms,
                             "cpu_attrib_ms": cpu_attr_ms,
                             "top_features": [t["feature"]
                                              for t in rep["top_features"]]})
            import json
            from pathlib import Path
            out_dir = Path(getattr(self.logger, "out_dir", "runs"))
            out_dir.mkdir(exist_ok=True)
            name = getattr(self.logger, "experiment_name", None) or "run"
            (out_dir / f"attribution-{name}-step{step}.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - 归因是附属产物，任何失败降级为日志
            self.logger.log({"event": "attribution_failed", "step": step,
                             "error": str(exc)[:200]})

    # ------------------------- 渐变 vs 突变检测 -------------------------

    def _is_gradual_drift(self, current_rmse: float, baseline_rmse: float) -> bool:
        """通过 RMSE 变化趋势判断是否为渐变概念漂移。

        - 突变：RMSE 单步骤升（如 0.2 → 4.0）→ 全量重训练
        - 渐变：RMSE 连续多步温和上升 → 增量 partial_fit 持续适应

        判据：最近 3 步 RMSE 单调上升，且相对基线的增幅每步 < 1.5x。
        """
        self._rmse_history.append(current_rmse)
        if len(self._rmse_history) > 10:
            self._rmse_history.pop(0)
        hist = self._rmse_history
        if len(hist) < 4:
            return False
        recent = hist[-4:]
        diffs = [recent[i + 1] - recent[i] for i in range(3)]
        # 连续上升，且单步增幅不超过基线的 1.5 倍（排除骤升）
        monotonic_up = all(d > 0.001 for d in diffs)
        not_sudden = all(d < 1.5 * baseline_rmse for d in diffs)
        # 总体已偏离基线超过 50%
        drifted = current_rmse > 1.5 * baseline_rmse
        return monotonic_up and not_sudden and drifted

    # ------------------------- P1：APD 跨窗内聚健康自检 -------------------------

    def _health_check(self, X_now) -> bool:
        """APD 退出条件：跨时间窗内聚 ∧ 参考分布仍然分离。

        内聚判据 = 当前批 vs APD_K 批前那批 的 MMD² z < apd_stable_z。
        慢单调漂移会让"窗内随机对半切分"失效（两半各含全量程、均值近等），
        必须跨时间窗比较；历史不足时回退窗内对半切分。
        """
        det = self.detector
        if det.encoder is None or det.repr_ref is None:
            return False
        z_now = det.encoder.transform(X_now)
        if len(self._recent_batches) >= self.apd_k:
            X_then = self._recent_batches[-self.apd_k][0]
            z_then = det.encoder.transform(X_then)
            m = self._mmd2(z_then, z_now)
            stable_z = (m - det.mmd_null_mean) / det.mmd_null_std
        else:
            n = len(z_now)
            half = n // 2
            rs = np.random.RandomState(0)
            vals = []
            for _ in range(8):
                perm = rs.permutation(n)
                vals.append(self._mmd2(z_now[perm[:half]], z_now[perm[half:2 * half]]))
            stable_z = (float(np.mean(vals)) - det.mmd_null_mean) / det.mmd_null_std
        n = len(z_now)
        ref_sub = (det.repr_ref[:n] if len(det.repr_ref) >= n
                   else det.repr_ref)
        sep_z = (self._mmd2(ref_sub, z_now) - det.mmd_null_mean) / det.mmd_null_std
        return bool(stable_z < self.apd_stable_z
                    and sep_z > det.mmd_z_threshold)

    def _mmd2(self, a, b) -> float:
        from .drift import mmd2_rbf
        return mmd2_rbf(a, b, gamma=self.detector.mmd_gamma)["mmd2"]

    # ------------------------- 主循环 -------------------------

    def run(self, stream, batch_size=200, init_batches=3):
        history = []
        X_buf, y_buf = [], []
        baseline_rmse = None
        cooldown = 0
        alarm_count = 0
        t_run0, c_run0 = self._clock()

        for step, (offset, X, y) in enumerate(stream.batches(batch_size)):
            t_step0, c_step0 = self._clock()
            if step < init_batches:
                X_buf.append(X)
                y_buf.append(y)
                if step == init_batches - 1:
                    X0 = np.vstack(X_buf)
                    y0 = np.concatenate(y_buf)
                    self.detector.fit_reference(X0)
                    self.model = self.model_cls().fit(X0, y0)
                    # P0：冻结静态基线（同起点、永不更新），全程作为门槛③对照
                    self.static_model = copy.deepcopy(self.model)
                    self.X_hist, self.y_hist = X0.copy(), y0.copy()
                    self._X_ref = X0.copy()
                    self._y_ref = y0.copy()
                    self.labels_used += len(y0)
                    self.n_retrains += 1
                    # 折外基线：避免把训练集自评当"线上水平"（会系统性低估）
                    baseline_rmse = self._oof_loss(X0, y0)
                    self.logger.log({"step": step, "event": "init_train",
                                     "rmse": baseline_rmse,
                                     "loss": baseline_rmse,
                                     "init_rmse_in_sample":
                                     self._loss(self.model, X0, y0)})
                self._lap("init", t_step0, c_step0)
                continue

            t_ev0, c_ev0 = self._clock()
            metrics = self.model.evaluate(X, y)
            # 任务无关：loss 是主循环唯一比较的误差标量（回归=RMSE，分类=1-macro_F1）
            loss = (float(metrics["rmse"]) if self.task is None
                    else self._loss(self.model, X, y))
            # 连续追踪模型（KalmanRegressor）：每批做逐样本在线更新，
            # 日志 RMSE 用一步预测口径（更接近真实线上误差）；evaluate 本身
            # 仅预测不更新，保证静态基线/留出验收不受污染。
            if getattr(self.model, "continuous_tracking", False):
                online_rmse = self.model.track(X, y)
                metrics = dict(metrics, rmse=online_rmse)
                loss = float(online_rmse)
            metrics = dict(metrics, loss=loss)
            # 交付合规：逐批预测落盘（默认关闭；run_pipeline 会显式开启）
            if self.pred_dump is not None:
                try:
                    _pred = np.asarray(self.model.predict(X)).ravel()
                except Exception:  # noqa: BLE001 - 落盘失败不影响主循环
                    _pred = None
                if _pred is not None:
                    try:
                        _true = np.asarray(y, dtype=float).ravel()
                    except (TypeError, ValueError):
                        _true = np.asarray(y).ravel()
                    for _i in range(min(len(_true), len(_pred))):
                        self.pred_dump.append(
                            (step, offset + _i, _true[_i], _pred[_i]))
            # P0：静态基线全程并行打点（门槛③"不低于组委会静态基线"）
            static_loss = self._loss(self.static_model, X, y)
            ratio = loss / max(static_loss, 1e-9)
            self._static_ratios.append(ratio)
            if ratio > 1.0:
                self.n_static_deficit += 1
                if self.n_static_deficit == 1 or self.n_static_deficit % 5 == 0:
                    self.logger.log({"step": step,
                                     "event": "static_baseline_deficit",
                                     "loss": loss,
                                     "static_loss": static_loss,
                                     "ratio": ratio,
                                     "count": self.n_static_deficit})
            t_eval_ms, cpu_eval_ms = self._lap("eval", t_ev0, c_ev0)

            t_det0, c_det0 = self._clock()
            drift = self.detector.score(X)
            # 创新点 1：双路联合判据（drift["drift_detected"] 已是 CKA∧KS）
            residual_alarm = loss > self.residual_factor * baseline_rmse
            # 分类专属旁证（置信度 / 预测熵 / 错误率）：回归适配器不提供 evidence，
            # task_ev 恒为 None，行为与改造前完全一致。
            task_ev = None
            if self.task is not None and hasattr(self.task, "evidence"):
                try:
                    task_ev = self.task.evidence(self.model, X, y)
                except Exception:  # noqa: BLE001 - 旁证失败不阻塞主循环
                    task_ev = None
            task_alarm = bool(task_ev and task_ev.get("alarm"))
            input_alarm = bool(drift["drift_detected"])
            # 性能侧一路 = 残差越界 或 分类旁证越界（回归任务下 task_alarm 恒 False）
            perf_alarm = bool(residual_alarm or task_alarm)
            # 三种融合模式（D5）：
            #   or        = 输入侧 / 性能侧任一路报警即重训（旧行为）
            #   and       = 输入侧与性能侧**同时**越界才报警（两路与，抑误报）
            #   perf_only = 只看性能侧；输入侧只用于漂移类型归因
            #               （分布变但精度没掉 = 工况变化，不该重训）
            if self.task_fuse == "and":
                alarm = bool(input_alarm and perf_alarm)
            elif self.task_fuse == "perf_only":
                alarm = perf_alarm
            else:  # or
                alarm = bool(input_alarm or perf_alarm)
            alarm_count = alarm_count + 1 if alarm else 0
            self.alarm_streak = self.alarm_streak + 1 if alarm else 0

            # 渐变漂移检测：loss 温和持续上升 → 进入增量适应模式
            is_gradual = self._is_gradual_drift(loss, baseline_rmse)
            t_det_ms, cpu_det_ms = self._lap("detect", t_det0, c_det0)
            if is_gradual and not self._in_gradual_mode:
                self._in_gradual_mode = True
                self.logger.log({"event": "gradual_mode_enter",
                                 "mmd_z": drift["mmd_z"]})
            elif not alarm and self._in_gradual_mode:
                self._in_gradual_mode = False
                self.logger.log({"event": "gradual_mode_exit"})

            # 渐变模式：每批 partial_fit 持续适应 + 周期性纯新数据全量重训练校准
            # 连续追踪模型（Kalman）已在每批 track() 中持续适应，跳过 MLP 适配块
            t_g0, c_g0 = self._clock()
            if (self._in_gradual_mode and self.labels_used < self.label_budget
                    and not getattr(self.model, "continuous_tracking", False)):
                split = self._accept_split(X, y, X, y)
                # 渐变期按比例付费（相邻批冗余），把预算留给后续漂移
                want = max(30, int(round(len(y) * self.gradual_label_ratio)))
                quota = min(want, len(y),
                            self.label_budget - self.labels_used)
                if quota >= 30:
                    # 留出集不参与 partial_fit，否则下面的全量重训验收被污染
                    X_pf = split[0][:quota] if split is not None else X[:quota]
                    y_pf = split[1][:quota] if split is not None else y[:quota]
                    for _ in range(10):
                        self.model.partial_fit(X_pf, y_pf)
                    self.labels_used += quota
                    self.logger.log({"event": "partial_fit_gradual",
                                     "labels_used": self.labels_used})
                # 每 5 批用最近 gradual_window 批纯新数据做全量重训练（不混旧标签）
                if (step % 5 == 0
                        and self.n_retrains < self.max_retrains
                        and (split is not None or self.gate_mode == "in_sample")):
                    snapshot = copy.deepcopy(self.model)
                    if split is not None:
                        X_fit, y_fit = split[0], split[1]
                        X_ac, y_ac = split[2], split[3]
                    else:  # in_sample 对照臂：同批自评（旧口径），仅用于消融
                        X_fit, y_fit = X, y
                        X_ac, y_ac = X_fit, y_fit
                    if self.gradual_window > 1 and self._recent_batches:
                        prev = self._recent_batches[-(self.gradual_window - 1):]
                        X_fit = np.vstack([b[0] for b in prev] + [X_fit])
                        y_fit = np.concatenate([b[1] for b in prev] + [y_fit])
                    new_model = self.model_cls().fit(X_fit, y_fit)
                    new_rmse = self._loss(new_model, X_ac, y_ac)
                    old_rmse = self._loss(snapshot, X_ac, y_ac)
                    anchor, anchor_limit = (self._static_anchor_eval(X_ac, y_ac)
                                            if split is not None else (None, None))
                    # 最小相对增益门：渐变周期重训同样接受边际无效槽位
                    rel_gain = ((old_rmse - new_rmse)
                                / max(old_rmse, 1e-9))
                    accept = (new_rmse <= old_rmse
                              and (anchor_limit is None or new_rmse <= anchor_limit)
                              and (self.min_relative_gain <= 0
                                   or rel_gain >= self.min_relative_gain))
                    if accept:
                        self.model = new_model
                        baseline_rmse = new_rmse
                        self.n_retrains += 1
                        self.logger.log({"step": step, "event": "gradual_retrain",
                                         "old_rmse": old_rmse,
                                         "new_rmse": new_rmse,
                                         "rel_gain": round(rel_gain, 4),
                                         "accept_n": len(y_ac),
                                         "n_retrains": self.n_retrains})
                    else:
                        self.model = snapshot
                        reason = "low_gain" if (self.min_relative_gain > 0
                                                and rel_gain < self.min_relative_gain
                                                and new_rmse <= old_rmse) else "worse_or_over_anchor"
                        if reason == "low_gain":
                            self.n_low_gain += 1
                        self.logger.log({"step": step,
                                         "event": "gradual_retrain_reject",
                                         "old_rmse": old_rmse,
                                         "new_rmse": new_rmse,
                                         "rel_gain": round(rel_gain, 4),
                                         "reason": reason,
                                         "static_limit": anchor_limit})
            t_grad_ms, cpu_grad_ms = self._lap("gradual", t_g0, c_g0)

            row = {"step": step, "offset": offset,
                   **{f"eval_{k}": v for k, v in metrics.items()},
                   "static_loss": static_loss,
                   "vs_static_ratio": ratio,
                   "mmd2": drift["mmd2"],
                   "mmd_z": drift["mmd_z"],
                   "ks_p_value": drift["ks_p_value"],
                   "structural_alarm": drift["structural_alarm"],
                   "marginal_alarm": drift["marginal_alarm"],
                   "dual_drift": drift["drift_detected"],
                   # 三路报警与最终决策分开记录：便于核对融合模式的实际作用
                   "input_alarm": input_alarm,
                   "residual_alarm": bool(residual_alarm),
                   "task_alarm": task_alarm,
                   "alarm": bool(alarm),
                   "task_fuse": self.task_fuse,
                   "mean_psi": drift["mean_psi"],
                   "alarm_count": alarm_count,
                   "cooldown": cooldown,
                   "gradual_mode": self._in_gradual_mode,
                   "labels_used": self.labels_used,
                   "n_retrains": self.n_retrains,
                   # 算力/时长证据：本批各阶段 wall-clock 与进程 CPU 时间（毫秒）
                   "t_eval_ms": t_eval_ms, "cpu_eval_ms": cpu_eval_ms,
                   "t_detect_ms": t_det_ms, "cpu_detect_ms": cpu_det_ms,
                   "t_gradual_ms": t_grad_ms, "cpu_gradual_ms": cpu_grad_ms,
                   "t_step_ms": round((time.perf_counter() - t_step0) * 1000, 1)}
            # 分类旁证入表（回归任务下 task_ev=None，行结构不变）
            if task_ev:
                row.update({f"task_{k}": v for k, v in task_ev.items()})
            self.logger.log(row, step=step)

            X_buf.append(X)
            y_buf.append(y)
            while len(X_buf) * batch_size > self.window_size:
                X_buf.pop(0)
                y_buf.pop(0)
            self._recent_batches.append((X, y))
            if len(self._recent_batches) > self.apd_k + 3:
                self._recent_batches.pop(0)

            if cooldown > 0:
                cooldown -= 1
                if cooldown == 0:
                    Xw = np.vstack(X_buf)
                    yw = np.concatenate(y_buf)
                    # 创新点 2：严重度分档 → 分级重训练
                    attr = self.detector.classify(
                        drift, loss, baseline_rmse,
                        residual_factor=self.residual_factor)
                    tier = attr["tier"]
                    # P1 策略控制器（可选）：证据 → 决策，异常自动回退规则
                    if self.controller is not None and tier != "observe":
                        try:
                            dec = self.controller.decide({
                                "tier": tier,
                                "severity": attr["severity"],
                                "residual_ratio": attr["residual_ratio"],
                                "psi_term": attr["psi_term"],
                                "psi": drift["mean_psi"],
                                "mean_psi": drift["mean_psi"],
                                "mmd_z": drift["mmd_z"],
                                "ks_p_value": drift["ks_p_value"],
                                "labels_left": self.label_budget - self.labels_used,
                                "retrains_left": self.max_retrains - self.n_retrains,
                                "batch_index": step,
                                "residual_alarm": bool(residual_alarm),
                                "input_alarm": bool(input_alarm),
                                "task_alarm": bool(task_alarm),
                                "mode": "sudden",
                            })
                            if dec.get("tier"):
                                tier = dec["tier"]
                            self.logger.log({"event": "controller_decision",
                                             "mode": self.controller.name,
                                             "tier": tier,
                                             "rationale": dec.get("rationale")})
                        except Exception as exc:  # noqa: BLE001
                            self.logger.log({"event": "controller_fallback",
                                             "error": str(exc)[:200]})
                    # P1：APD 闸门——持续报警但残差平坦 ⇒ 基准过期，抑制重训
                    if (self.apd_enabled and tier != "observe"
                            and self.alarm_streak >= self.apd_k
                            and not residual_alarm
                            and self.apd_suppress_streak
                            < self.apd_max_suppress):
                        # 只在"确实会发生重训"时抑制，避免虚假节省预算
                        self.apd_suppress_streak += 1
                        streak = self.alarm_streak
                        self.alarm_streak = 0
                        self.logger.log({"step": step, "event": "apd_suppress",
                                         "alarm_streak": streak,
                                         "tier": tier,
                                         "mmd_z": drift["mmd_z"],
                                         "cohesion": self._health_check(Xw),
                                         "suppressed": self.apd_suppress_streak})
                        history.append(row)
                        continue
                    if tier != "observe":
                        self._log_attribution(step, tier, Xw)
                    updated, new_rmse, mode = self._retrain_attributed(
                        Xw, yw, loss, baseline_rmse, tier,
                        severity=attr["severity"])
                    if updated:
                        baseline_rmse = new_rmse
                        self.detector.fit_reference(Xw)
                        self.logger.log({"step": step, "event": "retrained",
                                         "tier": tier, "severity": attr["severity"],
                                         "mode": mode,
                                         "old_rmse": loss,
                                         "new_rmse": new_rmse,
                                         "accept_n": len(yw),
                                         "labels_used": self.labels_used,
                                         "n_retrains": self.n_retrains})
                history.append(row)
                continue

            # 突变漂移（非渐变）才触发全量重训练 cooldown
            if alarm_count >= 2 and not self._in_gradual_mode:
                self.logger.log({"step": step, "event": "drift_detected",
                                 "structural": drift["structural_alarm"],
                                 "marginal": drift["marginal_alarm"]})
                cooldown = max(1, self.window_size // batch_size)
                # S2 快速恢复：高置信漂移把恢复期从 3 批降到 1 批
                # a1：连续两次报警即视为高置信；
                # a2：必须有双路漂移 OR 残差 > 4×基线，才加速恢复
                if self.fast_recovery != "off":
                    if self.fast_recovery == "a2":
                        confident = (drift["drift_detected"]
                                     or loss > 4 * baseline_rmse)
                    else:
                        confident = True  # a1：已在 alarm_count>=2 分支内
                    if confident:
                        cooldown = 1
                        self.logger.log({"event": "fast_recovery",
                                         "mode": self.fast_recovery,
                                         "cooldown": 1,
                                         "dual_drift": bool(drift["drift_detected"]),
                                         "residual_4x": bool(loss > 4 * baseline_rmse)})
                alarm_count = 0
            history.append(row)

        # 算力/时长证据：整轮汇总（客观分"单位增益消耗"的本地可审计记录）
        self.logger.log({
            "event": "compute_summary",
            "wall_s": round(time.perf_counter() - t_run0, 3),
            "cpu_s": round(time.process_time() - c_run0, 3),
            **{f"{k}_s": round(v, 3) for k, v in self._wall.items()},
            **{f"cpu_{k}_s": round(v, 3) for k, v in self._cpu.items()},
            "attrib_calls": self._attr_attempts,
            "attrib_skipped": self._attr_skipped,
            "labels_used": self.labels_used,
            "n_retrains": self.n_retrains,
            "cpu_count": os.cpu_count() or 1,
        })
        return history
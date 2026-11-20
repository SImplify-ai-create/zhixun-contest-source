# 智训引擎 · 工业模型自进化系统

![CI](https://github.com/SImplify-ai-create/zhixun-engine/actions/workflows/ci.yml/badge.svg)

第二届北京市大学生"人工智能+"创新大赛 · 企业赛题"智训引擎"参赛作品。

核心闭环：**漂移监测 → 诊断归因 → 分级重训练 → 评估上线 → 回滚**，
在预算约束（重训练次数 / 标注样本）下维持并提升模型在漂移数据流上的性能。

## 核心闭环创新 + 调度机制 + v1.1 比赛闸门保障

### 核心闭环三大创新点（v1.0 原始）

| # | 创新点 | 核心机制 | 解决的问题 |
|---|--------|---------|-----------|
| 1 | 解耦式双路漂移监测 | 冻结参考编码器（PCA）+ 自校准 MMD²（表征联合分布）+ KS 检验（边际分布，Bonferroni 校正）双路联合判据 | 用模型错误率监控会被自我适应欺骗（盲点悖论）；双路联合把稳定期空流误报从单路（KS 未校正）28/90 批压到 0/90 批 |
| 2 | 漂移严重度分级重训练 | 连续严重度 = max(残差倍数/残差系数, PSI 效应量)，输入侧证据封顶 2.6（低于 full 线）→ <1.0 `observe`（不重训）/ 1.0–3.0 `incremental`（增量微调 20 轮，keep_ratio 0.7）/ ≥3.0 `full`（清洗后全量） | 把预算花在真正需要全量重训练的时刻；轻微抖动不再白白消耗重训槽位，纯域偏移只走增量 |
| 3 | 选择性数据清洗 | 马氏距离剔除过时样本；`full` 档 keep_ratio=0（旧标签失效），`incremental` 档保留 0.7 | 避免新旧数据梯度错位，用更少标注量达到全量重训练精度 |

### 附加审计（创新点 2+，v1.0 新增）

| # | 审计机制 | 核心机制 | 价值 |
|---|---------|---------|------|
| 2+ | SHAP 误差归因（只读） | 重训前对当前模型在漂移窗口做 KernelSHAP，与参考窗口对照取"贡献增量 delta"；域漂移应定位到具体特征（实测锁定被注入平移的 f0-f2），深概念漂移 delta 天然平坦（SHAP 解释模型函数，映射换了它看不见）——平坦本身就是概念侧漂移的证据 | 把"报警 → 定位"做成可审计产物；归因失败自动降级为日志，绝不中断现场跑批 |

### 调度机制（创新点 4/5，v1.0 新增）

| # | 调度机制 | 默认 | 核心机制 | 价值 |
|---|---------|------|---------|------|
| 4 | 槽位压力预算节流 | 开 | 剩余重训槽位 >1 时不额外限额，只剩最后 1 个槽位才按剩余预算/槽位限额 | 避免"第一次重训吃光标注、晚到的漂移没钱修"的预算饥饿 |
| 5 | 漂移速率自适应窗口 | 开 | 残差连续两步上升才缩短取数窗口（快漂移 1 批 / 中 2 批 / 稳定 3 批） | 固定 600 条窗口在漂移期混入 30% 过渡期数据，滞后追不上漂移 |

### v1.1 比赛闸门保障（2026-11-02 整改）

针对比赛验收门槛（不低于静态基线、留出验证、无乐观偏差）新增六道闸门。实测效果见"已验证结果"一节。

| # | 闸门/保障 | 默认 | 核心机制 | 实测效果 |
|---|----------|------|---------|---------|
| 6 | 留出验收闸门（P0） | holdout | 重训候选在独立留出集（最近 20% 数据）上验收，训练/验收/回滚使用同一切片 | 训练集自评比留出集乐观 1.5–3.9 倍（runs/probe_accept_gate.csv）；当前 3 种子实测末段：突变 0.190、渐变 1.441 |
| 7 | 静态基线锚（P0） | 开 | 冻结初始静态基线并行打点，候选在验收集上劣于基线 +5% 时拒绝更新 | 全程劣于基线只发生在漂移刚到达 0–5 批（共 27 批/流） |
| 8 | APD 闸门（P1） | 开 | 持续报警抑制重训：连续 k 批报警且残差稳定不上升时，认定为慢漂移/假报警，跳过重训 | domain seed7 抑制 1 次不必要重训、省 600 标注；rebaseline 整体 A3 不必要重训 0.3 次（A0/A1/A2 为 0.6–0.8） |
| 9 | 简化版软门配额（P1） | 开 | 仅 incremental 档缩训练切片至 60%（下限 120），full 档永远全量；验收集从完整窗口取，与 quota 彻底解耦 | domain 同时降末段 0.187→0.166、省标注 660、降方差；full 档零副作用 |
| 10 | 抗遗忘 retention_ratio 回滚（P2） | 关 | 新旧数据分布重叠度超阈值时，触发半旧半新的保守重训而非全量清洗 | 默认关闭，作为深概念漂移场景的兜底安全件 |
| 11 | 渐变窗口优化 | gradual_window=1 | 渐变漂移场景把取数窗口从 3 批缩为 1 批，避免混入过渡期数据 | 渐变场景末段 RMSE 进一步收敛 |

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_all.py                         # 一键跑通全部实验，汇总到 runs/SUMMARY.md
python scripts/run_pipeline.py                    # 单次：突变漂移（默认 MLP）
python scripts/run_pipeline.py --model kalman     # 换模型：mlp | kalman | hybrid
python scripts/run_pipeline.py --drift gradual    # 单次：渐变漂移
python scripts/run_pipeline.py --source official --data-dir data/official   # 接入官方数据包
python scripts/run_pipeline.py --no-swanlab       # 离线模式，只写本地 CSV
```

**数据放哪**：官方数据包解压到 `data/official/`，主文件命名 `val.csv` / `val.npz` / `val.npy`（三选一）。
**跑完看哪**：终端汇总 + `runs/<实验名>-<时间戳>.csv`（逐批明细）+ `-summary.json` + `report-*.png`（曲线图）。

**展示台 / 控制台**（都是看板，不参与推理）：

```bash
python scripts/build_dashboard.py --skip-probe    # 生成 docs/dashboard.html，双击即开、离线可用
python scripts/serve_console.py                   # 本地控制台 http://127.0.0.1:8900，可选模型并实时看日志
```

配置统一在 `config.json`（命令行参数 > 环境变量 > config.json > 默认值）。
完整的运行、配置、镜像打包与故障排查见 [运行说明文档](docs/RUNBOOK.md)。

输出：控制台汇总 + `runs/` 下的实验记录 CSV、summary JSON 与 RMSE/PSI 曲线图。
登录 SwanLab 后实验自动同步云端（赛题指定实验记录平台）。
👉 **[SwanLab 实验看板](https://swanlab.cn/@Simplify-ai-create/zhixun-engine)**（合成流历史 run 已批量上传；
C-MAPSS / UCI 气体 / XJTU-SY 三套真实数据的每个 run 也已逐条上传，链接清单见
`real_data/最终结果表.md` §10）。

## 结构

```
src/data.py       合成漂移数据流（突变/渐变 + 三档生成器 domain/shallow/deep）
src/dataset.py    数据接入层：合成数据 / 官方数据包统一接口（npz/npy/csv）
src/config.py     配置加载（config.json + 环境变量）
src/drift.py      创新点1：冻结参考编码器 + MMD²+KS 双路检测；创新点2前置：严重度分档
src/cleaning.py   创新点3：选择性数据清洗（马氏距离）
src/model.py      预测模型三选：BaselineRegressor(MLP) / KalmanRegressor(CV-KF 追踪 w) / HybridRegressor(MLP+残差KF)
src/attribution.py  SHAP 误差归因（KernelSHAP，漂移窗口 vs 参考窗口贡献增量）
src/pipeline.py   自进化闭环（v1.1：留出验收/静态锚/APD/软门解耦 + model_cls 可切）
src/decision.py   v2.0 改进版：代价敏感动作决策（默认关，见"方案版本"）
src/advanced.py   创新点4/5 + v1.1 软门配额简化版（子类，最小侵入）
src/logger.py     SwanLab + CSV 双通道实验记录（含登录预检）
src/strategy.py   策略控制器（threshold/rule/llm 三种模式，LLM 异常自动回退）
config.json       运行配置（v1.1 默认）
Dockerfile        系统镜像（离线评测入口）
docs/RUNBOOK.md   系统运行说明文档
docs/方案版本说明.md  v1.0 / v1.1 / v2.0 切换说明
scripts/run_pipeline.py       单次运行入口
scripts/run_all.py            一键复现全部实验
scripts/ablation.py           消融实验（含三档漂移场景 + severity 签名修复）
scripts/tier_probe.py         三档漂移的严重度打点（先看分布再调阈值）
scripts/tier_actions.py       分级动作对照（observe / incremental / full / auto）
scripts/calibrate_actions.py  v2.0 动作反事实标定（生成代价敏感决策的标定表）
scripts/gating_ablation_v11.py 软门配额消融（6 场景 × 2 变体 × 3 种子）
scripts/eval_rebaseline.py    rebaseline A0–A6 完整评估（140 runs）
scripts/analyze_rebaseline.py H1–H5 + E3 假设判定
scripts/probe_accept_gate.py  留出验收 vs 训练集自评、APD 闸门回归
scripts/probe_correction.py   证伪实验：裁判模型伪标注（见"已证伪方案"一节）
scripts/probe_gold_judge.py   证伪补充：独立金标准裁判
scripts/probe_judge_ablation.py 证伪补充：融合权重/探针条数/实时裁判三组消融
scripts/probe_calibration.py  证伪实验：少样本输出校验层（对照组有缺陷，见 README）
scripts/probe_pure_retrain.py 公平对照：纯新样本重训 vs 校验层（证伪校验层）
scripts/sweep_scale.py        规模扫描框架（n/特征/宽度参数化，本地与云通用）
scripts/kalman_probe.py        卡尔曼追踪探针（验证 0.140 噪声下限，q_w/q_v 扫描）
scripts/upload_runs_to_swanlab.py 批量上传历史实验 CSV 到 SwanLab
```

## 已验证结果（合成数据）

生产入口 `scripts/run_pipeline.py`（`config.json`：6 个重训槽位 / 4000 标注预算），
三个随机种子（42 / 7 / 2024）的均值 ± 标准差：

| 场景 | 峰值 RMSE | 末段 RMSE | 初始（静态基线）RMSE | 重训练次数 | 标注消耗 |
|------|----------|----------|----------|-----------|---------|
| 突变漂移（MLP） | 7.286 ± 1.358 | **0.190 ± 0.039** | 0.168（自评）/ 0.167（折外均值） | 3.3 | 1833 |
| 渐变漂移（MLP） | 2.387 ± 0.592 | 1.441 ± 0.482 | 0.168（自评）/ 0.167（折外均值） | 4.7 | 1820 |
| **突变漂移（卡尔曼）** | **1.334 ± 0.104** | **0.140 ± 0.001** | 0.156~0.176（Ridge 折外） | 2.0 | 960 |
| **渐变漂移（卡尔曼）** | **0.164 ± 0.010** | **0.140 ± 0.001** | 0.156~0.176（Ridge 折外） | 2.0 | 960 |
| 突变/渐变漂移（混合） | 1.37 / 0.40 | 0.360 ± 0.060 | 0.24~0.29 | 1.7 | 840 |

> 卡尔曼模型：`--model kalman`，恒定速度 KF 追踪线性映射 w(t)。合成流 `y=X@w(t)+noise`
> 线性，KF 接近噪声下限（noise=0.1）。**渐变末段 −90%、峰值 −94%**；重训次数与标注减半。
> 混合模型 `--model hybrid`：MLP 主体 + 残差 KF，线性数据上非最优（MLP 主体是多余负担），
> 用于真实非线性数据。**模型选择：先跑 Ridge R²，高→kalman，中→hybrid，低→mlp**。

v1.1 整改（2026-11-02）集中在**评估口径**而非模型容量——旧实现"重训后在同一批数据上自评"，
留出验收后的真实末段误差与自评差 1.5~3.9 倍（见 `scripts/probe_accept_gate.py` 输出
`runs/probe_accept_gate.csv`）。上表为当前代码 3 种子实测（2026-11-02 重跑，原始记录
`runs/synthetic-sudden-1102-1805*.csv`、`runs/synthetic-gradual-1102-1806*.csv`）。

如实披露 v1.0→v1.1 的 tradeoff：v1.0 无闸门口径下渐变末段曾测得 0.859，但那是
"4 次渐变重训全被验收拒绝"的副产物（重训不生效，模型靠初始拟合硬扛漂移）；v1.1 让重训
真实生效并通过验收，换来 H1b（抑制 35.7% 不必要重训）、H3（过渡期命中 100%→0%）、
H5（复发记忆召回 100%）、E3（全臂不劣于 A0）成立。渐变最大单种子 1.977（seed=2024），
如实列为短板。

门槛③（不低于静态基线）全程用冻结静态基线并行打点：末段 `RMSE/静态基线` 突变 0.026~0.053、
渐变 0.175~0.421（本次 3 种子实测）；劣于基线只发生在漂移刚到达、cooldown 未结束的
0~5 个批（共 27 批/流，最差 5/27）。

误差曲线（由仓库当前代码生成，纵轴为 RMSE，可看到漂移点位置与恢复过程）：

- 突变漂移：[docs/results/synthetic-sudden.png](docs/results/synthetic-sudden.png)
- 渐变漂移：[docs/results/synthetic-gradual.png](docs/results/synthetic-gradual.png)

### 预测模型：三档自适应选择

三套模型共享同一套检测/验收/回滚/预算闸门，通过 `--model` 切换。选择策略：
**先跑 Ridge 拟合 y 的 R²**——R² 高（线性）→ kalman；R² 中（非线性但残差近似线性慢漂移）→ hybrid；R² 低（强非线性）→ mlp。

| | MLP（默认） | 卡尔曼（方案 A） | 混合（方案 B） |
|---|---|---|---|
| `--model` | `mlp` | `kalman` | `hybrid` |
| 思路 | 误差反应式：漂移→报警→重训快照 | 状态估计式：追踪线性映射 w(t)，CV-KF 逐样本更新 | MLP 非线性主体 + KF 追踪残差线性映射 a(t) |
| 预测式 | `y_hat = MLP(x)` | `y_hat = x^T w(t)` | `y_hat = MLP(x) + x^T a(t)` |
| 适用 | 通用，强非线性 | 特征→目标近似线性 | 非线性主体 + 残差慢漂移 |
| 渐变末段 | 1.441 ± 0.482 | **0.140 ± 0.001（−90%）** | 0.341（seed42） |
| 突变末段 | 0.190 ± 0.039 | **0.140 ± 0.001（−26%）** | 0.341（seed42） |
| 峰值 | 7.3 / 2.4 | 1.3 / 0.16 | 1.3 / 0.39 |
| 重训/标注 | 3.3/1833 · 4.7/1820 | 2.0/960 · 2.0/960 | 2.0/960 · 1.0/600 |

**为什么卡尔曼（方案 A）有效**：合成流 `y = X @ w(t) + noise`，渐变段 w(t) 是分段线性
时间函数——正好是恒定速度 KF 的建模假设。KF 每样本"时间更新→一步预测→测量更新"，
连续追踪 w 的水平与变化率，无需等待周期重训，末段逼近噪声下限（0.14 vs noise 0.1）。

**方案 B 的定位**：当数据非线性、方案 A 失配时，MLP 捕捉非线性主体，残差 KF 在线校正
慢漂移。合成流为线性，方案 B 不是最优（MLP 主体是多余负担），但在真实非线性数据上应优于纯 MLP。

**关键前提与红线**：
- 卡尔曼/混合只作用于**预测/跟踪**环节，检测、回滚、预算逻辑不动。
- **线性度检验优先**：官方数据到手先跑 Ridge R²，再决定 `--model`。
- 标签口径：track() 每批用 y 在线更新（与 evaluate 用 y 同口径，流式在线学习），不计入 `label_budget`。
- 突变跳变：CV 模型有约 5 批收敛滞后（峰值 1.3），可加"大创新检测→重置 P"压缩。


实验结果原始数据（CSV/JSON）：

- 软门配额消融：[runs/gating_ablation_v11.csv](runs/gating_ablation_v11.csv)
- rebaseline 完整评估（A0–A6）：[runs/rebaseline_eval.csv](runs/rebaseline_eval.csv)
- 留出验收探针：[runs/probe_accept_gate.csv](runs/probe_accept_gate.csv)
- rebaseline 假设判定汇总：[runs/rebaseline_verdicts.md](runs/rebaseline_verdicts.md)

### 消融实验（3 种子均值 ± 标准差，v1.0 旧口径——v1.1 留出验收后尚未全部重跑）

> 下表为 v1.0 自评口径的消融结果；v1.1 整改后主流程数字见上一节。
> 待办：v1.1 下重跑全部 4 配置 × 6 场景 × 3 种子消融，更新本表。

| 配置 | 突变末段 RMSE | 渐变末段 RMSE | domain 末段 RMSE |
|------|-------------|-------------|-----------------|
| 完整系统（三创新点） | 0.187 ± 0.058 | 2.040 ± 1.207 | **0.147 ± 0.020** |
| 去掉创新点1（仅残差检测） | 0.153 ± 0.009 | 3.237 ± 0.953 | 0.212 ± 0.044 |
| 去掉创新点2（无归因，恒全量） | 0.187 ± 0.058 | 1.772 ± 0.729 | 0.176 ± 0.013 |
| 去掉创新点3（无数据清洗） | 1.561 ± 0.474 | 2.071 ± 1.233 | 0.155 ± 0.017 |

- 创新点 3 是突变恢复的前提（去掉后 1.561）；创新点 1 是渐变场景的根因（去掉后 3.237）。
  突变场景去 1 反而 0.153——端到端差异落在种子噪声内，**创新点 1 的价值在误报抑制
  （0/90 vs 28/90），不声称端到端 RMSE 更优**。
- 创新点 2 在 domain 场景首次出现消融区分度（0.147 vs 0.176，差值大于双方标准差）。

### 分档有效性（创新点 2，`scripts/tier_actions.py`）

三档漂移生成器（`src/data.py::TieredDriftStream`）：domain 只换 X 边际、shallow 浅改
P(Y|X)、deep 整体换映射。自动分档命中（3 种子）非退化：
domain_mild {observe 5, incremental 3}、domain {incremental 3}、shallow {incremental 3}、
deep {incremental 1, full 6}。

| 场景 | 自动分档 末段/标注 | 最优固定动作 | 恒 full 末段/标注 |
|------|------------------|-------------|------------------|
| domain_mild | **0.156 / 1533** | observe 0.178 / 933 | 0.172 / 2200 |
| domain | **0.155 / 2933** | incremental 0.155 / 2933 | 0.176 / 2467 |
| shallow | 0.196 / 1400 | full 0.180 / 1733 | 0.180 / 1733 |
| deep | **0.185 / 1933** | full 0.180 / 1933 | 0.180 / 1933 |

如实披露：domain 类**同时**降误差省标注；shallow 以约 19% 标注节省换约 9% RMSE 退化；
sudden / deep 严重度天然落 full，与恒全量消融组必然相同（预期，非缺陷）。

### 软门配额消融（v1.1，`scripts/gating_ablation_v11.py`）

软门只缩 **incremental 档**的训练切片配额（固定 60%，下限 120 条），full 档永远全量；
验收集从完整窗口取，与 quota 彻底解耦（根治旧实现"quota 缩→训练+验收一起缩→误升级"的负例）。

| 场景 | baseline 末段/标注 | soft_quota 末段/标注 | 末段 Δ | 标注 Δ |
|------|-------------------|---------------------|--------|--------|
| sudden（full） | 0.190 ± 0.039 / 1833 | 0.190 ± 0.039 / 1833 | 0 | 0 |
| gradual | 1.359 ± 0.594 / 1933 | 1.441 ± 0.482 / 1820 | +0.082 | -113 |
| domain_mild | 0.177 ± 0.031 / 1000 | **0.164 ± 0.010** / 840 | -0.013 | -160 |
| domain | 0.187 ± 0.054 / 1933 | **0.166 ± 0.020** / 1273 | -0.021 | -660 |
| shallow | 0.317 ± 0.072 / 667 | 0.317 ± 0.072 / 667 | 0 | 0 |
| deep（full） | 0.185 ± 0.042 / 1867 | 0.185 ± 0.042 / 1787 | 0 | -80 |

- **domain seed7 正例保留且增强**：0.197/1800 → 0.153/1180（末段 -0.044，省 620 标注）。
- **deep seed42 负例消除**：旧实现 0.161/1800 → 0.183/2599（解耦前耦合 bug）；
  解耦后 baseline 与 soft_quota 完全一致（0.161/1800/3）。
- full 档场景（sudden/deep/shallow）零副作用；incremental 档主场（domain/domain_mild）
  同时降误差、省标注、降方差。`config.json` 已默认 `soft_quota=true`。

### 迟滞门方案（暂缓，端到端零作用）

迟滞门是一种抗抖安全件：当严重度来回穿越阈值时，用状态机锁定在当前档位，避免频繁切换。

**状态机逻辑正确**（震荡序列 7→1 次翻转），但**端到端零作用**。根因有三：

1. `classify` 只在 cooldown 结束后调用（一条流 2–4 次），迟滞门几乎没有出场机会；
2. 每次重训成功调用 `fit_reference` 会清空 `_tier` 状态，迟滞门刚生效就被重置；
3. 合成流 severity 是单调跳变（0→observe→incremental→full），天然不在阈值附近震荡。

**结论**：迟滞门保留为抗抖安全件（默认关闭），不声称收益。暂缓到官方数据包，
待确认存在阈值附近抖动信号后再启用。

### Rebaseline 再定标实验（140 runs，6 臂 × 4 流 × 5 种子）

验证"报警-性能背离 + 健康度自检"能否替代传统再定标，完整结果见 `runs/rebaseline_verdicts.md`。

| 假设 | 结论 | 说明 |
|------|------|------|
| H1 APD 省重训 | ❌ 未达标 | 相对 A0/A1 下降仅 7.1%，门槛 ≥30% |
| H1b A6 门控重训 | ✅ 成立 | 抑制不必要重训 35.7%，检出延迟 0 批 |
| H3 健康度自检 | ✅ 成立 | 过渡期命中从 100% 降到 0%，RMSE 上升段命中显著下降 |
| H5 复发记忆 | ✅ 成立 | 召回率 100%，标注从 1800 降到 1200，延迟 5 批 |
| H2 双尺度参考 | ❌ 证伪 | 检出延迟安全线被击穿（+16.4 批，约束 ≤+1） |
| H4 误报闭环 | ⏸ 待定 | 双路本身 0 误报，无事可做；需变噪声流才能公平裁决 |
| E3 回归约束 | ✅ 通过 | 两场景所有臂均未劣于 A0 |

**核心张力**：v1 早再定标可省 38% 重训但存在 78.3% 污染问题，v2 安全再定标实现 0% 污染
但无法节省预算。最终采用 A6 门控重训方案——APD 不触发再定标，改为直接抑制不必要重训。

## 真实数据与结果（三套公开数据集）

对外结论只引用三套公开学术数据集：NASA C-MAPSS（涡扇退化）、UCI 气体传感器阵列漂移、
XJTU-SY（轴承全寿命）。**数据文件不随仓库分发**（`real_data/` 在工作区本地，已被 `.gitignore` 排除），
仓库内只保留口径、结果与复现命令。

| 文档 | 内容 |
|------|------|
| [docs/最终结果表.md](docs/最终结果表.md) | 对外数字总表：C-MAPSS 四子集、UCI 气体、XJTU-SY 六轴承，含口径边界、作废数字与复现命令 |
| [docs/真实数据迁移规划.md](docs/真实数据迁移规划.md) | 三套数据的接入方案、批次构造、验收留出门槛与已拍板决策 |
| [docs/数据清单.md](docs/数据清单.md) | 逐文件字节数 / sha256 / 行数 / 维度与许可证，数据完整性校验记录 |
| [docs/DEPLOY_RUNBOOK.md](docs/DEPLOY_RUNBOOK.md) | 真实数据上的部署与运行手册（离线评测、镜像、故障排查） |
| [docs/XJTU_SY_校验报告.md](docs/XJTU_SY_校验报告.md) | XJTU-SY 9216 个 csv 的逐轴承文件数与形状核对报告 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方数据集与依赖许可证声明 |

口径提醒（引用任何数字时必须一并写）：不主张 SOTA，只主张同数据同预算下的臂间相对优势；
XJTU-SY 的 `rul_ratio` 需知道全寿命 N 才能构造，**只能离线回放**，且预测值已统一裁剪至 [0, 1]。
企业数据（`data/factory`、`real-j820189-*`）严格隔离，不进入本仓库、报告与任何对外材料。

## 方案版本（答辩口径）

- **v1.1 现行**（2026-11-02 整改）：MMD²+KS 双路 + 严重度三档 + 清洗/节流/窗口 +
  留出验收闸门 + 静态基线锚 + APD 闸门 + 简化版软门配额。`config.json` 已全部默认开启。
- **v1.0 历史**：MMD²+KS 双路 + 严重度三档 + 清洗/节流/窗口（自评口径，已被 v1.1 替换）。
- **v2.0 改进**：动作反事实标定（`scripts/calibrate_actions.py`）+ 代价敏感决策
  （`src/decision.py`，argmin 预测RMSE + λ·标注）。`config.json` 把
  `decision.mode` 改为 `cost_sensitive` 启用；标定表缺失时自动回退 v1.1 规则。
- **备用**：`psi_top_k`（高维特征敏感筛选，默认关）、`tier_bounds` 公开集反解、
  按漂移类型分档（幅度分档失灵时）。详见 [docs/方案版本说明.md](docs/方案版本说明.md)。

## 已证伪方案：裁判模型伪标注（不实现）

**设想**：引入一个固定在标准数据上的裁判模型，在主模型漂移时把错误样本标出来
（伪标注），重新投入训练，从而在不增加人工标注预算的前提下扩量。

**结论：否决。** `scripts/probe_correction.py` 在合成流（ground truth 已知）上
对 deep / shallow / domain 三种漂移各跑 3 个种子（42 / 7 / 2024），逐项检验
四项必要前提（P1–P4），均不成立。

### 实验设定

- 段0 [0, 2000) 稳定：训练主模型 `(64, 32)` 与冻结裁判 `(128, 64)`
  （不同结构 + 不同种子，用于检验"换模型是否带来独立信息"）
- 段1 [2000, 4000) 漂移1：适应裁判在此用 20% 真标签增量微调
- 段2 [4000, 6000) 漂移2：评测区，A 半（1000 条）用于取得标签，B 半 held-out
- 所有重训组统一按"段0 基线数据 + 新数据全量 fit"，避免"轮数 × 样本量"混淆对比

### P1 分歧携带信息 → 不成立

| 场景 | 主模型 RMSE | 冻结裁判 RMSE | 分歧 AUC | 同向大错占比 |
|------|------------|--------------|---------|------------|
| deep | 3.438 | 3.443 | **0.524**（等同随机） | **100%** |
| shallow | 0.264 | 0.270 | 0.517 | 0% |
| domain | 0.177 | 0.178 | 0.605 | 0% |

deep 场景即使换成完全不同的网络结构，两模型仍然 100% 同向错，分歧对"主模型真错"
的区分度等同随机。**分歧的独立信息来自数据独立性，不是模型独立性**——共享同一
训练分布时，裁判就是主模型的复制品。

### P2 伪标注可替代真标签 → 不成立

伪标注相对真实标签的 RMSE：deep 3.512（冻结）/ 2.172（适应），与主模型自身误差
3.438 同量级；domain 0.181（冻结），与主模型 0.177 同量级。伪标注不是"近似正确
的标签"，而是**主模型当前偏差的固化**。

### P3 伪标注训练优于少量真标签 → 不成立

| 场景 | 不重训 | 20% 真标签 | 100% 真标签 | 冻结伪标注 100% | 20%真 + 80%伪 |
|------|--------|-----------|------------|----------------|--------------|
| deep | 3.365 | 3.148 | **2.555** | 3.373 | 2.908 |
| shallow | 0.265 | 0.244 | **0.207** | 0.260 | 0.222 |
| domain | 0.176 | 0.140 | **0.129** | 0.180 | **1.201** |

- 纯伪标注在**三个场景全部输给 20% 真标签**（分别差 7%、6.5%、29%）
- domain 场景纯伪标注 0.180 **劣于不重训** 0.176
- 混合组在 deep / shallow 有小幅正向（相对 20% 真标签改善 7.6% / 9%），
  但在 domain 是 **7.6 倍劣化**（1.201 vs 0.140）

即：收益幅度是个位数百分比，风险量级是数倍，且两者取决于同一个不可靠的判据
（裁判自身是否可信）。

### P4 分歧可指导标注预算分配 → 不成立

同样 200 条真标签，按不同策略挑选：

| 场景 | 随机 | 分歧最大（不确定性采样） | 分歧最小 |
|------|------|------------------------|---------|
| deep | 3.204 | 3.164 | 3.244 |
| shallow | 0.251 | 0.254 | 0.250 |
| domain | **0.139** | 0.155 | 0.142 |

domain 场景不确定性采样反而更差；其余场景差异落在种子噪声内。

### 附带证伪：零标注输入矩匹配

把当前批 X 逐维对齐回参考分布（均值/方差匹配，CORAL 简化版）后直接预测：

| 场景 | 不处理 | 输入矩匹配 |
|------|--------|-----------|
| deep | 3.365 | 3.436（无收益，符合预期：P(Y|X) 已变） |
| shallow | 0.265 | 0.294（劣化） |
| domain | 0.176 | **1.606（灾难）** |

domain 的 X 均值平移是**真实信号**（y = X@w，X 变则 y 应随之变），把它拉回参考
分布等于抹掉合法输入变化。**"P(Y|X) 不变"不等于"Y 不变"**。

### 这份证伪的三处用途

1. **创新点 3 的正向依据**：选择"丢弃过时样本"不再只是保守做法，而是有实验支撑的
   决策——修复路线会污染训练集，且污染可量化。
2. **创新点 2 的独立验证**：domain 场景 200 条即吃掉大部分收益（0.140 vs 全额 0.129），
   deep 场景 20% 远不够（3.148 vs 2.555）。这正是"按漂移类型分配预算"的预期行为。
3. **答辩防守材料**：面对"为何不用伪标注省预算"，直接给出上述三组实验，而非结构性
   回答"我们没做"。

### 口径说明

- 全部数字为自研合成流（`src/data.py::TieredDriftStream`）实测、3 种子均值；
  执行 `python scripts/probe_correction.py` 可复现，原始记录在
  `runs/probe_correction.json`。
- 合成流的 domain 漂移只做前 3 维均值平移，真实场景协变量漂移形态可能不同，
  **本结论不外推到"所有协变量漂移"**。
- 本实验只测了"裁判多看段1 的 20% 真标签"这一种增强方式，**未测"裁判用更大规模
  独立金标准训练"**。后续补测（`scripts/probe_gold_judge.py`）显示：即使换成独立
  采样、覆盖漂移后分布的金标准裁判，分歧 AUC 上限也只有 0.695，判错器角色依然
  不成立；其作为预测源的边际收益（deep −6%）经下节实验确认同样不成立。

## 已证伪方案：少样本输出校验层（不实现）

**设想**：概念漂移的误差主要是全局幅度欠估，用 20–100 条当前分布真标签拟合
1 参数增益校准（y≈a·ŷ）或 3 参数非负双源融合，替代重训，从而**不消耗重训槽位**。

**结论：否决。** 该设想曾通过两轮预实验（`probe_calibration.py`、`sweep_scale.py`），
甚至做到"100 条校准优于 5000 条全量重训"；落地端到端时突变场景末段 RMSE 从
0.209 退化到 3.744，复查发现**前两轮预实验的重训对照组有系统性缺陷**：它们把
"段0 旧映射数据 + N 条新数据"混在一起训练——这正是创新点 3（清洗，keep_ratio=0）
在生产管线里避免的情形。旧标签失效却占训练集 2/3，梯度冲突使重训被严重低估。

补上公平对照（`scripts/probe_pure_retrain.py`：只用 N 条当前分布样本从零训练，
等价于清洗后重训；3 种子均值，held-out RMSE，"/"前为增益校准）：

| 场景 | 基线 | N=20 | N=100 | N=600 |
|------|------|------|-------|-------|
| deep | 3.365 | 2.542 / **1.033** | 2.433 / **0.378** | 2.430 / **0.175** |
| shallow | 0.264 | 0.242 / 1.205 | 0.231 / **0.380** | 0.231 / **0.170** |
| domain | 0.176 | 0.174 / 0.893 | 0.168 / **0.379** | 0.167 / **0.180** |

**纯样本重训在所有预算下都不劣于校准**（deep 场景 N=20 时已达 1.033 vs 2.542），
校验层宣称的成本优势不复存在——同样的标注预算，直接重训永远更值。

### 保留的资产

- `scripts/sweep_scale.py`：规模扫描框架（n / 特征数 / 宽度参数化，本地/云同一份
  代码），后续验证其他结论的规模稳健性时直接复用。
- 裁判模型的最终判定：判错器、伪标注源、预测源三个角色全部证伪，
  **项目中不引入裁判/教师/校验等任何第二模型组件**。

## 实验性实现（默认关闭）：策略层自改进闭环（LLM 提规则）

**实验性实现、默认关闭、7 条提案 0 采纳、结论为负、不作为主路径。**

把"决策策略本身"交给外部提案方（LLM 或离线启发式）写数值 if-then 规则，再在
真实流上以同一数据/切分/预算/种子与固定阈值基线逐位对比，只有通过既有闸门
（非退化、最小增益 0.03、静态锚、预算槽位）才允许采纳。2026-11-11 在 FD001 /
FD003 / UCI 气体上完成端到端验证：**7 条提案全部被拒，未放宽任何阈值**。

- 设计、规则 schema、闸门口径与已知边界：[docs/llm_rule_loop.md](docs/llm_rule_loop.md)
- 验收报告：[docs/llm_rule_report.md](docs/llm_rule_report.md)
- 采纳账本（含每条拒绝原因）：[docs/llm_rule_ledger.jsonl](docs/llm_rule_ledger.jsonl)

未接入生产入口（`run_pipeline.py` / `run_real_*.py` 均不加载规则），`config.json`
无新增键。原因是结构性的：FD001 上控制器插槽从未被询问（`n_gated=0`），
气体/FD003 上规则能改变行为但改变全是负收益——可动作空间被渐变模式与预算节流
压缩，不是调阈值能解决的问题。

## 设计原则汇总（证伪沉淀）

所有证伪实验共同指向四条设计原则，作为后续新增方案的预筛标准：

1. **对照组必须复现生产管线的完整动作**。预实验为隔离变量简化了重训（不做清洗），
   结果把创新点 3 的贡献算到了校验层头上。证伪别人之前，先确认对照组没有被
   自己的简化 handicap。
2. **"参数少 → 样本效率高"的直觉在此不成立**。目标映射是低维线性结构时，
   神经网用 20 条样本就能粗略重建，增益校准的 1 参数优势被淹没。复杂度和数据效率
   的关系取决于问题结构，不能一概而论。
3. **端到端验证不可替代**。两轮预实验（含 30k 规模扫描）全部通过，落地一次
   就暴露问题——预实验只负责筛掉明显坏的方向，不负责证明方向正确。最终结论必须
   在生产管线上用留出口径验证。
4. **零标注修正不能假设 P(Y|X) 不变**。domain 场景下输入矩匹配把合法 X 平移信号
   抹掉，RMSE 从 0.176 飙升到 1.606。**"P(Y|X) 不变" 不等于 "Y 不变"**——
   协变量漂移中 X 的分布变化可能就是 y 变化的原因。

## 数据规模适配（设计口径）

系统所有成本的计量单位是"批/窗口"而不是"全量数据"：检测 200 条/批、SHAP 三封顶
（背景 50 / 评估 150 / nsamples 128）、单次重训由窗口+配额封顶。**数据总量只改变
批数，不改变单批成本**。3 万样本规模扫描（`scripts/sweep_scale.py`）各结论与
6000 样本一致，CPU 单次几分钟。

三个已知规模风险已做自适应保险（默认开，现场无需改代码）：

1. `X_hist` 无界增长 → 封顶保留最近 `hist_max` 条（默认 5000，近期样本对漂移最有价值）；
2. 马氏距离 O(d³) → 维数超 `max_mahalanobis_dim`（默认 100）自动 PCA 降到 32 维再算
   距离（300 维实测正常，输出 32 维距离）；
3. KS 显著性随批大小爆炸 → 已加效应量门控 `ks_effect_min`（默认 0.1）：报警须同时满足
   p<α 且 D≥0.1；5000 条大批实测微扰（D=0.059）不误报、真漂移（D=0.268）不漏报，
   且 200 条批回归结果逐位不变（MMD/PSI 支路本是效应量口径，不受影响）。

## 理论依据与参考文献

各机制的立论来源按下表对应，参考文献采用 GB/T 7714—2015 顺序编码制。

| 机制 | 文献给出的关键结论 | 文献 |
|------|-------------------|------|
| 创新点1 解耦式双路监测 | 自适应模型会用内部更新吸收后漂移误差瞬变，累积型检测器因而系统性漏报；监控信号须与更新回路解耦；两批独立样本的分布比较属两样本检验（MMD²），CKA 需要样本配对、不适用（[15][16] 仅作表征相似度度量的背景引用） | [8][15][16][18][19][20] |
| 创新点2 严重度分级重训 | 漂移分为 abrupt / gradual / incremental / recurrent，形态不同则最优响应不同；预算受限时应在低成本动作与重干预之间分级 | [1][2][3][9][10] |
| 创新点3 选择性数据清洗 | 深层（特征）与浅层（分类头）遗忘不对称，小缓冲能锚住特征但难抑浅层遗忘；新旧数据分布重叠会引发突变式表征重构 | [5][6][7] |
| 创新点4 槽位压力节流 | 标注预算是硬约束；延迟与排队会造成 stranded budget，使有效预算静默腰斩；标注速率应随观测到的漂移动态调节 | [9][10][11][12] |
| 创新点5 自适应窗口 | 重训前应先检验后漂移数据是否已充分；窗口长度本应随漂移速率自适应；过渡期混合数据参与拟合会诱发表征突变 | [4][17][5] |
| 回滚保护 | 漂移后"何时重置、重置哪部分"决定长期适应成败，延迟重置有害；自适应 reset 可在预测崩溃前恢复 | [13][14] |

### 参考文献

[1] GAMA J, ŽLIOBAITĖ I, BIFET A, 等. A survey on concept drift adaptation[J]. ACM Computing Surveys, 2014, 46(4): Article 44. DOI: 10.1145/2523813.
[2] CELIK B, VANSCHOREN J. Adaptation strategies for automated machine learning on evolving data[EB/OL]. arXiv:2006.06480, 2020. https://arxiv.org/abs/2006.06480.
[3] YUAN Liheng, LI Heng, XIA Beihao, 等. Recent advances in concept drift adaptation methods for deep learning[C]//Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI-22), Survey Track. 2022. https://www.ijcai.org/proceedings/2022/0788.pdf.
[4] FUJIWARA R, 等. When to retrain after drift: a data-only test of post-drift data size sufficiency (CALIPER)[EB/OL]. arXiv:2603.09024, 2026. Accepted by ICLR 2026. https://arxiv.org/abs/2603.09024.
[5] CACCIA L, ALJUNDI R, ASADI A, 等. New insights on reducing abrupt representation change in online continual learning[C]//Proceedings of the 10th International Conference on Learning Representations (ICLR). 2022. https://arxiv.org/abs/2203.03798.
[6] LANZILLOTTA G, 等. Heads collapse, features stay: why replay needs big buffers[EB/OL]. arXiv:2512.07400, 2025. https://arxiv.org/abs/2512.07400.
[7] ZHU Yunqin, 等. Why do neural networks forget: a study of collapse in continual learning[EB/OL]. arXiv:2603.04580, 2026. https://arxiv.org/abs/2603.04580.
[8] MINATO R, 等. The blind spot paradox: when adaptive classifiers defeat drift detectors[C]//IEEE ICDM 2026 Workshops (OWAD 2026). 2026. https://arxiv.org/abs/2610.05853.
[9] LAMAAKAL I, 等. Drift-to-action controllers: budgeted interventions with online risk certificates[C]//CAO Workshop at ICLR 2026. 2026. https://arxiv.org/abs/2603.08578.
[10] PIASECNY A, 等. RCCDA: adaptive model updates in the presence of concept drift under a constrained resource budget[C]//Advances in Neural Information Processing Systems (NeurIPS). 2025. https://arxiv.org/abs/2505.24149.
[11] DASARI S. When to retrain: an empirical study of retraining policies for streaming ML under concept drift, budget, and latency constraints[EB/OL]. arXiv:2608.19488, 2026. https://arxiv.org/abs/2608.19488.
[12] AZIZI S, 等. Maintaining IoT device identification under concept drift via budget-aware traffic labeling[EB/OL]. arXiv:2608.15465, 2026. https://arxiv.org/abs/2608.15465.
[13] MISHRA H. RDumb++: drift-aware continual test-time adaptation[EB/OL]. arXiv:2601.15544, 2026. https://arxiv.org/abs/2601.15544.
[14] LIM Taejun, 等. When and where to reset matters for long-term test-time adaptation[C]//Proceedings of ICLR 2026. 2026. https://arxiv.org/abs/2603.03796.
[15] KORNBLITH S, NOROUZI M, LEE H, 等. Similarity of neural network representations revisited[C]//Proceedings of the 36th International Conference on Machine Learning (ICML). 2019: 3519-3529. https://arxiv.org/abs/1905.00414.
[16] VANCE N, FLYNN P. Measuring domain shifts using deep learning remote photoplethysmography model similarity[EB/OL]. arXiv:2404.08184, 2024. https://arxiv.org/abs/2404.08184.
[17] BIFET A, GAVALDÀ R. Learning from time-changing data with adaptive windowing[C]//Proceedings of the 2007 SIAM International Conference on Data Mining. 2007: 443-448.
[18] MASSEY F J. The Kolmogorov-Smirnov test for goodness of fit[J]. Journal of the American Statistical Association, 1951, 46(253): 68-78.
[19] SIDDIQI N. Credit risk scorecards: developing and implementing intelligent credit scoring[M]. Hoboken: John Wiley & Sons, 2006: 137.
[20] GRETTON A, BORGWARDT K M, RASCH M J, 等. A kernel two-sample test[J]. Journal of Machine Learning Research, 2012, 13: 723-773. https://www.jmlr.org/papers/v13/gretton12a.html.

### 引用边界（须与实测数字区分）

- 上文所有性能数字（稳定期空流误报、末段 RMSE、标注消耗）均为本项目自研合成流上的实测结果，文献仅提供机制依据，不为这些数值背书。
- 检测指标口径：取稳定期（首批 600 条为参考，其后连续 90 批、每批 200 条）统计报警率。实测结构路（MMD²）0/90、边际路（KS，Bonferroni 校正后）4/90、双路联合 0/90、单路 KS 未校正 28/90。双路的价值在抑制误报，不代表双路在漂移期命中率更高。
- [1][2][3][9][10] 支撑的是"不同漂移形态应有不同响应"的原理。**合成流下该创新点已出现非零区分度**（domain / domain_mild 场景自动分档与恒全量存在差异，见上文分档有效性表），但 shallow 场景仍以约 19% 标注节省换约 9% RMSE 退化，sudden / deep 场景严重度天然落 `full`、与恒全量消融组必然相同；对外表述须如实区分这三类情形，不得笼统声称"分级在所有场景均优于全量"。
- [4] 仅借鉴"重训前先检验后漂移数据是否充分"的思想，未实现其 locality 参数单调性检验，不得表述为"复现 CALIPER"。
- CKA 仅适用于同一批样本的两种表征对齐（[15]），本系统结构路已改用 MMD²（[20]）；[16] 的跨域相似度做法同样建立在配对表征之上，不作为本系统结构路的直接依据。
- 赛题官方评测指标（RMSE 口径、标注预算定义）尚无公开文本，以官网规则包为准，答辩材料不写具体口径。
- 多人合作的 arXiv 条目统一以第一作者 + "等" 著录，避免作者缩写误写；arXiv:2604.01319（实为非厄米光子晶体）与 arXiv:2606.09430（LargeMonitor，作者已撤回）确认与本主题无关或不可用，不予列入。
- "同向大错 100%" 是 deep 场景（映射整体更换）下的极端值，shallow / domain 场景实测为 0%；不得笼统表述为"双模型总是同向错"。同理，"伪标注无效"的结论建立在合成流的三种漂移形态上，不构成对所有任务、所有漂移类型的普适断言。

## 路线图

- [x] 校赛最小闭环（合成数据验证）
- [x] 三大创新点实现与消融实验
- [ ] 渐变漂移末段 RMSE 进一步优化（目标接近基线 0.175，长期优化）
- [x] SwanLab 实验记录迁移到云端（项目看板已上线，见快速开始）
- [ ] 官方数据包接入（评测平台开放后）
- [ ] 评测平台 SDK 对接与系统镜像打包
- [ ] 答辩 PPT 与 Q&A 准备

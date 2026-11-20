# 系统运行说明文档

"智训引擎"赛题参赛作品 · 工业模型自进化系统。

本系统在漂移数据流上自动完成 **漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚** 五步闭环，
全过程无人工干预，并受重训练次数与标注配额约束。

---

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | 3.10 及以上（推荐 3.12） |
| 依赖 | `pip install -r requirements.txt` |
| 硬件 | 普通 CPU 即可（官方数据包提供预提取特征） |
| 网络 | 跑云端实验记录时需要能访问 SwanLab；离线评测可完全断网 |

依赖清单：numpy、scikit-learn、scipy、matplotlib、swanlab。

---

## 2. 三步上手：数据放哪 → 怎么跑 → 去哪看

不需要打开任何展示台，命令行就能完成全部工作。展示台/控制台只是"看板"，不参与推理。

### 第 1 步：数据包放哪

把官方数据包解压后放进（或直接指定）这个目录：

```text
data/official/
├── val.csv      ← 推荐：有表头，y/target/label 列为标签，找不到就用最后一列
├── val.npz      ← 或：键名 X / y（不区分大小写）
└── val.npy      ← 或：最后一列是标签，其余列是特征
```

三种格式**任选其一**即可，文件名必须是 `val`（也认 `validation` / `dev` / `test` / `train`），
扩展名 `.csv` / `.npz` / `.npy` 都行。目录换位置也可以，运行时用 `--data-dir` 指过去即可：

```bash
python scripts/run_pipeline.py --source official --data-dir D:/数据/官方包 --split val
```

放好后先跑体检，确认能被正确解析：

```bash
python scripts/check_dataset.py --data-dir data/official --split val
```

体检会报 PASS/WARN/FAIL，并直接打印下一步可复制的命令；有 FAIL 就先按提示处理。

### 第 2 步：怎么跑（不走展示台）

```bash
# ① 官方数据（最常用）：默认 MLP
python scripts/run_pipeline.py --source official --data-dir data/official --split val --no-swanlab

# ② 官方数据换模型（线性度检验后再选）
python scripts/run_pipeline.py --source official --data-dir data/official --model kalman --no-swanlab
python scripts/run_pipeline.py --source official --data-dir data/official --model hybrid --no-swanlab

# ③ 合成流复现（三模型 × 两场景 × 三种子）
python scripts/run_pipeline.py --drift sudden  --model mlp    --seed 42 --no-swanlab --experiment-name mlp-sudden-s42
python scripts/run_pipeline.py --drift gradual --model kalman --seed 42 --no-swanlab --experiment-name kal-gradual-s42

# ④ 改预算/批大小（轻量数据包务必调小 batch-size）
python scripts/run_pipeline.py --source official --batch-size 100 --label-budget 1500 --max-retrains 3 --no-swanlab
```

`--no-swanlab` = 只写本地、不连云端；需要云端留痕时去掉它（见第 5 节）。
样本量小时，`--batch-size` 按体检建议调小（例如 100）。

不想用命令行，也可以开本地控制台（只监听 127.0.0.1，参数白名单）：

```bash
python scripts/serve_console.py            # 浏览器打开 http://127.0.0.1:8900
python scripts/serve_console.py --port 9000
```

控制台里可选**预测模型**（MLP / 卡尔曼 A / 混合 B）、数据源、批大小、预算、槽位，点"启动实验"后实时看日志与曲线。

### 第 3 步：跑完去哪看结果

| 位置 | 内容 | 用途 |
|---|---|---|
| 终端输出 | 峰值 RMSE、末段 RMSE、重训次数、标注消耗、验收比值、结果文件路径 | 最快的一眼结论 |
| `runs/<实验名>-<时间戳>.csv` | 逐批明细：RMSE、MMD² z、KS p、PSI、tier、事件 | 自己画图 / 复核 |
| `runs/<实验名>-<时间戳>-summary.json` | 汇总指标、配置快照、SwanLab 链接 | 归档、对比 |
| `runs/report-<实验名>.png` | RMSE 曲线 + 漂移点/重训练标注 | 直接贴答辩 PPT |
| `docs/dashboard.html` | 三模型 × 两场景 × 三种子的总看板 | 答辩演示（双击即开，离线可用） |

重新生成总看板（数据来自 `runs/`，不重新训练）：

```bash
python scripts/build_dashboard.py --skip-probe
```

> 注意：看板只会挑 `runs/` 里符合命名规则的最新一份。跑完新实验后重新生成一次即可同步。


---

## 3. 配置项

配置优先级：**命令行参数 > 环境变量 > config.json > 内置默认值**。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `data.source` | `synthetic` | 数据源：`synthetic` 合成自测 / `official` 官方数据包 |
| `data.drift_type` | `sudden` | 合成数据的漂移形态：`sudden` 突变 / `gradual` 渐变 |
| `data.data_dir` | `data/official` | 官方数据包目录 |
| `data.split` | `val` | 官方数据包使用的划分名 |
| `data.target_col` | `y` | CSV 中标签列名 |
| `stream.batch_size` | `200` | 每批样本数 |
| `stream.init_batches` | `3` | 用前几批建立参考分布与初始模型 |
| `pipeline.window_size` | `600` | 缓冲窗口，重训练时使用的样本量 |
| `pipeline.model_type` | `mlp` | 预测模型：`mlp`=MLP 反应式（默认，通用）；`kalman`=恒定速度卡尔曼追踪线性映射 w（合成流渐变末段 1.44→0.14，−90%）；`hybrid`=MLP+卡尔曼残差校正（非线性主体+慢漂移残差）。选择：先跑 Ridge 拟合 y 的 R²，高→kalman，中→hybrid，低→mlp |
| `pipeline.residual_factor` | `3.0` | 残差报警阈值（相对基线的倍数） |
| `pipeline.max_retrains` | `6` | 重训练预算上限（含初始训练占用的 1 次） |
| `pipeline.label_budget` | `4000` | 标注配额上限 |
| `pipeline.clean_keep_ratio_covariate` | `0.7` | `incremental` 档旧样本保留比例 |
| `pipeline.clean_keep_ratio_concept` | `0.0` | `full` 档旧样本保留比例 |
| `pipeline.attribution_every` | `2` | SHAP 归因降频：每 N 次重训跑一次（`1`=每次都跑），只影响审计产物，不影响决策 |
| `pipeline.min_relative_gain` | `0.03` | 最小相对增益门：候选在留出集上相对旧模型改善低于该比例（默认 3%）则放弃上线，不消耗槽位/标注；`0`=关闭 |
| `pipeline.use_rate_window` | `true` | 创新点 5：漂移速率自适应取数窗口 |
| `pipeline.use_budget_pacing` | `true` | 创新点 4：槽位压力预算节流 |
| `pipeline.pacing_trigger` | `pressure` | 节流触发方式：`pressure` 仅剩 1 个槽位才限额 / `even` 全程按槽位均分 |
| `detector.mmd_z_threshold` | `3.0` | 结构路（MMD²）报警阈值：相对参考数据零分布的 z 分数，零分布由参考数据对半切分自校准 |
| `detector.ks_alpha` | `0.05` | KS 边际路显著性水平 |
| `detector.tier_bounds` | `[1.0, 3.0]` | 严重度分档边界（observe/incremental/full），可在公开集上反解 |
| `detector.psi_top_k` | `0` | 备用：>0 时严重度改用 top-k 敏感特征 PSI（高维特征场景才开） |
| `decision.mode` | `threshold` | v1.0 阈值分档 / `cost_sensitive` v2.0 代价敏感查表 |
| `decision.calib_path` | `runs/action_calibration.json` | v2.0 标定表路径 |
| `decision.cost_label` | `1e-4` | v2.0 每条标注的代价权重 |
| `logging.swanlab` | `true` | 是否连接 SwanLab 云端 |
| `logging.out_dir` | `runs` | 本地记录输出目录 |

环境变量：`ZHIXUN_DATA_SOURCE`、`ZHIXUN_DATA_DIR`、`ZHIXUN_RUNS_DIR`、`SWANLAB_API_KEY`、`SWANLAB_MODE`。
示例见 `.env.example`。

---

## 4. 官方数据包接入（三步走）

数据加载层（[src/dataset.py](../src/dataset.py)）与管线解耦，把官方数据包放进目录即可切换，管线代码零改动。

### 第 0 步：目录与格式约定

```bash
data/official/val.csv     # 或 val.npz / val.npy（三选一，自动识别）
```

支持的格式：

| 扩展名 | 约定 |
|---|---|
| `.npz` | 键名 `X` / `y`（不区分大小写）；无明确键时取前两个数组 |
| `.npy` | 末列视为标签，其余列为特征 |
| `.csv` | 表头中 `y`/`target`/`label` 列为标签，找不到则取最后一列 |

文件查找顺序：`val` → `validation` → `dev` → `test` → `train`，任一扩展名匹配即用。
说明：隐藏测试流由评测平台内部使用，本地不可见，也不需要接入本系统。

### 第 1 步：体检（放好数据先跑这个）

```bash
python scripts/check_dataset.py                        # 默认 data/official
python scripts/check_dataset.py --data-dir data/official --split val
```

输出 PASS/WARN/FAIL 三级报告：文件识别、格式解析、NaN/inf、常数列、标签统计、
样本量与 batch_size 建议；通过时会**直接给出下一步可复制的运行命令**。
有 FAIL 项退出码为 1，按提示处理后重跑。

### 第 2 步：跑管线

```bash
python scripts/run_pipeline.py --source official --data-dir data/official --no-swanlab
# 换预测模型（默认 mlp；先按第 4 步的线性度检验再决定）
python scripts/run_pipeline.py --source official --data-dir data/official --model kalman --no-swanlab
```

样本量小时用体检脚本建议的 `--batch-size`（例如 `--batch-size 100`）。

### 第 3 步（可选）：v2.0 动作反事实标定 + 代价敏感决策

```bash
python scripts/calibrate_actions.py --source official --data-dir data/official --batch-size 100
# 然后把 config.json 的 decision.mode 改为 cost_sensitive，再跑第 2 步
```

标定把三种动作（observe/incremental/full）各强制跑一遍官方流，生成
`runs/action_calibration.json`；标定表缺失或查询点离表太远时运行期自动回退
v1.0 阈值规则。详细边界与答辩口径见 [方案版本说明](方案版本说明.md)。

### 常见体检 FAIL 对照

| 报错 | 处理 |
|---|---|
| 目录不存在 / 空目录 | 数据包解压到 `data/official/`，或用 `--data-dir` 指定实际路径 |
| 没有找到可识别的数据文件 | 按 `<split>.npz/.npy/.csv` 重命名主文件 |
| X 中 NaN/inf | 先与官方确认口径；或自行清洗后重新放入 |
| 样本量偏紧 | 按提示调小 `--batch-size` / `--init-batches` |

---

## 5. 实验记录（赛题红线 2）

赛题要求全部实验在云端实验记录平台完整留痕。

```bash
# 1) 注册并登录 SwanLab（校赛阶段免费）
pip install swanlab
swanlab login

# 2) 或者用环境变量
set SWANLAB_API_KEY=你的key

# 3) 正常运行，日志自动同步云端
python scripts/run_pipeline.py
```

每次运行会输出：

- 云端实验链接（终端打印，并写入 `runs/<name>-*-summary.json` 的 `swanlab_url`）
- 本地 CSV：`runs/<name>-<时间戳>.csv`（逐批指标 + 事件）
- 汇总 JSON：`runs/<name>-<时间戳>-summary.json`

评测环境的离线运行：镜像内默认 `SWANLAB_MODE=disabled`，只写本地 CSV。

> **实验记录格式声明**：官方《实验记录规范》尚未下发。当前 CSV 表头为**自建替代格式**，
> 通过 `src/logger.py` 顶部的 `OFFICIAL_COLUMN_MAP` 做可配置映射（重命名 / 注入常量列 /
> 表达式派生列），规范下发后只改映射表，不改记录逻辑与历史产物。
> 映射机制本身由 `tests/test_logger.py` 锁定。

### 5.1 真实数据集 run 的云端链接

三个真实数据脚本（`run_real_cmapss.py` / `run_real_gas.py` / `run_real_bearing.py`）已改为
读 `config.logging.swanlab`（默认 `true`），登录后自动建云端 run；未登录时 logger 自动降级
本地 CSV，不中断实验。下表是**开通云端记录后重跑**的产物，实验参数 / 预算 / 切分均未改动，
逐批指标与改动前的 `runs/*.csv` 逐字节一致（核对记录见 `real_data/最终结果表.md` §10.4）。

项目看板：<https://swanlab.cn/@Simplify-ai-create/zhixun-engine>

| 数据集 | run 名 | 云端链接 |
|---|---|---|
| C-MAPSS FD001 | `real-fd001`（`or` / `and` / `perf_only`） | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/czeq9r54> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/vws8392c> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/dcr08gtn> |
| C-MAPSS FD002 | `real-fd002`（`or` / `and` / `perf_only`） | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/dg14rn2k> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/435n3oyg> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/wvkcmug0> |
| C-MAPSS FD003 | `real-fd003`（`or` / `and` / `perf_only`） | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/eq5yjjf3> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/0lo7hh3p> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/hsw1qy6k> |
| C-MAPSS FD004 | `real-fd004`（`or` / `and` / `perf_only`） | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/1z5miz80> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/8zrdinc9> / <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/kw5ead8o> |
| UCI 气体 | `real-gas-uci-or` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/mdounchm> |
| XJTU-SY 2_1 | `real-bearing-2_1` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/n5985f32> |
| XJTU-SY 2_3 | `real-bearing-2_3` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/gahe0c9p> |
| XJTU-SY 3_1 | `real-bearing-3_1` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/vxytmiz4> |
| XJTU-SY 3_2 | `real-bearing-3_2` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/2p7kmthm> |
| XJTU-SY 3_3 | `real-bearing-3_3` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/d4eennqp> |
| XJTU-SY 3_4 | `real-bearing-3_4` | <https://swanlab.cn/@Simplify-ai-create/zhixun-engine/runs/h4nnk9jk> |

复跑命令（与上表一一对应，参数须原样保留）：

```bash
py -3.12 scripts/run_real_cmapss.py --subset FD001   # FD002 / FD003 / FD004 同理
py -3.12 scripts/run_real_gas.py --fuse or
py -3.12 scripts/run_real_bearing.py --bearings 2_1,2_3,3_1,3_2,3_3,3_4 \
    --batch-files 50 --init-batches 3 --period 8 --label rul_ratio
```

每个 run 的 URL 会写进同目录 `runs/<run 名>-<时间戳>-summary.json` 的 `swanlab_url` 字段，
可离线核对，不需要打开浏览器。

---

## 6. 镜像构建与离线评测

```bash
# 构建
docker build -t zhixun-engine:latest .

# 本地自测（合成数据）
docker run --rm zhixun-engine:latest --source synthetic --no-swanlab

# 官方数据包评测（数据目录挂载到 /data，结果落到 ./runs）
docker run --rm -v /绝对路径/官方数据:/data -v /绝对路径/runs:/app/runs \
  zhixun-engine:latest
```

镜像默认入口：`python scripts/run_pipeline.py --source official --data-dir /data --no-swanlab`。
系统在该配置下无人工干预地跑完全部阶段并输出结果。

**完整交付约定（接口 / 输出契约 / 超时与内存假设 / 外呼禁令）见
[交付运行说明](DEPLOY_RUNBOOK.md)。镜像构建与验证全部在 CI 的 `docker-image` job 内完成
（仓库本地不安装 Docker Desktop）。

---

## 7. 产物说明

| 文件 | 内容 |
|---|---|
| `runs/*.csv` | 逐批记录：RMSE、MMD² 与 z 分数、KS p 值（Bonferroni 校正后）、PSI、报警计数、重训练次数、标注消耗 |
| `runs/predictions-<实验名>-<时间戳>.csv` | **逐批预测结果**：`batch_step,sample_index,y_true,y_pred`，每行一个被评估样本（默认写出，`--no-pred-dump` 关闭） |
| `runs/*-summary.json` | 运行摘要 + 云端链接 |
| `runs/report-*.png` | RMSE 曲线 + 真实漂移点标注（答辩用） |
| `runs/SUMMARY.md` | 一键运行的三组实验汇总 |
| `docs/dashboard.html` | 三模型 × 两场景 × 三种子总看板（双击即开） |
| `docs/console.html` | 本地控制台页面（需 `serve_console.py` 提供 API） |

---

## 8. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `SwanLab 不可用，降级为本地 CSV` | 未登录或未设置 `SWANLAB_API_KEY`；不影响实验，但需补跑上传以满足红线 2 |
| `官方数据目录不存在` | 用 `--data-dir` 指定实际解压路径 |
| `下有多个数据文件，无法自动选择` | 用 `--split` 指定要用的划分名 |
| `数据形状不一致` | 检查标签列是否被错误地当成特征列 |
| 没有生成图表 | 未安装 matplotlib，`pip install matplotlib` |

---

## 9. 复现清单（对应评分点）

- [x] 五步闭环全程无人工干预，按时输出结果（红线 1）
- [x] 全部实验写入 SwanLab 云端 + 本地 CSV 双留痕（红线 2）
- [x] 综合性能跑赢静态基线（红线 3）
- [x] 消融实验证明各创新点有效（进阶加分）
- [x] 官方数据包可插拔接入，评测环境离线可运行

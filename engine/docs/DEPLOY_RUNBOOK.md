# 交付运行说明（离线镜像 / 复现口径）

> 面向比赛评测方：本文件只描述**怎么跑、吃什么、吐什么**，不依赖任何外部服务。
> 主运行手册见 [RUNBOOK.md](RUNBOOK.md)。

## 1. 镜像入口

```text
基础镜像   python:3.11-slim + libgomp1
工作目录   /app
入口       python scripts/run_pipeline.py
默认参数   --source official --data-dir /data --no-swanlab
```

自定义入口参数（追加在镜像名之后即可，全部为白名单参数）：

```bash
docker run --rm -v <数据目录>:/data -v <产物目录>:/app/runs zhixun-engine:latest \
  --source official --data-dir /data --split val --batch-size 200 --no-swanlab
```

构建与验证**全部在 CI**（`.github/workflows/ci.yml` 的 `docker-image` job）执行，
包括：构建镜像 → 离线跑通 → 校验预测文件契约 → 校验外呼开关不存在。
仓库本地不安装 Docker Desktop。

## 2. 数据挂载假设

| 项 | 约定 |
|---|---|
| 挂载点 | 数据只读挂载到 `/data`，产物目录挂载到 `/app/runs` |
| 主文件 | `/data/val.csv` 或 `val.npz` 或 `val.npy`（三选一） |
| 备用名 | `validation` / `dev` / `test` / `train`（任一命中即用，可用 `--split` 指定） |
| CSV | 有表头；`y` / `target` / `label` 列为标签，找不到则取最后一列 |
| NPZ | 键名 `X` / `y`（不区分大小写） |
| NPY | 末列为标签，其余为特征 |
| 网络 | **不需要**。镜像内 `SWANLAB_MODE=disabled`、`ZHIXUN_OFFLINE=1` |

放好数据后可先跑体检（PASS/WARN/FAIL，含 batch_size 建议）：

```bash
docker run --rm -v <数据目录>:/data --entrypoint python zhixun-engine:latest \
  scripts/check_dataset.py --data-dir /data --split val
```

## 3. 逐批调用接口

系统对外只有一个契约：**数据流适配器**（`src/dataset.py` / `src/real_stream.py`）。
任何数据源只要提供下列三个成员，即可零改动接入主循环：

```python
stream.name          -> str                       数据流名称，写入实验记录
stream.drift_points  -> list[int]                 已知漂移点（未知时返回 []）
stream.batches(bs)   -> Iterator[(offset, X, y)]  按批产出 (批起点, 特征矩阵, 标签)
```

每批在主循环内依次执行：**检测（MMD² + KS 双路）→ 决策（严重度分档 / 策略控制器）
→ 重训练（清洗 + 留出验收 + 静态基线锚）→ 评估 → 落盘**，无人工干预、无外部调用。

## 4. 输出位置与字段契约

| 路径 | 内容 | 是否必须 |
|---|---|---|
| `/app/runs/predictions-<实验名>-<时间戳>.csv` | **逐批预测结果**，见下表字段 | **必须**（`--no-pred-dump` 可关，评测请勿关） |
| `/app/runs/<实验名>-<时间戳>.csv` | 逐批指标与事件（RMSE、MMD² z、KS p、PSI、tier、重训/预算） | 必须 |
| `/app/runs/<实验名>-<时间戳>-summary.json` | 运行摘要 + 配置快照 | 必须 |
| `/app/runs/report-<实验名>.png` | RMSE 曲线 + 漂移/重训标注 | 可选（装 matplotlib 时生成） |

预测文件字段（固定顺序，UTF-8，逗号分隔，首行表头）：

| 列 | 含义 |
|---|---|
| `batch_step` | 第几批，0 起 |
| `sample_index` | 流内全局样本序号，0 起（= 批起点 + 批内偏移） |
| `y_true` | 真值；回归为数值（6 位小数），分类为类别名 |
| `y_pred` | 模型在该批的预测值（同一批内为同一模型的输出） |

## 5. 超时与内存假设

本机实测（CPU，单核，`python:3.11-slim` 同构环境；合成流 27 批 / 200 样本/批 / 3 次重训）：

| 指标 | 实测值 | 评测建议上限 |
|---|---|---|
| 端到端 wall time | 24.9 s（含解释器启动、SHAP 归因、PNG 绘图） | 单次运行 30 min |
| 进程峰值内存 | 211.5 MB | 2 GB |
| 磁盘产物 | 每万样本级预测 CSV < 1 MB | 1 GB |

说明：上述为**本机实测**，不含评测机差异；官方数据包若样本量级更大，耗时随批数线性增长，
建议按 `批数 × 1 s` 估算并预留 3 倍余量。内存瓶颈是历史缓冲（`hist_max`，默认 5000 条）
与清洗时的协方差矩阵：`max_mahalanobis_dim` 只决定降维阈值，特征维数超 100 时先 PCA 降到
≤32 维再算马氏距离，故实际协方差矩阵不超过 32×32。

## 6. 复现命令

```bash
# 合成流（离线自检，不需要任何数据挂载）
docker run --rm zhixun-engine:latest --source synthetic --drift sudden --no-swanlab

# 官方数据包
docker run --rm -v /abs/data:/data -v /abs/runs:/app/runs zhixun-engine:latest

# 自定义预算 / 批大小
docker run --rm -v /abs/data:/data -v /abs/runs:/app/runs zhixun-engine:latest \
  --batch-size 100 --label-budget 1500 --max-retrains 3 --no-swanlab
```

宿主机直接复现（与镜像等价）：

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --source synthetic --drift sudden --no-swanlab
```

## 7. 离线与外呼禁令（红线）

- 镜像内固定 `ZHIXUN_OFFLINE=1`，`ZHIXUN_LLM_BASE_URL` / `ZHIXUN_LLM_API_KEY` / `ZHIXUN_LLM_MODEL` 全部置空。
- `--strategy llm` 在镜像内**不可触发**：`src/strategy.py::build_controller` 在
  `ZHIXUN_OFFLINE=1` 时根本不会构造 `LLMController`，直接降级为规则控制器并打印
  `[offline] strategy=llm disabled ...`。
- `--no-swanlab` 为镜像默认参数，且 `SWANLAB_MODE=disabled`，云端实验记录开关在镜像内不可启用。
- CI 的 `docker-image` job 会显式断言上述离线降级提示存在，防止后续改动把外呼路径放回来。

## 8. 实验记录规范声明

官方《实验记录规范》**尚未下发**。当前 CSV 为**自建替代格式**，通过
`src/logger.py` 顶部 `OFFICIAL_COLUMN_MAP` 做可配置映射（重命名 / 注入常量列 / 表达式派生列）。
规范下发后只改映射表，不动记录逻辑，也不重写历史产物。
映射行为由 `tests/test_logger.py` 锁定。

## 9. 无法在本地验证的项（如实标注）

| 项 | 状态 |
|---|---|
| 镜像在评测机上的实际构建 | **未验证**（本地按要求不装 Docker Desktop，交由 CI 执行） |
| 评测机 CPU/内存差异对耗时的影响 | **未验证**，仅给出本机实测与线性外推 |
| 官方隐藏测试流的实际分布 | **无法验证**（评测平台内部使用，本地不可见） |

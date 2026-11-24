# zhixun-contest-source

> 源码与镜像复现说明

> 本文所有哈希、tag、文件数均为 2026-11-24 在本机实测，出处见 §0。
>
> **状态**：本仓库 `main` 的最新提交为本次文档更新（仅 `README.md`，不改变 `submission/app/`）；其上游 `74ba580` 对应 Release [`v7-run5`](https://github.com/SImplify-ai-create/zhixun-contest-source/releases/tag/v7-run5)，资产 sha256 与本文一致。

## 仓库结构

本仓库包含两个顶层目录：

### `submission/`

实际提交镜像所用的源码。

- 镜像名：`zhixun-harness:v7-run5`（取自归档 `manifest.json` 的 `RepoTags`）
- 镜像归档 `zhixun-harness-v7-run5.tar` 的 sha256：`9fc23370f17c422ee08cb9b9edcfc34af701b7bdbca5f32c33de5e38139ce214`
- 镜像内 `/app` 与 `submission/app/` 共 20 个文件，内容一致：归一化 CRLF/LF 后 0 个内容差异（7 个字节完全相同、13 个仅换行符不同）
- 目录内容：`app/`、`tests/`、`manifest.json`、`P0_FREEZE.json`、`P0设计说明.md`、`completeness_audit.md`、`赛方问询清单.md`。

### `engine/`

方法学与实验引擎的参考实现，**不是镜像内容**。

- 目录内容：`src/`、`docs/`、`tests/`、`config.json`、`README.md`、`THIRD_PARTY_NOTICES.md`、`requirements.txt`。
- 不含原始数据、运行记录、模型权重与私有配置；实验记录见公开实验记录页。

## 0. 本文数字的实测出处

| 数字 | 值 | 从哪来 |
|---|---|---|
| 归档来源 | Release `v7-run5` 资产 `zhixun-harness-v7-run5.tar` | `https://github.com/SImplify-ai-create/zhixun-contest-source/releases/tag/v7-run5` |
| 镜像 tag | `zhixun-harness:v7-run5` | `tar -xOf zhixun-harness-v7-run5.tar manifest.json` 的 `RepoTags` |
| tar 整份 sha256 | `9fc23370f17c422ee08cb9b9edcfc34af701b7bdbca5f32c33de5e38139ce214` | `Get-FileHash zhixun-harness-v7-run5.tar -Algorithm SHA256`（与 Release 资产摘要一致，回下载复算） |
| tar 体积 | 58,777,600 字节 | `(Get-Item zhixun-harness-v7-run5.tar).Length` |
| 镜像 ID（config digest） | `sha256:2d9ac534e52c601e085f9c303591c294a615ee848d151bc7942e9d97599b8411` | `blobs/sha256/2d9ac534…`，且其文件名与自身 sha256 相符 |
| 镜像 manifest digest | `sha256:dc34b3b1c439aab6a0fe8e22d8511dba91fa97ef3019f89ecf15327f93dafa05` | `index.json` 的 `manifests[0].digest`；本机 Docker 29（containerd 镜像存储）`docker image inspect .Id` 显示此值 |
| 主层 digest | `sha256:43e3a5e50a2e5146058040bc9f07906a0183f29bf9e97cde8c20551c2a72f29a`（58,763,657 B） | `manifest.json` 的 `Layers[0]`，文件名与自身 sha256 相符 |
| 空层 digest | `sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1`（32 B） | `Layers[1]`，文件名与自身 sha256 相符 |
| `/app` 文件数 | 20（19 个 `.py` + `preregistration.json`） | 解包主层后遍历 `/app` 统计文件数 |
| 仓库 `submission/app/` | 20 个文件，与镜像 `/app` 内容一致：7 个字节完全相同、13 个仅 CRLF/LF 不同、0 个内容差异（镜像内为 CRLF、仓库 blob 为 LF；本仓 `core.autocrlf=true` 检出后为 CRLF；只计 19 个 `.py` 时为 7 个相同、12 个仅换行差异） | `git archive` 取 HEAD 后逐文件 sha256，再做 CRLF→LF 归一化比对 |
| 基础系统 | Debian GNU/Linux 13 (trixie)，Python 3.11（`/usr/local/bin/python3.11`） | 主层 `usr/lib/os-release`、`usr/local/bin/` |
| 架构 | `linux/amd64` | config 的 `architecture` / `os` |

## 1. 镜像是什么

一个 OCI 布局的镜像归档（`index.json` + `oci-layout` + `blobs/sha256/`），两个层：主层是完整 rootfs（Debian 13 + Python 3.11 + `/app`），第二层是 32 字节空层（构建元数据层，不含内容）。

镜像元数据（取自 config）：

| 项 | 值 |
|---|---|
| `Entrypoint` | `["python", "/app/harness.py"]` |
| `Cmd` | `["describe"]` |
| `WorkingDir` | `/app` |
| `Env` | `PYTHONPATH=/app`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONUNBUFFERED=1` |
| 构建时间 | `2026-11-24T04:19:43Z` |
| 构建方式 | `ADD run5-rootfs.tar /` + `ENV` + `WORKDIR` + `ENTRYPOINT` + `CMD`（见 config 的 `history`），buildx `linux/amd64` |

运行时只依赖 Python 标准库与镜像内已装的 **numpy 2.4.6**（路径 `usr/local/lib/python3.11/site-packages/numpy/version.py`）；检测器、插值、MLP 训练全部用 numpy 实现，不引入其它第三方包。

## 2. `/app` 的 20 个文件与用途

用途取自各文件首行 docstring，未推测。

| 文件 | 字节 | 用途 |
|---|---:|---|
| `harness.py` | 23,075 | 固定入口。三个子命令 `describe` / `train` / `predict`；`train` 从 stdin 读 SDK 的 start 帧 |
| `contest_submission.py` | 42,408 | 选手模块（对应 `/app/contest_submission.py`）。`describe`、`predict`、官方 MLP 推理 `_infer_official` |
| `train_round.py` | 161,090 | 一轮 = 一次 `train(rnd)` 的主流程：固定顺序，任一步失败都不提交半成品；写 `configuration` 账本 |
| `train_mlp.py` | 78,629 | 在官方 CPU MLP 结构上做纯 numpy 梯度续训；含守卫、`rel_l2`、预测位移等闸门 |
| `train_stage.py` | 15,742 | `train` 阶段入口。**唯一**运行批级漂移检测的地方；读 `/data/train.jsonl` 与 `/data/validation.jsonl` |
| `selector.py` | 7,287 | 选臂。吃 train 侧批级检测报告，决定本轮造哪个候选；产出 `selector_trace` |
| `actions.py` | 7,188 | 动作库与阶段门禁（续训 / 插值 / 校准 / 回退） |
| `bias_family.py` | 37,752 | 输出侧校准候选族（只改 a、c，不碰梯度）；含 `a` 的端点截断与 `a_before_clamp` 留痕 |
| `detectors.py` | 5,653 | 批级漂移检测器（KL / 马氏 / BBSD），仅 `train` 阶段可运行 |
| `detector_lib/registry.py` | 6,632 | 19 条检测器注册表（`enabled` / `conditional` / `alarm_only` 三态） |
| `detector_lib/kernels.py` | 13,308 | 19 条检测器的 numpy 实现；拿不到的层显式标 `unavailable`，不造数 |
| `detector_lib/__init__.py` | 14,929 | 检测器库运行时（三态 + 标定门禁） |
| `llm_cage.py` | 46,234 | 提案笼子：候选性质测试、数字白名单、账本行引用校验、权重漂移上限 |
| `failure_modes.py` | 3,815 | 把失败签名拆成 `text`（无数字，喂 prompt）与 `evidence`（带数值，落账本） |
| `fact_probe.py` | 19,619 | 只写 stderr 的事实探针；stdout 零字节污染，异常自吞 |
| `v6alpha/alpha_lib.py` | 8,588 | 权重空间插值（α / soup 的公共纯函数） |
| `v6alpha/l2sp.py` | 5,536 | L2-SP 软近端（惩罚项、梯度项、λ 标定协议） |
| `v6alpha/grid_serialize.py` | 4,803 | `alpha_grid` 的 configuration 序列化与四级砍配 |
| `v6alpha/__init__.py` | 386 | 包容器，使 `from v6alpha import ...` 在扁平 `app` 布局下可用 |
| `preregistration.json` | 29,370 | 预登记（`arm-takeover-2026-11-22.v3` = active，`C4-calibration-2026-11-22.v3` = pending_activation） |

## 3. 还原镜像并校验（不需要我们的协助）

```bash
# 0) 从 Release 下载归档（仓库不入库 tar；资产名 zhixun-harness-v7-run5.tar）
# https://github.com/SImplify-ai-create/zhixun-contest-source/releases/tag/v7-run5

# 1) 校验整份归档（Linux/macOS）
sha256sum zhixun-harness-v7-run5.tar
# 期望：9fc23370f17c422ee08cb9b9edcfc34af701b7bdbca5f32c33de5e38139ce214

# Windows PowerShell
# (Get-FileHash .\zhixun-harness-v7-run5.tar -Algorithm SHA256).Hash.ToLower()

# 2) 导入
docker load -i zhixun-harness-v7-run5.tar
# 期望输出包含：Loaded image: zhixun-harness:v7-run5

# 3) 校验镜像本身
docker image inspect zhixun-harness:v7-run5 --format '{{.Id}}'
# 期望：sha256:dc34b3b1c439aab6a0fe8e22d8511dba91fa97ef3019f89ecf15327f93dafa05（manifest digest；
# 本机 Docker 29 containerd 镜像存储显示此值）。经典 graphdriver 会显示 config digest
# sha256:2d9ac534e52c601e085f9c303591c294a615ee848d151bc7942e9d97599b8411。两者均见 §0。

docker image inspect zhixun-harness:v7-run5 --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
# 期望：["python" "/app/harness.py"] ["describe"]

# 4) 校验文件清单（应输出 20 行）
docker run --rm --entrypoint sh zhixun-harness:v7-run5 -c 'find /app -type f | sort'

# 5) 跑入口自检（离线，不需要数据）
docker run --rm zhixun-harness:v7-run5
```

> 若 `docker load` 报不认识该格式：本归档是 OCI 布局（`index.json` + `oci-layout` + `blobs/sha256/`），Docker 20.10 及以上可直接 `load`；更低版本或非 Docker 运行时，请用 `skopeo copy oci-archive:zhixun-harness-v7-run5.tar docker-archive:zhixun-harness-v7-run5.tar` 转换后再导入。
>
> 镜像为 `linux/amd64`。在 arm64 机器上需 `--platform linux/amd64` 并启用模拟。

## 4. 入口与运行方式

**入口**：`python /app/harness.py {describe|train|predict}`（`Entrypoint` + `Cmd` 已把默认命令设为 `describe`，所以 `docker run --rm <tag>` 等价于 `describe`）。

| 命令 | 需要什么 | 能做什么 |
|---|---|---|
| `describe` | 无 | 返回选手模块的元信息，校验 `train` / `predict` / `PROTOCOL` 是否齐备。**可离线跑**，评审可用它确认镜像完整 |
| `train` | 平台 SDK（`contest_harness`）从 stdin 灌入 start 帧，宿主传入 `rnd`；`/data/train.jsonl`、`/data/validation.jsonl` | 跑完整一轮：检测 → 选型 → 训练/插值/校准 → 过闸门 → 提交。**镜像外没有 SDK 时按设计以 exit 3 退出**，并打印说明，不会假装成功 |
| `predict` | `/model`（或环境变量 `CONTEST_MODEL_DIR`）+ 预测上下文 | 加载指定检查点做因果推理 |

**数据路径**（写死在代码里，不读环境变量）：`/data/train.jsonl`、`/data/validation.jsonl`（`train_stage.py`、`train_round.py`）。宿主会通过 `rnd.task["public_projection"]` 投影为实际路径。

**运行时环境**：无外网；LLM 调用走平台宿主网关（`model_access: host_gateway_rpc`），不在镜像内配置任何 API Key；本地按 4 核自限。

因此一句话说清可复现边界：**评审可以完整复现"镜像能载入、sha256 对得上、`describe` 通过、20 个源文件与仓库一致"；但完整跑一轮 `train` 需要平台宿主，无法脱离平台在本地独立完成。**

## 5. 为什么没有 Dockerfile

镜像不是从 `submission/` 源码 `docker build` 出来的，而是把一份 rootfs 快照 `ADD` 进基础镜像（config 的 `history` 第一条即 `ADD run5-rootfs.tar /`）。仓库里只有 `submission/app/` 的源码，**没有那份 rootfs 快照，也没有基础镜像清单**，所以补一个 Dockerfile 只能造出一个"看起来一样、实际不等价"的镜像——反而会让评审以为可以逐字节复现。

因此本仓库给出的复现路径是**导入归档**（§3），而不是重建。若后续改为可重建的构建方式，会在本文件同步替换 §3 与 §7。

## 6. 来源与许可

- 训练数据只用平台下发的 `/data`；工业模型链路不含任何外部数据集。逐数据集（17 项）的许可原文、布尔、落地与实际用途，见随作品文档提交的《来源声明》§3–§5。
- Sampler 微调语料 126 行中，有 10 行来自 `shibing624/alpaca-zh`（CC BY 4.0），用于防塌缩；署名与使用量见《来源声明》§7.1，保留理由见 §7.2。
- 本仓库 `engine/` 是方法学参考实现，**不是镜像内容**，不含数据、权重与私有配置。

## 7. 本版实测值（v7-run5）与重建后的更新点

本文件写的是 v7-run5 的实测值。以下 5 类数字与 tag 在每次重建镜像后都会变（本次 tar 比 v7-run4 多 4,608 字节，主层多 1,897 字节），重建镜像后需同步更新本节数值，否则本文会再次变成过期文档：

| 项 | v7-run5 实测值 | 出处 |
|---|---|---|
| 镜像 tag | `zhixun-harness:v7-run5` | `manifest.json` 的 `RepoTags` |
| tar sha256 与字节数 | `9fc23370f17c422ee08cb9b9edcfc34af701b7bdbca5f32c33de5e38139ce214`，58,777,600 B | Release 资产（回下载复算一致）/ `Get-FileHash` |
| 镜像 ID（config digest） | `sha256:2d9ac534e52c601e085f9c303591c294a615ee848d151bc7942e9d97599b8411` | `blobs/sha256/2d9ac534…`，文件名与自身 sha256 相符 |
| manifest digest（Docker 29 `.Id`） | `sha256:dc34b3b1c439aab6a0fe8e22d8511dba91fa97ef3019f89ecf15327f93dafa05` | `index.json` 的 `manifests[0].digest` |
| 主层 digest 与体积 | `sha256:43e3a5e50a2e5146058040bc9f07906a0183f29bf9e97cde8c20551c2a72f29a`，58,763,657 B | `manifest.json` 的 `Layers[0]`，文件名与自身 sha256 相符 |
| `/app` 文件数 | 20（19 个 `.py` + `preregistration.json`） | 解包主层后 `find /app -type f` 计数 |
| 构建环境 / 时间 | Debian 13 + Python 3.11 + numpy 2.4.6；buildx `linux/amd64`，`ADD run5-rootfs.tar /`；`2026-11-24T04:19:43Z` | config 的 `history` / `created` |

判据：本仓库 `submission/app/` 树哈希与镜像 `/app` 一致，且 Release 附件 sha256 与本文相同。

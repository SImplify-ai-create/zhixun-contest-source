# v1/v2 完整度对账（只读）

- **核查时间**：2026-11-17
- **核查方式**：只读。未修改任何 v1/v2 文件，未补实现；本文件是唯一新增产物。
- **依据材料**：`contest-v2-freeze/outputs/FREEZE.md`、`contest-v2-freeze/outputs/archive_manifest.json`（794 条记录）、`2026-11-15` 与 `2026-11-16` 各目录的设计说明/报告/`experience_log.jsonl`、`contest-v2/P0设计说明.md`、`contest-v2/manifest.json`、`contest-v2/app/*.py`、`contest-v2/tests/*.py`。
- **三态定义**：`已落地且可验证`（代码或产物在盘上，且有可重跑的核验方式）／`只有文档没有代码`（只写到说明/报告/清单里）／`确实缺失`（文档和代码都没有，或只有外部问询）。

---

## 0. 结论先行

1. **数据预处理不是"没了"，是被大幅收窄**：v1 的预处理是动作注册表（`SIG_ACTION_MAP` 5 类签名 → 5 个动作），v2 只迁了 `identity` / `streaming_ema_norm` / `per_cond_seq` 三个因果动作。经验库 36 条里 **23 条**采用的动作 `per_cond_mono_norm`，在 v2 被判定为 transductive，**predict 侧显式拒绝**。这是"感觉预处理消失"的最大真实来源。
2. **图像预处理从未在本赛题落地过**：v1 冻结的 794 个文件里，"图像/图片/image preprocess/PIL/torchvision"零命中（唯一命中是 docker `image id` 和一个临时目录名）。全仓 `*.md` 里"图像"只出现在 3 份外部产品/故事生成项目的说明中，与本赛题 A/B/C 三条数据流无关。**不属于"冻结时被排除"，是从未写下过。**
3. **FREEZE 的 10 条未完成项**：5 条已迁入 v2；2 条在口径改写后已迁入；1 条属过期作废（分段器）；**2 条仍然悬空**（`dataset_key` 与 `inputs` 结构错配、四条平台待定项），另有 2 条含"真机未验"的尾巴。
4. **FREEZE 里有两处 v2 描述已过期**：#2 写的"v2 采用 fail-fast"、#5 写的"v2 直接判协议错误"。现行实现是"总输出 + 声明默认"和"上下文 > 行 > 产物"。FREEZE 是冻结快照，未回写，以 `P0设计说明.md` 为准。
5. **v2 三轮承诺全部有代码和测试**：P0 25 项 + P1 15 项 + P2 17 项 = 57 项通过；28 个可移植文件的 sha256 全部通过 `verify_upstream.py`。
6. **真正缺的东西只有四类**：B 流 floor（提交侧 536.27；留一 522.97 仅供参照；oracle 型均值 278.06 不可提交）没有进 v2 代码、检测器 12 模块只重写了 3 个、`dataset_key` 结构对账没有实现、平台侧 12 条问询与试运行回执仍然为空。

---

## 1. 专项：数据预处理与图像预处理

### 1.1 两条链路的结论

| 链路 | v1 是否存在 | 位置 | 现状（v2） | 三态 | 若缺失，最后可见位置 |
|---|---|---|---|---|---|
| 数据预处理（特征归一化/动作） | **存在，且是 C 流主线** | `2026-11-15/signature-routing/outputs/routing_contract.py`（`SIG_ACTION_MAP` L139–145、`_normalize` L197、`_cond_norm` L304、`_robust_norm` L256、`_causal_monotone` L367）；`2026-11-15/official-alignment/run_c_state.py::normalize` L117（`cond` / `unit` / `cond+unit` 三种口径）；`platform-s0/hybrid/hybrid_entrypoint.py`（L2）；`platform-s0/z0/baselines/data.py`（装载与禁用字段） | v2 重写为 5 个动作：`identity` / `streaming_ema_norm` / `per_cond_seq`（causal，predict 可用）+ `per_cond_mono_norm` / `batch_zscore`（transductive，predict 显式拒绝） | **部分落地**（3/8 动作进 predict；2 个进 train 侧但被拒于 predict；3 个未迁） | 见 1.2 逐行动作表 |
| 图像预处理 | **不存在** | 无 | 无 | **确实缺失**（且从未写过） | 无。全仓唯一"图像"出现在 `f344886cb76ffb31e172f8360026e32e_..._m.md`（AI 故事生成/语音合成项目 README）及其两份 bohr 导入副本；与本赛题无关 |

### 1.2 数据预处理逐动作去向

| v1 动作 / 口径 | v1 依据 | v2 对应 | v2 状态 | 三态 |
|---|---|---|---|---|
| `no_op_report_gap`（分类：报告 gap、不动作） | `routing_contract.py` L140；经验库 6 条 | 无 | 未迁移 | **确实缺失** |
| `rebuild_linear_ratio`（LDA + L2 重标定；经验库唯一 A 流 adopted，macro-F1 0.7649 vs 基线 0.7573） | `routing_contract.py` L140、L206；`experience_log.jsonl` `run-20261115-l2-lda` | 无 | 未迁移 | **确实缺失** |
| `per_cond_mono_norm`（读 train 工况统计表） | `routing_contract.py` L141–142；经验库 **23/36** 条采用 | `actions.per_cond_mono_norm`，标记 `TRANSDUCTIVE` | 已注册、已实现，但 predict 侧 `check_stage` 直接拒绝；selector 也不拿它当候选 | **只有代码没有入口**（最容易被误认为"没了"） |
| `per_unit_mono_norm`（单元自身整条轨迹） | `routing_contract.py` L142；`run_c_state.py::_normalize_own` mode=`unit` | 无同名动作；`batch_zscore` 是整批统计，不等价 | 未迁移 | **确实缺失** |
| `constant_life_prior`（B 流常数寿命先验） | `routing_contract.py` L143；经验库 6 条；`b-floor/floor_report.md`（提交侧 536.27；留一 522.97；oracle 278.06 不可提交） | 无 | 未迁移；v2 只保留"回归 clip ≥0" | **确实缺失** |
| `report_and_no_op`（B 流不动作） | `routing_contract.py` L144 | `identity` | 等价迁移 | 已落地 |
| 在线归一化（无状态 EMA） | v1 无此口径（v1 是批级/单元级） | `actions.streaming_ema_norm` | v2 新增，predict 可用 | 已落地（v2 新增项） |
| 分工况顺序归一（因果版） | v1 的 `cond` 统计量只用 train 估计（`run_c_state.py` L117–139） | `actions.per_cond_seq`（逐条件 Welford 子状态） | 已迁移为因果形态 | 已落地 |
| L2 归一（混合流入口） | `platform-s0/hybrid/hybrid_entrypoint.py` | 无 | 混合流已废弃（FREEZE 三节），不迁 | 属过期作废 |
| `StandardScaler`（多任务可行性实验） | `2026-11-16/multitask-feasibility/_work/routing_experiment.py` | 无 | 实验脚本，未产品化 | 只有文档没有代码（实验性质） |

### 1.3 "图像预处理"的三问三答

- **在 v1 到底存不存在？** 不存在。对 `2026-11-15` 与 `2026-11-16` 全目录按 `图像|图片|\bimage\b|preprocess|PIL|torchvision` 检索，命中 6 行，全部是 `docker image id`（`Z0_本地闭环.md` L144）、正则变量名（`routing_contract.py` L738）、以及 `out_host/evidence_crashed*` 里的临时目录名（`pef-framework-image`）。没有任何一条是图像数据读取或图像变换。
- **存在于哪个目录哪个文件？** 不存在。v2 `app/` 5 个模块、`tests/` 4 个测试、`manifest.json`、`P0设计说明.md` 同样零命中；`manifest.json` 里唯一的 "image" 是 `task_declaration.image_reports` 这个键名（本意是"镜像上报的 tasks"），与图像无关。
- **从哪一步开始没有的？** 不属于"曾写下后废弃"，也不属于"冻结时被排除在 portable 清单之外"——它**从未进入本赛题链路**。三条数据流全部是表格/序列（A：UCI 气体传感器；B/C：C-MAPSS 轴承），平台 SDK 的 `inputs` 也只收数值特征。仓内与"图像/多模态"有关的文字只出现在一份外部产品说明（AI 故事生成 + 语音合成，含配图/图生视频）和另一赛题的图纸解析方案里，属于材料混入，不是本项目的实现缺口。
- **为什么会有"被删了"的印象？** 两个可解释来源：① v1 的预处理集中在 C 流（`per_cond_*`），v2 默认产物（R1 续训父检查点）动作是 `identity`，看起来"预处理整段消失"；② v2 manifest 的 `image_reports` 键名会让人误以为这里原本有图像相关声明。

---

## 2. FREEZE.md 10 条未完成项的 v2 归属

| # | 未完成项（FREEZE 原文要点） | 依据 | v2 归属 | 现状与证据 | 若悬空，最后可见位置 |
|---|---|---|---|---|---|
| 1 | 训练侧 `train(rnd)` 从未实现 | FREEZE L92 | **已迁到 v2** | `app/train_round.py::run` 实现八步；`contest_submission.train` 委托；`tests/test_p2.py` 17 项通过 | — |
| 2 | predict 没有错误通道 | FREEZE L93；对照清单 C1 | **已迁到 v2（口径改写）** | 不是 FREEZE 写的 fail-fast，而是"总输出 + 声明默认"：每行必出一行，行级问题走预注册兜底，批级才 fail-fast。`P0设计说明.md` §一；`tests/test_p0.py` fallback 用例 | — |
| 3 | 逐行流式未落地 | FREEZE L94；C2 | **已迁到 v2** | `harness.py::cmd_predict` 逐行 readline→predict→`_emit` flush；`put` 行数事后自检 | — |
| 4 | checkpoint 来源变更后未重测 | FREEZE L95；C5 | **已迁到 v2（本地）／真机未验** | `CONTEST_ARTIFACT_PATH` 单文件装载，`tests/test_p0.py` 自建 npz 夹具；平台挂载仍未实测 | 真机待验：`platform-s0/STATUS_BLOCKED.md`（登录卡点） |
| 5 | 缺 `task` 的默认分支未清理 | FREEZE L96；C3 | **已迁到 v2（口径改写）** | FREEZE 说"直接判协议错误"，实际实现是"上下文 > 行 > 产物"，两者都缺时用产物 task 并在 stderr 标注（`contest_submission.py` L337–348、L356–357） | — |
| 6 | 一个镜像声明两个 task 的打包方式未定 | FREEZE L97；C7 | **已迁到 v2（平台部分悬空）** | `describe()` 报两个 task；按上下文/行分派；上下文与产物冲突启动期 exit 2（L392–395）。平台是否按 task 分进程仍无答案 | `2026-11-16/stream-contract-segmenter/outputs/SDK-1.0.2对照清单.md` §3 C7 |
| 7 | `dataset_key` 与 `inputs` 结构不符未覆盖 | FREEZE L98；C6 | **仍然悬空** | v2 只把 `dataset_key` 当作 `per_cond_seq` 的分组键（`contest_submission.py` L370），**没有任何结构对账**；错配时只能靠行级兜底，不会判定"这批数据与产物不符" | 同左 C6（清单 §3） |
| 8 | 检测器未接入路由 | FREEZE L99 | **已迁到 v2（红线口径变更）** | v1 红线是"检测器只进诊断，不参与 action 判定"；v2 P1 改为"只在 `train` 里参与候选选择，predict 不导入检测器"（`test_p1.py` 子进程断言 `False False`）。这是任务书明确批准的变更 | — |
| 9 | 分段器只在本地合成流验证 | FREEZE L100 | **属过期作废** | FREEZE §三已列为废弃：平台逐行、无错误帧，分段器的"这行拒绝、下行继续"语义不存在；v2 无分段器 | `2026-11-16/stream-contract-segmenter/outputs/stream_segmenter.py`（原样冻结） |
| 10 | 四条平台待定项（进程粒度、隐藏集 `dataset_key` 一致性、回归 `answer=0` 边界、样本行多余键） | FREEZE L101 | **仍然悬空** | 只有 `answer=0` 有代码层面的保守处理（回归 clip 到 ≥0，`contest_submission.py` L379）；其余三条无实现、无平台答复。`赛方问询清单.md` Q1–Q8 覆盖了相邻问题，但没有逐条回答这四项 | `2026-11-15/问询_终稿.md`（问 1–12）；`2026-11-17/contest-v2/赛方问询清单.md` |

**悬空项清单（单列）**：#7、#10；以及 #4 的真机重测、#6 的进程粒度确认。

---

## 3. 其他"已确认事项"对账

| 确认事项 | 依据出处 | 现状 | 三态 | 若缺失，最后可见位置 |
|---|---|---|---|---|
| A 流 S0 = 0.7140297…（官方末轮逐位一致） | `platform-s0/STATUS_BLOCKED.md` L10–13 | 本地闭环证据在盘；平台试运行未提交 | 已落地且可验证（本地） | — |
| B 流常数先验 floor：提交侧 536.27 / 留一参照 522.97 / oracle 型均值 278.06（不可提交），且是唯一 clip-free 数 | `b-floor/floor_report.md` L1–11、L81 | v1 报告与 `floor_results.json` 在盘；v2 **没有把它写进任何代码或判据**，只在 manifest upstream 钉了 hash | 只有文档没有代码（v2 侧） | `2026-11-15/b-floor/floor_results.json` |
| 检测器库 12 模块 / 19 个检测器 / 启用 3 个（argmax_kl、mahal_pen、bbsd） | `detector-lib/REPORT.md` L4、§10.1 | v2 只重写 KL / 马氏 / BBSD 三个，且是简化实现，不读 `detlib/`；其余 16 个条件激活动作未迁 | 只有文档没有代码（v2 侧） | `2026-11-16/detector-lib/detlib/` |
| 漂移基准（`bench_lib.py` + 报告）作为 P3 判据来源 | `drift-benchmark/report.md`、`bench_lib.py` | v2 未引用；P1 阈值 0.15 / 6.0 / 25.0 是另定的小样本口径 | 只有文档没有代码（v2 侧） | `2026-11-16/drift-benchmark/` |
| 分段器 200/200、CLI 55/55 逐字节一致 | FREEZE L15；两份验证报告 | 结论保留为离线证据；平台路径不使用 | 已落地且可验证（离线） | — |
| 策略库 `policy_library.json`（P4 复用） | `signature-routing/outputs/policy_library.json`；manifest p1 `P4` | v1 在盘；v2 未消费，selector 用自己的 3 候选表 | 只有文档没有代码（v2 侧） | 同左 |
| 经验库 36 条（35 条 adopted，4 类动作） | `experience_log.jsonl`（`run_id/action/outcome/adopted/rollback_to`） | v1 原样冻结；v2 不读该文件，改用 `rnd.history()` 作为经验入口 | 已落地且可验证（作为 v1 资产）／未迁入 v2 | — |
| 官方基线 `data.py` / `model.py` / `predict.py` 作为 v2 形状与数值参考 | FREEZE 可移植清单 #6–#8 | v2 重写为线性头 + npz，不直接调用；`data.py` 的禁用输入约束未在 v2 体现 | 已落地且可验证（参考关系） | — |
| 平台试运行（上传镜像、拿回执） | `platform-s0/STATUS_BLOCKED.md` | 从未提交；回执/平台分/六类 F1 为空 | **确实缺失** | `2026-11-15/platform-s0/试运行_交接记录.md` |
| 赛方 12 条问询（含问 8"推理阶段预处理变换是否算更换模型"） | `2026-11-15/问询_终稿.md` Q1–Q12 | 无答复记录 | **确实缺失**（信息缺口，非代码缺口） | 同左；v2 又追加 Q4–Q8 |
| v2 P0 承诺：总输出+声明默认、形状优先级、两守卫、无 pyc | `P0设计说明.md` §一/§四/§三；`tests/test_p0.py` | 25 项通过；`/app` 快照无新文件 | 已落地且可验证 | — |
| v2 P1 承诺：检测下沉 train、transductive 拒绝、`selector_trace` | `manifest.json` p1；`tests/test_p1.py` | 15 项通过；predict 子进程不导入 detectors/selector | 已落地且可验证 | — |
| v2 P2 承诺：训练循环、真提案、五字段账本、R1、可复现 | `manifest.json` p2；`tests/test_p2.py` | 17 项通过；同 history 两次运行账本一致 | 已落地且可验证 | — |
| v1 可移植 28 文件 sha256 钉定 | `manifest.json` upstream（`file_count: 28`）+ `verify_upstream.py` | 全部通过 | 已落地且可验证 | — |
| SDK 目录权限与入口形态（`/app`、`/data`、`/state` 只读，`/workspace` 可写） | SDK 1.0.2 说明；`P0设计说明.md` §三 | 按文档实现；平台从未实测 | 只有文档没有代码（权限部分无法本地证伪） | `ATTACHMENT: contest-harness-sdk-1.0.2.md` |

---

## 4. 三态统计

| 三态 | 条目 |
|---|---|
| 已落地且可验证 | FREEZE #1 #3；#2 #5 #6 #8（口径改写后）；A 流 S0；分段器离线结论；经验库（作为 v1 资产）；v2 P0/P1/P2 三组共 57 项；28 文件 hash |
| 只有文档没有代码 | B floor 536.27（提交）/ 522.97（留一参照）/ 278.06（oracle，不可提交）（v2 侧）；检测器 12 模块（v2 侧）；漂移基准（v2 侧）；策略库（v2 侧）；SDK 权限契约 |
| 确实缺失 | `dataset_key`↔`inputs` 结构对账（#7）；平台四条待定项（#10）；图像预处理（全链路）；平台试运行回执；赛方 12 问答复 |
| 过期作废 | 分段器（#9）；混合流；predict 时批级检测；独立 PyTRIO 端点 |

---

## 5. 最可能被误认为"东西没了"的几处，及真实原因

1. **"图像预处理没了"** —— 从未存在。全仓没有任何图像读写/变换代码；三条数据流都是表格或序列。唯一"图像"字样来自一份外部 AI 故事生成项目的 README 和另一赛题的图纸方案，属材料混入。
2. **"预处理整段没了"** —— 默认产物是 `identity`。R1 固定续训父检查点，动作字段缺省即不处理；`P0设计说明.md` 与 `train_round.py` 都是按"第一轮不追分、先保底"设计的。要看预处理，得看 A/B 候选与 `actions.py`，不在默认路径上。
3. **"`per_cond_mono_norm` 没了"** —— 它被划为 transductive：需要整条单元轨迹才能估统计量，与逐行因果冲突。v2 在 `actions.check_stage` 里显式拒绝并给明确报错，不是静默删掉。经验库 23/36 条都是它，所以"看起来主体被砍了"，实际是平台形态不允许。
4. **"检测器全没了"** —— v1 有 12 模块 / 19 个检测器，v2 只重写 KL、马氏、BBSD 三个。其余 16 个在 v1 里本就是 `conditional`（需标定才允许动作），迁移优先级低于入口正确性。三份 v1 报告与 `detlib/` 原样在盘。
5. **"经验库没了"** —— `experience_log.jsonl` 36 条原样冻结，但 v2 不读它，改由 `rnd.history()` 承接平台侧经验。两套经验体系目前不互通，这是真实缺口，不是文件丢失。
6. **"FREEZE 说的 fail-fast 和判协议错误不见了"** —— 那两条是冻结时的口径，后来被两轮修正取代（总输出+声明默认；上下文 > 行 > 产物）。FREEZE 按"不回写历史"的硬约束保持原样，所以文档看起来"和代码不一致"。
7. **"B 流 522.97 / 278.06 的锚点不见了"** —— 这条是真缺：v2 没有引用 `floor_results.json` 的任何数值，只在 manifest 里钉了 hash。回归侧只有 clip ≥0，没有 floor 对照（提交侧应作对照的是单一常数 536.27；522.97 是留一参照；278.06 依赖 `meta.bearing`、不可提交）。
8. **"`image_reports` 说明以前有图像链路"** —— 它是 `manifest.json` 的 `task_declaration` 子键，本意是"镜像上报的 tasks"（`classification`、`regression`），命名不当，与图像无关。未改动，仅登记。

---

## 附：本次核查用到的检索式（可复现）

- `rg -i "预处理|preprocess|图像|图片|image_|\.png|\.jpg" 2026-11-15 2026-11-16`
- `rg -i "图像预处理|图片预处理|image preprocessing" --glob "*.md" .` → 0 命中
- `rg -i "归一化|normaliz|StandardScaler|whiten|CORAL|zca|preproc" 2026-11-15 2026-11-16`
- 经验库统计：`json.loads` 逐行解析 `experience_log.jsonl`，36 条，adopted 35，动作分布 `per_cond_mono_norm 23 / no_op_report_gap 6 / constant_life_prior 6 / rebuild_linear_ratio 1`
- `archive_manifest.json`：794 条文件记录，`portable: true` 28 条
- v2 核验：`py -3 -B -m unittest tests.test_p0 tests.test_p1 tests.test_p2` → 57 passed；`py -3 -B tests/verify_upstream.py` → 28 文件全过

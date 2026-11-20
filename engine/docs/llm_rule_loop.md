# 策略层自改进闭环（LLM 提规则 → 数字闸门验收 → 采纳账本）

> **实验性实现、默认关闭、7 条提案 0 采纳、结论为负、不作为主路径。**
> 本目录下的规则提案/编译/验收代码只用于离线研究，**没有接入任何生产入口**
> （`scripts/run_pipeline.py`、`scripts/run_real_*.py` 均不加载规则），
> `config.json` 未新增任何相关键。请勿把这里的脚本当作可用能力引用。

状态：已在真实流上完成端到端验证（2026-11-11），结论为负。实现位于主仓库
`src/rule_program.py`、`src/rule_proposal.py`、`scripts/propose_rules.py`、
`scripts/verify_rules.py`；验收账本与报告随代码入库，见
`docs/llm_rule_ledger.jsonl`、`docs/llm_rule_report.md`。

## 1. 目标与边界

目标：让"决策层"本身可被外部提案方（LLM 或离线启发式）改进，但改动只有通过**现有数字闸门**才允许进入账本，且每次提案与判决全程可审计。

明确不做的事：
- 不改管线主循环、不改检测器、不改验收闸门阈值、不改预算/槽位/切分/种子。
- 不为"有采纳"放宽任何阈值。零采纳是合法结论。
- 不把模型凭据写入任何文件；PyTRIO 只走本机已登录的 profile，DeepSeek 只读环境变量。

## 2. 四层结构

```
[历史实验摘要] [闸门拒绝日志] [逐批指标] [21 个决策点证据]
        │
        ▼
① 规则提案层  scripts/propose_rules.py
   - 提案方：pytrio(LLM) / deepseek(LLM) / offline(确定性启发式)
   - 输出：结构化规则 JSON（数值 if-then，深度 ≤2，特征白名单）
   - 记录：provider / model / tokens_in / tokens_out / latency / attempts / raw_sha1
        │
        ▼
② 规则编译层  src/rule_program.py::RuleProgramController
   - 与 RuleController 同接口 decide(evidence) -> {tier, rationale}
   - 通过既有 controller 插槽注入 pipeline，不改其他部分
   - 保留与 RuleController 完全一致的预算护栏（full 且 labels_left<600 → incremental）
        │
        ▼
③ 数字闸门验收  scripts/verify_rules.py
   - 同一份真实流、同一切分、同一预算（4000 标注 / 6 槽位）、同种子
   - 候选规则 vs 固定阈值基线，逐批曲线逐位对照
   - 闸门：G0 基线可复现 / G1 非退化 / G2 最小增益 / G3 静态锚 / G4 预算与槽位
        │
        ▼
④ 采纳账本  docs/llm_rule_ledger.jsonl（运行时原文在 llm_rule_out/，默认 gitignore）
   - 每条：提案来源、规则正文、验收指标、采纳/拒绝、拒绝闸门与差值
```

## 3. 规则 JSON 与编译

规则是**数值 if-then**，不是自然语言：

```json
{
  "program_id": "p-...",
  "default_tier": "follow_severity",
  "rules": [
    {"id": "r1", "if": {"all": [
        {"feature": "residual_ratio", "op": "<", "value": 2.0},
        {"feature": "mmd_z", "op": ">=", "value": 3.0}]},
     "then": {"tier": "observe"}}
  ]
}
```

- 条件树最大深度 2：根节点只能是 `all` / `any`，叶子只能是 `{feature, op, value}`。
- 特征白名单：`severity / residual_ratio / psi / mmd_z / mean_psi / ks_p_value / labels_left / retrains_left / batch_index / residual_alarm / input_alarm / task_alarm`。
- 算子：`< <= > >= == !=`。
- 档位：`observe / incremental / full`。
- 默认档：`follow_severity` 表示沿用管线原来的严重度分档，否则为固定档。
- 规则按数组顺序匹配，**首条命中**生效；无命中走默认档。

编译产物是纯函数 `CompiledProgram.evaluate(evidence) -> tier | None`，可单测、可离线复算。控制器在返回前统一套用与 `RuleController` 相同的预算护栏，保证策略差异只来自规则本身。

## 4. 证据与提案输入

提案层使用三类输入：

1. **既有实验摘要**：`runs/real_cmapss_fd00*.csv`、`runs/real_gas_uci.csv` 的逐批曲线（最后一次生成的参考曲线）。
2. **闸门拒绝日志**：扫描 `runs/*.csv` 中含 `event` 的行，统计 `static_anchor_reject / accept_invalid / gradual_retrain_reject / holdout_untrusted / controller_*` 的次数与原因。
3. **逐批指标 + 决策点证据**：复用 `pytrio_20261111/replay_decision.py` 的 `build_cases`，按批从 `real_stream` 现算 21 个决策点（气体 9 + FD001 12）；不使用事件行 CSV 拼接。

另用两项既有工具（不另起一套）：
- `runs/action_calibration.json` + `CostSensitiveDetector`：在决策点证据上查 kNN 代价表，得到"代价最优动作"，供离线启发式提案与提示词摘要使用。标定表来自历史合成流，属于内部夹具派生，不能作为真实数据结论。
- `scripts/calibrate_actions.py`：本轮不重跑（它只支持合成/官方流），仅引用其已生成的标定表。

## 5. 验收闸门（与现有系统同口径，不放宽）

| 闸门 | 含义 | 判据 |
|---|---|---|
| G0 基线可复现 | 固定阈值基线必须与 `runs/*.csv` 参考曲线逐位一致 | 6 位小数逐位相等 |
| G1 非退化 | 候选在每条流上的末段指标不劣于基线 | `tail(candidate) <= tail(baseline)` |
| G2 最小增益 | 候选相对基线的改善达到仓库既有的 `min_relative_gain=0.03` | `(baseline-candidate)/baseline >= 0.03` |
| G3 静态锚 | 候选末段不劣于静态基线 + `static_margin=0.05` | 与 `_static_anchor_eval` 同口径 |
| G4 预算与槽位 | 不越预算、不越槽位 | `labels_used <= 4000` 且 `n_retrains <= 6` |

末段口径与现有脚本一致：回归取最后 5 批均值，分类取最后 3 批均值。任一闸门在任一条流上不通过 → 拒绝，并记录是哪道闸门、差多少。

为什么不在"决策一致率"上验收：决策一致率只能说明"像不像旧规则"，不能说明"是否更好"。本闭环只认数字。

## 6. 实测结论（7 条提案 / 0 采纳）

完整数字见 [llm_rule_report.md](llm_rule_report.md)，逐条记录见
[llm_rule_ledger.jsonl](llm_rule_ledger.jsonl)。

- 提案：6 条离线启发式（含 1 条空规则透传对照）+ 1 条 PyTRIO `Qwen/Qwen3.5-4B`
  （tokens_in=2292 / tokens_out=284 / 时延 32.72s）。
- 采纳：**0 条**。零采纳是合法结果，未为凑数放宽任何闸门。
- FD001：7 条规则全部与固定阈值基线逐批一致（61/61）。基线 `n_gated=0`、
  `n_accept=0`，事件里没有 `controller_decision`——控制器在该流上从未被询问，
  改规则改不动曲线（渐变模式下每批 `partial_fit` 持续适应，周期性全量重训占满槽位）。
- 气体流：规则确实能改变行为，但只有代价。`perf_soft_observe` 把标注从 2301
  压到 1045、重训 4→2，末段 macro-F1 从 0.467266 掉到 0.281418；反方向的
  `deep_drift_full` 多花 720 条标注，F1 一位不变（重训全被既有闸门拒绝）。
- FD003：`low_severity_observe` 与 LLM 提案的末段 RMSE 从 24.477395 抬到
  31.114873（36 批劣化 / 2 批改善），只省 110 条标注；LLM 写的 4 条规则在
  该流上退化成同一条 observe 规则。
- 透传对照（空规则）与基线逐位一致（气体 9/9、FD001 61/61），证明插槽接入与
  预算口径没有被动过。
- G0：FD001 61/61、FD003 70/70 逐位复现通过；气体参考 CSV 早于最近两次管线
  提交，按 `stale_reference` 记录（不阻塞候选比较，但如实报告 4 批不一致）。

**判断：这条路没跑通不是规则写得差，而是可动作空间被结构性压缩**——渐变模式
与预算节流让突发决策点大多不可达，可达点上既有闸门已经把无效重训拦掉。
要推进需改动作空间本身（例如让控制器调节渐变窗口/标注比例），而非调阈值。

## 7. 已知边界（必须写进报告）

1. **控制器插槽只覆盖部分决策点**：管线仅在 `tier != observe` 且冷却结束时询问控制器；规则无法把 observe 变成更新，也无法绕过渐变模式与 APD 抑制。规则能动的是 incremental/full/observe 之间的重排。
2. **气体流只在"不退化"上有意义**：实测规则能改变气体流行为（省标注/多花标注），
   但改变全是负收益（末段 F1 0.467266 → 0.281418 或持平），没有出现既省预算
   又不掉精度的规则。
3. **提案来自 LLM 不等于更好**：LLM 只写规则文本，判决只看数字；提案失败（解析失败/越权特征）直接记为拒绝，不进入验收。
4. **标定表是历史合成流产物**：用它做提案输入时必须在报告中标注"历史合成流结果，仅内部追溯"。
5. **单进程离线**：验收不联网、不调外部模型；提案阶段的模型调用单独计数，不混入策略效果。

## 8. 产物

入库（随代码走，作为证据）：
- `docs/llm_rule_ledger.jsonl`：采纳账本（按 program_id 合并，一条提案一条记录）。
- `docs/llm_rule_report.md`：最终验收报告（提案数/采纳数/逐流对照/逐位对照/失败明细）。

运行时（默认 gitignore，避免污染仓库）：
- `llm_rule_out/proposals-<stamp>.json`：提案原文与来源元数据。
- `llm_rule_out/verify/<stamp>/`：逐批曲线、闸门判决、逐位对照表。
- `llm_rule_out/rule_ledger.jsonl`：追加写的原始账本（`--compact-ledger` 可合并重写）。

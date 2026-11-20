# 规则提案验收报告 20261111-233323

- 提案数：7（含 1 条透传对照；提案阶段直接拒绝 0）
- 采纳数：0
- 拒绝数：7

## 提案输入摘要

- 来源文件：`digest-20261111-230103.json`（sha1 54d84b56e220be126d323b92df25e95bfeb45bc2）
- 决策点：{"gas": {"n": 9, "rule_tiers": {"full": 8, "incremental": 1}, "alarms": 9, "residual_alarms": 9}, "cmapss-fd001": {"n": 12, "rule_tiers": {"incremental": 9, "full": 3}, "alarms": 7, "residual_alarms": 12}}
- 规则档位分布：{"full": 11, "incremental": 10}
- 标定表查表（历史合成流，仅内部追溯）：{"observe": 13, "incremental": 0, "full": 0}
- 闸门日志事件：{"static_baseline_deficit": 157, "apd_suppress": 23, "gradual_retrain_reject": 10, "accept_invalid": 5, "static_anchor_reject": 3, "controller_decision": 2}

## 基线复现（G0）
- gas: FAIL（stale_reference：参考 CSV 早于最近一次管线提交，不阻塞候选比较）（参考 runs/real_gas_uci.csv，不一致批数 4）
  - 首个不一致：{'batch': 5, 'ours': '0.358048', 'reference': '0.283820'}
  - 说明：参考曲线早于最近一次管线变更（commit 时间戳晚于 CSV 修改时间），该 CSV 不是 HEAD 代码产物；G0 记为 stale_reference，不阻塞候选比较，但必须如实报告。
- cmapss-fd001: PASS（参考 runs/real_cmapss_fd001.csv，不一致批数 0）
- cmapss-fd003: PASS（参考 runs/real_cmapss_fd003.csv，不一致批数 0）

## 逐条提案

> 「决策点改动」来自重建的 21 个决策点（气体 9 + FD001 12），只是廉价代理；权威信号是下面的真实回放曲线。

| 提案 | 来源 | 规则数 | 决策点改动 | 摘要sha1 | 决策 | 失败闸门 |
|---|---|---|---|---|---|---|
| p-175ffe3ddaf1 | heuristic/-/low_severity_observe | 1 | 0/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G1_no_regression@cmapss-fd003; G2_min_gain@cmapss-fd003 |
| p-3ab154176297 | heuristic/-/deep_drift_full | 2 | 10/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G2_min_gain@cmapss-fd003 |
| p-7db272473264 | heuristic/-/budget_strict | 1 | 1/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G2_min_gain@cmapss-fd003 |
| p-cba9d9f1f8ba | heuristic/-/late_slot_incremental | 1 | 3/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G2_min_gain@cmapss-fd003 |
| p-drift-adapt | pytrio/Qwen/Qwen3.5-4B | 4 | 3/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G1_no_regression@cmapss-fd003; G2_min_gain@cmapss-fd003 |
| p-ea7621320370 | heuristic/-/passthrough_control | 0 | 0/21 | 54d84b56e220 | rejected | G2_min_gain@gas; G2_min_gain@cmapss-fd001; G2_min_gain@cmapss-fd003 |
| p-f12bbb55fa21 | heuristic/-/perf_soft_observe | 1 | 0/21 | 54d84b56e220 | rejected | G1_no_regression@gas; G2_min_gain@gas; G2_min_gain@cmapss-fd001; G2_min_gain@cmapss-fd003 |

## 与固定阈值基线的逐流对照

| 提案 | 流 | 指标 | 基线末段 | 候选末段 | Δloss(正=劣化) | 标注(候选/基线) | 重训(候选/基线) | 逐批劣化/改善 |
|---|---|---|---|---|---|---|---|---|
| p-175ffe3ddaf1 | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-175ffe3ddaf1 | cmapss-fd003 | rmse | 24.477395 | 31.114873 | +6.637478 | 3110/3220 | 6/6 | 36/2 |
| p-175ffe3ddaf1 | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 2301/2301 | 4/4 | 0/0 |
| p-3ab154176297 | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-3ab154176297 | cmapss-fd003 | rmse | 24.477395 | 24.477395 | +0.0 | 3220/3220 | 6/6 | 0/0 |
| p-3ab154176297 | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 3021/2301 | 4/4 | 0/0 |
| p-7db272473264 | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-7db272473264 | cmapss-fd003 | rmse | 24.477395 | 24.477395 | +0.0 | 3220/3220 | 6/6 | 0/0 |
| p-7db272473264 | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 2301/2301 | 4/4 | 0/0 |
| p-cba9d9f1f8ba | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-cba9d9f1f8ba | cmapss-fd003 | rmse | 24.477395 | 24.477395 | +0.0 | 3220/3220 | 6/6 | 0/0 |
| p-cba9d9f1f8ba | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 2301/2301 | 4/4 | 0/0 |
| p-drift-adapt | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-drift-adapt | cmapss-fd003 | rmse | 24.477395 | 31.114873 | +6.637478 | 3110/3220 | 6/6 | 36/2 |
| p-drift-adapt | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 2301/2301 | 4/4 | 0/0 |
| p-ea7621320370 | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-ea7621320370 | cmapss-fd003 | rmse | 24.477395 | 24.477395 | +0.0 | 3220/3220 | 6/6 | 0/0 |
| p-ea7621320370 | gas | macro_f1 | 0.467266 | 0.467266 | +0.0 | 2301/2301 | 4/4 | 0/0 |
| p-f12bbb55fa21 | cmapss-fd001 | rmse | 28.174374 | 28.174374 | +0.0 | 4000/4000 | 6/6 | 0/0 |
| p-f12bbb55fa21 | cmapss-fd003 | rmse | 24.477395 | 24.477395 | +0.0 | 3220/3220 | 6/6 | 0/0 |
| p-f12bbb55fa21 | gas | macro_f1 | 0.467266 | 0.281418 | +0.185848 | 1045/2301 | 2/4 | 1/1 |

## 与固定阈值的逐位对照

| 提案 | 流 | 完全一致批数/总批数 | 最大单批偏差(loss) |
|---|---|---|---|
| p-175ffe3ddaf1 | cmapss-fd001 | 61/61 | 0.0 |
| p-175ffe3ddaf1 | cmapss-fd003 | 32/70 | 97.321303 |
| p-175ffe3ddaf1 | gas | 9/9 | 0.0 |
| p-3ab154176297 | cmapss-fd001 | 61/61 | 0.0 |
| p-3ab154176297 | cmapss-fd003 | 70/70 | 0.0 |
| p-3ab154176297 | gas | 9/9 | 0.0 |
| p-7db272473264 | cmapss-fd001 | 61/61 | 0.0 |
| p-7db272473264 | cmapss-fd003 | 70/70 | 0.0 |
| p-7db272473264 | gas | 9/9 | 0.0 |
| p-cba9d9f1f8ba | cmapss-fd001 | 61/61 | 0.0 |
| p-cba9d9f1f8ba | cmapss-fd003 | 70/70 | 0.0 |
| p-cba9d9f1f8ba | gas | 9/9 | 0.0 |
| p-drift-adapt | cmapss-fd001 | 61/61 | 0.0 |
| p-drift-adapt | cmapss-fd003 | 32/70 | 97.321303 |
| p-drift-adapt | gas | 9/9 | 0.0 |
| p-ea7621320370 | cmapss-fd001 | 61/61 | 0.0 |
| p-ea7621320370 | cmapss-fd003 | 70/70 | 0.0 |
| p-ea7621320370 | gas | 9/9 | 0.0 |
| p-f12bbb55fa21 | cmapss-fd001 | 61/61 | 0.0 |
| p-f12bbb55fa21 | cmapss-fd003 | 70/70 | 0.0 |
| p-f12bbb55fa21 | gas | 7/9 | 0.581178 |

## 闸门失败明细
- p-175ffe3ddaf1 / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-175ffe3ddaf1 / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-175ffe3ddaf1 / G1_no_regression @ cmapss-fd003：{"candidate_metric": 31.114873, "baseline_metric": 24.477395, "metric": "rmse", "higher_is_better": false, "candidate_loss": 31.114873, "baseline_loss": 24.477395, "loss_delta": 6.637478, "limit": "candidate_loss <= baseline_loss"}
- p-175ffe3ddaf1 / G2_min_gain @ cmapss-fd003：{"relative_gain": -0.271168, "required": 0.03, "candidate_loss": 31.114873, "baseline_loss": 24.477395}
- p-3ab154176297 / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-3ab154176297 / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-3ab154176297 / G2_min_gain @ cmapss-fd003：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 24.477395, "baseline_loss": 24.477395}
- p-7db272473264 / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-7db272473264 / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-7db272473264 / G2_min_gain @ cmapss-fd003：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 24.477395, "baseline_loss": 24.477395}
- p-cba9d9f1f8ba / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-cba9d9f1f8ba / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-cba9d9f1f8ba / G2_min_gain @ cmapss-fd003：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 24.477395, "baseline_loss": 24.477395}
- p-drift-adapt / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-drift-adapt / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-drift-adapt / G1_no_regression @ cmapss-fd003：{"candidate_metric": 31.114873, "baseline_metric": 24.477395, "metric": "rmse", "higher_is_better": false, "candidate_loss": 31.114873, "baseline_loss": 24.477395, "loss_delta": 6.637478, "limit": "candidate_loss <= baseline_loss"}
- p-drift-adapt / G2_min_gain @ cmapss-fd003：{"relative_gain": -0.271168, "required": 0.03, "candidate_loss": 31.114873, "baseline_loss": 24.477395}
- p-ea7621320370 / G2_min_gain @ gas：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 0.532734, "baseline_loss": 0.532734}
- p-ea7621320370 / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-ea7621320370 / G2_min_gain @ cmapss-fd003：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 24.477395, "baseline_loss": 24.477395}
- p-f12bbb55fa21 / G1_no_regression @ gas：{"candidate_metric": 0.281418, "baseline_metric": 0.467266, "metric": "macro_f1", "higher_is_better": true, "candidate_loss": 0.718582, "baseline_loss": 0.532734, "loss_delta": 0.185848, "limit": "candidate_loss <= baseline_loss"}
- p-f12bbb55fa21 / G2_min_gain @ gas：{"relative_gain": -0.348858, "required": 0.03, "candidate_loss": 0.718582, "baseline_loss": 0.532734}
- p-f12bbb55fa21 / G2_min_gain @ cmapss-fd001：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 28.174374, "baseline_loss": 28.174374}
- p-f12bbb55fa21 / G2_min_gain @ cmapss-fd003：{"relative_gain": 0.0, "required": 0.03, "candidate_loss": 24.477395, "baseline_loss": 24.477395}

## 提案方元数据
| 提案 | provider | model | tokens_in | tokens_out | 时延(s) | 尝试 |
|---|---|---|---|---|---|---|
| p-drift-adapt | pytrio | Qwen/Qwen3.5-4B | 2292 | 284 | 32.72 | 1 |

## 为什么这条路还没跑通（实测）

- FD001：7 条规则里有 0 条改变了逐批曲线。基线 n_gated=0、n_accept=0、events={"partial_fit_gradual": 34, "gradual_retrain": 5, "gradual_mode_enter": 4, "static_baseline_deficit": 4, "gradual_mode_exit": 3, "init_train": 1, "compute_summary": 1}；候选 n_accept=0、静态锚拒绝=0、验收不可信=0。渐变模式下每批 partial_fit 持续适应，周期性全量重训占满槽位；突变重训路径在预算耗尽后基本不可达，控制器改 tier 改不动最终曲线。
  - 基线事件分布：{"partial_fit_gradual": 34, "gradual_retrain": 5, "gradual_mode_enter": 4, "static_baseline_deficit": 4, "gradual_mode_exit": 3, "init_train": 1, "compute_summary": 1}
  - 事件里没有 controller_decision：控制器在实际回放中从未被询问，改规则自然改不动曲线。
- 气体流：7 条规则里有 1 条改变了逐批曲线；基线标注 2301/重训 4，候选标注区间 1045~3021。改变最大的示例是 p-f12bbb55fa21：标注 1045/重训 2，末段 F1 0.467266 → 0.281418。规则在这里确实能改变行为，但省下的标注换来的是 F1 下降，没有出现既省预算又不掉精度的规则。
- FD003 上的额外反例：p-175ffe3ddaf1 把末段 RMSE 从 24.477395 抬到 31.114873（36 批劣化 / 2 批改善，最大单批 loss 偏差 97.321303），标注 3220 → 3110。
- LLM 提案（p-drift-adapt，4 条规则）在 FD003 上的逐批曲线与单条「低严重度跳过」规则完全相同：它看起来写了 4 条规则，实际生效路径退化成同一条 observe 规则，且代价同样是末段 RMSE 变差。
- 更激进的全量重训（p-3ab154176297）在气体流上多花 720 条标注，末段 F1 完全不变：这些重训都被验收闸门拒绝，预算白花。
- 结论：当前控制器的可动作空间被两层结构性因素压缩——（a）渐变模式与预算节流让突发决策点大多不可达；（b）可达的决策点上，验收闸门本身已经把无效重训拦掉，改档位不改变最终模型。这不是调阈值能解决的，需要改动作空间本身。

## 边界与诚实口径

1. 验收只看数字：同一份流、同一切分、同一预算（4000 标注 / 6 槽位）、同种子；基线复现参考曲线的结果单独列出（G0）。
2. 规则只能通过既有 controller 插槽在 incremental/full/observe 之间重排；无法把 observe 变成更新，也无法绕过渐变模式与 APD 抑制。
3. 零采纳是合法结果；本轮没有为了让账本好看而调参或放宽闸门。
4. 标定表来自历史合成流（内部夹具派生），只用于提案与提示词，不作为真实数据结论；对外材料需标注「历史合成流结果，仅内部追溯」。
5. 未验证项：FD002、FD004、XJTU-SY 轴承流；本轮未跑。

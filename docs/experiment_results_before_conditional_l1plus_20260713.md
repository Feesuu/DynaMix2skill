# 条件式 L1+ 抽象实验前的结果登记

记录日期：2026-07-13
仓库：`/mnt/data/yaodong/codes/DynaMix2skill`
Git HEAD：`06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a`

## 1. 记录边界

这份文档冻结当前能够由本地评测 artifact 核实的历史结果。它们全部产生于以下新逻辑之前：

- L1+ 去重后少于两张不同子卡时不调用 LLM。
- L1+ 只有存在共同不变量时才生成父卡。
- singleton 上层社区只保留社区链接，不复制 ExperienceCard。
- 拒绝与已有低层卡完全重复的高层卡。
- OfficeQA 与 SpreadsheetBench 使用各自的数据集专属经验抽取约束。

因此，本文中的任何旧树、旧 nodebank 和旧 heldout 分数都不能当作新逻辑的实验结果。新实验必须使用新 run dir 从建树阶段重新开始，不能 resume 旧树。

纳入主表的最低条件是：完整分母、评测完成、结果文件仍存在，并且能够确认主要 evaluator。因网络中断而未完成、selected slice、smoke test 和无法确认协议的结果不进入主表。

## 2. SpreadsheetBench：新 AWQ 轨迹协议

### 2.1 协议

- 数据集：SpreadsheetBench verified 400。
- Train：`[0, 200)`；Heldout：`[200, 400)`。
- 模型：`Qwen3.5-9B-AWQ`。
- Train agent：`cli_only`。
- DynaMix heldout agent：`cli_skill_preloaded`，动态检索 nodebank top-10。
- 主评测：LibreOffice recalc。
- Thinking：`true`；max turns：`30`。
- Embedding：`Qwen3-Embedding-8B`，`32000` 输入上限，`28000` chunk，`1000` overlap，mean pooling。

### 2.2 完整结果

| 运行 | Recalc | No-recalc audit | 说明 |
| --- | ---: | ---: | --- |
| Fresh train/no-skill | **97/200 = 48.5%** | 88/200 = 44.0% | 新 AWQ 环境重新采集的 train 轨迹及 train eval |
| Fresh static DynaMix heldout | **97/200 = 48.5%** | 75/200 = 37.5% | 使用 fresh 200 条 train records 建树，旧版无条件 L1+ 抽象 |
| Recorded vanilla heldout | **82/200 = 41.0%** | 68/200 = 34.0% | `cli_only`，不使用 nodebank/skills |

关键 artifact：

- Fresh run：`runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/`
- Train eval：`runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/trace2skill_train_eval.json`
- DynaMix heldout eval：`runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/trace2skill_heldout_eval.json`
- Tree summary：`runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/dynamix_tree/summary.json`
- Vanilla eval：`runs/spreadsheet_vanilla_heldout_qwen35_awq_20260708_170249/trace2skill_heldout_eval.json`
- 已封存结果：`runs/version_snapshots/20260710_spreadsheet_awq_retrain200_static_heldout97/`

### 2.3 公平性边界

`97/200` 与 `82/200` 都是各自 run 的有效记录，但不是严格匹配的因果对比：DynaMix 使用远端 HTTPS endpoint 和 32 workers，vanilla 使用本地 tunnel endpoint 和 16 workers；DynaMix 生成了 200/200 个结果，vanilla 仅完成 197/200 个 agent 输出。因此目前只能分别报告两项分数，不能把 `+7.5pp` 写成已经严格控制后的方法增益。

完整审计见：`research/skill_tree_refinement/research-loop/evidence/baseline_fairness_audit.md`。

## 3. SpreadsheetBench：旧版控制消融

这一组实验共享旧版 full soft GMM tree 的 `90/200` 锚点，内部可用于判断组件优先级；但 train trajectories 来自旧 `Qwen3.5-9B` run，train eval 不是当前 AWQ + LibreOffice 完整协议。因此它们是 historical controlled evidence，不能和上一节 fresh `97/200` 合并计算增益。

| 旧版实验 | 唯一主要改动 | Recalc | 相对旧锚点 |
| --- | --- | ---: | ---: |
| Full soft GMM hierarchy | GMM-BIC + cumulative-mass soft membership + full hierarchy | **90/200 = 45.0%** | 0.0pp |
| Hard assignment GMM | 每个 item 只进入 posterior argmax 社区 | 82/200 = 41.0% | -4.0pp |
| KMeans elbow | GMM-BIC 改为自动 elbow KMeans | 85/200 = 42.5% | -2.5pp |
| Fixed-K KMeans | 固定 `K=8` | 80/200 = 40.0% | -5.0pp |
| L0 single-card | 每个 L0 社区最多生成一张卡 | **90/200 = 45.0%** | 0.0pp |
| L1 only | 只做 L0 -> L1 | 89/200 = 44.5% | -0.5pp |
| Retrieve L1 only | 完整树只导出/检索 L1 | 82/200 = 41.0% | -4.0pp |
| Retrieve L2+ only | 完整树只导出/检索 L2 及以上 | 75/200 = 37.5% | -7.5pp |

关键 artifact：

- Full old anchor：`runs/static_qwen35_awq_8bembed_chunk28000_minsplit4_after_budget_fix_20260702_142314/scenarios/static_build/`
- Variants：`runs/ablations/static_controlled/`
- 消融审计：`research/skill_tree_refinement/research-loop/evidence/ablation_audit.md`
- 旧协议封存：`runs/version_snapshots/20260709_legacy_mixed_protocol_static_results/`

这组结果支持的最窄结论是：在旧协议中，soft GMM 和 GMM-BIC 值得保留；L0 多卡没有显示额外收益；完整深层树相对 L1-only 只多 1 个正确任务；只检索 L2+ 明显较弱。它不能证明这些结论在 fresh AWQ 轨迹上仍成立，也没有多 seed 显著性证据。

## 4. OfficeQA：旧版无条件 L1+ 抽象

### 4.1 共同协议

- Train：`train + val`，共 74 条；Heldout：`test`，172 条。
- 模型：`Qwen3.5-9B-AWQ`。
- Oracle Parsed Pages + candidate-file absolute-path allowlist。
- 工具：OfficeQA 文档工具；max tool turns：`30`。
- Temperature：`0.6`；thinking：`true`；无 completion token 人工上限。
- 主指标：SkillOpt-compatible EM/F1。
- Official reward 仅作为 audit。
- DynaMix heldout 动态检索 top-10 nodebank experience；vanilla 不注入经验。

### 4.2 完整结果

| 运行 | SkillOpt EM | SkillOpt F1 | Official audit hard | Agent OK |
| --- | ---: | ---: | ---: | ---: |
| OfficeQA Static DynaMix new-prompt | 69/172 = 40.12% | 40.91% | **77/172 = 44.77%** | 165/172 |
| OfficeQA vanilla | 70/172 = 40.70% | 42.34% | **76/172 = 44.19%** | 169/172 |

关键 artifact：

- DynaMix：`runs/officeqa_dynamix_newprompt_full16_20260706_142153/`
- DynaMix report：`runs/officeqa_dynamix_newprompt_full16_20260706_142153/officeqa_experiment_report.json`
- DynaMix per-task results：`runs/officeqa_dynamix_newprompt_full16_20260706_142153/heldout_rollout/officeqa_results.json`
- Vanilla：`runs/officeqa_vanilla_test_16w_20260704_1524/`
- Vanilla report：`runs/officeqa_vanilla_test_16w_20260704_1524/officeqa_vanilla_report.json`
- Vanilla per-task results：`runs/officeqa_vanilla_test_16w_20260704_1524/vanilla_rollout/officeqa_results.json`

两次 run 的 endpoint 分别为本地 tunnel `11802` 与 `11803`，因此差异只能作描述性结果。主指标下 DynaMix 少 1 题且 F1 更低；official audit 下 DynaMix 多 1 题。这个量级不能支持 DynaMix 优于 vanilla 的结论。

## 5. 不进入主表的结果

- 旧 static `86/200 = 43.0%`、`81/200 = 40.5%`：保留在 legacy snapshot 中，只作历史复盘。
- min-split 8 的 `88/200 = 44.0%`：运行受后来确认的 temperature propagation bug 和 task-local workdir guard 变更影响，只能作诊断，不作为公平 headline。
- 未完成、网络超时、selected slice、OfficeQA 24 条早期 heldout、20 条 smoke：不用于方法结论。
- Dynamic SpreadsheetBench：当前没有满足新协议并可作为主结果的完整可信 run，因此不登记 headline 分数。

## 6. 下一轮新实验的对照要求

下一轮实验测试的是“条件式 L1+ 抽象 + 跨层完全重复拦截 + 数据集专属 analyst prompt”。为隔离变量，应：

1. 使用 fresh AWQ run 的同一份 200 条 train records，不重新混入 legacy records。
2. 使用新 run dir，从 tree build、nodebank export 开始重建；不得复用旧 tree 或 nodebank。
3. 固定 GMM-BIC、cumulative-mass soft membership、embedding、chunk、top-k、模型、thinking、temperature、max turns 和 LibreOffice evaluator。
4. 使用相同 endpoint、workers、timeout/retry 和 heldout 200 条分母运行旧逻辑对照或至少 matched vanilla，避免再次形成跨运行混杂。
5. 除 heldout accuracy 外，记录各层输入卡数、singleton community 数、跳过原因、父卡生成数、跨层重复拦截数、nodebank 层级分布、检索层级分布和 token/runtime。

当前新逻辑尚无真实建树或 heldout 结果。本文是实验前基线登记，不是新方法结果报告。

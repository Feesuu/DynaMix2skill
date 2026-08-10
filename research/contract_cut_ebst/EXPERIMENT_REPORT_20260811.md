# Contract-Cut EBST v1 正式实验报告

## 1. 结论先行

这次实验完成了用户要求的完整链路：复用 SpreadsheetBench `0:200` 已有轨迹，按 `batch_size=8` 构建 Contract-Cut EBST，导出多个 Skill folders，再对 `200:400` 使用 top-1 完整 `SKILL.md` 做 heldout，并以 LibreOffice recalc 作为主评测。

工程结论是通过：流程完整结束，200 条 heldout 全部进入分母，树的单父、平衡、容量、精确局部 split、精确 Atom cover 和固定候选树上的最优 cut 均有 artifact 支持。

研究结论是不通过：当前方法得到 `72/200 = 36.0%`，低于历史同模型 vanilla artifact 的 `82/200 = 41.0%`。153 个 active skills 中 147 个只覆盖 1 个 Atom，200 次检索中 195 次命中 singleton skill。这个版本没有形成预期的“少量、结构化、可复用的 Skill folders”，而是主要退化成逐轨迹经验库。

因此，本分支应保留为一个可复现的负结果和算法诊断 checkpoint，不应替代当前最佳方法，也不能支持 Contract-Cut 已提升泛化能力的论文 claim。

## 2. 版本与输入身份

| 项目 | 值 |
|---|---|
| Repository | `https://github.com/Feesuu/DynaMix2skill.git` |
| Branch | `research/contract-cut-ebst-v1` |
| 正式 run 使用的代码 commit | `1ebcbd1` |
| Base commit | `5aea7dc` |
| Run directory | `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2` |
| Train records | SpreadsheetBench dataset order `0:200` |
| Heldout | SpreadsheetBench dataset order `200:400` |
| Records path | `/mnt/data/yaodong/codes/DynaMix2skill/runs/spreadsheet_splitthink_rolloutfalse_analysttrue_best52_20260720_180410/ordered_records.json` |
| Records SHA256 | `f07519085195e8fa77d036e4cea5cc3654c57722d8cd9476fffc93da51075db1` |
| Records count | 200 |
| Source success labels | 99 success, 101 failure |
| Dataset SHA256 | `bcecaa89a005bd4e3bbe98da150a86e8062c27f262e575d5e47bd9861b3525e7` |

Run 名中的 `r3_compiler_v2` 表示第三次 downstream build 和第二版 compiler contract。它不是一次 fresh Atom extraction，也不是对 heldout 调参后的选择。正式 run 声明复用了前一轮按照相同 Atom analyst 协议生成的 200 行 `atom_drafts.jsonl`，其 SHA256 为 `e49efa0b431fd2ec993cf9724840dc182019fe2ce1a33eb1a8e5a744241ad987`。它没有复用旧 compiler cache、旧 tree、旧 cut、旧 skills 或旧 heldout 输出，具体记录在：

- `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/cache_reuse_manifest.json`
- `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/stages/build.complete.json`
- `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/stages/heldout.complete.json`

补充 post-run identity audit 将 r2 与 r3 drafts 的字节级 SHA、200 个唯一 record indexes、统一 analyst protocol hash `2b79d6341e0b859ad3a50c22cbdf92453197130aa9a90a8ccbd96280dad315a7` 和 r2 generation protocol 绑定起来。r2 protocol 明确记录 no-thinking、temperature 0、`max_tokens=null`、workers 8 和同一个 records SHA。该证据只支持“已验证的 Atom reuse + fresh downstream build”，不支持“r3 重新执行了 Atom analyst”：

- `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/research/contract_cut_ebst/FORMAL_RUN_AUDIT_20260811.json`

## 3. 冻结实验协议

### 3.1 模型与并发

| 参数 | 值 |
|---|---|
| LLM | `Qwen3.5-9B-AWQ` |
| LLM endpoint | `http://10.26.1.184:18085/v1` |
| Context length | 100,000 |
| Thinking | `false`，Atom、compiler、heldout 一致 |
| Temperature | `0.0` |
| Request `max_tokens` | 未设置，服务健康检查例外为 16 |
| Request timeout | 1,200 秒 |
| LLM concurrency/workers | 8 |
| Atom arrival batch size | 8 |
| Heldout max turns | 30 |
| Embedding model | `Qwen3-Embedding-8B` |
| Embedding endpoint | `http://10.26.1.184:18007/v1` |
| Embedding dimension | 4,096 |
| Embedding max model length | 32,000 |
| Embedding batch/concurrency | 8 / 8 |

API key 只以 SHA256 指纹写入运行摘要，没有提交明文 key。

### 3.2 Atom 协议

每条完整训练轨迹最多生成一个 Contract Atom。成功和失败记录使用相同 prompt 和 schema。输入包括 instruction、完整 rollout、produced answer、LibreOffice evaluator evidence 和 success/failure 状态。系统不按字符数截断 evidence；如果完整 prompt 超过 92,000 tokens，则显式排除。

Atom 只用以下文本建立边界 embedding：

```text
Applicability: {trigger}
Scope: {scope}
Success condition: {verification}
```

Gold、evaluator、task ID、路径、answer position 和输出不会进入 embedding、检索 query 或导出的公开 skill 文本。每个 Atom 的结构权重固定为 1。

对 153 个实际导出 skill 的 459 个语义文件执行了全量扫描。公开 reference Atom 只有七个允许字段，公开 provenance 只有十个允许字段；未发现 private structured keys、带边界的 source task ID、record SHA、明确 evaluator/gold 标签或明文 API key。agent-readable mirror 中也未发现 `/home/yaodong` 或 `/mnt/data` 私有路径。这个扫描可以排除已知结构化和标识符泄漏，但不能从数学上证明自然语言或常见数字与某个 gold value 没有偶然重合。完整 scan definition 和结果记录在 `FORMAL_RUN_AUDIT_20260811.json`。

### 3.3 EBST 结构协议

- 单父在线树，Atom 只存于叶子。
- `M=8`，非 root 最小 occupancy `m=4`。
- 第九个 entry 导致 overflow 时，枚举全部 `C(9,4)=126` 个 4/5 partition。
- 每个 partition 先最小化两侧最大 exact cover radius，再最小化 radius sum，最后用 canonical ID 稳定打破平局。
- 插入下降使用 `(radius enlargement, representative distance, stable ID)` 的字典序。
- 所有 representative 都是实际 descendant Atom，cover radius 对所有 descendant 精确计算。

### 3.4 Contract cut 与 Skill compiler

固定 augmented candidate tree 上的代价是：

```math
F(u)=\min\left(D(u)+\beta,\sum_{v\in children(u)}F(v)\right),
```

其中 `beta=1.0`，`D(u)` 是该候选区域内 Atom 到 representative 的平方 chord distance 总和。物理 EBST leaf 还增加每个 Atom 的 deterministic terminal candidate，从而保证每个 accepted Atom 最终恰好被一个 skill 覆盖。

compiler 只接收结构化 Atom，不接收原始 heldout 或训练轨迹。它只能输出一个包含条件分支的 contract，或 `split_required`。格式错误只允许一次 repair。没有额外 heuristic semantic validator，也没有第二个 LLM critic。

### 3.5 Heldout retrieval

- 检索 corpus 只包含 final optimal-cut 的 active Skill folders。
- Skill embedding 文本只含 `name + applicability + objective`。
- Query 精确为 `instruction + "\n\nTask type: " + instruction_type`。
- `answer_position` 作为 audit 字段出现在 selection log 中，但不会额外拼入 query。部分 instruction 本身自然包含单元格范围，这与泄漏 metadata 是两件事。
- SpreadsheetBench agent 的冻结任务上下文仍包含 benchmark 提供的 `answer_position`，与历史 vanilla/skill runs 一致。这里的控制变量是它不参与 Skill retrieval query，而不是从任务执行环境删除它。
- Dense cosine top-1。
- 注入完整 `SKILL.md`，并提供仅指向公开 skill mirror 的可选 reference 目录。
- 主评测为 LibreOffice 24.2.7.2 recalc 后的 workbook correctness。

## 4. 建树结果

### 4.1 Atom 接收与排除

| 项目 | 数量 |
|---|---:|
| 输入轨迹 | 200 |
| Accepted Atoms | 180 |
| Excluded records | 20 |
| Accepted success-source | 91 |
| Accepted failure-source | 89 |

排除原因：

| 原因 | 数量 |
|---|---:|
| revision 后仍复制 source literal | 18 |
| revision 后仍包含 evaluator artifact | 1 |
| 完整 analyst prompt 超 92,000 tokens | 1 |

20/200 的排除率是一个真实风险。特别是 18 条 literal exclusion 表明当前 reusable-content checker 可能过严。它避免训练实例细节进入公开 skill，但也丢掉了 10% 训练 evidence。这个结果不能被描述为“200 条训练轨迹全部进入树”。

### 4.2 物理树审计

| 指标 | 结果 |
|---|---:|
| Atom count | 180 |
| Physical node count | 38 |
| Leaf count | 32 |
| Internal count | 6 |
| Height | 2 |
| Certified height bound | 4 |
| Exact split count | 35 |
| Leaf occupancy | 4 到 8 |
| Internal occupancy | 5 到 7 |
| Leaves at same depth | 是 |
| Unique Atom placement | 是 |
| Exact cover radius nodes | 38/38 |
| All Atom weights equal 1 | 是 |

这部分支持结构性 claim：该 run 的物理树是平衡、单父、容量合法的 EBST，并且每次 overflow split 都是指定 126 partition 搜索空间上的精确局部最优。

这部分不支持语义 claim：结构合法不等于 embedding boundary 正确，也不等于每个树节点都可编译成有用 skill。

### 4.3 Final cut 与 active skills

| 指标 | 结果 |
|---|---:|
| Final active skills | 153 |
| 物理 EBST node skills | 6 |
| 单 Atom terminal skills | 147 |
| 多 Atom 覆盖 | 33 Atoms |
| 单 Atom 覆盖 | 147 Atoms |
| Final objective | 169.5421 |
| Distortion | 16.5421 |
| Opening cost | 153.0 |
| Registry versions | 224 |
| Active / superseded / archived | 153 / 63 / 8 |

Active support histogram：

| Skill support | Skill 数量 |
|---|---:|
| 1 Atom | 147 |
| 5 Atoms | 5 |
| 8 Atoms | 1 |

`beta=1.0` 时，一个含 `n` 个 Atom 的候选区域只有在 `D(u) <= beta*(n-1)` 且 compiler 接受时，才会优于拆成 `n` 个 singleton。更重要的是，如果一个 physical leaf 被 compiler 判为 `split_required`，当前 augmented tree 的下一层就是单 Atom terminals，没有 leaf 内部的中间 subset candidate。因此 cut 可以从 5 到 8 个 Atom 的 region 直接退化为 5 到 8 个 singleton。当前 147 个 singleton 表明这不是边缘情况，而是主导行为。

### 4.4 Skill 文本规模与示例质量

153 个 `SKILL.md` 的 token cost：

| 统计 | Tokens |
|---|---:|
| Minimum | 341 |
| Median | 577 |
| P90 | 729 |
| Maximum | 1,501 |
| Total | 92,016 |

多 Atom skill 示例 `cc_skill_0015` 是一个结构较好的 “Spreadsheet Row Deletion Skill”，覆盖 5 条 row deletion evidence，并将不同适用条件编译成 5 条 conditional rules。这个例子证明 compiler 能生成条件化 skill。

但多数 active skill 是单 Atom 改写。例如高频命中的 `cc_skill_0185` 提出 “Use a static lookup dictionary instead of VLOOKUP formulas”。这种来自单轨迹的局部策略容易在不同 workbook 上产生负迁移，且不具备多 evidence 支撑。

## 5. Heldout 结果

### 5.1 主指标

| 指标 | 结果 |
|---|---:|
| Heldout denominator | 200 |
| LibreOffice recalc correct | 72 |
| Instance accuracy | **36.0%** |
| Testcase accuracy | **36.0%** |
| Soft / Hard | 36.0% / 36.0% |
| Raw cached-value audit | 67/200 = 33.5% |

verified `200:400` 中每个 instance 只有一个 testcase，因此本 run 的 Soft、Hard、instance accuracy 和 testcase accuracy 数值相同。

LibreOffice 改变了 7 条 raw 判定：6 条 `raw false -> recalc true`，1 条 `raw true -> recalc false`。因此 36.0% 必须作为主结果，33.5% 只能作为 cached-value audit。

### 5.2 Runtime 与 correctness 分解

| Heldout 结果层 | 数量 |
|---|---:|
| Agent runtime success | 146 |
| Max turns exceeded | 50 |
| Runtime invalid timeout | 4 |
| Recalc correct | 72 |
| Value/content mismatch | 71 |
| Missing output workbook | 55 |
| Worksheet missing | 2 |

成功生成 workbook 的 146 条中有 72 条最终正确，条件正确率为 `72/146 = 49.3%`。但正式指标必须保留全部 200 条分母；不能删掉 max-turn 或 timeout task。

50 条 max-turn failure 的 trace 中存在重复 action 和无进展循环。4 条 timeout 是 1200 秒单请求 timeout，并且没有自动重发。运行后服务 health 仍为 200，因此不能把整个 36.0% 简单归因于服务中断，但 timeout 本身仍属于 runtime confounder，而不是可归因于 workbook reasoning 的模型错误。

### 5.3 Retrieval 使用情况

| 指标 | 结果 |
|---|---:|
| Query 数 | 200 |
| Top-k | 1 |
| Unique selected skills | 48 |
| Singleton skill selections | 195 |
| Multi-Atom skill selections | 5 |
| Multi-Atom selection accuracy | 1/5 = 20.0% |
| Singleton selection accuracy | 71/195 = 36.4% |

只有两个多 Atom skills 在 heldout 被选中过：`cc_skill_0080` 被选 2 次且 0 次正确，`cc_skill_0170` 被选 3 次且 1 次正确。其余 4 个多 Atom active skills从未被检索。

公开 reference mirror 被 bubblewrap 以只读方式挂载，完整 `SKILL.md` 也给出了对应 reference 路径。但冻结的 SpreadsheetBench system prompt 同时要求 agent 不访问当前 task directory 以外的文件。真实 200 条 trajectory 中没有发现访问 reference mirror 的 action。因此，本 run 只验证了完整 `SKILL.md` 注入，没有验证“agent 按需读取 references”会工作或有收益；这也是一处需要在下一协议版本中消除的 prompt contract 冲突。

按 source evidence outcome 分组：

| Selected skill evidence | Correct / Selected | Accuracy |
|---|---:|---:|
| 全 failure-source | 36/114 | 31.6% |
| 全 success-source | 35/81 | 43.2% |
| Mixed-source | 1/5 | 20.0% |

相似度四分位准确率约为 40%、36%、28%、40%，没有单调关系。也就是说，当前 boundary embedding 相似度不能稳定区分“这个 skill 会帮助任务”与“这个 skill 只是文字上相似”。

## 6. Token 与时间成本

### 6.1 正式 r3 build

| 指标 | 结果 |
|---|---:|
| Build wall time | 1,514.6 秒，约 25.2 分钟 |
| Logged LLM requests | 184 |
| Prompt tokens | 1,258,097 |
| Completion tokens | 232,631 |
| Total tokens | 1,490,728 |
| Max single prompt | 45,543 |
| Max single completion | 66,877 |
| Max total context | 100,000 |

这不是完整 Atom extraction 成本，因为 r3 合法复用了 r2 的 Atom drafts。r2 的 usage log 还混合了旧 compiler 调用，因此不能从现有通用 usage log 精确拆出 Atom extraction-only token 成本。报告不能把 1.49M 误称为从原始轨迹开始的完整建树成本。

### 6.2 Heldout

| 指标 | 结果 |
|---|---:|
| Rollout wall time | 5,471.4 秒，约 91.2 分钟 |
| LibreOffice evaluation | 263.3 秒，约 4.4 分钟 |
| LLM requests | 2,978 |
| Prompt tokens | 24,810,730 |
| Completion tokens | 2,783,035 |
| Total tokens | 27,593,765 |
| Max single prompt | 49,086 |
| Max single completion | 96,085 |
| Max total context | 100,000 |

所有正式 heldout 请求的 temperature 均为 0.0，`max_tokens` 均为 null。极端 completion 接近 100k，说明 no-thinking 并不能自动防止 runaway generation。这是 agent/runtime 层的成本风险。

## 7. 与历史结果的关系

| Artifact | Recalc | 与当前差值 | 可比性 |
|---|---:|---:|---|
| Contract-Cut EBST v1 | 72/200 = 36.0% | 0 | 当前正式结果 |
| 历史 vanilla no-skill | 82/200 = 41.0% | -5.0pp | 同 heldout 和 evaluator，但不是本分支新跑的 paired control |
| 历史 EBST static | 93/200 = 46.5% | -10.5pp | 同模型/no-thinking，但 top-10、workers16，协议非严格控制 |
| 历史 EBST dynamic | 88/200 = 44.0% | -8.0pp | 同样存在 retrieval 与并发差异 |
| 旧 GMM static | 97/200 = 48.5% | -12.5pp | thinking 和方法协议差异更大，不可作因果对比 |

当前与历史 vanilla 的 task-level交集为：both correct 48、Contract-Cut only 24、vanilla only 34、both wrong 94，净少 10 条正确。这个对比提示负迁移，但由于 vanilla 不是本次同进程 paired rerun，只能作为诊断证据，不能作为严格显著性结论。

历史 artifact 身份：

- Vanilla 41.0%: `/mnt/data/yaodong/codes/DynaMix2skill/runs/version_snapshots/20260710_spreadsheet_awq_retrain200_static_heldout97/vanilla_heldout_eval.json`, SHA256 `4bb046583779bbb829f6af06bc60ccab24556ce5429e6f0c91d453c29a273d74`。
- EBST static 46.5%: `/mnt/data/yaodong/codes/DynaMix2skill/runs/ebst_v4_strict_online_yd5_20260729_103548/01_open_loop/scenarios/static_build/trace2skill_heldout_eval.json`, SHA256 `2cb0bca002d8aba1b97492d19105f2e9672c7cc0b1d3ad8af9b20dacca103a19`。
- EBST dynamic 44.0%: `/mnt/data/yaodong/codes/DynaMix2skill/runs/ebst_v4_strict_online_yd5_20260729_103548/01_open_loop/scenarios/dynamic_update/trace2skill_heldout_eval.json`, SHA256 `d5bbbe8856d0674f34361373f56b5174ea13900f0334eb143b7be1f6ae303447`。
- 旧 GMM static 48.5%: `/mnt/data/yaodong/codes/DynaMix2skill/runs/version_snapshots/20260710_spreadsheet_awq_retrain200_static_heldout97/trace2skill_heldout_eval.json`, SHA256 `d980cc6ee8f99f153a0bfb0dba35fe554d39121ad5882d91e95cfc722e010ce7`。

## 8. Case study

### 8.1 Skill 注入后循环，vanilla 正确

- Task `54144`
- Contract-Cut 选择 singleton `cc_skill_0002`，score 0.6225。
- Agent 重复动作并耗尽 30 turns，最终没有 output workbook。
- 历史 vanilla 对该 task 正确。

这说明 top-1 singleton skill 不只可能无帮助，还可能强化错误执行路径。

### 8.2 Workbook 生成成功但结果错误

- Task `52807`
- Contract-Cut 选择 singleton `cc_skill_0133`，score 0.5787。
- Runtime success，但 LibreOffice 重算后 `H6` expected `1000`，got `0`。
- 历史 vanilla 正确。

这类失败不能由 timeout 或 missing output 解释，属于检索经验与任务执行没有形成正确迁移。

### 8.3 Contract-Cut 独有正确案例

- Task `52917`
- Contract-Cut 选择 singleton `cc_skill_0162`，score 0.7234。
- Contract-Cut 正确，历史 vanilla 日期处理错误。

这说明经验库中确实存在有帮助的局部知识，但当前 index 无法稳定选择这种知识。

### 8.4 多 Atom skill 未显示优势

- `cc_skill_0080` 在 task `3002` 上得到 Single/Multiple 错误，历史 vanilla 也错。
- `cc_skill_0170` 在 task `32902` 上导致错误 bonus，历史 vanilla 正确。
- `cc_skill_0170` 在 task `54513` 上正确，但历史 vanilla 同样正确。

5 次多 Atom命中不足以估计真实效果，但至少当前样本没有显示 compiler-generated grouped skill 的明显优势。

## 9. 理论保证的准确边界

本 run 有 artifact 支持的保证：

1. 单父、唯一 Atom placement。
2. `M=8,m=4` 下的 occupancy 与等深叶子。
3. 每次 9-entry overflow 在指定 126 partitions 上的精确局部 split。
4. 每个物理节点 exact cover radius。
5. augmented candidate tree 上 exact Atom cover。
6. 在固定 candidate tree、固定 `beta`、固定 compiler feasibility cache 下的精确最优 cut。
7. Dataset-order batch insertion 和 batch-end compilation 的审计可复现性。

本 run 不支持的保证：

1. Embedding 距离对应真实 Skill 边界。
2. Contract compiler 会产生最优或有帮助的抽象。
3. `beta=1.0` 对该数据分布最优。
4. Top-1 相似度等价于因果帮助。
5. Contract-Cut 比 vanilla、GMM tree 或旧 EBST 泛化更好。

## 10. 研究判断与下一步

当前版本不应继续靠微调 `beta` 或 top-k 刷 heldout。最主要的问题是 method identity 层面的：物理树成功了，但 skill boundary、compiler feasibility 和 retrieval utility 没有形成闭环。

下一步最小且可证伪的研究方向应是：

1. 保留本分支作为 frozen negative-result checkpoint，不覆盖现有最佳版本。
2. 先在 train-only 或独立 validation 上审计 20 条 Atom exclusion，区分真实泄漏和过严 literal matching，再决定是否修改 Atom policy。
3. 给 leaf 内部增加由同一几何目标产生的 nested subset candidates，避免“leaf contract 不可行就直接全 singleton”的离散退化。这个改变需要重新定义 candidate family 和最优 cut claim，不能作为小修补静默加入 v1。
4. 将 skill selection 从纯文本相关性扩展为 train-only utility calibration，明确验证“选中的 skill 是否提高执行成功概率”。任何 calibration 都不能使用 heldout label。
5. 重新跑一个严格 paired no-skill control，冻结相同模型、endpoint、workers8、no-thinking、temperature0、max-turn30、timeout、evaluator 和 task order，才能对 Contract-Cut 的净贡献做正式判断。

## 11. 可审计 artifacts

- Algorithm contract: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/research/contract_cut_ebst/ALGORITHM_CONTRACT.md`
- Experiment protocol: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/experiment_protocol.json`
- Accepted/excluded Atoms: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/contract_cut/contract_atoms.json`
- Tree state/audit: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/contract_cut/tree_state.json`
- Tree audit: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/contract_cut/tree_audit.json`
- Skill registry: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/contract_cut/skill_registry.json`
- Exported skills: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/skills`
- Retrieval selections: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/heldout/skill_selections.jsonl`
- Rollout ledger: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/heldout/rollout_results.jsonl`
- LibreOffice evaluation: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/heldout/libreoffice_eval.json`
- Build usage: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/logs/build_usage.jsonl`
- Heldout usage: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2/logs/heldout_usage.jsonl`
- Post-run identity/leakage audit: `/home/yaodong/codes/DynaMix2skill_contract_cut_ebst/research/contract_cut_ebst/FORMAL_RUN_AUDIT_20260811.json`

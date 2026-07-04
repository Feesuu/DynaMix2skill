# OfficeQA DynaMix Prompt-Fix TODO And Context

更新时间：2026-07-05

## 当前任务结论

当前 DynaMix2skill 的 OfficeQA 路线处在 `new prompt` 版本：我们已经把 OfficeQA analyst prompt 改成更严格的“公开可复用经验卡”抽取方式，核心目标是阻止训练集诊断信息泄漏进 nodebank experience。

本轮代码已经修改，但重新挂 OfficeQA DynaMix 建树时没有跑完。失败原因不是 prompt 泄漏复发，而是远端 Qwen3.5-9B-AWQ 服务在 LLM analyst 阶段返回 `503 no healthy upstream`，并且日志显示存在一次远端实际 token 长度超限警告。

## 这次 Prompt 改了什么

修改文件：

- `src/dynamix_trace2skill/summary.py`

新增/调整的 OfficeQA prompt 语义：

1. `predicted_answer`、`ground_truth`、`fail_reason`、verifier/evaluator details、reward audit fields 只能作为 private diagnostic inputs。
2. LLM 可以用这些 train-only 诊断字段在内部判断错误类型和根因，但输出的 ExperienceCard 必须是 public reusable guidance。
3. ExperienceCard 的 `name`、`trigger`、`content`、`placement` 不能包含：
   - 训练集 exact answer
   - predicted answer
   - gold / ground truth answer
   - fail_reason 标签
   - verifier/evaluator 字段名
   - 从轨迹里复制出来的 task-specific literal
4. 如果经验依赖某个具体诊断值，必须替换成抽象占位符，例如：
   - `<computed_value>`
   - `<expected_value>`
   - `<requested_precision>`
   - `<source_period>`
   - `<target_period>`
   - `<unit>`
   - `<metric>`
5. 新增“禁止具体例子数字”的硬规则：不得把输入 members 里的具体百分比、小数、年份、answer string、bracketed values、vectors 等抄进经验卡。
6. OfficeQA dynamic prompt 也加了同样的 private diagnostic / numeric-copy 约束，但只在 `task_profile == "officeqa"` 时追加，避免污染 SpreadsheetBench prompt。

关键代码位置可用如下命令查看：

```bash
rg -n "private diagnostic|concrete numeric|officeqa_experience_policy|task_profile" \
  /mnt/data/yaodong/codes/DynaMix2skill/src/dynamix_trace2skill/summary.py
```

## 为什么要改这个 Prompt

上一轮 OfficeQA DynaMix 建树时，nodebank audit 发现经验卡泄漏训练集诊断信息，典型包括：

- `ground truth (...)`
- `answer_mismatch`
- 训练集中的具体百分比、小数、向量、答案数字

这说明原 prompt 虽然允许 analyst 看 train rollout 的 `predicted_answer` 和 `ground_truth` 来做错误分析，但没有足够强地约束输出不能暴露这些信息。

正确目标是：

- train diagnostic 可以帮助经验抽取；
- 经验卡必须能用于 heldout；
- heldout system prompt 里不能出现训练样本答案、gold label、verifier 字段和具体训练数字。

## 已做验证

本轮 prompt 修改后已做过：

```bash
/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python \
  -m py_compile \
  /mnt/data/yaodong/codes/DynaMix2skill/src/dynamix_trace2skill/summary.py
```

结果：通过。

```bash
cd /mnt/data/yaodong/codes/DynaMix2skill
/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python \
  -m pytest tests/test_officeqa_integration.py -q
```

结果：`34 passed`。

```bash
git -C /mnt/data/yaodong/codes/DynaMix2skill diff --check -- src/dynamix_trace2skill/summary.py
```

结果：通过。

还用 prompt smoke 确认了 OfficeQA static/dynamic system prompt 都包含：

- `private diagnostic inputs`
- `Do not use concrete numeric examples copied from input members`

## Independent Review 状态

本轮用 3 个 independent-code-review subagents 审过 prompt 修改。

结论：

- spec reviewer：通过，只有提示 `task_profile` 必须由 OfficeQA runner 设置为 `officeqa`。
- regression reviewer：最初发现 dynamic prompt 的 OfficeQA 约束一度无条件加进了 generic hard constraints，可能污染 SpreadsheetBench；已修复为只在 `task_profile == "officeqa"` 时追加，复核通过。
- research-protocol reviewer：对当前整个 dirty worktree 给出 blocked，因为工作区里还有之前遗留的 `tree_builder.py`、`skillbank.py`、`pipeline.py` 等非 prompt 改动。这个 finding 不是本轮 prompt patch 新增的，但提醒当前仓库状态不干净，后续提交前必须分清楚哪些改动属于 OfficeQA，哪些属于之前实验遗留。

## 重新挂的 Run

新 run 目录：

```text
/mnt/data/yaodong/codes/DynaMix2skill/runs/officeqa_dynamix_promptfix_rebuild_16_http_20260704_232713
```

日志：

```text
/mnt/data/yaodong/codes/DynaMix2skill/runs/officeqa_dynamix_promptfix_rebuild_16_http_20260704_232713/logs/officeqa_dynamix_wrapper.log
```

这次 run 没有重跑 train rollout，而是复用上一轮已经生成的 train records：

```text
/mnt/data/yaodong/codes/DynaMix2skill/runs/officeqa_dynamix_diag_newprompt_16_http_20260704_220541/records.json
```

原因：本轮只改 analyst prompt，train rollout 本身没有变，直接从已有 `records.json` 重新建树可以省时间。

## 本轮 Run 的关键配置

从 `tree/analysis/runtime_config.json` 读取到的关键参数：

- `task_profile`: `officeqa`
- LLM base URL: `http://asmiatbrqksz.10.27.127.9.nip.io/v1`
- LLM model: `Qwen3.5-9B-AWQ`
- generation temperature: `0.6`
- generation timeout: `1200`
- generation max concurrency: `16`
- thinking: `true`
- embedding base URL: `http://10.26.1.184:18007/v1`
- embedding model: `Qwen3-Embedding-8B`
- embedding max model len: `32000`
- chunked embedding: enabled
- chunk tokens: `28000`
- chunk overlap: `1000`
- embedding pooling: `mean`
- summary max model tokens: `100000`
- summary budget ratio: `0.85`
- analyst prompt budget: `85000`
- member evidence effective budget: `77000`
- GMM min split size: `4`
- GMM min effective samples per component: `4`
- heldout split: `test`
- skillbank top-k: `10`

## 已发生的失败

建树流程到 LLM analyst 阶段失败。

日志中出现：

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (143543 > 131072).
```

随后大量请求返回：

```text
openai.InternalServerError: no healthy upstream
```

远端接口健康检查也返回：

```text
HTTP/2 503
no healthy upstream
```

并且 SSH 只读检查远端时返回过：

```text
devbox ns-sicij3kv/yaodong-test is not running
```

所以当前 blocker 是远端 LLM 服务不可用，不是本地 embedding 服务或 prompt patch 本身失败。

## Token 相关发现

本地 token report：

```text
/mnt/data/yaodong/codes/DynaMix2skill/runs/officeqa_dynamix_promptfix_rebuild_16_http_20260704_232713/tree/analysis/cluster_prompt_token_report.json
```

统计：

- prompt token events: `37`
- 最大本地估算 prompt tokens: `84674`
- 本地 budget: `85000`

最大的一些社区：

```text
L0_C2_R003: 84674 tokens, member_count=4
L0_C3_R000: 84467 tokens, member_count=4
L0_C15_R002: 84290 tokens, member_count=2
L0_C15_R005: 84103 tokens, member_count=3
L0_C0_R001: 82319 tokens, member_count=2
```

但是远端日志出现 `143543 > 131072`，说明存在一个重要风险：

- 本地 analyst prompt budget 用的 tokenizer/估算口径可能低估了 Qwen3.5-9B-AWQ 的实际 chat prompt token。
- 当前 `analyst.tokenizer_model` 为 `null`，运行时似乎使用了 embedding tokenizer `/mnt/data/grouph_share/models/modelscope/models/Qwen/Qwen3-Embedding-8B` 来做 token report。
- 这个口径和远端 Qwen3.5-9B-AWQ 的 chat template / tokenizer 不一定一致。

## 当前服务状态

截至 2026-07-05 检查：

本地 embedding 服务仍在：

```text
tmux: qwen3_embedding_8b_gpu7_18007_32k_eager
endpoint: http://10.26.1.184:18007/v1
model: Qwen3-Embedding-8B
```

OfficeQA DynaMix run tmux 已退出。

远端 Qwen3.5-9B-AWQ 服务仍不可用：

```bash
curl --noproxy '*' -k -sS -m 20 -i \
  https://asmiatbrqksz.10.27.127.9.nip.io/v1/models \
  -H 'Authorization: Bearer <key>'
```

返回：

```text
HTTP/2 503
no healthy upstream
```

## 下一步 TODO

### P0：等远端 LLM 服务恢复

先确认：

```bash
curl --noproxy '*' -k -sS -m 20 -i \
  https://asmiatbrqksz.10.27.127.9.nip.io/v1/models \
  -H 'Authorization: Bearer <key>'
```

或：

```bash
curl --noproxy '*' -sS -m 20 -i \
  http://asmiatbrqksz.10.27.127.9.nip.io/v1/models \
  -H 'Authorization: Bearer <key>'
```

必须返回 200 后再继续。

### P0：不要直接 16 并发重打

上一轮 16 并发 analyst 请求触发大量 `503 no healthy upstream`。远端恢复后建议先用更保守方式重跑：

- analyst/generation concurrency 先降到 `4` 或 `8`
- summary budget ratio 先降到 `0.75` 或更低，避免实际 chat tokens 超限
- 或显式设置正确的 Qwen3.5 analyst tokenizer，而不是用 embedding tokenizer 估算

### P1：修 token 预算口径

需要确认代码里 analyst prompt token report 使用的 tokenizer。

重点检查：

- `src/dynamix_trace2skill/pipeline.py`
- `src/dynamix_trace2skill/summary.py`
- `src/dynamix_core/tree_builder.py`
- `scripts/run_officeqa_dynamix_experiment.py`

目标：

- analyst prompt 预算必须用和 generation model 对齐的 tokenizer / chat template 口径；
- 或者保守地降低 budget，给 chat template / thinking / system prompt 留更大余量；
- 不要只看 `cluster_prompt_token_report.json` 里的 embedding-tokenizer 估算。

### P1：远端恢复后重新建树并观察 nodebank audit

重新 run 时仍可复用：

```text
/mnt/data/yaodong/codes/DynaMix2skill/runs/officeqa_dynamix_diag_newprompt_16_http_20260704_220541/records.json
```

因为 train rollout 没变。

重点观察：

- build tree 是否完成
- nodebank audit 是否还发现 train diagnostic leakage
- `officeqa_nodebank_diagnostic_audit.json` 是否为 clean
- heldout 是否能自动继续跑 test split

### P1：如果 audit 仍失败，再加输出后处理/repair

当前只改了 prompt。Prompt 不能 100% 保证模型不泄漏训练数字。

如果新 prompt 后 audit 仍发现泄漏，应考虑增加一个 deterministic sanitizer / repair gate：

- 对 ExperienceCard 的 `name/trigger/content/placement` 做 train diagnostic pattern audit；
- 命中 exact train answer、gold answer、predicted answer、fail_reason、明显数字泄漏时，要求 LLM repair；
- repair 仍失败则跳过该 card 或标记 diagnostic unsafe，不进入 nodebank。

这个属于下一步代码修改，不是本轮已做内容。

## 唤醒这个任务时应该先做什么

1. 先检查远端 Qwen3.5-9B-AWQ 是否恢复。
2. 再检查当前 dirty worktree，避免把 prompt patch 和旧的 tree/skillbank/pipeline 改动混在一起提交。
3. 复查 `summary.py` 的 OfficeQA prompt 是否还包含 private diagnostic 和 no concrete numeric examples 规则。
4. 用较低并发/更保守 token budget 从已有 `records.json` 重新建树。
5. 如果建树完成，立刻检查 nodebank diagnostic audit。

## 当前不应声称的结论

不能说 new prompt 已经解决了 OfficeQA leakage，因为这一轮没有跑到 nodebank audit 完成。

目前只能说：

- prompt 规则已经加入；
- 单元测试和 prompt smoke 通过；
- 重新建树被远端 LLM 服务 503 阻断；
- 还有 token 预算口径可能低估的风险，需要下轮优先处理。

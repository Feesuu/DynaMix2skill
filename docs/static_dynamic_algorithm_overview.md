# DynaMix 静态与动态算法流程概览

本文档记录当前代码版本中 DynaMix2skill 的两个主要场景：

- `static_build`：从已有训练轨迹一次性建树，再导出 nodebank。
- `dynamic_update`：先用一部分轨迹静态建初始树，再把后续轨迹按动态方式插入并更新树，最后导出 nodebank。

目标是给后续接手实验的 agent 或研究人员一个“不改变协议”的高层地图。本文档描述当前实现的大致过程，不把 future work 写成已实现功能。

## 共同组件

两个场景共享以下核心组件。

- 轨迹输入：由 `scripts/run_dynamix_trace2skill_experiment.py` 收集或读取 Trace2Skill/SpreadsheetBench 训练轨迹，形成 `records.json`。
- 轨迹表示：`src/dynamix_trace2skill/trace_views.py` 把每条轨迹渲染成用于 embedding 和 LLM 分析的文本。
- embedding：`EmbeddingClient` 调用 OpenAI-compatible embedding endpoint。长轨迹可走 `chunked_embedding`，把轨迹按 token chunk 切开、分别 embedding、再 mean pooling 成一条轨迹向量。
- 聚类：`ProjectedGmmTreeBuilder` 使用 local PCA + weighted GMM-BIC。
- 软分配：默认 `cumulative_mass`，并结合 `max_membership_gap` 控制多社区归属。
- 经验抽取：`ClusterAnalyst` 调 LLM，把一个 community 的成员总结为 `ExperienceCard`。
- nodebank 导出：`skill_export.py` 只导出可检索的 `ExperienceCard` 节点，不导出 L0 原始轨迹，也不导出 diagnostic/oversize 节点。
- heldout 检索：对 nodebank 做 embedding index，heldout 每个 task 用 query 检索 top-k 经验节点，把节点内容注入 system prompt。

当前 nodebank 的 embedding 文本只包含：

```text
name: ...
trigger: ...
content: ...
```

不会把 `level`、`support_mass`、`confidence`、`item_id`、`source_community_id` 等元信息放入 embedding 文本。

## 静态算法：`static_build`

静态场景对应代码入口：

- `build_tree_from_records()` in `src/dynamix_trace2skill/pipeline.py`
- `ProjectedGmmTreeBuilder` in `src/dynamix_core/tree_builder.py`

整体流程如下。

1. 读取训练轨迹。

   `DynaMixRunConfig` 指定 `records_path`、数据范围、模型 endpoint、embedding endpoint、树构建超参数等。代码会把轨迹加载成 `RawTrajectoryRecord`，再转成 `ExperienceItem(kind=trajectory, level=0)`。

2. 生成每条轨迹的 embedding。

   如果 `chunked_embedding.enabled=false`，每条轨迹直接作为一个文本请求 embedding。

   如果 `chunked_embedding.enabled=true`，一条长轨迹会被切成带 overlap 的多个 chunk，每个 chunk 分别 embedding，最后 mean pooling 成一个向量。这样可以避免 embedding 模型上下文长度不足导致长轨迹被硬截断。

3. 从 L0 开始逐层聚类。

   第 `level` 层的输入是当前层所有 `ExperienceItem`：

   - L0 输入是原始轨迹。
   - L1 输入是从 L0 community 抽取出的 ExperienceCard。
   - L2+ 输入是更低层 ExperienceCard 的抽象卡。

   每一层都会先检查输入是否为空、是否满足 `gmm_bic.min_split_size`。如果输入太少且不需要 L0 budget refinement，则停止。

4. 对当前层做 local PCA。

   代码先对输入 embedding 做 normalize，再用 local PCA 投影。默认设置是：

   - `projection.variance_ratio=0.90`
   - `projection.max_dim=32`
   - `projection.min_dim=2`
   - `projection.whiten=false`

   这里保存的 PCA mean/components 会进入该层的 `routing_model`，供动态更新阶段复用。

5. 用 weighted GMM-BIC 选择 K。

   对候选 K 拟合 GMM，用 BIC 选择最优 K。K 上限受 `abs_kmax` 和 `min_effective_samples_per_component` 限制。当前默认：

   - `gmm_bic.min_split_size=4`
   - `gmm_bic.min_effective_samples_per_component=2`
   - `gmm_bic.abs_kmax=64`

   如果 BIC 选择 `K <= 1`，且没有触发 L0 超预算 refinement，则该层停止，不强行生成人工 root card。

6. 用 soft membership 生成 overlapping communities。

   默认 `soft_membership.recursive_assignment=cumulative_mass`。对每个 item：

   - 根据 GMM posterior 从高到低排序。
   - 累积 posterior mass，直到达到 `cumulative_mass_coverage`。
   - 同时用 `max_membership_gap` 阻止把很弱的尾部 component 也纳入。

   当前默认：

   - `cumulative_mass_coverage=0.90`
   - `max_membership_gap=0.25`

   因此一个 item 可以属于多个 community，这也是当前树支持 multi-parent 的核心机制。

7. 保存静态 routing model。

   每一层正常 GMM-BIC 聚类后，会保存一份 `routing_model`：

   - `pca_mean`
   - `pca_components`
   - GMM `pi`
   - GMM `means`
   - GMM `variances`
   - `community_ids`
   - component effective counts
   - soft assignment 配置

   动态插入新 item 时会用这些参数把新 item 投影回同一个 PCA/GMM 空间，然后计算 posterior。

8. L0 超预算 community 的 refinement。

   只有 L0 原始轨迹层会应用 `budget_refinement`。原因是 L0 轨迹文本可能非常长，直接把一个大 community 输入 analyst 会超过 prompt budget。

   当前策略是：

   - 先保留正常 GMM-BIC 得到的 coarse community。
   - 如果某个 coarse community 的成员证据 token 超过 `summary_budget.effective_token_budget`，就对这个 community 内部递归做 local GMM-BIC refinement。
   - refinement 得到的可行 leaf 会 flatten 回 L0，作为最终 L0 community。
   - 如果 GMM 不能继续切，使用 token packing fallback。
   - packing 仍无法放下时拆成 singleton。
   - 单条轨迹本身仍超过同一个 prompt budget 时，才标记为 `excluded_oversize_singleton`，不送入 LLM 分析。

   注意：这里没有为 singleton 发明单独 budget。所有 community 都遵守同一个 analyst prompt budget：

   ```text
   effective_token_budget = summary_budget.max_model_tokens * summary_budget.budget_ratio - prompt_overhead_reserve_tokens
   ```

9. 调用 LLM analyst 生成 ExperienceCard。

   `ClusterAnalyst.summarize()` 使用 guided JSON schema 要求输出：

   - `name`
   - `trigger`
   - `content`
   - `placement`
   - `confidence`

   当前 cardinality policy：

   - L0 raw trajectory community 可以输出一张或多张卡。
   - L1+ ExperienceCard community 必须压缩成一张更高层抽象卡。

   LLM 不输出 `support_mass`。`support_mass` 由 state 根据 community 和 card confidence 重新分配。

10. 提交本层并进入下一层。

   生成的 ExperienceCard 会成为下一层输入。静态 build 会一直向上构建，直到：

   - 没有下一层 item。
   - 某层 BIC 选择 `K<=1`。
   - 某层输入太少。
   - 某层无法生成新的 ExperienceCard。
   - 达到 `max_levels`。

11. 导出 nodebank。

   静态树建完后，`export_skill_files()` 会把所有可导出的 `ExperienceCard` 节点写入：

   ```text
   <run_dir>/skills/node_bank_manifest.json
   ```

   导出过滤规则：

   - 只导出 `kind == experience_card`。
   - 不导出 L0 raw trajectory。
   - 不导出 `llm_summary_skipped`、`oversize_singleton` 等 diagnostic 节点。
   - 不生成 legacy `SKILL.md` 文件夹语义。

12. 构建 nodebank embedding index 并跑 heldout。

   `build_skillbank_index.py` 或 pipeline 内部刷新 nodebank index。heldout 阶段每个 task 会检索 top-k 相关 ExperienceCard 节点，并把这些经验片段注入 agent system prompt。

## 动态算法：`dynamic_update`

动态场景对应代码入口：

- `build_dynamic_tree_from_records()` in `src/dynamix_trace2skill/pipeline.py`
- `ExperienceHierarchyDynamicUpdater` in `src/dynamix_core/update.py`

动态场景的目标不是重新全量建树，而是模拟训练轨迹逐步到来时如何维护已有经验树。

### 数据划分

当前 dynamic config 使用：

- `dynamic.initial_count`：初始静态建树使用多少条轨迹，默认 120。
- `dynamic.arrival_count`：后续动态插入多少条轨迹，默认 80。
- `dynamic.update_batch_size`：每轮 admission 后按层并发 LLM 更新的批大小，默认 8。
- `dynamic.shuffle_seed`：arrival 轨迹是否可复现随机打乱，默认 42；如果设为 `null`，则保持输入顺序。

流程上：

1. 前 `initial_count` 条轨迹先按静态算法建一棵初始树。
2. 后续 `arrival_count` 条轨迹作为动态到来的输入。
3. 如果配置了 `shuffle_seed`，arrival 轨迹会被可复现打乱。
4. arrival 轨迹按 `update_batch_size` 切成若干 update batch。

注意：当前 batch 不是“一起拟合一个新 GMM”。batch 内轨迹 admission 仍然是一条一条顺序插入；batch 的作用是攒一批 affected communities 后，按层做 LLM 并发更新，从而提高吞吐。

### 单条轨迹 admission

对每条新轨迹，动态 updater 执行以下步骤。

1. 检查轨迹 embedding 和 token budget。

   如果这条轨迹作为单独 L0 community 时，动态 analyst prompt 也会超过同一个 prompt budget，则它被标记为 `excluded_oversize_singleton`，不会插入树中。

2. 用已有 L0 routing model 路由。

   新轨迹先用 L0 保存的 PCA/GMM routing model 计算 posterior：

   - 使用静态 build 保存的 `pca_mean` 和 `pca_components` 投影。
   - 使用保存的 GMM `pi/means/variances` 计算 posterior。
   - 使用 dynamic soft membership 配置选择候选 community。

   dynamic soft membership 默认也使用：

   - `assignment=cumulative_mass`
   - `cumulative_mass_coverage=0.90`
   - `max_membership_gap=0.25`

3. 如果 L0 曾发生 budget refinement，则继续路由到 refinement leaf。

   静态 build 对 L0 超长 community 做过 refinement 时，会保存 `refinement_routing_tree`。动态路由会先到 coarse community，再沿 refinement tree 路由到 active leaf community。

4. 对候选 L0 communities 做 token budget gate。

   新轨迹可能被 soft membership 选到多个候选 community。动态 updater 会逐个检查：

   - 如果把新轨迹加入候选 community 后，动态 analyst prompt 不超过 budget，则该候选可接受。
   - 如果某个候选加入后会超过 budget，则该候选被拒绝。

   当前实现允许多社区插入：所有通过 budget gate 的候选都保留。这样保留了静态 build 的 multi-parent 语义。

5. 如果所有候选都超预算，则 growing-K 新建 L0 dynamic community。

   如果新轨迹自己不超 budget，但所有候选 community 加上它都会超 budget，则新建一个 L0 dynamic community：

   ```text
   L0_DYN_...
   ```

   这个新 community 是一个新的 GMM component，表示“这条轨迹不能安全插入已有 L0 community，但它本身可以被 analyst 分析”。它不会强行污染已有超预算 community。

6. 更新 L0 state 和 routing model。

   新轨迹插入后，state 会更新它的 selected membership 和 posterior membership。若创建了新的 L0 dynamic community，则 routing model 也会 append 一个新 component。

   当前 routing model 的在线更新使用 remove/add sufficient statistics：

   - 新 item：添加它对各 component 的贡献。
   - 已存在 card 被更新：先移除旧贡献，再加入新贡献。
   - 更新 `pi/means/variances/component_effective_counts`。

   这样下一条动态轨迹会看到已经更新过的 routing model。

### Batch 后的 LLM 更新与向上传播

一个 update batch 内的所有轨迹完成 admission 后，updater 会把受影响的 community 作为起点，向上更新 ExperienceCard。

1. 收集 affected L0 communities。

   新轨迹插入到哪些 L0 community，哪些 L0 community 就需要重新分析。

2. 同一层 community 并发调用 dynamic analyst。

   `_propagate_affected_communities()` 会按 level 串行推进，但同一个 level 内多个 community 会用 `asyncio.gather` 并发调用。

3. L0 dynamic patch 允许 updates 和 new_cards。

   L0 的成员是 raw trajectories。动态 prompt 会让 LLM：

   - 用 `updates` 按明确 `item_id` 修改已有 ExperienceCard。
   - 用 `new_cards` 增加新发现的独立经验卡。
   - 不允许按位置、confidence rank 或列表顺序去覆盖旧卡。

4. L1+ dynamic patch 只允许 updates。

   L1+ 的成员已经是 ExperienceCard。当前设计不在 L1+ 直接新增 sibling card，而是只允许对已有高层 abstraction card 做 update。

   代码层面使用 update-only schema：

   ```json
   {
     "updates": [
       {
         "item_id": "...",
         "name": "...",
         "trigger": "...",
         "content": "...",
         "placement": {"target": "...", "reference_kind": "..."},
         "confidence": 0.8
       }
     ]
   }
   ```

   如果 L1+ LLM 输出违反 schema，代码会做有限 repair retry；最终仍无效时记录并返回空 patch，不让整个 dynamic run 直接崩掉。

5. changed cards 继续向上 reroute。

   被 update 或 add 的 ExperienceCard 会变成其所在 level 的 changed item。它们会被路由到上一层的 community：

   - 使用对应 level 的 routing model。
   - 使用 dynamic cumulative mass soft membership。
   - 重新计算 affected parent communities。
   - 对 affected parent communities 再做 dynamic analyst update。

   这个过程一直传播到没有 affected community，或到达顶层 terminal level。

6. support mass 跟随 state 重新分配。

   LLM patch 不直接给 `support_mass`。当一个 community 的 generated cards 被更新或新增时，state 会基于 community support 和 card confidence 重新分配 support mass。未变化的旧卡保留旧 confidence，并参与新的 support mass 分配。

7. 每个 batch 保存 snapshot。

   动态 pipeline 会在：

   ```text
   <run_dir>/dynamic_snapshots/batch_XXX/
   ```

   写入：

   - `hierarchy_state.json`
   - 当前 batch 的 nodebank manifest
   - skillbank index
   - `snapshot_meta.json`

   最终完整动态树写到 run dir 顶层。

### 动态场景与静态场景的关键差异

静态 build 的 L0 超预算策略：

- 对超预算 coarse community 递归 GMM-BIC refinement。
- 如果 GMM 不能继续切，则 token packing。
- packing 不行则 singleton。
- singleton 仍超过同一个 prompt budget 才 exclude。

动态 update 的 L0 超预算策略：

- 不拆已有 community。
- 新轨迹先路由到候选 community。
- 对每个候选检查“加入新轨迹后的 prompt 是否超预算”。
- 能放下的候选都保留，保持 multi-parent。
- 所有候选都放不下，但新轨迹自己能放下，则 growing-K 新建 L0 dynamic community。
- 新轨迹自己也放不下，才 exclude。

这个差异是有意设计的：动态插入时拆已有 community 会牵动大量 parent/subtree 结构，当前版本选择用 growing-K 和在线 routing model 更新来保持局部、可审计的动态维护。

## Heldout 使用方式

不管静态还是动态，最终都导出 nodebank：

```text
<scenario_output_dir>/skills/node_bank_manifest.json
<scenario_output_dir>/skills/node_bank_index.json
```

heldout 阶段：

1. 用 task query 对 nodebank 做 embedding 检索。
2. 取 top-k ExperienceCard 节点。
3. 把这些节点的 `prompt_text` 注入 agent system prompt。
4. agent 继续用 SpreadsheetBench/Trace2Skill 风格 rollout 解题。
5. 评测侧应使用 LibreOffice recalc 后的 evaluator，避免 openpyxl 写公式但 cached value 未刷新的误判。

当前文档只描述 nodebank 检索机制，不把旧的 `SKILL.md` folder export 作为正式路径。

## 当前重要超参数

常见主实验默认值在 `scripts/run_handoff_static_dynamic_experiment.sh` 和 `scripts/run_dynamix_trace2skill_experiment.py` 中暴露。

| 参数 | 当前含义 |
| --- | --- |
| `TREE_SCENARIO` | `static_build` 或 `dynamic_update` |
| `TRAIN_START/TRAIN_END` | 使用的训练轨迹范围 |
| `HELDOUT_START/HELDOUT_END` | heldout 评测范围 |
| `SKILLBANK_TOP_K` | heldout 注入的 nodebank top-k |
| `MAX_TURNS` | 每个 heldout rollout 最大步数 |
| `THINKING` | rollout/static analyst 是否开启 thinking；dynamic patch analyst 当前强制 `thinking=false` |
| `EMBEDDING_MAX_MODEL_LEN` | embedding 服务最大上下文 |
| `CHUNKED_EMBEDDING_ENABLED` | 是否对长轨迹做 chunked mean embedding |
| `GMM_MIN_SPLIT_SIZE` | GMM-BIC 聚类最小样本数，当前默认 4 |
| `GMM_MIN_EFFECTIVE_SAMPLES_PER_COMPONENT` | K 上界约束，当前默认 2 |
| `SOFT_CUMULATIVE_MASS_COVERAGE` | soft membership 累积质量阈值，当前默认 0.90 |
| `SOFT_MAX_MEMBERSHIP_GAP` | soft membership tail stop，当前默认 0.25 |
| `SUMMARY_MAX_MODEL_TOKENS` | analyst 模型上下文上限 |
| `SUMMARY_BUDGET_RATIO` | analyst prompt 预算比例，当前主实验常用 0.85 |
| `DYNAMIC_INITIAL_COUNT` | dynamic 初始静态树轨迹数，默认 120 |
| `DYNAMIC_ARRIVAL_COUNT` | dynamic 后续插入轨迹数，默认 80 |
| `DYNAMIC_UPDATE_BATCH_SIZE` | dynamic 每批 admission 后并发 summary 的大小，默认 8 |
| `DYNAMIC_SHUFFLE_SEED` | dynamic arrival 可复现打乱 seed，默认 42 |

## 明确不是当前版本做的事

- 当前版本不是 Trace2Skill 原始 iterative verifier loop；metadata 中也标记 `iterative_rca_loop=false`。
- 当前版本不导出旧语义的巨大 `SKILL.md` skill folder。
- 当前版本不在动态插入时 split 已有 L0 community。
- 当前版本不在 L1+ dynamic update 直接新增 new_cards。
- 当前版本不把 diagnostic/oversize/raw trajectory 节点放进 nodebank 检索。

# Static SpreadsheetBench Checkpoint: 2026-07-02 Chunk-28000 / MinSplit-4

## Verdict

This checkpoint is the strongest completed static SpreadsheetBench run currently recorded in this workspace.

- Official LibreOffice-recalculated heldout accuracy: **90/200 = 45.0%**
- Raw no-recalc audit accuracy: **68/200 = 34.0%**
- Heldout rollout completion: **198/200 successful harness runs**

## Run Identity

- Run directory: `/mnt/data/yaodong/codes/DynaMix2skill/runs/static_qwen35_awq_8bembed_chunk28000_minsplit4_after_budget_fix_20260702_142314/scenarios/static_build`
- Scenario: `static_build`
- Data path: `/mnt/data/yaodong/codes/DynaMix2skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400`
- Train records source: `/mnt/data/yaodong/codes/DynaMix2skill/runs/static_qwen35_awq_xml8_20260618_144604/records.json`
- Train range: `0..200`
- Heldout range: `200..400`
- Completion markers:
  - `stage_markers/04_build_tree.done`
  - `stage_markers/06_heldout_collect.done`
  - `stage_markers/07_heldout_eval.done`

## Model And Runtime

- Generation model: `Qwen3.5-9B-AWQ`
- Generation base URL used by run: `http://127.0.0.1:11802/v1`
- Thinking: `true`
- Workers: `8`
- Max turns: `30`
- Rollout temperature: `0.0`
- Generation temperature for analyst: `0.6`
- Python: `/home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python`

## Embedding And Tree Parameters

- Embedding model: `Qwen3-Embedding-8B`
- Embedding base URL used by run: `http://10.26.1.184:18007/v1`
- Embedding max model/input tokens: `32000`
- Chunked embedding: enabled
- Chunk tokens: `28000`
- Chunk overlap: `1000`
- Chunk pooling: `mean`
- GMM `min_split_size`: `4`
- GMM `min_effective_samples_per_component`: `2`
- Summary max model tokens: `100000`
- Summary budget ratio: `0.85`
- Skillbank top-k: `10`

## Tree Summary

- Records: `200`
- Total hierarchy items: `565`
- Communities: `238`
- Committed layers: `6`
- Layer records: `7`
- Exported nodebank nodes: `365`
- Nodebank format: `dynamix_node_skill_bank_v1`
- Nodebank artifact: `dynamix_tree/skills/node_bank_manifest.json`
- Skillbank index artifact: `dynamix_tree/skills/.dynamix_skillbank_index.json`
- Embedding truncation events: `0`

Layer progression:

| Level | Input items | Generated cards | Chosen K | Committed | Stop reason |
| --- | ---: | ---: | ---: | --- | --- |
| L0 | 200 | 257 | 6 | yes | |
| L1 | 257 | 56 | 56 | yes | |
| L2 | 56 | 28 | 28 | yes | |
| L3 | 28 | 14 | 14 | yes | |
| L4 | 14 | 7 | 7 | yes | |
| L5 | 7 | 3 | 3 | yes | |
| L6 | 3 | 0 | 1 | no | too_small |

## Chunked Embedding Summary

- Strategy: `sliding_window_chunk_mean`
- Text count: `200`
- Total chunks: `222`
- Records over model limit before chunking: `18`
- Maximum original token count: `90566`
- Maximum chunks for one record: `4`

## Heldout Evaluation

Evaluation artifact:

`trace2skill_heldout_eval.json`

Summary:

| Metric | Value |
| --- | ---: |
| Total instances | 200 |
| Fully correct instances | 90 |
| Instance accuracy | 0.45 |
| Total test cases | 200 |
| Passed test cases | 90 |
| Test-case accuracy | 0.45 |
| Raw passed test cases | 68 |
| Raw test-case accuracy | 0.34 |
| Average soft score | 0.45 |
| Average hard score | 0.45 |
| Official evaluation mode | `libreoffice_recalc` |
| Raw audit mode | `audit_only_no_recalc` |

LibreOffice recalculated outputs:

`trace2skill_heldout_outputs/eval_artifacts/libreoffice_recalculated_outputs`

## Known Runtime Note

During heldout, one request hit the generation backend context limit:

```text
This model's maximum context length is 100000 tokens.
However, you requested 0 output tokens and your prompt contains at least 100001 input tokens.
```

The run did not abort; the full heldout collect and final evaluation completed. This should remain a tracked risk for prompt budget / top-k retrieval tuning.

## Comparison To Immediately Previous Static Run

The preceding `chunk_tokens=8000 / overlap=1000 / min_split_size=4` run achieved:

- Official LibreOffice-recalculated heldout accuracy: `81/200 = 40.5%`

This checkpoint improves that to:

- Official LibreOffice-recalculated heldout accuracy: `90/200 = 45.0%`

## Commit Scope

This checkpoint records the current local code/config state together with this result note. Large run artifacts under `runs/` are not intended to be committed.

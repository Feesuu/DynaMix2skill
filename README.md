# DynaMix-HSPI + Pruned Trace2Skill Runtime

This is the audit-ready DynaMix-HSPI project. It includes:

1. DynaMix core hierarchy code.
2. DynaMix Trace2Skill bridge code.
3. Only the Trace2Skill runtime files actually reused by this project.

It does **not** include Trace2Skill's patch-merge implementation, released skills, upstream README, or unused experiment scripts.

## What this project does

```text
Trace2Skill SpreadsheetBench runner
-> ReAct trajectory logs
-> DynaMix trajectory normalization
-> Qwen3-Embedding-8B embeddings
-> weighted PCA/GMM-BIC hierarchy
-> cluster-level Trace2Skill-style ExperienceCards
-> skill folder export
-> top-k skillbank selection
-> Trace2Skill skill-preloaded heldout evaluation
```

## Main directories

```text
src/dynamix_core/          DynaMix hierarchy core
src/dynamix_trace2skill/   Trace2Skill bridge, prompt adaptation, skillbank selection
spreadsheet_agent/         pruned Trace2Skill spreadsheet agent runtime
src/react_agent/           pruned Trace2Skill ReAct/LLM/tool runtime
analysis/                  Trace2Skill prompt templates used by DynaMix analyst
scripts/                   DynaMix run/build/smoke scripts
configs/                   DynaMix configs
tests/                     DynaMix contract tests
```

## Files intentionally not included

See `docs/dynamix/PRUNED_TRACE2SKILL_REUSE_MANIFEST.md`.

## Setup

```bash
cd DynaMix_Trace2Skill_audit_ready_v1
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install openai numpy scikit-learn tqdm openpyxl pandas transformers pytest
```

## Smoke tests

```bash
pytest -q

python scripts/smoke_correctness_synthetic.py \
  --work-dir /tmp/dynamix_correctness_smoke
```

If you have `data/spreadsheetbench_verified/spreadsheetbench_verified_400`:

```bash
python scripts/smoke_real_data_trace2skill_dynamix.py \
  --data-path /path/to/spreadsheetbench_verified_400 \
  --limit 4 \
  --work-dir /tmp/dynamix_realdata_smoke
```

## Real Qwen small experiment

```bash
python scripts/run_dynamix_trace2skill_experiment.py \
  --data-path /path/to/spreadsheetbench_verified_400 \
  --run-dir runs/qwen_smoke_001 \
  --train-start 0 \
  --train-end 5 \
  --heldout-start 200 \
  --heldout-end 201 \
  --workers 1 \
  --model Qwen3.5-9B \
  --openai-base-url http://127.0.0.1:18002/v1 \
  --openai-api-key EMPTY \
  --embedding-base-url http://127.0.0.1:18000/v1 \
  --embedding-model Qwen3-Embedding-8B \
  --embedding-tokenizer Qwen3-Embedding-8B \
  --max-turns 100 \
  --thinking true \
  --skillbank-top-k 3
```

## Skillbank behavior

DynaMix may export multiple skill folders. At inference time, for each task:

1. Embed task query with Qwen3-Embedding-8B.
2. Embed all `SKILL.md` files in the skillbank.
3. Select top-k skills by cosine similarity, default top-k = 3.
4. Preload the selected `SKILL.md` content into Trace2Skill's normal skill-preloaded prompt.
5. Keep references/scripts in their fixed skill directories; no per-query copying.

This is concurrency-safe for local multi-worker runs because each task only reads the immutable skillbank.

## Notes for future remote/Docker runs

Current code assumes local path visibility. If a future remote/Docker runner is used, stage or mount the whole skillbank root once before the run, then rewrite the skillbank index paths to that visible root. Do not copy selected folders per query.

## Bundled mock data

This package includes a tiny synthetic SpreadsheetBench-compatible dataset under:

```text
data/spreadsheetbench_verified/spreadsheetbench_verified_400
```

It is only for smoke tests and layout checks. It is not the real benchmark.

For detailed Codex instructions, see [`CODEX_NEXT_STEPS.md`](CODEX_NEXT_STEPS.md).

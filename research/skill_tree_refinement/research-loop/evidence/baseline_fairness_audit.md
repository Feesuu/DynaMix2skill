# Fresh DynaMix Versus Recorded Vanilla: Fairness Audit

## Verdict

The two scores are valid records of their own runs, but they are not a fully
matched method comparison. Report `48.5%` and `41.0%` separately; do not claim
a causal `+7.5pp` DynaMix gain until vanilla is rerun under the current protocol.

## Shared Protocol

- Dataset: SpreadsheetBench verified heldout `[200,400)`, denominator 200.
- Model identity: `Qwen3.5-9B-AWQ`.
- Generation JSON: temperature 0 and thinking enabled.
- Agent limit: 30 turns.
- Primary evaluator: LibreOffice recalc.

## Mismatches and Runtime Outcomes

| Field | Fresh DynaMix | Recorded vanilla |
|---|---|---|
| Run | `spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319` | `spreadsheet_vanilla_heldout_qwen35_awq_20260708_170249` |
| Endpoint | `https://evirdwimyrmm.10.27.127.9.nip.io/v1` | `http://127.0.0.1:11802/v1` |
| Workers | 32 | 16 |
| Agent execution completed | 200/200 | 197/200 |
| Recalc score | 97/200 (48.5%) | 82/200 (41.0%) |
| Harness/runtime failures | recoverable stream retries | one 100k context rejection; two max-turn no-output cases |

Workers alone should not define model quality, but the endpoint/service and
execution-failure differences make the pair non-identical. The vanilla context
failure is task `32789`; max-turn no-output tasks are `37378` and `50051`.

## Required Matched Anchor

Rerun vanilla with no nodebank/skills while freezing the fresh run's endpoint,
model, generation config, workers, tools, 30-turn limit, timeout/retry policy,
dataset order, split, LibreOffice evaluator, and denominator. Report model
errors, max-turn failures, missing workbooks, retries, and final score separately.

## Claim Boundary

- Supported: the fresh DynaMix run scored 48.5%; the recorded vanilla run scored
  41.0% under their documented runtimes.
- Unsupported: DynaMix caused a 7.5pp gain under a strictly controlled protocol.
- Unaffected: build-only analysis of how minsplit changes tree topology and card
  quality, provided all build variables except minsplit remain frozen.

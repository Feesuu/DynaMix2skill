# Experiment Contract: IDEA-001 Global Split Granularity

Status: amended after independent review; build-only run blocked until the
post-fix review is recorded.

## Research Question

Does increasing `gmm_bic.min_split_size` from 4 to 8 produce a structurally
cleaner and cheaper skill tree without losing source coverage or increasing
card redundancy under the otherwise frozen fresh protocol?

This is not a pure depth ablation. The parameter jointly controls:

1. ordinary-layer `too_small` stopping;
2. whether `compute_kmax` returns 1 below the threshold; and
3. whether an over-budget L0 refinement node uses another local GMM split or
   the token-packing fallback.

All three effects are part of the intervention and must be reported.

## Fixed Data and Baseline

- Records: exact file
  `runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/ordered_records.json`.
- Records SHA256:
  `5b7657d5ebfdf9e37d29f20fc97958dc59de579aaef04958053db90780b70752`.
- Baseline tree: the same run's `dynamix_tree/`.
- Baseline heldout score: `97/200=48.5%`, treated as one development run.
- Recorded vanilla `82/200=41.0%` is contextual, not matched; its endpoint,
  workers, and execution completion differ.

## Source-State Preflight

Before launch, recompute the existing stage fingerprints and require exact
matches to the baseline marker:

| Source | Required SHA256 |
|---|---|
| runner | `28cab759903628244c4e1c3144ce14746a878345430cd3183a561f518d8a2b93` |
| build script | `55dcdfd31ffdf7eea854fd631b0ceb8b6ac31565da01a96c4aae4e90697f8eb9` |
| `src/dynamix_core` | `832abc0f080cf3d66e7abe9a7be9c8a3d1edd3d479333c11700cb6f2cc522a26` |
| `src/dynamix_trace2skill` | `dec37b2df56231f234c7d4b981f1ca9abc0c3005f782ecaec2c30165dc7e418d` |

These hashes were rechecked after review and currently match. If any differ,
do not launch; restore an isolated baseline source snapshot or amend the
contract. The working tree may remain dirty only if the required source hashes
still match and unrelated files are not used by the build.

## Allowed Config Diff

Copy the baseline `dynamix_config.json`. Only these fields may differ:

- `output_dir`;
- `generation.debug_dir`;
- `embedding.cache_path`, provided it starts as a byte-for-byte copy of the
  baseline cache; and
- `hierarchy.gmm_bic.min_split_size`, changed from 4 to 8.

No core code change is expected. Dataset order, model/endpoint, generation
temperature 0.6, thinking, concurrency 32, embedding model/config, random seed,
GMM-BIC settings, soft membership, token budget, prompts, max levels, and
nodebank export remain unchanged.

## Build-Only Metrics

All metrics are computed from `summary.json`, `hierarchy_state.json`,
`node_bank_manifest.json`, `.dynamix_skillbank_index.json`, stage logs, and
usage JSONL. No heldout query, label, or score is used for selection.

1. Reported and committed layer count; per-level `input_count`,
   `generated_count`, `chosen_k`, `tested_k`, and `stop_reason`.
2. L0 final community count; singleton count/fraction; min, mean, median, and
   max member count.
3. L0 `split_events` grouped by `statistical_split` and `split_reason`, plus all
   selected/tested K values and token-packing fallback count.
4. Excluded oversize singleton IDs/count; raw level-0 item count and source
   record coverage.
5. L1 card count, total support mass, total node count, and nodes by level.
6. Empty/malformed card count, where any exported node has an empty `name`,
   `trigger`, or `content`.
7. Within each level, L2-normalize saved node embeddings, compute each node's
   maximum cosine similarity to another node in the same level, then report the
   mean and fraction at or above 0.90. The baseline L1 fraction is
   `0.1799163179916318`.
8. Generation prompt/completion/total tokens, embedding tokens, build wall
   time, retry count, and failed generation count.

## Predeclared Selection Rule

The minsplit8 build must first pass every hard gate:

- source and records hashes match;
- 200 raw items are present and no new item is excluded;
- every exported card has non-empty `name`, `trigger`, and `content`;
- the build finishes without an unrecovered generation or embedding failure.

It has a structural signal if either committed layers decrease from 6 to at
most 5 or total exported nodes decrease by at least 5%, while the within-L1
cosine-neighbor fraction at 0.90 does not rise by more than 0.02 absolute and
generation total tokens do not rise by more than 10%.

- If hard gates and the structural signal pass, select minsplit8 for one full
  development-heldout evaluation and do not run minsplit10.
- If a hard gate fails, reject this branch without heldout evaluation.
- Run minsplit10 only if minsplit8 passes hard gates but misses the structural
  signal and its L0 audit shows that the unresolved difference can arise from
  refinement nodes affected by threshold 10. It is a second global-granularity
  diagnostic, not a stronger pure-depth setting.
- If any metric lies exactly on a threshold or missing usage data prevents a
  decision, mark the result ambiguous; do not promote by judgment call.

## Heldout Protocol and Claim Boundary

There is no 40-task pilot. Selecting on 40 heldout tasks and later including
them in a 200-task result would tune on heldout data. After build-only selection,
at most one candidate receives a full `[200,400)` evaluation.

Because this heldout set has already been observed repeatedly during method
development, the result is development evidence, not an untouched final test.
It uses the same model endpoint, workers, generation settings, tools, top-k,
retrieval query/index, 30-turn limit, timeout/retry policy, and LibreOffice
recalc evaluator as the fresh baseline. All failures remain in denominator 200.

Before any DynaMix-versus-vanilla claim, rerun vanilla under that exact current
runtime and separately report context, max-turn, timeout, and missing-workbook
failures.

## Compute and Stop Conditions

Reuse train trajectories; maximum 32 workers. Start with one full-record
build-only minsplit8 run. Stop on a hard-gate failure, protocol contamination,
or two fair flat/negative global-granularity diagnostics. Do not implement a
new framework or core algorithm for this experiment.

## Amendment Log

- Independent review reclassified the intervention from pure depth to global
  granularity, removed the overlapping 40-task pilot, defined objective metrics
  and thresholds, and froze source hashes.
- Baseline-fairness audit reclassified the recorded vanilla run as contextual.

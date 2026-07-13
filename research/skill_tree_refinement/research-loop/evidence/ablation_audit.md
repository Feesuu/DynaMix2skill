# Historical Static Ablation Audit

## Verdict

The old controlled suite is useful for prioritization but cannot be directly
combined with the fresh `97/200=48.5%` run. Its full-tree anchor is
`90/200=45.0%`, its source records come from the July 2 run, and some endpoint
forms differ. New research claims require matched reruns from the fresh records.

## Results

| Variant | Correct / 200 | Delta vs old full 45.0% | Mechanism | Initial reading |
|---|---:|---:|---|---|
| old full static | 90 | 0.0pp | GMM-BIC, cumulative mass, full hierarchy | historical anchor |
| hard assignment GMM | 82 | -4.0pp | argmax parent only | soft multi-parent appears useful |
| KMeans elbow | 85 | -2.5pp | hard KMeans, elbow K | GMM-BIC appears useful |
| fixed-K KMeans | 80 | -5.0pp | hard KMeans, K=8 | automatic/probabilistic clustering appears useful |
| L0 single-card | 90 | 0.0pp | one card per L0 community | multiple L0 cards not supported in this run |
| L1 only | 89 | -0.5pp | stop after L0->L1 | deep hierarchy adds little in this run |
| retrieve L1 only | 82 | -4.0pp | full tree, export L1 only | L1 alone did not match full retrieval |
| retrieve L2+ only | 75 | -7.5pp | full tree, export L2+ only | high-level cards alone are weak |

Artifacts: `runs/ablations/static_controlled/<variant>/<timestamp>/` and
`runs/static_qwen35_awq_8bembed_chunk28000_minsplit4_after_budget_fix_20260702_142314/scenarios/static_build/`.

## What These Runs Support

- Keep GMM-BIC and cumulative-mass soft assignment as the default starting point.
- Test a shallower hierarchy because L1-only was within one task of the old full
  tree, while L2+-only was clearly worse.
- Do not assume L0 multi-card generation is necessary; its value needs a fresh
  matched rerun or a more targeted card-quality analysis.

## What They Do Not Support

- They do not prove that L1-only matches the fresh 48.5% method.
- They do not isolate tree depth from `compute_kmax`; changing min split affects both.
- They do not establish statistical significance or robustness across seeds.
- They do not show why individual tasks flipped; task-level paired analysis is pending.
- Equal aggregate scores do not imply identical behavior or that a component is
  useless; positive and negative flips can cancel.

## Required Fresh Controls

1. Freeze the current fresh records/config/evaluator and compare fingerprints.
2. Build minsplit8 before adding a new algorithm.
3. If the structural hypothesis survives, compare chosen hierarchy versus a
   fresh L1-only control on exactly the same heldout rollouts/protocol.
4. Record paired task flips and retrieved cards, not only aggregate accuracy.

## Baseline Fairness Correction

The fresh DynaMix run and the recorded vanilla run are not a strict matched
pair. Both use `Qwen3.5-9B-AWQ`, temperature 0, thinking enabled, 30 turns,
the verified heldout `[200,400)`, and LibreOffice recalc. However:

- DynaMix used `https://evirdwimyrmm.10.27.127.9.nip.io/v1` and 32 workers.
- Vanilla used `http://127.0.0.1:11802/v1` and 16 workers.
- DynaMix generated 200/200 results. Its stream timeouts recovered on retry.
- Vanilla generated 197/200 results: task `32789` exceeded the 100k context
  window, while tasks `37378` and `50051` exhausted 30 turns without output.

Therefore `48.5% - 41.0% = 7.5pp` is a descriptive cross-run difference, not a
clean estimate of DynaMix's causal gain. A current-endpoint vanilla rerun is
required before using the delta as headline evidence. This does not block the
build-only minsplit diagnostic, but it blocks comparative performance claims.

# High-Level Fragmentation Diagnostic

Status: structural diagnostic only, 2026-07-11. No model rollout or heldout
label was used.

## Verdict

The current hierarchy is not deep primarily because `min_split_size=4` is
small. The more direct failure is that upper-layer GMM-BIC repeatedly selects
many near-degenerate components, including singleton components whose
spherical variance is pinned to `min_covar=1e-6`. Recursive summarization then
rewrites those singleton or two-card groups as if they were higher-level
abstractions.

Changing only `min_split_size` does not constrain community size. In
`ProjectedGmmTreeBuilder.cluster_layer`, it only stops an entire layer when the
layer input count is below the threshold. It does not reject small child
components. `GmmBicConfig.min_effective_samples_per_component` is also only a
`Kmax` bound; `_finalize_candidate` explicitly accepts every non-empty primary
component.

## Fresh Minsplit4 Tree

Authoritative artifact:
`runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/dynamix_tree/`.

| Input level | Input items | Chosen K | Community singletons | Community median size | Variances at `1e-6` |
|---:|---:|---:|---:|---:|---:|
| L0 | 200 | 2 coarse, then token-budget refinement to 101 leaves | 57/101 | 1 | 0/2 coarse components |
| L1 | 239 | 64, equal to `abs_kmax` | 16/64 | 3 | 16/64 |
| L2 | 64 | 25 | 15/25 | 1 | 15/25 |
| L3 | 25 | 11 | 9/11 | 1 | 9/11 |
| L4 | 11 | 5 | 4/5 | 1 | 4/5 |
| L5 | 5 | 2 | 1/2 | 2.5 | 1/2 |

The equality between singleton counts and floor-variance components from L1
through L5 is direct evidence of spike components. At L1, BIC selected the
largest tested K=64 with a margin of 356.4. At later levels the hierarchy is
mostly singleton rewrites, not multi-item abstraction.

The resulting card language becomes more generic with depth. A simple
diagnostic for verification/check/inspection language rises from 181/239 L1
cards (75.7%) to 54/64 L2 (84.4%), 23/25 L3 (92.0%), and 100% at L4-L6. This
keyword statistic is not a semantic-quality metric, but it agrees with the
singleton and near-duplicate evidence.

## Why Minsplit8 Did Not Repair It

Authoritative artifact:
`runs/research_minsplit8_build_only_20260710_121345/dynamix_tree/`.

- L1: 249 inputs -> K=51, with 11 singleton communities.
- L2: 51 inputs -> K=24, with 14 singleton communities.
- L3: 24 inputs -> K=12, with 9 singleton communities.
- L4: 12 inputs -> K=6, with 5 singleton communities.

`min_split_size=8` stopped the final six-item layer, but did not prevent the
earlier high-K, singleton-heavy layers. Its completed heldout result was
88/200, so increasing the same global parameter again is not justified.

## CPU-Only Replay Reproducibility Gate

The first no-LLM replay was invalid for parameter selection. Local PCA used
scikit-learn's randomized SVD path without a `random_state`; identical frozen
L1 vectors and the same configured GMM seed could therefore produce different
projected vectors and different selected K. The earlier reported `K=63` and
`K=16` values must not be used as structural evidence or as a reason to choose
one `min_effective_samples_per_component` setting.

The live implementation now derives one deterministic seed per layer or
budget-refinement node, uses it for both PCA and the downstream split selector,
and normalizes only the PCA copy into scikit-learn's 32-bit seed domain. The
configured GMM seed and search protocol are otherwise unchanged. Regression
tests cover randomized-SVD selection, large seeds, fresh builders with reversed
input order, and budget-refinement splits. The full repository test suite is
`162 passed`.

A post-fix sweep over the exact frozen 239 L1 vectors changed only
`min_effective_samples_per_component`. Candidate K fits were computed once per
repeat and then truncated at each setting's Kmax; this is mathematically
equivalent because the parameter only changes Kmax. Candidate scheduling used
16 CPU workers, while every fit retained the production random seed, five
restarts, BIC, and membership policy.

| Min effective | Kmax | Selected K | Singletons | Median size | Multi-parent items | Membership hash prefix |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 64 | 64 | 16/64 | 3.0 | 0 | `a5c28f8560d3` |
| 4 | 59 | 48 | 10/48 | 4.5 | 1 | `3d0ad95b7e4c` |
| 6 | 39 | 18 | 1/18 | 12.0 | 2 | `12eb63f56b07` |
| 8 | 29 | 18 | 1/18 | 12.0 | 2 | `12eb63f56b07` |
| 10 | 23 | 18 | 1/18 | 12.0 | 2 | `12eb63f56b07` |

Both complete repeats were byte-identical for every setting. The result shows
a stable structural transition once K values above 39 are removed: settings
6, 8, and 10 select the same fit and memberships. It does not establish that
K=18 improves heldout performance. The smallest controlled structural variant
is therefore `min_effective_samples_per_component=6` at L1 only, with L0/L1
cards frozen and no recursive higher-level summarization; it must remain an
ablation until paired outcome evidence exists.

## Research Consequence

The next structural control should keep the completed L0 communities and L1
cards fixed, then change only upper-level clustering. The smallest credible
variant is one coarser, component-mass-validated L1 clustering layer used as a
router over L1 advice, with no recursive singleton summarization. This is a
method repair/ablation, not a novelty claim. Its value must be tested against
the frozen dense and current L2-router controls under the same rollout
protocol.

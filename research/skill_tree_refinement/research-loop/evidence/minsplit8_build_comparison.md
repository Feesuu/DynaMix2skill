# min_split_size=8 Build Comparison

## Verdict

- Predeclared build gates: PASS.
- The tree is one committed level shallower, but the nodebank only shrinks slightly because L1 cards increase.
- This parameter also changes L0 budget refinement behavior, including token-pack fallback; it is not a pure depth knob.

## Controlled Results

| Metric | minsplit4 | minsplit8 | Delta |
|---|---:|---:|---:|
| Committed layers | 6 | 5 | -1 |
| Nodebank nodes | 346 | 342 | -4 |
| L1 cards | 239 | 249 | +10 |
| L0 communities | 101 | 90 | -11 |
| L0 singletons | 57 | 48 | -9 |
| Analyst requests | 208 | 183 | -25 |
| Total generation tokens | 3427798 | 3277437 | -4.4% |
| L1 NN cosine >= 0.90 | 18.0% | 18.9% | +0.9% |
| Exact duplicate fraction | 4.3% | 5.3% | +0.9% |
| Statistical L0 refinement events | 9 | 6 | -3 |
| Token-pack fallback events | 0 | 4 | +4 |

## Decision

The predeclared build gates pass, so the next controlled action is one full heldout-200 run on the minsplit8 tree. Do not run minsplit10 before that result unless minsplit8 exposes a concrete safety failure.

## Final Heldout Result

The completed Stage 07 LibreOffice-recalc evaluation scores 88/200 (44.0%);
the raw cached-value audit scores 71/200 (35.5%). Agent collection produced
workbooks for 198/200 tasks, with two missing-output failures. Relative to the
frozen minsplit4 result of 97/200 (48.5%), there are 18 positive flips and 27
negative flips, for a net loss of nine tasks.

This is not a clean causal estimate of the tree parameter. Both historical
runs were affected by the now-fixed temperature propagation bug: the CLI asked
for 0.0 but React requests used 0.7. They also predate the task-local temporary
directory guard. The valid conclusion is narrower: minsplit8 does not show the
large, robust gain required to justify another minsplit10 depth sweep, and its
build already failed to reduce L2+ redundancy. Do not continue to minsplit10;
move to the fresh paired retrieval-role diagnostic.

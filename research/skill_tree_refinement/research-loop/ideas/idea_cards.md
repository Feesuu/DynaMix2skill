# Idea Cards

## IDEA-001: Global Granularity Control

Origin: `idea.md`
Status: stopped after selected full heldout diagnostic; final Stage 07 audit pending

### Thesis
The current `min_split_size=4` may create an unhelpful global granularity: it
affects ordinary-layer stopping, the GMM K search below threshold, and L0
over-budget refinement/fallback. A larger value may simplify the hierarchy, but
cannot be interpreted as a pure upper-depth change.

### Mechanism
Hold the complete fresh SpreadsheetBench protocol fixed and rebuild from the
same 200 train records with `min_split_size` 4, 8, and conditionally 10. Measure
L0 refinement outcomes and card structure before spending on heldout rollouts.

### Novelty Claim
None yet. This is a mechanism diagnostic, not the proposed final method.

### Closest Work
Pending source-verified literature review on hierarchical retrieval, recursive
abstraction, agent memory, and skill evolution.

### Minimum Experiment
Build a minsplit8 tree using the fresh run's ordered records; run 10 only if 8
passes safety gates but leaves a predeclared global-granularity question
unresolved. Compare depth, L0 GMM/fallback events, node/community counts,
support, singleton rate, prompt/token cost, semantic redundancy, and exclusions.

### Killer Baseline
Current fresh static DynaMix tree and heldout result: `97/200=48.5%`.

### Metrics
Tree depth, per-level K/items/cards, support and singleton distributions,
summary token cost, nodebank size, within-level embedding redundancy, and
LibreOffice-recalculated heldout accuracy for at most one selected setting.

### Expected Failure Modes
Larger communities may produce generic cards; shallower depth may merely remove
cards without improving retrieval; build stochasticity may confound comparisons.

### Compute / Resource Constraints
Reuse existing trajectories. Up to 32 workers are authorized, but full heldout
is gated on a structural signal and a reviewed experiment contract.

### Current Evidence
The fresh tree has 7 reported layers, 6 committed layers, and 9 statistical L0
budget-refinement splits. Older L1-only reached `89/200` versus an old full-tree
`90/200`, but that comparison predates the fresh `97/200` anchor.

### Next Action
Preserve the final Stage 07 artifact when the active runner exits. Do not run
minsplit10: minsplit8 is one level shallower but has worse L2+ redundancy and a
44.0% provisional heldout score with a 44.5% mathematical upper bound.

### Risks
`min_split_size` also changes GMM K bounds, not only tree stopping depth; the
analysis must report both effects.

## IDEA-002: Evidence-Preserving Upper-Level Evolution

Origin: `idea.md` / IDEA-001 evidence
Status: Phase A implemented, independently reviewed, and queued for matched run

### Thesis
Repeated N-to-1 summaries may discard operational evidence. Upper-level skills
should remain traceable to supporting successes/failures and earn inclusion via
utility evidence rather than abstraction alone.

### Mechanism
Phase A uses generated L2 cards only as router identities. Each router is a
membership-weighted centroid of its direct L1 embeddings; query routing creates
an L1 candidate set and only L1 procedural cards are injected. Phase B is
contingent: each unique L1 card must help at least one source-excluded failed
train task and hurt no source-excluded successful train task under LibreOffice
recalc before admission.

### Novelty Claim
Unverified until closest-work collision analysis is complete.

### Minimum Experiment
One isolated strategy variant on the same fresh records, followed by a paired
diagnostic against current DynaMix.

### Killer Baseline
Current static DynaMix plus L1-only and retrieve-layer ablations under a matched
fresh protocol.

### Metrics
Card specificity/provenance, retrieved-card usefulness, positive/negative task
flips, token cost, and LibreOffice-recalculated accuracy.

### Expected Failure Modes
Validation overfits train tasks; provenance makes prompts too long; added
scoring duplicates retrieval similarity without new signal.

### Compute / Resource Constraints
Implement only one selected mechanism at a time using existing abstractions.

### Current Evidence
Older retrieve-L2+-only (`75/200`) underperformed retrieve-L1-only (`82/200`),
suggesting weak high-level cards, but the old protocol is only diagnostic. Live
Phase-A preflight changes every query's top-10, returns exactly ten L1 cards for
200/200 queries, uses zero global backfill, and preserves the exact source
embedding index. Phase-B feasibility finds valid source-excluded success and
failure controls for all 239 L1 cards.

### Next Action
Run the already queued fresh dense control followed by the routed variant and
apply the frozen pair gate before writing any Phase-B code.

### Risks
Paper mechanisms may rely on different tasks, supervision, or information
access; method identity and fairness must be preserved.

## IDEA-003: Component Pruning Under Fresh Protocol

Origin: `idea.md` / prior ablations
Status: queued

### Thesis
Remove only components that fail a fresh matched comparison; historical
ablation results are insufficient because their anchor differs from 48.5%.

### Mechanism
Rerun only decision-critical ablations after IDEA-001 narrows the tree design.

### Novelty Claim
None; this is scientific control and simplification.

### Minimum Experiment
Fresh matched L1-only versus chosen full hierarchy, then one retrieval-layer
ablation if task-level evidence still implicates upper levels.

### Killer Baseline
Chosen hierarchy with all otherwise identical settings.

### Metrics
Accuracy, paired task flips, retrieval overlap, token/runtime cost, and tree
complexity.

### Expected Failure Modes
Run-to-run variance obscures small differences; pruning may improve cost but not
quality.

### Compute / Resource Constraints
No broad ablation sweep; only experiments tied to a concrete decision.

### Current Evidence
The older controlled suite ranges from 37.5% to 45.0%, with L1-only at 44.5%.

### Next Action
Wait for IDEA-001 and IDEA-002 evidence.

### Risks
Post-hoc selection; all promotion rules must be written before results.

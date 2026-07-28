# Evidence-Balanced Skill Tree v4

## Status

This document is the implementation contract for the
`research/ebst-v4-strict-online` branch. It is a new experimental
tree policy. It does not replace or silently change the legacy projected
GMM-BIC tree or CDOST v3.

## Problem Observed In CDOST v3

The completed CDOST artifact is structurally valid but unsuitable as a
deployment skill bank:

- 200 atoms produced a 399-node binary tree with height 97.
- 152 of 199 internal nodes had a singleton child.
- 1,989 of 2,000 heldout selections were L1 atom nodes.
- Experience atoms and transferable skills were treated as the same
  retrievable object.
- The OTD approximation condition was not observed (`beta = 0`), so the
  separation-dependent approximation statement did not apply to the artifact.

The v4 design fixes these problems at construction time. Retrieval-side
deduplication is not the primary repair.

## Method Identity

The method has three distinct objects:

1. **Evidence atom**: one trajectory-local, provenance-preserving analysis.
   Atoms are immutable evidence and are never injected into heldout prompts.
2. **Balanced metric tree**: the online structural index over atom embeddings.
   The evolutionary history starts from an empty tree and uses one
   source-ordered insertion operation. The open-loop run performs that full
   history directly; the formal closed-loop run resumes an exact checkpoint
   from that empty-start history before processing policy-dependent arrivals.
   Every committed arrival is followed by dirty-path capsule refresh,
   validation, and an atomic, fingerprinted checkpoint.
3. **Skill capsule**: a transferable procedure supported by a non-singleton
   evidence bucket or by at least two distinct child capsules. Only accepted
   capsules are exported to the nodebank.

This differs from Skill-SP's method identity. Skill-SP uses skills to generate
training tasks and trains a prompt-only solver. V4 adopts only its
evidence-driven skill lifecycle, structured packages, and validation
discipline; it does not claim or implement Skill-SP self-play.

## Dual-View Metric

For unit-normalized trigger and procedure embeddings, define

\[
d_\lambda(i,j) =
\sqrt{
  \lambda \lVert t_i-t_j\rVert_2^2 +
  (1-\lambda)\lVert p_i-p_j\rVert_2^2
},
\qquad \lambda\in[0,1].
\]

This is Euclidean distance after mapping each atom to

\[
\phi_\lambda(i) =
[\sqrt{\lambda}\,t_i;\sqrt{1-\lambda}\,p_i].
\]

It is a metric on the resulting vector space and a pseudometric on Atom
identities because two distinct Atoms may have identical embeddings (and an
endpoint value of `lambda` ignores one view). Triangle-inequality pruning
remains valid. This is a structural guarantee, not a guarantee that the
embeddings capture semantic truth.

## Balanced Online Tree

The tree is a B+-style metric hierarchy with:

- maximum fanout or leaf capacity `B`;
- minimum non-root occupancy `m = ceil(B / 2)`;
- all atoms stored only in leaves;
- one representative atom and a certified covering-radius upper bound per
  child entry;
- deterministic insertion and deterministic split tie-breaking.

Insertion descends through the child requiring the smallest covering-radius
enlargement, then the shortest representative distance, then stable ID order.
An overflowing node is split by farthest-pair pivots. The remaining entries
are assigned by distance-margin order while preserving `m` occupancy on both
sides. Splits propagate only along the insertion path.

The structural update is identical in both settings:

```text
for atom in source_order:
    tree.insert(atom)
```

`open_loop_replay` uses fixed, previously collected `0:200` train
trajectories. Atom generation may be cached because it is policy-independent,
but only the committed prefix is visible to the tree and capsule registry.
`closed_loop_skill_evolution` uses an open-loop prefix checkpoint, then
reruns each later train task with the current nodebank before extracting and
inserting its Atom. The current formal protocol uses `0:120` as the bootstrap
prefix and `120:200` as policy-dependent arrivals. This split is configurable
but must be fixed before looking at heldout results.

## Verifiable Structural Guarantees

For `B >= 4`, insertion maintains the following invariants:

1. Every non-root node has occupancy in `[ceil(B/2), B]`.
2. Every leaf has the same depth.
3. Every atom appears exactly once and only in a leaf.
4. Leaf radii are exact over at most `B` Atoms. Internal radii satisfy
   `R(v)=max_c[d(rep(v), rep(c)) + R(c)]`, a certified upper bound maintained
   from at most `B` children.
5. An insertion changes only the root-to-leaf path, nodes created by split
   propagation, and the at most `B + 1` direct children reparented by each
   internal split. The number of structurally touched nodes is therefore
   `O(B h)`. Radius maintenance also inspects at most `B` entries per level,
   so online structural maintenance costs `O(B h d)` for embedding dimension
   `d`, excluding the optional full audit.
6. The height is logarithmic in the number of atoms under the occupancy
   invariant. For `n` atoms and minimum internal fanout `m`, the reported
   conservative bound is:

\[
h \le 1 + \left\lceil \log_m
  \left(\max\left(1,\frac{n}{2m}\right)\right)
\right\rceil.
\]

Exact nearest-evidence search may use triangle-inequality branch and bound.
Its correctness follows from the pseudometric triangle inequality and
certified covering upper bounds. Looser internal radii can reduce pruning but
cannot remove a valid candidate. No sublinear high-dimensional runtime claim
is made; worst-case search remains linear.

The property tests run full validation after every insertion. Strict online
production runs refresh the dirty path, validate the complete committed
prefix, and write an atomic checkpoint after every arrival. The structural
`insert()` itself remains local; the full validation is an explicit research
audit cost and is not included in the `O(B h d)` structural-update claim.

## Skill Construction

### Leaf capsules

A leaf is eligible only when it contains at least two atoms. The analyst must
produce one capsule with:

- `name`
- `trigger`
- `content`
- `scope`
- `verification`
- `failure_modes`

The prompt receives the complete structured atoms and their provenance IDs.
Task-specific answers, workbook paths, exact task IDs, and accidental literals
must not be copied into reusable text.

The analyst receives successful atoms as `positive_evidence_atoms` and failed
atoms as `negative_evidence_atoms`. Negative evidence may support a root cause,
boundary, guardrail, or corrected procedure, but an observed failed action
cannot be promoted as a recommendation. This separation uses only the recorded
train outcome; it never exposes a gold answer to retrieval or heldout prompts.

### Parent capsules

An internal node is eligible only when it has at least two distinct accepted
child capsules. A parent is accepted only if it expresses a cross-child
invariant and is not a normalized duplicate of any child. Structural nodes
that do not qualify remain in the tree but are not retrievable.

This prevents singleton rewrites by construction. Exact normalized child
duplicates are rejected deterministically. Semantic paraphrases are filtered
by the parent prompt and second promotion audit, but are not ruled out by a
mathematical guarantee.

## Evidence-Driven Lifecycle

Each capsule has one of:

- `candidate`: generated and awaiting validation;
- `active`: eligible for retrieval under the configured validation policy;
- `rejected`: failed schema, provenance, leakage, or abstraction checks;
- `archived`: invalidated after evidence/tree change and retained for audit;
  the event records `archive_disposition=invalidated_stale`, so archival is
  not misreported as a successful replacement.

Every capsule records its evidence atom IDs, source item IDs, validation
events, and exposure counts. In the closed loop, selected capsule IDs and the
subsequent LibreOffice outcome are recorded before the new trajectory is
inserted. These counts are carried into a replacement capsule on the dirty
path. They are labelled `exposure_outcome_not_causal`: co-selection does not
prove which skill caused the outcome. The lifecycle ledger records the
candidate event before the final active or rejected event. A smoothed
reliability
estimate is:

\[
\hat r = \frac{1 + n_{\text{pass}}}{2 + n_{\text{trial}}}.
\]

This is a lifecycle statistic, not a generalization guarantee. Optional
source-task replay may promote or reject candidates, but replay success only
supports a source-task safety claim.

## Retrieval

Atoms are excluded. Retrieval operates on complete active skill capsules.
The default policy is token-budgeted capsule retrieval:

1. Embed the query with the frozen embedding protocol.
2. Score all active capsules with the frozen shifted-cosine relevance rule.
3. Select an ancestor-descendant antichain under the prompt token budget.
4. Inject each selected capsule as one coherent unit.

The implementation must report selected capsule count, level distribution,
lineage conflicts, token budget utilization, and evidence support.

Replay reliability is stored for audit but is not used by the v4 selector
because behavioral replay has not been implemented. Adding reliability to the
ranking before measuring it would create a fictitious quality signal and break
the first tree-policy-only comparison.

The structural update bound does not imply a bounded semantic-refresh prompt.
If intermediate capsules are unavailable, a parent may receive the active
capsule frontier of its subtree, which is `O(S_subtree)` in the worst case.
Prompt budgeting is therefore checked explicitly. A prompt-budget or runtime
generation failure is recorded separately from semantic rejection and blocks
heldout evaluation rather than silently producing a partial skill bank.

## Required Artifacts

- `balanced_tree_state.json`
- `experience_atoms.json`
- `skill_capsules.jsonl`
- `skill_lifecycle_events.jsonl`
- `node_bank_manifest.json`
- `tree_quality_audit.json`
- `summary.json`
- `dynamic_snapshots/arrival_XXXX/checkpoint.complete.json` for strict
  open-loop replay
- `checkpoints/arrival_XXXX/checkpoint.complete.json`,
  `online_records.json`, and `experiment_contract.json` for closed-loop
  evolution

`tree_quality_audit.json` must include occupancy, leaf-depth equality, height
and bound, split count, locality diagnostics, covering-radius checks,
retrievable atom count (required to be zero), lifecycle counts, duplicate
counts, and capsule support distribution.

## Explicit Non-Claims

V4 does not claim:

- optimal hierarchical clustering;
- semantic correctness from structural invariants;
- sublinear nearest-neighbor runtime in the worst case;
- downstream benchmark improvement before a controlled experiment;
- Skill-SP reproduction;
- behavioral validation when source-task replay was not run.

## Relationship To Skill-SP

Skill-SP (Qwen Applications, arXiv:2607.22529) co-evolves a proposer, solver,
and skill controller through self-play. V4 does not implement that training
loop. It adopts three narrower engineering lessons from the official
implementation: preserve per-skill lifecycle state, separate positive support
from negative evidence, and gate promotion instead of treating every generated
summary as deployable. In v4 the separation is explicit in the leaf analyst
payload, while promotion uses deterministic checks and a second LLM audit.
V4's parent-promotion validator is structural and evidence-based; it is not a
substitute for Skill-SP's execution feedback or reinforcement-learning
objective.

## Fair Experiment Gate

The first controlled comparison must reuse the exact CDOST-v3 Atom artifact
and keep records, model identity, decoding, workers, evaluator, retrieval
query, top-k/token budget, retry policy, and heldout denominator fixed. The
runner verifies these invariants against the baseline control manifest; only
service URLs and credentials may differ and are still logged. The changed
variables are the declared tree policy and capsule construction/export
semantics. Runtime-confounded cases must be reported separately from workbook
correctness. Shared rollout/evaluator, nodebank rendering, retrieval query, and
antichain-selection source fingerprints must also match. Consequently, a
pre-v4 baseline manifest cannot serve as the formal control; the control must
be rerun from the same commit, while old results remain historical context.

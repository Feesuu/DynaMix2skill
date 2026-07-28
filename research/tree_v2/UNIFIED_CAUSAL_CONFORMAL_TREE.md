# Unified Causal-Conformal Skill Tree

Status: superseded historical design; not implemented and not normative.

This document records an earlier multi-parent conformal/GMM proposal. The
current implemented method is the single-parent binary
**Certified Dual-view Online Skill Tree (CDOST)** specified in
`CDOST_CODING_PLAN.md` and audited in `CDOST_ALIGNMENT_MATRIX.md`. No result,
artifact, or theoretical claim from the current branch may be attributed to
UCST.

## 1. Verdict

The primary defect is not simply that GMM is soft or that `min_split_size` is
too small. The current pipeline conflates three different decisions:

1. what semantic experience a trajectory contains;
2. which semantic community that experience belongs to;
3. how to fit a community into an LLM prompt.

This causes long raw traces and token limits to determine the semantic tree.
The proposed redesign separates those decisions and makes static construction
and dynamic insertion two execution modes of one maintained state.

Working name: **Unified Causal-Conformal Skill Tree (UCST)**.

## 2. What is adopted from prior work

### adRAP

Useful mechanism:

- preserve fitted projection and mixture state;
- update sufficient statistics when an item arrives;
- locally split affected clusters;
- regenerate only affected summaries and ancestors.

Required correction for our setting:

- adRAP treats document cluster size as a reason to split;
- DynaMix must not treat LLM prompt token overflow as evidence of a new semantic
  community;
- semantic clustering and prompt-capacity compression must be separate states.

### MemTree and online hierarchical clustering

Useful mechanism:

- top-down local insertion;
- update only the affected path;
- evaluate hierarchy quality with a structural objective.

Limit:

- the OTD approximation result is for a single-parent hierarchy under a
  well-separated-data assumption;
- the current DynaMix object is a multi-parent DAG, not a strict tree;
- LLM aggregation is not automatically guaranteed to preserve the separation
  assumption.

UCST therefore does not claim to inherit MemTree's approximation theorem. It
uses a separate conformal set-routing guarantee for overlap and reports a
single-parent backbone metric only as an auxiliary structural diagnostic.

### SkillCAT

Useful mechanism:

- extract evidence near a same-task success/failure divergence instead of
  summarizing a complete trace;
- validate a candidate experience by replay before promoting it;
- keep only routed experience at execution time.

Required correction for our data:

- one trajectory alone cannot establish a causal contrast;
- a single-trace atom is labeled local evidence, not causal evidence;
- a causal label is allowed only when a same-task success/failure pair exists.

## 3. State model

For each trajectory \(\tau_i\), keep two separate objects.

### Immutable evidence

The original task, trace, verifier outcome, predicted output, reference output,
and runtime metadata remain immutable provenance. They are never used directly
as the clustering vector.

### Verified Experience Atom

By default, one trajectory yields one atom:

```text
trigger:
decision:
invariant:
verification:
scope:
failure_mode:
```

The extraction prompt must:

- distinguish observed evidence from inferred guidance;
- use verifier differences on training data;
- forbid concrete task IDs, workbook paths, answer coordinates, and incidental
  answer values in the reusable text;
- explain when the lesson does not apply;
- produce one coherent atom rather than many generic cards.

If a same-task success/failure pair is available, the extractor compares the
first meaningful action divergence and records contrastive evidence. Otherwise
the atom is explicitly marked `single_trace`.

An atom is promoted to the semantic index only after source-task replay:

- failure -> success: strong positive evidence;
- success -> success: preservation evidence;
- failure -> failure: quarantine or low reliability;
- success -> failure: reject.

Reliability becomes a weight \(w_i\), not an unverified LLM confidence.

## 4. Semantic clustering

Experience embeddings are normalized:

\[
z_i \in \mathbb{S}^{d-1}, \qquad \lVert z_i\rVert_2 = 1.
\]

Because the vectors are directional, the preferred model is a von Mises-Fisher
mixture or a spherical mixture expressed through cosine likelihood. For level
\(\ell\):

\[
p(z_i) = \sum_{k=1}^{K_\ell}
\pi_k\,\mathrm{vMF}(z_i\mid\mu_k,\kappa_k).
\]

The penalized objective is:

\[
\mathcal{J}_{\ell}
=
-2\sum_i w_i\log p(z_i)
+ \operatorname{pen}(K_\ell,N_\ell).
\]

`pen` is an MDL/BIC-style model-complexity penalty. Parent-child duplication is
checked separately by the abstraction certificate because mixing an
LLM-dependent duplication score into the mixture objective would make the
optimization claim unclear. A local split is accepted only if:

1. it decreases \(\mathcal{J}_{\ell}\);
2. every child has effective support at least \(m_{\min}\);
3. the result is stable across configured restarts;
4. the split is semantic, not triggered only by prompt length.

This directly prevents the tiny-variance singleton exploit seen in the current
GMM-BIC implementation.

## 5. Conformal set-valued routing

Current cumulative-mass routing is relative: some component always has the
largest posterior even for an outlier. UCST adds an absolute compatibility
test.

Using a held-out calibration subset of training atoms, define a nonconformity
score for label \(k\):

\[
a(z,k) = -\log p(k\mid z).
\]

The split-conformal p-value is:

\[
p_k(z)
=
\frac{1+\sum_{j\in\mathcal{C}}
\mathbf{1}[a(z_j,y_j)\ge a(z,k)]}
{|\mathcal{C}|+1}.
\]

The multi-parent candidate set is:

\[
\Gamma_\alpha(z)=\{k:p_k(z)>\alpha\}.
\]

Consequences:

- multiple compatible communities remain possible;
- an empty set means calibrated novelty instead of forced assignment;
- under exchangeability and a fixed calibration protocol, split conformal
  prediction provides marginal coverage of at least \(1-\alpha\) for the
  frozen model's reference assignments;
- tiny communities use pooled or shrinkage calibration rather than unreliable
  per-community quantiles.

Because this is unsupervised clustering, the reference labels are fitted
pseudo-labels, not ground-truth semantic classes. The coverage result therefore
does not prove that a human-defined "true community" is in the set. It only
calibrates routing uncertainty relative to a frozen clustering model.

Within \(\Gamma_\alpha(z)\), posterior weights are renormalized so that:

\[
q_{ik}\ge 0,\qquad
\sum_{k\in\Gamma_\alpha(z_i)}q_{ik}=1.
\]

This preserves support mass without an arbitrary top-R membership rule.

## 6. Growing K without singleton explosion

If \(\Gamma_\alpha(z)\) is empty, the atom enters a provisional novelty buffer.
It does not immediately become a retrievable singleton skill.

A new semantic component is promoted only when:

- accumulated effective support reaches \(m_{\min}\); and
- fitting the component improves the penalized objective.

Until promotion, buffered atoms remain auditable evidence and can be replayed,
but they do not create a semantic node. This replaces the current behavior
where one extreme point can create a tiny-variance GMM component.

## 7. Token capacity is a compression tree, not a semantic split

Each semantic community owns an evidence-compression tree. Its leaves are
token packs, each satisfying:

\[
\operatorname{tokens}(P_j) \le B.
\]

When a new atom makes a pack exceed \(B\):

1. split or repack only that evidence pack;
2. summarize the changed packs;
3. regenerate their community schema from current child summaries;
4. leave the semantic community identity unchanged.

This is the central static/dynamic unification:

- static construction packs all evidence after semantic fitting;
- dynamic insertion updates only affected packs and ancestors;
- token budget can change summarization cost but cannot fabricate a new
  semantic community.

## 8. Hierarchical abstraction

Higher nodes are generated only from at least two semantically distinct,
validated child schemas.

A singleton child is passed through as a structural link. It is not rewritten
into a duplicate parent.

A proposed parent is retrievable only if it passes an abstraction certificate:

1. **coverage:** it is supported by at least two children;
2. **novelty:** it is not near-duplicate to any single child;
3. **compression:** it is shorter than the combined child evidence;
4. **traceability:** every assertion maps to child provenance;
5. **behavioral safety:** source-task replay does not regress supported tasks.

Rejected parents remain provenance links and are not placed in the node bank.

Parent generation uses a canonical child order and a deterministic decoding
protocol when order-invariance is being measured. Otherwise order-invariance
must not be claimed.

## 9. One algorithm, two modes

UCST exposes one maintained state:

```text
experience atoms
semantic mixture sufficient statistics
conformal calibration state
semantic memberships and support mass
community compression trees
validated schema hierarchy
provenance and replay results
```

Static mode:

```text
extract/verify all atoms
-> fit semantic mixture to convergence
-> build compression trees
-> build validated abstractions
```

Dynamic mode:

```text
extract/verify one arriving atom
-> conformal route or novelty buffer
-> online sufficient-statistic update
-> objective-gated local split/merge if needed
-> update affected compression packs
-> regenerate only affected validated ancestors
```

Both modes optimize and maintain the same objects. Dynamic mode is not a
separate heuristic that routes by prompt capacity.

## 10. Honest guarantees

The design can support the following claims if implemented and tested exactly.

1. **Prompt safety.** Every analyst call is built from packs with measured token
   count at most \(B\).
2. **Support-mass conservation.** Normalized set-valued weights preserve total
   input mass at each semantic level.
3. **Finite-sample routing coverage.** Split-conformal routing has marginal
   \(1-\alpha\) coverage for frozen reference assignments under exchangeability
   and a frozen calibration protocol; it is not semantic ground truth.
4. **Monotone accepted local refinement.** Full local EM and split/merge moves
   are committed only when the penalized objective does not increase.
5. **Online-EM convergence boundary.** With standard stochastic-approximation
   conditions, online EM converges to stationary points, not necessarily the
   global optimum.
6. **Local update complexity.** Routing is \(O(Kd)\); sufficient-statistic and
   summary work is limited to selected communities and affected compression
   paths.
7. **Conditional static/dynamic consistency.** If both modes reach the same
   memberships and deterministic summaries are generated from the same current
   children, their schema content matches. The final partition itself is not
   guaranteed order-independent.

The design does not guarantee causal truth from a single trajectory, global
clustering optimality, perfect semantic communities, or downstream accuracy
improvement.

## 11. Why not directly use MemTree's theorem

MemTree's OTD connection is valuable as a structural reference, but claiming
its approximation result for current DynaMix would be invalid:

- OTD assumes a single-parent hierarchy;
- DynaMix allows overlapping memberships;
- OTD's theorem requires well-separated data;
- LLM-generated parent embeddings need not preserve that assumption.

If a strict theorem on a tree objective is required, UCST can expose an
auxiliary primary-parent backbone and evaluate Moseley-Wang revenue on that
backbone. Secondary conformal links remain typed cross-links. The guarantee
would apply only to the backbone, not to the full overlap graph.

## 12. Falsifiable experiment ladder

All experiments must reuse the same train/heldout split, model endpoint,
rollout protocol, evaluator, top-k, and records unless the tested component
requires an explicitly logged change.

### E0: frozen baseline reproduction

- Reproduce the audited `78/200` run from the stable tag.
- Purpose: verify that the checkpoint is executable.
- No research claim if reproduction fails.

### E1: Experience Atom only

- Replace raw-trajectory clustering text with one verified atom per trajectory.
- Keep the current clustering and hierarchy otherwise unchanged.
- Tests whether evidence representation, not clustering, is the main failure.

### E2: semantic/capacity separation

- Keep E1 semantic clustering.
- Replace token-driven semantic refinement with per-community compression
  packs.
- Primary structural success criterion: singleton ratio and L0 K no longer
  change when only \(B\) changes.

### E3: constrained mixture

- Enforce effective support at fit acceptance.
- Compare GMM, vMF mixture, and hard assignment under the same atom inputs.
- Report likelihood/MDL, singleton ratio, overlap rate, and heldout score.

### E4: conformal multi-parent routing

- Replace cumulative mass with calibrated set-valued routing.
- Report empirical calibration coverage, average set size, empty-set novelty
  rate, and downstream score.

### E5: dynamic consistency

- Build on the first 60% and insert the remaining 40% in dataset order.
- Compare with a full static build over 100%.
- Report assignment Jaccard, adjusted mutual information on the primary
  backbone, center drift, hierarchy duplicate rate, update cost, and heldout
  score.

### E6: replay promotion gate

- Compare all extracted atoms against only replay-validated atoms.
- Report source-task transition counts and heldout transfer.

## 13. Decision gates

Proceed from one stage to the next only if:

- the run is fully auditable and LibreOffice-recalculated;
- no environment/runtime failure is silently counted as a method failure;
- the tested stage improves its intended structural metric;
- any downstream claim uses the full heldout denominator;
- dynamic and static comparisons share the same evidence, model, evaluator,
  and retrieval budget.

The first implementation should be E1 plus E2. Conformal routing and a new
mixture family should not be implemented until we know that atom quality and
semantic/capacity separation fix the observed tree pathology.

## 14. Minimal implementation boundary

The intended implementation should reuse existing state and call paths:

- `src/dynamix_core/tree_builder.py`
- `src/dynamix_core/update.py`
- `src/dynamix_core/gmm_bic.py`
- `src/dynamix_trace2skill/summary.py`
- `src/dynamix_trace2skill/pipeline.py`

Only three new responsibilities are justified:

1. experience-atom extraction and replay status;
2. semantic mixture constraints/calibration;
3. per-community token-pack compression state.

No second benchmark framework, duplicate tree implementation, or parallel
legacy dynamic path should be introduced.

## 15. Primary references

- Chucri et al., *Recursive Abstractive Processing for Retrieval in Dynamic
  Datasets*, arXiv:2410.01736.
- Rezazadeh et al., *From Isolated Conversations to Hierarchical Schemas:
  Dynamic Tree Memory Representation for LLMs*, arXiv:2410.14052.
- Menon et al., *Online Hierarchical Clustering Approximations*,
  arXiv:1909.09667.
- *SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution
  for LLM Agents*, arXiv:2606.13317.
- Cappé and Moulines, *Online EM Algorithm for Latent Data Models*,
  arXiv:0712.4273.
- Straub et al., *Small-Variance Nonparametric Clustering on the Hypersphere*,
  arXiv:1607.06407.
- Diebold et al., *A Unified Framework for Hard and Soft Clustering with
  Regularized Optimal Transport*, arXiv:1711.04366.

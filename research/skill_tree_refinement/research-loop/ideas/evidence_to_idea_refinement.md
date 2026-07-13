# Evidence-to-Idea Refinement

## 2026-07-10: Fresh static result and historical ablation audit

- Triggering evidence: fresh static DynaMix `97/200=48.5%`, recorded vanilla
  heldout `82/200=41.0%`, 7-layer tree, and older controlled ablations.
- Supported: L0 community experience remains a plausible useful mechanism;
  the fresh DynaMix run itself is complete at 200/200.
- Not supported as matched: the 7.5pp cross-run difference, because endpoint,
  workers, and agent-execution completion differ.
- Weakened: deep recursive summarization is not supported by the old L1-only
  and retrieval-layer ablations.
- Untested: whether shallower stopping improves the fresh 48.5% tree; whether a
  provenance/validation mechanism improves upper-level experience.
- Main risk: historical ablations use an older 45.0% anchor, so direct claims
  against the fresh result would be protocol-mismatched.
- Literature status: initial source-verified mechanism/collision map complete;
  novelty remains unresolved.
- Next minimum experiment: build-only `min_split_size=8` from the same fresh
  ordered records; run 10 only under the predeclared conditional gate.
- If successful: supports a preliminary global-granularity diagnostic result.
- If unsuccessful: stop tuning this parameter and move to IDEA-002.
- Final action: `refine_mechanism`.

| Date | Triggering Evidence | Parent Idea | Supported | Weakened | Untested | Final Action | Next Minimum Experiment |
|---|---|---|---|---|---|---|---|
| 2026-07-10 | fresh 48.5%, recorded vanilla 41.0%, 7 layers, old ablations | IDEA-001 | community extraction plausibility | deep summary utility; matched vanilla delta | min split 8/10; validated evolution | refine_mechanism | frozen build-only minsplit8 |

## 2026-07-10: Minsplit8 and retrieval-effect evidence

- `min_split_size=8` passed all frozen build safety gates and reduced committed
  summary levels from 6 to 5, communities from 208 to 183, and generation
  tokens by 4.4%.
- It did not solve the skill-structure problem: total nodes only fell 346 to
  342, L1 cards increased 239 to 249, and L2+ near-duplicate rate at cosine
  0.90 worsened from 58.9% to 62.4%.
- Across 200 heldout retrievals, similarity scores do not separate positive
  flips, negative flips, shared passes, or shared failures. A trace-level
  negative flip confirms that a highly similar card can prescribe the wrong
  output-layout semantics.
- Supported: GMM community discovery remains useful enough to retain; global
  `min_split_size` is not a sufficient repair.
- Weakened: injecting recursively generated L2+ summaries as direct advice;
  treating cosine similarity as skill utility.
- Literature collision: SkillAudit already contributes paired trajectory
  auditing and harmful-update rollback, so generic validation is not novel.
- Refined mechanism: IDEA-002 uses membership-weighted L1 centroids as L2
  routers, injects only L1 executable advice, then conditionally adds
  source-excluded LibreOffice counterfactual admission.
- Next minimum experiment: Phase A of
  `contracts/IDEA-002-evidence-routed-community-skills.md` after the already
  running minsplit8 heldout finishes and before any new source is loaded.
- Final action: `refine_mechanism`.

| Date | Triggering Evidence | Parent Idea | Supported | Weakened | Untested | Final Action | Next Minimum Experiment |
|---|---|---|---|---|---|---|---|
| 2026-07-10 | minsplit8 build, retrieval flips, SkillAudit collision | IDEA-002 | L0 communities; hierarchy as routing structure | blind L2+ injection; similarity-as-utility | L2-router/L1-advice; source-excluded gate | refine_mechanism | Phase A retrieval strategy |

## 2026-07-11: Minsplit8 negative result and Phase-A/B feasibility

- Triggering evidence: minsplit8 provisional LibreOffice 88/200 with at most
  89/200 after the unfinished task; 18 positive versus 27 negative flips;
  Phase-A 200-query dense/routed preflights; source-excluded controls for all
  239 L1 cards; SkillRet and Skill-as-Pseudocode collision checks.
- Supported: L0 soft-community structure remains a plausible routing prior;
  Phase A is a real intervention because all 200 top-10 sets change and routed
  retrieval injects only L1 cards with zero global backfill.
- Weakened: global depth/granularity tuning as the primary repair; high cosine
  as utility evidence; free-form prose relevance as sufficient executability.
- Still untested: whether role-separated routing improves paired outcomes;
  whether source-excluded verifier evidence can remove harmful cards without
  discarding useful advice.
- Main risks: the frozen tree was induced from pre-isolation trajectories; the
  paired run reuses a repeatedly observed development heldout; Phase B costs up
  to 478 card-only rerolls and collides with generic validation literature.
- Literature boundary: SkillRet occupies generic learned skill retrieval and
  Skill-as-Pseudocode occupies typed procedural refactoring. The remaining
  narrow DynaMix distinction is source-excluded deterministic-verifier evidence
  attached to cards induced from overlapping soft communities.
- Next minimum experiment: fresh isolated-harness dense control followed by the
  reviewed L2-router/L1-advice arm, exact 200-task denominator and LibreOffice
  pair audit.
- If successful: regenerate clean train trajectories/tree, then implement the
  already-preflighted Phase-B admission ledger. If failed: stop hierarchy-text
  routing as a method branch and use traces to decide between card admission or
  representation as a new, separately contracted ablation.
- Final action: `refine_mechanism`.

| Date | Triggering Evidence | Parent Idea | Supported | Weakened | Untested | Final Action | Next Minimum Experiment |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | minsplit8 <=44.5%, Phase-A/B preflights, SkillRet/SaP | IDEA-002 | soft-community routing prior; source-excluded feasibility | minsplit tuning; similarity/executability assumptions | paired routing utility; verifier admission utility | refine_mechanism | fresh dense/routed pair |

## 2026-07-11: HDSO collision and Phase-B reframing

- Triggering evidence: HDSO's full paired hypothesis lifecycle, GoSkills'
  role-structured retrieval, SRA's retrieval-versus-incorporation decomposition,
  and DynaMix task `55468` retrieving the right risk but failing execution.
- Supported: the observed DynaMix bottleneck is skill incorporation and
  executable utility, not only nearest-neighbor relevance.
- Weakened: the claim that source-excluded paired verifier admission is itself
  novel; the planned one-success/one-failure control is also underpowered
  relative to HDSO's staged target and guardrail validation.
- Untested: whether soft overlapping communities improve candidate hypotheses,
  validation-stratum selection, or transfer compared with a non-community
  HDSO-style control.
- Protocol decision: do not implement old IDEA-002B merely because it is
  feasible. Finish Phase A first. If promoted, treat HDSO-style admission as a
  baseline and test IDEA-002C as the DynaMix-specific variable.
- Final action: `refine_mechanism`.

| Date | Triggering Evidence | Parent Idea | Supported | Weakened | Untested | Final Action | Next Minimum Experiment |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | HDSO, GoSkills, SRA, task 55468 | IDEA-002B | incorporation/utility bottleneck | generic verifier-admission novelty | community-conditioned hypotheses vs non-community HDSO control | refine_mechanism | finish Phase A; redesign Phase B only if promoted |

## 2026-07-11: Upper-layer GMM degeneration and causal-utility collision

- The fresh tree's L1 GMM selects K=64 at the search boundary for 239 cards.
  Sixteen L1 components are singleton and exactly sixteen variances are pinned
  to `min_covar=1e-6`. L2 and L3 repeat the same pattern with 15/25 and 9/11
  singleton, floor-variance components.
- `min_split_size` only stops a whole layer and does not constrain component
  size. `min_effective_samples_per_component` currently bounds Kmax but is not
  enforced as component validity. This explains why minsplit8 made the tree one
  layer shallower without repairing high-level abstraction.
- Only 3/200 L0 trajectories have more than one selected parent under the
  current cumulative-mass/gap rule. A future contribution cannot rely on the
  phrase "multi-parent" alone; it must demonstrate value from the full saved
  posterior/community geometry.
- SkillAxe's SpreadsheetBench 52% is a 50-task common-subset result where its
  69-skill and 22-skill libraries tie. It supports compact libraries, not a
  same-protocol accuracy target for DynaMix.
- ASSAY establishes task-specific causal heterogeneity and occupies generic
  randomized skill masking. It strengthens the utility-routing hypothesis but
  raises the required baseline: DynaMix must show that community conditioning
  improves utility transfer or lowers validation cost at a matched budget.
- Action: finish the already frozen Phase-A pair. Independently prepare a
  structural repair control that freezes L0/L1 and uses one mass-validated
  upper router layer, with no recursive singleton summaries. Do not implement
  the old 239-card Phase B as the claimed method.
- Final action: `refine_mechanism`.

| Date | Triggering Evidence | Parent Idea | Supported | Weakened | Untested | Final Action | Next Minimum Experiment |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | variance-floor singleton audit; SkillAxe; ASSAY | IDEA-002C / IDEA-003A | community/posterior as possible utility prior | minsplit-only repair; recursive summaries; multi-parent-edge novelty | mass-validated router; equal-budget utility transfer | refine_mechanism | Phase A, then frozen-L1 structural control |

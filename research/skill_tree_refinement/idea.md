# DynaMix Skill-Tree Refinement

## Source and Scope

This idea is distilled from the user's July 10, 2026 research direction for
`/mnt/data/yaodong/codes/DynaMix2skill`. The source of truth is the user's
request plus live experiment artifacts; external papers may refine or falsify
this direction, but must not replace it with an unrelated project.

The project asks whether DynaMix can retain the useful community-level
experience extraction at L0 while replacing weak or unnecessary recursive
summarization with a simpler, more effective skill evolution mechanism.

## Current Evidence

- Fresh, current-protocol SpreadsheetBench static DynaMix:
  `97/200 = 48.5%` LibreOffice-recalculated heldout accuracy.
- Matched no-skill heldout baseline:
  `82/200 = 41.0%`, so the observed gain is `+7.5` percentage points.
- The same fresh run's no-skill train rollout is also `97/200 = 48.5%`, which
  shows that the current model/runtime is strong and that method gains must be
  interpreted against a high baseline.
- The fresh tree contains 7 build layers, 346 nodes, and 208 communities.
  Its root configuration uses `min_split_size = 4` and
  `min_effective_samples_per_component = 2`.
- Earlier controlled ablations used an older `90/200 = 45.0%` static anchor:
  hard-assignment GMM `82/200`, KMeans elbow `85/200`, fixed-K KMeans
  `80/200`, L0 single-card `90/200`, L1-only `89/200`, retrieve-L1-only
  `82/200`, and retrieve-L2+-only `75/200`.
- These ablations suggest that community-level extraction matters, but do not
  establish that deep recursive abstraction helps. In particular, L1-only
  nearly matched the older full hierarchy, while L2+-only was weaker.
- The old ablations and the fresh `48.5%` run are not directly interchangeable:
  records, run time, and baseline anchor differ. New claims require matched
  reruns under the fresh protocol.

## Core Research Question

Can we build a shallower, evidence-preserving, validated skill hierarchy that
beats the current `48.5%` static DynaMix result under an otherwise identical
SpreadsheetBench protocol?

## Working Hypotheses

### H1: Excessive recursive depth weakens experience quality

`min_split_size = 4` may permit many small upper-level communities and a deep
chain of lossy summaries. Increasing it to 8 or 10 may produce fewer, broader,
better-supported abstractions and reduce prompt clutter.

This is a testable structural hypothesis, not an assumed fix. The experiment
must measure tree depth, per-level node/community counts, singleton and support
distributions, summary specificity, retrieval overlap, and heldout task flips.

### H2: L0 community extraction is useful, but repeated N-to-1 summarization is weak

The GMM-BIC community step and multi-card L0 extraction may supply the useful
signal, while L2+ cards add little or lose operational detail. A better method
should preserve trace-backed evidence and avoid repeatedly compressing already
compressed cards without validation.

### H3: Skill evolution should be evidence-conditioned and validated

Recent work on skill evolution, agent memory, hierarchical retrieval, and
experience distillation may provide mechanisms such as contrastive failure
analysis, success/failure controls, provenance-preserving synthesis, utility
validation, or query-conditioned composition. Candidate mechanisms must be
mapped to DynaMix's current evidence before implementation.

## Initial Research Branches

### IDEA-001: Depth and granularity control

Compare `min_split_size` 4, 8, and 10 while keeping records, model, endpoint,
thinking, decoding, embedding, GMM-BIC, soft membership, evaluator, retrieval,
top-k, max turns, split, and denominator fixed. Start with build-only structural
diagnostics. Expand only promising settings to a paired heldout pilot, then a
full run if justified.

### IDEA-002: Evidence-preserving upper-level evolution

Survey recent skill-evolution and hierarchical-memory methods, then design one
minimal mechanism that improves upper-level cards without discarding supporting
L0/L1 evidence. Candidate mechanisms must expose a falsifiable difference from
the current repeated N-to-1 summary.

### IDEA-003: Remove components unsupported by matched evidence

Re-audit previous ablations under the fresh protocol. Components are removed
only when a controlled comparison shows no useful contribution or when they are
subsumed by a simpler mechanism. Historical mismatched runs are diagnostic,
not final evidence.

## Experiment Invariants

- Dataset: SpreadsheetBench verified 400, train `[0, 200)`, heldout `[200, 400)`.
- Primary evaluator: LibreOffice recalc; failures remain in the denominator.
- Model: `Qwen3.5-9B-AWQ` through the user-provided endpoint.
- Embedding: the same Qwen3-Embedding-8B endpoint/config as the fresh baseline.
- Rollout: same task prompt, tools, max turns, thinking mode, decoding,
  timeout/retry policy, and workers unless an explicit contract amendment is
  reviewed.
- Retrieval: same query, nodebank fields, top-k, and injection protocol unless
  the experiment explicitly isolates retrieval as the only changed variable.
- Reuse the fresh 200 train trajectories where the experiment concerns only
  tree construction; do not regenerate them without a separate rationale.
- Every run records command, config, commit/diff state, runtime, token usage,
  tree statistics, metrics, task-level flips, and known confounders.
- Do not report selected-slice or smoke results as full benchmark evidence.

## Resource Policy

- The generation endpoint is
  `https://evirdwimyrmm.10.27.127.9.nip.io/v1`, model
  `Qwen3.5-9B-AWQ`, API key label `dummy`, TLS verification disabled as required
  by the endpoint, and request timeout 300 seconds for service probes.
- Serious experiment wrappers may use up to 32 workers as authorized, but first
  run the smallest credible smoke or paired diagnostic that answers the current
  decision question.
- Never log secrets. The endpoint uses the non-secret placeholder key `dummy`.

## Success Criteria

- First gate: a complete, source-verified literature/closest-work map and an
  audit of all existing ablations against the fresh experiment protocol.
- Structural gate: a candidate produces a measurably cleaner tree and more
  evidence-specific cards without changing protocol invariants.
- Pilot gate: paired task-level improvement over the current method with no
  harness/evaluator/runtime confounder.
- Full gate: exceed `97/200` on the same heldout set and evaluator, with at least
  matched vanilla/current-DynaMix controls and a trace-level explanation of
  gains and regressions. One run is preliminary evidence, not a final paper
  claim; robustness requires repeated seeds/runs where stochasticity matters.

## Stop or Escalation Conditions

Pause for user review before changing the formal dataset split, evaluator,
baseline information access, model endpoint, task tools, retrieval query/index
semantics, or headline claim. Also pause before overwriting the baseline tree,
mixing results into an existing run directory, committing unrelated dirty-tree
changes, or launching a substantially larger compute campaign than one matched
200-train/200-heldout experiment.

Stop a branch when a fair paired diagnostic shows no signal twice, when the
mechanism only improves by changing information access or evaluator semantics,
or when literature shows the claimed novelty is already subsumed without a
meaningful DynaMix-specific distinction.

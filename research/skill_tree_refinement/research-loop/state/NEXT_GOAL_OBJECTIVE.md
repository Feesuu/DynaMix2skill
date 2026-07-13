# Next Goal Objective

## Active 16-Worker Objective

Keep the current goal active and continuously advance it with 16-worker task
rollouts. Complete the already queued same-protocol Phase-A dense-versus-routed
pair, then the frozen 239-L1 K18 dense-versus-routed pair. While shared vLLM
traffic prevents a clean launch boundary, continue train-only implementation,
matching, audit, and literature work rather than marking the goal blocked.

The next method gate is IDEA-003: compare GMM-community-conditioned evidence
with a source-disjoint, equal-count, equal-outcome, exact-token-budget matched
non-community control. Run candidate generation and utility validation only
after the construction preflight passes. Use 16 workers for task rollouts,
keep embedding concurrency at the frozen protocol value, and permit no heldout
labels during construction. Only a frozen candidate that passes the registered
train-side utility gate may proceed to a complete 200-task LibreOffice-recalc
evaluation against its matched dense baseline.

Continue the DynaMix skill-tree research loop until a better mechanism is
implemented and fairly validated, rather than stopping after one ablation or
because a shared endpoint is busy.

Use the current live repository and artifacts as authoritative. Preserve the
useful L0 trajectory-community to L1 ExperienceCard extraction, but challenge
the unsupported parts of the current method: recursive L2+ text summaries,
singleton-heavy upper GMM components, and similarity-only retrieval. Use
recent skill-evolution work including HDSO, GoSkills, Skill Retrieval
Augmentation, SkillAxe, ASSAY, and SkillComposer to identify what is already
known and what remains falsifiably distinct.

First complete the frozen, matched Phase-A comparison between dense all-node
retrieval and L2-community-router/L1-advice retrieval. Use 16 workers in both
arms, Qwen3.5-9B-AWQ, thinking=true, temperature=0, max turns 30, top-k 10,
the exact frozen 200 train records/tree/index, instruction plus task-type query,
Qwen3-Embedding-8B, and LibreOffice-recalc evaluation over heldout 200:400.
Keep cache disabled, never retry ambiguous read/stream timeouts, require stable
vLLM drain boundaries, and retain every failure in the 200-task denominator.
If ingress or endpoint failures recur, preserve the invalid run and continue
offline analysis or a pre-registered runtime repair; do not mark the whole goal
blocked while meaningful work remains.

Use the completed deterministic L1 structural sweep as a controlled diagnostic,
not as an accuracy claim or a min-effective-only replay of the historical tree.
The source tree used older unseeded PCA code and selected K=64 at L1. The new
control freezes the 239 L1 items and both stored embedding views, then refits
PCA/GMM with the current deterministic implementation; under that implementation,
min-effective values 6, 8, and 10 all select the same K=18 memberships. Build
the smallest non-generative K=18 routing layer with value 6 and no new LLM
summaries. Its controlled performance comparison is dense L1 retrieval versus
K=18 routed L1 retrieval over the exact same 239-node bank; comparisons to the
historical all-level dense or L2 router runs are informative but not single-
variable ablations.

For the next method candidate, do not claim hierarchy, top-k retrieval,
verification, rejection ledgers, or progressive disclosure as novel. Test the
narrow DynaMix hypothesis that saved GMM community/posterior geometry improves
experience generation or task-specific utility estimation. Compare
community-conditioned evidence against an equal-count, equal-outcome,
equal-token, instruction-type-matched non-community HDSO/ASSAY-style control.
Use source-disjoint train tasks for candidate validation and never use heldout
labels for construction, filtering, threshold selection, or prompt tuning.

Implement only the minimum code required by evidence. Use Codebase-Memory for
architecture impact, Serena for symbol/reference inspection, governed
development before edits, Ponytail to avoid unnecessary abstractions, and
independent spec/regression/research-protocol reviewers after nontrivial
changes. Preserve unrelated dirty worktree changes. Record commands, configs,
hashes, token/runtime usage, task-level positive/negative flips, evaluator
artifacts, rejected hypotheses, and claim boundaries. Do not git push until the
user reviews the resulting code and evidence.

The target is a same-protocol method that reliably exceeds the matched dense
baseline and preferably exceeds the historical 48.5% run, with improvements
supported by complete 200-task LibreOffice evaluation rather than selected
slices or runtime-confounded runs. If a candidate fails its pre-registered
gate, stop that branch, record why, and continue with the next evidence-backed
mechanism instead of declaring the research objective blocked.

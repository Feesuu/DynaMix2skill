# Experiment Contract: IDEA-002 Evidence-Routed Community Skills

Status: design frozen before implementation. Phase A is the next code change;
Phase B remains contingent on Phase A and its train-only preflight.

## Problem Established by Current Evidence

The current tree uses every `ExperienceCard` as both a hierarchy summary and an
executable instruction. That conflates two roles:

1. L2+ cards should organize and route lower-level experience; and
2. injected advice should contain concrete, trace-grounded procedures.

This role conflation is empirically visible. In the fresh minsplit4 tree,
58.9% of L2+ nodes have a same-level nearest-neighbor cosine of at least 0.90.
The minsplit8 build is one committed level shallower, but this fraction worsens
to 62.4%. Existing heldout retrieval is already 78.15% L1, while L3+ occupies
only 5.55% of slots. Similarity does not predict utility, and one inspected
negative flip shows a highly similar card inducing the wrong positional
operation.

The intervention therefore changes node roles and admission evidence, not the
GMM-BIC community discovery mechanism.

## Invariants

- Keep L0 GMM-BIC, PCA, budget refinement, and cumulative-mass membership.
- Preserve all community/member weights and multi-parent provenance.
- Keep the retrieval query exactly `instruction + instruction_type`.
- Keep heldout `top_k=10`, model, endpoint, tools, thinking, temperature,
  workers, max turns, registered completion cap, request timeout, and
  LibreOffice evaluator identical between arms.
- Do not use heldout labels, scores, or trajectories for construction,
  selection, thresholds, or admission.
- When re-exporting the unchanged baseline tree, copy and reuse the exact
  baseline skillbank index after document/protocol validation. Re-embedding
  identical cards is not allowed because observed service-level float drift
  would contaminate the retrieval comparison.
- Do not add PPO, a learned controller, an LLM reranker, or a new verifier.
- All new behavior is behind an explicit retrieval/admission strategy; the
  existing dense-all-nodes strategy remains a reproducible baseline.

## Pre-Outcome Protocol Amendment: Runtime Isolation

This amendment was made before any Phase A heldout rollout or score. Trace-level
audit found that the historical 97/200 dense run was not fully isolated:

- four heldout tasks referenced generic absolute `/tmp/*.py` files, and stale
  files owned by another user were executed after writes failed;
- thirteen heldout tasks created local `inspect.py` files that shadowed
  Python's standard-library `inspect` module and broke `openpyxl` imports;
- the same failure classes also occur in the train trajectories used by the
  frozen tree.

The routed comparison therefore cannot use 97/200 as its sole control. Both
arms must use the same minimal runtime-isolation patch: reject absolute `/tmp`
paths outside the current task directory, retain relative task paths, and warn
when a local script shadows an imported module. No task-specific spreadsheet
hint, tool, answer, query field, or evaluator behavior is added.

Run a fresh `dense_all_nodes` heldout control under this patched harness, then
run `l2_router_l1_advice` with the exact same tree, index, model, settings, and
harness. The existing tree is acceptable for this paired mechanism diagnostic
because both arms share it, but neither result is paper-grade evidence for a
fully clean training pipeline. If Phase A survives the paired gate, regenerate
the 200 train trajectories and tree under the isolated harness before making a
headline method claim.

A second pre-outcome runtime audit found that a public-ingress stream timeout
was retried as if it were a pre-send connection failure. One logical request
could therefore leave up to four backend generations, and the routed arm could
start while dense-arm generations were still running. The formal pair now
disables response-cache reuse, never retries an ambiguous read/stream timeout,
sets OpenAI SDK internal retries to zero, and records every actual request
attempt. Ordinary pre-send connection errors
retain the client retry capability, but any retry, timeout, or cached response
invalidates that arm. The launcher requires three consecutive idle vLLM metric
samples before dense, between arms, and after routed. These guards apply
identically to both arms and change no model or method variable.

## Pre-Outcome Protocol Amendment: Continuously Shared Endpoint

Before either Phase A arm emitted a model request, the idle gate was observed
for more than three hours with zero local DynaMix workers while the global
vLLM metrics continuously reported three or four running requests. The global
counters also advanced, proving that the endpoint was healthy but shared with
unrelated clients. Consequently, `running=0` is not an attainable or
attributable isolation condition on this endpoint.

The formal pair therefore uses `shared_endpoint_concurrent_v1`: launch dense
and routed arms in the same time window with eight workers per arm and a fixed
total client budget of sixteen. This replaces temporal separation with
counterbalanced exposure to the same external load. It does not change the
task set, tree, index, query, top-k, model, decoding, tools, timeout, or
LibreOffice evaluator. Cache remains disabled, ambiguous timeouts are never
retried, and any timeout, retry, nonzero arm exit, incomplete denominator, or
protocol mismatch invalidates the pair.

For this amendment, the frozen worker invariant means equal per-arm workers:
both arms use eight. The total client concurrency remains sixteen. This
explicitly supersedes the earlier sequential plan in which one sixteen-worker
arm would run at a time; results from the two execution shapes are not mixed.

The launcher records both arm intervals, requires their start times to differ
by at most five seconds, verifies that the intervals overlap, and samples the
global vLLM running/waiting counters throughout the pair. Global occupancy is
audit context rather than a validity gate because it cannot identify request
ownership. The abandoned sequential-idle run remains an audit artifact and
must not be mixed with the concurrent pair. This remains paired development
evidence, not paper-grade endpoint-isolated evidence.

The ingress currently requires disabled local TLS verification. This is
retained symmetrically in both arms and recorded as a transport confound; it is
not evidence of endpoint identity or endpoint-isolated execution.

## Pre-Outcome Protocol Amendment: Ingress Timeout Containment

The first concurrent frozen-query attempt was stopped before evaluation and
before inspecting any task-level pass/fail outcome. Its request-attempt ledgers
recorded repeated `InternalServerError` failures at exactly 600 seconds even
though the client timeout was 1200 seconds. The endpoint identifies its proxy
as `istio-envoy`; successful requests in the same window completed in at most
about 117 seconds, while the failures were terminated at `600.00-600.02`
seconds. This establishes an ingress route-timeout mismatch rather than a
configured DynaMix client timeout.

A second attempt sent Envoy's documented per-request override
`x-envoy-upstream-rq-timeout-ms: 1200000` in both arms. The endpoint accepted
the header on short probes, but the formal request ledgers still recorded three
dense and four routed failures at approximately 600 seconds. The ingress
therefore ignores the override or enforces a higher-level 600-second maximum.
The ineffective header implementation was removed rather than retained as a
dead protocol knob.

The next corrected pair keeps `thinking=true`, client timeout 1200 seconds, no
ambiguous-timeout retry, and all method settings unchanged, but explicitly
sets `max_tokens=16384` for every ReAct model call in both arms. This value is
more than 2.7 times the largest successful completion observed in the failed
uncapped attempts (5901 tokens), while bounding runaway reasoning below the
ingress wall. Runtime, generation-config, Stage-06, usage and pair audits all
require exactly 16384. Cap-hit counts are reported. Any timeout, retry, error,
incomplete denominator or protocol mismatch still invalidates the pair.

This is a symmetric transport-containment protocol change, not a method
improvement. It supports the new matched dense/routed comparison only; it is
not silently compared as protocol-identical to the historical uncapped 48.5%
run. Both stopped 600-second attempts remain infrastructure audit history and
are not mixed with the capped pair.

## Pre-Outcome Protocol Amendment: Frozen Heldout Query Embeddings

Before either arm emitted a heldout model request, concurrent retrieval
preflight exposed two reproducibility defects. First, the embedding service's
dynamic batching produced slightly different vectors for the same query
(maximum observed component difference about `2.33e-4`) even though cosine
similarity remained above `0.999998`. Second, the old Stage-05 selection hash
was computed from train records `[0,200)`, while Stage 06 actually runs
heldout dataset rows `[200,400)`. The old train-query hashes are therefore
invalid as heldout retrieval guards.

The amended pair freezes the exact 200 heldout query embeddings once before
outcome generation. The cache uses only `instruction + Task type`; it excludes
`answer_position`, workbook output, gold answers, evaluator labels, and model
responses. Embeddings are requested serially (`concurrency=1`, one query per
request) from the same Qwen3-Embedding-8B endpoint and then stored in a
hash-bound read-only artifact. Both arms must use that exact artifact, require
all 200 cache hits, and make zero live query-embedding requests during heldout
retrieval. Node/card embeddings and the existing nodebank indexes remain
byte-for-byte unchanged.

The frozen cache is
`runs/research_query_embedding_freeze_heldout200_20260711_192028/heldout_query_embeddings.json`
with SHA256
`60d77ea4ba6a7b0f062bdf5f1bab919f06d747c48572cd31ff58dccdf1e41571`.
It contains 200 unique query hashes and 4096-dimensional vectors. Under this
artifact the full-nodebank dense heldout selection SHA256 is
`ec58751f918f10d94d308457b1c65a705e7645f5a459ed6b2c191649c4c363dc`;
the prepared 239-card L1-only dense heldout selection SHA256 is
`9fbb4fd428fac4c5e9f21b2e3f471704db6b45c556d0b0ae6c04b1a630c7f896`.
This amendment changes only deterministic retrieval input materialization and
applies symmetrically. It does not inspect or use heldout outcomes.

## Phase A: L2 Routers, L1 Advice

Phase A reuses the exact existing minsplit4 hierarchy and changes retrieval
only. It does not regenerate trajectories, cards, communities, or embeddings.

For query embedding `q`:

1. Treat every level-2 card as a router identity, never as injected advice and
   never use its generated summary text as the router representation.
2. Recover each router's direct L1 descendants from
   `L2 item -> generated_from_community_id -> community.member_weights`.
3. Compute each router representation from its direct descendants:
   `normalize(sum_i membership_weight_i * normalized_L1_embedding_i)`. This
   tests the saved GMM community structure without trusting redundant L2 text.
4. Rank all L2 routers by cosine similarity between `q` and that weighted L1
   centroid.
5. Add routers in that order until the union of unique L1 descendants contains
   at least the existing `top_k` candidates. This avoids a new `router_k`
   hyperparameter.
6. Rank only that L1 candidate union by cosine similarity to `q` and inject the
   best 10 unique L1 cards.
7. If all valid routers are exhausted before ten candidates are available,
   backfill from globally ranked L1 cards and record the exact backfilled IDs.
8. Log selected routers, centroid scores, descendant edges/weights, candidate
   pool size, final L1 scores, and backfill count for every task.

All L3+ cards are excluded from selection and injection. They remain in the
tree artifact for audit but have no runtime role in Phase A.

The fresh baseline tree already contains 64 L2 routers and 239 L1 cards. Direct
L1 descendant count per L2 router has min 1, median 3, mean 3.73, and max 27.
All 239 L1 cards currently have a direct L2 parent; the implementation must
also preserve union semantics if a future tree has multiple parents.

## Phase A Pre-Heldout Gates

Run retrieval-only preflight over the exact 200 heldout queries without agent
rollout or outcome access:

- every query returns exactly ten unique L1 cards;
- no L2+ card appears in rendered `Retrieved Experience`;
- every selected L1 is a real descendant of a selected L2 router or is
  explicitly logged as backfill;
- query text and stored embedding protocol are byte-for-byte unchanged;
- source and Phase A skillbank-index SHA256 values are identical;
- router/candidate ordering is deterministic under equal scores;
- router scores are computed from membership-weighted L1 centroids, not from
  L2 summary embeddings;
- candidate-pool and backfill distributions are reported;
- baseline dense retrieval remains unchanged when the new strategy is off.
- every query embedding is read from the same frozen cache SHA, and the
  reconstructed query hash matches the heldout dataset row.

After tests and independent code review, run one full `[200,400)` dense control
and one full routed arm concurrently under `shared_endpoint_concurrent_v1`.
This is repeated development evidence, not an untouched test-set claim.

## Phase A Falsifiable Prediction

If the GMM community structure is useful but high-level summary text is not,
Phase A should reduce negative behavior flips without losing hierarchical
coverage. The decision statistic is LibreOffice hard accuracy on all 200
tasks, with all runtime failures retained in the denominator. Task-level
positive/negative flips against the fresh minsplit4 run must also be reported.

- Continue to Phase B if Phase A is non-inferior to the new matched dense
  control or if it reduces negative flips by at least 25% without losing more
  than two total passes.
- Stop the routing branch if it loses more than two passes and provides no
  compensating reduction in negative flips.

These thresholds are frozen before the Phase A heldout result.

## Phase B: Source-Excluded Counterfactual Admission

Phase B addresses the second observed failure: semantic relevance is not
empirical utility. It applies only to L1 advice cards and only after Phase A.

For each L1 candidate card:

1. Recover all source trajectories from its generating L0 community. Because
   membership is soft, the source set is the union of every trajectory with a
   nonzero stored membership weight.
2. From train tasks outside that source set, select deterministically the
   nearest baseline-failed task and nearest baseline-successful task by the
   frozen query/card embedding protocol.
3. Run each selected task twice under the same runtime: vanilla and with only
   that candidate card. Evaluate both outputs with LibreOffice recalc.
4. Record `help` (0->1), `hurt` (1->0), `preserve` (1->1), or `inert_fail`
   (0->0), plus both trajectories and evaluator artifacts.
5. Admit a card as executable advice only if it produces at least one `help`
   and zero `hurt`. Cards with no positive delta are `unsupported`; cards with
   any regression are `harmful`. Both remain auditable but are excluded from
   the primary nodebank.

This is deliberately strict and small: at most two paired task evaluations per
card, no learned credit model, and no heldout feedback. A candidate tested on
its own source task is not accepted as transfer evidence.

If either outcome class has no source-excluded train task, the card is marked
`insufficient_evidence` rather than silently admitted. The utility ledger is
metadata only and is not included in embedding text.

## Related-Work Boundary

- Unlike Trace2Skill's trajectory-local reroll, Phase B requires transfer to a
  source-excluded train task and retains overlapping community provenance.
- Unlike SkillOpt's monolithic skill-document selection, admission is per
  community card and routing is separated from executable advice.
- Unlike SkillAudit, DynaMix uses the available deterministic train verifier,
  does not claim ground-truth-free evolution, and does not use PACE or an LLM
  verdict as the commit gate.
- Unlike MemSkill, there is no learned selector/controller or RL loop.
- Unlike SkillX, semantic hierarchy roles are induced from DynaMix's community
  graph and validated L1 cards rather than a fixed planning/function/tool
  ontology.
- Unlike SkillPyramid, Phase A does not claim abstract-versus-atomic role
  separation or task-time skill composition as new; it only tests whether
  DynaMix's saved soft-community graph is a better router than its generated
  high-level text.
- Unlike SkillOps, Phase B does not claim utility tracking, redundancy
  maintenance, merge, retire, or typed validators as new; its narrower test is
  source-excluded deterministic-verifier admission tied to overlapping
  trajectory communities.

The possible research contribution is therefore narrow: source-excluded,
verifier-backed admission of procedural cards derived from overlapping soft
communities, combined with a strict separation between hierarchy routing and
runtime advice. Utility gating or hierarchy alone is not claimed as novel.

## Implementation Scope

Phase A should require only:

- an explicit selector strategy in `src/dynamix_trace2skill/skillbank.py`;
- exported ancestry needed for deterministic routing, preferably as a compact
  manifest field derived from the existing hierarchy state;
- exact compatible-index reuse in `scripts/export_dynamix_nodebank.py`, with
  source/output hashes recorded in the reuse contract;
- CLI/config plumbing in the existing experiment runner;
- focused tests in `tests/test_dynamix_reuse_contracts.py`;
- run/report logging for router and final advice selections.

Do not change `tree_builder.py`, GMM logic, analyst prompts, or the evaluator
for Phase A. Phase B gets a separate implementation plan only after Phase A's
gate, because it introduces real rollout cost and new artifacts.

## Claim Boundary

Phase A can test whether hierarchical routing is better than injecting every
summary level. Phase B can test whether train-side counterfactual transfer
evidence filters harmful cards. Neither proves broad cross-benchmark skill
generalization, untouched-test performance, or causal utility beyond the
paired tasks actually evaluated.

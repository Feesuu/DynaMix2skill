# Related Work and Mechanism Map

Status: initial source-verified map, 2026-07-10. This is not yet a complete
systematic review. Claims below are grounded in the linked paper or official
repository, not secondary summaries.

## Decision Summary

The literature does not support blindly deepening the current recursive
summary tree. The strongest recurring pattern is different:

1. extract trace-grounded, procedural candidates;
2. use success/failure contrast or semantic gradients to make the update
   specific;
3. validate or score candidates before retaining them;
4. prune redundancy and preserve compact, executable guidance;
5. use levels with different semantic roles, not merely repeated N-to-1
   summaries of the previous level.

This points to a DynaMix-specific refinement: keep L0 GMM-BIC communities, but
treat upper-level evolution as evidence-preserving candidate generation plus a
utility gate, rather than unconstrained recursive abstraction. This is only a
candidate direction until collision and feasibility analysis are complete.

## Closest Methods

### Trace2Skill

Source: [paper](https://arxiv.org/abs/2603.25158),
[official code](https://github.com/Qwen-Applications/Trace2Skill).

Mechanism:

- collect labeled success/failure trajectories;
- propose trajectory-local patches in parallel with separate success/error
  analysis;
- agentic analysts can inspect produced artifacts and validate fixes;
- hierarchically consolidate patches into portable procedures;
- the paper reports that semantic overlap and patch interference are real, and
  that selecting patch subsets can help but is validation-expensive.

Reported evidence:

- SpreadsheetBench uses 200 evolution and 200 heldout tasks, with results
  averaged over three seeds;
- the paper reports substantial deltas from human/parametric skills, and finds
  parallel consolidation generally stronger/faster than sequential editing;
- its own analysis warns that new patches can fix some tasks while regressing
  previously correct tasks.

What DynaMix can adopt:

- contrastive success/error analysis within an L0 community;
- artifact/result-aware diagnosis rather than generic summary;
- explicit duplicate/interference audit before upper-level consolidation.

What not to copy blindly:

- one monolithic skill artifact: DynaMix's nodebank is a different method;
- validation on test/heldout data;
- unrestricted many-to-one merging that reproduces the same lossy-summary issue.

Collision judgment: high conceptual overlap for trace-grounded diagnosis and
hierarchical consolidation. A DynaMix contribution must be specifically about
community-conditioned, multi-parent, utility-gated skill structure, not merely
"analyze trajectories and merge lessons."

### SkillOpt

Source: [paper](https://arxiv.org/abs/2605.23904),
[official code](https://github.com/microsoft/SkillOpt).

Mechanism:

- bounded patch-style edits from success and failure rollout batches;
- a strict held-out selection gate accepts only a strictly improved candidate;
- rejected edits become negative optimization history;
- a slow/meta update preserves durable lessons across epochs.

Reported evidence:

- the paper reports SpreadsheetBench 77.5 for its default optimizer setting;
- removing the rejected buffer lowers its reported SpreadsheetBench result
  from 77.5 to 72.9;
- removing both meta skill and slow update lowers it to 55.0;
- these numbers use SkillOpt's models, splits, harness, and evaluator and are not
  directly comparable to DynaMix's 48.5%.

What DynaMix can adopt:

- candidate acceptance based on a train-only selection signal;
- retain rejected-card evidence to prevent repeated bad summaries;
- bounded changes that preserve already useful procedural content.

What not to copy blindly:

- optimize one large skill document;
- create a new validation split without explicitly accounting for reduced
  training evidence and baseline fairness;
- treat their absolute score as a target under our different protocol.

Collision judgment: strong overlap for validation-gated evolution. A new method
must explain why community-local candidates and node-level utility differ from
SkillOpt's single-skill text optimizer.

### Skill-Pro

Source: [paper](https://arxiv.org/abs/2602.01869),
[official code](https://github.com/Miracle1207/Skill-Pro).

Mechanism:

- represents a skill with activation, execution, and termination conditions;
- derives structured semantic gradients by hindsight attribution;
- aggregates gradients across a batch to filter trajectory-specific noise;
- uses a PPO-style counterfactual gate before accepting candidates;
- maintains online utility scores and prunes non-positive/redundant skills.

Reported evidence:

- the paper reports 816 total skill tokens and 0.90 ALFWorld success in its
  setting, with component ablations for semantic gradients, gate, and scoring;
- its counterfactual gate relies on action likelihood/return information not
  currently emitted by DynaMix.

What DynaMix can adopt:

- distinguish trigger from executable procedure and verification/termination;
- aggregate community-level update evidence instead of plain summaries;
- maintain empirical utility metadata outside embedding text;
- prune redundant or consistently harmful cards.

What not to copy blindly:

- call an LLM score "PPO" without the required likelihood/advantage data;
- add a complex policy layer before a simpler reroll gate is tested.

Collision judgment: high for utility-gated procedural skills. A DynaMix version
should be presented as a simpler verifier-backed community skill gate unless it
actually implements the paper's counterfactual objective.

### SkillX

Source: [paper](https://arxiv.org/abs/2604.04804).

Mechanism:

- uses semantic levels with distinct roles: planning, functional, and atomic;
- extracts planning skills by removing exploration/backtracking noise;
- refines a library through merge/filter/update operations;
- retrieves planning skills first, rewrites a task-specific pseudo-plan, then
  retrieves lower-level skills;
- validates tool-grounded atomic skills against actual tool schemas.

Reported evidence:

- on Qwen3-32B the paper reports 63.67 Avg@4 on BFCL-v3 versus 53.67 no memory,
  and gains on AppWorld and tau2-Bench;
- its analysis says the best level combination depends on the base model, and
  warns text-only optimization can overfit with limited training data.

What DynaMix can adopt:

- levels should have explicit semantic responsibilities rather than simply
  summarizing the previous level;
- remove exploratory dead ends from successful procedural cards while retaining
  failure evidence separately;
- validate any tool-level statement against the actual environment.

What not to copy blindly:

- a tool-schema-specific three-level ontology as a universal hierarchy;
- pseudo-plan rewriting before proving current retrieval is the bottleneck;
- four rollouts per training task without a compute/decision justification.

Collision judgment: very high for hierarchical skill libraries. A DynaMix
contribution cannot be just "three levels of skills"; it must tie levels to
community evidence, soft membership, or validated abstraction.

### ReasoningBank / MaTTS

Source: [paper](https://arxiv.org/abs/2509.25140).

Mechanism:

- distills strategy-level memory from successes and failures;
- parallel multiple trajectories for the same query provide direct contrastive
  evidence about which choices cause success or failure;
- aggregation is important; independently storing every trajectory is noisy.

Reported evidence:

- on SWE-Bench-Verified, the paper reports 38.8 versus 34.2 no-memory for
  Gemini-2.5-flash, and 57.4 versus 54.0 for Gemini-2.5-pro;
- its correctness labels can use LLM-as-judge, which the paper acknowledges may
  be noisy. DynaMix has a stronger spreadsheet verifier and should use it.

What DynaMix can adopt:

- within a mixed-outcome L0 community, contrast successful and failed traces
  directly instead of summarizing them independently;
- use the LibreOffice result as the correctness signal;
- aggregate recurring causal differences into one candidate card.

Collision judgment: high for contrastive success/failure memory. DynaMix's
specific opening is verified community-level contrast across different but
semantically clustered tasks, rather than test-time scaling on one query.

### CoEvoSkills, MUSE-Autoskill, and MetaSkill-Evolve

Sources: [CoEvoSkills](https://arxiv.org/abs/2604.01687),
[MUSE-Autoskill](https://arxiv.org/abs/2605.27366), and
[MetaSkill-Evolve](https://arxiv.org/abs/2607.05297).

Useful signals:

- CoEvoSkills co-evolves a surrogate verifier with the skill generator;
- MUSE-Autoskill treats skills as lifecycle-managed, testable assets with
  skill-level memory and runtime feedback;
- MetaSkill-Evolve separates fast task-skill updates from slow meta-skill
  evolution and reports gains over raw backbones on OfficeQA, SealQA, and
  ALFWorld.

Why not first:

- co-evolving a verifier or a five-agent meta-skill pipeline is much more
  machinery than DynaMix currently needs;
- SpreadsheetBench already supplies a strong executable verifier;
- these are valuable future baselines/related work, but violate Ponytail's
  smallest-falsifiable-change principle as the first implementation.

### Skill1, SkillOS, and SkillsVote

Sources: [Skill1](https://arxiv.org/abs/2605.06130),
[SkillOS](https://arxiv.org/abs/2605.06614), and
[SkillsVote](https://arxiv.org/abs/2605.18401).

Mechanism distinctions:

- Skill1 trains one policy to jointly credit skill selection, use, and
  distillation from task outcomes. DynaMix currently freezes retrieval while
  changing construction, so it must not claim joint co-evolution.
- SkillOS learns a curator from delayed downstream reward over related task
  streams. A DynaMix verifier gate would be a non-learned, community-local
  curator and should be named accordingly.
- SkillsVote attributes outcomes among skill use, exploration, environment,
  and result signals, then admits reusable discoveries through evidence-gated
  updates. This is a strong collision with generic "validated skill admission."

Implication for VCCS:

Evidence gating, provenance, lifecycle management, and credit assignment are
not novel by themselves. The remaining possible distinction is narrower:
verifier-backed contrastive skill admission over overlapping GMM communities,
with source trajectories and multi-parent support preserved. That distinction
still needs a formal objective and direct collision test before any novelty
claim.

### SkillAudit and MemSkill

Sources: [SkillAudit](https://arxiv.org/abs/2606.14239) and
[MemSkill](https://arxiv.org/abs/2602.02474).

SkillAudit is the strongest direct collision discovered in the second pass. It
runs the same task with and without a candidate skill, uses twelve
process-aligned evaluators to localize harmful skill passages, combines them
with a fixed structural verifier, and commits or rolls back edits. Its stated
contribution is specifically ground-truth-free evolution; it does not use a
hidden test or external reward during evolution.

MemSkill learns a controller for top-k memory-skill selection, mines a bounded
hard-case buffer, periodically evolves a shared skill bank with an LLM
designer, and rolls back regressions. It is substantially more machinery than
the next DynaMix experiment and already occupies the learned-selector route.

Consequences for DynaMix:

- do not claim paired execution, rollback, a hard-case buffer, or a learned
  selector as new;
- use SpreadsheetBench's deterministic train verifier honestly rather than
  imitating a ground-truth-free PACE judge;
- distinguish source-task self-consistency from transfer evidence;
- the remaining narrow opening is source-excluded, verifier-backed utility for
  cards induced by overlapping communities, plus separating router nodes from
  executable advice.

The first minimal test should not implement PACE, twelve judges, PPO, or a new
designer. Those additions would increase cost without addressing the observed
role conflation in the current tree.

### AgentSkillOS

Source: [paper](https://arxiv.org/abs/2603.02176),
[official code](https://github.com/ynulihao/AgentSkillOS).

AgentSkillOS recursively categorizes large skill collections into a capability
tree and composes retrieved skills as DAGs. It strengthens the case that
structured retrieval can matter, but also means a DynaMix contribution cannot
rest on "organize skills in a tree" alone. DynaMix must show that its
experience-derived overlapping communities and verified update rule add value
beyond tree indexing and orchestration.

### SkillRet and Skill-as-Pseudocode

Sources: [SkillRet](https://arxiv.org/abs/2605.05726) and
[Skill-as-Pseudocode](https://arxiv.org/abs/2605.27955), with the latter's
[official code](https://github.com/InternLM/Skill-as-Pseudocode).

SkillRet isolates retrieval as a first-class problem over 17,810 public agent
skills and disjoint train/evaluation skill pools. Its strongest reported signal
is that task-specific retrieval training improves NDCG@10 substantially over
off-the-shelf retrievers because the useful skill signal is sparse inside long,
noisy queries. This supports DynaMix's observed result that raw cosine scores do
not separate positive from negative behavior flips. It also means generic
"better skill retrieval" is not a novel contribution and that any future
learned retriever requires a separate, fair contract.

Skill-as-Pseudocode converts clustered free-form procedural passages into typed
pseudocode contracts, applies deterministic coverage/binding/replacement/risk
checks, and restores concrete invocation templates. Its paired ALFWorld result
supports the claim that a relevant prose skill can still be hard to execute.
This is directly relevant to DynaMix's generic L1 cards, but typed contracts,
deterministic skill checks, and prose-to-pseudocode refactoring are occupied
ideas. They are a possible representation ablation after utility admission,
not the current novelty claim.

Consequences for the current sequence:

- finish the frozen Phase-A router/advice test before changing representation;
- if Phase A fails, do not conclude that hierarchy is useless until task-level
  traces distinguish retrieval mismatch from non-executable card text;
- retain source-excluded LibreOffice outcome evidence as Phase B's differentiator;
- consider typed procedural contracts only as a controlled later ablation, not
  as an unreviewed extra component in Phase A or Phase B.

### SkillPyramid, SkillOps, and CoEvoSkills

Sources: [SkillPyramid](https://arxiv.org/abs/2606.03692),
[SkillOps](https://arxiv.org/abs/2605.13716), and
[CoEvoSkills](https://arxiv.org/abs/2604.01687).

SkillPyramid is a direct collision with generic hierarchy-role separation. It
constructs downward atomic skills for reusable executable operations and upward
abstract skills for task structure. At task time it first uses abstract skills
to construct a framework, then lower-level skills to instantiate executable
details. Its DeepSeek-V3.2 ablation reports smaller but nonzero losses from
removing either atomic or abstract skills, and a larger loss from disabling
self-evolution. Phase A's L2-router/L1-advice split is therefore a diagnostic
adaptation to DynaMix's observed redundancy, not a standalone novelty claim.

SkillOps is a direct collision with generic library maintenance. It represents
skills as typed contracts over preconditions, operation, artifacts, validators,
and failure modes; connects them with dependency, compatibility, redundancy,
and alternative edges; and applies merge, repair, retire, validator, and
adapter actions using observable utility and failure logs. Its plug-in gains
over retrieval baselines are reported as 0.68--2.90 percentage points. DynaMix
cannot claim utility logging, redundancy removal, typed validation, or skill
retirement as new by themselves.

CoEvoSkills provides another strong result that one-shot skill generation is
not enough: a skill generator iterates against an information-isolated
surrogate verifier and an opaque ground-truth oracle signal. Its ablation
attributes most of the gain to iterative verification rather than a stronger
generation prompt. SpreadsheetBench already gives DynaMix a deterministic
train verifier, so inventing a surrogate verifier would be unnecessary and
would weaken rather than strengthen the evidence source.

Consequences:

- treat Phase A as a falsifiable retrieval intervention, not the final research
  contribution;
- do not claim abstract/atomic role separation, hierarchy, validation, utility
  maintenance, merge, retire, or co-evolution in isolation;
- retain the narrow Phase B distinction only if experiments support it:
  source-excluded deterministic-verifier admission for procedural cards induced
  from overlapping soft communities with multi-parent provenance;
- if that distinction fails empirically, the honest conclusion is that current
  community induction is not adding enough value, not that another unvalidated
  meta-agent is needed.

### HDSO, GoSkills, and Skill Retrieval Augmentation

Sources: [HDSO](https://arxiv.org/abs/2606.22330),
[GoSkills](https://arxiv.org/abs/2605.06978), and
[Skill Retrieval Augmentation](https://arxiv.org/abs/2604.24594).

HDSO is now the closest collision to the planned Phase-B admission loop. It
treats every persistent skill update as a falsifiable hypothesis, runs matched
control/treatment executions on train tasks, measures treatment-only versus
control-only wins, checks out-of-scope guardrail regressions, confirms on
independent train indices, and preserves rejected hypotheses. Its validation
stages use 4, then 8, then 32 target tasks plus guardrail tasks. Therefore
paired verifier admission, a rejection ledger, net utility, source-independent
confirmation, and progressive disclosure are occupied contributions. The
current DynaMix feasibility plan of one source-excluded success and one failure
control per card is weaker evidence and must not be presented as a stronger
general admission method.

GoSkills is a direct collision with generic role-structured retrieval. It
renders compact groups with Start, Support, Check, and Avoid roles instead of
flat atomic skill lists. Phase A remains a useful DynaMix diagnostic, but L2
routing, grouped skill presentation, and router/advice role separation are not
standalone novelty claims.

Skill Retrieval Augmentation decomposes retrieval, incorporation, and final
execution. Its main finding is that agents load skills at similar rates even
when no gold skill is retrieved or no external capability is needed. This
matches DynaMix task `55468`: the top retrieved card named the exact column-index
risk, yet the executor still miscomputed `AE` as column 29 and wrote a
non-formula string. Retrieval relevance is therefore not the final bottleneck.

Consequences for DynaMix:

- do not implement the old Phase B as a novelty claim;
- if Phase A passes, use HDSO-style paired validation as a strong reference or
  baseline, not as a DynaMix invention;
- the remaining testable distinction is whether overlapping GMM communities
  improve candidate generation, source-disjoint validation strata, and
  multi-parent provenance relative to non-community candidate generation;
- any community-conditioned admission result needs a controlled non-community
  HDSO-style baseline under the same rollout budget;
- progressive disclosure or Start/Support/Check/Avoid rendering may be useful
  engineering variants, but they are occupied by HDSO/GoSkills and require
  explicit attribution.

### SkillAxe, ASSAY, and SkillComposer

Sources: [SkillAxe](https://arxiv.org/abs/2606.10546),
[ASSAY](https://arxiv.org/abs/2606.15390), and
[SkillComposer](https://arxiv.org/abs/2606.32025).

SkillAxe is directly relevant because it reports SpreadsheetBench, but its
headline must not be compared numerically with DynaMix without the protocol
details. It uses Claude Opus 4.5, OfficeJS/Excel COM, a random seed-42 split,
optional tool-loaded skills, and reports 52.0% on the common 50-task subset
where all conditions completed. Both its 69-skill LLM-self library and its
22-skill refined library score the same 52.0%; the demonstrated benefit is
library compression and higher activation, not an accuracy gain over its own
LLM-self library. Its three-zone FORK/IMPROVE/SKIP construction and four
diagnostics are useful references, but the 16-to-52 headline is not a matched
baseline for our 200-task Qwen/LibreOffice protocol.

ASSAY is a stronger explanation for DynaMix's negative flips. It estimates a
per-skill/per-development-task causal matrix using 12 randomized skill masks on
15 development tasks (180 rollouts per model), then uses task-conditioned
masking to suppress skills predicted to hurt. Its ablation attributes +7.5
points to per-task masking, larger than its offline restructuring gain. This
occupies generic causal skill masking and confirms that globally plausible
skills can have opposite effects on different task regions.

SkillComposer jointly predicts skill subset, count, and order. It occupies the
generic claim that a learned structural composer is better than independent
top-k retrieval. DynaMix should not add a learned composer before showing that
its community structure provides a cheaper or better-conditioned utility
signal.

Consequences for DynaMix:

- do not cite SkillAxe's 52% as a same-protocol target or as proof that its
  refinement improves accuracy over its own generated library;
- do not claim per-task causal masking, randomized skill attribution, or
  subset/count/order composition as new;
- the remaining possible opening is a lower-cost community-conditioned utility
  model: use DynaMix's saved posterior/community provenance to choose validation
  strata and transfer measured skill utility, then compare directly with a
  non-community HDSO/ASSAY-style control at the same rollout budget;
- first repair the current singleton-heavy upper clustering, because a
  community-conditioned method is not credible if its communities are
  variance-floor spike components.

## Older Structural Memory Context

- [A-MEM](https://arxiv.org/abs/2502.12110) dynamically links and revises notes;
- [G-Memory](https://arxiv.org/abs/2506.07398) combines high-level insights with
  condensed lower-level interaction traces in a three-tier graph.

These support provenance links and multi-resolution retrieval, but their tasks
and memory objects differ from executable spreadsheet skills. They motivate
preserving links from abstract cards to concrete evidence; they do not by
themselves justify deeper recursive summarization.

## Candidate DynaMix Mechanism After This Pass

Working name: **Community-Conditioned Skill Utility (CCSU)**.

Minimal form:

1. Keep current L0 embeddings, GMM-BIC, cumulative-mass multi-parent assignment,
   and token-budget refinement.
2. Replace singleton-heavy recursive abstraction with at most one validated
   upper community layer used for routing, not injected prose.
3. For each retained community, use verified success/failure labels and full
   traces to generate a small number of falsifiable procedural candidates.
4. Validate candidates on source-disjoint training evidence. HDSO-style paired
   execution is a baseline; the DynaMix variable is community-conditioned
   candidate generation and validation-stratum construction.
5. Store utility as metadata and predict task-conditioned utility from the
   community/posterior representation. Compare this against an equal-budget
   non-community control and against plain semantic retrieval.
6. Keep only candidates with positive net utility and no guardrail regression;
   preserve rejected candidates for audit.

This candidate combines known ingredients, so novelty is not yet established.
The possible contribution is now narrower: soft GMM posterior structure as a
low-dimensional prior for generating, validating, and routing skill utility.
It must beat non-community HDSO-style admission and ASSAY-style task-neighbor
utility transfer at the same rollout budget. The current run has only three
L0 trajectories with more than one selected parent, so novelty cannot rest on
selected multi-parent edges alone; it must use and validate the saved posterior
geometry.

## What the Literature Says to Avoid

- deeper hierarchy without semantic role separation;
- accepting every LLM-generated summary;
- evaluating skill candidates on heldout test tasks;
- storing every lesson without redundancy/interference control;
- adding meta-agents before showing that a simple verifier gate helps;
- claiming PPO, co-evolution, or recursive self-improvement without implementing
  their defining optimization signal.
- treating higher embedding similarity as proof that a skill is useful or
  executable;
- claiming typed/pseudocode skill refactoring as new without comparing against
  Skill-as-Pseudocode.

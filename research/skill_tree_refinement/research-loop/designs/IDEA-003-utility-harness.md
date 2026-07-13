# IDEA-003 Minimal Utility Harness Design

Status: implementation design only. Candidate generation and task rollout remain
gated on completion of the frozen-L1 K18 matched retrieval pair.

## Objective

Execute the already reviewed `preflight_final.json` without introducing a new
agent, evaluator, retrieval policy or hidden prompt difference. Generate 18
paired candidates and evaluate each pair on its frozen source-excluded success
and failure tasks under no-card, source-evidence-card and matched-evidence-card
conditions.

## Candidate Generation

1. Read only `preflight_final.candidate_prompts.json` and require the hash stored
   in `preflight_final.json`.
2. Submit its 36 already rendered message lists through the existing
   `GenerationClient.chat_json` with the frozen Qwen3.5-9B-AWQ endpoint,
   `thinking=true`, temperature `0.6`, the recorded schema, no response cache,
   zero JSON repair retries and the existing global concurrency semaphore.
3. Validate exactly `name`, `trigger`, `content`, `confidence`; retain generation
   failure as an invalid pair rather than falling back or regenerating evidence.
4. Persist request attempt, usage, raw response, parsed card, prompt hash and
   pair validity before any task rollout.

## Identical Agent Conditions

The no-card condition must not use `cli_only`, because its system template is
not byte-identical to `cli_skill_preloaded`. Add one narrow research-harness
input to the existing preloaded agent: a fixed retrieved-experience payload that
may be either empty or one validated card.

- Environment/CLI input absent: existing nodebank retrieval behavior is exactly
  unchanged.
- Explicit empty payload: render the normal preloaded system template with an
  empty `<retrieved_experience>` block.
- Explicit one-card payload: render the same template with that card only and
  bypass dynamic retrieval.
- Reject simultaneous fixed payload and nodebank retrieval settings.
- Log the fixed payload hash and selected condition, never a hidden fallback.

This isolates the treatment to experience content while keeping model, task
prompt, tools, answer-position access, filesystem sandbox, max turns and system
template identical.

## Rollout Scheduling

Materialize 108 immutable job descriptors before launch:

- 18 community pairs;
- two frozen validation tasks per pair;
- three conditions per task;
- deterministic counterbalanced condition order by community rank;
- one task-local work/output/log directory per descriptor.

Run descriptors with one global `asyncio.Semaphore(16)`. Each descriptor invokes
the existing SpreadsheetBench task execution path with one instance ID and one
worker, so total model concurrency is 16 rather than 16 per subprocess. Use
temperature `0.0`, thinking `true`, max turns `30`, timeout `1200`, disabled
cache, one transport attempt and no ambiguous-timeout retry.

## Evaluation

Reuse the current LibreOffice-recalc evaluator functions for every produced
workbook. Record recalc hard pass as primary and cached-value pass as raw audit.
Missing workbook, timeout, parser failure and evaluator error stay in the fixed
108-job denominator. The fresh no-card outcome for the same pair/task is the
only help/hurt baseline.

## Fail-Closed Gates

- Require final preflight, prompt-sidecar, posterior-sidecar and script hashes.
- Require 18 valid construction pairs and exactly 108 descriptors before launch.
- Require fixed task IDs and source-exclusion proofs from the reviewed artifact.
- Reject payload/template/model/temperature/thinking/max-turn/worker drift.
- Reject cache hits, retries, duplicate job IDs, shared workdirs or missing
  attempt/evaluator artifacts.
- Compute the registered utility gate only after all planned jobs terminate.

## Minimal Code Impact

Likely behavioral changes after the K18 gate:

1. one explicit fixed-experience branch in
   `spreadsheet_agent/agents/cli_skill_preloaded_agent.py`;
2. one matching CLI validation path in `run_spreadsheetbench.py`;
3. one IDEA-003 orchestration script and targeted tests.

No change is needed in GMM, tree construction, nodebank selection, dynamic
update, task prompt, tools or evaluator semantics. Independent spec, regression
and research-protocol review is required before the 36+108 requests start.

# Experiment Contract: IDEA-003 Community-Conditioned Utility

Status: pre-registered design. Implementation and rollout are contingent on
completion of the frozen-L1 K18 matched retrieval pair.

## Research Question

Does DynaMix's saved GMM community/posterior geometry improve the quality or
train-side transfer utility of an abstracted experience, beyond what can be
obtained from an equal-budget collection of semantically similar L1 cards that
does not share the community?

This is narrower than claiming that hierarchy, retrieval, validation, or skill
evolution is novel. The falsifiable mechanism is community-conditioned evidence.

## Frozen Inputs

- The 200 train tasks and their LibreOffice-recalc outcomes.
- The 239 existing L1 cards and both frozen embedding views.
- The deterministic reprojected K18 GMM fit identified by membership fingerprint
  `1946e7f87eecd29adc92d0a342b7d54ee7f09d54f5e30fe36f05855c6d8893ad`.
- The query policy `instruction + Task type`; `answer_position` remains excluded.
- No heldout task, label, trajectory, retrieval result, or evaluator output may
  influence construction, matching, task selection, prompts, thresholds, or
  admission.

## Paired Evidence Sets

For each of the 18 K18 communities, construct one true set and one matched-null
set.

### Blinding Amendment (2026-07-11, Before Candidate Generation)

The first executable preflight showed that some frozen L1 card text itself
contains train task identifiers or the condition-label words prohibited below.
Before any token counting, matching or prompt rendering, apply the same
deterministic sanitizer to every L1 card in both arms:

- replace every exact frozen train trajectory ID with `<trajectory_identifier>`;
- replace every exact frozen train task ID with `<task_identifier>`;
- replace standalone case-insensitive `true`, `control`, and `community` with
  `<boolean_value>`, `<comparison_group>`, and `<evidence_group>` respectively.

The sanitizer uses only the already frozen train identity set and the original
blinding rule; it uses no outcome, candidate, validation or heldout result.
Token lengths, pair matching, exact quotas and final prompt hashes are computed
after sanitization. Raw card text remains available only in the frozen source
artifact and is never sent to the candidate analyst. This amendment implements
the preregistered blinding requirement rather than changing its research
hypothesis or decision threshold.

### True Community Set

- Include every L1 card with a selected cumulative-mass edge to the community.
- Preserve the selected membership weights and deterministic item-ID tie order.
- Record the full posterior separately for audit, but do not place unselected
  posterior values in the analyst prompt.

### Matched-Null Set

Break community co-membership while preserving the observable evidence budget.
For each true member, select one unused replacement L1 card outside the target
community with the following deterministic priority:

1. same source outcome mode;
2. same primary source task instruction type when available;
3. closest token length and support-mass quantile;
4. highest embedding similarity, with item ID as the final tie-break;
5. target-component posterior no greater than the median posterior among all
   cards outside the selected community.

Copy the true set's sorted membership-weight list onto the matched replacements.
This keeps count and weight shape fixed while breaking actual community identity.
A replacement cannot be reused within the same null set.

### Exact Budget Matching

- Pair true and matched cards one-to-one in deterministic order.
- Allocate each pair `min(true_tokens, matched_tokens)` evidence tokens.
- Use the same tokenizer, rendering, truncation side and special-token policy.
- The final analyst prompts must have exactly equal evidence-token counts and
  the same fixed prompt template; otherwise the pair is invalid.
- Community IDs, task IDs, gold answers, evaluator outcomes and the words
  `true`, `control`, or `community` are not exposed to the analyst.

## Candidate Generation

Generate exactly one ExperienceCard from each evidence set with the same model,
thinking setting, temperature, JSON schema, timeout and output budget. The
prompt asks for a concrete transferable procedure grounded only in the supplied
cards. It must return only `name`, `trigger`, `content`, and `confidence`.

There are 18 true candidates and 18 matched-null candidates. Generation failure
does not trigger a semantic fallback; the pair is marked invalid and retained
in the denominator/audit.

## Source-Excluded Utility Tasks

For each true community:

1. Recover the union of all train task IDs that generated any selected true
   member card, following all nonzero L0 provenance edges.
2. Exclude this source union from validation-task selection.
3. Embed the remaining train queries with the frozen query protocol.
4. Using the true community centroid, select the nearest original-train success
   and nearest original-train failure. Selection is independent of generated
   candidate text and is reused for both candidates.
5. If either outcome class is unavailable, record `insufficient_evidence` for
   the whole community pair rather than substituting a source task.

This yields at most 36 validation tasks. Every selected task is run under three
conditions with identical agent/tool/runtime settings:

- no injected card;
- true-community candidate only;
- matched-null candidate only.

Condition order is deterministically counterbalanced across communities. Each
output workbook is evaluated with LibreOffice recalc. The fresh no-card reroll,
not the historical trajectory label, is the causal baseline for help/harm.

Maximum planned rollout cost is 108 task runs plus 36 analyst generations.
Use 16 workers for task rollout and the existing global generation concurrency
guard. No result cache or ambiguous-timeout retry is allowed.

## Metrics

For each condition and task, record hard pass, raw audit pass, runtime, token
usage, trajectory, workbook, evaluator artifact, request-attempt ledger and
source-exclusion proof.

Relative to the fresh no-card result:

- `help`: 0 -> 1;
- `hurt`: 1 -> 0;
- `preserve`: 1 -> 1;
- `inert_fail`: 0 -> 0.

Primary development statistic:

`net_utility = help_count - hurt_count`

Compare true versus matched-null on the same task IDs. Also report paired hard
accuracy, success-task preservation, failure-task correction, exact task-level
flips, and a paired permutation interval. Report all invalid/runtime failures in
the planned denominator; do not silently drop them.

## Decision Gate

Continue to a full 200-task heldout candidate only if all conditions hold:

1. true-community net utility exceeds matched-null by at least three task
   outcomes over the complete valid source-excluded set;
2. true-community `hurt_count` is no greater than matched-null `hurt_count`;
3. at least half of the 18 community pairs produce valid candidates and both
   source-excluded outcome controls;
4. no protocol, source-exclusion, cache, retry, evaluator or artifact-integrity
   violation is found.

Failure of this gate rejects community-conditioned generation as the next
method branch. It does not prove communities are useless for routing.

If the gate passes, freeze the admitted candidate artifact before any heldout
run. The eventual 200-task evaluation must compare against a token-matched dense
control under the same model, endpoint, workers, top-k, runtime and LibreOffice
evaluator. Heldout results may evaluate the frozen decision but may not modify it.

## Claim Boundary

This experiment can support only the claim that, on the specified train-side
source-excluded controls, shared K18 community evidence produced candidates with
different paired utility from matched non-community evidence. It cannot establish
untouched generalization, broad skill-evolution novelty, or a headline benchmark
improvement without the later frozen full evaluation.

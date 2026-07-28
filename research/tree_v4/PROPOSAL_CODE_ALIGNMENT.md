# V4 Proposal-to-Code Alignment

## Identity

| Proposal object | Implementation | Runtime artifact |
|---|---|---|
| Immutable Experience Atom | existing `ExperienceAtomAnalyst` reused by `evidence_balanced_skill_pipeline.py` | `experience_atoms.json` |
| Balanced online metric tree | `BalancedMetricTreeState` | `balanced_tree_state.json` |
| Deployable Skill Capsule | `SkillCapsule` and `SkillCapsuleRegistry` | `skill_capsules.jsonl` |
| Strict online state | `EvidenceBalancedOnlineSession` | per-arrival checkpoint with prefix and artifact hashes |
| Dynamic local update | sequential `insert`, then `_refresh_capsules` on evidence-changed ancestors | insertion/refresh events and checkpoints |
| Policy-dependent arrival | existing SpreadsheetBench runner + LibreOffice evaluator + existing Atom analyst | per-task rollout, evaluation, selection, record, and checkpoint artifacts |
| Retrieval exclusion of raw evidence | `_write_nodebank_manifest` exports active capsules only | `skills/node_bank_manifest.json` |

## Structural Claims And Checks

| Claim | Code check | Test |
|---|---|---|
| non-root occupancy in `[ceil(B/2), B]` | `BalancedMetricTreeState.validate` | randomized insertion property test |
| equal leaf depth | `validate` and `structural_audit` | randomized insertion property test |
| each atom exactly once, leaves only | `validate` | placement and randomized tests |
| certified covering upper bounds | leaf exact radius plus recursive internal bound | every insertion in randomized test |
| `O(Bh)` local insertion mutation | affected/retired IDs include the path, split nodes, and directly reparented children | randomized locality test |
| exact metric nearest search | branch-and-bound lower bound | brute-force equivalence tests |
| logarithmic height under occupancy invariant | `height_bound` plus audit | randomized insertion property test |

The guarantees above are data-structure guarantees. They do not prove that an
embedding is semantically correct, that an LLM capsule is true, or that heldout
performance improves.

Property tests call full validation after each insert. Production code
maintains radii from at most `B` local entries per level, then runs the
expensive descendant audit after each strict online arrival and before
export.

## Skill Promotion

- A leaf with one Atom is never summarized or exported.
- A leaf capsule needs at least two evidence atoms.
- Leaf generation separates recorded successful atoms from failed atoms;
  failures may define root causes or guardrails but are not copied as actions.
- A parent needs at least two distinct active child capsules.
- Deterministic validators reject empty fields, unsupported provenance,
  leakage patterns, and exact normalized child duplicates.
- A second structured LLM audit accepts a parent only when multiple children
  support it, it adds a cross-child abstraction, and it does not copy
  example-specific content.
- Rejected candidates remain in lifecycle artifacts and never enter the
  nodebank.
- The lifecycle ledger records `candidate` before the final `active` or
  `rejected` event.
- Semantic rejection, validation rejection, runtime generation errors, and
  prompt-budget errors are counted separately. Runtime and budget errors block
  heldout rather than masquerading as semantic decisions.
- Capsule validation does not perform a separate counterfactual behavioral
  replay. The closed-loop setting instead evaluates each naturally arriving
  train task with LibreOffice before that task becomes new tree evidence.

## Two Online Settings

Both settings call the same `BalancedMetricTreeState.insert` operation in
dataset order and use the same state transition:

```text
arrival -> Atom -> insert -> archive stale capsules
        -> bottom-up dirty-path refresh -> validate -> atomic checkpoint
```

`open_loop_replay` starts from an empty tree and consumes fixed historical
train trajectories. Current skills never change a later train trajectory.
Frozen Atoms are allowed only as a policy-independent cache; every tree,
capsule, and checkpoint artifact is restricted to the committed prefix.

`closed_loop_skill_evolution` loads a fingerprint-matched open-loop prefix
checkpoint. Each later train task first retrieves from the current nodebank,
runs the existing SpreadsheetBench agent, receives the authoritative
LibreOffice-recalculated outcome, and is then converted to one Atom and
inserted. The selected capsule IDs and outcome are preserved as exposure
audit, not causal credit.

There is no structural batch. Each task must finish rollout, evaluation, Atom
extraction, tree update, capsule refresh, validation, and checkpoint before the
next task can observe the state. Fingerprinted resume rejects a different Atom
protocol or a nonmatching record prefix.

`affected_node_ids` audits every structural pointer/radius change, while
`capsule_refresh_node_ids` excludes children whose parent pointer changed but
whose evidence set did not. The `O(B h d)` bound applies to structural
insertion and certified-radius maintenance only. Capsule refresh may inspect
an active subtree frontier in the absence of intermediate capsules, so
semantic prompt size is worst-case linear in the number of frontier capsules
and is guarded by the configured prompt budget.

## Retrieval

The nodebank contains complete active capsules, not Atoms or trajectory text.
Embedding text remains exactly:

```text
name: ...
trigger: ...
content: ...
```

Prompt injection uses the complete capsule package, including scope,
verification, and failure modes. Existing token-budgeted antichain retrieval
prevents selecting an ancestor together with its descendant. No retrieval-side
deduplication is used to conceal a duplicate construction problem.
Selection retains the existing shifted-cosine relevance rule for the first
controlled comparison. Closed-loop exposure reliability is serialized and
passed to a revised capsule, but it does not affect ranking in this version.
Using co-exposure outcomes as a rank feature requires a separate controlled
ablation because they do not identify causal credit.

## Fairness Controls

`ebst_control_manifest.json` binds the static/dynamic pair to the same:

- records hash and split;
- model, decoding, tokenizer, embedding protocol, and cache;
- rollout/retry/max-turn protocol;
- shared rollout, evaluator, skillbank rendering, and antichain-selection
  source fingerprints;
- evaluator identity;
- retrieval query, top-k, and token budget;
- source-code fingerprints.

The static treatment additionally requires the exact CDOST-v3 Atom cache and
checks all non-treatment protocol invariants against its
`cdost_control_manifest.json`. The dynamic treatment inherits that baseline
binding from its paired static v4 manifest. Because older control manifests do
not contain the new fine-grained retrieval fingerprints, the formal control
must be rerun from this same commit; an old score may be reported only as
historical context.

The new tree policy does not alter SpreadsheetBench or OfficeQA evaluators,
rollout tools, heldout denominator, or legacy tree policies.

# Contract-Cut EBST Experiment

This directory contains the only supported v1 launch path. The method contract
is `research/contract_cut_ebst/ALGORITHM_CONTRACT.md`.

## Protocol

- Reuse the frozen, dataset-ordered SpreadsheetBench `0:200` no-thinking
  rollouts. Re-extract Contract Atoms because prior Atom caches are incompatible.
- Insert in dataset order, eight trajectories per batch.
- Use `M=8`, `m=4`, exact local overflow splits, `beta=1.0`, and lazy
  contract-feasible optimal cut.
- Run heldout `200:400` with dense top-1 skill retrieval and complete
  `SKILL.md` injection.
- Model and analyst thinking are disabled. No `max_tokens` field is sent for
  Atom analysis, skill compilation, or heldout rollout.
- Primary score is LibreOffice-recalculated correctness over all 200 heldout
  tasks.

## Launch

Set the yd-5 key without writing it to the repository:

```bash
export YD5_API_KEY='<yd-5 bearer key>'
bash experiments/contract_cut_ebst/run.sh
```

Override `RUN_DIR`, endpoints, model names, records, or dataset paths through
environment variables only when creating a separately named protocol variant.
The Python preflight fails if the frozen records hash, order, slices, workers,
batch size, or no-thinking contract changes.

## Outputs

```text
<run>/experiment_protocol.json
<run>/contract_cut/contract_atoms.json
<run>/contract_cut/tree_state.json
<run>/contract_cut/tree_audit.json
<run>/contract_cut/skill_registry.json
<run>/skills/skill_*/SKILL.md
<run>/skills/skill_*/references/atoms.json
<run>/skills/node_bank_manifest.json
<run>/heldout/skill_selections.jsonl
<run>/heldout/libreoffice_eval.json
<run>/reports/experiment_report.{json,md}
```

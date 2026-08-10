# Contract-Cut EBST Implementation Plan

## Scope

Implement Contract-Cut EBST as an isolated method path on top of the clean
`5aea7dc` repository checkpoint. Existing GMM, CDOST, EBST, SpreadsheetBench,
and evaluator behavior remain unchanged unless a small method registration is
required.

## Artifacts

1. `src/dynamix_core/contract_cut_ebst.py`
   - Contract Atom data model.
   - Single-parent balanced metric tree.
   - Exact 126-partition local split.
   - Distortion and exact antichain DP over EBST nodes plus deterministic
     Atom-entry terminals in physical leaves.
2. `src/dynamix_trace2skill/contract_cut_pipeline.py`
   - Full-evidence, same-prompt Atom analyst.
   - Prompt-budget exclusion audit.
   - Boundary embedding.
   - Lazy Contract Compiler and integrity checks.
   - Per-batch Atom extraction, sequential geometry updates, batch-end compile.
   - Stable registry, private audit, sanitized skill folder and nodebank export.
3. `scripts/run_contract_cut_ebst_experiment.py`
   - Provenance preflight and dataset-order enforcement.
   - Build from existing `0:200` records.
   - Top-1 `200:400` heldout rollout through the existing SpreadsheetBench
     agent and existing LibreOffice-recalc evaluator.
   - Stage logs, usage logs, resume markers, and final report.
4. `experiments/contract_cut_ebst/run.sh`
   - One explicit reproducible launch example with no embedded secret.
5. `tests/test_contract_cut_ebst.py`
   - Structural, split-optimality, cut, budget, compiler, registry, export, and
     retrieval-protocol tests.

## Verification gates

1. Static checks and targeted unit tests.
2. Existing skillbank and SpreadsheetBench regression tests.
3. Mock end-to-end build and nodebank retrieval.
4. Real service health checks and 1-2 record smoke build.
5. Independent spec/research-protocol and regression/security/Ponytail review.
6. Full `0:200` batch build, structure audit, and skill folder inspection.
7. Full `200:400` top-1 heldout with LibreOffice recalc.
8. Secret scan, clean diff review, commit, and push of only the new branch.

## Frozen inputs

The initial run uses:

```text
records: /mnt/data/yaodong/codes/DynaMix2skill/runs/
  spreadsheet_splitthink_rolloutfalse_analysttrue_best52_20260720_180410/
  ordered_records.json
record_count: 200
records_sha256: f07519085195e8fa77d036e4cea5cc3654c57722d8cd9476fffc93da51075db1
dataset: /mnt/data/yaodong/codes/DynaMix2skill/data/
  spreadsheetbench_verified/spreadsheetbench_verified_400
train_slice: 0:200
heldout_slice: 200:400
```

The old `evidence_cover_experience_atoms_v1` cache is intentionally not reused:
it used separate success/failure prompts, a 60,000-character compact evidence
bundle, and dual boundary/procedure embeddings, which violates this contract.

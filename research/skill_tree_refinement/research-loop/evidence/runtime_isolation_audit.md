# Spreadsheet Runtime Isolation Audit

## Verdict

The historical minsplit4 result and the active minsplit8 result are useful
development diagnostics, but they are not paper-grade clean method estimates.
The bash tool executes with a task-specific current working directory but does
not confine absolute paths. A model command can therefore execute a stale
global `/tmp/*.py` file after its attempted overwrite fails. Local helper files
named `inspect.py` also shadow Python's standard-library module and break
NumPy/openpyxl imports.

This conclusion is based on trace text, evaluator rows, the current bash-tool
implementation, and concrete foreign-path stack traces. It is not an inference
from aggregate accuracy.

## Fresh Minsplit4 Train Evidence

Run: `runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319`

- Train denominator: 200.
- Tasks referencing generic non-task `/tmp` paths: 3 (`48620`, `51289`,
  `130-9`). LibreOffice result: 0/3 passed.
- Tasks with the `partially initialized module 'openpyxl'` shadowing signature:
  10 (`409-45`, `39046`, `13-1`, `48620`, `47798`, `120-24`, `41-47`,
  `52575`, `531-18`, `130-9`). LibreOffice result: 3/10 passed.
- These traces feed `records.json` and therefore can affect downstream card
  construction even when both retrieval arms reuse the same frozen tree.

## Fresh Minsplit4 Heldout Evidence

- Heldout denominator: 200.
- Tasks referencing generic non-task `/tmp` paths: 4 (`54196`, `59129`,
  `51249`, `39190`). LibreOffice result: 3/4 passed.
- Tasks with the module-shadowing signature: 13 (`56786`, `7902`, `3002`,
  `54196`, `59794`, `59129`, `43657`, `37456`, `9111`, `15387`, `44296`,
  `39190`, `55427`). LibreOffice result: 4/13 passed.
- In task `54196`, the model attempted to create `/tmp/solution.py`; the write
  failed because another user's file already existed, and Python then executed
  that stale script, which referenced a foreign HiTab workbook path.

## Minsplit8 Running Snapshot

Run: `runs/research_minsplit8_heldout_20260710_191426`

At the running snapshot, five task logs referenced generic non-task `/tmp`
paths and thirteen contained the module-shadowing signature. The run was not
complete, so no final counts or score are asserted here. The same failure mode
is present, which means a completed score must be labeled under the old,
unisolated harness.

## Root Cause

`spreadsheet_agent/tools/bash.py` sets `cwd=working_dir`, which makes relative
paths task-local, but `shell=True` still permits explicit absolute paths. The
system prompts already tell the model not to search `/tmp`; prompt compliance
alone is insufficient as an isolation boundary.

## Frozen Correction

Before comparing a new retrieval strategy:

1. Reject an absolute `/tmp` path unless its resolved path is inside the current
   task directory. Do not rewrite commands silently.
2. Keep ordinary relative commands and task-local absolute paths valid.
3. Tell both vanilla and skill-preloaded agents not to name helper scripts after
   installed modules, and return a generic recovery hint if shadowing occurs.
4. Run a new dense-all-nodes control and routed intervention with the same
   patched harness, exact frozen tree/index, endpoint, 32 workers, top-10,
   thinking, max turns, and LibreOffice evaluator.
5. Treat this paired run as a mechanism diagnostic because the shared tree was
   built from some contaminated train traces. If the mechanism passes, rerun
   train and tree construction under the isolated harness before a headline
   method claim.

## Claim Boundary

This audit does not prove that every affected task would change outcome after
isolation, nor does it explain the full accuracy gap. It proves that non-model
runtime interference was possible and actually occurred, so aggregate scores
from the old harness cannot be interpreted as uncontaminated model or method
capability.

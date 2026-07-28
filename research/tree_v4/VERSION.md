# Version Record

- Date: 2026-07-28
- Branch: `research/ebst-v4-strict-online`
- Base commit: `7e4b914`
- Implementation commit: `9bce954`
- Remote: `https://github.com/Feesuu/DynaMix2skill.git`
- Method: Evidence-Balanced Skill Tree v4, strict online protocols

## Implemented Settings

1. `open_loop_replay`
   - starts from an empty tree;
   - consumes frozen train trajectories `0:200` in dataset order;
   - inserts one Atom at a time;
   - refreshes only dirty capsule paths;
   - validates and atomically checkpoints every arrival;
   - never injects the current skill tree into later train trajectories.
2. `closed_loop_skill_evolution`
   - resumes a fingerprint-matched open-loop prefix, or starts from null;
   - retrieves the current nodebank before each later train rollout;
   - evaluates the task with LibreOffice recalc;
   - records selected-skill exposure/outcome without causal attribution;
   - extracts one Atom, inserts it, revises dirty capsules, validates, and
     checkpoints before the next train task.

Both settings reserve SpreadsheetBench `200:400` for heldout and enforce the
same paired model, decoding, retrieval, cache, evaluator, timeout, retry, and
worker protocol through the control manifest.

## Verification

- Full test suite: `323 passed`, one pre-existing NumPy deprecation warning.
- Targeted EBST/closed-loop/reuse tests: `184 passed`.
- Python compilation passed.
- Shell syntax passed for all tree-v4 launchers.
- `git diff --check` passed.
- Secret scan found no API key or token in the changed source and artifacts.
- Two independent reviewers completed final review with no remaining P0/P1.

## Result Status

No live LLM, embedding, LibreOffice, or heldout benchmark run is attached to
this implementation checkpoint. The next formal run must execute the paired
open-loop and closed-loop protocols under the same control manifest; smoke or
partial results must not be reported as benchmark evidence.

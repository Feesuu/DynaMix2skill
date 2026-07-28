# Independent Review Gate

Date: 2026-07-28

## Reviewers

- Spec and research-protocol reviewer:
  `019fa840-86f8-7140-9279-cd3bf8ee0e7d`
- Regression, security/data, and Ponytail-complexity reviewer:
  `019fa842-34f3-78b2-bddb-981c9082f2d1`

Both reviewers inspected the live diff and untracked files without editing
them. After three repair passes, both returned `PASS` with no remaining P0/P1.

## Main Findings Resolved

- Resume now binds experiment contract, source, dataset and workbook content,
  Atom protocol, tree protocol, record prefix, evaluator, and checkpoint.
- Bootstrap must be an owned strict `open_loop_replay` prefix with matching
  record and tree fingerprints.
- Only strict open-loop permits an empty initial tree; legacy `fixed_replay`
  retains its previous minimum-one seed behavior.
- Closed-loop selection logs are bound to the exact task, query, top-k, active
  node IDs, and score cardinality.
- Open-loop/static capsule refresh keeps the prior prompt semantics; only
  closed-loop revision receives the prior capsule and exposure audit.
- The paired control manifest enforces model endpoint, thinking, temperature,
  turns, workers, timeout, retries, response-cache policy, top-k, and evaluator.
- Closed-loop train and the enclosing train-plus-heldout launcher both use
  nonblocking run-directory locks.
- Heldout resume binds input and ground-truth workbook hashes, nodebank/index,
  retrieval environment, source query-vector manifest, and runtime sources.
  Successful output workbooks are verified by SHA256 before a task is skipped.
- Source query vectors are validated before heldout and audited afterward
  against the paired open-loop manifest.
- API-key values and hashes are excluded from persisted commands/contracts.

## Ponytail Review

Six repeated Python config reads in the closed-loop shell were reduced to one.
No remaining simplification was accepted that would weaken experiment
identity, logging, checkpoint safety, testing, or research-protocol evidence.

## Verification

- Full tests: `323 passed`, one pre-existing NumPy deprecation warning.
- Targeted tests: `184 passed`.
- `py_compile`, launcher `bash -n`, and `git diff --check`: passed.
- No real model or LibreOffice experiment was run in this implementation goal.

## Claim Boundary

This review supports code/spec alignment, online-order correctness, structural
invariants, resume integrity, and paired-protocol enforcement. It does not
support a claim that either setting improves SpreadsheetBench performance,
runtime, or skill quality; those claims require the pending full experiments.

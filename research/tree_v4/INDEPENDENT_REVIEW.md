# Independent Review Gate

Date: 2026-07-28

## Reviewers

- Spec and research-protocol reviewer:
  `019fa738-9f6e-7d61-bf49-600eb26695f2`
- Regression, fairness, security, and Ponytail reviewer:
  `019fa739-5fde-7f12-a876-77e96997241b`

Both reviewers inspected the live diff and did not edit files.
After the fixes below, both completed a second pass with no remaining
actionable findings.

## Findings And Resolutions

### P1: legacy nodebank prompt regression

The initial implementation used any non-empty `prompt_text`, which changed
legacy GMM nodebank injection. The renderer now uses full prompts only for
CDOST or EBST analyst modes. A regression test locks legacy, CDOST, and EBST
rendering behavior.

### P1: dynamic capsule vector-cache miss

The initial dynamic configuration required every capsule text to already exist
in the static cache. EBST now reuses existing vectors under
`first_write_wins` and permits only newly created capsule texts to be added.
CDOST's previous strict-cache behavior remains unchanged.

### P1: incomplete build could leave a done marker

Capsule runtime and prompt-budget failures are written to the audit artifacts,
then the build process exits non-zero. The experiment runner therefore writes
only a failed marker, and a resumed run rebuilds after the service recovers.

### P1: shared retrieval code was not bound by the fairness gate

The control contract now fingerprints `skillbank.py` and the antichain
selection implementation in addition to the rollout and evaluator sources.
Old control manifests without those fingerprints are not accepted as formal
controls; the CDOST control must be rerun from the same commit.

### P2: incomplete EBST topology and schema validation

Heldout preflight and the selector both validate unique parents, closure,
reachability, lineage, active lifecycle, non-empty semantic and prompt fields,
the exact embedding-text contract, allowed analyst modes, and at least two
unique evidence atoms. Malformed manifests fail before any heldout worker is
started.

### P2/P3: lifecycle wording and unused API parameter

Archive events now record `archive_disposition=invalidated_stale`; the
documentation no longer equates every archive with successful replacement.
The unused `tree` argument was removed from `rebuild_active_links`.
Documentation also states that the current dynamic run deterministically
rebuilds the initial 120 frozen atoms and does not yet resume a serialized
tree snapshot.

## Verification

- Full test suite: `305 passed`, one pre-existing NumPy deprecation warning.
- Targeted EBST/fairness tests: `23 passed`.
- Offline frozen-Atom audit: 200 atoms, 41 structural nodes, 35 leaves,
  6 internal nodes, height 2, occupancy 4-8.
- Python compilation, shell syntax, and `git diff --check`: passed.
- No live LLM or LibreOffice benchmark was run because model resources were
  stopped.

## Claim Boundary

This gate supports implementation consistency, structural invariants, and
experiment-protocol enforcement. It does not yet support a claim of improved
capsule semantics, runtime, or downstream benchmark performance.

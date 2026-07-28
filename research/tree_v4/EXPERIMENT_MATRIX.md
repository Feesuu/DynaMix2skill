# Evidence-Balanced Skill Tree v4 Experiment Matrix

## Fixed Protocol

Both online settings must use the same:

- SpreadsheetBench verified train `0:200` and heldout `200:400`;
- source dataset order, with no shuffle or future-item access;
- Qwen3.5-9B-AWQ rollout/analyst model identity;
- Qwen3-Embedding-8B embedding protocol, tokenizer, cache namespace, chunk
  length, overlap, and pooling;
- max turns, rollout thinking/temperature, timeout, retry policy, disabled
  response cache, tools, worker count, and failure denominator;
- capsule analyst and validator configuration;
- retrieval query (`instruction + Task type`), top-k, prompt token budget,
  and complete-capsule injection format;
- LibreOffice-recalculated evaluator and 200-task heldout denominator.

Runtime-invalid tasks remain in the denominator and are reported separately
from workbook correctness. A partial or runtime-confounded run is diagnostic,
not headline evidence.

## Primary Settings

| ID | Train trajectory source | Online update | Heldout |
|---|---|---|---|
| E1 Open-loop replay | Existing fixed `0:200` trajectories; current skills do not affect later train trajectories | Start empty; after every arrival run `insert -> dirty-path refresh -> validate -> atomic checkpoint` | Current final nodebank on `200:400` |
| E2 Closed-loop skill evolution | Fixed open-loop prefix `0:120`; each task in `120:200` is rerolled after retrieving the current nodebank | Same structural update as E1; selected skill exposure and LibreOffice outcome are attached before Atom extraction | Current final nodebank on `200:400` |

The `120` bootstrap boundary is a preregistered protocol choice, not a tuned
result. A from-null closed-loop run is supported by setting the bootstrap count
to zero, but it is a separate experiment because early tasks cannot retrieve a
skill before any non-singleton capsule exists.

## Controlled Interpretation

E1 tests whether the online data structure can maintain a skill tree from a
fixed stream without future leakage. E2 tests the whole feedback system, where
the current skill bank changes later train trajectories. Their scores are not
an isolated ablation of the tree because E2 also changes the train data
distribution.

Selected skills are co-exposed. A later success or failure is recorded as
`exposure_outcome_not_causal`; it cannot identify which selected skill caused
the outcome. Exposure reliability is carried through capsule revision for
audit but does not affect retrieval ranking in this version.

## Required Reports

Each run must preserve:

- full command, resolved config, source commit, and source/data fingerprints;
- credential environment-variable names, never credential values or hashes;
- exact train/heldout split and ordered trajectory IDs;
- Atom, tree, capsule, lifecycle, nodebank, retrieval, and evaluator artifacts;
- one completed checkpoint per committed train arrival;
- closed-loop selected capsule IDs, scores, query, evaluation, and resulting
  record for every policy-dependent train task;
- wall time and token usage by stage;
- runtime failure taxonomy;
- LibreOffice numerator and denominator;
- strongest supported conclusion and explicit non-claims.

# Evidence-Balanced Skill Tree v4 Experiment Matrix

## Fixed Protocol

All paired runs must use the same:

- SpreadsheetBench verified split: train `0:200`, heldout `200:400`;
- frozen `records.json` and ordered Experience Atom identities;
- rollout model, endpoint, thinking mode, temperature, max turns, workers,
  timeout, retry policy, and tool permissions;
- frozen CDOST-v3 Atom embeddings and their 32k / 8k / 1k / mean-pooling
  protocol;
- retrieval query (`instruction + Task type`), top-k ceiling, prompt token
  budget, and complete-capsule injection format;
- LibreOffice-recalculated evaluator and 200-task denominator.

Runtime-invalid tasks must remain in the denominator and be reported separately
from workbook correctness. A partial or runtime-confounded run is diagnostic,
not headline evidence.

## Experiments

| ID | Research question | Control | Treatment | Decision signal |
|---|---|---|---|---|
| E0 | Does v4 repair the observed structural pathology? | Frozen v3 200-Atom artifact | Offline v4 insertion over the same ordered Atoms | occupancy, height/bound, certified radius upper bounds, unique placement, retrievable Atom count |
| E1 | Does the evidence-balanced tree and gated capsule bank improve heldout utility? | Reproducible v3 static nodebank | v4 static build from the same frozen records/Atoms | LibreOffice accuracy, execution-failure count, active/rejected capsules, retrieval level/support distribution |
| E2 | Can online local maintenance preserve useful skills? | E1 v4 static build over all 200 Atoms | v4 dynamic build: reconstruct `0:120`, insert `120:200` sequentially, refresh capsules every 8 arrivals | heldout accuracy gap to E1, changed-node/capsule counts, runtime/token cost |
| E3 | Is the Skill-SP-inspired evidence separation useful? | v4 with outcome labels hidden from capsule consolidation | v4 positive/negative/unknown evidence groups | heldout accuracy and unsafe/failure-derived recommendation audit |
| E4 | Is promotion gating necessary? | v4 candidate capsules promoted without the second parent audit | full v4 deterministic + LLM promotion gate | duplicate/paraphrase rate, rejected-parent count, heldout accuracy |

E0 is already executable offline and supports only structural claims. E1 and E2
are the first required model experiments. E3 and E4 are ablations and must not
be run until E1 establishes a clean treatment result.

## Required Reports

Each model run must preserve:

- full command and resolved config;
- source and protocol fingerprints;
- Atom, tree, capsule, lifecycle, nodebank, retrieval, and evaluator artifacts;
- wall time and token usage by stage;
- runtime failure taxonomy;
- LibreOffice numerator and denominator;
- strongest supported conclusion and explicit non-claims.

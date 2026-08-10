# Contract-Cut EBST Algorithm Contract

This document freezes the method identity for the
`research/contract-cut-ebst-v1` branch. Implementations and experiment reports
must not silently weaken or reinterpret these rules.

## Inputs and protocol

- SpreadsheetBench train records are the dataset-ordered `0:200` trajectories.
- Every accepted trajectory produces exactly one Contract Atom.
- Success and failure records use the same analyst prompt and output schema.
- The analyst receives the instruction, complete rollout, produced answer,
  LibreOffice-recalculated evaluator evidence, and success/failure status.
- Evidence is never character-truncated. A prompt that exceeds the configured
  analyst input budget is excluded with an explicit audit record.
- Reusable Atom fields must not contain task IDs, paths, workbook coordinates,
  exact answers, or incidental task literals.
- Gold/evaluator evidence is diagnostic only. It is excluded from boundary
  embeddings, retrieval documents, retrieval queries, and exported skill text.
- Every Atom has weight one.

## Contract Atom and boundary metric

Each Atom stores:

```text
trigger
scope
decision
invariant
verification
failure_mode
provenance
```

Its only structural embedding text is:

```text
Applicability: {trigger}
Scope: {scope}
Success condition: {verification}
```

The embedding is L2-normalized. Tree distance is Euclidean chord distance,
which is monotone in cosine distance on unit vectors:

```math
d(a_i,a_j)^2 = 2 - 2 cos(b_i,b_j).
```

Decision, invariant, and failure mode affect the compiled skill contract, not
the skill boundary.

## Evidence-balanced metric tree

The tree is online, single-parent, and B+-tree balanced. Leaves contain Atom
IDs; internal nodes contain child IDs. Every representative is a real
descendant Atom. Every node stores an exact cover radius over its descendant
Atoms.

Insertion descends from the root. For every child `c` and new Atom `a`, rank:

```math
(max(0, d(rep(c), a) - rho(c)), d(rep(c), a), stable_id(c)).
```

The lexicographically smallest child is selected. The Atom is inserted once.

The fixed capacity is `M=8`, with minimum non-root occupancy `m=4`. An
overflow has exactly nine entries. All 126 four/five partitions are enumerated.
For each side `S`, the local cover objective is:

```math
R(S) = min_{r in S} max_{e in S} (d(r,e) + rho_e).
```

The split minimizes, lexicographically:

```math
(max(R(L),R(R)), R(L)+R(R), canonical_partition_id).
```

Parent overflow recurses; root overflow creates a new root. All leaves remain
at equal depth and every non-root occupancy remains in `[4,8]`.

## Contract-feasible optimal cut

The geometric EBST is not modified by skill compilation. For the cut only, we
define an augmented candidate tree `T+`: every physical EBST node is a
candidate, and every Atom entry inside a physical leaf is also represented by
one deterministic terminal child. These Atom-entry terminals are not extra
geometric nodes and do not affect insertion, occupancy, depth, radii, or the
single-parent balance guarantee. They formalize the required fallback that one
Atom can always be rendered as one terminating skill.

Every subtree is only a candidate skill region. Its distortion is:

```math
D(u) = sum_{a in A(u)} d(a, rep(u))^2.
```

The one-skill geometric cost is `D(u) + beta`. The initial experiment uses
`beta=1.0`, an explicit scale-aware regularizer because squared chord distance
on unit vectors lies in `[0,4]`. Beta is recorded in every run and must not be
tuned on heldout labels.

For a fixed augmented candidate tree `T+` and current infeasible-candidate
cache, dynamic programming computes:

```math
F(u) = min(D(u)+beta, sum_{v in children(u)} F(v)).
```

For a physical leaf, the split alternative is the sum of its Atom-entry
terminal costs. The resulting selected candidates form an antichain in `T+`
and cover every accepted Atom exactly once. Lazy feasibility then compiles only
selected roots. A successful contract remains selected. A `split_required` or
over-budget physical candidate is cached as infeasible for that exact compiler
input hash, and the DP reruns. Atom-entry terminals are rendered
deterministically and are always feasible, so the process terminates. The
optimal-cut guarantee is exact on `T+`; it is not claimed for the plain EBST
node set alone.

## Skill compiler and export

The compiler receives all structured Atoms under one selected root, never raw
trajectories. It may output only `contract` or `split_required`. A contract has:

```text
name
applicability
objective
conditional_rules
invariants
verification
recovery
source_atom_ids
```

Each conditional rule has:

```text
condition
procedure
invariant
verification
recovery
source_atom_ids
```

There is one JSON format-repair attempt. Deterministic integrity checking
validates schema, non-empty required fields, source IDs, complete Atom
provenance coverage, and reference paths. There is no additional heuristic or
LLM semantic validator.

The final cut is exported as:

```text
skill_<stable_id>/
  SKILL.md
  references/atoms.json
  provenance.json
```

`SKILL.md` has exactly these sections: When To Use, Objective, Decision Rules,
Procedure, Invariants, Verification, Recovery, and References.

## Batch updates and identity

Atoms arrive in dataset order with `batch_size=8`. Within a batch, insertion is
sequential and updates only geometry. No compiler call occurs until the batch
ends. The batch-end cut is recomputed, and only candidate regions whose
compiler input hash is new are compiled.

Stable skill identity follows member sets:

- unchanged members reuse the skill ID and files;
- a unique existing skill that only gains Atoms keeps its ID and increments its
  version;
- a split supersedes the predecessor and creates derived skills;
- a merge creates a new skill with all predecessor IDs.

Every private registry version records member Atom IDs, source item IDs, tree
region, compiler input hash, `SKILL.md` hash, batch, and lineage. A merge never
inherits one predecessor ID: it creates a new ID whose `derived_from` contains
all predecessors. A strict one-predecessor expansion may retain its ID.

The private audit artifacts under `contract_cut/` retain source item IDs,
verifier evidence, hashes, and boundary embeddings. Agent-readable skill
folders expose only synthetic Atom IDs, reusable semantic Atom fields, safe
lineage, and the compiled skill. They never expose task IDs, verifier feedback,
record hashes, boundary vectors, paths from the source task, or gold answers.
Exact numeric quantities copied from an instruction are treated as incidental
task literals and trigger one semantic revision before an Atom can be accepted.

## Heldout protocol

- Dataset: SpreadsheetBench verified heldout `200:400`.
- Model: `Qwen3.5-9B-AWQ` at the declared run endpoint.
- Thinking: disabled for Atom analysis, compilation, and heldout rollout.
- LLM concurrency: 8; Atom batch size: 8; max turns: 30.
- Model context length: 100,000. No artificial rollout or analyst
  `max_tokens` field is sent.
- Retrieval corpus: only final optimal-cut skill folders.
- Skill embedding text: name, applicability, objective.
- Query: instruction plus `Task type: instruction_type`.
- Query excludes answer position, gold, evaluator evidence, and outputs.
- Retrieval is dense cosine top-1.
- The complete selected `SKILL.md` is injected. Its reference directory comes
  from a public-only mirror containing only the exported skill folders.
- Model-generated shell commands run in a separate PID namespace. The private
  run directory is masked, and model/API secrets and DynaMix control variables
  are removed from the shell environment.
- Resume markers seal the complete build and heldout artifacts, not only their
  summary JSON files.
- Primary evaluation is LibreOffice-recalculated correctness over all 200
  heldout tasks. Runtime failures remain in the denominator.

## Claim boundary

The implementation may claim balanced structure, exact local overflow split,
exact Atom cover, fixed-tree optimal cut, local geometric insertion, batch-end
compilation equivalence in open-loop mode, deterministic identity, and
auditable provenance. It must not claim that the embedding is semantically
correct, that the LLM compiler is optimal, or that heldout quality is guaranteed
without benchmark evidence.

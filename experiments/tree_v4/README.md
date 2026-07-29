# Evidence-Balanced Skill Tree v4 Experiments

These launchers implement two online SpreadsheetBench settings with one shared
EBST state transition:

```text
Atom -> insert -> dirty-path capsule refresh -> validate -> atomic checkpoint
```

They do not copy or replace the SpreadsheetBench agent, LibreOffice evaluator,
Experience Atom analyst, balanced tree, capsule analyst, nodebank exporter, or
retriever.

## Shared Protocol

- train `0:200`, heldout `200:400`;
- dataset order, no shuffle;
- single-parent balanced metric tree;
- only active Skill Capsules are retrievable;
- retrieval query is `instruction + Task type`;
- heldout top-k defaults to 10;
- LibreOffice recalc is authoritative;
- runtime failures remain in the denominator.

All model, endpoint, tokenizer, embedding, timeout, retry, and worker settings
must be exported explicitly. Start from `base_env.example.sh`; it contains no
API key.

## 1. Open-Loop Replay

`run_dynamic.sh` consumes existing fixed `0:200` train trajectories. It starts
from an empty tree. Current skills never affect later train trajectories.
Frozen Experience Atoms are allowed only as a policy-independent cache. The
tree, registry, and each `arrival_XXXX` checkpoint contain only the committed
prefix.

The current launcher first requires a same-commit CDOST control so the frozen
Atom and protocol manifest cannot come from an incompatible historical run:

```bash
RUN_DIR=/absolute/path/to/control_run \
  SOURCE_CDOST_RUN_DIR=/absolute/path/to/frozen_atom_source \
  bash experiments/tree_v4/run_control.sh

RUN_DIR=/absolute/path/to/open_loop_run \
  BASELINE_CDOST_RUN_DIR=/absolute/path/to/control_run/scenarios/static_build \
  bash experiments/tree_v4/run_static.sh

RUN_DIR=/absolute/path/to/open_loop_run \
  bash experiments/tree_v4/run_dynamic.sh
```

The strict open-loop controls are fixed by the wrapper:

```text
initial_count=0
arrival_count=200
update_batch_size=1
trajectory_source=open_loop_replay
shuffle_seed=null
resume_from_snapshots=false
```

`run_dynamic.sh` also runs heldout `200:400` through the common experiment
driver. EBST snapshot resume remains disabled until snapshot identity and
fingerprint validation are implemented; a failed launch must restart the
dynamic build from its empty-tree input rather than trust a partial snapshot.

## 2. Closed-Loop Skill Evolution

The formal default uses the completed open-loop checkpoint after train item
119 as bootstrap. Tasks `120:200` are rerolled one at a time:

```text
retrieve current nodebank
-> inject selected capsules into the system prompt
-> run one SpreadsheetBench task
-> LibreOffice recalc evaluation
-> record selected capsule exposure and outcome
-> extract one Atom
-> apply the shared EBST state transition
```

The next task cannot start until the current task's checkpoint is complete.
There is no structural batch.

```bash
export OPEN_LOOP_SCENARIO_DIR=/absolute/path/to/open_loop_run/scenarios/dynamic_update
export CLOSED_LOOP_RUN_DIR=/absolute/path/to/closed_loop_run
bash experiments/tree_v4/run_closed_loop.sh
```

The wrapper runs the policy-dependent train suffix and then heldout `200:400`
with 16 workers by default. `BOOTSTRAP_COUNT=0` is supported for a separate
from-null closed-loop experiment; in that mode the launcher omits bootstrap
checkpoint arguments.

Heldout resume is guarded by the existing
`results.jsonl.manifest.json` identity. It binds the final nodebank manifest,
model/decoding, split, top-k, embedding endpoint/model, rollout agent sources,
and skillbank source before `--missing_only` may reuse any task.

## Claim Boundary

The two settings use the same tree algorithm but not the same train
trajectories. Open-loop isolates online maintenance over a fixed stream.
Closed-loop evaluates the whole feedback system because current skills change
later train rollouts.

Skill exposure outcomes are not causal attribution. They are preserved as
`exposure_outcome_not_causal`, carried through capsule revision for audit, and
not used by retrieval ranking.

No quality claim is valid until the complete 200-task heldout and LibreOffice
evaluation finish without unresolved runtime confounders.

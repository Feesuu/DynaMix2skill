# Evidence-Balanced Skill Tree v4

These launchers run the new `evidence_balanced_skill_tree` policy without
copying the main experiment driver. Both use
`scripts/run_handoff_static_dynamic_experiment.sh`; the wrappers only freeze
the mechanism-specific controls.

## Protocol

- SpreadsheetBench verified train `0:200`, heldout `200:400`
- source dataset order
- one immutable Experience Atom per train trajectory
- exact frozen CDOST-v3 Experience Atoms and their recorded embeddings
- `Qwen3-Embedding-8B`, 32k model limit, baseline 8k/1k mean-pooled chunks
- rollout and analyst settings inherited from explicit environment variables
- atoms never enter the nodebank
- only active Skill Capsules are retrievable
- LibreOffice-recalculated heldout evaluation

First rerun the CDOST control from this same commit while reusing the
historical run's exact ordered `experience_atoms.json`. Then point
`BASELINE_CDOST_RUN_DIR` at that new control scenario. The runner verifies the shared records,
splits, generation, embedding, analyst, retrieval, rollout, and evaluator
protocol against that run's control manifest. It then builds all 200 atoms with
the same sequential insertion operator used by the dynamic run. The dynamic
run reuses the static v4 run's frozen Atoms, reconstructs inserts `0:120`, and
then inserts `120:200` sequentially. Capsule refreshes are batched only to
parallelize LLM calls; structural insertion remains sequential.

## Required Environment

Start from the example and replace local paths/endpoints:

```bash
cp experiments/tree_v4/base_env.example.sh /tmp/ebst_env.sh
source /tmp/ebst_env.sh
```

The file contains no API key. Export `OPENAI_API_KEY` in the shell when the
endpoint requires one.

`SOURCE_CDOST_RUN_DIR` supplies only the historical frozen Atom artifact.
Create the same-commit formal control with:

```bash
RUN_DIR=/absolute/path/to/runs/ebst_v4_control \
  bash experiments/tree_v4/run_control.sh
```

Then set:

```bash
export BASELINE_CDOST_RUN_DIR="$RUN_DIR/scenarios/static_build"
```

`BASELINE_CDOST_RUN_DIR` must point to the completed same-commit control
scenario containing
`dynamix_tree/experience_atoms.json`,
`analysis/cdost_control_manifest.json`, and the successful build marker.

## Static

```bash
bash experiments/tree_v4/run_static.sh
```

The default run root is:

```text
runs/ebst_v4_<timestamp>/
```

## Dynamic

Use the exact same `RUN_DIR` as the completed static run:

```bash
RUN_DIR=/absolute/path/to/runs/ebst_v4_<timestamp> \
  bash experiments/tree_v4/run_dynamic.sh
```

The dynamic wrapper requires:

```text
<RUN_DIR>/scenarios/static_build/dynamix_tree/experience_atoms.json
```

It fails before launch if that controlled source artifact is absent.

## Claim Boundary

These scripts establish a fair, paired experiment. Structural invariants are
checked by code. No benchmark improvement or semantic correctness is claimed
until the complete heldout run and LibreOffice evaluator finish without
runtime confounders.

The controlled questions, fixed variables, and reporting requirements are in
`research/tree_v4/EXPERIMENT_MATRIX.md`.

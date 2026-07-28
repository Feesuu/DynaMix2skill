# DynaMix Static Baseline Checkpoint

## 1. Git identity

- Repository: `/mnt/data/yaodong/codes/DynaMix2skill`
- Audited code commit: `06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a`
- Stable branch: `stable/static-thinkingfalse-20260724`
- Annotated tag: `static-thinkingfalse-20260724`
- Research branch: `research/tree-v2-constrained-gmm`
- Isolated research worktree: `/mnt/data/yaodong/codes/DynaMix2skill_tree_v2`

Both branches and the tag currently resolve to the audited commit above. They are
local checkpoints; they have not been pushed by this task.

The original worktree remains on `reproduce-48p5-20260714` and still contains
pre-existing tracked and untracked changes. Those changes were deliberately not
committed into the stable checkpoint because the audited run records commit
`06bb7803...` as its code identity.

## 2. Audited run identity

- Run:
  `/mnt/data/yaodong/codes/DynaMix2skill/runs/spreadsheet_splitthink_rolloutfalse_analysttrue_best52_20260720_180410`
- Dataset: `spreadsheetbench_verified_400`
- Train range: `[0, 200)`
- Heldout range: `[200, 400)`
- Model: `Qwen3.5-9B-AWQ`
- Rollout workers: `16`
- Rollout max turns: `30`
- Rollout temperature: `0.0`
- Retrieval top-k: `10`
- Embedding model: `Qwen3-Embedding-8B`
- Embedding chunking: `28000` tokens with `1000` overlap, mean pooling
- Tree policy: `projected_gmm_bic`
- `min_split_size`: `8`
- `min_effective_samples_per_component`: `2`
- Membership: cumulative mass `0.9`, max gap `0.25`, minimum weight `0.05`
- Tree maximum levels: `8`
- Evaluator: LibreOffice recalculation

Important protocol correction:

- The directory name contains `analysttrue`.
- The saved `dynamix_config.json` actually has
  `generation.thinking_mode=false` and
  `chat_template_kwargs.enable_thinking=false`.
- The runtime config also records `thinking="false"`.

Therefore this artifact must not be described as an analyst-thinking-true run.

## 3. Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `dynamix_config.json` | `0e3c5e9575c6a0a4e8a74250a922801f24095cd6af39c6715bfcb1f15853020a` |
| `ordered_records.json` | `f07519085195e8fa77d036e4cea5cc3654c57722d8cd9476fffc93da51075db1` |
| `records.json` | `fcf95508fefd8bf84047c80ee1c89c4704d7137181e100f5dd144b5b34441d6d` |
| `node_bank_manifest.json` | `d5d078f8d2de8bc2b79903e3bea04f98896b04052672b25e4ae2339fa908c88d` |
| `trace2skill_heldout_eval.json` | `fb6f5b520dd839c0bd0239acf581866fac2c362062b391a54788480bfab9cbb1` |

## 4. Actual result

The run name contains `best52`, but the saved LibreOffice-recalculated result is:

```text
78 / 200 = 39.0%
```

The no-recalculation cached-value audit is:

```text
52 / 200 = 26.0%
```

The run name is not evidence of a 52% result and must not be used in tables,
branch names, or paper claims as if it were one.

## 5. Actual tree

The saved artifacts show:

```text
200 trajectories
  -> root GMM-BIC chooses K=5
  -> four token-oversized root communities are recursively refined
  -> 103 flattened L0 communities
  -> 244 L1 experience cards
  -> 38 L2 cards
  -> 8 L3 cards
  -> 2 L4 cards
  -> 292 retrievable node-bank nodes
```

The node bank level distribution is:

| Level | Nodes |
|---|---:|
| L1 | 244 |
| L2 | 38 |
| L3 | 8 |
| L4 | 2 |

The detailed audit found 62 singleton L0 communities and 120 L1 cards generated
from those singleton communities. The hierarchy is therefore dominated by
low-level, singleton-derived cards rather than by a small number of stable
semantic communities.

## 6. Known structural problems

1. Token-budget refinement changes semantic cluster identity. Oversized
   communities are recursively split and their leaves are flattened into L0,
   so prompt capacity determines the semantic topology.
2. `min_effective_samples_per_component=2` restricts the candidate K bound but
   does not reject fitted GMM components whose effective support is below two.
3. Raw trajectory action and observation text dominates the embedding input;
   task instruction semantics are a small fraction of the representation.
4. Cumulative-mass soft assignment is almost hard in this run, so the claimed
   overlap mechanism is barely exercised.
5. Singleton upper communities stop instead of contributing a distinct higher
   abstraction; support mass decreases across levels.
6. Higher nodes can be paraphrases of children, producing duplicate retrievable
   content without adding a new invariant.

This checkpoint is preserved to reproduce and diagnose those behaviors, not to
claim that they are the intended final DynaMix algorithm.

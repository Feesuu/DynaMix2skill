# Static Build Ablations

These ablations test one mechanism at a time while reusing the same DynaMix
codebase and experiment runner. No variant copies `src/`; each variant only
changes controlled CLI/config knobs.

## Variants

- `hard_assignment_gmm`: keep projected GMM-BIC, but use one hard argmax parent
  per item.
- `kmeans_elbow`: replace GMM-BIC with projected weighted KMeans and elbow K.
- `kmeans_fixed_k`: replace GMM-BIC with projected weighted KMeans at a fixed K.
- `l0_single_card`: keep the full hierarchy, but cap each L0 community to one
  generated card.
- `l1_only`: build only L0 -> L1, with no L2+ abstraction.
- `retrieve_l1_only`: reuse a full baseline tree, export only L1 cards.
- `retrieve_l2plus_only`: reuse a full baseline tree, export only L2+ cards.
- `retrieve_all_cards`: reuse a full baseline tree, export all cards. This is a
  sanity check against the baseline nodebank export.

## Running

1. Copy `common/base_env.example.sh` to a local file such as
   `common/base_env.local.sh`.
2. Fill in paths/endpoints in that local file.
3. Run a variant:

```bash
DYNAMIX_ABLATION_ENV=experiments/ablations/static/common/base_env.local.sh \
  experiments/ablations/static/variants/hard_assignment_gmm/run.sh
```

Retrieval-only variants require `BASELINE_TREE_DIR` to point to an already
built full static `dynamix_tree/` directory, and require either `RECORDS_PATH`
or `REUSE_TRAIN_RUN_DIR` so train rollout/eval/extraction stages are reused.
They do not rebuild the tree; they only re-export a filtered nodebank and run
heldout with the same retrieval top-k.

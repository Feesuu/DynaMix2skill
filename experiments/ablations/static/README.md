# Static Ablation Experiments

This directory keeps static-build ablation definitions separate from the main
implementation. Variants do not copy `src/`; each variant only declares the
mechanism difference in `variant.json` and delegates execution to
`common/run_variant.sh`.

## Variants

- `flat_individual_cards`: `tree_policy=identity_singleton`, `max_levels=1`; each budget-fit trajectory is summarized independently, while oversize singletons keep the shared excluded audit path.
- `hard_assignment_gmm`: keep GMM-BIC, but use `primary_argmax` assignment.
- `kmeans_elbow`: use PCA + weighted KMeans with inertia elbow K selection.
- `l1_only`: keep GMM-BIC, but stop after L0 -> L1.
- `retrieve_all_cards`: re-export an existing full tree with all ExperienceCards.
- `retrieve_l1_only`: re-export an existing full tree with `level == 1`.
- `retrieve_l2plus_only`: re-export an existing full tree with `level >= 2`.

## Usage

Copy and edit the environment template:

```bash
cp experiments/ablations/static/common/base_env.example.sh /tmp/dynamix_ablation_env.sh
vim /tmp/dynamix_ablation_env.sh
```

Run one variant:

```bash
bash experiments/ablations/static/variants/hard_assignment_gmm/run.sh /tmp/dynamix_ablation_env.sh
```

For retrieval-only variants, first run or point to an unfiltered full static tree
from the main method and set:

```bash
export REUSE_TREE_DIR=/path/to/full/static/run/dynamix_tree
bash experiments/ablations/static/variants/retrieve_all_cards/run.sh /tmp/dynamix_ablation_env.sh
```

All outputs are written under:

```text
runs/ablations/static/<variant_name>/<timestamp>/
```

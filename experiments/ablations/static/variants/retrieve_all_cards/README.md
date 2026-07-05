# retrieve_all_cards

Question: sanity check for the full-tree nodebank export path.

This reuses an existing full static tree and exports all ExperienceCards. It
should match the baseline nodebank policy except for run directory and resume
fingerprints. Set `BASELINE_TREE_DIR` to the full baseline `dynamix_tree/`
directory, plus `RECORDS_PATH` or `REUSE_TRAIN_RUN_DIR` to reuse train
artifacts.

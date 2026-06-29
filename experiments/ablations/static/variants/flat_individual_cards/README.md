# flat_individual_cards

No clustering. Each non-excluded L0 trajectory forms its own singleton
community, and the run stops after L0 -> L1 (`max_levels=1`). Single trajectories
that exceed the shared analyst prompt budget still use the normal
oversize/excluded audit path. This tests whether community-level experience
extraction is better than per-trajectory card extraction.

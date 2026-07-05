# kmeans_fixed_k

Question: is automatic K selection important?

This replaces GMM-BIC with weighted KMeans at a fixed target K. The default
target is `8`, but the effective K is still clipped by the same `compute_kmax`
and item-count constraints used by the main tree builder. This keeps the
ablation safe under small layers while testing a non-BIC K policy.

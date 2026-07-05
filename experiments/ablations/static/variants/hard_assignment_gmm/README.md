# hard_assignment_gmm

Question: is multi-parent soft clustering necessary?

This keeps the baseline projected GMM-BIC clustering, prompt, hierarchy depth,
nodebank export, and heldout retrieval unchanged. The only mechanism change is
`soft_recursive_assignment=primary_argmax`, so each input item contributes to
exactly one community with weight `1.0`.

# kmeans_elbow

Question: does the probabilistic GMM-BIC clustering matter, or is a simpler
hard KMeans split enough?

This keeps local PCA, support weighting, K upper bounds, budget refinement,
LLM summarization, hierarchy depth, and nodebank retrieval unchanged. The split
selector is replaced by weighted KMeans with an elbow-selected K.

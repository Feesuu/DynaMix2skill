from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .config import GmmBicConfig, KMeansConfig
from .gmm_bic import GmmBicSelection, GmmCandidateFit, compute_kmax

_EPS = 1.0e-12


def select_kmeans_split(
    projected: np.ndarray,
    *,
    gmm_config: GmmBicConfig,
    kmeans_config: KMeansConfig,
    mode: str,
    random_seed: int,
    sample_weights: np.ndarray | None = None,
    kmax_effective_n: float | None = None,
) -> GmmBicSelection:
    """Return a GMM-selection-shaped weighted KMeans split."""

    gmm_config.validate()
    kmeans_config.validate()
    if mode not in {"elbow", "fixed"}:
        raise ValueError("KMeans split mode must be 'elbow' or 'fixed'")
    x = _finite_2d(projected, name="projected")
    weights = _sample_weights(sample_weights, int(x.shape[0]))
    kmax = compute_kmax(
        int(x.shape[0]),
        gmm_config,
        total_weight=float(weights.sum()),
        kmax_effective_n=kmax_effective_n,
    )
    if mode == "fixed":
        effective_k = max(1, min(int(kmeans_config.fixed_k), int(kmax), int(x.shape[0])))
        k_values = [effective_k]
    else:
        min_k = max(1, min(int(kmeans_config.min_k), int(kmax)))
        k_values = list(range(min_k, int(kmax) + 1))
    candidates = [
        _fit_kmeans(
            x,
            k=int(k),
            config=kmeans_config,
            random_seed=int(random_seed) + int(k) * 9973,
            sample_weights=weights,
            min_covar=float(gmm_config.min_covar),
        )
        for k in k_values
    ]
    if not candidates:
        raise ValueError("no KMeans candidates")
    chosen = _choose_fixed(candidates, int(kmeans_config.fixed_k)) if mode == "fixed" else _choose_elbow(candidates)
    return GmmBicSelection(
        chosen=chosen,
        candidates=candidates,
        bic_margin=_score_margin(candidates, chosen),
    )


def _fit_kmeans(
    x: np.ndarray,
    *,
    k: int,
    config: KMeansConfig,
    random_seed: int,
    sample_weights: np.ndarray,
    min_covar: float,
) -> GmmCandidateFit:
    k = max(1, min(int(k), int(x.shape[0])))
    if k == 1:
        labels = np.zeros(int(x.shape[0]), dtype=int)
        centers = _weighted_mean(x, sample_weights)[None, :]
        inertia = _weighted_inertia(x, labels, centers, sample_weights)
        return _candidate_from_labels(x, labels, centers, inertia, sample_weights, min_covar=min_covar, n_iter=0)

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=int(config.num_restarts),
        max_iter=int(config.max_iter),
        tol=float(config.tol),
        random_state=int(random_seed),
        algorithm="lloyd",
    )
    model.fit(x, sample_weight=sample_weights)
    labels = np.asarray(model.labels_, dtype=int)
    centers = np.asarray(model.cluster_centers_, dtype=float)
    return _candidate_from_labels(
        x,
        labels,
        centers,
        float(model.inertia_),
        sample_weights,
        min_covar=min_covar,
        n_iter=int(getattr(model, "n_iter_", 0) or 0),
    )


def _candidate_from_labels(
    x: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    inertia: float,
    sample_weights: np.ndarray,
    *,
    min_covar: float,
    n_iter: int,
) -> GmmCandidateFit:
    k, dim = means.shape
    responsibilities = np.zeros((x.shape[0], k), dtype=float)
    responsibilities[np.arange(x.shape[0]), labels] = 1.0
    masses = np.array([float(sample_weights[labels == component].sum()) for component in range(k)], dtype=float)
    total = max(float(sample_weights.sum()), np.finfo(float).eps)
    pi = np.maximum(masses / total, np.finfo(float).eps)
    pi = pi / float(pi.sum())
    variances = np.empty((k, dim), dtype=float)
    for component in range(k):
        mask = labels == component
        if np.any(mask):
            diff = x[mask] - means[component]
            local_weights = sample_weights[mask]
            variances[component] = np.maximum(
                (local_weights[:, None] * diff * diff).sum(axis=0) / max(float(local_weights.sum()), np.finfo(float).eps),
                min_covar,
            )
        else:
            variances[component] = min_covar
    child_sizes = [int(np.sum(labels == component)) for component in range(k)]
    valid = all(size > 0 for size in child_sizes)
    return GmmCandidateFit(
        k=int(k),
        valid=bool(valid),
        bic=float(inertia),
        log_likelihood=float(-inertia),
        pi=pi,
        means=np.asarray(means, dtype=float),
        variances=np.asarray(variances, dtype=float),
        responsibilities=responsibilities,
        primary_labels=np.asarray(labels, dtype=int),
        component_masses=[float(value) for value in masses],
        child_sizes=child_sizes,
        reason="" if valid else "empty_primary_child",
        converged=True,
        n_iter=int(n_iter),
    )


def _choose_fixed(candidates: list[GmmCandidateFit], fixed_k: int) -> GmmCandidateFit:
    target = max(1, int(fixed_k))
    by_distance = sorted(candidates, key=lambda item: (abs(int(item.k) - target), int(item.k)))
    valid = [candidate for candidate in by_distance if candidate.valid]
    return valid[0] if valid else by_distance[0]


def _choose_elbow(candidates: list[GmmCandidateFit]) -> GmmCandidateFit:
    valid = [candidate for candidate in candidates if candidate.valid]
    pool = valid or candidates
    if len(pool) <= 2:
        return pool[-1]
    xs = np.asarray([float(candidate.k) for candidate in pool], dtype=float)
    ys = np.asarray([float(candidate.bic) for candidate in pool], dtype=float)
    if float(xs[-1] - xs[0]) <= _EPS or float(np.max(ys) - np.min(ys)) <= _EPS:
        return pool[0]
    points = np.column_stack(((xs - xs[0]) / (xs[-1] - xs[0]), (ys - ys[-1]) / (ys[0] - ys[-1] + _EPS)))
    start = points[0]
    end = points[-1]
    line = end - start
    denom = max(float(np.linalg.norm(line)), _EPS)
    distances = np.abs(np.cross(line, points - start)) / denom
    return pool[int(np.argmax(distances))]


def _score_margin(candidates: list[GmmCandidateFit], chosen: GmmCandidateFit) -> float:
    valid = sorted((candidate for candidate in candidates if candidate.valid), key=lambda item: item.bic)
    if len(valid) < 2:
        return 0.0
    if valid[0].k == chosen.k:
        return float(valid[1].bic - valid[0].bic)
    return float(chosen.bic - valid[0].bic)


def _finite_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} values must be finite")
    return matrix


def _sample_weights(values: np.ndarray | None, n_items: int) -> np.ndarray:
    if values is None:
        return np.ones(n_items, dtype=float)
    weights = np.asarray(values, dtype=float)
    if weights.shape != (n_items,):
        raise ValueError("sample_weights shape must match item count")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample_weights must be positive finite values")
    return weights


def _weighted_mean(x: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
    return (x * sample_weights[:, None]).sum(axis=0) / max(float(sample_weights.sum()), np.finfo(float).eps)


def _weighted_inertia(x: np.ndarray, labels: np.ndarray, means: np.ndarray, sample_weights: np.ndarray) -> float:
    distances = np.sum((x - means[labels]) ** 2, axis=1)
    return float(np.dot(sample_weights, distances))


__all__ = ["select_kmeans_split"]

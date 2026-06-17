from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np

from .config import GmmBicConfig, SoftMembershipConfig

_EPS = 1.0e-12


@dataclass(frozen=True)
class GmmCandidateFit:
    k: int
    valid: bool
    bic: float
    log_likelihood: float
    pi: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    responsibilities: np.ndarray
    primary_labels: np.ndarray
    component_masses: list[float]
    child_sizes: list[int]
    reason: str = ""
    converged: bool = False
    n_iter: int = 0


@dataclass(frozen=True)
class GmmBicSelection:
    chosen: GmmCandidateFit
    candidates: list[GmmCandidateFit]
    bic_margin: float


def compute_kmax(
    n_items: int,
    config: GmmBicConfig,
    *,
    total_weight: float | None = None,
    kmax_effective_n: float | None = None,
) -> int:
    """Return the maximum K to search for the current layer.

    ``kmax_effective_n`` is the independent item/card count used to bound
    model search.  ``total_weight`` is kept for backwards compatibility with
    older callers, but inherited support_mass totals should not inflate Kmax.
    """
    _validate_gmm_config(config)
    n_items = int(n_items)
    if n_items < 1:
        raise ValueError("n_items must be positive")
    if n_items < config.min_split_size:
        return 1
    effective_source = kmax_effective_n if kmax_effective_n is not None else (n_items if total_weight is None else total_weight)
    effective_n = float(effective_source)
    if not math.isfinite(effective_n) or effective_n <= 0.0:
        raise ValueError("kmax effective sample count must be positive and finite")
    mass_bound = int(math.floor(effective_n / float(config.min_effective_samples_per_component)))
    return max(1, min(int(config.abs_kmax), mass_bound, n_items))


def select_gmm_bic(
    projected: np.ndarray,
    *,
    config: GmmBicConfig,
    soft_config: SoftMembershipConfig,
    random_seed: int,
    kmax_override: int | None = None,
    sample_weights: np.ndarray | None = None,
    kmax_effective_n: float | None = None,
) -> GmmBicSelection:
    _validate_gmm_config(config)
    _validate_soft_config(soft_config)
    x = _as_finite_2d_array(projected, name="projected")
    weights = _checked_sample_weights(sample_weights, int(x.shape[0]))
    kmax = _resolve_kmax(int(x.shape[0]), config, weights, kmax_override, kmax_effective_n=kmax_effective_n)

    candidates: list[GmmCandidateFit] = []
    for k in range(1, kmax + 1):
        fit = _fit_best_restart(
            x,
            k=k,
            config=config,
            random_seed=int(random_seed) + k * 9973,
            sample_weights=weights,
        )
        candidates.append(_finalize_candidate(fit, config=config, sample_weights=weights))
    return _select_candidate(candidates)


async def select_gmm_bic_async(
    projected: np.ndarray,
    *,
    config: GmmBicConfig,
    soft_config: SoftMembershipConfig,
    random_seed: int,
    kmax_override: int | None = None,
    sample_weights: np.ndarray | None = None,
    kmax_effective_n: float | None = None,
    max_concurrent_candidates: int = 1,
    max_concurrent_restarts: int = 1,
) -> GmmBicSelection:
    _validate_gmm_config(config)
    _validate_soft_config(soft_config)
    x = _as_finite_2d_array(projected, name="projected")
    weights = _checked_sample_weights(sample_weights, int(x.shape[0]))
    kmax = _resolve_kmax(int(x.shape[0]), config, weights, kmax_override, kmax_effective_n=kmax_effective_n)

    candidate_limit = max(1, int(max_concurrent_candidates))
    restart_limit = max(1, int(max_concurrent_restarts))
    candidate_sem = asyncio.Semaphore(candidate_limit)

    async def fit_k(k: int) -> GmmCandidateFit:
        async with candidate_sem:
            fit = await _fit_best_restart_async(
                x,
                k=k,
                config=config,
                random_seed=int(random_seed) + k * 9973,
                sample_weights=weights,
                max_concurrent_restarts=restart_limit,
            )
            return _finalize_candidate(fit, config=config, sample_weights=weights)

    candidates = await asyncio.gather(*(fit_k(k) for k in range(1, kmax + 1)))
    return _select_candidate(list(candidates))


def soft_memberships(
    item_ids: list[str],
    child_ids: list[str],
    responsibilities: np.ndarray,
    config: SoftMembershipConfig,
) -> dict[str, list[dict[str, float | str]]]:
    _validate_soft_config(config)
    if _has_dupes(item_ids):
        raise ValueError("duplicate item_ids")
    if _has_dupes(child_ids):
        raise ValueError("duplicate child_ids")
    matrix = _as_finite_2d_array(responsibilities, name="responsibilities")
    if matrix.shape != (len(item_ids), len(child_ids)):
        raise ValueError("responsibilities shape must be (len(item_ids), len(child_ids))")
    if np.any(matrix < -_EPS):
        raise ValueError("responsibilities must be non-negative")
    matrix = np.maximum(matrix, 0.0)
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= _EPS):
        raise ValueError("each responsibility row must have positive total mass")
    if not np.allclose(row_sums, 1.0, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("responsibility rows must already be normalized posterior probabilities")

    result: dict[str, list[dict[str, float | str]]] = {}
    for row_index, item_id in enumerate(item_ids):
        selected = _select_membership_indices(matrix[row_index], config)
        result[item_id] = [
            {"child_id": child_ids[int(index)], "weight": float(matrix[row_index, int(index)])}
            for index in selected
        ]
    return result


async def soft_memberships_async(
    item_ids: list[str],
    child_ids: list[str],
    responsibilities: np.ndarray,
    config: SoftMembershipConfig,
) -> dict[str, list[dict[str, float | str]]]:
    return await asyncio.to_thread(soft_memberships, item_ids, child_ids, responsibilities, config)


def membership_weight_dicts(
    item_ids: list[str],
    child_ids: list[str],
    responsibilities: np.ndarray,
    config: SoftMembershipConfig,
) -> dict[str, dict[str, float]]:
    nested = soft_memberships(item_ids, child_ids, responsibilities, config)
    return {
        item_id: {str(edge["child_id"]): float(edge["weight"]) for edge in edges}
        for item_id, edges in nested.items()
    }


async def membership_weight_dicts_async(
    item_ids: list[str],
    child_ids: list[str],
    responsibilities: np.ndarray,
    config: SoftMembershipConfig,
) -> dict[str, dict[str, float]]:
    return await asyncio.to_thread(membership_weight_dicts, item_ids, child_ids, responsibilities, config)


def _resolve_kmax(
    n_items: int,
    config: GmmBicConfig,
    weights: np.ndarray,
    kmax_override: int | None,
    *,
    kmax_effective_n: float | None = None,
) -> int:
    if kmax_override is None:
        return compute_kmax(
            n_items,
            config,
            total_weight=float(weights.sum()),
            kmax_effective_n=kmax_effective_n,
        )
    kmax = int(kmax_override)
    if kmax < 1:
        raise ValueError("kmax_override must be >= 1")
    return max(1, min(kmax, n_items))


def _select_candidate(candidates: list[GmmCandidateFit]) -> GmmBicSelection:
    if not candidates:
        raise ValueError("no GMM candidates")
    valid = [candidate for candidate in candidates if candidate.valid]
    chosen = min(valid, key=lambda item: (item.bic, item.k)) if valid else candidates[0]
    sorted_valid = sorted(valid, key=lambda item: (item.bic, item.k))
    margin = float(sorted_valid[1].bic - sorted_valid[0].bic) if len(sorted_valid) >= 2 else 0.0
    return GmmBicSelection(chosen=chosen, candidates=candidates, bic_margin=margin)


def _fit_best_restart(
    x: np.ndarray,
    *,
    k: int,
    config: GmmBicConfig,
    random_seed: int,
    sample_weights: np.ndarray,
) -> GmmCandidateFit:
    best: GmmCandidateFit | None = None
    for restart in range(config.num_restarts):
        candidate = _fit_single_restart(
            x,
            k=k,
            config=config,
            random_seed=random_seed + restart * 104729,
            sample_weights=sample_weights,
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate
    if best is None:
        raise RuntimeError("no GMM restart ran")
    return best


async def _fit_best_restart_async(
    x: np.ndarray,
    *,
    k: int,
    config: GmmBicConfig,
    random_seed: int,
    sample_weights: np.ndarray,
    max_concurrent_restarts: int,
) -> GmmCandidateFit:
    if max_concurrent_restarts <= 1:
        return await asyncio.to_thread(
            _fit_best_restart,
            x,
            k=k,
            config=config,
            random_seed=random_seed,
            sample_weights=sample_weights,
        )
    restart_sem = asyncio.Semaphore(max(1, int(max_concurrent_restarts)))

    async def run_restart(restart: int) -> GmmCandidateFit:
        async with restart_sem:
            return await asyncio.to_thread(
                _fit_single_restart,
                x,
                k=k,
                config=config,
                random_seed=random_seed + restart * 104729,
                sample_weights=sample_weights,
            )

    candidates = await asyncio.gather(*(run_restart(restart) for restart in range(config.num_restarts)))
    return max(candidates, key=lambda candidate: candidate.log_likelihood)


def _fit_single_restart(
    x: np.ndarray,
    *,
    k: int,
    config: GmmBicConfig,
    random_seed: int,
    sample_weights: np.ndarray,
) -> GmmCandidateFit:
    rng = np.random.default_rng(random_seed)
    pi, means, variances = _initialize_gmm(
        x,
        k,
        rng,
        min_covar=config.min_covar,
        kmeans_init_iters=config.kmeans_init_iters,
        sample_weights=sample_weights,
    )
    variances = _normalize_covariance_family(
        variances,
        covariance_type=config.covariance_type,
        min_covar=config.min_covar,
        weights=pi,
    )
    previous_ll = -math.inf
    responsibilities = np.zeros((x.shape[0], k), dtype=float)
    log_likelihood = -math.inf
    converged = False
    n_iter = 0
    for iteration in range(1, config.max_iter + 1):
        log_prob = _estimate_log_prob(x, pi, means, variances)
        log_norm = _logsumexp(log_prob, axis=1)
        responsibilities = np.exp(log_prob - log_norm[:, None])
        log_likelihood = float(np.dot(sample_weights, log_norm))
        if not math.isfinite(log_likelihood):
            raise FloatingPointError("GMM log-likelihood became non-finite")
        if math.isfinite(previous_ll) and abs(log_likelihood - previous_ll) <= config.tol * max(1.0, abs(previous_ll)):
            converged = True
            n_iter = iteration
            break
        previous_ll = log_likelihood
        weighted_resp = responsibilities * sample_weights[:, None]
        nk = weighted_resp.sum(axis=0) + 10.0 * np.finfo(float).eps
        pi = nk / float(sample_weights.sum())
        means = (weighted_resp.T @ x) / nk[:, None]
        variances = _estimate_variances(
            x,
            means,
            responsibilities,
            nk,
            covariance_type=config.covariance_type,
            min_covar=config.min_covar,
            sample_weights=sample_weights,
        )
        n_iter = iteration

    log_prob = _estimate_log_prob(x, pi, means, variances)
    log_norm = _logsumexp(log_prob, axis=1)
    responsibilities = np.exp(log_prob - log_norm[:, None])
    log_likelihood = float(np.dot(sample_weights, log_norm))
    return _raw_candidate(
        x,
        k,
        pi,
        means,
        variances,
        responsibilities,
        log_likelihood,
        covariance_type=config.covariance_type,
        sample_weights=sample_weights,
        converged=converged,
        n_iter=n_iter,
    )


def _initialize_gmm(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    min_covar: float,
    kmeans_init_iters: int,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_items, dim = x.shape
    total_weight = float(sample_weights.sum())
    if k == 1:
        means = ((x * sample_weights[:, None]).sum(axis=0) / total_weight)[None, :]
        labels = np.zeros(n_items, dtype=int)
    else:
        chosen = _weighted_kmeans_plus_plus_indices(x, k, rng, sample_weights=sample_weights)
        means = x[chosen].copy()
        labels = np.zeros(n_items, dtype=int)
        for _ in range(kmeans_init_iters):
            distances = np.sum((x[:, None, :] - means[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            changed = False
            for component in range(k):
                mask = labels == component
                if not np.any(mask):
                    weighted_distance = sample_weights * np.min(distances, axis=1)
                    farthest = int(np.argmax(weighted_distance))
                    means[component] = x[farthest]
                    labels[farthest] = component
                    changed = True
                else:
                    local_weights = sample_weights[mask]
                    new_mean = (x[mask] * local_weights[:, None]).sum(axis=0) / max(float(local_weights.sum()), np.finfo(float).eps)
                    if float(np.linalg.norm(new_mean - means[component])) > 1.0e-8:
                        changed = True
                    means[component] = new_mean
            if not changed:
                break

    global_mean = (x * sample_weights[:, None]).sum(axis=0) / max(total_weight, np.finfo(float).eps)
    global_var = np.maximum(
        (sample_weights[:, None] * (x - global_mean) ** 2).sum(axis=0) / max(total_weight, np.finfo(float).eps),
        min_covar,
    )
    variances = np.empty((k, dim), dtype=float)
    pi = np.empty(k, dtype=float)
    for component in range(k):
        mask = labels == component
        if np.any(mask):
            local_weights = sample_weights[mask]
            local_weight = float(local_weights.sum())
            pi[component] = local_weight / total_weight
            local_mean = (x[mask] * local_weights[:, None]).sum(axis=0) / max(local_weight, np.finfo(float).eps)
            variances[component] = np.maximum(
                (local_weights[:, None] * (x[mask] - local_mean) ** 2).sum(axis=0) / max(local_weight, np.finfo(float).eps),
                min_covar,
            )
        else:
            pi[component] = np.finfo(float).eps
            variances[component] = global_var
    pi = pi / pi.sum()
    return pi, means, variances


def _weighted_kmeans_plus_plus_indices(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    sample_weights: np.ndarray,
) -> list[int]:
    n_items = x.shape[0]
    first = int(rng.choice(n_items, p=sample_weights / float(sample_weights.sum())))
    chosen = [first]
    distances = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, k):
        weighted_distances = np.maximum(distances, 0.0) * sample_weights
        total = float(weighted_distances.sum())
        if total <= _EPS:
            candidate = int(rng.choice(n_items, p=sample_weights / float(sample_weights.sum())))
        else:
            candidate = int(rng.choice(n_items, p=weighted_distances / total))
        if candidate in chosen:
            candidate = int(np.argmax(weighted_distances))
        chosen.append(candidate)
        distances = np.minimum(distances, np.sum((x - x[candidate]) ** 2, axis=1))
    return chosen


def _raw_candidate(
    x: np.ndarray,
    k: int,
    pi: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    responsibilities: np.ndarray,
    log_likelihood: float,
    *,
    covariance_type: str,
    sample_weights: np.ndarray,
    converged: bool,
    n_iter: int,
) -> GmmCandidateFit:
    primary = np.argmax(responsibilities, axis=1)
    masses = (responsibilities * sample_weights[:, None]).sum(axis=0)
    child_sizes = [int(np.sum(primary == component)) for component in range(k)]
    params = _parameter_count(k, int(x.shape[1]), covariance_type)
    bic = -2.0 * log_likelihood + params * math.log(max(1.0, float(sample_weights.sum())))
    return GmmCandidateFit(
        k=k,
        valid=True,
        bic=float(bic),
        log_likelihood=float(log_likelihood),
        pi=np.asarray(pi, dtype=float),
        means=np.asarray(means, dtype=float),
        variances=np.asarray(variances, dtype=float),
        responsibilities=np.asarray(responsibilities, dtype=float),
        primary_labels=np.asarray(primary, dtype=int),
        component_masses=[float(value) for value in masses],
        child_sizes=child_sizes,
        converged=bool(converged),
        n_iter=int(n_iter),
    )


def _parameter_count(k: int, dim: int, covariance_type: str) -> int:
    weights = k - 1
    means = k * dim
    if covariance_type == "diag":
        covariance = k * dim
    elif covariance_type == "spherical":
        covariance = k
    elif covariance_type == "tied":
        covariance = dim
    else:
        raise ValueError(f"unsupported covariance_type={covariance_type!r}")
    return weights + means + covariance


def _estimate_variances(
    x: np.ndarray,
    means: np.ndarray,
    responsibilities: np.ndarray,
    nk: np.ndarray,
    *,
    covariance_type: str,
    min_covar: float,
    sample_weights: np.ndarray,
) -> np.ndarray:
    k, dim = means.shape
    weighted_resp = responsibilities * sample_weights[:, None]
    diff = x[:, None, :] - means[None, :, :]
    weighted_squared = weighted_resp[:, :, None] * diff * diff

    if covariance_type == "diag":
        variances = weighted_squared.sum(axis=0) / nk[:, None]
        return np.maximum(variances, min_covar)

    if covariance_type == "spherical":
        scalar = weighted_squared.sum(axis=(0, 2)) / (nk * dim)
        scalar = np.maximum(scalar, min_covar)
        return np.repeat(scalar[:, None], dim, axis=1)

    if covariance_type == "tied":
        shared = weighted_squared.sum(axis=(0, 1)) / max(float(sample_weights.sum()), np.finfo(float).eps)
        shared = np.maximum(shared, min_covar)
        return np.repeat(shared[None, :], k, axis=0)

    raise ValueError(f"unsupported covariance_type={covariance_type!r}")


def _normalize_covariance_family(
    variances: np.ndarray,
    *,
    covariance_type: str,
    min_covar: float,
    weights: np.ndarray,
) -> np.ndarray:
    variances = np.maximum(np.asarray(variances, dtype=float), min_covar)
    if covariance_type == "diag":
        return variances
    if covariance_type == "spherical":
        return np.repeat(np.mean(variances, axis=1, keepdims=True), variances.shape[1], axis=1)
    if covariance_type == "tied":
        normalized_weights = weights / max(float(np.sum(weights)), np.finfo(float).eps)
        shared = np.sum(variances * normalized_weights[:, None], axis=0)
        return np.repeat(shared[None, :], variances.shape[0], axis=0)
    raise ValueError(f"unsupported covariance_type={covariance_type!r}")


def _finalize_candidate(
    fit: GmmCandidateFit,
    *,
    config: GmmBicConfig,
    sample_weights: np.ndarray,
) -> GmmCandidateFit:
    if fit.k == 1:
        return fit
    # Hard component-mass gates were removed.  BIC already penalizes unnecessary
    # components, and cumulative soft assignment can make small components useful
    # for budget refinement.  The only structural invalidity here is a component
    # with no primary member at all.
    if min(fit.child_sizes) < 1:
        return _replace_valid(fit, False, "empty_primary_child")
    return fit


def _replace_valid(fit: GmmCandidateFit, valid: bool, reason: str) -> GmmCandidateFit:
    return GmmCandidateFit(
        k=fit.k,
        valid=valid,
        bic=fit.bic,
        log_likelihood=fit.log_likelihood,
        pi=fit.pi,
        means=fit.means,
        variances=fit.variances,
        responsibilities=fit.responsibilities,
        primary_labels=fit.primary_labels,
        component_masses=fit.component_masses,
        child_sizes=fit.child_sizes,
        reason=reason,
        converged=fit.converged,
        n_iter=fit.n_iter,
    )


def _estimate_log_prob(x: np.ndarray, pi: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    dim = x.shape[1]
    safe_variances = np.maximum(variances, np.finfo(float).tiny)
    log_pi = np.log(np.maximum(pi, np.finfo(float).tiny))
    log_det = np.sum(np.log(safe_variances), axis=1)
    diff = x[:, None, :] - means[None, :, :]
    mahal = np.sum((diff * diff) / safe_variances[None, :, :], axis=2)
    return log_pi[None, :] - 0.5 * (dim * math.log(2.0 * math.pi) + log_det[None, :] + mahal)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    if not np.all(np.isfinite(max_values)):
        raise FloatingPointError("logsumexp received non-finite values")
    stable = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.sum(stable, axis=axis))


def _select_membership_indices(row: np.ndarray, config: SoftMembershipConfig) -> np.ndarray:
    order = np.argsort(-row)
    if order.size == 0:
        raise ValueError("responsibility row is empty")
    best = float(row[order[0]])

    if not config.save_soft_edges or config.recursive_assignment == "primary_argmax":
        return order[:1]

    if config.recursive_assignment == "top_r_threshold":
        selected = [int(order[0])]
        for index in order[1 : min(len(order), int(config.top_r_memberships))]:
            weight = float(row[index])
            if weight < config.min_membership_weight:
                continue
            if best - weight > config.max_membership_gap:
                continue
            selected.append(int(index))
        return np.asarray(selected, dtype=int)

    if config.recursive_assignment == "cumulative_mass":
        # Select enough high-posterior components to cover the requested mass,
        # but do not pull in low-confidence tail components just to reach the
        # target.  The max-gap gate keeps cumulative assignment as "near-tie"
        # soft membership rather than noisy long-tail coverage.
        selected: list[int] = [int(order[0])]
        mass = float(row[order[0]])
        for index in order[1:]:
            weight = float(row[index])
            if best - weight > config.max_membership_gap:
                break
            selected.append(int(index))
            mass += weight
            if mass >= config.cumulative_mass_coverage:
                break
        return np.asarray(selected, dtype=int)

    raise ValueError(f"unsupported recursive_assignment={config.recursive_assignment!r}")


def _as_finite_2d_array(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} values must be finite")
    return values


def _checked_sample_weights(sample_weights: np.ndarray | None, n_items: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(n_items, dtype=float)
    weights = np.asarray(sample_weights, dtype=float)
    if weights.shape != (n_items,):
        raise ValueError("sample_weights must have shape (n_items,)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("sample_weights must be positive")
    return weights


def _validate_gmm_config(config: GmmBicConfig) -> None:
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()


def _validate_soft_config(config: SoftMembershipConfig) -> None:
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()


def _has_dupes(values: list[str]) -> bool:
    return len(set(values)) != len(values)


def gmm_responsibilities_from_model(
    projected: np.ndarray,
    *,
    pi: np.ndarray | list[float],
    means: np.ndarray | list[list[float]],
    variances: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Return full GMM posterior responsibilities for saved routing models."""
    x = _as_finite_2d_array(projected, name="projected")
    pi_arr = np.asarray(pi, dtype=float)
    means_arr = _as_finite_2d_array(np.asarray(means, dtype=float), name="means")
    var_arr = _as_finite_2d_array(np.asarray(variances, dtype=float), name="variances")
    if pi_arr.ndim != 1:
        raise ValueError("pi must be a 1D vector")
    if means_arr.shape != var_arr.shape or means_arr.shape[0] != pi_arr.shape[0]:
        raise ValueError("GMM parameter shapes are inconsistent")
    if x.shape[1] != means_arr.shape[1]:
        raise ValueError("projected dimension does not match GMM means")
    log_prob = _estimate_log_prob(x, pi_arr, means_arr, var_arr)
    log_norm = _logsumexp(log_prob, axis=1)
    return np.exp(log_prob - log_norm[:, None])


def online_em_update_saved_gmm(
    projected: np.ndarray,
    *,
    sample_weights: np.ndarray | list[float],
    responsibilities: np.ndarray,
    model: dict,
    min_covar: float = 1.0e-6,
) -> dict:
    """Fixed-K online-EM update for a saved diagonal/spherical GMM.

    The update uses full posterior responsibilities and item support_mass as
    sample weights.  The returned statistics are internal routing-model
    effective counts, not ExperienceCommunity.support_mass.
    """
    x = _as_finite_2d_array(projected, name="projected")
    resp = _as_finite_2d_array(np.asarray(responsibilities, dtype=float), name="responsibilities")
    if resp.shape[0] != x.shape[0]:
        raise ValueError("responsibilities row count must match projected rows")
    if np.any(resp < -_EPS):
        raise ValueError("responsibilities must be non-negative")
    resp = np.maximum(resp, 0.0)
    row_sums = resp.sum(axis=1, keepdims=True)
    resp = resp / np.where(row_sums <= _EPS, 1.0, row_sums)
    weights = _checked_sample_weights(np.asarray(sample_weights, dtype=float), x.shape[0])

    pi_old = np.asarray(model.get("pi"), dtype=float)
    means_old = _as_finite_2d_array(np.asarray(model.get("means"), dtype=float), name="means")
    var_old = _as_finite_2d_array(np.asarray(model.get("variances"), dtype=float), name="variances")
    if pi_old.ndim != 1 or means_old.shape != var_old.shape or means_old.shape[0] != pi_old.shape[0]:
        raise ValueError("saved GMM model parameter shapes are inconsistent")
    k, dim = means_old.shape
    if x.shape[1] != dim or resp.shape[1] != k:
        raise ValueError("saved GMM model shape does not match projected/responsibilities")

    old_counts = np.asarray(model.get("component_effective_counts", model.get("component_masses", [])), dtype=float)
    if old_counts.shape != (k,) or not np.all(np.isfinite(old_counts)) or np.any(old_counts < 0.0):
        total_old = float(model.get("total_effective_count", np.sum(old_counts) if old_counts.size else 0.0))
        if not math.isfinite(total_old) or total_old <= 0.0:
            total_old = float(np.sum(pi_old))
        old_counts = np.maximum(pi_old, 0.0) * max(total_old, _EPS)

    old_first = np.asarray(model.get("first_moments", []), dtype=float)
    old_second = np.asarray(model.get("second_moments", []), dtype=float)
    if old_first.shape != (k, dim):
        old_first = old_counts[:, None] * means_old
    if old_second.shape != (k, dim):
        old_second = old_counts[:, None] * (var_old + means_old * means_old)

    weighted_resp = resp * weights[:, None]
    delta_counts = weighted_resp.sum(axis=0)
    delta_first = weighted_resp.T @ x
    delta_second = weighted_resp.T @ (x * x)

    counts = old_counts + delta_counts
    safe_counts = np.maximum(counts, np.finfo(float).eps)
    first = old_first + delta_first
    second = old_second + delta_second
    means = first / safe_counts[:, None]
    variances = np.maximum(second / safe_counts[:, None] - means * means, float(min_covar))
    total = float(counts.sum())
    pi = counts / max(total, np.finfo(float).eps)

    updated = dict(model)
    updated.update(
        {
            "pi": pi.astype(float).tolist(),
            "means": means.astype(float).tolist(),
            "variances": variances.astype(float).tolist(),
            "component_effective_counts": counts.astype(float).tolist(),
            "total_effective_count": total,
            "first_moments": first.astype(float).tolist(),
            "second_moments": second.astype(float).tolist(),
            "online_em_updates": int(model.get("online_em_updates", 0)) + 1,
        }
    )
    return updated

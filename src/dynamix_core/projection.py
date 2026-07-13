from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .config import ProjectionConfig

_EPS = 1.0e-12
_SKLEARN_RANDOM_SEED_MODULUS = 2**32


@dataclass(frozen=True)
class ProjectionResult:
    projected: np.ndarray
    dim: int
    explained_variance_ratio: float
    spectrum: list[float]
    mean: np.ndarray
    components: np.ndarray


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = _as_finite_2d_array(matrix, name="matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms <= _EPS, 1.0, norms)
    return values / norms


async def normalize_rows_async(matrix: np.ndarray) -> np.ndarray:
    return await asyncio.to_thread(normalize_rows, matrix)


def local_pca_project(
    embeddings: np.ndarray,
    config: ProjectionConfig,
    *,
    random_seed: int,
) -> ProjectionResult:
    _validate_projection_config(config)
    values = _as_finite_2d_array(embeddings, name="embeddings")

    n_items, input_dim = values.shape
    if n_items < 2:
        raise ValueError("local PCA requires at least two items")

    max_components = min(input_dim, n_items - 1, int(config.max_dim))
    if max_components < 1:
        raise ValueError("projection has no valid component dimension")

    if _is_nearly_constant(values):
        return _constant_projection(values, max_components=max_components, min_dim=int(config.min_dim))

    pca = PCA(
        n_components=max_components,
        whiten=bool(config.whiten),
        svd_solver="auto",
        random_state=int(random_seed) % _SKLEARN_RANDOM_SEED_MODULUS,
    )
    projected_full = pca.fit_transform(values)

    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    dim = _select_dimension(
        ratios=ratios,
        variance_ratio=float(config.variance_ratio),
        min_dim=int(config.min_dim),
        max_components=max_components,
    )

    return ProjectionResult(
        projected=np.asarray(projected_full[:, :dim], dtype=float),
        dim=dim,
        explained_variance_ratio=float(ratios[:dim].sum()),
        spectrum=[float(value) for value in np.asarray(pca.explained_variance_, dtype=float)],
        mean=np.asarray(pca.mean_, dtype=float),
        components=np.asarray(pca.components_[:dim], dtype=float),
    )


async def local_pca_project_async(
    embeddings: np.ndarray,
    config: ProjectionConfig,
    *,
    random_seed: int,
) -> ProjectionResult:
    return await asyncio.to_thread(
        local_pca_project,
        embeddings,
        config,
        random_seed=random_seed,
    )


def project_with_basis(embeddings: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = _as_finite_2d_array(embeddings, name="embeddings")
    mean_values = np.asarray(mean, dtype=float)
    component_values = _as_finite_2d_array(components, name="components")

    if mean_values.ndim != 1:
        raise ValueError("mean must be a 1D vector")
    if values.shape[1] != mean_values.shape[0]:
        raise ValueError("embedding dimension does not match PCA mean dimension")
    if component_values.shape[1] != mean_values.shape[0]:
        raise ValueError("components dimension does not match PCA mean dimension")

    return (values - mean_values[None, :]) @ component_values.T


async def project_with_basis_async(embeddings: np.ndarray, *, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return await asyncio.to_thread(project_with_basis, embeddings, mean=mean, components=components)


def _validate_projection_config(config: ProjectionConfig) -> None:
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()


def _as_finite_2d_array(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} values must be finite")
    return values


def _is_nearly_constant(values: np.ndarray) -> bool:
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered)) <= _EPS


def _constant_projection(values: np.ndarray, *, max_components: int, min_dim: int) -> ProjectionResult:
    dim = max(1, min(min_dim, max_components))
    components = np.eye(values.shape[1], dim, dtype=float).T
    return ProjectionResult(
        projected=np.zeros((values.shape[0], dim), dtype=float),
        dim=dim,
        explained_variance_ratio=0.0,
        spectrum=[0.0 for _ in range(max_components)],
        mean=values.mean(axis=0),
        components=components,
    )


def _select_dimension(
    *,
    ratios: np.ndarray,
    variance_ratio: float,
    min_dim: int,
    max_components: int,
) -> int:
    if ratios.ndim != 1 or ratios.size == 0:
        return max(1, min(min_dim, max_components))
    cumulative = np.cumsum(ratios)
    dim_by_variance = int(np.searchsorted(cumulative, variance_ratio, side="left") + 1)
    lower = max(1, min(min_dim, max_components))
    return max(lower, min(dim_by_variance, max_components))

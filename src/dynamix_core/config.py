from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProjectionConfig:
    method: str = "local_pca"
    variance_ratio: float = 0.90
    max_dim: int = 32
    min_dim: int = 2
    whiten: bool = False

    def validate(self) -> None:
        if self.method != "local_pca":
            raise ValueError("projection.method must be 'local_pca'")
        if not (0.0 < self.variance_ratio <= 1.0):
            raise ValueError("projection.variance_ratio must be in (0, 1]")
        if self.max_dim < 1:
            raise ValueError("projection.max_dim must be >= 1")
        if self.min_dim < 1:
            raise ValueError("projection.min_dim must be >= 1")
        if self.min_dim > self.max_dim:
            raise ValueError("projection.min_dim must be <= projection.max_dim")
        if self.whiten:
            raise ValueError("projection.whiten=false is required for stable routing reuse")


@dataclass(frozen=True)
class GmmBicConfig:
    covariance_type: str = "spherical"
    num_restarts: int = 5
    kmeans_init_iters: int = 15
    max_iter: int = 100
    tol: float = 1.0e-4
    min_covar: float = 1.0e-6
    min_split_size: int = 2
    # Used only to upper-bound searched K: kmax <= floor(total_weight / this value).
    # It is not a hard candidate-validity constraint.
    min_effective_samples_per_component: int = 2
    abs_kmax: int = 64
    max_concurrent_candidates: int = 1
    max_concurrent_restarts: int = 1

    def validate(self) -> None:
        if self.covariance_type not in {"diag", "spherical", "tied"}:
            raise ValueError("gmm_bic.covariance_type must be one of: diag, spherical, tied")
        if self.num_restarts < 1:
            raise ValueError("gmm_bic.num_restarts must be >= 1")
        if self.kmeans_init_iters < 1:
            raise ValueError("gmm_bic.kmeans_init_iters must be >= 1")
        if self.max_iter < 1:
            raise ValueError("gmm_bic.max_iter must be >= 1")
        if self.tol <= 0.0:
            raise ValueError("gmm_bic.tol must be positive")
        if self.min_covar <= 0.0:
            raise ValueError("gmm_bic.min_covar must be positive")
        if self.min_split_size < 2:
            raise ValueError("gmm_bic.min_split_size must be >= 2")
        if self.min_effective_samples_per_component < 1:
            raise ValueError("gmm_bic.min_effective_samples_per_component must be >= 1")
        if self.abs_kmax < 1:
            raise ValueError("gmm_bic.abs_kmax must be >= 1")
        if self.max_concurrent_candidates < 1:
            raise ValueError("gmm_bic.max_concurrent_candidates must be >= 1")
        if self.max_concurrent_restarts < 1:
            raise ValueError("gmm_bic.max_concurrent_restarts must be >= 1")


@dataclass(frozen=True)
class SoftMembershipConfig:
    save_soft_edges: bool = True
    top_r_memberships: int = 2
    recursive_assignment: str = "cumulative_mass"
    min_membership_weight: float = 0.05
    max_membership_gap: float = 0.25
    cumulative_mass_coverage: float = 0.90

    def validate(self) -> None:
        allowed = {"primary_argmax", "top_r_threshold", "cumulative_mass"}
        if self.recursive_assignment not in allowed:
            raise ValueError(f"soft_membership.recursive_assignment must be one of {sorted(allowed)}")
        if self.top_r_memberships < 1:
            raise ValueError("soft_membership.top_r_memberships must be >= 1")
        if not (0.0 <= self.min_membership_weight <= 1.0):
            raise ValueError("soft_membership.min_membership_weight must be in [0, 1]")
        if not (0.0 <= self.max_membership_gap <= 1.0):
            raise ValueError("soft_membership.max_membership_gap must be in [0, 1]")
        if not (0.0 < self.cumulative_mass_coverage <= 1.0):
            raise ValueError("soft_membership.cumulative_mass_coverage must be in (0, 1]")
        if self.recursive_assignment != "primary_argmax" and self.top_r_memberships < 2:
            raise ValueError("overlapping assignment requires top_r_memberships >= 2")


@dataclass(frozen=True)
class SummaryBudgetConfig:
    max_model_tokens: int = 100_000
    budget_ratio: float = 0.70
    prompt_overhead_reserve_tokens: int = 0
    token_count_metadata_keys: tuple[str, ...] = (
        "analysis_token_count",
        "prompt_token_count",
        "token_count",
        "tokens",
    )

    @property
    def analyst_prompt_token_budget(self) -> int:
        return int(self.max_model_tokens * self.budget_ratio)

    @property
    def member_evidence_token_budget(self) -> int:
        return self.analyst_prompt_token_budget - int(self.prompt_overhead_reserve_tokens)

    @property
    def effective_token_budget(self) -> int:
        return self.member_evidence_token_budget

    def validate(self) -> None:
        if self.max_model_tokens < 1:
            raise ValueError("summary_budget.max_model_tokens must be >= 1")
        if not (0.0 < self.budget_ratio <= 1.0):
            raise ValueError("summary_budget.budget_ratio must be in (0, 1]")
        if int(self.prompt_overhead_reserve_tokens) < 0:
            raise ValueError("summary_budget.prompt_overhead_reserve_tokens must be >= 0")
        if self.member_evidence_token_budget < 1:
            raise ValueError("summary_budget.member_evidence_token_budget must be >= 1")
        if not self.token_count_metadata_keys:
            raise ValueError("summary_budget.token_count_metadata_keys must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["analyst_prompt_token_budget"] = self.analyst_prompt_token_budget
        payload["member_evidence_token_budget"] = self.member_evidence_token_budget
        payload["effective_token_budget"] = self.effective_token_budget
        payload["token_count_metadata_keys"] = list(self.token_count_metadata_keys)
        return payload


@dataclass(frozen=True)
class BudgetRefinementConfig:
    """GMM-BIC budget-refinement policy for over-long raw communities.

    This is not a semantic hierarchy.  It first keeps the normal coarse GMM-BIC
    layer split, then recursively refines only over-budget raw communities with
    local GMM-BIC splits and flattens feasible leaves back into the same layer.
    """

    enabled: bool = True
    apply_to_level: int = 0
    selection_policy: str = "bic_best_with_token_progress"
    min_token_reduction_fraction: float = 0.10
    fallback: str = "gmm_bic_recursive"
    flatten_refinement_leaves_to_l0: bool = True
    skip_oversize_singleton: bool = True

    def validate(self) -> None:
        if self.apply_to_level < 0:
            raise ValueError("budget_refinement.apply_to_level must be >= 0")
        if self.selection_policy != "bic_best_with_token_progress":
            raise ValueError("budget_refinement.selection_policy must be 'bic_best_with_token_progress'")
        if not (0.0 <= self.min_token_reduction_fraction < 1.0):
            raise ValueError("budget_refinement.min_token_reduction_fraction must be in [0, 1)")
        if self.fallback != "gmm_bic_recursive":
            raise ValueError("budget_refinement.fallback must be 'gmm_bic_recursive'")
        if not self.flatten_refinement_leaves_to_l0:
            raise ValueError("budget_refinement.flatten_refinement_leaves_to_l0 must be true")



@dataclass(frozen=True)
class DynamicUpdateConfig:
    """Budget-constrained online GMM dynamic update configuration.

    Trajectories are inserted sequentially.  L0 candidates are accepted only
    when adding the trajectory keeps the analyst prompt under budget; otherwise
    a new L0 community/GMM component is created when the trajectory itself fits
    the budget.  Existing communities are not split during online insertion.
    """

    mode: str = "budget_constrained_online_gmm"
    assignment: str = "cumulative_mass"
    top_r: int = 2
    min_membership_weight: float = 0.05
    max_membership_gap: float = 0.25
    cumulative_mass_coverage: float = 0.90
    update_routing_model: bool = True
    clear_stale_after_propagation: bool = True
    confidence_metadata_key: str = "confidence"

    def validate(self) -> None:
        if self.mode != "budget_constrained_online_gmm":
            raise ValueError("dynamic_update.mode must be 'budget_constrained_online_gmm'")
        allowed = {"primary_argmax", "top_r_threshold", "cumulative_mass"}
        if self.assignment not in allowed:
            raise ValueError(f"dynamic_update.assignment must be one of {sorted(allowed)}")
        if self.top_r < 1:
            raise ValueError("dynamic_update.top_r must be >= 1")
        if self.assignment != "primary_argmax" and self.top_r < 2:
            raise ValueError("overlapping dynamic assignment requires top_r >= 2")
        if not (0.0 <= self.min_membership_weight <= 1.0):
            raise ValueError("dynamic_update.min_membership_weight must be in [0, 1]")
        if not (0.0 <= self.max_membership_gap <= 1.0):
            raise ValueError("dynamic_update.max_membership_gap must be in [0, 1]")
        if not (0.0 < self.cumulative_mass_coverage <= 1.0):
            raise ValueError("dynamic_update.cumulative_mass_coverage must be in (0, 1]")
        if not self.update_routing_model:
            raise ValueError("dynamic_update.update_routing_model must be true for budget_constrained_online_gmm")
        if not self.confidence_metadata_key:
            raise ValueError("dynamic_update.confidence_metadata_key must be non-empty")

    def to_soft_membership_config(self) -> SoftMembershipConfig:
        return SoftMembershipConfig(
            save_soft_edges=True,
            top_r_memberships=int(self.top_r),
            recursive_assignment=str(self.assignment),
            min_membership_weight=float(self.min_membership_weight),
            max_membership_gap=float(self.max_membership_gap),
            cumulative_mass_coverage=float(self.cumulative_mass_coverage),
        )


@dataclass(frozen=True)
class ProjectedGmmDynamicTreeConfig:
    tree_policy: str = "projected_gmm_bic"
    graph_kind: str = "overlapping_experience_hierarchy"
    allow_overlap: bool = True
    allow_multi_parent: bool = True
    use_support_mass: bool = True
    random_seed: int = 42
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    gmm_bic: GmmBicConfig = field(default_factory=GmmBicConfig)
    soft_membership: SoftMembershipConfig = field(default_factory=SoftMembershipConfig)
    summary_budget: SummaryBudgetConfig = field(default_factory=SummaryBudgetConfig)
    budget_refinement: BudgetRefinementConfig = field(default_factory=BudgetRefinementConfig)
    dynamic_update: DynamicUpdateConfig = field(default_factory=DynamicUpdateConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ProjectedGmmDynamicTreeConfig":
        data = dict(payload or {})
        config = cls(
            tree_policy=str(data.get("tree_policy", "projected_gmm_bic")),
            graph_kind=str(data.get("graph_kind", "overlapping_experience_hierarchy")),
            allow_overlap=bool(data.get("allow_overlap", True)),
            allow_multi_parent=bool(data.get("allow_multi_parent", True)),
            use_support_mass=bool(data.get("use_support_mass", True)),
            random_seed=int(data.get("random_seed", 42)),
            projection=ProjectionConfig(**dict(data.get("projection", {}))),
            gmm_bic=GmmBicConfig(**_clean_gmm_bic_payload(dict(data.get("gmm_bic", {})))),
            soft_membership=SoftMembershipConfig(**_clean_soft_membership_payload(dict(data.get("soft_membership", {})))),
            summary_budget=SummaryBudgetConfig(**_clean_summary_budget_payload(dict(data.get("summary_budget", {})))),
            budget_refinement=BudgetRefinementConfig(**dict(data.get("budget_refinement", {}))),
            dynamic_update=DynamicUpdateConfig(**_normalize_dynamic_update_payload(dict(data.get("dynamic_update", {})))),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.tree_policy != "projected_gmm_bic":
            raise ValueError("tree_policy must be 'projected_gmm_bic'")
        if self.graph_kind not in {"overlapping_experience_hierarchy", "overlapping_hierarchy_dag"}:
            raise ValueError("graph_kind must be overlapping_experience_hierarchy or overlapping_hierarchy_dag")
        if not self.allow_overlap:
            raise ValueError("allow_overlap must be true")
        if not self.allow_multi_parent:
            raise ValueError("allow_multi_parent must be true")
        if not self.use_support_mass:
            raise ValueError("use_support_mass must be true")
        self.projection.validate()
        self.gmm_bic.validate()
        self.soft_membership.validate()
        self.summary_budget.validate()
        self.budget_refinement.validate()
        self.dynamic_update.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_gmm_bic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # These old hard validity gates conflict with cumulative soft assignment and
    # budget-driven refinement.  BIC still penalizes unnecessary components;
    # kmax is still bounded by min_effective_samples_per_component.
    payload.pop("min_child_mass", None)
    payload.pop("min_child_fraction", None)
    return payload


def _clean_soft_membership_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("normalize_selected_memberships", None)
    return payload


def _clean_summary_budget_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "token_count_metadata_keys" in payload and not isinstance(payload["token_count_metadata_keys"], tuple):
        payload["token_count_metadata_keys"] = tuple(payload["token_count_metadata_keys"])
    return payload


def _normalize_dynamic_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept old payload aliases, but normalize to the current online surface."""
    mapping = {
        "affected_assignment": "assignment",
        "affected_top_r": "top_r",
        "nearest_leaf_top_r": "top_r",
        "affected_min_weight": "min_membership_weight",
        "affected_max_gap": "max_membership_gap",
        "affected_mass_coverage": "cumulative_mass_coverage",
    }
    for old, new in mapping.items():
        if old in payload and new not in payload:
            payload[new] = payload[old]
    allowed = set(DynamicUpdateConfig.__dataclass_fields__)
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload.setdefault("mode", "budget_constrained_online_gmm")
    return payload


__all__ = [
    "BudgetRefinementConfig",
    "DynamicUpdateConfig",
    "GmmBicConfig",
    "ProjectedGmmDynamicTreeConfig",
    "ProjectionConfig",
    "SoftMembershipConfig",
    "SummaryBudgetConfig",
]

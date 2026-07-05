from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Sequence

import numpy as np

from .config import ProjectedGmmDynamicTreeConfig, SoftMembershipConfig
from .data_structures import (
    ExperienceCommunity,
    ExperienceHierarchyState,
    ExperienceItem,
    ITEM_KIND_EXPERIENCE_CARD,
)
from .gmm_bic import GmmBicSelection, GmmCandidateFit, compute_kmax, membership_weight_dicts, select_gmm_bic_async
from .kmeans_split import select_kmeans_split
from .projection import ProjectionResult, local_pca_project, normalize_rows

SummaryFn = Callable[
    [ExperienceCommunity, list[ExperienceItem], "LayerClusteringResult"],
    list[ExperienceItem] | Awaitable[list[ExperienceItem]],
]
PromptTokenEstimator = Callable[
    [ExperienceCommunity, list[ExperienceItem]],
    int | Awaitable[int],
]


@dataclass(frozen=True)
class LayerRoutingModel:
    level: int
    community_ids: list[str]
    pca_mean: list[float]
    pca_components: list[list[float]]
    pi: list[float]
    means: list[list[float]]
    variances: list[list[float]]
    covariance_type: str
    component_effective_counts: list[float]
    total_effective_count: float
    soft_assignment: dict

    def to_dict(self) -> dict:
        return {
            "routing_model_kind": "fixed_k_pca_gmm",
            "level": self.level,
            "community_ids": list(self.community_ids),
            "pca_mean": list(self.pca_mean),
            "pca_components": [list(row) for row in self.pca_components],
            "pi": list(self.pi),
            "means": [list(row) for row in self.means],
            "variances": [list(row) for row in self.variances],
            "covariance_type": self.covariance_type,
            "component_effective_counts": list(self.component_effective_counts),
            "total_effective_count": float(self.total_effective_count),
            "soft_assignment": dict(self.soft_assignment),
        }


@dataclass(frozen=True)
class LayerClusteringResult:
    level: int
    input_item_ids: list[str]
    communities: list[ExperienceCommunity]
    member_item_ids_by_community: dict[str, list[str]]
    stop_reason: str
    excluded_input_item_ids: list[str] = field(default_factory=list)
    projection_dim: int = 0
    explained_variance_ratio: float = 0.0
    pca_spectrum: list[float] = field(default_factory=list)
    chosen_k: int = 1
    tested_k: list[int] = field(default_factory=list)
    bic_by_k: dict[str, float] = field(default_factory=dict)
    log_likelihood_by_k: dict[str, float] = field(default_factory=dict)
    bic_margin: float = 0.0
    selection_score_kind: str = "bic"
    split_selector: str = "weighted_gmm_bic"
    routing_model: LayerRoutingModel | None = None
    summary_budget: dict[str, Any] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return bool(self.stop_reason) and not self.communities


@dataclass(frozen=True)
class LayerBuildResult:
    clustering: LayerClusteringResult
    generated_item_ids: list[str]
    committed: bool


@dataclass(frozen=True)
class HierarchyBuildResult:
    state: ExperienceHierarchyState
    layers: list[LayerBuildResult]


@dataclass
class ProjectedGmmTreeBuilder:
    """Static bottom-up builder for the v1 overlapping experience hierarchy.

    The builder owns only projection + weighted GMM-BIC clustering + safe layer
    commits.  It stops before LLM summarization when BIC selects K=1, so no
    artificial root ExperienceCard is generated.
    """

    config: ProjectedGmmDynamicTreeConfig

    async def build(
        self,
        items: Iterable[ExperienceItem],
        *,
        summary_fn: SummaryFn,
        prompt_token_estimator: PromptTokenEstimator | None = None,
        prompt_token_budget: int | None = None,
        max_levels: int = 8,
    ) -> HierarchyBuildResult:
        if max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items(list(items))
        layer_results: list[LayerBuildResult] = []
        for level in range(max_levels):
            layer_items = await state.item_objects_at_level(level)
            if not layer_items:
                break
            result = await self.build_layer(
                state,
                level=level,
                items=layer_items,
                summary_fn=summary_fn,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )
            layer_results.append(result)
            if not result.committed:
                break
        return HierarchyBuildResult(state=state, layers=layer_results)

    async def build_layer(
        self,
        state: ExperienceHierarchyState,
        *,
        level: int,
        summary_fn: SummaryFn,
        items: Sequence[ExperienceItem] | None = None,
        prompt_token_estimator: PromptTokenEstimator | None = None,
        prompt_token_budget: int | None = None,
    ) -> LayerBuildResult:
        layer_items = list(items) if items is not None else await state.item_objects_at_level(level)
        clustering = await self.cluster_layer(
            layer_items,
            level=level,
            prompt_token_estimator=prompt_token_estimator,
            prompt_token_budget=prompt_token_budget,
        )
        if clustering.should_stop:
            return LayerBuildResult(clustering=clustering, generated_item_ids=[], committed=False)

        items_by_id = {item.item_id: item for item in layer_items}
        generated = await self._summarize_communities(clustering, items_by_id=items_by_id, summary_fn=summary_fn)
        if not generated:
            return LayerBuildResult(clustering=clustering, generated_item_ids=[], committed=False)

        metadata = {
            "builder": self._builder_name(),
            "tree_policy": self.config.tree_policy,
            "chosen_k": clustering.chosen_k,
            "bic_margin": clustering.bic_margin,
            "projection_dim": clustering.projection_dim,
            "routing_model": clustering.routing_model.to_dict() if clustering.routing_model else None,
            "budget_refinement": clustering.summary_budget,
            "excluded_input_item_ids": list(clustering.excluded_input_item_ids),
            "summary_budget_contract": {
                "token_budget_source": "config.summary_budget",
                "note": "Upstream items should provide token-count metadata if over-budget splitting is required before summary_fn.",
            },
        }
        await state.commit_layer(
            level=level,
            communities=clustering.communities,
            generated_items=generated,
            stop_reason="split",
            metadata=metadata,
            excluded_input_item_ids=clustering.excluded_input_item_ids,
        )
        return LayerBuildResult(
            clustering=clustering,
            generated_item_ids=[item.item_id for item in generated],
            committed=True,
        )

    async def cluster_layer(
        self,
        items: Sequence[ExperienceItem],
        *,
        level: int,
        prompt_token_estimator: PromptTokenEstimator | None = None,
        prompt_token_budget: int | None = None,
    ) -> LayerClusteringResult:
        self.config.validate()
        ordered = sorted(list(items), key=lambda item: item.item_id)
        input_ids = [item.item_id for item in ordered]
        if not ordered:
            return _stopped(level, input_ids, "empty_layer")
        for item in ordered:
            if item.level != level:
                raise ValueError(f"item {item.item_id!r} has level {item.level}, expected {level}")
            if not item.embedding:
                raise ValueError(f"item {item.item_id!r} is missing embedding")

        n_items = len(ordered)
        items_by_id = {item.item_id: item for item in ordered}
        token_counts = {item.item_id: _item_token_count(item, self.config.summary_budget.token_count_metadata_keys) for item in ordered}
        active_prompt_token_budget = int(prompt_token_budget or self.config.summary_budget.analyst_prompt_token_budget)
        token_budget = (
            active_prompt_token_budget
            if prompt_token_estimator is not None
            else self.config.summary_budget.effective_token_budget
        )
        total_prompt_tokens = sum(token_counts.values())
        if prompt_token_estimator is not None and level == self.config.budget_refinement.apply_to_level:
            success_count, failure_count, outcome_mode = _outcome_counts(ordered)
            coarse_for_budget = ExperienceCommunity(
                community_id=f"L{level}_C0",
                level=level,
                member_weights={item_id: 1.0 for item_id in input_ids},
                posterior_member_weights={item_id: 1.0 for item_id in input_ids},
                clustering_method="weighted_gmm_bic_budget_probe",
                support_mass=_support_mass(items_by_id, {item_id: 1.0 for item_id in input_ids}),
                outcome_mode=outcome_mode,
                success_count=success_count,
                failure_count=failure_count,
                metadata={"budget_probe": True},
            )
            total_prompt_tokens = await self._estimate_community_token_cost(
                coarse_for_budget,
                input_ids,
                items_by_id=items_by_id,
                token_counts=token_counts,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=active_prompt_token_budget,
            )

        budget_refinement_enabled = (
            self.config.budget_refinement.enabled
            and level == self.config.budget_refinement.apply_to_level
            and total_prompt_tokens > token_budget
        )

        if n_items < self.config.gmm_bic.min_split_size and not budget_refinement_enabled:
            return _stopped(level, input_ids, "too_small")

        effective_kmax = compute_kmax(
            n_items,
            self.config.gmm_bic,
            kmax_effective_n=float(n_items),
        )
        if effective_kmax <= 1 and not budget_refinement_enabled:
            return _stopped(level, input_ids, "bic_selected_one")

        if n_items < self.config.gmm_bic.min_split_size and budget_refinement_enabled:
            success_count, failure_count, outcome_mode = _outcome_counts(ordered)
            coarse = ExperienceCommunity(
                community_id=f"L{level}_C0",
                level=level,
                member_weights={item_id: 1.0 for item_id in input_ids},
                posterior_member_weights={item_id: 1.0 for item_id in input_ids},
                clustering_method="weighted_gmm_bic_single_refined",
                support_mass=_support_mass(items_by_id, {item_id: 1.0 for item_id in input_ids}),
                outcome_mode=outcome_mode,
                success_count=success_count,
                failure_count=failure_count,
                metadata={
                    "component_index": 0,
                    "budget_refinement_coarse_root": True,
                    "too_few_items_for_gmm": True,
                    "too_few_items_for_pca": n_items < 2,
                },
            )
            refined = await self._refine_overbudget_coarse_communities(
                [coarse],
                {coarse.community_id: input_ids},
                level=level,
                input_ids=input_ids,
                items_by_id=items_by_id,
                token_counts=token_counts,
                token_budget=token_budget,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=active_prompt_token_budget,
            )
            return LayerClusteringResult(
                level=level,
                input_item_ids=input_ids,
                communities=list(refined["communities"]),
                member_item_ids_by_community=dict(refined["member_item_ids_by_community"]),
                stop_reason="" if refined["communities"] else "budget_refinement_no_active_communities",
                excluded_input_item_ids=list(refined["excluded_input_item_ids"]),
                chosen_k=1,
                tested_k=[1],
                summary_budget={
                    "effective_token_budget": int(token_budget),
                    "selection_score_kind": "bic",
                    "split_selector": "weighted_gmm_bic",
                    "bic_selected_k": 1,
                    "coarse_selected_k": 1,
                    "budget_enforced": bool(refined["summary_budget"].get("refinement_routing_tree") or refined["excluded_input_item_ids"]),
                    **refined["summary_budget"],
                },
            )

        embeddings = normalize_rows(np.asarray([item.embedding for item in ordered], dtype=float))
        projection = local_pca_project(embeddings, self.config.projection)
        weights = _normalized_gmm_sample_weights(ordered)
        selection = await self._select_projected_split(
            projection.projected,
            level=level,
            n_items=n_items,
            random_seed=int(self.config.random_seed) + level * 100003 + n_items,
            sample_weights=weights,
            kmax_effective_n=float(n_items),
        )
        if selection.chosen.k <= 1:
            if budget_refinement_enabled:
                coarse = ExperienceCommunity(
                    community_id=f"L{level}_C0",
                    level=level,
                    member_weights={item_id: 1.0 for item_id in input_ids},
                    posterior_member_weights={item_id: 1.0 for item_id in input_ids},
                    clustering_method="weighted_gmm_bic_single_refined",
                    support_mass=_support_mass(items_by_id, {item_id: 1.0 for item_id in input_ids}),
                    outcome_mode=_outcome_counts(ordered)[2],
                    success_count=_outcome_counts(ordered)[0],
                    failure_count=_outcome_counts(ordered)[1],
                    metadata={
                        "component_index": 0,
                        "chosen_k": int(selection.chosen.k),
                        "bic": float(selection.chosen.bic),
                        "log_likelihood": float(selection.chosen.log_likelihood),
                        "budget_refinement_coarse_root": True,
                    },
                )
                refined = await self._refine_overbudget_coarse_communities(
                    [coarse],
                    {coarse.community_id: input_ids},
                    level=level,
                    input_ids=input_ids,
                    items_by_id=items_by_id,
                    token_counts=token_counts,
                    token_budget=token_budget,
                    prompt_token_estimator=prompt_token_estimator,
                    prompt_token_budget=active_prompt_token_budget,
                )
                return _clustering_from_refinement(
                    level=level,
                    input_ids=input_ids,
                    projection=projection,
                    selection=selection,
                    fit=selection.chosen,
                    routing_model=None,
                    refined=refined,
                    selection_score_kind=self._selection_score_kind(),
                    split_selector=self._community_clustering_method(),
                )
            return _stopped_from_selection(level, input_ids, "bic_selected_one", projection, selection, selection_score_kind=self._selection_score_kind(), split_selector=self._community_clustering_method())

        fit = selection.chosen
        parts = _candidate_layer_parts(
            fit,
            level=level,
            input_ids=input_ids,
            items_by_id=items_by_id,
            token_counts=token_counts,
            soft_config=self.config.soft_membership,
            clustering_method=self._community_clustering_method(),
        )
        if parts is None:
            return _stopped_from_selection(level, input_ids, "membership_collapsed", projection, selection, selection_score_kind=self._selection_score_kind(), split_selector=self._community_clustering_method())

        communities = parts["communities"]
        member_ids_by_community = parts["member_ids_by_community"]
        active_child_ids = parts["active_child_ids"]
        active_components = parts["active_components"]

        routing_model = LayerRoutingModel(
            level=level,
            community_ids=[community.community_id for community in communities],
            pca_mean=projection.mean.astype(float).tolist(),
            pca_components=projection.components.astype(float).tolist(),
            pi=_renormalized_pi(fit.pi, active_components),
            means=fit.means[active_components].astype(float).tolist(),
            variances=fit.variances[active_components].astype(float).tolist(),
            covariance_type=self.config.gmm_bic.covariance_type,
            component_effective_counts=[float(fit.component_masses[index]) for index in active_components],
            total_effective_count=float(sum(fit.component_masses[index] for index in active_components)),
            soft_assignment=self.config.soft_membership.__dict__,
        )
        refined = await self._refine_overbudget_coarse_communities(
            communities,
            member_ids_by_community,
            level=level,
            input_ids=input_ids,
            items_by_id=items_by_id,
            token_counts=token_counts,
            token_budget=token_budget,
            prompt_token_estimator=prompt_token_estimator,
            prompt_token_budget=active_prompt_token_budget,
        )
        return LayerClusteringResult(
            level=level,
            input_item_ids=input_ids,
            communities=refined["communities"],
            member_item_ids_by_community=refined["member_item_ids_by_community"],
            stop_reason="" if refined["communities"] else "budget_refinement_no_active_communities",
            excluded_input_item_ids=refined["excluded_input_item_ids"],
            projection_dim=int(projection.dim),
            explained_variance_ratio=float(projection.explained_variance_ratio),
            pca_spectrum=list(projection.spectrum),
            chosen_k=int(fit.k),
            tested_k=[candidate.k for candidate in selection.candidates],
            bic_by_k={str(candidate.k): float(candidate.bic) for candidate in selection.candidates},
            log_likelihood_by_k={str(candidate.k): float(candidate.log_likelihood) for candidate in selection.candidates},
            bic_margin=float(selection.bic_margin),
            selection_score_kind=self._selection_score_kind(),
            split_selector=self._community_clustering_method(),
            routing_model=routing_model,
            summary_budget={
                "effective_token_budget": int(token_budget),
                "selection_score_kind": self._selection_score_kind(),
                "split_selector": self._community_clustering_method(),
                "kmeans_fixed_k_requested": int(self.config.kmeans.fixed_k) if self.config.tree_policy == "projected_kmeans_fixed" else None,
                "bic_selected_k": int(selection.chosen.k),
                "coarse_selected_k": int(fit.k),
                "budget_enforced": bool(refined["summary_budget"].get("refinement_routing_tree") or refined["excluded_input_item_ids"]),
                **refined["summary_budget"],
            },
        )

    def _builder_name(self) -> str:
        mapping = {
            "projected_gmm_bic": "projected_weighted_gmm_bic",
            "projected_kmeans_elbow": "projected_weighted_kmeans_elbow",
            "projected_kmeans_fixed": "projected_weighted_kmeans_fixed",
        }
        return mapping.get(self.config.tree_policy, self.config.tree_policy)

    def _community_clustering_method(self) -> str:
        mapping = {
            "projected_gmm_bic": "weighted_gmm_bic",
            "projected_kmeans_elbow": "weighted_kmeans_elbow",
            "projected_kmeans_fixed": "weighted_kmeans_fixed",
        }
        return mapping.get(self.config.tree_policy, self.config.tree_policy)

    def _selection_score_kind(self) -> str:
        if self.config.tree_policy in {"projected_kmeans_elbow", "projected_kmeans_fixed"}:
            return "weighted_inertia"
        return "bic"

    async def _select_projected_split(
        self,
        projected: np.ndarray,
        *,
        level: int,
        n_items: int,
        random_seed: int,
        sample_weights: np.ndarray,
        kmax_effective_n: float,
    ) -> GmmBicSelection:
        if self.config.tree_policy == "projected_gmm_bic":
            return await select_gmm_bic_async(
                projected,
                config=self.config.gmm_bic,
                soft_config=self.config.soft_membership,
                random_seed=random_seed,
                sample_weights=sample_weights,
                kmax_effective_n=kmax_effective_n,
                max_concurrent_candidates=self.config.gmm_bic.max_concurrent_candidates,
                max_concurrent_restarts=self.config.gmm_bic.max_concurrent_restarts,
            )
        if self.config.tree_policy in {"projected_kmeans_elbow", "projected_kmeans_fixed"}:
            mode = "fixed" if self.config.tree_policy == "projected_kmeans_fixed" else "elbow"
            return select_kmeans_split(
                projected,
                gmm_config=self.config.gmm_bic,
                kmeans_config=self.config.kmeans,
                mode=mode,
                random_seed=random_seed,
                sample_weights=sample_weights,
                kmax_effective_n=kmax_effective_n,
            )
        raise ValueError(f"unsupported projected tree_policy={self.config.tree_policy!r} at level {level} with n_items={n_items}")

    async def _estimate_community_token_cost(
        self,
        community: ExperienceCommunity,
        member_ids: Sequence[str],
        *,
        items_by_id: dict[str, ExperienceItem],
        token_counts: dict[str, int],
        prompt_token_estimator: PromptTokenEstimator | None,
        prompt_token_budget: int | None = None,
    ) -> int:
        metadata_cost = _selected_prompt_token_cost(token_counts, member_ids)
        if prompt_token_estimator is None:
            return metadata_cost
        active_prompt_token_budget = int(prompt_token_budget or self.config.summary_budget.analyst_prompt_token_budget)
        if metadata_cost > active_prompt_token_budget:
            # The metadata sum is already over the full analyst prompt budget,
            # so this community must be split or excluded.  Avoid constructing
            # enormous prompts just to learn that they are still over budget.
            return metadata_cost
        members = [items_by_id[item_id] for item_id in member_ids]
        value = prompt_token_estimator(community, members)
        if inspect.isawaitable(value):
            value = await value
        return max(1, int(value))

    def _refinement_node_community(
        self,
        parent: ExperienceCommunity,
        community_id: str,
        member_ids: Sequence[str],
        member_weights: dict[str, float],
        *,
        items_by_id: dict[str, ExperienceItem],
        metadata: dict[str, Any] | None = None,
    ) -> ExperienceCommunity:
        active_weights = {item_id: float(member_weights.get(item_id, 1.0)) for item_id in member_ids}
        success_count, failure_count, outcome_mode = _outcome_counts([items_by_id[item_id] for item_id in member_ids])
        return ExperienceCommunity(
            community_id=community_id,
            level=parent.level,
            member_weights=active_weights,
            posterior_member_weights=dict(active_weights),
            clustering_method="budget_refinement_prompt_probe",
            support_mass=_support_mass(items_by_id, active_weights),
            outcome_mode=outcome_mode,
            success_count=success_count,
            failure_count=failure_count,
            metadata={
                **dict(parent.metadata or {}),
                **dict(metadata or {}),
            },
        )

    async def _refine_overbudget_coarse_communities(
        self,
        communities: list[ExperienceCommunity],
        member_ids_by_community: dict[str, list[str]],
        *,
        level: int,
        input_ids: list[str],
        items_by_id: dict[str, ExperienceItem],
        token_counts: dict[str, int],
        token_budget: int,
        prompt_token_estimator: PromptTokenEstimator | None,
        prompt_token_budget: int | None = None,
    ) -> dict[str, Any]:
        if not (self.config.budget_refinement.enabled and level == self.config.budget_refinement.apply_to_level):
            return {
                "communities": communities,
                "member_item_ids_by_community": member_ids_by_community,
                "excluded_input_item_ids": [],
                "summary_budget": {
                    "budget_refinement_mode": "disabled",
                    "excluded_oversize_singletons": [],
                    "refinement_routing_tree": None,
                },
            }

        final_communities: list[ExperienceCommunity] = []
        final_members: dict[str, list[str]] = {}
        excluded: dict[str, dict[str, Any]] = {}
        refined_roots: dict[str, Any] = {}
        refined_nodes: dict[str, Any] = {}
        split_events: list[dict[str, Any]] = []
        budget_summary = {
            "token_budget_kind": "analyst_prompt" if prompt_token_estimator is not None else "member_evidence",
            "member_evidence_token_budget": int(self.config.summary_budget.effective_token_budget),
            "analyst_prompt_token_budget": int(prompt_token_budget or self.config.summary_budget.analyst_prompt_token_budget),
        }

        for community in communities:
            member_ids = list(member_ids_by_community.get(community.community_id, []))
            prompt_tokens = await self._estimate_community_token_cost(
                community,
                member_ids,
                items_by_id=items_by_id,
                token_counts=token_counts,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )
            if prompt_tokens <= token_budget:
                final_communities.append(community)
                final_members[community.community_id] = member_ids
                continue

            result = await self._refine_one_coarse_community(
                community,
                member_ids,
                level=level,
                items_by_id=items_by_id,
                token_counts=token_counts,
                token_budget=token_budget,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )
            final_communities.extend(result["communities"])
            final_members.update(result["member_item_ids_by_community"])
            for item_id, payload in result["excluded"].items():
                excluded[item_id] = payload
            if result["root_node_id"]:
                refined_roots[community.community_id] = result["root_node_id"]
            refined_nodes.update(result["routing_nodes"])
            split_events.extend(result["split_events"])

        excluded_ids = set(excluded)
        if excluded_ids:
            cleaned_communities: list[ExperienceCommunity] = []
            cleaned_members: dict[str, list[str]] = {}
            for community in final_communities:
                member_weights = {iid: weight for iid, weight in community.member_weights.items() if iid not in excluded_ids}
                posterior_weights = {iid: weight for iid, weight in community.posterior_member_weights.items() if iid not in excluded_ids}
                if not member_weights:
                    continue
                member_ids = [iid for iid in final_members.get(community.community_id, []) if iid in member_weights]
                cleaned_communities.append(replace(
                    community,
                    member_weights=member_weights,
                    posterior_member_weights=posterior_weights or dict(member_weights),
                    support_mass=_support_mass(items_by_id, member_weights),
                ))
                cleaned_members[community.community_id] = member_ids
            final_communities = cleaned_communities
            final_members = cleaned_members

        if not final_communities:
            return {
                "communities": [],
                "member_item_ids_by_community": {},
                "excluded_input_item_ids": sorted(excluded),
                "summary_budget": {
                    "budget_refinement_mode": "coarse_then_refine",
                    "effective_token_budget": int(token_budget),
                    **budget_summary,
                    "excluded_oversize_singletons": [excluded[item_id] for item_id in sorted(excluded)],
                    "refinement_routing_tree": {
                        "kind": "coarse_refinement_routing_tree",
                        "coarse_roots": refined_roots,
                        "nodes": refined_nodes,
                    } if refined_nodes else None,
                    "split_events": split_events,
                },
            }

        return {
            "communities": final_communities,
            "member_item_ids_by_community": final_members,
            "excluded_input_item_ids": sorted(excluded),
            "summary_budget": {
                "budget_refinement_mode": "coarse_then_refine",
                "effective_token_budget": int(token_budget),
                **budget_summary,
                "excluded_oversize_singletons": [excluded[item_id] for item_id in sorted(excluded)],
                "refinement_routing_tree": {
                    "kind": "coarse_refinement_routing_tree",
                    "coarse_roots": refined_roots,
                    "nodes": refined_nodes,
                } if refined_nodes else None,
                "split_events": split_events,
            },
        }

    async def _refine_one_coarse_community(
        self,
        community: ExperienceCommunity,
        member_ids: list[str],
        *,
        level: int,
        items_by_id: dict[str, ExperienceItem],
        token_counts: dict[str, int],
        token_budget: int,
        prompt_token_estimator: PromptTokenEstimator | None,
        prompt_token_budget: int | None = None,
    ) -> dict[str, Any]:
        root_node_id = f"{community.community_id}_R0"
        queue: list[dict[str, Any]] = [{
            "node_id": root_node_id,
            "item_ids": member_ids,
            "path_weights": {iid: float(community.member_weights.get(iid, 1.0)) for iid in member_ids},
            "depth": 0,
        }]
        final_specs: list[dict[str, Any]] = []
        excluded: dict[str, dict[str, Any]] = {}
        routing_nodes: dict[str, Any] = {}
        split_events: list[dict[str, Any]] = []
        node_serial = 1

        while queue:
            node = queue.pop(0)
            node_item_ids = list(node["item_ids"])
            node_token_cost = await self._estimate_community_token_cost(
                self._refinement_node_community(
                    community,
                    node["node_id"],
                    node_item_ids,
                    node["path_weights"],
                    items_by_id=items_by_id,
                    metadata={"budget_refinement_node": True},
                ),
                node_item_ids,
                items_by_id=items_by_id,
                token_counts=token_counts,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )
            if node_token_cost <= token_budget:
                final_specs.append({
                    "source_node_id": node["node_id"],
                    "item_ids": node_item_ids,
                    "member_weights": dict(node["path_weights"]),
                    "token_cost": int(node_token_cost),
                    "refinement_depth": int(node["depth"]),
                })
                continue

            if len(node_item_ids) <= 1:
                item_id = node_item_ids[0] if node_item_ids else ""
                if item_id:
                    excluded[item_id] = {
                        "item_id": item_id,
                        "source_community_id": community.community_id,
                        "node_id": node["node_id"],
                        "token_cost": int(node_token_cost),
                        "budget": int(token_budget),
                        "reason": "oversize_singleton",
                    }
                routing_nodes[node["node_id"]] = {
                    "node_id": node["node_id"],
                    "kind": "excluded_oversize_singleton",
                    "item_ids": node_item_ids,
                    "excluded_item_id": item_id,
                    "token_cost": int(node_token_cost),
                    "budget": int(token_budget),
                }
                continue

            if len(node_item_ids) < self.config.gmm_bic.min_split_size:
                split = {
                    "accepted": False,
                    "reason": "too_few_items_for_gmm_refinement",
                    "node_id": node["node_id"],
                    "item_count": len(node_item_ids),
                    "min_split_size": int(self.config.gmm_bic.min_split_size),
                }
            else:
                local_items = [items_by_id[item_id] for item_id in node_item_ids]
                split = await self._budget_refinement_gmm_split(
                    local_items,
                    node=node,
                    level=level,
                    token_counts=token_counts,
                    parent_token_cost=node_token_cost,
                    token_budget=token_budget,
                    serial_start=node_serial,
                )
            if not split.get("accepted"):
                fallback = await self._fallback_overbudget_node(
                    community,
                    node,
                    items_by_id=items_by_id,
                    token_counts=token_counts,
                    token_budget=token_budget,
                    split_rejection=split,
                    serial_start=node_serial,
                    prompt_token_estimator=prompt_token_estimator,
                    prompt_token_budget=prompt_token_budget,
                )
                final_specs.extend(fallback["final_specs"])
                excluded.update(fallback["excluded"])
                routing_nodes.update(fallback["routing_nodes"])
                split_events.append(fallback["event"])
                node_serial += int(fallback["node_count"])
                continue

            children = list(split["children"])
            routing_node = dict(split["routing_node"])
            routing_node["source_community_id"] = community.community_id
            routing_nodes[node["node_id"]] = routing_node
            split_events.append({key: value for key, value in split.items() if key not in {"children", "routing_node"}})
            node_serial += len(children)
            queue.extend(children)

        final_communities: list[ExperienceCommunity] = []
        final_members: dict[str, list[str]] = {}
        final_index = 0

        async def append_final_spec(spec: dict[str, Any]) -> None:
            nonlocal final_index
            member_ids_for_leaf = [item_id for item_id in member_ids if item_id in set(spec["item_ids"])]
            member_weights = {item_id: float(spec["member_weights"].get(item_id, 1.0)) for item_id in member_ids_for_leaf if item_id not in excluded}
            if not member_weights:
                return
            community_id = f"{community.community_id}_R{final_index:03d}"
            success_count, failure_count, outcome_mode = _outcome_counts([items_by_id[item_id] for item_id in member_weights])
            routing_kind = str(spec.get("routing_kind", "leaf"))
            fallback_metadata = {
                key: value
                for key, value in {
                    "fallback_kind": spec.get("routing_kind"),
                    "fallback_parent_node_id": spec.get("fallback_parent_node_id"),
                    "fallback_reason": spec.get("fallback_reason"),
                }.items()
                if value is not None
            }
            candidate = ExperienceCommunity(
                community_id=community_id,
                level=level,
                member_weights=member_weights,
                posterior_member_weights=dict(member_weights),
                clustering_method=str(spec.get("clustering_method", "budget_refined_weighted_gmm_bic_leaf")),
                support_mass=_support_mass(items_by_id, member_weights),
                outcome_mode=outcome_mode,
                success_count=success_count,
                failure_count=failure_count,
                metadata={
                    **dict(community.metadata or {}),
                    "refined_from_community_id": community.community_id,
                    "source_refinement_node_id": spec["source_node_id"],
                    "refinement_depth": int(spec["refinement_depth"]),
                    "prompt_token_cost": int(spec["token_cost"]),
                    "budget": int(token_budget),
                    "split_reason": str(spec.get("split_reason", "budget_refinement_leaf")),
                    **fallback_metadata,
                },
            )
            exact_token_cost = await self._estimate_community_token_cost(
                candidate,
                list(member_weights),
                items_by_id=items_by_id,
                token_counts=token_counts,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )
            if exact_token_cost > token_budget:
                if len(member_weights) <= 1:
                    item_id = next(iter(member_weights))
                    excluded[item_id] = {
                        "item_id": item_id,
                        "source_community_id": community.community_id,
                        "node_id": spec["source_node_id"],
                        "token_cost": int(exact_token_cost),
                        "budget": int(token_budget),
                        "reason": "oversize_singleton",
                        "fallback_reason": "final_prompt_token_cost_exceeds_budget",
                    }
                    routing_nodes[spec["source_node_id"]] = {
                        "node_id": spec["source_node_id"],
                        "kind": "excluded_oversize_singleton",
                        "item_ids": [item_id],
                        "excluded_item_id": item_id,
                        "token_cost": int(exact_token_cost),
                        "budget": int(token_budget),
                        "fallback_reason": "final_prompt_token_cost_exceeds_budget",
                    }
                    return
                child_node_ids: list[str] = []
                for child_index, item_id in enumerate(member_ids_for_leaf):
                    child_node_id = f"{spec['source_node_id']}_V{child_index}"
                    child_node_ids.append(child_node_id)
                    await append_final_spec({
                        **spec,
                        "source_node_id": child_node_id,
                        "item_ids": [item_id],
                        "member_weights": {item_id: float(member_weights[item_id])},
                        "token_cost": int(_selected_prompt_token_cost(token_counts, [item_id])),
                        "refinement_depth": int(spec["refinement_depth"]) + 1,
                        "routing_kind": "singleton_leaf",
                        "clustering_method": "budget_fallback_singleton_leaf",
                        "split_reason": "final_prompt_token_singleton_leaf",
                        "fallback_parent_node_id": spec["source_node_id"],
                        "fallback_reason": "final_prompt_token_cost_exceeds_budget",
                    })
                routing_nodes[spec["source_node_id"]] = {
                    "node_id": spec["source_node_id"],
                    "kind": "fallback_token_router",
                    "routing_model_kind": "fallback_centroid_softmax_v1",
                    "routing_temperature": 8.0,
                    "soft_assignment": self.config.soft_membership.__dict__,
                    "source_community_id": community.community_id,
                    "item_ids": member_ids_for_leaf,
                    "child_node_ids": child_node_ids,
                    "token_cost": int(exact_token_cost),
                    "budget": int(token_budget),
                    "fallback_reason": "final_prompt_token_cost_exceeds_budget",
                }
                split_events.append({
                    "split_reason": "final_prompt_token_cost_exceeds_budget",
                    "node_id": spec["source_node_id"],
                    "parent_prompt_tokens": int(exact_token_cost),
                    "budget": int(token_budget),
                    "child_node_ids": child_node_ids,
                })
                return

            routing_payload = {
                "node_id": spec["source_node_id"],
                "kind": routing_kind,
                "community_id": community_id,
                "source_community_id": community.community_id,
                "token_cost": int(exact_token_cost),
                "refinement_depth": int(spec["refinement_depth"]),
            }
            if routing_kind in {"token_packing_leaf", "singleton_leaf"}:
                routing_payload.update({
                    "item_ids": list(member_weights),
                    "fallback_parent_node_id": spec.get("fallback_parent_node_id"),
                    "fallback_reason": spec.get("fallback_reason"),
                    "centroid_embedding": _centroid_embedding(items_by_id, list(member_weights)),
                    "token_budget": int(token_budget),
                })
            routing_nodes[spec["source_node_id"]] = routing_payload
            final_communities.append(candidate)
            final_members[community_id] = list(member_weights)
            final_index += 1

        for spec in final_specs:
            await append_final_spec(spec)

        return {
            "root_node_id": root_node_id if routing_nodes else "",
            "communities": final_communities,
            "member_item_ids_by_community": final_members,
            "excluded": excluded,
            "routing_nodes": routing_nodes,
            "split_events": split_events,
        }

    async def _fallback_overbudget_node(
        self,
        community: ExperienceCommunity,
        node: dict[str, Any],
        *,
        items_by_id: dict[str, ExperienceItem],
        token_counts: dict[str, int],
        token_budget: int,
        split_rejection: dict[str, Any],
        serial_start: int,
        prompt_token_estimator: PromptTokenEstimator | None,
        prompt_token_budget: int | None = None,
    ) -> dict[str, Any]:
        node_item_ids = list(node["item_ids"])
        path_weights = dict(node["path_weights"])
        packs: list[list[str]] = []
        pack_costs: list[int] = []
        excluded: dict[str, dict[str, Any]] = {}
        routing_nodes: dict[str, Any] = {}
        child_node_ids: list[str] = []
        original_order = {item_id: index for index, item_id in enumerate(node_item_ids)}

        def add_child_node_id() -> str:
            child_id = f"{node['node_id']}_F{serial_start + len(child_node_ids)}"
            child_node_ids.append(child_id)
            return child_id

        async def estimate_pack_cost(probe_id: str, pack_ids: list[str]) -> int:
            probe = self._refinement_node_community(
                community,
                probe_id,
                pack_ids,
                path_weights,
                items_by_id=items_by_id,
                metadata={
                    "budget_fallback_probe": True,
                    "fallback_parent_node_id": node["node_id"],
                    "fallback_reason": split_rejection.get("reason"),
                },
            )
            return await self._estimate_community_token_cost(
                probe,
                pack_ids,
                items_by_id=items_by_id,
                token_counts=token_counts,
                prompt_token_estimator=prompt_token_estimator,
                prompt_token_budget=prompt_token_budget,
            )

        parent_prompt_tokens = await estimate_pack_cost(node["node_id"], node_item_ids)
        sortable_ids = sorted(node_item_ids, key=lambda item_id: (-int(token_counts.get(item_id, 0)), original_order[item_id]))
        for item_id in sortable_ids:
            count = await estimate_pack_cost(f"{node['node_id']}_singleton_probe_{item_id}", [item_id])
            if count > token_budget:
                child_id = add_child_node_id()
                excluded[item_id] = {
                    "item_id": item_id,
                    "source_community_id": community.community_id,
                    "node_id": child_id,
                    "source_node_id": node["node_id"],
                    "token_cost": count,
                    "budget": int(token_budget),
                    "reason": "oversize_singleton",
                    "fallback_reason": split_rejection.get("reason"),
                }
                routing_nodes[child_id] = {
                    "node_id": child_id,
                    "kind": "excluded_oversize_singleton",
                    "item_ids": [item_id],
                    "excluded_item_id": item_id,
                    "token_cost": count,
                    "budget": int(token_budget),
                    "fallback_parent_node_id": node["node_id"],
                    "fallback_reason": split_rejection.get("reason"),
                }
                continue
            placed = False
            for index, cost in enumerate(pack_costs):
                candidate_pack = packs[index] + [item_id]
                candidate_cost = await estimate_pack_cost(f"{node['node_id']}_pack_probe_{index}", candidate_pack)
                if cost <= token_budget and candidate_cost <= token_budget:
                    packs[index].append(item_id)
                    pack_costs[index] = candidate_cost
                    placed = True
                    break
            if not placed:
                packs.append([item_id])
                pack_costs.append(count)

        final_specs: list[dict[str, Any]] = []
        for pack, cost in zip(packs, pack_costs):
            child_id = add_child_node_id()
            ordered_pack = [item_id for item_id in node_item_ids if item_id in set(pack)]
            routing_kind = "singleton_leaf" if len(ordered_pack) == 1 else "token_packing_leaf"
            final_specs.append({
                "source_node_id": child_id,
                "item_ids": ordered_pack,
                "member_weights": {item_id: float(path_weights.get(item_id, 1.0)) for item_id in ordered_pack},
                "token_cost": int(cost),
                "refinement_depth": int(node["depth"]) + 1,
                "routing_kind": routing_kind,
                "clustering_method": f"budget_fallback_{routing_kind}",
                "split_reason": routing_kind,
                "fallback_parent_node_id": node["node_id"],
                "fallback_reason": split_rejection.get("reason"),
            })

        routing_nodes[node["node_id"]] = {
            "node_id": node["node_id"],
            "kind": "fallback_token_router",
            "routing_model_kind": "fallback_centroid_softmax_v1",
            "routing_temperature": 8.0,
            "soft_assignment": self.config.soft_membership.__dict__,
            "source_community_id": community.community_id,
            "item_ids": node_item_ids,
            "child_node_ids": child_node_ids,
            "token_cost": int(parent_prompt_tokens),
            "budget": int(token_budget),
            "fallback_reason": split_rejection.get("reason"),
            "last_rejection": split_rejection,
        }
        return {
            "final_specs": final_specs,
            "excluded": excluded,
            "routing_nodes": routing_nodes,
            "node_count": len(child_node_ids),
            "event": {
                "split_reason": "budget_fallback_token_pack_singleton",
                "node_id": node["node_id"],
                "parent_prompt_tokens": int(parent_prompt_tokens),
                "budget": int(token_budget),
                "fallback_reason": split_rejection.get("reason"),
                "pack_count": len(final_specs),
                "excluded_count": len(excluded),
                "pack_prompt_tokens": [int(cost) for cost in pack_costs],
                "excluded_item_ids": sorted(excluded),
            },
        }

    async def _budget_refinement_gmm_split(
        self,
        local_items: list[ExperienceItem],
        *,
        node: dict[str, Any],
        level: int,
        token_counts: dict[str, int],
        parent_token_cost: int,
        token_budget: int,
        serial_start: int,
    ) -> dict[str, Any]:
        if len(local_items) < 2:
            return {"accepted": False, "reason": "too_few_items_for_gmm_refinement"}
        local_ids = [item.item_id for item in local_items]
        embeddings = normalize_rows(np.asarray([item.embedding for item in local_items], dtype=float))
        projection = local_pca_project(embeddings, self.config.projection)
        path_weights = dict(node["path_weights"])
        weights = _normalized_gmm_sample_weights(local_items, path_weights=path_weights)
        selection = await self._select_projected_split(
            projection.projected,
            level=level,
            n_items=len(local_items),
            random_seed=int(self.config.random_seed) + level * 100003 + len(local_items) + int(node["depth"]) * 7919,
            sample_weights=weights,
            kmax_effective_n=float(len(local_items)),
        )

        candidates: list[dict[str, Any]] = []
        for fit in selection.candidates:
            if not fit.valid or fit.k <= 1:
                continue
            child_ids = [f"{node['node_id']}_K{fit.k}_C{component}" for component in range(fit.k)]
            child_weights = membership_weight_dicts(local_ids, child_ids, fit.responsibilities, self.config.soft_membership)
            selected_by_child: dict[str, dict[str, float]] = {child_id: {} for child_id in child_ids}
            for item_id, local_memberships in child_weights.items():
                for child_id, local_weight in local_memberships.items():
                    if local_weight > 0.0:
                        selected_by_child[child_id][item_id] = float(path_weights.get(item_id, 1.0)) * float(local_weight)
            selected_by_child = {cid: weights for cid, weights in selected_by_child.items() if weights}
            if len(selected_by_child) <= 1:
                continue
            child_token_costs = {cid: _selected_prompt_token_cost(token_counts, list(weights)) for cid, weights in selected_by_child.items()}
            max_child_token = max(child_token_costs.values())
            candidates.append(
                {
                    "fit": fit,
                    "child_member_weights": selected_by_child,
                    "child_token_costs": child_token_costs,
                    "max_child_token_cost": int(max_child_token),
                    "progress_fraction": 1.0 - float(max_child_token) / max(float(parent_token_cost), 1.0),
                }
            )

        tested = [int(candidate.k) for candidate in selection.candidates]
        bic_by_k = {f"{node['node_id']}:k={candidate.k}": float(candidate.bic) for candidate in selection.candidates}
        ll_by_k = {f"{node['node_id']}:k={candidate.k}": float(candidate.log_likelihood) for candidate in selection.candidates}
        split_selector = self._community_clustering_method()
        if not candidates:
            return {
                "accepted": False,
                "reason": f"no_nontrivial_{split_selector}_candidate",
                "split_selector": split_selector,
                "node_id": node["node_id"],
                "tested_k": tested,
                "bic_by_k": bic_by_k,
                "log_likelihood_by_k": ll_by_k,
            }

        min_reduction = float(self.config.budget_refinement.min_token_reduction_fraction)
        threshold = float(parent_token_cost) * (1.0 - min_reduction)
        progressive = [candidate for candidate in candidates if candidate["max_child_token_cost"] <= threshold]
        if not progressive:
            progressive = [candidate for candidate in candidates if candidate["max_child_token_cost"] < parent_token_cost]
        if not progressive:
            best = min(candidates, key=lambda cand: (cand["max_child_token_cost"], cand["fit"].bic, cand["fit"].k))
            return {
                "accepted": False,
                "reason": f"{split_selector}_candidates_do_not_reduce_prompt_tokens",
                "split_selector": split_selector,
                "node_id": node["node_id"],
                "parent_prompt_tokens": int(parent_token_cost),
                "best_max_child_prompt_tokens": int(best["max_child_token_cost"]),
                "tested_k": tested,
                "bic_by_k": bic_by_k,
                "log_likelihood_by_k": ll_by_k,
            }

        selected = min(progressive, key=lambda cand: (cand["fit"].bic, cand["fit"].k))
        fit = selected["fit"]
        children: list[dict[str, Any]] = []
        child_node_ids: list[str] = []
        active_components: list[int] = []
        for child_index, (child_id, member_weights) in enumerate(selected["child_member_weights"].items()):
            member_ids = [item_id for item_id in local_ids if item_id in member_weights]
            child_node_id = f"{node['node_id']}_R{serial_start + child_index}"
            component = int(child_id.rsplit("C", 1)[1])
            child_node_ids.append(child_node_id)
            active_components.append(component)
            children.append(
                {
                    "node_id": child_node_id,
                    "source_gmm_child_id": child_id,
                    "item_ids": member_ids,
                    "path_weights": dict(member_weights),
                    "depth": int(node["depth"]) + 1,
                }
            )
        safe_active = active_components or list(range(int(fit.k)))
        return {
            "accepted": True,
            "split_reason": f"budget_forced_{split_selector}_progress",
            "split_selector": split_selector,
            "statistical_split": True,
            "node_id": node["node_id"],
            "parent_prompt_tokens": int(parent_token_cost),
            "budget": int(token_budget),
            "bic_selected_k": int(selection.chosen.k),
            "selected_k": int(fit.k),
            "selected_bic": float(fit.bic),
            "selected_max_child_prompt_tokens": int(selected["max_child_token_cost"]),
            "selected_progress_fraction": float(selected["progress_fraction"]),
            "child_prompt_tokens": {cid: int(value) for cid, value in selected["child_token_costs"].items()},
            "tested_k": tested,
            "bic_by_k": bic_by_k,
            "log_likelihood_by_k": ll_by_k,
            "routing_node": {
                "node_id": node["node_id"],
                "kind": "gmm_split",
                "split_selector": split_selector,
                "level": int(level),
                "pca_mean": projection.mean.astype(float).tolist(),
                "pca_components": projection.components.astype(float).tolist(),
                "pi": _renormalized_pi(fit.pi, safe_active),
                "means": fit.means[safe_active].astype(float).tolist(),
                "variances": fit.variances[safe_active].astype(float).tolist(),
                "covariance_type": self.config.gmm_bic.covariance_type,
                "child_node_ids": child_node_ids,
                "component_indices": [int(index) for index in safe_active],
                "soft_assignment": self.config.soft_membership.__dict__,
                "selected_k": int(fit.k),
                "selected_bic": float(fit.bic),
            },
            "children": children,
        }

    async def _summarize_communities(self, clustering: LayerClusteringResult, *, items_by_id: dict[str, ExperienceItem], summary_fn: SummaryFn) -> list[ExperienceItem]:
        async def run_one(index: int, community: ExperienceCommunity) -> tuple[int, list[ExperienceItem]]:
            member_ids = clustering.member_item_ids_by_community[community.community_id]
            members = [items_by_id[item_id] for item_id in member_ids]
            if _community_skips_llm_summary(community):
                return index, []
            produced = summary_fn(community, members, clustering)
            if inspect.isawaitable(produced):
                produced = await produced
            cards = list(produced)
            _validate_generated_cards(community, cards)
            return index, cards

        results = await asyncio.gather(*(run_one(index, community) for index, community in enumerate(clustering.communities)))
        generated: list[ExperienceItem] = []
        for _, cards in sorted(results, key=lambda pair: pair[0]):
            generated.extend(cards)
        return generated


ExperienceHierarchyTreeBuilder = ProjectedGmmTreeBuilder


def _candidate_layer_parts(
    fit: GmmCandidateFit,
    *,
    level: int,
    input_ids: list[str],
    items_by_id: dict[str, ExperienceItem],
    token_counts: dict[str, int],
    soft_config: SoftMembershipConfig,
    clustering_method: str = "weighted_gmm_bic",
) -> dict[str, Any] | None:
    child_ids = [f"L{level}_C{component}" for component in range(fit.k)]
    selected_memberships = membership_weight_dicts(input_ids, child_ids, fit.responsibilities, soft_config)
    selected_memberships = {iid: {cid: w for cid, w in weights.items() if float(w) > 1.0e-12} for iid, weights in selected_memberships.items()}
    if not soft_config.save_soft_edges or soft_config.recursive_assignment == "primary_argmax":
        full_memberships = selected_memberships
    else:
        full_memberships = {
            item_id: {child_ids[col]: float(fit.responsibilities[row, col]) for col in range(len(child_ids))}
            for row, item_id in enumerate(input_ids)
        }

    selected_by_child: dict[str, dict[str, float]] = {cid: {} for cid in child_ids}
    posterior_by_child: dict[str, dict[str, float]] = {cid: {} for cid in child_ids}
    for item_id, memberships in selected_memberships.items():
        for child_id, weight in memberships.items():
            if weight > 0.0:
                selected_by_child[child_id][item_id] = float(weight)
    for item_id, memberships in full_memberships.items():
        for child_id, weight in memberships.items():
            if weight > 0.0:
                posterior_by_child[child_id][item_id] = float(weight)
    selected_by_child = {cid: weights for cid, weights in selected_by_child.items() if weights}
    if len(selected_by_child) <= 1:
        return None

    active_child_ids = [child_id for child_id in child_ids if child_id in selected_by_child]
    active_components = [int(child_id.rsplit("C", 1)[1]) for child_id in active_child_ids]
    communities: list[ExperienceCommunity] = []
    member_ids_by_community: dict[str, list[str]] = {}
    community_token_masses: dict[str, float] = {}
    for child_id in active_child_ids:
        member_weights = selected_by_child[child_id]
        component = int(child_id.rsplit("C", 1)[1])
        member_ids = [item_id for item_id in input_ids if item_id in member_weights]
        success_count, failure_count, outcome_mode = _outcome_counts([items_by_id[item_id] for item_id in member_ids])
        communities.append(
            ExperienceCommunity(
                community_id=child_id,
                level=level,
                member_weights=member_weights,
                posterior_member_weights=posterior_by_child.get(child_id, {}),
                clustering_method=clustering_method,
                support_mass=_support_mass(items_by_id, member_weights),
                outcome_mode=outcome_mode,
                success_count=success_count,
                failure_count=failure_count,
                metadata={
                    "component_index": component,
                    "component_effective_count": float(fit.component_masses[component]),
                    "primary_size": int(fit.child_sizes[component]),
                    "mixture_weight": float(fit.pi[component]),
                    "chosen_k": int(fit.k),
                    "bic": float(fit.bic),
                    "log_likelihood": float(fit.log_likelihood),
                    "selection_score": float(fit.bic),
                },
            )
        )
        member_ids_by_community[child_id] = member_ids
        community_token_masses[child_id] = _community_token_mass(token_counts, member_weights)
    return {
        "communities": communities,
        "member_ids_by_community": member_ids_by_community,
        "active_child_ids": active_child_ids,
        "active_components": active_components,
        "community_token_masses": community_token_masses,
    }


def _clustering_from_refinement(
    *,
    level: int,
    input_ids: list[str],
    projection: ProjectionResult,
    selection: GmmBicSelection,
    fit: GmmCandidateFit,
    routing_model: LayerRoutingModel | None,
    refined: dict[str, Any],
    selection_score_kind: str = "bic",
    split_selector: str = "weighted_gmm_bic",
) -> LayerClusteringResult:
    return LayerClusteringResult(
        level=level,
        input_item_ids=input_ids,
        communities=list(refined["communities"]),
        member_item_ids_by_community=dict(refined["member_item_ids_by_community"]),
        stop_reason="" if refined["communities"] else "budget_refinement_no_active_communities",
        excluded_input_item_ids=list(refined["excluded_input_item_ids"]),
        projection_dim=int(projection.dim),
        explained_variance_ratio=float(projection.explained_variance_ratio),
        pca_spectrum=list(projection.spectrum),
        chosen_k=int(fit.k),
        tested_k=[candidate.k for candidate in selection.candidates],
        bic_by_k={str(candidate.k): float(candidate.bic) for candidate in selection.candidates},
        log_likelihood_by_k={str(candidate.k): float(candidate.log_likelihood) for candidate in selection.candidates},
        bic_margin=float(selection.bic_margin),
        selection_score_kind=selection_score_kind,
        split_selector=split_selector,
        routing_model=routing_model,
        summary_budget={
            "selection_score_kind": selection_score_kind,
            "split_selector": split_selector,
            **dict(refined["summary_budget"]),
        },
    )


def _item_token_count(item: ExperienceItem, keys: Sequence[str]) -> int:
    for key in keys:
        value = item.metadata.get(key)
        if value is None:
            continue
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    # Fallback estimate: conservative enough to trigger splitting on very long
    # text, but deterministic and independent of tokenizer availability.
    text = item.text or ""
    return max(1, int((len(text) + 3) // 4))


def _selected_prompt_token_cost(token_counts: dict[str, int], item_ids: Sequence[str]) -> int:
    # Prompt cost is unweighted: if a trajectory is selected for a community,
    # the analyst sees it once regardless of posterior membership weight.
    return int(sum(int(token_counts.get(item_id, 0)) for item_id in set(item_ids)))


def _centroid_embedding(items_by_id: dict[str, ExperienceItem], item_ids: Sequence[str]) -> list[float]:
    vectors = [items_by_id[item_id].embedding for item_id in item_ids if item_id in items_by_id and items_by_id[item_id].embedding]
    if not vectors:
        return []
    return normalize_rows(np.asarray(vectors, dtype=float)).mean(axis=0).astype(float).tolist()


def _community_token_mass(token_counts: dict[str, int], member_weights: dict[str, float]) -> float:
    return float(_selected_prompt_token_cost(token_counts, list(member_weights)))


def _normalized_gmm_sample_weights(items: Sequence[ExperienceItem], *, path_weights: dict[str, float] | None = None) -> np.ndarray:
    """Return temporary GMM weights normalized to the current item count.

    Persistent ``item.support_mass`` remains unchanged and is still used for
    community mass accounting.  These weights are only for EM/weighted BIC, so
    inherited support_mass cannot inflate the layer's effective sample count.
    """
    n_items = len(items)
    if n_items < 1:
        return np.ones(0, dtype=float)
    path_weights = dict(path_weights or {})
    masses = np.asarray([items_support_weight(item, path_weights) for item in items], dtype=float)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        return np.ones(n_items, dtype=float)
    total = float(masses.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.ones(n_items, dtype=float)
    return masses * (float(n_items) / total)


def _community_skips_llm_summary(community: ExperienceCommunity) -> bool:
    metadata = dict(community.metadata or {})
    return bool(metadata.get("llm_summary_skipped") or metadata.get("oversize_singleton"))


def items_support_weight(item: ExperienceItem, path_weights: dict[str, float]) -> float:
    return float(item.support_mass) * float(path_weights.get(item.item_id, 1.0))


def _validate_generated_cards(community: ExperienceCommunity, cards: list[ExperienceItem]) -> None:
    if not cards:
        if _community_skips_llm_summary(community):
            return
        raise ValueError(f"community {community.community_id!r} generated no experience cards")
    seen: set[str] = set()
    for card in cards:
        if card.item_id in seen:
            raise ValueError(f"duplicate generated item_id={card.item_id!r}")
        seen.add(card.item_id)
        if card.kind != ITEM_KIND_EXPERIENCE_CARD:
            raise ValueError("summary_fn must return experience_card items")
        if card.level != community.level + 1:
            raise ValueError("generated experience_card level must be community.level + 1")
        if list(card.generated_from_community_ids) != [community.community_id]:
            raise ValueError("generated experience_card must have generated_from_community_ids == [source_community_id]")
        if not card.embedding:
            raise ValueError("generated experience_card must include embedding")
        _require_confidence(card.metadata)


def _require_confidence(metadata: dict[str, Any]) -> float:
    value = metadata.get("confidence")
    if value is None:
        raise ValueError("generated experience_card metadata must include confidence")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("generated experience_card confidence must be positive and finite")
    return value


def _support_mass(items: dict[str, ExperienceItem], member_weights: dict[str, float]) -> float:
    return float(sum(items[item_id].support_mass * weight for item_id, weight in member_weights.items()))


def _outcome_counts(items: list[ExperienceItem]) -> tuple[int, int, str]:
    success = 0
    failure = 0
    for item in items:
        value = _infer_success(item)
        if value is True:
            success += 1
        elif value is False:
            failure += 1
    if success and failure:
        mode = "mixed"
    elif success:
        mode = "success"
    elif failure:
        mode = "failure"
    else:
        mode = "mixed"
    return success, failure, mode


def _infer_success(item: ExperienceItem) -> bool | None:
    metadata = item.metadata or {}
    for key in ("success", "succeeded", "is_success"):
        if key in metadata:
            return bool(metadata[key])
    for key in ("verifier_score", "score", "reward"):
        if key in metadata:
            try:
                return float(metadata[key]) > 0.0
            except (TypeError, ValueError):
                return None
    outcome = str(metadata.get("outcome", metadata.get("outcome_mode", ""))).lower()
    if outcome in {"success", "succeeded", "pass", "passed"}:
        return True
    if outcome in {"failure", "failed", "fail"}:
        return False
    return None


def _stopped(level: int, input_ids: list[str], reason: str) -> LayerClusteringResult:
    return LayerClusteringResult(level=level, input_item_ids=list(input_ids), communities=[], member_item_ids_by_community={}, stop_reason=reason)


def _stopped_from_selection(
    level: int,
    input_ids: list[str],
    reason: str,
    projection: ProjectionResult,
    selection: GmmBicSelection,
    *,
    selection_score_kind: str = "bic",
    split_selector: str = "weighted_gmm_bic",
) -> LayerClusteringResult:
    return LayerClusteringResult(
        level=level,
        input_item_ids=list(input_ids),
        communities=[],
        member_item_ids_by_community={},
        stop_reason=reason,
        projection_dim=int(projection.dim),
        explained_variance_ratio=float(projection.explained_variance_ratio),
        pca_spectrum=list(projection.spectrum),
        chosen_k=int(selection.chosen.k),
        tested_k=[candidate.k for candidate in selection.candidates],
        bic_by_k={str(candidate.k): float(candidate.bic) for candidate in selection.candidates},
        log_likelihood_by_k={str(candidate.k): float(candidate.log_likelihood) for candidate in selection.candidates},
        bic_margin=float(selection.bic_margin),
        selection_score_kind=selection_score_kind,
        split_selector=split_selector,
    )


def _renormalized_pi(pi: np.ndarray, indices: list[int]) -> list[float]:
    values = np.asarray(pi, dtype=float)[indices]
    total = float(values.sum())
    if total <= 1.0e-12:
        return (np.ones(len(indices), dtype=float) / max(1, len(indices))).tolist()
    return (values / total).astype(float).tolist()


__all__ = [
    "ExperienceHierarchyTreeBuilder",
    "HierarchyBuildResult",
    "LayerBuildResult",
    "LayerClusteringResult",
    "LayerRoutingModel",
    "ProjectedGmmTreeBuilder",
]

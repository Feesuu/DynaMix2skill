from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
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
from .projection import ProjectionResult, local_pca_project_async, normalize_rows

SummaryFn = Callable[
    [ExperienceCommunity, list[ExperienceItem], "LayerClusteringResult"],
    list[ExperienceItem] | Awaitable[list[ExperienceItem]],
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
    projection_dim: int = 0
    explained_variance_ratio: float = 0.0
    pca_spectrum: list[float] = field(default_factory=list)
    chosen_k: int = 1
    tested_k: list[int] = field(default_factory=list)
    bic_by_k: dict[str, float] = field(default_factory=dict)
    log_likelihood_by_k: dict[str, float] = field(default_factory=dict)
    bic_margin: float = 0.0
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
            result = await self.build_layer(state, level=level, items=layer_items, summary_fn=summary_fn)
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
    ) -> LayerBuildResult:
        layer_items = list(items) if items is not None else await state.item_objects_at_level(level)
        clustering = await self.cluster_layer(layer_items, level=level)
        if clustering.should_stop:
            return LayerBuildResult(clustering=clustering, generated_item_ids=[], committed=False)

        items_by_id = {item.item_id: item for item in layer_items}
        generated = await self._summarize_communities(clustering, items_by_id=items_by_id, summary_fn=summary_fn)
        if not generated:
            return LayerBuildResult(clustering=clustering, generated_item_ids=[], committed=False)

        metadata = {
            "builder": "projected_weighted_gmm_bic",
            "chosen_k": clustering.chosen_k,
            "bic_margin": clustering.bic_margin,
            "projection_dim": clustering.projection_dim,
            "routing_model": clustering.routing_model.to_dict() if clustering.routing_model else None,
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
        )
        return LayerBuildResult(
            clustering=clustering,
            generated_item_ids=[item.item_id for item in generated],
            committed=True,
        )

    async def cluster_layer(self, items: Sequence[ExperienceItem], *, level: int) -> LayerClusteringResult:
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
        token_budget = self.config.summary_budget.effective_token_budget
        total_prompt_tokens = sum(token_counts.values())

        if n_items < self.config.gmm_bic.min_split_size and not (
            self.config.budget_refinement.enabled
            and level == self.config.budget_refinement.apply_to_level
            and total_prompt_tokens > token_budget
        ):
            return _stopped(level, input_ids, "too_small")

        if (
            self.config.budget_refinement.enabled
            and level == self.config.budget_refinement.apply_to_level
            and total_prompt_tokens > token_budget
        ):
            return await self._cluster_budget_refined_flat_l0(
                ordered,
                level=level,
                items_by_id=items_by_id,
                token_counts=token_counts,
                token_budget=token_budget,
            )

        embeddings = normalize_rows(np.asarray([item.embedding for item in ordered], dtype=float))
        projection = await local_pca_project_async(embeddings, self.config.projection)
        weights = _normalized_gmm_sample_weights(ordered)
        selection = await select_gmm_bic_async(
            projection.projected,
            config=self.config.gmm_bic,
            soft_config=self.config.soft_membership,
            random_seed=int(self.config.random_seed) + level * 100003 + n_items,
            sample_weights=weights,
            kmax_effective_n=float(n_items),
            max_concurrent_candidates=self.config.gmm_bic.max_concurrent_candidates,
            max_concurrent_restarts=self.config.gmm_bic.max_concurrent_restarts,
        )
        if selection.chosen.k <= 1:
            return _stopped_from_selection(level, input_ids, "bic_selected_one", projection, selection)

        fit, parts, budget_info = _select_budget_compatible_fit(
            selection,
            level=level,
            input_ids=input_ids,
            items_by_id=items_by_id,
            token_counts=token_counts,
            soft_config=self.config.soft_membership,
            token_budget=token_budget,
        )
        if parts is None:
            return _stopped_from_selection(level, input_ids, "membership_collapsed", projection, selection)

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
        return LayerClusteringResult(
            level=level,
            input_item_ids=input_ids,
            communities=communities,
            member_item_ids_by_community=member_ids_by_community,
            stop_reason="",
            projection_dim=int(projection.dim),
            explained_variance_ratio=float(projection.explained_variance_ratio),
            pca_spectrum=list(projection.spectrum),
            chosen_k=int(fit.k),
            tested_k=[candidate.k for candidate in selection.candidates],
            bic_by_k={str(candidate.k): float(candidate.bic) for candidate in selection.candidates},
            log_likelihood_by_k={str(candidate.k): float(candidate.log_likelihood) for candidate in selection.candidates},
            bic_margin=float(selection.bic_margin),
            routing_model=routing_model,
            summary_budget=budget_info,
        )

    async def _cluster_budget_refined_flat_l0(
        self,
        ordered: list[ExperienceItem],
        *,
        level: int,
        items_by_id: dict[str, ExperienceItem],
        token_counts: dict[str, int],
        token_budget: int,
    ) -> LayerClusteringResult:
        """Refine over-budget raw communities into flat, analyst-feasible L0 leaves.

        Internal refinement nodes are not committed as semantic hierarchy nodes.
        They are only a budget-feasibility work queue.  The returned communities
        are all at ``level`` and are the only communities that will be passed to
        the cluster analyst.
        """
        input_ids = [item.item_id for item in ordered]
        queue: list[dict[str, Any]] = [
            {
                "node_id": f"L{level}_R0",
                "item_ids": input_ids,
                "path_weights": {item_id: 1.0 for item_id in input_ids},
                "depth": 0,
            }
        ]
        final_specs: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        skipped_oversize_item_ids: set[str] = set()
        split_events: list[dict[str, Any]] = []
        tested_k: set[int] = set()
        bic_by_k: dict[str, float] = {}
        log_likelihood_by_k: dict[str, float] = {}
        node_serial = 1

        while queue:
            node = queue.pop(0)
            node_item_ids = list(node["item_ids"])
            node_token_cost = _selected_prompt_token_cost(token_counts, node_item_ids)
            if node_token_cost <= token_budget:
                final_specs.append(
                    {
                        "source_node_id": node["node_id"],
                        "item_ids": node_item_ids,
                        "member_weights": dict(node["path_weights"]),
                        "token_cost": int(node_token_cost),
                        "refinement_depth": int(node["depth"]),
                    }
                )
                continue

            if len(node_item_ids) <= 1:
                llm_summary_skipped = bool(self.config.budget_refinement.skip_oversize_singleton)
                item_id = node_item_ids[0] if node_item_ids else ""
                if item_id and item_id in skipped_oversize_item_ids:
                    continue
                if item_id:
                    skipped_oversize_item_ids.add(item_id)
                skipped.append(
                    {
                        "node_id": node["node_id"],
                        "item_ids": node_item_ids,
                        "token_cost": int(node_token_cost),
                        "budget": int(token_budget),
                        "stop_reason": "oversize_singleton",
                        "llm_summary_skipped": llm_summary_skipped,
                    }
                )
                # Keep a diagnostic community so layer coverage remains explicit,
                # but do not generate a fallback ExperienceCard for it.
                final_specs.append(
                    {
                        "source_node_id": node["node_id"],
                        "item_ids": node_item_ids,
                        "member_weights": dict(node["path_weights"]),
                        "token_cost": int(node_token_cost),
                        "refinement_depth": int(node["depth"]),
                        "oversize_singleton": True,
                        "llm_summary_skipped": llm_summary_skipped,
                    }
                )
                continue

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
            tested_k.update(split.get("tested_k", []))
            bic_by_k.update(split.get("bic_by_k", {}))
            log_likelihood_by_k.update(split.get("log_likelihood_by_k", {}))

            if not split.get("accepted"):
                split = await self._deterministic_budget_split(
                    local_items,
                    node=node,
                    level=level,
                    token_counts=token_counts,
                    parent_token_cost=node_token_cost,
                    token_budget=token_budget,
                    serial_start=node_serial,
                )

            if not split.get("accepted"):
                skipped.append(
                    {
                        "node_id": node["node_id"],
                        "item_ids": node_item_ids,
                        "token_cost": int(node_token_cost),
                        "budget": int(token_budget),
                        "stop_reason": "oversize_unsplittable",
                        "last_rejection": split,
                    }
                )
                continue

            children = list(split["children"])
            split_events.append({key: value for key, value in split.items() if key != "children"})
            node_serial += len(children)
            queue.extend(children)

        if not final_specs:
            return LayerClusteringResult(
                level=level,
                input_item_ids=input_ids,
                communities=[],
                member_item_ids_by_community={},
                stop_reason="budget_refinement_no_feasible_communities",
                summary_budget={
                    "mode": "flat_l0_budget_refinement",
                    "effective_token_budget": int(token_budget),
                    "input_prompt_tokens": int(_selected_prompt_token_cost(token_counts, input_ids)),
                    "skipped": skipped,
                    "split_events": split_events,
                },
            )

        communities: list[ExperienceCommunity] = []
        member_ids_by_community: dict[str, list[str]] = {}
        for index, spec in enumerate(final_specs):
            community_id = f"L{level}_C{index:03d}"
            member_ids = [item_id for item_id in input_ids if item_id in set(spec["item_ids"])]
            member_weights = {item_id: float(spec["member_weights"].get(item_id, 1.0)) for item_id in member_ids}
            success_count, failure_count, outcome_mode = _outcome_counts([items_by_id[item_id] for item_id in member_ids])
            is_oversize_singleton = bool(spec.get("oversize_singleton"))
            llm_summary_skipped = bool(spec.get("llm_summary_skipped"))
            communities.append(
                ExperienceCommunity(
                    community_id=community_id,
                    level=level,
                    member_weights=member_weights,
                    posterior_member_weights=dict(member_weights),
                    clustering_method="budget_refined_weighted_gmm_bic",
                    support_mass=_support_mass(items_by_id, member_weights),
                    outcome_mode=outcome_mode,
                    success_count=success_count,
                    failure_count=failure_count,
                    metadata={
                        "source_refinement_node_id": spec["source_node_id"],
                        "refinement_depth": int(spec["refinement_depth"]),
                        "prompt_token_cost": int(spec["token_cost"]),
                        "budget": int(token_budget),
                        "split_reason": "budget_refinement_oversize_singleton" if is_oversize_singleton else "budget_refinement_leaf",
                        "oversize_singleton": is_oversize_singleton,
                        "llm_summary_skipped": llm_summary_skipped,
                    },
                )
            )
            member_ids_by_community[community_id] = member_ids

        return LayerClusteringResult(
            level=level,
            input_item_ids=input_ids,
            communities=communities,
            member_item_ids_by_community=member_ids_by_community,
            stop_reason="",
            chosen_k=len(communities),
            tested_k=sorted(tested_k),
            bic_by_k=bic_by_k,
            log_likelihood_by_k=log_likelihood_by_k,
            bic_margin=0.0,
            routing_model=None,
            summary_budget={
                "mode": "flat_l0_budget_refinement",
                "effective_token_budget": int(token_budget),
                "input_prompt_tokens": int(_selected_prompt_token_cost(token_counts, input_ids)),
                "final_community_count": len(communities),
                "skipped_count": len(skipped),
                "oversize_singleton_skipped_count": sum(1 for spec in final_specs if spec.get("llm_summary_skipped")),
                "oversize_singleton_fallback_count": 0,
                "skipped": skipped,
                "split_events": split_events,
                "soft_assignment": self.config.soft_membership.__dict__,
                "budget_refinement": self.config.budget_refinement.__dict__,
                "note": "Internal over-budget raw communities were refined and flattened back to this layer; oversized singleton leaves are retained as diagnostic communities only and do not generate ExperienceCards.",
            },
        )

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
        projection = await local_pca_project_async(embeddings, self.config.projection)
        path_weights = dict(node["path_weights"])
        weights = _normalized_gmm_sample_weights(local_items, path_weights=path_weights)
        selection = await select_gmm_bic_async(
            projection.projected,
            config=self.config.gmm_bic,
            soft_config=self.config.soft_membership,
            random_seed=int(self.config.random_seed) + level * 100003 + len(local_items) + int(node["depth"]) * 7919,
            sample_weights=weights,
            kmax_effective_n=float(len(local_items)),
            max_concurrent_candidates=self.config.gmm_bic.max_concurrent_candidates,
            max_concurrent_restarts=self.config.gmm_bic.max_concurrent_restarts,
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
        if not candidates:
            return {
                "accepted": False,
                "reason": "no_nontrivial_gmm_candidate",
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
                "reason": "gmm_candidates_do_not_reduce_prompt_tokens",
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
        for child_index, (child_id, member_weights) in enumerate(selected["child_member_weights"].items()):
            member_ids = [item_id for item_id in local_ids if item_id in member_weights]
            children.append(
                {
                    "node_id": f"L{level}_R{serial_start + child_index}",
                    "source_gmm_child_id": child_id,
                    "item_ids": member_ids,
                    "path_weights": dict(member_weights),
                    "depth": int(node["depth"]) + 1,
                }
            )
        return {
            "accepted": True,
            "split_reason": "budget_forced_gmm_bic_progress",
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
            "children": children,
        }

    async def _deterministic_budget_split(
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
            return {"accepted": False, "reason": "too_few_items_for_deterministic_split"}
        local_ids = [item.item_id for item in local_items]
        embeddings = normalize_rows(np.asarray([item.embedding for item in local_items], dtype=float))
        projection = await local_pca_project_async(embeddings, self.config.projection)
        if projection.projected.shape[1] >= 1:
            scores = projection.projected[:, 0]
        else:
            scores = np.zeros(len(local_items), dtype=float)
        ordered_ids = [item_id for _, item_id in sorted(zip(scores.tolist(), local_ids), key=lambda pair: (pair[0], pair[1]))]
        if len(ordered_ids) < 2:
            return {"accepted": False, "reason": "deterministic_order_too_small"}
        best_index = 1
        best_max = float("inf")
        total = _selected_prompt_token_cost(token_counts, ordered_ids)
        running = 0
        for index in range(1, len(ordered_ids)):
            running += int(token_counts.get(ordered_ids[index - 1], 1))
            left = running
            right = total - running
            largest = max(left, right)
            if largest < best_max:
                best_max = float(largest)
                best_index = index
        left_ids = ordered_ids[:best_index]
        right_ids = ordered_ids[best_index:]
        if not left_ids or not right_ids or max(_selected_prompt_token_cost(token_counts, left_ids), _selected_prompt_token_cost(token_counts, right_ids)) >= parent_token_cost:
            return {"accepted": False, "reason": "deterministic_split_no_token_progress"}
        path_weights = dict(node["path_weights"])
        children = [
            {
                "node_id": f"L{level}_R{serial_start}",
                "item_ids": left_ids,
                "path_weights": {item_id: float(path_weights.get(item_id, 1.0)) for item_id in left_ids},
                "depth": int(node["depth"]) + 1,
            },
            {
                "node_id": f"L{level}_R{serial_start + 1}",
                "item_ids": right_ids,
                "path_weights": {item_id: float(path_weights.get(item_id, 1.0)) for item_id in right_ids},
                "depth": int(node["depth"]) + 1,
            },
        ]
        return {
            "accepted": True,
            "split_reason": "budget_forced_pca_token_balanced_binary",
            "statistical_split": False,
            "node_id": node["node_id"],
            "parent_prompt_tokens": int(parent_token_cost),
            "budget": int(token_budget),
            "child_prompt_tokens": {
                children[0]["node_id"]: int(_selected_prompt_token_cost(token_counts, left_ids)),
                children[1]["node_id"]: int(_selected_prompt_token_cost(token_counts, right_ids)),
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


def _select_budget_compatible_fit(
    selection: GmmBicSelection,
    *,
    level: int,
    input_ids: list[str],
    items_by_id: dict[str, ExperienceItem],
    token_counts: dict[str, int],
    soft_config: SoftMembershipConfig,
    token_budget: int,
) -> tuple[GmmCandidateFit, dict[str, Any] | None, dict[str, Any]]:
    """Choose a candidate that is both BIC-valid and summary-budget valid.

    The summary budget is enforced before calling ``summary_fn``.  If the BIC
    optimum produces an over-budget community, we select the best higher-K
    valid GMM candidate whose selected communities all fit the budget.  This is
    a flat, routing-consistent split: the final committed communities still have
    a single saved PCA/GMM routing model, so dynamic fixed-K routing remains
    well-defined.  We never truncate member text to satisfy the budget.
    """
    token_budget = int(token_budget)
    if token_budget < 1:
        raise ValueError("summary token budget must be positive")

    evaluated: list[tuple[GmmCandidateFit, dict[str, Any], dict[str, Any]]] = []
    for candidate in selection.candidates:
        if not candidate.valid or candidate.k <= 1:
            continue
        parts = _candidate_layer_parts(
            candidate,
            level=level,
            input_ids=input_ids,
            items_by_id=items_by_id,
            token_counts=token_counts,
            soft_config=soft_config,
        )
        if parts is None:
            continue
        token_masses = parts["community_token_masses"]
        max_token_mass = max(token_masses.values()) if token_masses else 0.0
        info = {
            "effective_token_budget": token_budget,
            "bic_selected_k": int(selection.chosen.k),
            "candidate_k": int(candidate.k),
            "max_community_token_mass": float(max_token_mass),
            "community_token_masses": {cid: float(value) for cid, value in token_masses.items()},
            "over_budget": bool(max_token_mass > token_budget),
        }
        evaluated.append((candidate, parts, info))

    if not evaluated:
        return selection.chosen, None, {
            "effective_token_budget": token_budget,
            "bic_selected_k": int(selection.chosen.k),
            "budget_enforced": False,
            "reason": "no_valid_nontrivial_candidate",
        }

    # Keep the BIC winner if it already satisfies the summary budget.
    for candidate, parts, info in evaluated:
        if candidate.k == selection.chosen.k and not info["over_budget"]:
            info.update({"budget_enforced": True, "budget_override": False, "selected_k": int(candidate.k)})
            return candidate, parts, info

    # Otherwise split by choosing the best budget-valid candidate with K at
    # least as large as the BIC optimum.  This prevents summary-time truncation.
    valid_budget = [triple for triple in evaluated if not triple[2]["over_budget"] and triple[0].k >= selection.chosen.k]
    if not valid_budget:
        valid_budget = [triple for triple in evaluated if not triple[2]["over_budget"]]
    if valid_budget:
        candidate, parts, info = min(valid_budget, key=lambda triple: (triple[0].bic, triple[0].k))
        info.update({"budget_enforced": True, "budget_override": candidate.k != selection.chosen.k, "selected_k": int(candidate.k)})
        return candidate, parts, info

    worst = max((triple[2]["max_community_token_mass"] for triple in evaluated), default=0.0)
    raise ValueError(
        "no valid GMM candidate satisfies summary_budget.effective_token_budget; "
        f"budget={token_budget}, max_candidate_community_tokens={worst:.3f}. "
        "Increase gmm_bic.abs_kmax / lower min_child constraints, or reduce upstream item token sizes."
    )


def _candidate_layer_parts(
    fit: GmmCandidateFit,
    *,
    level: int,
    input_ids: list[str],
    items_by_id: dict[str, ExperienceItem],
    token_counts: dict[str, int],
    soft_config: SoftMembershipConfig,
) -> dict[str, Any] | None:
    child_ids = [f"L{level}_C{component}" for component in range(fit.k)]
    selected_memberships = membership_weight_dicts(input_ids, child_ids, fit.responsibilities, soft_config)
    selected_memberships = {iid: {cid: w for cid, w in weights.items() if float(w) > 1.0e-12} for iid, weights in selected_memberships.items()}
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
                clustering_method="weighted_gmm_bic",
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


def _stopped_from_selection(level: int, input_ids: list[str], reason: str, projection: ProjectionResult, selection: GmmBicSelection) -> LayerClusteringResult:
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

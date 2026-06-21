from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Iterable, Sequence

import numpy as np

from .config import ProjectedGmmDynamicTreeConfig, SoftMembershipConfig
from .data_structures import (
    DynamicPatchResult,
    ExperienceCardPatch,
    ExperienceCommunity,
    ExperienceHierarchyState,
    ExperienceItem,
    RerouteResult,
)
from .gmm_bic import gmm_responsibilities_from_model, membership_weight_dicts
from .projection import normalize_rows, project_with_basis

EmbedFn = Callable[[Sequence[ExperienceItem]], dict[str, list[float]] | Awaitable[dict[str, list[float]]]]
DynamicSummaryFn = Callable[["DynamicCommunityContext"], Iterable[ExperienceCardPatch] | Awaitable[Iterable[ExperienceCardPatch]]]
DynamicPromptTokenEstimator = Callable[
    [ExperienceCommunity, Sequence[ExperienceItem], Sequence[dict[str, Any]]],
    int | Awaitable[int],
]
FALLBACK_ROUTING_TEMPERATURE = 8.0


@dataclass(frozen=True)
class DynamicCommunityContext:
    """Input contract for the external dynamic LLM edit prompt.

    The updater never calls an LLM directly.  The caller receives an updated
    community, its selected member items, and old generated ExperienceCards.

    Required patch contract:
    - L0 raw trajectory communities may return ``update`` patches plus ``add``
      patches for newly discovered independent cards.
    - L1+ ExperienceCard communities must return ``update`` patches only.
    - Every patch must include ``metadata['confidence']``.
    - Patches must include fresh text and embedding.
    - Patches must not provide support_mass.  The state reallocates the source
      community support_mass across all active generated cards by normalized
      confidence.  Unchanged old cards keep their previous confidence.
    """

    level: int
    community: ExperienceCommunity
    member_items: list[dict[str, Any]]
    previous_generated_experiences: list[dict[str, Any]]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class DynamicRoutingResult:
    level: int
    item_ids: list[str]
    selected_assignments: dict[str, dict[str, float]]
    posterior_assignments: dict[str, dict[str, float]]
    routing_model_kind: str


@dataclass(frozen=True)
class DynamicUpdateResult:
    inserted_item_ids: list[str]
    initial_affected_community_ids: list[str]
    updated_community_ids: list[str]
    changed_item_ids: list[str]
    excluded_item_ids: list[str] = field(default_factory=list)
    excluded_oversize_singletons: list[dict[str, Any]] = field(default_factory=list)
    terminal_changed_item_ids: list[str] = field(default_factory=list)
    requires_skill_export: bool = False
    patch_results: list[DynamicPatchResult] = field(default_factory=list)
    reroute_results: list[RerouteResult] = field(default_factory=list)
    routing_results: list[DynamicRoutingResult] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceHierarchyDynamicUpdater:
    """Sequential budget-constrained online updater.

    Dynamic update assumptions:
    - trajectory arrivals are processed one at a time in caller order;
    - L0 uses token-budget admission before inserting into candidate communities;
    - if every selected L0 candidate would exceed budget, a new L0 community is
      created and the saved GMM routing model grows by one component;
    - existing communities are not split during online insertion;
    - routing GMM parameters are maintained with remove/add sufficient stats;
    - changed ExperienceCards propagate upward; terminal top-level changes only
      emit ``requires_skill_export`` and are handled by skill_export.py.
    """

    config: ProjectedGmmDynamicTreeConfig
    max_propagation_rounds: int = 16

    async def update(
        self,
        *,
        state: ExperienceHierarchyState,
        new_trajectory_items: Iterable[ExperienceItem],
        dynamic_summary_fn: DynamicSummaryFn,
        embed_fn: EmbedFn | None = None,
        dynamic_prompt_token_estimator: DynamicPromptTokenEstimator | None = None,
        dynamic_prompt_token_budget: int | None = None,
    ) -> DynamicUpdateResult:
        new_items = [copy.deepcopy(item) for item in new_trajectory_items]
        if not new_items:
            validation = await state.validate_hierarchy(require_no_pending_reroute=True)
            return DynamicUpdateResult([], [], [], [], validation=validation)
        item_ids = [str(item.item_id) for item in new_items]
        if len(set(item_ids)) != len(item_ids):
            duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
            raise ValueError(f"duplicate dynamic trajectory item_id(s): {duplicates[:10]}")

        inserted_ids: list[str] = []
        excluded_ids: list[str] = []
        excluded_oversize_singletons: list[dict[str, Any]] = []
        initial_affected: set[str] = set()
        patch_results: list[DynamicPatchResult] = []
        reroute_results: list[RerouteResult] = []
        routing_results: list[DynamicRoutingResult] = []
        updated_communities: set[str] = set()
        changed_items: set[str] = set()
        terminal_changed: set[str] = set()

        for raw_item in new_items:
            item = copy.deepcopy(raw_item)
            if embed_fn is not None:
                item = (await _ensure_embeddings([item], embed_fn))[0]
            _validate_new_trajectories([item])
            await self._ensure_all_routing_contribution_caches(state)
            result = await self._admit_one_trajectory(
                state=state,
                item=item,
                dynamic_prompt_token_estimator=dynamic_prompt_token_estimator,
                dynamic_prompt_token_budget=dynamic_prompt_token_budget,
            )
            inserted_ids.extend(result.inserted_item_ids)
            excluded_ids.extend(result.excluded_item_ids)
            excluded_oversize_singletons.extend(result.excluded_oversize_singletons)
            initial_affected.update(result.initial_affected_community_ids)
            reroute_results.extend(result.reroute_results)
            routing_results.extend(result.routing_results)

        propagated = await self._propagate_affected_communities(
            state=state,
            affected=sorted(initial_affected),
            dynamic_summary_fn=dynamic_summary_fn,
        )
        patch_results.extend(propagated.patch_results)
        reroute_results.extend(propagated.reroute_results)
        routing_results.extend(propagated.routing_results)
        updated_communities.update(propagated.updated_community_ids)
        changed_items.update(propagated.changed_item_ids)
        terminal_changed.update(propagated.terminal_changed_item_ids)

        await state.clear_pending_reroute_items()
        if self.config.dynamic_update.clear_stale_after_propagation:
            await state.clear_layer_stale(0)
        validation = await state.validate_hierarchy(require_no_pending_reroute=True, require_no_stale_layers=False)
        if not validation.get("ok", False):
            raise ValueError(f"dynamic update produced invalid hierarchy: {validation}")
        return DynamicUpdateResult(
            inserted_item_ids=inserted_ids,
            initial_affected_community_ids=sorted(initial_affected),
            updated_community_ids=sorted(updated_communities),
            changed_item_ids=sorted(changed_items),
            excluded_item_ids=excluded_ids,
            excluded_oversize_singletons=excluded_oversize_singletons,
            terminal_changed_item_ids=sorted(terminal_changed),
            requires_skill_export=bool(terminal_changed),
            patch_results=patch_results,
            reroute_results=reroute_results,
            routing_results=routing_results,
            validation=validation,
        )

    async def _admit_one_trajectory(
        self,
        *,
        state: ExperienceHierarchyState,
        item: ExperienceItem,
        dynamic_prompt_token_estimator: DynamicPromptTokenEstimator | None = None,
        dynamic_prompt_token_budget: int | None = None,
    ) -> DynamicUpdateResult:
        await self._require_saved_routing_model(state, level=0)
        item_token_count = self._item_token_count(item)
        if dynamic_prompt_token_estimator is not None:
            token_budget = int(dynamic_prompt_token_budget or self.config.summary_budget.analyst_prompt_token_budget)
        else:
            token_budget = int(self.config.summary_budget.effective_token_budget)
        singleton_prompt_tokens = await self._estimate_l0_singleton_prompt_tokens(
            item=item,
            token_budget=token_budget,
            dynamic_prompt_token_estimator=dynamic_prompt_token_estimator,
        )
        if singleton_prompt_tokens > token_budget:
            validation = await state.validate_hierarchy(require_no_pending_reroute=True, require_no_stale_layers=False)
            excluded = _dynamic_excluded_oversize_singleton(item, singleton_prompt_tokens, token_budget)
            return DynamicUpdateResult(
                [],
                [],
                [],
                [],
                excluded_item_ids=[item.item_id],
                excluded_oversize_singletons=[excluded],
                validation=validation,
            )

        await state.insert_trajectory_items([item])
        inserted_ids = [item.item_id]

        initial_routing = await self.route_existing_items(state, level=0, item_ids=inserted_ids)
        gated_selected, gated_posterior, new_communities = await self._gate_l0_budget_or_grow(
            state,
            item=item,
            routing=initial_routing,
            item_token_count=item_token_count,
            token_budget=token_budget,
            dynamic_prompt_token_estimator=dynamic_prompt_token_estimator,
        )
        gated_routing = DynamicRoutingResult(
            level=0,
            item_ids=inserted_ids,
            selected_assignments=gated_selected,
            posterior_assignments=gated_posterior,
            routing_model_kind=initial_routing.routing_model_kind + "+budget_constrained",
        )
        initial_reroute = await state.reroute_items_at_level(
            level=0,
            assignments=gated_selected,
            posterior_assignments=gated_posterior,
            new_communities=new_communities,
        )
        if self.config.dynamic_update.update_routing_model:
            for community in new_communities:
                await self._append_routing_component(state, level=0, community=community, seed_item=item)
            await self._update_layer_routing_model(state, level=0, item_ids=inserted_ids, posterior_assignments=gated_posterior)

        return DynamicUpdateResult(
            inserted_item_ids=inserted_ids,
            initial_affected_community_ids=sorted(set(initial_reroute.affected_community_ids)),
            updated_community_ids=[],
            changed_item_ids=[],
            reroute_results=[initial_reroute],
            routing_results=[gated_routing],
        )

    async def _update_one_trajectory(
        self,
        *,
        state: ExperienceHierarchyState,
        item: ExperienceItem,
        dynamic_summary_fn: DynamicSummaryFn,
    ) -> DynamicUpdateResult:
        admission = await self._admit_one_trajectory(state=state, item=item)
        propagated = await self._propagate_affected_communities(
            state=state,
            affected=admission.initial_affected_community_ids,
            dynamic_summary_fn=dynamic_summary_fn,
        )

        await state.clear_pending_reroute_items()
        validation = await state.validate_hierarchy(require_no_pending_reroute=True, require_no_stale_layers=False)
        if not validation.get("ok", False):
            raise ValueError(f"dynamic update produced invalid hierarchy: {validation}")
        return DynamicUpdateResult(
            inserted_item_ids=admission.inserted_item_ids,
            initial_affected_community_ids=admission.initial_affected_community_ids,
            updated_community_ids=propagated.updated_community_ids,
            changed_item_ids=propagated.changed_item_ids,
            excluded_item_ids=admission.excluded_item_ids,
            excluded_oversize_singletons=admission.excluded_oversize_singletons,
            terminal_changed_item_ids=propagated.terminal_changed_item_ids,
            requires_skill_export=propagated.requires_skill_export,
            patch_results=propagated.patch_results,
            reroute_results=[*admission.reroute_results, *propagated.reroute_results],
            routing_results=[*admission.routing_results, *propagated.routing_results],
            validation=validation,
        )

    async def _propagate_affected_communities(
        self,
        *,
        state: ExperienceHierarchyState,
        affected: Sequence[str],
        dynamic_summary_fn: DynamicSummaryFn,
    ) -> DynamicUpdateResult:
        patch_results: list[DynamicPatchResult] = []
        reroute_results: list[RerouteResult] = []
        routing_results: list[DynamicRoutingResult] = []
        updated_communities: set[str] = set()
        changed_items: set[str] = set()
        terminal_changed: set[str] = set()

        rounds = 0
        pending = sorted(set(affected))
        while pending:
            rounds += 1
            if rounds > self.max_propagation_rounds:
                raise RuntimeError("dynamic update exceeded max_propagation_rounds")
            current = sorted(set(pending))
            pending = []
            communities = await state.community_objects(current)
            by_level: dict[int, list[ExperienceCommunity]] = {}
            for community in communities:
                by_level.setdefault(int(community.level), []).append(community)

            for level in sorted(by_level):
                level_communities = sorted(by_level[level], key=lambda community: community.community_id)

                async def run_one(community: ExperienceCommunity) -> tuple[ExperienceCommunity, list[ExperienceCardPatch]]:
                    context = await self._build_context(state, community)
                    patches = list(await _maybe_await(dynamic_summary_fn(context)))
                    return community, patches

                llm_results = await asyncio.gather(*(run_one(community) for community in level_communities))
                for community, patches in llm_results:
                    patch_result = await state.commit_dynamic_community_update(community=community, patches=patches)
                    patch_results.append(patch_result)
                    updated_communities.add(community.community_id)
                    changed_items.update(patch_result.changed_item_ids)

                    reroute_ids = list(dict.fromkeys(patch_result.requires_reroute_item_ids))
                    support_only_ids = [iid for iid in patch_result.support_changed_item_ids if iid not in set(reroute_ids)]

                    next_affected, next_terminal = await self._propagate_reroute_items(state, reroute_ids)
                    pending.extend(next_affected)
                    terminal_changed.update(next_terminal)

                    support_affected, support_terminal = await self._propagate_support_only_items(state, support_only_ids)
                    pending.extend(support_affected)
                    terminal_changed.update(support_terminal)

                    routing_results.extend(getattr(self, "_last_routing_results", []))
                    reroute_results.extend(getattr(self, "_last_reroute_results", []))
                    self._last_routing_results = []
                    self._last_reroute_results = []

        return DynamicUpdateResult(
            inserted_item_ids=[],
            initial_affected_community_ids=[],
            updated_community_ids=sorted(updated_communities),
            changed_item_ids=sorted(changed_items),
            terminal_changed_item_ids=sorted(terminal_changed),
            requires_skill_export=bool(terminal_changed),
            patch_results=patch_results,
            reroute_results=reroute_results,
            routing_results=routing_results,
        )

    async def _propagate_reroute_items(self, state: ExperienceHierarchyState, item_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        self._last_routing_results = []
        self._last_reroute_results = []
        if not item_ids:
            return [], []
        affected: list[str] = []
        terminal: list[str] = []
        items = await state.item_objects(item_ids)
        by_level: dict[int, list[str]] = {}
        for item in items:
            by_level.setdefault(item.level, []).append(item.item_id)
        for level, ids in sorted(by_level.items()):
            model = await self._layer_routing_model(state, level)
            if model is None:
                if await self._level_without_routing_model_is_terminal(state, level):
                    terminal.extend(ids)
                    continue
                raise ValueError(f"dynamic update requires saved routing_model for non-terminal level {level}")
            routing = await self.route_existing_items(state, level=level, item_ids=ids)
            reroute = await state.reroute_items_at_level(
                level=level,
                assignments=routing.selected_assignments,
                posterior_assignments=routing.posterior_assignments,
            )
            if self.config.dynamic_update.update_routing_model:
                await self._update_layer_routing_model(state, level=level, item_ids=ids, posterior_assignments=routing.posterior_assignments)
            self._last_routing_results.append(routing)
            self._last_reroute_results.append(reroute)
            affected.extend(reroute.affected_community_ids)
        return affected, terminal

    async def _propagate_support_only_items(self, state: ExperienceHierarchyState, item_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        if not item_ids:
            return [], []
        affected: list[str] = []
        terminal: list[str] = []
        items = await state.item_objects(item_ids)
        by_level: dict[int, list[str]] = {}
        for item in items:
            by_level.setdefault(item.level, []).append(item.item_id)
        for level, ids in sorted(by_level.items()):
            model = await self._layer_routing_model(state, level)
            if model is None:
                if await self._level_without_routing_model_is_terminal(state, level):
                    terminal.extend(ids)
                    continue
                raise ValueError(f"dynamic update requires saved routing_model for non-terminal level {level}")
            selected: dict[str, dict[str, float]] = {}
            posterior: dict[str, dict[str, float]] = {}
            for item_id in ids:
                current_selected = await state.communities_for_item(item_id)
                current_posterior = await state.posterior_communities_for_item(item_id)
                if not current_selected:
                    # If the item has no existing assignment, route it normally.
                    route_affected, route_terminal = await self._propagate_reroute_items(state, [item_id])
                    affected.extend(route_affected)
                    terminal.extend(route_terminal)
                    continue
                selected[item_id] = current_selected
                posterior[item_id] = current_posterior
            if selected:
                reroute = await state.reroute_items_at_level(level=level, assignments=selected, posterior_assignments=posterior)
                if self.config.dynamic_update.update_routing_model:
                    await self._update_layer_routing_model(state, level=level, item_ids=list(selected), posterior_assignments=posterior)
                self._last_reroute_results.append(reroute)
                affected.extend(reroute.affected_community_ids)
        return affected, terminal

    async def route_existing_items(self, state: ExperienceHierarchyState, *, level: int, item_ids: Sequence[str]) -> DynamicRoutingResult:
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            return DynamicRoutingResult(level, [], {}, {}, "empty")
        items = await state.item_objects(item_ids)
        for item in items:
            if item.level != level:
                raise ValueError(f"item {item.item_id!r} has level {item.level}, expected {level}")
            if not item.embedding:
                raise ValueError(f"item {item.item_id!r} is missing embedding")
        model = await self._layer_routing_model(state, level)
        refinement_tree = (await state.layer_metadata(level)).get("budget_refinement", {}).get("refinement_routing_tree")
        if model is None:
            coarse_roots = dict((refinement_tree or {}).get("coarse_roots", {}))
            if len(coarse_roots) != 1:
                raise ValueError(f"no routing model for level {level}; dynamic update expects a stable built hierarchy")
            coarse_id = next(iter(coarse_roots))
            selected = {}
            posterior = {}
            for item in items:
                selected_leaf = _route_through_refinement_tree(
                    item=item,
                    coarse_community_id=coarse_id,
                    coarse_weight=1.0,
                    tree=refinement_tree,
                    soft_config=self._dynamic_soft_membership_config(),
                    selected_only=True,
                )
                posterior_leaf = _route_through_refinement_tree(
                    item=item,
                    coarse_community_id=coarse_id,
                    coarse_weight=1.0,
                    tree=refinement_tree,
                    soft_config=self._dynamic_soft_membership_config(),
                    selected_only=False,
                )
                if not selected_leaf or not posterior_leaf:
                    raise ValueError(f"budget refinement tree produced no active leaf assignment for item {item.item_id!r}")
                selected[item.item_id] = selected_leaf
                posterior[item.item_id] = posterior_leaf
            return DynamicRoutingResult(level, item_ids, selected, posterior, "budget_refinement_tree")
        embeddings = normalize_rows(np.asarray([item.embedding for item in items], dtype=float))
        projected = project_with_basis(embeddings, mean=np.asarray(model["pca_mean"], dtype=float), components=np.asarray(model["pca_components"], dtype=float))
        responsibilities = gmm_responsibilities_from_model(projected, pi=model["pi"], means=model["means"], variances=model["variances"])
        child_ids = list(model["community_ids"])
        active_communities = set(await state.communities_at_level(level))
        active_indices = [
            index
            for index, child_id in enumerate(child_ids)
            if child_id in active_communities or _refinement_tree_has_active_leaf(refinement_tree, child_id)
        ]
        if not active_indices:
            raise ValueError(f"routing model for level {level} has no active routable communities")
        if len(active_indices) != len(child_ids):
            child_ids = [child_ids[index] for index in active_indices]
            responsibilities = responsibilities[:, active_indices]
        selected = _membership_weight_dicts_preserve_mass(item_ids, child_ids, responsibilities, self._dynamic_soft_membership_config())
        selected = {iid: {cid: w for cid, w in weights.items() if float(w) > 1.0e-12} for iid, weights in selected.items()}
        posterior = {
            item_id: {
                child_ids[col]: float(responsibilities[row, col])
                for col in range(len(child_ids))
                if float(responsibilities[row, col]) > 1.0e-12
            }
            for row, item_id in enumerate(item_ids)
        }
        if refinement_tree:
            refined_selected: dict[str, dict[str, float]] = {}
            refined_posterior: dict[str, dict[str, float]] = {}
            for row, item_id in enumerate(item_ids):
                selected_leaf: dict[str, float] = {}
                posterior_leaf: dict[str, float] = {}
                for coarse_id, coarse_weight in selected.get(item_id, {}).items():
                    routed = _route_through_refinement_tree(
                        item=items[row],
                        coarse_community_id=coarse_id,
                        coarse_weight=float(coarse_weight),
                        tree=refinement_tree,
                        soft_config=self._dynamic_soft_membership_config(),
                        selected_only=True,
                    )
                    if routed is None:
                        selected_leaf[coarse_id] = float(coarse_weight)
                    else:
                        for leaf_id, weight in routed.items():
                            selected_leaf[leaf_id] = selected_leaf.get(leaf_id, 0.0) + float(weight)
                for coarse_id, coarse_weight in posterior.get(item_id, {}).items():
                    routed = _route_through_refinement_tree(
                        item=items[row],
                        coarse_community_id=coarse_id,
                        coarse_weight=float(coarse_weight),
                        tree=refinement_tree,
                        soft_config=self._dynamic_soft_membership_config(),
                        selected_only=False,
                    )
                    if routed is None:
                        posterior_leaf[coarse_id] = float(coarse_weight)
                    else:
                        for leaf_id, weight in routed.items():
                            posterior_leaf[leaf_id] = posterior_leaf.get(leaf_id, 0.0) + float(weight)
                refined_selected[item_id] = {cid: weight for cid, weight in selected_leaf.items() if weight > 1.0e-12}
                refined_posterior[item_id] = {cid: weight for cid, weight in posterior_leaf.items() if weight > 1.0e-12}
                if not refined_selected[item_id] or not refined_posterior[item_id]:
                    raise ValueError(f"budget refinement tree produced no active leaf assignment for item {item_id!r}")
            selected = refined_selected
            posterior = refined_posterior
        return DynamicRoutingResult(
            level=level,
            item_ids=item_ids,
            selected_assignments=selected,
            posterior_assignments=posterior,
            routing_model_kind=str(model.get("routing_model_kind", "fixed_k_pca_gmm")) + ("+budget_refinement_tree" if refinement_tree else ""),
        )

    async def _update_layer_routing_model(self, state: ExperienceHierarchyState, *, level: int, item_ids: Sequence[str], posterior_assignments: dict[str, dict[str, float]]) -> None:
        """Online remove/add update of the saved GMM routing model.

        Each routed item has a cached sufficient-statistics contribution.  When
        an item is new, we only add its contribution.  When an existing
        ExperienceCard changes text/embedding/support_mass, we first remove its
        previous contribution and then add the new one.  This keeps the update
        online without double-counting changed cards.
        """
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            return
        await self._ensure_layer_routing_contributions(state, level)
        model = await self._layer_routing_model(state, level)
        if model is None:
            raise ValueError(f"cannot update level {level} routing model because no routing_model is saved")
        metadata = await state.layer_metadata(level)
        refinement_tree = metadata.get("budget_refinement", {}).get("refinement_routing_tree")
        child_ids = list(model["community_ids"])
        if not child_ids:
            return
        counts, first, second, contributions = self._routing_stat_arrays(model)
        index_by_child = {child_id: index for index, child_id in enumerate(child_ids)}
        items = await state.item_objects(item_ids)
        projected = self._project_embeddings(model, [item.embedding for item in items])

        for row, item in enumerate(items):
            old = contributions.pop(item.item_id, None)
            if old:
                self._remove_contribution(counts, first, second, index_by_child, old)

            posterior = _coarsen_posterior_for_routing_model(
                posterior_assignments.get(item.item_id, {}),
                child_ids=child_ids,
                refinement_tree=refinement_tree,
            )
            if not posterior:
                posterior = _coarsen_posterior_for_routing_model(
                    await state.posterior_communities_for_item(item.item_id),
                    child_ids=child_ids,
                    refinement_tree=refinement_tree,
                )
            posterior = _positive_weight_values({cid: posterior.get(cid, 0.0) for cid in child_ids})
            if not posterior:
                raise ValueError(f"item {item.item_id!r} has no posterior membership at level {level}")

            contribution = {
                "support_mass": float(item.support_mass),
                "projected": [float(value) for value in projected[row].tolist()],
                "posterior": posterior,
            }
            self._add_contribution(counts, first, second, index_by_child, contribution)
            contributions[item.item_id] = contribution

        safe_counts = np.maximum(counts, np.finfo(float).eps)
        previous_means = np.asarray(model["means"], dtype=float)
        previous_variances = np.asarray(model["variances"], dtype=float)
        means = np.divide(first, safe_counts[:, None], out=previous_means.copy(), where=counts[:, None] > 1.0e-12)
        variances = np.divide(second, safe_counts[:, None], out=(previous_variances + previous_means * previous_means), where=counts[:, None] > 1.0e-12) - means * means
        variances = np.maximum(variances, self.config.gmm_bic.min_covar)
        total_count = float(max(float(counts.sum()), np.finfo(float).eps))
        pi = counts / total_count

        updated_model = dict(model)
        updated_model.update(
            {
                "pi": pi.astype(float).tolist(),
                "means": means.astype(float).tolist(),
                "variances": variances.astype(float).tolist(),
                "component_effective_counts": counts.astype(float).tolist(),
                "total_effective_count": total_count,
                "first_moments": first.astype(float).tolist(),
                "second_moments": second.astype(float).tolist(),
                "online_em_updates": int(model.get("online_em_updates", 0)) + 1,
                "item_contributions": contributions,
                "item_contributions_initialized": True,
                "update_rule": "online_remove_add_sufficient_statistics",
            }
        )
        metadata["routing_model"] = updated_model
        await state.update_layer_metadata(level, metadata)

    async def _gate_l0_budget_or_grow(
        self,
        state: ExperienceHierarchyState,
        *,
        item: ExperienceItem,
        routing: DynamicRoutingResult,
        item_token_count: int,
        token_budget: int,
        dynamic_prompt_token_estimator: DynamicPromptTokenEstimator | None = None,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], list[ExperienceCommunity]]:
        item_id = item.item_id
        selected = dict(routing.selected_assignments.get(item_id, {}))
        posterior = dict(routing.posterior_assignments.get(item_id, {}))
        ordered_candidates = sorted(selected, key=lambda cid: float(selected[cid]), reverse=True)
        candidate_costs = await self._community_prompt_token_costs(
            state,
            ordered_candidates,
            item=item,
            selected_weights=selected,
            posterior_weights=posterior,
            fallback_item_token_count=item_token_count,
            dynamic_prompt_token_estimator=dynamic_prompt_token_estimator,
        )
        accepted: list[str] = []
        for community_id in ordered_candidates:
            if int(candidate_costs.get(community_id, 0)) <= int(token_budget):
                accepted.append(community_id)

        if accepted:
            selected_weights = {
                community_id: float(selected[community_id])
                for community_id in accepted
                if float(selected.get(community_id, 0.0)) > 1.0e-12
            }
            posterior_weights = {
                community_id: float(weight)
                for community_id, weight in posterior.items()
                if float(weight) > 1.0e-12
            }
            if not posterior_weights:
                posterior_weights = dict(selected_weights)
            return {item_id: selected_weights}, {item_id: posterior_weights}, []

        community = self._new_l0_dynamic_community(
            item=item,
            item_token_count=item_token_count,
            token_budget=token_budget,
            rejected_candidates=ordered_candidates,
            rejected_candidate_posterior_weights={
                community_id: float(posterior.get(community_id, 0.0))
                for community_id in ordered_candidates
                if float(posterior.get(community_id, 0.0)) > 1.0e-12
            },
            rejected_candidate_token_costs=candidate_costs,
        )
        return {item_id: {community.community_id: 1.0}}, {item_id: {community.community_id: 1.0}}, [community]

    async def _estimate_l0_singleton_prompt_tokens(
        self,
        *,
        item: ExperienceItem,
        token_budget: int,
        dynamic_prompt_token_estimator: DynamicPromptTokenEstimator | None,
    ) -> int:
        item_token_count = self._item_token_count(item)
        if dynamic_prompt_token_estimator is None:
            return int(item_token_count)
        community = self._new_l0_dynamic_community(
            item=item,
            item_token_count=item_token_count,
            token_budget=token_budget,
            rejected_candidates=[],
            rejected_candidate_posterior_weights={},
            rejected_candidate_token_costs={},
        )
        return int(await _maybe_await(dynamic_prompt_token_estimator(community, [item], [])))

    async def _community_prompt_token_costs(
        self,
        state: ExperienceHierarchyState,
        community_ids: Sequence[str],
        *,
        item: ExperienceItem,
        selected_weights: dict[str, float],
        posterior_weights: dict[str, float],
        fallback_item_token_count: int,
        dynamic_prompt_token_estimator: DynamicPromptTokenEstimator | None,
    ) -> dict[str, int]:
        if dynamic_prompt_token_estimator is None:
            member_costs = await self._community_token_costs(state, community_ids)
            return {
                community_id: int(member_costs.get(community_id, 0)) + int(fallback_item_token_count)
                for community_id in community_ids
            }
        costs: dict[str, int] = {}
        communities = await state.community_objects(community_ids)
        for community in communities:
            proposed = replace(
                community,
                member_weights={
                    **dict(community.member_weights),
                    item.item_id: float(selected_weights.get(community.community_id, 0.0)),
                },
                posterior_member_weights={
                    **dict(community.posterior_member_weights),
                    item.item_id: float(posterior_weights.get(community.community_id, selected_weights.get(community.community_id, 0.0))),
                },
            )
            context = await self._build_context(state, proposed)
            member_items = await state.item_objects(list(proposed.member_weights))
            costs[community.community_id] = int(
                await _maybe_await(
                    dynamic_prompt_token_estimator(
                        proposed,
                        member_items,
                        list(context.previous_generated_experiences or []),
                    )
                )
            )
        return costs

    async def _community_token_costs(self, state: ExperienceHierarchyState, community_ids: Sequence[str]) -> dict[str, int]:
        community_ids = list(dict.fromkeys(community_ids))
        if not community_ids:
            return {}
        communities = await state.community_objects(community_ids)
        member_ids = sorted({item_id for community in communities for item_id in community.member_weights})
        member_items = await state.item_objects(member_ids) if member_ids else []
        token_by_item = {item.item_id: self._item_token_count(item) for item in member_items}
        return {
            community.community_id: int(sum(token_by_item.get(item_id, 0) for item_id in community.member_weights))
            for community in communities
        }

    def _new_l0_dynamic_community(
        self,
        *,
        item: ExperienceItem,
        item_token_count: int,
        token_budget: int,
        rejected_candidates: Sequence[str],
        rejected_candidate_posterior_weights: dict[str, float],
        rejected_candidate_token_costs: dict[str, int],
    ) -> ExperienceCommunity:
        community_id = "L0_DYN_" + hashlib.sha1(f"l0-dyn:{item.item_id}".encode("utf-8")).hexdigest()[:12]
        success_count, failure_count, outcome_mode = _single_item_outcome(item)
        return ExperienceCommunity(
            community_id=community_id,
            level=0,
            member_weights={item.item_id: 1.0},
            posterior_member_weights={item.item_id: 1.0},
            clustering_method="dynamic_budget_overflow_singleton",
            support_mass=float(item.support_mass),
            outcome_mode=outcome_mode,
            success_count=success_count,
            failure_count=failure_count,
            metadata={
                "created_by": "dynamic_budget_constrained_online_gmm",
                "seed_item_id": item.item_id,
                "prompt_token_cost": int(item_token_count),
                "budget": int(token_budget),
                "rejected_candidate_community_ids": list(rejected_candidates),
                "rejected_candidate_posterior_weights": {cid: float(rejected_candidate_posterior_weights.get(cid, 0.0)) for cid in rejected_candidates},
                "rejected_candidate_token_costs": {cid: int(rejected_candidate_token_costs.get(cid, 0)) for cid in rejected_candidates},
                "split_reason": "dynamic_l0_budget_overflow_new_component",
            },
        )

    async def _append_routing_component(
        self,
        state: ExperienceHierarchyState,
        *,
        level: int,
        community: ExperienceCommunity,
        seed_item: ExperienceItem,
    ) -> None:
        metadata = await state.layer_metadata(level)
        model = metadata.get("routing_model")
        if not model:
            raise ValueError(f"cannot grow level {level} routing model because no routing_model is saved")
        if community.community_id in list(model.get("community_ids", [])):
            return
        projected = self._project_embeddings(model, [seed_item.embedding])[0]
        counts, first, second, contributions = self._routing_stat_arrays(model)
        variance = self._initial_component_variance(model)
        dim = int(projected.shape[0])
        if variance.shape[0] != dim:
            variance = np.full(dim, float(self.config.gmm_bic.min_covar), dtype=float)

        updated_model = dict(model)
        updated_model["community_ids"] = list(model.get("community_ids", [])) + [community.community_id]
        updated_model["means"] = [list(row) for row in model.get("means", [])] + [[float(value) for value in projected.tolist()]]
        updated_model["variances"] = [list(row) for row in model.get("variances", [])] + [[float(value) for value in variance.tolist()]]
        updated_model["component_effective_counts"] = np.concatenate([counts, np.asarray([0.0])]).astype(float).tolist()
        zero = np.zeros((1, dim), dtype=float)
        updated_model["first_moments"] = np.vstack([first, zero]).astype(float).tolist()
        updated_model["second_moments"] = np.vstack([second, zero]).astype(float).tolist()
        updated_model["item_contributions"] = contributions
        updated_model["item_contributions_initialized"] = True
        updated_model["grow_k_components_added"] = int(model.get("grow_k_components_added", 0)) + 1
        metadata["routing_model"] = updated_model
        await state.update_layer_metadata(level, metadata)

    async def _require_saved_routing_model(self, state: ExperienceHierarchyState, *, level: int) -> None:
        metadata = await state.layer_metadata(level)
        if not metadata.get("routing_model"):
            raise ValueError(f"dynamic update requires saved routing_model for level {level}")

    async def _ensure_all_routing_contribution_caches(self, state: ExperienceHierarchyState) -> None:
        snapshot = await state.to_dict(include_embeddings=False, validate=False)
        for level_text, layer in sorted(snapshot.get("layers", {}).items(), key=lambda kv: int(kv[0])):
            if dict(layer.get("metadata") or {}).get("routing_model"):
                await self._ensure_layer_routing_contributions(state, int(level_text))

    async def _ensure_layer_routing_contributions(self, state: ExperienceHierarchyState, level: int) -> None:
        metadata = await state.layer_metadata(level)
        model = metadata.get("routing_model")
        if not model or model.get("item_contributions_initialized"):
            return
        refinement_tree = metadata.get("budget_refinement", {}).get("refinement_routing_tree")
        child_ids = list(model["community_ids"])
        item_ids = await state.layer_input_item_ids(level)
        items = await state.item_objects(item_ids)
        if not items:
            return
        projected = self._project_embeddings(model, [item.embedding for item in items])
        contributions: dict[str, dict[str, Any]] = {}
        for row, item in enumerate(items):
            posterior = _coarsen_posterior_for_routing_model(
                await state.posterior_communities_for_item(item.item_id),
                child_ids=child_ids,
                refinement_tree=refinement_tree,
            )
            posterior = _positive_weight_values({cid: posterior.get(cid, 0.0) for cid in child_ids})
            if not posterior:
                continue
            contribution = {
                "support_mass": float(item.support_mass),
                "projected": [float(value) for value in projected[row].tolist()],
                "posterior": posterior,
            }
            contributions[item.item_id] = contribution
        if not contributions:
            return
        updated = dict(model)
        updated.update({
            "item_contributions": contributions,
            "item_contributions_initialized": True,
            "item_contributions_source": "existing_state_preserve_routing_parameters",
            "update_rule": "online_remove_add_sufficient_statistics",
        })
        metadata["routing_model"] = updated
        await state.update_layer_metadata(level, metadata)

    def _routing_stat_arrays(self, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
        means = np.asarray(model["means"], dtype=float)
        variances = np.asarray(model["variances"], dtype=float)
        counts = np.asarray(model.get("component_effective_counts") or [], dtype=float)
        if counts.size != means.shape[0]:
            total = float(model.get("total_effective_count", 0.0)) or float(means.shape[0])
            pi = np.asarray(model.get("pi", []), dtype=float)
            counts = pi * total if pi.size == means.shape[0] else np.ones(means.shape[0], dtype=float)
        first = np.asarray(model.get("first_moments") or [], dtype=float)
        second = np.asarray(model.get("second_moments") or [], dtype=float)
        if first.shape != means.shape:
            first = counts[:, None] * means
        if second.shape != means.shape:
            second = counts[:, None] * (variances + means * means)
        contributions = {
            str(item_id): {
                "support_mass": float(payload.get("support_mass", 0.0)),
                "projected": [float(value) for value in payload.get("projected", [])],
                "posterior": {str(cid): float(weight) for cid, weight in dict(payload.get("posterior", {})).items()},
            }
            for item_id, payload in dict(model.get("item_contributions") or {}).items()
        }
        return counts.astype(float), first.astype(float), second.astype(float), contributions

    def _project_embeddings(self, model: dict[str, Any], embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        normalized = normalize_rows(np.asarray([list(embedding) for embedding in embeddings], dtype=float))
        return project_with_basis(
            normalized,
            mean=np.asarray(model["pca_mean"], dtype=float),
            components=np.asarray(model["pca_components"], dtype=float),
        )

    def _initial_component_variance(self, model: dict[str, Any]) -> np.ndarray:
        variances = np.asarray(model.get("variances") or [], dtype=float)
        if variances.ndim == 2 and variances.size:
            base = np.median(variances, axis=0)
        else:
            means = np.asarray(model.get("means") or [[0.0]], dtype=float)
            base = np.full(means.shape[1], float(self.config.gmm_bic.min_covar), dtype=float)
        return np.maximum(base, float(self.config.gmm_bic.min_covar))

    @staticmethod
    def _add_contribution(
        counts: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        index_by_child: dict[str, int],
        contribution: dict[str, Any],
    ) -> None:
        support = float(contribution.get("support_mass", 0.0))
        projected = np.asarray(contribution.get("projected", []), dtype=float)
        for community_id, weight in dict(contribution.get("posterior", {})).items():
            index = index_by_child.get(str(community_id))
            if index is None:
                continue
            delta = support * float(weight)
            counts[index] += delta
            first[index] += delta * projected
            second[index] += delta * (projected * projected)

    @staticmethod
    def _remove_contribution(
        counts: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        index_by_child: dict[str, int],
        contribution: dict[str, Any],
    ) -> None:
        support = float(contribution.get("support_mass", 0.0))
        projected = np.asarray(contribution.get("projected", []), dtype=float)
        for community_id, weight in dict(contribution.get("posterior", {})).items():
            index = index_by_child.get(str(community_id))
            if index is None:
                continue
            delta = support * float(weight)
            counts[index] = max(0.0, counts[index] - delta)
            first[index] -= delta * projected
            second[index] -= delta * (projected * projected)

    def _item_token_count(self, item: ExperienceItem) -> int:
        for key in self.config.summary_budget.token_count_metadata_keys:
            value = item.metadata.get(key)
            if value is None:
                continue
            try:
                count = int(float(value))
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
        return max(1, (len(item.text or "") + 3) // 4)

    async def _build_context(self, state: ExperienceHierarchyState, community: ExperienceCommunity) -> DynamicCommunityContext:
        payload = await state.build_dynamic_prompt_payload(community)
        return DynamicCommunityContext(
            level=community.level,
            community=community,
            member_items=list(payload.get("member_items", [])),
            previous_generated_experiences=list(payload.get("previous_generated_experiences", [])),
            raw_payload=payload,
        )

    async def _layer_routing_model(self, state: ExperienceHierarchyState, level: int) -> dict[str, Any] | None:
        metadata = await state.layer_metadata(level)
        model = metadata.get("routing_model")
        return copy.deepcopy(model) if model else None

    async def _level_without_routing_model_is_terminal(self, state: ExperienceHierarchyState, level: int) -> bool:
        next_level = int(level) + 1
        return not await state.items_at_level(next_level) and not await state.communities_at_level(next_level) and not await state.layer_metadata(next_level)

    def _dynamic_soft_membership_config(self) -> SoftMembershipConfig:
        return self.config.dynamic_update.to_soft_membership_config()


async def _ensure_embeddings(items: list[ExperienceItem], embed_fn: EmbedFn) -> list[ExperienceItem]:
    missing = [item for item in items if not item.embedding]
    if not missing:
        return items
    embeddings = await _maybe_await(embed_fn(missing))
    updated: list[ExperienceItem] = []
    for item in items:
        if item.embedding:
            updated.append(item)
        else:
            if item.item_id not in embeddings:
                raise KeyError(f"embed_fn did not return embedding for item_id={item.item_id!r}")
            updated.append(copy.copy(item).updated(embedding=list(embeddings[item.item_id])))
    return updated


def _validate_new_trajectories(items: Sequence[ExperienceItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"duplicate new trajectory item_id={item.item_id!r}")
        seen.add(item.item_id)
        if item.kind != "trajectory":
            raise ValueError("dynamic insertion accepts only trajectory items")
        if item.level != 0:
            raise ValueError("new trajectory items must have level 0")
        if not item.embedding:
            raise ValueError(f"new trajectory item {item.item_id!r} is missing embedding")


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _positive_weight_values(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(key): float(value) for key, value in dict(weights or {}).items() if float(value) > 1.0e-12}
    return clean


def _membership_weight_dicts_preserve_mass(
    item_ids: list[str],
    child_ids: list[str],
    weights: np.ndarray,
    config: SoftMembershipConfig,
) -> dict[str, dict[str, float]]:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2 or arr.shape != (len(item_ids), len(child_ids)):
        raise ValueError("weights shape must be (len(item_ids), len(child_ids))")
    if np.any(~np.isfinite(arr)):
        raise ValueError("weights must be finite")
    if np.any(arr < -1.0e-12):
        raise ValueError("weights must be non-negative")
    arr = np.maximum(arr, 0.0)
    return {
        item_id: {
            child_ids[int(index)]: float(arr[row, int(index)])
            for index in _select_preserved_membership_indices(arr[row], config)
            if float(arr[row, int(index)]) > 1.0e-12
        }
        for row, item_id in enumerate(item_ids)
    }


def _select_preserved_membership_indices(row: np.ndarray, config: SoftMembershipConfig) -> np.ndarray:
    order = [int(index) for index in np.argsort(-row) if float(row[int(index)]) > 1.0e-12]
    if not order:
        return np.asarray([], dtype=int)
    best = float(row[order[0]])
    if not config.save_soft_edges or config.recursive_assignment == "primary_argmax":
        return np.asarray(order[:1], dtype=int)
    if config.recursive_assignment == "top_r_threshold":
        selected = [order[0]]
        for index in order[1 : min(len(order), int(config.top_r_memberships))]:
            weight = float(row[index])
            if weight < config.min_membership_weight:
                continue
            if best - weight > config.max_membership_gap:
                continue
            selected.append(index)
        return np.asarray(selected, dtype=int)
    if config.recursive_assignment == "cumulative_mass":
        selected = [order[0]]
        mass = best
        for index in order[1:]:
            weight = float(row[index])
            if best - weight > config.max_membership_gap:
                break
            selected.append(index)
            mass += weight
            if mass >= config.cumulative_mass_coverage:
                break
        return np.asarray(selected, dtype=int)
    raise ValueError(f"unsupported recursive_assignment={config.recursive_assignment!r}")


def _single_item_outcome(item: ExperienceItem) -> tuple[int, int, str]:
    metadata = dict(item.metadata or {})
    record = metadata.get("record") if isinstance(metadata.get("record"), dict) else {}
    value = metadata.get("success", record.get("success"))
    if value is True:
        return 1, 0, "success"
    if value is False:
        return 0, 1, "failure"
    return 0, 0, "mixed"


def _dynamic_excluded_oversize_singleton(item: ExperienceItem, item_token_count: int, token_budget: int) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "source_community_id": None,
        "token_cost": int(item_token_count),
        "budget": int(token_budget),
        "reason": "oversize_singleton",
        "dynamic_arrival": True,
    }


def _route_through_refinement_tree(
    *,
    item: ExperienceItem,
    coarse_community_id: str,
    coarse_weight: float,
    tree: dict[str, Any],
    soft_config: SoftMembershipConfig,
    selected_only: bool,
) -> dict[str, float] | None:
    roots = dict(tree.get("coarse_roots", {}))
    root_id = roots.get(coarse_community_id)
    if not root_id:
        return None
    nodes = dict(tree.get("nodes", {}))
    pending: dict[str, float] = {str(root_id): float(coarse_weight)}
    leaves: dict[str, float] = {}
    while pending:
        node_id, weight = pending.popitem()
        if weight <= 1.0e-12:
            continue
        node = dict(nodes.get(node_id, {}))
        kind = str(node.get("kind", ""))
        if _is_refinement_leaf_kind(kind):
            community_id = str(node.get("community_id", ""))
            if community_id:
                leaves[community_id] = leaves.get(community_id, 0.0) + float(weight)
            continue
        if kind.startswith("excluded_"):
            continue
        if kind == "fallback_token_router":
            child_ids = [
                str(child_id)
                for child_id in node.get("child_node_ids", [])
                if _refinement_node_has_active_leaf(nodes, str(child_id))
            ]
            child_weights = _fallback_router_child_weights(
                item=item,
                nodes=nodes,
                child_ids=child_ids,
                selected_only=selected_only,
                soft_config=soft_config,
                temperature=float(node.get("routing_temperature", FALLBACK_ROUTING_TEMPERATURE)),
            )
            for child_id, child_weight in child_weights.items():
                pending[child_id] = pending.get(child_id, 0.0) + float(weight) * float(child_weight)
            continue
        if kind != "gmm_split":
            raise ValueError(f"unsupported refinement routing node kind={kind!r}")

        child_ids = [str(child_id) for child_id in node.get("child_node_ids", [])]
        if not child_ids:
            raise ValueError(f"refinement routing node {node_id!r} has no child_node_ids")
        embeddings = normalize_rows(np.asarray([item.embedding], dtype=float))
        projected = project_with_basis(
            embeddings,
            mean=np.asarray(node["pca_mean"], dtype=float),
            components=np.asarray(node["pca_components"], dtype=float),
        )
        resp = gmm_responsibilities_from_model(
            projected,
            pi=node["pi"],
            means=node["means"],
            variances=node["variances"],
        )
        active_indices = [
            index
            for index, child_id in enumerate(child_ids)
            if _refinement_node_has_active_leaf(nodes, child_id)
        ]
        if not active_indices:
            continue
        if len(active_indices) != len(child_ids):
            child_ids = [child_ids[index] for index in active_indices]
            resp = resp[:, active_indices]
        if selected_only:
            child_weights = _membership_weight_dicts_preserve_mass([item.item_id], child_ids, resp, soft_config)[item.item_id]
        else:
            child_weights = {
                child_id: float(resp[0, col])
                for col, child_id in enumerate(child_ids)
                if float(resp[0, col]) > 1.0e-12
            }
        for child_id, child_weight in child_weights.items():
            if child_id not in nodes:
                raise ValueError(f"refinement routing node {node_id!r} references missing child {child_id!r}")
            pending[child_id] = pending.get(child_id, 0.0) + float(weight) * float(child_weight)
    return {community_id: weight for community_id, weight in leaves.items() if weight > 1.0e-12}


def _refinement_tree_has_active_leaf(tree: dict[str, Any] | None, coarse_community_id: str) -> bool:
    if not tree:
        return False
    root_id = dict(tree.get("coarse_roots", {})).get(coarse_community_id)
    if not root_id:
        return False
    return _refinement_node_has_active_leaf(dict(tree.get("nodes", {})), str(root_id))


def _coarsen_posterior_for_routing_model(
    posterior: dict[str, float],
    *,
    child_ids: list[str],
    refinement_tree: dict[str, Any] | None,
) -> dict[str, float]:
    child_set = set(child_ids)
    leaf_to_coarse = _refinement_leaf_to_coarse_map(refinement_tree)
    coarse: dict[str, float] = {}
    for community_id, weight in dict(posterior).items():
        if community_id in child_set:
            coarse[community_id] = coarse.get(community_id, 0.0) + float(weight)
            continue
        parent = leaf_to_coarse.get(community_id)
        if parent in child_set:
            coarse[parent] = coarse.get(parent, 0.0) + float(weight)
    return {community_id: weight for community_id, weight in coarse.items() if weight > 1.0e-12}


def _refinement_leaf_to_coarse_map(tree: dict[str, Any] | None) -> dict[str, str]:
    if not tree:
        return {}
    mapping: dict[str, str] = {}
    nodes = dict(tree.get("nodes", {}))
    for coarse_id, root_id in dict(tree.get("coarse_roots", {})).items():
        pending = [str(root_id)]
        seen: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = dict(nodes.get(node_id, {}))
            if _is_refinement_leaf_kind(str(node.get("kind", ""))) and node.get("community_id"):
                mapping[str(node["community_id"])] = str(coarse_id)
            else:
                pending.extend(str(child_id) for child_id in node.get("child_node_ids", []))
    return mapping


def _refinement_node_has_active_leaf(nodes: dict[str, Any], node_id: str, memo: dict[str, bool] | None = None) -> bool:
    memo = memo if memo is not None else {}
    if node_id in memo:
        return memo[node_id]
    node = dict(nodes.get(node_id, {}))
    kind = str(node.get("kind", ""))
    if _is_refinement_leaf_kind(kind):
        result = bool(node.get("community_id"))
    elif kind.startswith("excluded_") or not node:
        result = False
    else:
        result = any(_refinement_node_has_active_leaf(nodes, str(child_id), memo) for child_id in node.get("child_node_ids", []))
    memo[node_id] = result
    return result


def _is_refinement_leaf_kind(kind: str) -> bool:
    return kind in {"leaf", "token_packing_leaf", "singleton_leaf"}


def _fallback_router_child_weights(
    *,
    item: ExperienceItem,
    nodes: dict[str, Any],
    child_ids: list[str],
    selected_only: bool,
    soft_config: SoftMembershipConfig,
    temperature: float = FALLBACK_ROUTING_TEMPERATURE,
) -> dict[str, float]:
    if not child_ids:
        return {}
    if len(child_ids) == 1:
        return {child_ids[0]: 1.0}
    centroids = []
    active_ids = []
    for child_id in child_ids:
        centroid = nodes.get(child_id, {}).get("centroid_embedding")
        if centroid:
            centroids.append(centroid)
            active_ids.append(child_id)
    if not active_ids:
        weight = 1.0 / float(len(child_ids))
        return {child_id: weight for child_id in child_ids}
    item_vec = normalize_rows(np.asarray([item.embedding], dtype=float))
    centroid_vecs = normalize_rows(np.asarray(centroids, dtype=float))
    scores = np.matmul(item_vec, centroid_vecs.T)[0]
    shifted = scores - float(np.max(scores))
    probs = np.exp(shifted * float(temperature))
    total = float(np.sum(probs))
    if total <= 0.0 or not np.isfinite(total):
        probs = np.ones(len(active_ids), dtype=float) / float(len(active_ids))
    else:
        probs = probs / total
    if selected_only:
        return membership_weight_dicts([item.item_id], active_ids, probs.reshape(1, -1), soft_config)[item.item_id]
    return {
        child_id: float(probs[index])
        for index, child_id in enumerate(active_ids)
        if float(probs[index]) > 1.0e-12
    }




__all__ = [
    "DynamicCommunityContext",
    "DynamicRoutingResult",
    "DynamicSummaryFn",
    "DynamicUpdateResult",
    "EmbedFn",
    "ExperienceHierarchyDynamicUpdater",
]

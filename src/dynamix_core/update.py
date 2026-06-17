from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
    terminal_changed_item_ids: list[str] = field(default_factory=list)
    requires_skill_export: bool = False
    patch_results: list[DynamicPatchResult] = field(default_factory=list)
    reroute_results: list[RerouteResult] = field(default_factory=list)
    routing_results: list[DynamicRoutingResult] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceHierarchyDynamicUpdater:
    """Fixed-K online-EM updater for a stable experience hierarchy.

    Dynamic update v1 assumptions:
    - no online birth / split / merge / delete of communities;
    - new trajectories are routed to existing communities;
    - routing GMM parameters are updated with full posterior responsibilities;
    - selected memberships are stored separately for prompts/support_mass;
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
    ) -> DynamicUpdateResult:
        new_items = [copy.deepcopy(item) for item in new_trajectory_items]
        if embed_fn is not None:
            new_items = await _ensure_embeddings(new_items, embed_fn)
        _validate_new_trajectories(new_items)
        if not new_items:
            validation = await state.validate_hierarchy(require_no_pending_reroute=True)
            return DynamicUpdateResult([], [], [], [], validation=validation)

        await state.insert_trajectory_items(new_items)
        inserted_ids = [item.item_id for item in new_items]

        initial_routing = await self.route_existing_items(state, level=0, item_ids=inserted_ids)
        initial_reroute = await state.reroute_items_at_level(
            level=0,
            assignments=initial_routing.selected_assignments,
            posterior_assignments=initial_routing.posterior_assignments,
        )
        if self.config.dynamic_update.update_routing_model:
            await self._update_layer_routing_model(state, level=0, item_ids=inserted_ids, posterior_assignments=initial_routing.posterior_assignments)

        affected = sorted(initial_reroute.affected_community_ids)
        patch_results: list[DynamicPatchResult] = []
        reroute_results: list[RerouteResult] = [initial_reroute]
        routing_results: list[DynamicRoutingResult] = [initial_routing]
        updated_communities: set[str] = set()
        changed_items: set[str] = set()
        terminal_changed: set[str] = set()

        rounds = 0
        while affected:
            rounds += 1
            if rounds > self.max_propagation_rounds:
                raise RuntimeError("dynamic update exceeded max_propagation_rounds")
            current = sorted(set(affected))
            affected = []
            for community in await state.community_objects(current):
                context = await self._build_context(state, community)
                patches = list(await _maybe_await(dynamic_summary_fn(context)))
                patch_result = await state.commit_dynamic_community_update(community=community, patches=patches)
                patch_results.append(patch_result)
                updated_communities.add(community.community_id)
                changed_items.update(patch_result.changed_item_ids)

                # New or embedding/text-changed cards need full reroute.
                reroute_ids = list(dict.fromkeys(patch_result.requires_reroute_item_ids))
                support_only_ids = [iid for iid in patch_result.support_changed_item_ids if iid not in set(reroute_ids)]

                next_affected, next_terminal = await self._propagate_reroute_items(state, reroute_ids)
                affected.extend(next_affected)
                terminal_changed.update(next_terminal)

                support_affected, support_terminal = await self._propagate_support_only_items(state, support_only_ids)
                affected.extend(support_affected)
                terminal_changed.update(support_terminal)

                # Collect routing/reroute results emitted by helper calls.
                routing_results.extend(getattr(self, "_last_routing_results", []))
                reroute_results.extend(getattr(self, "_last_reroute_results", []))
                self._last_routing_results = []
                self._last_reroute_results = []

        await state.clear_pending_reroute_items()
        if self.config.dynamic_update.clear_stale_after_propagation:
            # L0 can be stale because new trajectories were inserted before routing.
            await state.clear_layer_stale(0)
        validation = await state.validate_hierarchy(require_no_pending_reroute=True, require_no_stale_layers=False)
        if not validation.get("ok", False):
            raise ValueError(f"dynamic update produced invalid hierarchy: {validation}")
        return DynamicUpdateResult(
            inserted_item_ids=inserted_ids,
            initial_affected_community_ids=sorted(set(initial_reroute.affected_community_ids)),
            updated_community_ids=sorted(updated_communities),
            changed_item_ids=sorted(changed_items),
            terminal_changed_item_ids=sorted(terminal_changed),
            requires_skill_export=bool(terminal_changed),
            patch_results=patch_results,
            reroute_results=reroute_results,
            routing_results=routing_results,
            validation=validation,
        )

    async def update_batches(
        self,
        *,
        state: ExperienceHierarchyState,
        trajectory_batches: Iterable[Iterable[ExperienceItem]],
        dynamic_summary_fn: DynamicSummaryFn,
        embed_fn: EmbedFn | None = None,
    ) -> list[DynamicUpdateResult]:
        results: list[DynamicUpdateResult] = []
        for batch in trajectory_batches:
            results.append(await self.update(state=state, new_trajectory_items=list(batch), dynamic_summary_fn=dynamic_summary_fn, embed_fn=embed_fn))
        return results

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
                terminal.extend(ids)
                continue
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
                terminal.extend(ids)
                continue
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
            responsibilities = _renormalize_responsibility_columns(responsibilities[:, active_indices])
        selected = membership_weight_dicts(item_ids, child_ids, responsibilities, self._dynamic_soft_membership_config())
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
        """Refresh the saved fixed-K routing model from current state statistics.

        This intentionally recomputes sufficient statistics from all current
        items at the layer instead of blindly adding the provided batch.  Dynamic
        updates may change an existing ExperienceCard's text, embedding,
        confidence, or support_mass.  In those cases an additive online-EM update
        would double-count the old contribution.  A full sufficient-statistics
        refresh preserves the fixed-K assumption while keeping the routing model
        consistent with state.member_weights / state.posterior_member_weights.
        """
        model = await self._layer_routing_model(state, level)
        if model is None:
            return
        metadata = await state.layer_metadata(level)
        refinement_tree = metadata.get("budget_refinement", {}).get("refinement_routing_tree")
        active_item_ids = await state.layer_input_item_ids(level)
        items = await state.item_objects(active_item_ids)
        if not items:
            return
        child_ids = list(model["community_ids"])
        active_communities = set(await state.communities_at_level(level))
        active_child_ids = {
            child_id
            for child_id in child_ids
            if child_id in active_communities or _refinement_tree_has_active_leaf(refinement_tree, child_id)
        }
        if not active_child_ids:
            return
        embeddings = normalize_rows(np.asarray([item.embedding for item in items], dtype=float))
        projected = project_with_basis(
            embeddings,
            mean=np.asarray(model["pca_mean"], dtype=float),
            components=np.asarray(model["pca_components"], dtype=float),
        )
        responsibilities = []
        for item in items:
            posterior = _coarsen_posterior_for_routing_model(
                await state.posterior_communities_for_item(item.item_id),
                child_ids=child_ids,
                refinement_tree=refinement_tree,
            )
            # During insertion/update, use the fresh posterior assignments as a
            # safety fallback.  State should normally already contain them after
            # reroute_items_at_level(), but this keeps fixed-K statistic refresh
            # robust for just-inserted items and makes the provided argument
            # semantically meaningful.
            if not posterior and item.item_id in posterior_assignments:
                posterior = _coarsen_posterior_for_routing_model(
                    posterior_assignments[item.item_id],
                    child_ids=child_ids,
                    refinement_tree=refinement_tree,
                )
            row = [float(posterior.get(cid, 0.0)) if cid in active_child_ids else 0.0 for cid in child_ids]
            total = float(sum(row))
            if total <= 1.0e-12:
                # Last-resort deterministic fallback: route through the saved
                # model rather than failing the entire dynamic update because an
                # index did not yet expose posterior memberships.
                resp_row = gmm_responsibilities_from_model(projected[[items.index(item)]], pi=model["pi"], means=model["means"], variances=model["variances"])[0]
                row = [float(resp_row[j]) if child_ids[j] in active_child_ids else 0.0 for j in range(len(child_ids))]
                total = float(sum(row))
            if total <= 1.0e-12:
                raise ValueError(f"item {item.item_id!r} has no posterior membership at level {level}")
            responsibilities.append([value / total for value in row])
        resp = np.asarray(responsibilities, dtype=float)
        sample_weights = np.asarray([item.support_mass for item in items], dtype=float)
        weighted_resp = resp * sample_weights[:, None]
        counts = weighted_resp.sum(axis=0)
        safe_counts = np.maximum(counts, np.finfo(float).eps)
        first = weighted_resp.T @ projected
        second = weighted_resp.T @ (projected * projected)
        means = first / safe_counts[:, None]
        variances = np.maximum(second / safe_counts[:, None] - means * means, self.config.gmm_bic.min_covar)
        total_count = float(counts.sum())
        pi = counts / max(total_count, np.finfo(float).eps)

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
                "update_rule": "fixed_k_full_sufficient_statistics_refresh",
            }
        )
        metadata["routing_model"] = updated_model
        await state.update_layer_metadata(level, metadata)

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
        if kind == "leaf":
            community_id = str(node.get("community_id", ""))
            if community_id:
                leaves[community_id] = leaves.get(community_id, 0.0) + float(weight)
            continue
        if kind.startswith("excluded_"):
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
            resp = _renormalize_responsibility_columns(resp[:, active_indices])
        if selected_only:
            child_weights = membership_weight_dicts([item.item_id], child_ids, resp, soft_config)[item.item_id]
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
            if str(node.get("kind", "")) == "leaf" and node.get("community_id"):
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
    if kind == "leaf":
        result = bool(node.get("community_id"))
    elif kind.startswith("excluded_") or not node:
        result = False
    else:
        result = any(_refinement_node_has_active_leaf(nodes, str(child_id), memo) for child_id in node.get("child_node_ids", []))
    memo[node_id] = result
    return result


def _renormalize_responsibility_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("responsibility values must be a 2D array")
    if arr.shape[1] == 0:
        raise ValueError("cannot normalize zero responsibility columns")
    totals = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=float)
    valid = totals[:, 0] > 1.0e-12
    out[valid] = arr[valid] / totals[valid]
    if np.any(~valid):
        out[~valid] = 1.0 / float(arr.shape[1])
    return out


__all__ = [
    "DynamicCommunityContext",
    "DynamicRoutingResult",
    "DynamicSummaryFn",
    "DynamicUpdateResult",
    "EmbedFn",
    "ExperienceHierarchyDynamicUpdater",
]

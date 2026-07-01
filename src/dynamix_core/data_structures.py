from __future__ import annotations

import asyncio
import copy
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Literal

ITEM_KIND_TRAJECTORY = "trajectory"
ITEM_KIND_EXPERIENCE_CARD = "experience_card"
VALID_ITEM_KINDS = {ITEM_KIND_TRAJECTORY, ITEM_KIND_EXPERIENCE_CARD}
PatchOperation = Literal["update", "add"]
_EPS = 1.0e-12


def _uniq(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _finite_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    return vector


def _positive_weights(weights: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in dict(weights or {}).items():
        if not key:
            raise ValueError("membership id must be non-empty")
        value = float(value)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("membership weights must be positive and finite")
        clean[str(key)] = value
    return clean


def _support_mass(items: dict[str, "ExperienceItem"], weights: dict[str, float]) -> float:
    return float(sum(items[item_id].support_mass * weight for item_id, weight in weights.items() if item_id in items))


def _confidence_from_metadata(metadata: dict[str, Any], *, default: float | None = None) -> float:
    value = metadata.get("confidence", default)
    if value is None:
        raise ValueError("ExperienceCard confidence is required")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ExperienceCard confidence must be positive and finite")
    return value


@dataclass
class ExperienceItem:
    item_id: str
    level: int
    kind: str
    text: str
    embedding: list[float] = field(default_factory=list)
    support_mass: float = 1.0
    generated_from_community_ids: list[str] = field(default_factory=list)
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if self.level < 0:
            raise ValueError("level must be >= 0")
        if self.kind not in VALID_ITEM_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_ITEM_KINDS)}")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        self.embedding = _finite_vector(self.embedding)
        self.support_mass = float(self.support_mass)
        if self.support_mass <= 0.0 or not math.isfinite(self.support_mass):
            raise ValueError("support_mass must be positive and finite")
        self.generated_from_community_ids = _uniq(self.generated_from_community_ids)
        self.metadata = dict(self.metadata or {})

    def updated(
        self,
        *,
        text: str | None = None,
        embedding: list[float] | None = None,
        support_mass: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExperienceItem":
        return replace(
            self,
            text=self.text if text is None else text,
            embedding=list(self.embedding if embedding is None else embedding),
            support_mass=self.support_mass if support_mass is None else float(support_mass),
            version=self.version + 1,
            metadata={**self.metadata, **dict(metadata or {})},
        )

    def to_dict(self, *, include_embedding: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_embedding:
            payload.pop("embedding", None)
        return payload


@dataclass
class ExperienceCommunity:
    community_id: str
    level: int
    member_weights: dict[str, float]
    posterior_member_weights: dict[str, float] = field(default_factory=dict)
    generated_item_ids: list[str] = field(default_factory=list)
    clustering_method: str = ""
    support_mass: float = 0.0
    outcome_mode: str = "mixed"
    success_count: int = 0
    failure_count: int = 0
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.community_id:
            raise ValueError("community_id must be non-empty")
        if self.level < 0:
            raise ValueError("community level must be >= 0")
        self.member_weights = _positive_weights(self.member_weights)
        self.posterior_member_weights = _positive_weights(self.posterior_member_weights or dict(self.member_weights))
        missing_selected = set(self.member_weights) - set(self.posterior_member_weights)
        if missing_selected:
            raise ValueError(f"posterior_member_weights missing selected members: {sorted(missing_selected)[:5]}")
        self.generated_item_ids = _uniq(self.generated_item_ids)
        self.support_mass = float(self.support_mass)
        if self.support_mass < 0.0 or not math.isfinite(self.support_mass):
            raise ValueError("community support_mass must be non-negative and finite")
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceLayer:
    level: int
    input_item_ids: list[str] = field(default_factory=list)
    community_ids: list[str] = field(default_factory=list)
    generated_item_ids: list[str] = field(default_factory=list)
    stop_reason: str = ""
    stale: bool = False
    stale_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("layer level must be >= 0")
        self.input_item_ids = _uniq(self.input_item_ids)
        self.community_ids = _uniq(self.community_ids)
        self.generated_item_ids = _uniq(self.generated_item_ids)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceCardPatch:
    operation: PatchOperation
    item_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in {"update", "add"}:
            raise ValueError("operation must be update or add")
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.text:
            raise ValueError("text must be non-empty")
        if not self.embedding:
            raise ValueError("dynamic patches must include a fresh embedding")
        self.embedding = _finite_vector(self.embedding)
        self.metadata = dict(self.metadata or {})
        _confidence_from_metadata(self.metadata)


@dataclass(frozen=True)
class DynamicPatchResult:
    source_community_id: str
    updated_item_ids: list[str]
    added_item_ids: list[str]
    changed_item_ids: list[str]
    support_changed_item_ids: list[str]
    requires_reroute_item_ids: list[str]


@dataclass(frozen=True)
class RerouteResult:
    level: int
    changed_item_ids: list[str]
    affected_community_ids: list[str]
    unassigned_item_ids: list[str]
    new_community_ids: list[str]


@dataclass
class ExperienceHierarchyIndex:
    items_by_level: dict[int, set[str]] = field(default_factory=dict)
    communities_by_level: dict[int, set[str]] = field(default_factory=dict)
    generated_items_by_level: dict[int, set[str]] = field(default_factory=dict)
    item_to_communities: dict[str, dict[str, float]] = field(default_factory=dict)
    item_to_posterior_communities: dict[str, dict[str, float]] = field(default_factory=dict)
    community_to_items: dict[str, dict[str, float]] = field(default_factory=dict)
    community_to_posterior_items: dict[str, dict[str, float]] = field(default_factory=dict)
    community_to_generated_items: dict[str, set[str]] = field(default_factory=dict)
    item_to_source_communities: dict[str, set[str]] = field(default_factory=dict)
    primary_community_by_item: dict[str, str] = field(default_factory=dict)
    item_rank: dict[str, int] = field(default_factory=dict)
    community_rank: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_state(cls, *, items: dict[str, ExperienceItem], communities: dict[str, ExperienceCommunity]) -> "ExperienceHierarchyIndex":
        index = cls()
        for rank, (item_id, item) in enumerate(items.items()):
            index.items_by_level.setdefault(item.level, set()).add(item_id)
            index.item_rank[item_id] = rank
            if item.kind == ITEM_KIND_EXPERIENCE_CARD:
                index.generated_items_by_level.setdefault(item.level, set()).add(item_id)
                index.item_to_source_communities[item_id] = set(item.generated_from_community_ids)
        for rank, (community_id, community) in enumerate(communities.items()):
            index.communities_by_level.setdefault(community.level, set()).add(community_id)
            index.community_rank[community_id] = rank
            index.community_to_items[community_id] = dict(community.member_weights)
            index.community_to_posterior_items[community_id] = dict(community.posterior_member_weights)
            index.community_to_generated_items[community_id] = set(community.generated_item_ids)
            for item_id, weight in community.member_weights.items():
                index.item_to_communities.setdefault(item_id, {})[community_id] = weight
            for item_id, weight in community.posterior_member_weights.items():
                index.item_to_posterior_communities.setdefault(item_id, {})[community_id] = weight
        index.refresh_primary_all()
        return index

    def refresh_primary_all(self) -> None:
        self.primary_community_by_item = {}
        for item_id, memberships in self.item_to_communities.items():
            if memberships:
                self.primary_community_by_item[item_id] = max(
                    memberships,
                    key=lambda cid: (memberships[cid], -self.community_rank.get(cid, 10**12)),
                )

    def communities_for_item(self, item_id: str) -> dict[str, float]:
        return dict(self.item_to_communities.get(item_id, {}))

    def posterior_communities_for_item(self, item_id: str) -> dict[str, float]:
        return dict(self.item_to_posterior_communities.get(item_id, {}))

    def items_in_community(self, community_id: str) -> dict[str, float]:
        return dict(self.community_to_items.get(community_id, {}))

    def generated_items_for_community(self, community_id: str) -> list[str]:
        return list(self.community_to_generated_items.get(community_id, set()))

    def source_communities_for_item(self, item_id: str) -> list[str]:
        return list(self.item_to_source_communities.get(item_id, set()))

    def items_at_level(self, level: int) -> list[str]:
        return list(self.items_by_level.get(level, set()))

    def communities_at_level(self, level: int) -> list[str]:
        return list(self.communities_by_level.get(level, set()))

    def generated_items_at_level(self, level: int) -> list[str]:
        return list(self.generated_items_by_level.get(level, set()))


@dataclass
class ExperienceHierarchyState:
    _items: dict[str, ExperienceItem] = field(default_factory=dict, init=False, repr=False)
    _communities: dict[str, ExperienceCommunity] = field(default_factory=dict, init=False, repr=False)
    _layers: dict[int, ExperienceLayer] = field(default_factory=dict, init=False, repr=False)
    _pending_reroute_item_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _index: ExperienceHierarchyIndex | None = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def initialize_trajectory_items(self, items: Iterable[ExperienceItem]) -> None:
        batch = [copy.deepcopy(item) for item in items]
        async with self._lock:
            self._check_trajectory_batch(batch, existing_allowed=False)
            self._items = {item.item_id: item for item in batch}
            self._communities = {}
            self._layers = {}
            self._pending_reroute_item_ids = set()
            self._index = None

    async def insert_trajectory_items(self, items: Iterable[ExperienceItem]) -> None:
        batch = [copy.deepcopy(item) for item in items]
        async with self._lock:
            self._check_trajectory_batch(batch, existing_allowed=True)
            for item in batch:
                self._items[item.item_id] = item
            self._mark_layer_stale(0, "new_trajectory_items")
            if 0 in self._layers:
                layer = self._layers[0]
                inserted_ids = [item.item_id for item in batch]
                self._layers[0] = replace(layer, input_item_ids=_uniq([*layer.input_item_ids, *inserted_ids]))
            self._index = None

    async def commit_layer(
        self,
        *,
        level: int,
        communities: Iterable[ExperienceCommunity],
        generated_items: Iterable[ExperienceItem],
        stop_reason: str = "",
        metadata: dict[str, Any] | None = None,
        excluded_input_item_ids: Iterable[str] = (),
    ) -> None:
        community_batch = [copy.deepcopy(community) for community in communities]
        item_batch = [copy.deepcopy(item) for item in generated_items]
        excluded_batch = _uniq(excluded_input_item_ids)
        async with self._lock:
            self._commit_layer_unlocked(
                level=level,
                communities=community_batch,
                generated_items=item_batch,
                stop_reason=stop_reason,
                metadata=metadata,
                excluded_input_item_ids=excluded_batch,
            )

    async def build_dynamic_prompt_payload(self, community: ExperienceCommunity) -> dict[str, Any]:
        community = copy.deepcopy(community)
        async with self._lock:
            for item_id in community.member_weights:
                if item_id not in self._items:
                    raise KeyError(f"community {community.community_id!r} references missing item {item_id!r}")
                if self._items[item_id].level != community.level:
                    raise ValueError(f"community {community.community_id!r} member {item_id!r} has wrong level")
            old = self._communities.get(community.community_id)
            previous_ids = list(old.generated_item_ids if old else [])
            member_items = [self._items[iid] for iid in community.member_weights]
            is_raw_l0 = community.level == 0 and any(item.kind == ITEM_KIND_TRAJECTORY for item in member_items)
            if is_raw_l0:
                patch_contract = {
                    "analyst_mode": "raw_extractor",
                    "dynamic_summary_fn": "For this L0 raw trajectory community, return update patches for changed old cards and add patches for genuinely new independent cards.",
                    "allowed_patch_operations": ["update", "add"],
                    "update": "Use operation='update' only when revising an item_id from previous_generated_experiences.",
                    "add": "Use operation='add' only for a newly discovered independent reusable lesson in this raw trajectory community.",
                    "confidence": "Each update/add patch must include metadata['confidence']; unchanged old cards keep their stored confidence.",
                }
            else:
                patch_contract = {
                    "analyst_mode": "experience_abstractor",
                    "dynamic_summary_fn": "For this L1+ ExperienceCard community, return update patches only for existing previous_generated_experiences.",
                    "allowed_patch_operations": ["update"],
                    "update": "Each patch must use operation='update' and an item_id copied from previous_generated_experiences.",
                    "forbidden": "Do not use operation='add' for L1+ dynamic abstraction updates.",
                    "confidence": "Each update patch must include metadata['confidence']; unchanged old cards keep their stored confidence.",
                }
            return {
                "contract": {
                    **patch_contract,
                    "support_mass": "Do not set support_mass in patches. The state redistributes source community support_mass across active cards by normalized confidence.",
                    "concurrency": "External LLM concurrency must be controlled outside this state object.",
                },
                "proposed_community": community.to_dict(),
                "member_items": [item.to_dict(include_embedding=False) for item in member_items],
                "previous_generated_experiences": [self._items[iid].to_dict(include_embedding=False) for iid in previous_ids if iid in self._items],
            }

    async def commit_dynamic_community_update(self, *, community: ExperienceCommunity, patches: Iterable[ExperienceCardPatch]) -> DynamicPatchResult:
        community = copy.deepcopy(community)
        patch_batch = [copy.deepcopy(patch) for patch in patches]
        async with self._lock:
            return self._commit_dynamic_update_atomic_unlocked(community=community, patches=patch_batch)

    async def apply_experience_card_patches(self, *, source_community_id: str, patches: Iterable[ExperienceCardPatch]) -> DynamicPatchResult:
        patch_batch = [copy.deepcopy(patch) for patch in patches]
        async with self._lock:
            return self._apply_patches_atomic_unlocked(source_community_id=source_community_id, patches=patch_batch)

    async def reroute_items_at_level(
        self,
        *,
        level: int,
        assignments: dict[str, dict[str, float]],
        posterior_assignments: dict[str, dict[str, float]] | None = None,
        new_communities: Iterable[ExperienceCommunity] = (),
    ) -> RerouteResult:
        community_batch = [copy.deepcopy(community) for community in new_communities]
        assignments = copy.deepcopy(assignments)
        posterior_assignments = copy.deepcopy(posterior_assignments if posterior_assignments is not None else assignments)
        async with self._lock:
            return self._reroute_atomic_unlocked(level=level, assignments=assignments, posterior_assignments=posterior_assignments, new_communities=community_batch)

    async def communities_for_item(self, item_id: str) -> dict[str, float]:
        async with self._lock:
            return self._index_unlocked().communities_for_item(item_id)

    async def posterior_communities_for_item(self, item_id: str) -> dict[str, float]:
        async with self._lock:
            return self._index_unlocked().posterior_communities_for_item(item_id)

    async def items_in_community(self, community_id: str) -> dict[str, float]:
        async with self._lock:
            return self._index_unlocked().items_in_community(community_id)

    async def generated_items_for_community(self, community_id: str) -> list[str]:
        async with self._lock:
            return self._index_unlocked().generated_items_for_community(community_id)

    async def source_communities_for_item(self, item_id: str) -> list[str]:
        async with self._lock:
            return self._index_unlocked().source_communities_for_item(item_id)

    async def items_at_level(self, level: int) -> list[str]:
        async with self._lock:
            return self._index_unlocked().items_at_level(level)

    async def communities_at_level(self, level: int) -> list[str]:
        async with self._lock:
            return self._index_unlocked().communities_at_level(level)

    async def generated_items_at_level(self, level: int) -> list[str]:
        async with self._lock:
            return self._index_unlocked().generated_items_at_level(level)

    async def item_objects_at_level(self, level: int) -> list[ExperienceItem]:
        async with self._lock:
            return [copy.deepcopy(item) for item in self._items.values() if item.level == level]

    async def layer_input_item_ids(self, level: int) -> list[str]:
        async with self._lock:
            layer = self._layers.get(level)
            if layer is None:
                return [item_id for item_id, item in self._items.items() if item.level == level]
            return list(layer.input_item_ids)

    async def community_objects_at_level(self, level: int) -> list[ExperienceCommunity]:
        async with self._lock:
            return [copy.deepcopy(community) for community in self._communities.values() if community.level == level]

    async def item_objects(self, item_ids: Iterable[str]) -> list[ExperienceItem]:
        async with self._lock:
            return [copy.deepcopy(self._items[str(item_id)]) for item_id in item_ids]

    async def community_objects(self, community_ids: Iterable[str]) -> list[ExperienceCommunity]:
        async with self._lock:
            return [copy.deepcopy(self._communities[str(community_id)]) for community_id in community_ids]

    async def layer_metadata(self, level: int) -> dict[str, Any]:
        async with self._lock:
            layer = self._layers.get(level)
            return copy.deepcopy(layer.metadata if layer else {})

    async def update_layer_metadata(self, level: int, metadata: dict[str, Any]) -> None:
        async with self._lock:
            if level not in self._layers:
                raise KeyError(f"unknown layer={level}")
            self._layers[level] = replace(self._layers[level], metadata=copy.deepcopy(metadata))
            self._index = None

    async def update_community_metadata(self, community_id: str, metadata: dict[str, Any]) -> None:
        async with self._lock:
            if community_id not in self._communities:
                raise KeyError(f"unknown community_id={community_id!r}")
            community = self._communities[community_id]
            self._communities[community_id] = replace(community, metadata={**community.metadata, **copy.deepcopy(metadata)})
            self._index = None

    async def clear_pending_reroute_items(self, item_ids: Iterable[str] | None = None) -> None:
        async with self._lock:
            if item_ids is None:
                self._pending_reroute_item_ids.clear()
            else:
                self._pending_reroute_item_ids.difference_update(str(item_id) for item_id in item_ids)

    async def clear_layer_stale(self, level: int) -> None:
        async with self._lock:
            if level in self._layers:
                self._layers[level] = replace(self._layers[level], stale=False, stale_reason="")

    async def compute_community_support_mass(self, community_id: str) -> float:
        async with self._lock:
            return _support_mass(self._items, self._communities[community_id].member_weights)

    async def primary_community_assignment(self) -> dict[str, str]:
        async with self._lock:
            return dict(self._index_unlocked().primary_community_by_item)

    async def validate_hierarchy(
        self,
        *,
        strict_layers: bool = True,
        check_support_mass: bool = False,
        require_no_pending_reroute: bool = True,
        require_no_stale_layers: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            return _validate_maps(
                self._items,
                self._communities,
                self._layers,
                pending_reroute_item_ids=self._pending_reroute_item_ids,
                strict_layers=strict_layers,
                check_support_mass=check_support_mass,
                require_no_pending_reroute=require_no_pending_reroute,
                require_no_stale_layers=require_no_stale_layers,
            )

    async def to_dict(self, *, include_embeddings: bool = False, validate: bool = True) -> dict[str, Any]:
        async with self._lock:
            payload = {
                "items": {iid: item.to_dict(include_embedding=include_embeddings) for iid, item in self._items.items()},
                "communities": {cid: community.to_dict() for cid, community in self._communities.items()},
                "layers": {str(level): layer.to_dict() for level, layer in self._layers.items()},
                "pending_reroute_item_ids": sorted(self._pending_reroute_item_ids),
            }
            if validate:
                payload["validation"] = _validate_maps(self._items, self._communities, self._layers, pending_reroute_item_ids=self._pending_reroute_item_ids)
            return copy.deepcopy(payload)

    def _index_unlocked(self) -> ExperienceHierarchyIndex:
        if self._index is None:
            self._index = ExperienceHierarchyIndex.from_state(items=self._items, communities=self._communities)
        return self._index

    def _check_trajectory_batch(self, items: list[ExperienceItem], *, existing_allowed: bool) -> None:
        seen: set[str] = set()
        for item in items:
            if item.kind != ITEM_KIND_TRAJECTORY:
                raise ValueError("trajectory batch accepts only trajectory items")
            if item.level != 0:
                raise ValueError("trajectory items must have level 0")
            if item.item_id in seen:
                raise ValueError(f"duplicate item_id={item.item_id!r}")
            if not existing_allowed and item.item_id in self._items:
                raise ValueError(f"duplicate item_id={item.item_id!r}")
            if existing_allowed and item.item_id in self._items:
                raise ValueError(f"duplicate item_id={item.item_id!r}")
            if not item.embedding:
                raise ValueError("trajectory items must include embeddings")
            seen.add(item.item_id)

    def _mark_layer_stale(self, level: int, reason: str) -> None:
        if level in self._layers:
            layer = self._layers[level]
            self._layers[level] = replace(layer, stale=True, stale_reason=reason)

    def _clear_from_level_unlocked(self, level: int) -> tuple[dict[str, ExperienceItem], dict[str, ExperienceCommunity], dict[int, ExperienceLayer], set[str]]:
        items = {iid: item for iid, item in self._items.items() if item.level <= level}
        communities = {cid: c for cid, c in self._communities.items() if c.level < level}
        layers = {lvl: layer for lvl, layer in self._layers.items() if lvl < level}
        pending = {iid for iid in self._pending_reroute_item_ids if iid in items}
        return items, communities, layers, pending

    def _commit_layer_unlocked(
        self,
        *,
        level: int,
        communities: list[ExperienceCommunity],
        generated_items: list[ExperienceItem],
        stop_reason: str,
        metadata: dict[str, Any] | None,
        excluded_input_item_ids: list[str] | None = None,
    ) -> None:
        new_items, new_communities, new_layers, new_pending = self._clear_from_level_unlocked(level)
        all_input_item_ids = [iid for iid, item in new_items.items() if item.level == level]
        excluded = set(_uniq(excluded_input_item_ids or []))
        for item_id in excluded:
            if item_id not in new_items:
                raise ValueError(f"excluded input item {item_id!r} is missing")
            if new_items[item_id].level != level:
                raise ValueError(f"excluded input item {item_id!r} is not at level {level}")
        input_item_ids = [iid for iid in all_input_item_ids if iid not in excluded]
        if not input_item_ids:
            raise ValueError(f"no input items at level {level}")
        if not communities:
            raise ValueError("commit_layer requires at least one community")

        community_ids = _uniq([c.community_id for c in communities])
        if len(community_ids) != len(communities):
            raise ValueError("duplicate community ids")
        generated_ids = _uniq([item.item_id for item in generated_items])
        if len(generated_ids) != len(generated_items):
            raise ValueError("duplicate generated item ids")

        covered: set[str] = set()
        generated_by_community: dict[str, list[str]] = {cid: [] for cid in community_ids}
        for community in communities:
            if community.level != level:
                raise ValueError("community level mismatch")
            for item_id in community.member_weights:
                if item_id in excluded:
                    raise ValueError(f"community {community.community_id!r} includes excluded input item {item_id!r}")
                if item_id not in new_items:
                    raise ValueError(f"community {community.community_id!r} references missing item {item_id!r}")
                if new_items[item_id].level != level:
                    raise ValueError("community member level mismatch")
                covered.add(item_id)
        if set(input_item_ids) - covered:
            raise ValueError(f"unassigned input items at level {level}: {sorted(set(input_item_ids) - covered)[:10]}")

        for item in generated_items:
            if item.kind != ITEM_KIND_EXPERIENCE_CARD:
                raise ValueError("generated_items must be experience_card items")
            if item.level != level + 1:
                raise ValueError("generated item level must be level + 1")
            if not item.embedding:
                raise ValueError("generated experience_card items must include embeddings")
            if item.item_id in new_items:
                raise ValueError(f"duplicate generated item_id={item.item_id!r}")
            if len(item.generated_from_community_ids) != 1:
                raise ValueError("generated item must reference exactly one source community")
            _confidence_from_metadata(item.metadata)
            for cid in item.generated_from_community_ids:
                if cid not in generated_by_community:
                    raise ValueError(f"generated item {item.item_id!r} references non-committed community {cid!r}")
                generated_by_community[cid].append(item.item_id)
            new_items[item.item_id] = item
        community_by_id = {community.community_id: community for community in communities}
        for cid, ids in generated_by_community.items():
            if not ids and not _community_allows_no_generated_card(community_by_id[cid]):
                raise ValueError(f"community {cid!r} did not generate an experience_card")

        for community in communities:
            weights = dict(community.member_weights)
            new_communities[community.community_id] = replace(
                community,
                support_mass=_support_mass(new_items, weights),
                generated_item_ids=_uniq(generated_by_community[community.community_id]),
            )
        for cid, ids in generated_by_community.items():
            if ids:
                _redistribute_generated_support_mass(new_items, new_communities[cid], ids)
        layer_metadata = copy.deepcopy(metadata or {})
        if excluded:
            layer_metadata["excluded_input_item_ids"] = sorted(excluded)
        new_layers[level] = ExperienceLayer(
            level=level,
            input_item_ids=input_item_ids,
            community_ids=community_ids,
            generated_item_ids=generated_ids,
            stop_reason=stop_reason,
            metadata=layer_metadata,
        )
        result = _validate_maps(new_items, new_communities, new_layers, pending_reroute_item_ids=new_pending, strict_layers=True)
        if not result["ok"]:
            raise ValueError(f"invalid layer commit: {result}")
        self._items, self._communities, self._layers, self._pending_reroute_item_ids = new_items, new_communities, new_layers, new_pending
        self._index = None

    def _reroute_atomic_unlocked(
        self,
        *,
        level: int,
        assignments: dict[str, dict[str, float]],
        posterior_assignments: dict[str, dict[str, float]],
        new_communities: list[ExperienceCommunity],
    ) -> RerouteResult:
        for community in new_communities:
            if community.community_id in self._communities:
                raise ValueError(f"duplicate new community_id={community.community_id!r}")
            if community.level != level:
                raise ValueError("new community level mismatch")
            self._communities[community.community_id] = community
        if new_communities and level in self._layers:
            layer = self._layers[level]
            self._layers[level] = replace(
                layer,
                community_ids=_uniq([*layer.community_ids, *[community.community_id for community in new_communities]]),
            )

        communities_at_level = [c for c in self._communities.values() if c.level == level]
        if not communities_at_level:
            raise ValueError(f"no communities available at level {level}")
        by_id = {community.community_id: community for community in communities_at_level}
        changed_item_ids = list(assignments)
        for item_id in changed_item_ids:
            if item_id not in self._items:
                raise KeyError(f"unknown item_id={item_id!r}")
            if self._items[item_id].level != level:
                raise ValueError(f"item {item_id!r} is not at level {level}")

        for item_id, weights in assignments.items():
            weights = _positive_weights(weights)
            full_weights = _positive_weights(posterior_assignments.get(item_id, weights))
            for cid in weights:
                if cid not in by_id:
                    raise KeyError(f"assignment references unknown community {cid!r}")
            for cid in full_weights:
                if cid not in by_id:
                    raise KeyError(f"posterior assignment references unknown community {cid!r}")
            for community in communities_at_level:
                selected = dict(community.member_weights)
                posterior = dict(community.posterior_member_weights)
                selected.pop(item_id, None)
                posterior.pop(item_id, None)
                if community.community_id in weights:
                    selected[item_id] = float(weights[community.community_id])
                if community.community_id in full_weights:
                    posterior[item_id] = float(full_weights[community.community_id])
                self._communities[community.community_id] = replace(
                    community,
                    member_weights=selected,
                    posterior_member_weights=posterior,
                    support_mass=_support_mass(self._items, selected),
                    version=community.version + 1,
                )
                by_id[community.community_id] = self._communities[community.community_id]

        affected = sorted({cid for item_id in assignments for cid in assignments[item_id]})
        self._mark_layer_stale(level + 1, "lower_level_reroute")
        self._index = None
        return RerouteResult(level=level, changed_item_ids=changed_item_ids, affected_community_ids=affected, unassigned_item_ids=[], new_community_ids=[c.community_id for c in new_communities])

    def _commit_dynamic_update_atomic_unlocked(self, *, community: ExperienceCommunity, patches: list[ExperienceCardPatch]) -> DynamicPatchResult:
        if community.community_id not in self._communities:
            raise KeyError(f"unknown community_id={community.community_id!r}")
        self._apply_community_to_maps(community)
        return self._apply_patches_atomic_unlocked(source_community_id=community.community_id, patches=patches)

    def _apply_community_to_maps(self, community: ExperienceCommunity) -> None:
        for item_id in community.member_weights:
            if item_id not in self._items:
                raise KeyError(f"community references missing item {item_id!r}")
            if self._items[item_id].level != community.level:
                raise ValueError("community member level mismatch")
        existing = self._communities[community.community_id]
        self._communities[community.community_id] = replace(
            community,
            generated_item_ids=list(existing.generated_item_ids),
            support_mass=_support_mass(self._items, community.member_weights),
            version=existing.version + 1,
        )

    def _apply_patches_atomic_unlocked(self, *, source_community_id: str, patches: list[ExperienceCardPatch]) -> DynamicPatchResult:
        if source_community_id not in self._communities:
            raise KeyError(f"unknown source_community_id={source_community_id!r}")
        community = self._communities[source_community_id]
        existing_ids = list(community.generated_item_ids)
        updates: dict[str, ExperienceCardPatch] = {}
        adds: list[ExperienceCardPatch] = []
        add_ids: set[str] = set()
        for patch in patches:
            if patch.operation == "update":
                if patch.item_id not in existing_ids:
                    raise KeyError(f"update patch references non-generated card {patch.item_id!r}")
                if patch.item_id in updates:
                    raise ValueError(f"duplicate update patch item_id={patch.item_id!r}")
                updates[patch.item_id] = patch
            elif patch.operation == "add":
                if community.level > 0:
                    raise ValueError("add patches are allowed only for L0 raw trajectory communities")
                if patch.item_id in self._items or patch.item_id in add_ids:
                    raise ValueError(f"add patch duplicates existing item_id={patch.item_id!r}")
                add_ids.add(patch.item_id)
                adds.append(patch)
            else:
                raise ValueError(f"unsupported patch operation={patch.operation!r}")

        updated_ids: list[str] = []
        added_ids: list[str] = []
        reroute_ids: list[str] = []
        support_changed: set[str] = set()

        old_support = {iid: self._items[iid].support_mass for iid in existing_ids if iid in self._items}
        for item_id, patch in updates.items():
            old = self._items[item_id]
            embedding_changed = list(old.embedding) != list(patch.embedding)
            text_changed = old.text != patch.text
            self._items[item_id] = old.updated(text=patch.text, embedding=patch.embedding, metadata=patch.metadata)
            updated_ids.append(item_id)
            if embedding_changed or text_changed:
                reroute_ids.append(item_id)

        for patch in adds:
            item = ExperienceItem(
                item_id=patch.item_id,
                level=community.level + 1,
                kind=ITEM_KIND_EXPERIENCE_CARD,
                text=patch.text,
                embedding=patch.embedding,
                support_mass=1.0,
                generated_from_community_ids=[source_community_id],
                metadata=dict(patch.metadata),
            )
            self._items[item.item_id] = item
            existing_ids.append(item.item_id)
            added_ids.append(item.item_id)
            reroute_ids.append(item.item_id)
        if added_ids and (community.level + 1) in self._layers:
            target_layer = self._layers[community.level + 1]
            self._layers[community.level + 1] = replace(target_layer, input_item_ids=_uniq([*target_layer.input_item_ids, *added_ids]))

        # Redistribute source community support_mass across all active generated cards.
        active_ids = [iid for iid in existing_ids if iid in self._items]
        if active_ids or not _community_allows_no_generated_card(community):
            new_support_by_item = _support_mass_allocation(self._items, community, active_ids)
            for item_id, new_support in new_support_by_item.items():
                old = self._items[item_id]
                if abs(old.support_mass - new_support) > 1.0e-9:
                    support_changed.add(item_id)
                    self._items[item_id] = replace(old, support_mass=float(new_support), version=old.version + 1)

        self._communities[source_community_id] = replace(community, generated_item_ids=_uniq(active_ids))
        for item_id in support_changed:
            if item_id not in reroute_ids:
                self._pending_reroute_item_ids.add(item_id)
        changed = sorted(set(updated_ids) | set(added_ids) | support_changed)
        self._index = None
        return DynamicPatchResult(
            source_community_id=source_community_id,
            updated_item_ids=sorted(updated_ids),
            added_item_ids=sorted(added_ids),
            changed_item_ids=changed,
            support_changed_item_ids=sorted(support_changed),
            requires_reroute_item_ids=sorted(set(reroute_ids)),
        )


def _community_allows_no_generated_card(community: ExperienceCommunity) -> bool:
    metadata = dict(community.metadata or {})
    return bool(
        metadata.get("llm_summary_skipped")
        or metadata.get("dynamic_llm_summary_skipped")
        or metadata.get("oversize_singleton")
    )


def _support_mass_allocation(items: dict[str, ExperienceItem], community: ExperienceCommunity, generated_item_ids: list[str]) -> dict[str, float]:
    if not generated_item_ids:
        raise ValueError(f"community {community.community_id!r} has no generated cards for support redistribution")
    confidences = {item_id: _confidence_from_metadata(items[item_id].metadata) for item_id in generated_item_ids}
    total_conf = float(sum(confidences.values()))
    if total_conf <= _EPS or not math.isfinite(total_conf):
        raise ValueError("total confidence must be positive and finite")
    return {item_id: float(community.support_mass) * confidences[item_id] / total_conf for item_id in generated_item_ids}


def _redistribute_generated_support_mass(items: dict[str, ExperienceItem], community: ExperienceCommunity, generated_item_ids: list[str]) -> None:
    for item_id, support in _support_mass_allocation(items, community, generated_item_ids).items():
        items[item_id] = replace(items[item_id], support_mass=float(support))


def _validate_maps(
    items: dict[str, ExperienceItem],
    communities: dict[str, ExperienceCommunity],
    layers: dict[int, ExperienceLayer],
    *,
    pending_reroute_item_ids: set[str],
    strict_layers: bool = True,
    check_support_mass: bool = False,
    require_no_pending_reroute: bool = True,
    require_no_stale_layers: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for item_id, item in items.items():
        if item.item_id != item_id:
            errors.append({"type": "item_id_mismatch", "item_id": item_id})
        for cid in item.generated_from_community_ids:
            if cid not in communities:
                errors.append({"type": "missing_source_community", "item_id": item_id, "community_id": cid})
    for cid, community in communities.items():
        if community.community_id != cid:
            errors.append({"type": "community_id_mismatch", "community_id": cid})
        for item_id in community.member_weights:
            if item_id not in items:
                errors.append({"type": "missing_member", "community_id": cid, "item_id": item_id})
            elif items[item_id].level != community.level:
                errors.append({"type": "member_level_mismatch", "community_id": cid, "item_id": item_id})
        for item_id in community.posterior_member_weights:
            if item_id not in items:
                errors.append({"type": "missing_posterior_member", "community_id": cid, "item_id": item_id})
        if set(community.member_weights) - set(community.posterior_member_weights):
            errors.append({"type": "selected_not_in_full_posterior", "community_id": cid})
        for item_id in community.generated_item_ids:
            if item_id not in items:
                errors.append({"type": "missing_generated_item", "community_id": cid, "item_id": item_id})
        if check_support_mass:
            expected = _support_mass(items, community.member_weights)
            if abs(expected - community.support_mass) > 1.0e-6:
                errors.append({"type": "support_mass_mismatch", "community_id": cid, "expected": expected, "actual": community.support_mass})
    if strict_layers:
        for level, layer in layers.items():
            for item_id in layer.input_item_ids:
                if item_id not in items or items[item_id].level != level:
                    errors.append({"type": "invalid_layer_input", "level": level, "item_id": item_id})
            for cid in layer.community_ids:
                if cid not in communities or communities[cid].level != level:
                    errors.append({"type": "invalid_layer_community", "level": level, "community_id": cid})
            for item_id in layer.generated_item_ids:
                if item_id not in items or items[item_id].level != level + 1:
                    errors.append({"type": "invalid_layer_generated", "level": level, "item_id": item_id})
            if require_no_stale_layers and layer.stale:
                errors.append({"type": "stale_layer", "level": level, "reason": layer.stale_reason})
        for cid, community in communities.items():
            layer = layers.get(community.level)
            if layer is not None and cid not in set(layer.community_ids):
                errors.append({"type": "community_missing_from_layer", "level": community.level, "community_id": cid})
    if require_no_pending_reroute and pending_reroute_item_ids:
        errors.append({"type": "pending_reroute_items", "item_ids": sorted(pending_reroute_item_ids)})
    return {"ok": not errors, "errors": errors}


__all__ = [
    "DynamicPatchResult",
    "ExperienceCardPatch",
    "ExperienceCommunity",
    "ExperienceHierarchyIndex",
    "ExperienceHierarchyState",
    "ExperienceItem",
    "ExperienceLayer",
    "ITEM_KIND_EXPERIENCE_CARD",
    "ITEM_KIND_TRAJECTORY",
    "PatchOperation",
    "RerouteResult",
]

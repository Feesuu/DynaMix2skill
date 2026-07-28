from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .certified_otd import ExperienceAtom

__all__ = [
    "BalancedInsertionResult",
    "BalancedMetricNode",
    "BalancedMetricTreeState",
]


def _unit_vector(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError(f"{field_name} must have positive norm")
    return tuple(value / norm for value in vector)


@dataclass
class BalancedMetricNode:
    node_id: str
    parent_id: str | None
    kind: str
    representative_atom_id: str
    cover_radius: float = 0.0
    atom_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.kind == "leaf"

    @property
    def occupancy(self) -> int:
        return len(self.atom_ids if self.is_leaf else self.child_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "representative_atom_id": self.representative_atom_id,
            "cover_radius": self.cover_radius,
            "atom_ids": list(self.atom_ids),
            "child_ids": list(self.child_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BalancedMetricNode":
        data = dict(payload)
        data["atom_ids"] = [str(value) for value in data.get("atom_ids", [])]
        data["child_ids"] = [str(value) for value in data.get("child_ids", [])]
        return cls(**data)


@dataclass(frozen=True)
class BalancedInsertionResult:
    atom_id: str
    path_before_split: tuple[str, ...]
    affected_node_ids: tuple[str, ...]
    capsule_refresh_node_ids: tuple[str, ...]
    created_node_ids: tuple[str, ...]
    retired_node_ids: tuple[str, ...]
    split_count: int
    root_changed: bool


class BalancedMetricTreeState:
    """Deterministic online B+-style metric tree over experience atoms."""

    def __init__(
        self,
        *,
        max_entries: int = 8,
        dual_view_lambda: float = 0.5,
    ) -> None:
        if int(max_entries) < 4:
            raise ValueError("max_entries must be at least 4")
        if not 0.0 <= float(dual_view_lambda) <= 1.0:
            raise ValueError("dual_view_lambda must be in [0, 1]")
        self.max_entries = int(max_entries)
        self.dual_view_lambda = float(dual_view_lambda)
        self.root_id: str | None = None
        self.atoms: dict[str, ExperienceAtom] = {}
        self.nodes: dict[str, BalancedMetricNode] = {}
        self.next_node_sequence = 0
        self.total_split_count = 0

    @property
    def min_entries(self) -> int:
        return (self.max_entries + 1) // 2

    def distance(self, left_atom_id: str, right_atom_id: str) -> float:
        left = self.atoms[left_atom_id]
        right = self.atoms[right_atom_id]
        trigger_sq = sum(
            (a - b) ** 2
            for a, b in zip(left.trigger_embedding, right.trigger_embedding)
        )
        procedure_sq = sum(
            (a - b) ** 2
            for a, b in zip(left.procedure_embedding, right.procedure_embedding)
        )
        return math.sqrt(
            self.dual_view_lambda * trigger_sq
            + (1.0 - self.dual_view_lambda) * procedure_sq
        )

    def insert(self, atom: ExperienceAtom) -> BalancedInsertionResult:
        if atom.atom_id in self.atoms:
            raise ValueError(f"duplicate atom_id: {atom.atom_id}")
        if self.atoms:
            expected = next(iter(self.atoms.values())).embedding_dimension
            if atom.embedding_dimension != expected:
                raise ValueError(
                    "all atoms must use the same embedding dimension"
                )
        self.atoms[atom.atom_id] = atom
        if self.root_id is None:
            root = self._new_leaf([atom.atom_id], representative=atom.atom_id)
            self.root_id = root.node_id
            return BalancedInsertionResult(
                atom_id=atom.atom_id,
                path_before_split=(root.node_id,),
                affected_node_ids=(root.node_id,),
                capsule_refresh_node_ids=(root.node_id,),
                created_node_ids=(root.node_id,),
                retired_node_ids=(),
                split_count=0,
                root_changed=True,
            )

        path = self._descent_path(atom.atom_id)
        leaf = self.nodes[path[-1]]
        leaf.atom_ids.append(atom.atom_id)
        leaf.atom_ids.sort()
        self._recompute_radius(leaf.node_id)

        affected = set(path)
        capsule_refresh = set(path)
        created: list[str] = []
        retired: list[str] = []
        split_count = 0
        root_changed = False
        current_id = leaf.node_id
        while self.nodes[current_id].occupancy > self.max_entries:
            old_node = self.nodes[current_id]
            old_parent_id = old_node.parent_id
            left, right = self._split_node(current_id)
            created.extend((left.node_id, right.node_id))
            retired.append(current_id)
            split_count += 1
            self.total_split_count += 1
            affected.update((left.node_id, right.node_id))
            capsule_refresh.update((left.node_id, right.node_id))
            if not left.is_leaf:
                affected.update(left.child_ids)
                affected.update(right.child_ids)
            if old_parent_id is None:
                root = self._new_internal(
                    [left.node_id, right.node_id],
                    representative=left.representative_atom_id,
                )
                left.parent_id = root.node_id
                right.parent_id = root.node_id
                self.root_id = root.node_id
                created.append(root.node_id)
                affected.add(root.node_id)
                capsule_refresh.add(root.node_id)
                root_changed = True
                del self.nodes[current_id]
                break

            parent = self.nodes[old_parent_id]
            parent.child_ids = [
                child_id
                for child_id in parent.child_ids
                if child_id != current_id
            ]
            parent.child_ids.extend((left.node_id, right.node_id))
            parent.child_ids.sort()
            left.parent_id = parent.node_id
            right.parent_id = parent.node_id
            del self.nodes[current_id]
            self._recompute_radius(parent.node_id)
            affected.add(parent.node_id)
            capsule_refresh.add(parent.node_id)
            current_id = parent.node_id

        self._recompute_ancestors(current_id, affected)
        return BalancedInsertionResult(
            atom_id=atom.atom_id,
            path_before_split=tuple(path),
            affected_node_ids=tuple(sorted(affected)),
            capsule_refresh_node_ids=tuple(sorted(capsule_refresh)),
            created_node_ids=tuple(created),
            retired_node_ids=tuple(retired),
            split_count=split_count,
            root_changed=root_changed,
        )

    def nearest_atom_ids(
        self,
        *,
        trigger_embedding: Sequence[float],
        procedure_embedding: Sequence[float],
        k: int,
    ) -> tuple[str, ...]:
        if self.root_id is None or k <= 0:
            return ()
        trigger = _unit_vector(
            trigger_embedding,
            field_name="trigger_embedding",
        )
        procedure = _unit_vector(
            procedure_embedding,
            field_name="procedure_embedding",
        )
        expected = next(iter(self.atoms.values())).embedding_dimension
        if len(trigger) != expected or len(procedure) != expected:
            raise ValueError("query embedding dimension does not match the tree")

        queue: list[tuple[float, str]] = []
        root = self.nodes[self.root_id]
        heapq.heappush(
            queue,
            (
                max(
                    0.0,
                    self._query_distance(
                        trigger,
                        procedure,
                        root.representative_atom_id,
                    )
                    - root.cover_radius,
                ),
                root.node_id,
            ),
        )
        best: list[tuple[float, str]] = []
        worst = math.inf
        while queue:
            lower_bound, node_id = heapq.heappop(queue)
            tolerance = 1e-12 * max(1.0, abs(lower_bound), abs(worst))
            if len(best) >= k and lower_bound > worst + tolerance:
                break
            node = self.nodes[node_id]
            if node.is_leaf:
                for atom_id in node.atom_ids:
                    distance = self._query_distance(
                        trigger,
                        procedure,
                        atom_id,
                    )
                    best.append((distance, atom_id))
                best.sort(key=lambda item: (item[0], item[1]))
                del best[k:]
                if len(best) >= k:
                    worst = best[-1][0]
                continue
            for child_id in node.child_ids:
                child = self.nodes[child_id]
                distance = self._query_distance(
                    trigger,
                    procedure,
                    child.representative_atom_id,
                )
                child_lower_bound = max(0.0, distance - child.cover_radius)
                tolerance = 1e-12 * max(
                    1.0,
                    abs(child_lower_bound),
                    abs(worst),
                )
                if len(best) < k or child_lower_bound <= worst + tolerance:
                    heapq.heappush(queue, (child_lower_bound, child_id))
        return tuple(atom_id for _, atom_id in best)

    def descendant_atom_ids(self, node_id: str) -> tuple[str, ...]:
        node = self.nodes[node_id]
        if node.is_leaf:
            return tuple(node.atom_ids)
        atom_ids: list[str] = []
        for child_id in node.child_ids:
            atom_ids.extend(self.descendant_atom_ids(child_id))
        return tuple(sorted(atom_ids))

    def leaf_depths(self) -> dict[str, int]:
        if self.root_id is None:
            return {}
        depths: dict[str, int] = {}
        stack = [(self.root_id, 0)]
        while stack:
            node_id, depth = stack.pop()
            node = self.nodes[node_id]
            if node.is_leaf:
                depths[node_id] = depth
            else:
                stack.extend((child_id, depth + 1) for child_id in node.child_ids)
        return depths

    def height_bound(self) -> int:
        atom_count = len(self.atoms)
        if atom_count <= self.max_entries:
            return 0
        ratio = max(1.0, atom_count / (2.0 * self.min_entries))
        return 1 + math.ceil(math.log(ratio, self.min_entries))

    def structural_audit(self) -> dict[str, Any]:
        self.validate()
        leaf_depths = self.leaf_depths()
        occupancies = [node.occupancy for node in self.nodes.values()]
        leaf_occupancies = [
            node.occupancy for node in self.nodes.values() if node.is_leaf
        ]
        internal_occupancies = [
            node.occupancy for node in self.nodes.values() if not node.is_leaf
        ]
        height = max(leaf_depths.values(), default=0)
        exact_radius_count = 0
        for node_id, node in self.nodes.items():
            exact_radius = max(
                (
                    self.distance(node.representative_atom_id, atom_id)
                    for atom_id in self.descendant_atom_ids(node_id)
                ),
                default=0.0,
            )
            if math.isclose(
                node.cover_radius,
                exact_radius,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                exact_radius_count += 1
        return {
            "atom_count": len(self.atoms),
            "node_count": len(self.nodes),
            "leaf_count": len(leaf_depths),
            "internal_count": len(self.nodes) - len(leaf_depths),
            "max_entries": self.max_entries,
            "min_entries": self.min_entries,
            "height": height,
            "height_bound": self.height_bound(),
            "height_within_bound": height <= self.height_bound(),
            "all_leaves_same_depth": len(set(leaf_depths.values())) <= 1,
            "occupancy_min": min(occupancies, default=0),
            "occupancy_max": max(occupancies, default=0),
            "leaf_occupancies": leaf_occupancies,
            "internal_occupancies": internal_occupancies,
            "split_count": self.total_split_count,
            "cover_radius_certified_upper_bound": True,
            "exact_cover_radius_node_count": exact_radius_count,
            "inexact_cover_radius_node_count": (
                len(self.nodes) - exact_radius_count
            ),
            "atoms_only_in_leaves": True,
            "unique_atom_placement": True,
        }

    def validate(self) -> None:
        if self.root_id is None:
            if self.atoms or self.nodes:
                raise ValueError("empty tree must not contain atoms or nodes")
            return
        if self.root_id not in self.nodes:
            raise ValueError("root_id does not reference a node")
        root = self.nodes[self.root_id]
        if root.parent_id is not None:
            raise ValueError("root must not have a parent")

        visited: set[str] = set()
        placed_atoms: list[str] = []
        leaf_depths: set[int] = set()

        def visit(node_id: str, depth: int) -> None:
            if node_id in visited:
                raise ValueError("tree contains a cycle or repeated child")
            visited.add(node_id)
            node = self.nodes[node_id]
            if node.kind not in {"leaf", "internal"}:
                raise ValueError(f"invalid node kind: {node.kind}")
            if node.is_leaf:
                if node.child_ids:
                    raise ValueError("leaf nodes must not contain child_ids")
                if not node.atom_ids:
                    raise ValueError("leaf nodes must not be empty")
                placed_atoms.extend(node.atom_ids)
                leaf_depths.add(depth)
            else:
                if node.atom_ids:
                    raise ValueError("internal nodes must not contain atom_ids")
                if len(node.child_ids) < 2:
                    raise ValueError("internal nodes must have at least two children")
                for child_id in node.child_ids:
                    if child_id not in self.nodes:
                        raise ValueError(f"missing child node: {child_id}")
                    if self.nodes[child_id].parent_id != node_id:
                        raise ValueError("child parent link is inconsistent")
                    visit(child_id, depth + 1)

            if node_id != self.root_id:
                if not self.min_entries <= node.occupancy <= self.max_entries:
                    raise ValueError(
                        f"non-root occupancy out of bounds for {node_id}: "
                        f"{node.occupancy}"
                    )
            elif node.occupancy > self.max_entries:
                raise ValueError("root occupancy exceeds max_entries")

            descendants = self.descendant_atom_ids(node_id)
            if node.representative_atom_id not in descendants:
                raise ValueError("node representative must be a descendant atom")
            expected_radius = self._certified_radius(node_id)
            if not math.isclose(
                node.cover_radius,
                expected_radius,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"invalid certified cover radius for {node_id}: "
                    f"{node.cover_radius} != {expected_radius}"
                )
            exact_radius = max(
                (
                    self.distance(node.representative_atom_id, atom_id)
                    for atom_id in descendants
                ),
                default=0.0,
            )
            if node.cover_radius + 1e-9 < exact_radius:
                raise ValueError(
                    f"cover radius is not an upper bound for {node_id}: "
                    f"{node.cover_radius} < {exact_radius}"
                )

        visit(self.root_id, 0)
        if visited != set(self.nodes):
            raise ValueError("tree contains unreachable nodes")
        if len(leaf_depths) > 1:
            raise ValueError("all leaves must have the same depth")
        if len(placed_atoms) != len(set(placed_atoms)):
            raise ValueError("an atom appears in more than one leaf")
        if set(placed_atoms) != set(self.atoms):
            raise ValueError("tree atom placement does not match atom registry")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "tree_policy": "evidence_balanced_skill_tree",
            "max_entries": self.max_entries,
            "min_entries": self.min_entries,
            "dual_view_lambda": self.dual_view_lambda,
            "root_id": self.root_id,
            "next_node_sequence": self.next_node_sequence,
            "total_split_count": self.total_split_count,
            "atoms": {
                atom_id: atom.to_dict()
                for atom_id, atom in sorted(self.atoms.items())
            },
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in sorted(self.nodes.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BalancedMetricTreeState":
        state = cls(
            max_entries=int(payload["max_entries"]),
            dual_view_lambda=float(payload["dual_view_lambda"]),
        )
        state.root_id = (
            str(payload["root_id"]) if payload.get("root_id") is not None else None
        )
        state.next_node_sequence = int(payload.get("next_node_sequence", 0))
        state.total_split_count = int(payload.get("total_split_count", 0))
        state.atoms = {
            str(atom_id): ExperienceAtom.from_dict(atom_payload)
            for atom_id, atom_payload in dict(payload.get("atoms", {})).items()
        }
        state.nodes = {
            str(node_id): BalancedMetricNode.from_dict(node_payload)
            for node_id, node_payload in dict(payload.get("nodes", {})).items()
        }
        state.validate()
        return state

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _descent_path(self, atom_id: str) -> list[str]:
        if self.root_id is None:
            raise RuntimeError("cannot descend an empty tree")
        path = [self.root_id]
        while not self.nodes[path[-1]].is_leaf:
            node = self.nodes[path[-1]]
            ranked: list[tuple[float, float, str]] = []
            for child_id in node.child_ids:
                child = self.nodes[child_id]
                distance = self.distance(
                    child.representative_atom_id,
                    atom_id,
                )
                enlargement = max(0.0, distance - child.cover_radius)
                ranked.append((enlargement, distance, child_id))
            path.append(min(ranked)[2])
        return path

    def _split_node(
        self,
        node_id: str,
    ) -> tuple[BalancedMetricNode, BalancedMetricNode]:
        node = self.nodes[node_id]
        entries = list(node.atom_ids if node.is_leaf else node.child_ids)
        representatives = {
            entry_id: (
                entry_id
                if node.is_leaf
                else self.nodes[entry_id].representative_atom_id
            )
            for entry_id in entries
        }
        left_pivot, right_pivot = self._farthest_pair(entries, representatives)
        left_ids, right_ids = self._balanced_partition(
            entries,
            representatives,
            left_pivot,
            right_pivot,
        )
        left_rep = representatives[left_pivot]
        right_rep = representatives[right_pivot]
        if node.is_leaf:
            left = self._new_leaf(
                left_ids,
                representative=left_rep,
                parent_id=node.parent_id,
            )
            right = self._new_leaf(
                right_ids,
                representative=right_rep,
                parent_id=node.parent_id,
            )
        else:
            left = self._new_internal(
                left_ids,
                representative=left_rep,
                parent_id=node.parent_id,
            )
            right = self._new_internal(
                right_ids,
                representative=right_rep,
                parent_id=node.parent_id,
            )
            for child_id in left.child_ids:
                self.nodes[child_id].parent_id = left.node_id
            for child_id in right.child_ids:
                self.nodes[child_id].parent_id = right.node_id
        return left, right

    def _farthest_pair(
        self,
        entries: Sequence[str],
        representatives: Mapping[str, str],
    ) -> tuple[str, str]:
        ordered = sorted(entries)
        best_pair = (ordered[0], ordered[1])
        best_distance = -1.0
        for index, left_id in enumerate(ordered):
            for right_id in ordered[index + 1 :]:
                distance = self.distance(
                    representatives[left_id],
                    representatives[right_id],
                )
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (left_id, right_id)
        return best_pair

    def _balanced_partition(
        self,
        entries: Sequence[str],
        representatives: Mapping[str, str],
        left_pivot: str,
        right_pivot: str,
    ) -> tuple[list[str], list[str]]:
        remaining = [
            entry_id
            for entry_id in entries
            if entry_id not in {left_pivot, right_pivot}
        ]
        remaining.sort(
            key=lambda entry_id: (
                self.distance(
                    representatives[entry_id],
                    representatives[left_pivot],
                )
                - self.distance(
                    representatives[entry_id],
                    representatives[right_pivot],
                ),
                entry_id,
            )
        )
        target_left = len(entries) // 2
        left = [left_pivot, *remaining[: target_left - 1]]
        right = [right_pivot, *remaining[target_left - 1 :]]
        left.sort()
        right.sort()
        if len(left) < self.min_entries or len(right) < self.min_entries:
            raise RuntimeError("balanced split failed the minimum occupancy bound")
        return left, right

    def _new_leaf(
        self,
        atom_ids: Iterable[str],
        *,
        representative: str,
        parent_id: str | None = None,
    ) -> BalancedMetricNode:
        node = BalancedMetricNode(
            node_id=self._new_node_id(),
            parent_id=parent_id,
            kind="leaf",
            representative_atom_id=representative,
            atom_ids=sorted(atom_ids),
        )
        self.nodes[node.node_id] = node
        self._recompute_radius(node.node_id)
        return node

    def _new_internal(
        self,
        child_ids: Iterable[str],
        *,
        representative: str,
        parent_id: str | None = None,
    ) -> BalancedMetricNode:
        node = BalancedMetricNode(
            node_id=self._new_node_id(),
            parent_id=parent_id,
            kind="internal",
            representative_atom_id=representative,
            child_ids=sorted(child_ids),
        )
        self.nodes[node.node_id] = node
        self._recompute_radius(node.node_id)
        return node

    def _new_node_id(self) -> str:
        node_id = f"ebst_node_{self.next_node_sequence:08d}"
        self.next_node_sequence += 1
        return node_id

    def _recompute_radius(self, node_id: str) -> None:
        self.nodes[node_id].cover_radius = self._certified_radius(node_id)

    def _certified_radius(self, node_id: str) -> float:
        node = self.nodes[node_id]
        if node.is_leaf:
            return max(
                (
                    self.distance(node.representative_atom_id, atom_id)
                    for atom_id in node.atom_ids
                ),
                default=0.0,
            )
        return max(
            (
                self.distance(
                    node.representative_atom_id,
                    self.nodes[child_id].representative_atom_id,
                )
                + self.nodes[child_id].cover_radius
                for child_id in node.child_ids
            ),
            default=0.0,
        )

    def _recompute_ancestors(
        self,
        node_id: str,
        affected: set[str],
    ) -> None:
        current_id: str | None = node_id
        while current_id is not None and current_id in self.nodes:
            self._recompute_radius(current_id)
            affected.add(current_id)
            current_id = self.nodes[current_id].parent_id

    def _query_distance(
        self,
        trigger: Sequence[float],
        procedure: Sequence[float],
        atom_id: str,
    ) -> float:
        atom = self.atoms[atom_id]
        trigger_sq = sum(
            (a - b) ** 2 for a, b in zip(trigger, atom.trigger_embedding)
        )
        procedure_sq = sum(
            (a - b) ** 2 for a, b in zip(procedure, atom.procedure_embedding)
        )
        return math.sqrt(
            self.dual_view_lambda * trigger_sq
            + (1.0 - self.dual_view_lambda) * procedure_sq
        )

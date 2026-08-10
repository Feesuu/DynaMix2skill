from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .balanced_metric_tree import (
    BalancedInsertionResult,
    BalancedMetricNode,
    BalancedMetricTreeState,
)

__all__ = [
    "ATOM_ENTRY_PREFIX",
    "ContractAtom",
    "ContractCutResult",
    "ContractCutTreeState",
    "ExactContractCutOptimizer",
]

ATOM_ENTRY_PREFIX = "contract_atom_entry::"


def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("boundary_embedding must be a non-empty finite vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError("boundary_embedding must have positive norm")
    if math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return vector
    return tuple(value / norm for value in vector)


@dataclass(frozen=True)
class ContractAtom:
    atom_id: str
    source_item_id: str
    trigger: str
    scope: str
    decision: str
    invariant: str
    verification: str
    failure_mode: str
    boundary_embedding: tuple[float, ...]
    provenance: dict[str, Any]
    weight: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "atom_id",
            "source_item_id",
            "trigger",
            "scope",
            "decision",
            "invariant",
            "verification",
            "failure_mode",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Contract Atom field is empty: {field_name}")
        if not math.isclose(float(self.weight), 1.0):
            raise ValueError("Contract-Cut EBST v1 requires every Atom weight to be 1")
        object.__setattr__(
            self,
            "boundary_embedding",
            _unit_vector(self.boundary_embedding),
        )

    @property
    def embedding_dimension(self) -> int:
        return len(self.boundary_embedding)

    @property
    def boundary_text(self) -> str:
        return "\n".join(
            (
                f"Applicability: {self.trigger}",
                f"Scope: {self.scope}",
                f"Success condition: {self.verification}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "source_item_id": self.source_item_id,
            "trigger": self.trigger,
            "scope": self.scope,
            "decision": self.decision,
            "invariant": self.invariant,
            "verification": self.verification,
            "failure_mode": self.failure_mode,
            "boundary_embedding": list(self.boundary_embedding),
            "provenance": dict(self.provenance),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractAtom":
        data = dict(payload)
        data["boundary_embedding"] = tuple(data["boundary_embedding"])
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


@dataclass(frozen=True)
class ContractCutResult:
    selected_node_ids: tuple[str, ...]
    objective: float
    distortion: float
    opening_cost: float
    covered_atom_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DpValue:
    objective: float
    selected_node_ids: tuple[str, ...]


class ContractCutTreeState(BalancedMetricTreeState):
    """Single-parent EBST with exact local min-max overflow splits."""

    TREE_POLICY = "contract_cut_ebst"

    def __init__(self, *, max_entries: int = 8) -> None:
        if int(max_entries) != 8:
            raise ValueError("Contract-Cut EBST v1 fixes max_entries=8")
        super().__init__(max_entries=8, dual_view_lambda=1.0)
        self.atoms: dict[str, ContractAtom] = {}
        self.split_events: list[dict[str, Any]] = []

    def distance(self, left_atom_id: str, right_atom_id: str) -> float:
        left = self.atoms[left_atom_id].boundary_embedding
        right = self.atoms[right_atom_id].boundary_embedding
        if len(left) != len(right):
            raise ValueError("Contract Atom embedding dimensions do not match")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def insert(self, atom: ContractAtom) -> BalancedInsertionResult:
        return super().insert(atom)  # type: ignore[arg-type]

    def node_distortion(self, node_id: str) -> float:
        if self.is_atom_entry(node_id):
            return 0.0
        node = self.nodes[node_id]
        return sum(
            self.atoms[atom_id].weight
            * self.distance(atom_id, node.representative_atom_id) ** 2
            for atom_id in self.descendant_atom_ids(node_id)
        )

    @staticmethod
    def atom_entry_id(atom_id: str) -> str:
        return ATOM_ENTRY_PREFIX + atom_id

    @staticmethod
    def is_atom_entry(candidate_id: str) -> bool:
        return str(candidate_id).startswith(ATOM_ENTRY_PREFIX)

    @staticmethod
    def atom_id_from_entry(candidate_id: str) -> str:
        if not ContractCutTreeState.is_atom_entry(candidate_id):
            raise ValueError(f"not an Atom-entry terminal: {candidate_id}")
        atom_id = str(candidate_id)[len(ATOM_ENTRY_PREFIX) :]
        if not atom_id:
            raise ValueError("Atom-entry terminal has an empty Atom ID")
        return atom_id

    def candidate_atom_ids(self, candidate_id: str) -> tuple[str, ...]:
        if self.is_atom_entry(candidate_id):
            atom_id = self.atom_id_from_entry(candidate_id)
            if atom_id not in self.atoms:
                raise ValueError(f"Atom-entry terminal is not in the tree: {atom_id}")
            return (atom_id,)
        return self.descendant_atom_ids(candidate_id)

    def candidate_parent_id(self, candidate_id: str) -> str | None:
        if not self.is_atom_entry(candidate_id):
            return self.nodes[candidate_id].parent_id
        atom_id = self.atom_id_from_entry(candidate_id)
        return next(
            node_id
            for node_id, node in self.nodes.items()
            if node.is_leaf and atom_id in node.atom_ids
        )

    def structural_audit(self) -> dict[str, Any]:
        audit = super().structural_audit()
        audit.update(
            {
                "tree_policy": self.TREE_POLICY,
                "exact_local_split": True,
                "split_partition_count": 126,
                "all_atom_weights_one": all(
                    math.isclose(atom.weight, 1.0)
                    for atom in self.atoms.values()
                ),
                "single_boundary_embedding": True,
                "cut_domain": "ebst_nodes_plus_leaf_atom_entries",
                "atom_entry_terminal_count": len(self.atoms),
                "optimal_cut_guarantee": "exact_on_augmented_candidate_tree",
                "split_events": list(self.split_events),
            }
        )
        return audit

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": "contract_cut_ebst_tree_v1",
            "tree_policy": self.TREE_POLICY,
            "max_entries": self.max_entries,
            "min_entries": self.min_entries,
            "root_id": self.root_id,
            "next_node_sequence": self.next_node_sequence,
            "total_split_count": self.total_split_count,
            "split_events": list(self.split_events),
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractCutTreeState":
        if payload.get("format") != "contract_cut_ebst_tree_v1":
            raise ValueError("unsupported Contract-Cut EBST tree format")
        state = cls(max_entries=int(payload["max_entries"]))
        state.root_id = (
            str(payload["root_id"])
            if payload.get("root_id") is not None
            else None
        )
        state.next_node_sequence = int(payload.get("next_node_sequence", 0))
        state.total_split_count = int(payload.get("total_split_count", 0))
        state.split_events = [
            dict(event) for event in payload.get("split_events", [])
        ]
        state.atoms = {
            str(atom_id): ContractAtom.from_dict(atom_payload)
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

    def _split_node(
        self,
        node_id: str,
    ) -> tuple[BalancedMetricNode, BalancedMetricNode]:
        node = self.nodes[node_id]
        entries = sorted(node.atom_ids if node.is_leaf else node.child_ids)
        if len(entries) != self.max_entries + 1:
            raise RuntimeError(
                "exact Contract-Cut split requires exactly nine entries"
            )
        representatives = {
            entry_id: (
                entry_id
                if node.is_leaf
                else self.nodes[entry_id].representative_atom_id
            )
            for entry_id in entries
        }
        radii = {
            entry_id: 0.0 if node.is_leaf else self.nodes[entry_id].cover_radius
            for entry_id in entries
        }

        best: tuple[
            tuple[float, float, tuple[str, ...], tuple[str, ...]],
            tuple[str, ...],
            tuple[str, ...],
            str,
            str,
            float,
            float,
        ] | None = None
        all_entries = set(entries)
        for left_tuple in itertools.combinations(entries, self.min_entries):
            left = tuple(sorted(left_tuple))
            right = tuple(sorted(all_entries - set(left)))
            left_radius, left_rep_entry = self._entry_cover(
                left,
                representatives=representatives,
                radii=radii,
            )
            right_radius, right_rep_entry = self._entry_cover(
                right,
                representatives=representatives,
                radii=radii,
            )
            key = (
                max(left_radius, right_radius),
                left_radius + right_radius,
                left,
                right,
            )
            candidate = (
                key,
                left,
                right,
                left_rep_entry,
                right_rep_entry,
                left_radius,
                right_radius,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise RuntimeError("exact Contract-Cut split found no partition")
        (
            objective,
            left_ids,
            right_ids,
            left_rep_entry,
            right_rep_entry,
            left_radius,
            right_radius,
        ) = best
        left_rep = representatives[left_rep_entry]
        right_rep = representatives[right_rep_entry]

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

        # Preserve the representatives selected by the exact split objective.
        left.representative_atom_id = left_rep
        right.representative_atom_id = right_rep
        left.cover_radius = self._exact_radius(left.node_id)
        right.cover_radius = self._exact_radius(right.node_id)
        self.split_events.append(
            {
                "retired_node_id": node_id,
                "kind": node.kind,
                "entry_count": len(entries),
                "enumerated_partition_count": math.comb(
                    len(entries), self.min_entries
                ),
                "left_entries": list(left_ids),
                "right_entries": list(right_ids),
                "left_objective_radius": left_radius,
                "right_objective_radius": right_radius,
                "max_objective_radius": objective[0],
                "sum_objective_radius": objective[1],
            }
        )
        return left, right

    def _entry_cover(
        self,
        entries: Sequence[str],
        *,
        representatives: Mapping[str, str],
        radii: Mapping[str, float],
    ) -> tuple[float, str]:
        ranked: list[tuple[float, str]] = []
        for candidate in entries:
            candidate_atom = representatives[candidate]
            radius = max(
                self.distance(candidate_atom, representatives[entry])
                + radii[entry]
                for entry in entries
            )
            ranked.append((radius, candidate))
        return min(ranked)

    def _new_node_id(self) -> str:
        node_id = f"cceb_node_{self.next_node_sequence:08d}"
        self.next_node_sequence += 1
        return node_id

    def _recompute_radius(self, node_id: str) -> None:
        node = self.nodes[node_id]
        candidates = (
            tuple(node.atom_ids)
            if node.is_leaf
            else tuple(
                self.nodes[child_id].representative_atom_id
                for child_id in node.child_ids
            )
        )
        descendants = self.descendant_atom_ids(node_id)
        if not candidates or not descendants:
            node.cover_radius = 0.0
            return
        ranked = [
            (
                max(
                    self.distance(candidate, atom_id)
                    for atom_id in descendants
                ),
                candidate,
            )
            for candidate in candidates
        ]
        radius, representative = min(ranked)
        node.representative_atom_id = representative
        node.cover_radius = radius

    def _certified_radius(self, node_id: str) -> float:
        return self._exact_radius(node_id)

    def _exact_radius(self, node_id: str) -> float:
        node = self.nodes[node_id]
        return max(
            (
                self.distance(node.representative_atom_id, atom_id)
                for atom_id in self.descendant_atom_ids(node_id)
            ),
            default=0.0,
        )


class ExactContractCutOptimizer:
    """Exact DP over EBST nodes plus per-leaf Atom-entry terminals."""

    def __init__(self, state: ContractCutTreeState, *, beta: float) -> None:
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("beta must be a positive finite value")
        self.state = state
        self.beta = float(beta)

    def solve(
        self,
        *,
        infeasible_node_ids: Sequence[str] = (),
    ) -> ContractCutResult:
        if self.state.root_id is None:
            return ContractCutResult((), 0.0, 0.0, 0.0, ())
        blocked = set(infeasible_node_ids)
        unknown = blocked - set(self.state.nodes)
        if unknown:
            raise ValueError(f"infeasible cut nodes are not in the tree: {unknown}")
        memo: dict[str, _DpValue] = {}

        def visit(node_id: str) -> _DpValue:
            if node_id in memo:
                return memo[node_id]
            node = self.state.nodes[node_id]
            one_cost = self.state.node_distortion(node_id) + self.beta
            one = _DpValue(one_cost, (node_id,))
            if node.is_leaf:
                atom_terminals = _DpValue(
                    self.beta * len(node.atom_ids),
                    tuple(
                        self.state.atom_entry_id(atom_id)
                        for atom_id in sorted(node.atom_ids)
                    ),
                )
                if node_id in blocked:
                    value = atom_terminals
                else:
                    value = min(
                        (one, atom_terminals),
                        key=lambda candidate: (
                            candidate.objective,
                            len(candidate.selected_node_ids),
                            candidate.selected_node_ids,
                        ),
                    )
            else:
                children = [visit(child_id) for child_id in node.child_ids]
                split = _DpValue(
                    sum(child.objective for child in children),
                    tuple(
                        sorted(
                            node_id
                            for child in children
                            for node_id in child.selected_node_ids
                        )
                    ),
                )
                if node_id in blocked:
                    value = split
                else:
                    value = min(
                        (one, split),
                        key=lambda candidate: (
                            candidate.objective,
                            len(candidate.selected_node_ids),
                            candidate.selected_node_ids,
                        ),
                    )
            memo[node_id] = value
            return value

        optimum = visit(self.state.root_id)
        selected = tuple(sorted(optimum.selected_node_ids))
        covered = tuple(
            sorted(
                atom_id
                for node_id in selected
                for atom_id in self.state.candidate_atom_ids(node_id)
            )
        )
        if len(covered) != len(set(covered)) or set(covered) != set(
            self.state.atoms
        ):
            raise RuntimeError("Contract cut does not cover every Atom exactly once")
        self._require_antichain(selected)
        distortion = sum(
            self.state.node_distortion(node_id) for node_id in selected
        )
        opening_cost = self.beta * len(selected)
        if not math.isclose(
            optimum.objective,
            distortion + opening_cost,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Contract cut objective accounting is inconsistent")
        return ContractCutResult(
            selected_node_ids=selected,
            objective=optimum.objective,
            distortion=distortion,
            opening_cost=opening_cost,
            covered_atom_ids=covered,
        )

    def _require_antichain(self, selected_node_ids: Sequence[str]) -> None:
        selected = set(selected_node_ids)
        for node_id in selected:
            parent_id = self.state.candidate_parent_id(node_id)
            while parent_id is not None:
                if parent_id in selected:
                    raise RuntimeError("Contract cut contains ancestor/descendant nodes")
                parent_id = self.state.nodes[parent_id].parent_id

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_SIMILARITY_NUMERIC_TOLERANCE = 1.0e-10
_UnconstrainedState = tuple[float, int, tuple[str, ...]]


def _unit_vector(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{field_name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} contains a non-finite value")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{field_name} must have non-zero norm")
    if abs(norm - 1.0) <= 8.0 * np.finfo(float).eps:
        return tuple(float(value) for value in vector)
    return tuple(float(value) for value in vector / norm)


def _canonical_similarity_average(value: float) -> float:
    """Bound only floating-point drift around the exact shifted-cosine range."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("similarity must be finite")
    if value < -_SIMILARITY_NUMERIC_TOLERANCE or value > (
        1.0 + _SIMILARITY_NUMERIC_TOLERANCE
    ):
        raise ValueError(f"shifted-cosine similarity outside [0, 1]: {value}")
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _required_text(value: str, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperienceAtom:
    atom_id: str
    source_item_id: str
    trigger: str
    scope: str
    decision: str
    invariant: str
    verification: str
    failure_mode: str
    evidence_type: str
    reliability: float
    trigger_embedding: tuple[float, ...]
    procedure_embedding: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_id", _required_text(self.atom_id, field_name="atom_id"))
        object.__setattr__(
            self,
            "source_item_id",
            _required_text(self.source_item_id, field_name="source_item_id"),
        )
        for name in (
            "trigger",
            "scope",
            "decision",
            "invariant",
            "verification",
            "failure_mode",
            "evidence_type",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), field_name=name))
        reliability = float(self.reliability)
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be finite and in [0, 1]")
        object.__setattr__(self, "reliability", reliability)
        trigger = _unit_vector(self.trigger_embedding, field_name="trigger_embedding")
        procedure = _unit_vector(self.procedure_embedding, field_name="procedure_embedding")
        if len(trigger) != len(procedure):
            raise ValueError("trigger and procedure embeddings must have the same dimension")
        object.__setattr__(self, "trigger_embedding", trigger)
        object.__setattr__(self, "procedure_embedding", procedure)
        metadata = json.loads(
            json.dumps(
                dict(self.metadata),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    @property
    def embedding_dimension(self) -> int:
        return len(self.trigger_embedding)

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
            "evidence_type": self.evidence_type,
            "reliability": self.reliability,
            "trigger_embedding": list(self.trigger_embedding),
            "procedure_embedding": list(self.procedure_embedding),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceAtom":
        data = dict(payload)
        data["trigger_embedding"] = tuple(data["trigger_embedding"])
        data["procedure_embedding"] = tuple(data["procedure_embedding"])
        return cls(**data)


@dataclass
class OtdNode:
    node_id: str
    parent_id: str | None
    left_id: str | None
    right_id: str | None
    atom_id: str | None
    leaf_count: int
    sum_trigger: tuple[float, ...]
    sum_procedure: tuple[float, ...]
    sum_trigger_squared_norms: float
    sum_procedure_squared_norms: float
    within_similarity: float
    min_atom_id: str
    skill: dict[str, Any] = field(default_factory=dict)
    retrievable: bool = False
    validation_mode: str = "structural_only"
    structural_certificate: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.atom_id is not None

    @property
    def child_ids(self) -> tuple[str, ...]:
        if self.is_leaf:
            return ()
        if self.left_id is None or self.right_id is None:
            raise ValueError(f"internal node {self.node_id} must have two children")
        return (self.left_id, self.right_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OtdNode":
        data = dict(payload)
        data["sum_trigger"] = tuple(data["sum_trigger"])
        data["sum_procedure"] = tuple(data["sum_procedure"])
        return cls(**data)


@dataclass(frozen=True)
class OtdDecision:
    node_id: str
    action: str
    within_similarity: float
    cross_similarity: float
    within_similarity_sum: float
    cross_similarity_sum: float
    left_cross_similarity: float | None = None
    right_cross_similarity: float | None = None
    selected_child_id: str | None = None


@dataclass(frozen=True)
class OtdInsertionResult:
    atom_id: str
    leaf_node_id: str
    root_node_id: str
    created_node_ids: tuple[str, ...]
    changed_internal_node_ids: tuple[str, ...]
    decisions: tuple[OtdDecision, ...]


class OtdTreeState:
    """Exact single-parent Online Top-Down tree over fixed dual-view atoms."""

    def __init__(
        self,
        *,
        dual_view_lambda: float = 0.5,
        tie_epsilon: float = 0.0,
    ) -> None:
        dual_view_lambda = float(dual_view_lambda)
        tie_epsilon = float(tie_epsilon)
        if not 0.0 <= dual_view_lambda <= 1.0:
            raise ValueError("dual_view_lambda must be in [0, 1]")
        if not math.isfinite(tie_epsilon) or tie_epsilon < 0.0:
            raise ValueError("tie_epsilon must be finite and non-negative")
        self._dual_view_lambda = dual_view_lambda
        self._tie_epsilon = tie_epsilon
        self.root_node_id: str | None = None
        self.nodes: dict[str, OtdNode] = {}
        self.atoms: dict[str, ExperienceAtom] = {}
        self.insertion_order: list[str] = []
        self._next_internal_index = 1
        self._embedding_dimension: int | None = None

    @property
    def dual_view_lambda(self) -> float:
        return self._dual_view_lambda

    @property
    def tie_epsilon(self) -> float:
        return self._tie_epsilon

    def insert(self, atom: ExperienceAtom) -> OtdInsertionResult:
        if atom.atom_id in self.atoms:
            raise ValueError(f"duplicate atom_id: {atom.atom_id}")
        if self._embedding_dimension is None:
            self._embedding_dimension = atom.embedding_dimension
        elif atom.embedding_dimension != self._embedding_dimension:
            raise ValueError(
                f"atom embedding dimension {atom.embedding_dimension} does not match "
                f"tree dimension {self._embedding_dimension}"
            )

        leaf = self._leaf_node(atom)
        if leaf.node_id in self.nodes:
            raise ValueError(f"leaf node id collision: {leaf.node_id}")
        self.atoms[atom.atom_id] = atom
        self.nodes[leaf.node_id] = leaf
        self.insertion_order.append(atom.atom_id)

        if self.root_node_id is None:
            self.root_node_id = leaf.node_id
            return OtdInsertionResult(
                atom_id=atom.atom_id,
                leaf_node_id=leaf.node_id,
                root_node_id=leaf.node_id,
                created_node_ids=(leaf.node_id,),
                changed_internal_node_ids=(),
                decisions=(),
            )

        old_root = self.root_node_id
        new_root, changed, created, decisions = self._insert_at(old_root, leaf.node_id)
        self.root_node_id = new_root
        self.nodes[new_root].parent_id = None
        return OtdInsertionResult(
            atom_id=atom.atom_id,
            leaf_node_id=leaf.node_id,
            root_node_id=new_root,
            created_node_ids=(leaf.node_id, *created),
            changed_internal_node_ids=tuple(changed),
            decisions=tuple(decisions),
        )

    def pair_similarity(self, left: ExperienceAtom, right: ExperienceAtom) -> float:
        trigger = float(np.dot(left.trigger_embedding, right.trigger_embedding))
        procedure = float(np.dot(left.procedure_embedding, right.procedure_embedding))
        value = 0.5 + 0.5 * (
            self.dual_view_lambda * trigger
            + (1.0 - self.dual_view_lambda) * procedure
        )
        return _canonical_similarity_average(value)

    def cross_similarity(self, left_node_id: str, right_node_id: str) -> float:
        left = self.nodes[left_node_id]
        right = self.nodes[right_node_id]
        trigger = float(np.dot(left.sum_trigger, right.sum_trigger))
        procedure = float(np.dot(left.sum_procedure, right.sum_procedure))
        pair_count = left.leaf_count * right.leaf_count
        raw_sum = 0.5 * pair_count + 0.5 * (
            self.dual_view_lambda * trigger
            + (1.0 - self.dual_view_lambda) * procedure
        )
        return pair_count * _canonical_similarity_average(raw_sum / pair_count)

    def average_within_similarity(self, node_id: str) -> float:
        node = self.nodes[node_id]
        if node.leaf_count < 2:
            raise ValueError("average within similarity is undefined for a leaf")
        pair_count = node.leaf_count * (node.leaf_count - 1) / 2.0
        return _canonical_similarity_average(node.within_similarity / pair_count)

    def average_cross_similarity(
        self,
        left_node_id: str,
        right_node_id: str,
    ) -> float:
        left = self.nodes[left_node_id]
        right = self.nodes[right_node_id]
        return self.cross_similarity(left_node_id, right_node_id) / (
            left.leaf_count * right.leaf_count
        )

    def average_similarity_to_atom(
        self,
        node_id: str,
        atom: ExperienceAtom,
    ) -> float:
        node = self.nodes[node_id]
        if atom.embedding_dimension != self._embedding_dimension:
            raise ValueError(
                f"atom embedding dimension {atom.embedding_dimension} does not match "
                f"tree dimension {self._embedding_dimension}"
            )
        trigger = float(np.dot(node.sum_trigger, atom.trigger_embedding))
        procedure = float(np.dot(node.sum_procedure, atom.procedure_embedding))
        value = 0.5 + 0.5 * (
            self.dual_view_lambda * trigger / node.leaf_count
            + (1.0 - self.dual_view_lambda) * procedure / node.leaf_count
        )
        return _canonical_similarity_average(value)

    def descendant_atom_ids(self, node_id: str) -> tuple[str, ...]:
        node = self.nodes[node_id]
        if node.is_leaf:
            return (str(node.atom_id),)
        values: list[str] = []
        for child_id in node.child_ids:
            values.extend(self.descendant_atom_ids(child_id))
        return tuple(values)

    def height(self, node_id: str | None = None) -> int:
        if node_id is None:
            if self.root_node_id is None:
                return 0
            node_id = self.root_node_id
        node = self.nodes[node_id]
        if node.is_leaf:
            return 0
        return 1 + max(self.height(child_id) for child_id in node.child_ids)

    def moseley_wang_revenue(self) -> float:
        if self.root_node_id is None:
            return 0.0
        total_leaves = len(self.atoms)
        leaf_ids = {
            atom_id: self._leaf_node_id(atom_id)
            for atom_id in self.insertion_order
        }
        revenue = 0.0
        for index, left_atom_id in enumerate(self.insertion_order):
            for right_atom_id in self.insertion_order[index + 1 :]:
                lca_id = self._lowest_common_ancestor(
                    leaf_ids[left_atom_id],
                    leaf_ids[right_atom_id],
                )
                similarity = self.pair_similarity(
                    self.atoms[left_atom_id],
                    self.atoms[right_atom_id],
                )
                revenue += similarity * (
                    total_leaves - self.nodes[lca_id].leaf_count
                )
        return float(revenue)

    def structural_diagnostics(self) -> dict[str, Any]:
        theorem_implementation_eligible = self.tie_epsilon == 0.0
        return {
            "tree_policy": "certified_dual_view_otd",
            "atom_count": len(self.atoms),
            "node_count": len(self.nodes),
            "internal_node_count": sum(not node.is_leaf for node in self.nodes.values()),
            "height": self.height(),
            "dual_view_lambda": self.dual_view_lambda,
            "tie_epsilon": self.tie_epsilon,
            "moseley_wang_revenue": self.moseley_wang_revenue(),
            "exact_otd_comparisons": theorem_implementation_eligible,
            "comparison_arithmetic": "ieee754_binary64",
            "exact_comparison_semantics": (
                "formula_exact_without_engineering_tie_epsilon"
            ),
            "similarity_canonicalization": (
                "shifted_cosine_clamped_only_within_numeric_tolerance"
            ),
            "similarity_boundary_tolerance": (
                _SIMILARITY_NUMERIC_TOLERANCE
            ),
            "theorem_implementation_eligible": theorem_implementation_eligible,
            "theorem_claim_enabled": False,
            "well_separation_assumption_verified": False,
            "guarantee_scope": (
                (
                    "beta/3 Moseley-Wang revenue approximation only under the "
                    "beta-well-separated assumption for fixed nonnegative similarities"
                )
                if theorem_implementation_eligible
                else (
                    "disabled because nonzero tie_epsilon changes exact "
                    "Online Top-Down comparisons"
                )
            ),
        }

    def validate(self) -> None:
        if self.root_node_id is None:
            if self.nodes or self.atoms or self.insertion_order:
                raise ValueError("empty tree cannot contain nodes or atoms")
            return
        if self.root_node_id not in self.nodes:
            raise ValueError("root_node_id does not exist")
        if self.nodes[self.root_node_id].parent_id is not None:
            raise ValueError("root node must not have a parent")
        if len(self.insertion_order) != len(set(self.insertion_order)):
            raise ValueError("insertion_order contains duplicate atom ids")
        if set(self.insertion_order) != set(self.atoms):
            raise ValueError("insertion_order and atoms differ")
        if any(atom_id != atom.atom_id for atom_id, atom in self.atoms.items()):
            raise ValueError("atom dictionary key does not match atom_id")
        if any(node_id != node.node_id for node_id, node in self.nodes.items()):
            raise ValueError("node dictionary key does not match node_id")

        visited: set[str] = set()
        leaf_atoms: set[str] = set()

        def visit(
            node_id: str,
            parent_id: str | None,
        ) -> tuple[int, np.ndarray, np.ndarray, float, float]:
            if node_id in visited:
                raise ValueError(f"cycle or multi-parent node detected: {node_id}")
            visited.add(node_id)
            node = self.nodes[node_id]
            if node.parent_id != parent_id:
                raise ValueError(f"incorrect parent pointer for {node_id}")
            if node.is_leaf:
                if node.left_id is not None or node.right_id is not None:
                    raise ValueError(f"leaf node {node_id} cannot have children")
                if node.atom_id not in self.atoms:
                    raise ValueError(f"leaf node {node_id} references an unknown atom")
                if node.atom_id in leaf_atoms:
                    raise ValueError(f"atom {node.atom_id} appears in multiple leaves")
                leaf_atoms.add(str(node.atom_id))
                atom = self.atoms[str(node.atom_id)]
                expected_trigger = np.asarray(atom.trigger_embedding)
                expected_procedure = np.asarray(atom.procedure_embedding)
                if node.leaf_count != 1 or abs(node.within_similarity) > 1.0e-9:
                    raise ValueError(f"invalid leaf aggregates for {node_id}")
                if not np.allclose(node.sum_trigger, expected_trigger):
                    raise ValueError(f"invalid trigger sum for {node_id}")
                if not np.allclose(node.sum_procedure, expected_procedure):
                    raise ValueError(f"invalid procedure sum for {node_id}")
                if not math.isclose(
                    node.sum_trigger_squared_norms,
                    float(np.dot(expected_trigger, expected_trigger)),
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(f"invalid trigger squared norms for {node_id}")
                if not math.isclose(
                    node.sum_procedure_squared_norms,
                    float(np.dot(expected_procedure, expected_procedure)),
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(f"invalid procedure squared norms for {node_id}")
                return (
                    1,
                    expected_trigger,
                    expected_procedure,
                    node.sum_trigger_squared_norms,
                    node.sum_procedure_squared_norms,
                )

            if node.atom_id is not None or node.left_id is None or node.right_id is None:
                raise ValueError(f"internal node {node_id} must be binary")
            (
                left_count,
                left_trigger,
                left_procedure,
                left_trigger_sq,
                left_procedure_sq,
            ) = visit(node.left_id, node_id)
            (
                right_count,
                right_trigger,
                right_procedure,
                right_trigger_sq,
                right_procedure_sq,
            ) = visit(node.right_id, node_id)
            count = left_count + right_count
            trigger = left_trigger + right_trigger
            procedure = left_procedure + right_procedure
            if node.leaf_count != count:
                raise ValueError(f"invalid leaf_count for {node_id}")
            if not np.allclose(node.sum_trigger, trigger):
                raise ValueError(f"invalid trigger aggregate for {node_id}")
            if not np.allclose(node.sum_procedure, procedure):
                raise ValueError(f"invalid procedure aggregate for {node_id}")
            trigger_sq = left_trigger_sq + right_trigger_sq
            procedure_sq = left_procedure_sq + right_procedure_sq
            if not math.isclose(
                node.sum_trigger_squared_norms,
                trigger_sq,
                abs_tol=1.0e-9,
            ):
                raise ValueError(f"invalid trigger squared norms for {node_id}")
            if not math.isclose(
                node.sum_procedure_squared_norms,
                procedure_sq,
                abs_tol=1.0e-9,
            ):
                raise ValueError(f"invalid procedure squared norms for {node_id}")
            expected_within = self._within_from_stats(node)
            if not math.isclose(
                node.within_similarity,
                expected_within,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                raise ValueError(f"invalid within_similarity for {node_id}")
            expected_min = min(
                self.nodes[node.left_id].min_atom_id,
                self.nodes[node.right_id].min_atom_id,
            )
            if node.min_atom_id != expected_min:
                raise ValueError(f"invalid min_atom_id for {node_id}")
            return count, trigger, procedure, trigger_sq, procedure_sq

        visit(self.root_node_id, None)
        if visited != set(self.nodes):
            raise ValueError(f"unreachable nodes: {sorted(set(self.nodes) - visited)}")
        if leaf_atoms != set(self.atoms):
            raise ValueError("tree leaves and atoms differ")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": "certified_dual_view_otd_v1",
            "dual_view_lambda": self.dual_view_lambda,
            "tie_epsilon": self.tie_epsilon,
            "root_node_id": self.root_node_id,
            "next_internal_index": self._next_internal_index,
            "embedding_dimension": self._embedding_dimension,
            "insertion_order": list(self.insertion_order),
            "atoms": {
                atom_id: atom.to_dict()
                for atom_id, atom in sorted(self.atoms.items())
            },
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in sorted(self.nodes.items())
            },
        }

    def to_structural_dict(self) -> dict[str, Any]:
        """Serialize only theorem-bearing state, excluding mutable skill text."""

        payload = self.to_dict()
        payload["nodes"] = {
            node_id: {
                key: value
                for key, value in node.items()
                if key
                not in {
                    "skill",
                    "retrievable",
                    "validation_mode",
                    "structural_certificate",
                }
            }
            for node_id, node in payload["nodes"].items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OtdTreeState":
        if payload.get("format") != "certified_dual_view_otd_v1":
            raise ValueError(f"unsupported OTD format: {payload.get('format')!r}")
        state = cls(
            dual_view_lambda=float(payload["dual_view_lambda"]),
            tie_epsilon=float(payload["tie_epsilon"]),
        )
        state.root_node_id = payload.get("root_node_id")
        state._next_internal_index = int(payload.get("next_internal_index", 1))
        dimension = payload.get("embedding_dimension")
        state._embedding_dimension = None if dimension is None else int(dimension)
        state.insertion_order = [str(value) for value in payload.get("insertion_order", [])]
        state.atoms = {
            str(atom_id): ExperienceAtom.from_dict(atom)
            for atom_id, atom in dict(payload.get("atoms", {})).items()
        }
        state.nodes = {
            str(node_id): OtdNode.from_dict(node)
            for node_id, node in dict(payload.get("nodes", {})).items()
        }
        state.validate()
        return state

    def _insert_at(
        self,
        node_id: str,
        leaf_id: str,
    ) -> tuple[str, list[str], list[str], list[OtdDecision]]:
        node = self.nodes[node_id]
        within_sum = node.within_similarity
        cross_sum = self.cross_similarity(node_id, leaf_id)
        cross_average = self.average_cross_similarity(node_id, leaf_id)
        if node.is_leaf:
            should_merge = True
            within_average = 0.0
        else:
            within_average = self.average_within_similarity(node_id)
            should_merge = (
                within_average + self.tie_epsilon >= cross_average
            )
        if should_merge:
            parent_id = self._new_internal_node_id()
            parent = self._combine_nodes(
                parent_id,
                node_id,
                leaf_id,
                parent_id=node.parent_id,
            )
            self.nodes[parent_id] = parent
            self.nodes[node_id].parent_id = parent_id
            self.nodes[leaf_id].parent_id = parent_id
            return (
                parent_id,
                [parent_id],
                [parent_id],
                [
                    OtdDecision(
                        node_id=node_id,
                        action="merge",
                        within_similarity=within_average,
                        cross_similarity=cross_average,
                        within_similarity_sum=within_sum,
                        cross_similarity_sum=cross_sum,
                    )
                ],
            )

        left_id, right_id = node.child_ids
        left_cross = self.average_cross_similarity(left_id, leaf_id)
        right_cross = self.average_cross_similarity(right_id, leaf_id)
        selected_id = self._select_child(
            left_id,
            right_id,
            left_cross,
            right_cross,
        )
        new_child_root, changed, created, nested_decisions = self._insert_at(
            selected_id,
            leaf_id,
        )
        if selected_id == left_id:
            node.left_id = new_child_root
        else:
            node.right_id = new_child_root
        self.nodes[new_child_root].parent_id = node_id
        self._recompute_internal(node_id)
        decision = OtdDecision(
            node_id=node_id,
            action="recurse",
            within_similarity=within_average,
            cross_similarity=cross_average,
            within_similarity_sum=within_sum,
            cross_similarity_sum=cross_sum,
            left_cross_similarity=left_cross,
            right_cross_similarity=right_cross,
            selected_child_id=selected_id,
        )
        return node_id, [*changed, node_id], created, [decision, *nested_decisions]

    def _select_child(
        self,
        left_id: str,
        right_id: str,
        left_cross: float,
        right_cross: float,
    ) -> str:
        if left_cross > right_cross + self.tie_epsilon:
            return left_id
        if right_cross > left_cross + self.tie_epsilon:
            return right_id
        left_key = (self.nodes[left_id].min_atom_id, left_id)
        right_key = (self.nodes[right_id].min_atom_id, right_id)
        return left_id if left_key <= right_key else right_id

    def _leaf_node(self, atom: ExperienceAtom) -> OtdNode:
        node_id = self._leaf_node_id(atom.atom_id)
        skill = {
            "name": atom.invariant,
            "trigger": f"{atom.trigger} Scope: {atom.scope}",
            "content": (
                f"Decision: {atom.decision}\n"
                f"Invariant: {atom.invariant}\n"
                f"Verification: {atom.verification}\n"
                f"Failure mode: {atom.failure_mode}"
            ),
            "confidence": atom.reliability,
        }
        return OtdNode(
            node_id=node_id,
            parent_id=None,
            left_id=None,
            right_id=None,
            atom_id=atom.atom_id,
            leaf_count=1,
            sum_trigger=atom.trigger_embedding,
            sum_procedure=atom.procedure_embedding,
            sum_trigger_squared_norms=float(
                np.dot(atom.trigger_embedding, atom.trigger_embedding)
            ),
            sum_procedure_squared_norms=float(
                np.dot(atom.procedure_embedding, atom.procedure_embedding)
            ),
            within_similarity=0.0,
            min_atom_id=atom.atom_id,
            skill=skill,
            retrievable=True,
            validation_mode="source_verifier",
            structural_certificate={
                "passed": True,
                "source_item_id": atom.source_item_id,
                "evidence_type": atom.evidence_type,
                "reliability_semantics": (
                    "source_weight_not_calibrated_probability"
                ),
            },
        )

    def _combine_nodes(
        self,
        node_id: str,
        left_id: str,
        right_id: str,
        *,
        parent_id: str | None,
    ) -> OtdNode:
        left = self.nodes[left_id]
        right = self.nodes[right_id]
        node = OtdNode(
            node_id=node_id,
            parent_id=parent_id,
            left_id=left_id,
            right_id=right_id,
            atom_id=None,
            leaf_count=left.leaf_count + right.leaf_count,
            sum_trigger=tuple(
                float(a + b)
                for a, b in zip(left.sum_trigger, right.sum_trigger)
            ),
            sum_procedure=tuple(
                float(a + b)
                for a, b in zip(left.sum_procedure, right.sum_procedure)
            ),
            sum_trigger_squared_norms=(
                left.sum_trigger_squared_norms
                + right.sum_trigger_squared_norms
            ),
            sum_procedure_squared_norms=(
                left.sum_procedure_squared_norms
                + right.sum_procedure_squared_norms
            ),
            within_similarity=0.0,
            min_atom_id=min(left.min_atom_id, right.min_atom_id),
        )
        node.within_similarity = self._within_from_stats(node)
        return node

    def _recompute_internal(self, node_id: str) -> None:
        node = self.nodes[node_id]
        if node.is_leaf:
            raise ValueError(f"cannot recompute leaf node {node_id}")
        replacement = self._combine_nodes(
            node_id,
            str(node.left_id),
            str(node.right_id),
            parent_id=node.parent_id,
        )
        replacement.skill = dict(node.skill)
        replacement.retrievable = node.retrievable
        replacement.validation_mode = node.validation_mode
        replacement.structural_certificate = dict(node.structural_certificate)
        self.nodes[node_id] = replacement

    def _within_from_stats(self, node: OtdNode) -> float:
        pairs = node.leaf_count * (node.leaf_count - 1) / 2.0
        trigger_dot_sum = (
            float(np.dot(node.sum_trigger, node.sum_trigger))
            - node.sum_trigger_squared_norms
        ) / 2.0
        procedure_dot_sum = (
            float(np.dot(node.sum_procedure, node.sum_procedure))
            - node.sum_procedure_squared_norms
        ) / 2.0
        value = 0.5 * pairs + 0.5 * (
            self.dual_view_lambda * trigger_dot_sum
            + (1.0 - self.dual_view_lambda) * procedure_dot_sum
        )
        if pairs <= 0.0:
            return 0.0
        return pairs * _canonical_similarity_average(value / pairs)

    def _new_internal_node_id(self) -> str:
        while True:
            node_id = f"otd_{self._next_internal_index:08d}"
            self._next_internal_index += 1
            if node_id not in self.nodes:
                return node_id

    @staticmethod
    def _leaf_node_id(atom_id: str) -> str:
        digest = hashlib.sha256(atom_id.encode("utf-8")).hexdigest()[:20]
        return f"atom_{digest}"

    def _lowest_common_ancestor(self, left_id: str, right_id: str) -> str:
        left_ancestors: set[str] = set()
        current: str | None = left_id
        while current is not None:
            left_ancestors.add(current)
            current = self.nodes[current].parent_id
        current = right_id
        while current not in left_ancestors:
            parent = self.nodes[current].parent_id
            if parent is None:
                raise ValueError("nodes do not share a root")
            current = parent
        return current


def audit_observed_beta_separation(
    atoms: Sequence[ExperienceAtom],
    *,
    dual_view_lambda: float = 0.5,
    tie_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Replay arrivals and audit Assumption 1 of Menon et al. on that stream.

    This is an offline diagnostic over observed arrivals, not part of the
    O(hd) insertion path and not a guarantee for future points.
    """

    state = OtdTreeState(
        dual_view_lambda=dual_view_lambda,
        tie_epsilon=tie_epsilon,
    )
    beta = 1.0
    antecedent_count = 0
    constraint_count = 0
    zero_rhs_constraint_count = 0
    worst_constraint: dict[str, Any] | None = None

    for arrival_index, atom in enumerate(atoms):
        for node in tuple(state.nodes.values()):
            if node.is_leaf:
                continue
            subtree_cross = state.average_similarity_to_atom(node.node_id, atom)
            subtree_within = state.average_within_similarity(node.node_id)
            if subtree_cross <= subtree_within:
                continue
            left_id, right_id = node.child_ids
            left_cross = state.average_similarity_to_atom(left_id, atom)
            right_cross = state.average_similarity_to_atom(right_id, atom)
            candidates: list[tuple[str, str, float, float]] = []
            if left_cross <= right_cross:
                candidates.append((left_id, right_id, left_cross, right_cross))
            if right_cross <= left_cross:
                candidates.append((right_id, left_id, right_cross, left_cross))
            for child_id, sibling_id, child_cross, sibling_cross in candidates:
                antecedent_count += 1
                if child_cross <= 0.0:
                    zero_rhs_constraint_count += 1
                    continue
                child = state.nodes[child_id]
                child_within = (
                    0.0
                    if child.is_leaf
                    else state.average_within_similarity(child_id)
                )
                ratio = min(1.0, child_within / child_cross)
                constraint_count += 1
                if ratio < beta:
                    beta = ratio
                    worst_constraint = {
                        "arrival_index": arrival_index,
                        "arrival_atom_id": atom.atom_id,
                        "subtree_node_id": node.node_id,
                        "subtree_within_similarity": subtree_within,
                        "subtree_to_arrival_similarity": subtree_cross,
                        "less_similar_child_id": child_id,
                        "more_similar_sibling_id": sibling_id,
                        "child_within_similarity": child_within,
                        "child_to_arrival_similarity": child_cross,
                        "sibling_to_arrival_similarity": sibling_cross,
                        "beta_upper_bound": ratio,
                    }
        state.insert(atom)

    exact_comparisons = float(tie_epsilon) == 0.0
    vacuous = antecedent_count == 0
    assumption_satisfied = exact_comparisons and beta > 0.0
    return {
        "diagnostic": "observed_arrival_beta_separation_v3",
        "scope": "observed_arrivals_only",
        "atom_count": len(atoms),
        "antecedent_count": antecedent_count,
        "constraint_count": constraint_count,
        "zero_rhs_constraint_count": zero_rhs_constraint_count,
        "vacuous": vacuous,
        "observed_beta": beta,
        "observed_beta_over_3": beta / 3.0,
        "exact_otd_comparisons": exact_comparisons,
        "comparison_arithmetic": "ieee754_binary64",
        "exact_comparison_semantics": (
            "formula_exact_without_engineering_tie_epsilon"
        ),
        "similarity_canonicalization": (
            "shifted_cosine_clamped_only_within_numeric_tolerance"
        ),
        "similarity_boundary_tolerance": _SIMILARITY_NUMERIC_TOLERANCE,
        "assumption_satisfied_on_observed_stream": assumption_satisfied,
        "assumption_status": (
            "exact_otd_comparison_disabled"
            if not exact_comparisons
            else (
                "satisfied_vacuously_on_observed_stream"
                if vacuous
                else (
                    "satisfied_trivially_zero_rhs_on_observed_stream"
                    if constraint_count == 0
                    else (
                        "satisfied_nonvacuously_on_observed_stream"
                        if beta > 0.0
                        else "not_satisfied_with_positive_beta"
                    )
                )
            )
        ),
        "theorem_claim_enabled": (
            assumption_satisfied and not vacuous
        ),
        "worst_constraint": worst_constraint,
        "future_arrivals_certified": False,
        "semantic_atom_quality_certified": False,
    }


@dataclass(frozen=True)
class AntichainSelection:
    node_ids: tuple[str, ...]
    score: float
    token_cost: int


def select_budgeted_antichain(
    *,
    root_node_id: str,
    children_by_node: Mapping[str, Sequence[str]],
    relevance_by_node: Mapping[str, float],
    token_cost_by_node: Mapping[str, int],
    max_nodes: int,
    token_budget: int,
    token_unit: int = 128,
    total_cost_for_nodes: Callable[[tuple[str, ...]], int] | None = None,
    max_exact_states: int = 250_000,
) -> AntichainSelection:
    """Return the exact antichain optimum under the declared token budget.

    Additive node costs use the polynomial tree DP. A complete rendered-prompt
    cost is non-additive, so it is optimized separately with exact
    branch-and-bound. The latter is exponential in the worst case; the
    cardinality and relevance bounds make the intended small-top-k retrieval
    practical without pretending that non-additive costs have DP optimal
    substructure.
    """

    max_nodes = int(max_nodes)
    token_budget = int(token_budget)
    token_unit = int(token_unit)
    if max_nodes <= 0:
        return AntichainSelection((), 0.0, 0)
    if token_budget <= 0:
        return AntichainSelection((), 0.0, 0)
    if token_unit <= 0:
        raise ValueError("token_unit must be positive")
    if int(max_exact_states) <= 0:
        raise ValueError("max_exact_states must be positive")
    budget_units = token_budget // token_unit
    if total_cost_for_nodes is None and budget_units <= 0:
        return AntichainSelection((), 0.0, 0)

    def better(
        current: tuple[float, tuple[str, ...], int] | None,
        candidate: tuple[float, tuple[str, ...], int],
    ) -> tuple[float, tuple[str, ...], int]:
        if current is None:
            return candidate
        current_key = (current[0], -current[2], tuple(reversed(current[1])))
        candidate_key = (candidate[0], -candidate[2], tuple(reversed(candidate[1])))
        return candidate if candidate_key > current_key else current

    # Validate one reachable single-parent tree for both the additive and
    # complete-render-cost solvers.
    reachable: list[str] = []
    ancestors_by_node: dict[str, frozenset[str]] = {}
    parent_by_node: dict[str, str] = {}
    topology_visiting: set[str] = set()

    def visit(node_id: str, ancestors: frozenset[str]) -> None:
        if node_id in topology_visiting:
            raise ValueError(f"cycle detected in antichain tree at {node_id}")
        if node_id in ancestors_by_node:
            raise ValueError(
                f"node has multiple structural parents in antichain tree: {node_id}"
            )
        topology_visiting.add(node_id)
        ancestors_by_node[node_id] = ancestors
        reachable.append(node_id)
        for child_id in tuple(children_by_node.get(node_id, ())):
            existing_parent = parent_by_node.get(child_id)
            if existing_parent is not None and existing_parent != node_id:
                raise ValueError(
                    "node has multiple structural parents in antichain tree: "
                    f"{child_id}"
                )
            parent_by_node[child_id] = node_id
            visit(child_id, ancestors | {node_id})
        topology_visiting.remove(node_id)

    visit(root_node_id, frozenset())

    visiting: set[str] = set()

    def solve_additive(
        node_id: str,
    ) -> dict[tuple[int, int], tuple[float, tuple[str, ...], int]]:
        if node_id in visiting:
            raise ValueError(f"cycle detected in antichain tree at {node_id}")
        visiting.add(node_id)
        states: dict[tuple[int, int], tuple[float, tuple[str, ...], int]] = {
            (0, 0): (0.0, (), 0)
        }

        if node_id in relevance_by_node and node_id in token_cost_by_node:
            actual_cost = max(1, int(token_cost_by_node[node_id]))
            units = max(1, math.ceil(actual_cost / token_unit))
            if units <= budget_units:
                states[(1, units)] = (
                    float(relevance_by_node[node_id]),
                    (node_id,),
                    actual_cost,
                )

        children = tuple(children_by_node.get(node_id, ()))
        if children:
            descendant_states: dict[
                tuple[int, int],
                tuple[float, tuple[str, ...], int],
            ] = {(0, 0): (0.0, (), 0)}
            for child_id in children:
                child_states = solve_additive(child_id)
                combined: dict[
                    tuple[int, int],
                    tuple[float, tuple[str, ...], int],
                ] = {}
                for (left_count, left_units), left_value in descendant_states.items():
                    for (right_count, right_units), right_value in child_states.items():
                        count = left_count + right_count
                        units = left_units + right_units
                        if count > max_nodes or units > budget_units:
                            continue
                        node_ids = tuple(
                            sorted((*left_value[1], *right_value[1]))
                        )
                        actual_cost = left_value[2] + right_value[2]
                        candidate = (
                            left_value[0] + right_value[0],
                            node_ids,
                            actual_cost,
                        )
                        key = (count, units)
                        combined[key] = better(combined.get(key), candidate)
                descendant_states = combined
            for key, value in descendant_states.items():
                states[key] = better(states.get(key), value)

        visiting.remove(node_id)
        return states

    if total_cost_for_nodes is None:
        all_states = solve_additive(root_node_id)
        additive_winner: tuple[
            float,
            tuple[str, ...],
            int,
        ] | None = None
        for (count, _), value in all_states.items():
            if count <= max_nodes and value[2] <= token_budget:
                additive_winner = better(additive_winner, value)
        if additive_winner is None:
            return AntichainSelection((), 0.0, 0)
        return AntichainSelection(
            node_ids=additive_winner[1],
            score=float(additive_winner[0]),
            token_cost=int(additive_winner[2]),
        )

    cost_cache: dict[tuple[str, ...], int] = {(): 0}

    def rendered_cost(node_ids: tuple[str, ...]) -> int:
        canonical = tuple(sorted(node_ids))
        if canonical not in cost_cache:
            cost_cache[canonical] = int(total_cost_for_nodes(canonical))
        cost = cost_cache[canonical]
        if cost < 0:
            raise ValueError("selection token cost must be non-negative")
        return cost

    def is_feasible(cost: int) -> bool:
        return cost <= token_budget

    candidates = [
        node_id
        for node_id in reachable
        if node_id in relevance_by_node and node_id in token_cost_by_node
    ]
    candidates.sort(
        key=lambda node_id: (
            -float(relevance_by_node[node_id]),
            node_id,
        )
    )
    candidate_set = set(candidates)
    descendants_by_node: dict[str, set[str]] = {
        node_id: set() for node_id in candidates
    }
    for node_id in candidates:
        for ancestor_id in ancestors_by_node[node_id]:
            if ancestor_id in candidate_set:
                descendants_by_node[ancestor_id].add(node_id)
    conflicts_by_node = {
        node_id: (
            set(ancestors_by_node[node_id]).intersection(candidate_set)
            | descendants_by_node[node_id]
        )
        for node_id in candidates
    }

    # Solve the cardinality-only problem while also counting whether the
    # relevance optimum is unique. A unique optimum that fits can return
    # immediately; tied optima still require complete-cost tie-breaking.
    def merge_unconstrained(
        current: _UnconstrainedState | None,
        candidate: _UnconstrainedState,
    ) -> _UnconstrainedState:
        if current is None or candidate[0] > current[0]:
            return candidate
        if candidate[0] < current[0]:
            return current
        representative = better(
            (current[0], current[2], 0),
            (candidate[0], candidate[2], 0),
        )
        return (
            current[0],
            min(2, current[1] + candidate[1]),
            representative[1],
        )

    def solve_unconstrained(node_id: str) -> dict[int, _UnconstrainedState]:
        descendant_states: dict[int, _UnconstrainedState] = {
            0: (0.0, 1, ())
        }
        for child_id in tuple(children_by_node.get(node_id, ())):
            child_states = solve_unconstrained(child_id)
            combined: dict[int, _UnconstrainedState] = {}
            for left_count, left_value in descendant_states.items():
                for right_count, right_value in child_states.items():
                    count = left_count + right_count
                    if count > max_nodes:
                        continue
                    node_ids = tuple(
                        sorted((*left_value[2], *right_value[2]))
                    )
                    candidate = (
                        left_value[0] + right_value[0],
                        min(2, left_value[1] * right_value[1]),
                        node_ids,
                    )
                    combined[count] = merge_unconstrained(
                        combined.get(count),
                        candidate,
                    )
            descendant_states = combined
        if node_id in candidate_set:
            candidate = (
                float(relevance_by_node[node_id]),
                1,
                (node_id,),
            )
            descendant_states[1] = merge_unconstrained(
                descendant_states.get(1),
                candidate,
            )
        return descendant_states

    unconstrained_state: _UnconstrainedState | None = None
    for state in solve_unconstrained(root_node_id).values():
        unconstrained_state = merge_unconstrained(
            unconstrained_state,
            state,
        )
    if unconstrained_state is None:
        return AntichainSelection((), 0.0, 0)
    unconstrained = AntichainSelection(
        node_ids=unconstrained_state[2],
        score=unconstrained_state[0],
        token_cost=rendered_cost(unconstrained_state[2]),
    )
    unconstrained_cost = rendered_cost(unconstrained.node_ids)
    if unconstrained_state[1] == 1 and is_feasible(unconstrained_cost):
        return AntichainSelection(
            node_ids=unconstrained.node_ids,
            score=unconstrained.score,
            token_cost=unconstrained_cost,
        )

    winner: tuple[float, tuple[str, ...], int] = (
        (
            unconstrained.score,
            unconstrained.node_ids,
            unconstrained_cost,
        )
        if is_feasible(unconstrained_cost)
        else (0.0, (), 0)
    )

    def consider(selected: tuple[str, ...], score: float) -> None:
        nonlocal winner
        canonical = tuple(sorted(selected))
        cost = rendered_cost(canonical)
        if is_feasible(cost):
            winner = better(winner, (score, canonical, cost))

    # Establish a useful incumbent before exhaustive proof.
    greedy: list[str] = []
    blocked: set[str] = set()
    greedy_score = 0.0
    for node_id in candidates:
        if len(greedy) >= max_nodes:
            break
        if node_id in blocked:
            continue
        proposed = tuple((*greedy, node_id))
        if is_feasible(rendered_cost(tuple(sorted(proposed)))):
            greedy.append(node_id)
            greedy_score += float(relevance_by_node[node_id])
            blocked.update(conflicts_by_node[node_id])
            blocked.add(node_id)
    consider(tuple(greedy), greedy_score)

    def optimistic_remaining_score(
        *,
        start: int,
        slots: int,
        blocked_nodes: set[str],
    ) -> float:
        if slots <= 0:
            return 0.0
        values: list[float] = []
        for node_id in candidates[start:]:
            if node_id in blocked_nodes:
                continue
            values.append(max(0.0, float(relevance_by_node[node_id])))
            if len(values) == slots:
                break
        return sum(values)

    def search(
        *,
        index: int,
        selected: tuple[str, ...],
        selected_score: float,
        blocked_nodes: set[str],
    ) -> None:
        nonlocal visited_search_states
        visited_search_states += 1
        if visited_search_states > int(max_exact_states):
            raise RuntimeError(
                "exact non-additive antichain search exceeded its declared "
                f"state limit ({max_exact_states}); increase the retrieval "
                "token budget so the unique unconstrained optimum fits, "
                "reduce top_k, or explicitly raise max_exact_states"
            )
        consider(selected, selected_score)
        slots = max_nodes - len(selected)
        if slots <= 0 or index >= len(candidates):
            return
        upper_bound = selected_score + optimistic_remaining_score(
            start=index,
            slots=slots,
            blocked_nodes=blocked_nodes,
        )
        if upper_bound < winner[0]:
            return

        node_id = candidates[index]
        if node_id not in blocked_nodes:
            next_blocked = set(blocked_nodes)
            next_blocked.update(conflicts_by_node[node_id])
            next_blocked.add(node_id)
            search(
                index=index + 1,
                selected=(*selected, node_id),
                selected_score=(
                    selected_score + float(relevance_by_node[node_id])
                ),
                blocked_nodes=next_blocked,
            )
        search(
            index=index + 1,
            selected=selected,
            selected_score=selected_score,
            blocked_nodes=blocked_nodes,
        )

    visited_search_states = 0
    search(
        index=0,
        selected=(),
        selected_score=0.0,
        blocked_nodes=set(),
    )
    return AntichainSelection(
        node_ids=winner[1],
        score=float(winner[0]),
        token_cost=int(winner[2]),
    )


__all__ = [
    "AntichainSelection",
    "ExperienceAtom",
    "OtdDecision",
    "OtdInsertionResult",
    "OtdNode",
    "OtdTreeState",
    "audit_observed_beta_separation",
    "select_budgeted_antichain",
]

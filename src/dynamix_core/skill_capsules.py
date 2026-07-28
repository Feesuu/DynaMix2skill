from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .balanced_metric_tree import BalancedMetricTreeState

__all__ = [
    "SkillCapsule",
    "SkillCapsuleRegistry",
    "normalized_capsule_text",
    "validate_capsule_candidate",
]


_WHITESPACE_RE = re.compile(r"\s+")


def normalized_capsule_text(capsule: "SkillCapsule") -> str:
    return _WHITESPACE_RE.sub(
        " ",
        "\n".join(
            (
                capsule.name,
                capsule.trigger,
                capsule.content,
                capsule.scope,
                capsule.verification,
                *capsule.failure_modes,
            )
        ).casefold(),
    ).strip()


@dataclass
class SkillCapsule:
    capsule_id: str
    tree_node_id: str
    version: int
    level: int
    name: str
    trigger: str
    content: str
    scope: str
    verification: str
    failure_modes: tuple[str, ...]
    evidence_atom_ids: tuple[str, ...]
    source_item_ids: tuple[str, ...]
    child_capsule_ids: tuple[str, ...] = ()
    parent_capsule_id: str | None = None
    status: str = "candidate"
    validation_mode: str = "structural_only"
    validation_reasons: tuple[str, ...] = ()
    replay_passes: int = 0
    replay_trials: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reliability(self) -> float:
        return (1.0 + self.replay_passes) / (2.0 + self.replay_trials)

    @property
    def retrievable(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "tree_node_id": self.tree_node_id,
            "version": self.version,
            "level": self.level,
            "name": self.name,
            "trigger": self.trigger,
            "content": self.content,
            "scope": self.scope,
            "verification": self.verification,
            "failure_modes": list(self.failure_modes),
            "evidence_atom_ids": list(self.evidence_atom_ids),
            "source_item_ids": list(self.source_item_ids),
            "child_capsule_ids": list(self.child_capsule_ids),
            "parent_capsule_id": self.parent_capsule_id,
            "status": self.status,
            "validation_mode": self.validation_mode,
            "validation_reasons": list(self.validation_reasons),
            "replay_passes": self.replay_passes,
            "replay_trials": self.replay_trials,
            "reliability": self.reliability,
            "metadata": json.loads(
                json.dumps(
                    self.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillCapsule":
        data = dict(payload)
        data.pop("reliability", None)
        for field_name in (
            "failure_modes",
            "evidence_atom_ids",
            "source_item_ids",
            "child_capsule_ids",
            "validation_reasons",
        ):
            data[field_name] = tuple(str(value) for value in data.get(field_name, ()))
        return cls(**data)


def validate_capsule_candidate(
    capsule: SkillCapsule,
    *,
    child_capsules: Sequence[SkillCapsule] = (),
) -> tuple[str, ...]:
    reasons: list[str] = []
    for field_name in (
        "name",
        "trigger",
        "content",
        "scope",
        "verification",
    ):
        if not str(getattr(capsule, field_name)).strip():
            reasons.append(f"missing_{field_name}")
    if not capsule.failure_modes:
        reasons.append("missing_failure_modes")
    if len(capsule.evidence_atom_ids) < 2:
        reasons.append("insufficient_evidence_atoms")
    if len(capsule.evidence_atom_ids) != len(set(capsule.evidence_atom_ids)):
        reasons.append("duplicate_evidence_atom_ids")
    if len(capsule.source_item_ids) != len(set(capsule.source_item_ids)):
        reasons.append("duplicate_source_item_ids")
    if len(capsule.evidence_atom_ids) != len(capsule.source_item_ids):
        reasons.append("incomplete_source_provenance")

    if child_capsules:
        distinct_children = {
            normalized_capsule_text(child) for child in child_capsules
        }
        if len(distinct_children) < 2:
            reasons.append("fewer_than_two_distinct_child_capsules")
        if normalized_capsule_text(capsule) in distinct_children:
            reasons.append("duplicates_child_capsule")
        expected_child_ids = tuple(
            sorted(child.capsule_id for child in child_capsules)
        )
        if tuple(sorted(capsule.child_capsule_ids)) != expected_child_ids:
            reasons.append("child_capsule_provenance_mismatch")

    return tuple(sorted(set(reasons)))


class SkillCapsuleRegistry:
    def __init__(self) -> None:
        self.capsules: dict[str, SkillCapsule] = {}
        self.active_by_tree_node: dict[str, str] = {}
        self.next_version_by_tree_node: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []

    def next_identity(self, tree_node_id: str) -> tuple[str, int]:
        version = self.next_version_by_tree_node.get(tree_node_id, 0) + 1
        self.next_version_by_tree_node[tree_node_id] = version
        return f"skill_{tree_node_id}_v{version:04d}", version

    def register(
        self,
        capsule: SkillCapsule,
        *,
        event_reason: str,
    ) -> None:
        if capsule.capsule_id in self.capsules:
            raise ValueError(f"duplicate capsule_id: {capsule.capsule_id}")
        if capsule.tree_node_id in self.active_by_tree_node:
            raise ValueError(
                "tree node already has an active capsule; archive it before "
                "registering a replacement"
            )
        self.capsules[capsule.capsule_id] = capsule
        self.events.append(
            {
                "capsule_id": capsule.capsule_id,
                "tree_node_id": capsule.tree_node_id,
                "status": "candidate",
                "reason": "generated_candidate",
                "validation_reasons": [],
            }
        )
        if capsule.retrievable:
            self.active_by_tree_node[capsule.tree_node_id] = capsule.capsule_id
        self.events.append(
            {
                "capsule_id": capsule.capsule_id,
                "tree_node_id": capsule.tree_node_id,
                "status": capsule.status,
                "reason": event_reason,
                "validation_reasons": list(capsule.validation_reasons),
            }
        )

    def archive_tree_nodes(
        self,
        tree_node_ids: Sequence[str],
        *,
        reason: str,
    ) -> None:
        for tree_node_id in sorted(set(tree_node_ids)):
            capsule_id = self.active_by_tree_node.pop(tree_node_id, None)
            if capsule_id is None:
                continue
            capsule = self.capsules[capsule_id]
            capsule.status = "archived"
            capsule.parent_capsule_id = None
            self.events.append(
                {
                    "capsule_id": capsule_id,
                    "tree_node_id": tree_node_id,
                    "status": "archived",
                    "reason": reason,
                    "archive_disposition": "invalidated_stale",
                }
            )

    def active_capsule_for_tree_node(
        self,
        tree_node_id: str,
    ) -> SkillCapsule | None:
        capsule_id = self.active_by_tree_node.get(tree_node_id)
        return self.capsules.get(capsule_id) if capsule_id else None

    def active_frontier(
        self,
        tree: BalancedMetricTreeState,
        tree_node_id: str,
    ) -> tuple[SkillCapsule, ...]:
        capsule = self.active_capsule_for_tree_node(tree_node_id)
        if capsule is not None:
            return (capsule,)
        node = tree.nodes[tree_node_id]
        if node.is_leaf:
            return ()
        frontier: list[SkillCapsule] = []
        for child_id in node.child_ids:
            frontier.extend(self.active_frontier(tree, child_id))
        return tuple(
            sorted(frontier, key=lambda item: item.capsule_id)
        )

    def rebuild_active_links(self) -> None:
        children_by_capsule: dict[str, list[str]] = {
            capsule_id: []
            for capsule_id in self.active_by_tree_node.values()
        }
        active_capsule_ids = set(children_by_capsule)
        parent_by_child: dict[str, str] = {}
        for capsule_id in active_capsule_ids:
            self.capsules[capsule_id].parent_capsule_id = None
        for parent_id in sorted(active_capsule_ids):
            parent = self.capsules[parent_id]
            for child_id in parent.child_capsule_ids:
                if child_id not in active_capsule_ids:
                    raise ValueError(
                        "active parent references a non-active child capsule"
                    )
                existing = parent_by_child.get(child_id)
                if existing is not None and existing != parent_id:
                    raise ValueError(
                        "active capsule has more than one evidence parent"
                    )
                parent_by_child[child_id] = parent_id
                children_by_capsule[parent_id].append(child_id)
        for child_id, parent_id in parent_by_child.items():
            self.capsules[child_id].parent_capsule_id = parent_id
        for capsule_id, child_ids in children_by_capsule.items():
            self.capsules[capsule_id].metadata[
                "active_child_capsule_ids"
            ] = sorted(child_ids)

    def validate(self, tree: BalancedMetricTreeState) -> None:
        tree.validate()
        for tree_node_id, capsule_id in self.active_by_tree_node.items():
            if tree_node_id not in tree.nodes:
                raise ValueError("active capsule references a missing tree node")
            capsule = self.capsules[capsule_id]
            if not capsule.retrievable:
                raise ValueError("active registry contains a non-active capsule")
            descendants = set(tree.descendant_atom_ids(tree_node_id))
            evidence = set(capsule.evidence_atom_ids)
            if not evidence or not evidence.issubset(descendants):
                raise ValueError(
                    "active capsule evidence must be a non-empty subset of its "
                    "tree descendants"
                )
            node = tree.nodes[tree_node_id]
            if node.is_leaf and evidence != descendants:
                raise ValueError(
                    "active leaf capsule evidence must equal its leaf atoms"
                )
            if node.is_leaf and capsule.child_capsule_ids:
                raise ValueError("active leaf capsule cannot reference children")
            if not node.is_leaf:
                if len(capsule.child_capsule_ids) < 2:
                    raise ValueError(
                        "active parent capsule requires at least two children"
                    )
                child_capsules = [
                    self.capsules.get(child_id)
                    for child_id in capsule.child_capsule_ids
                ]
                if any(child is None for child in child_capsules):
                    raise ValueError(
                        "active parent capsule references a missing child"
                    )
                typed_children = [
                    child
                    for child in child_capsules
                    if child is not None
                ]
                if any(not child.retrievable for child in typed_children):
                    raise ValueError(
                        "active parent capsule children must remain active"
                    )
                child_evidence = {
                    atom_id
                    for child in typed_children
                    for atom_id in child.evidence_atom_ids
                }
                if evidence != child_evidence:
                    raise ValueError(
                        "active parent evidence must equal the union of its "
                        "recorded child evidence"
                    )
                direct_branches = {
                    self._direct_child_branch(
                        tree,
                        ancestor_id=tree_node_id,
                        descendant_id=child.tree_node_id,
                    )
                    for child in typed_children
                }
                if len(direct_branches) < 2:
                    raise ValueError(
                        "active parent capsule requires evidence from at least "
                        "two direct child branches"
                    )
        for capsule in self.capsules.values():
            if capsule.parent_capsule_id is None:
                continue
            parent = self.capsules.get(capsule.parent_capsule_id)
            if parent is None or not parent.retrievable:
                raise ValueError("capsule parent must reference an active capsule")
            if capsule.capsule_id not in parent.child_capsule_ids:
                raise ValueError(
                    "capsule parent link must match evidence provenance"
                )

    @staticmethod
    def _direct_child_branch(
        tree: BalancedMetricTreeState,
        *,
        ancestor_id: str,
        descendant_id: str,
    ) -> str:
        current_id = descendant_id
        while True:
            parent_id = tree.nodes[current_id].parent_id
            if parent_id == ancestor_id:
                return current_id
            if parent_id is None:
                raise ValueError(
                    "child capsule is not inside its parent capsule subtree"
                )
            current_id = parent_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "evidence_balanced_skill_capsules_v1",
            "capsules": [
                capsule.to_dict()
                for capsule in sorted(
                    self.capsules.values(),
                    key=lambda item: item.capsule_id,
                )
            ],
            "active_by_tree_node": dict(sorted(self.active_by_tree_node.items())),
            "next_version_by_tree_node": dict(
                sorted(self.next_version_by_tree_node.items())
            ),
            "events": list(self.events),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillCapsuleRegistry":
        if payload.get("format") != "evidence_balanced_skill_capsules_v1":
            raise ValueError("unsupported skill capsule registry format")
        registry = cls()
        registry.capsules = {
            capsule.capsule_id: capsule
            for capsule in (
                SkillCapsule.from_dict(item)
                for item in payload.get("capsules", [])
            )
        }
        registry.active_by_tree_node = {
            str(key): str(value)
            for key, value in dict(
                payload.get("active_by_tree_node", {})
            ).items()
        }
        registry.next_version_by_tree_node = {
            str(key): int(value)
            for key, value in dict(
                payload.get("next_version_by_tree_node", {})
            ).items()
        }
        registry.events = [
            dict(event) for event in payload.get("events", [])
        ]
        return registry

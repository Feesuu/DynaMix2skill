from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

ITEM_KIND_EXPERIENCE_CARD = "experience_card"


class ExperienceHierarchyStateLike(Protocol):
    async def to_dict(self, *, include_embeddings: bool = False, validate: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SkillExportConfig:
    """Export DynaMix ExperienceCards as a node-level retrieval bank.

    Required ExperienceCard metadata schema is intentionally minimal and strict:

    - name: experience name
    - trigger: when to use it
    - content: the reusable guidance body
    - confidence: positive float

    This project evolves as one versioned codebase: artifacts must use this
    schema and should be regenerated when the schema changes.
    """

    output_dir_name: str = "skills"
    max_node_count: int | None = None


@dataclass(frozen=True)
class ExportedSkillNode:
    node_id: str
    item_id: str
    level: int
    support_mass: float
    confidence: float
    name: str
    trigger: str
    content: str
    embedding_text: str
    prompt_text: str
    source_community_id: str = ""
    source_member_count: int = 0
    analyst_mode: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class SkillExportResult:
    output_dir: str
    manifest_path: str
    node_count: int
    nodes: list[ExportedSkillNode] = field(default_factory=list)


async def export_skill_files(
    state: ExperienceHierarchyStateLike,
    output_dir: str | Path,
    *,
    config: SkillExportConfig | None = None,
) -> SkillExportResult:
    payload = await state.to_dict(include_embeddings=False, validate=True)
    return export_skill_files_from_payload(payload, output_dir, config=config)


def export_skill_files_from_payload(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    *,
    config: SkillExportConfig | None = None,
) -> SkillExportResult:
    config = config or SkillExportConfig()
    out = Path(output_dir) / config.output_dir_name
    out.mkdir(parents=True, exist_ok=True)

    items = dict(payload.get("items", {}))
    if not items:
        raise ValueError("cannot export node bank from an empty hierarchy")

    nodes = _sort_nodes_for_bank([
        _node_from_payload(item)
        for item in items.values()
        if _is_exportable_experience_card(item)
    ])
    if config.max_node_count is not None:
        nodes = nodes[: max(0, int(config.max_node_count))]
    if not nodes:
        raise ValueError("no exportable ExperienceCard nodes found for node bank export")

    item_to_node_ids = {node.item_id: [node.node_id] for node in nodes}

    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "export_policy": {
            "retrieval_unit": "experience_card_node",
            "required_node_schema": ["name", "trigger", "content", "confidence"],
            "embedding_fields": ["name", "trigger", "content"],
            "strict_minimal_schema": True,
            "node_order": "descending_support_mass_then_descending_level_then_item_id",
            "trajectory_items_exported": False,
            "diagnostic_oversize_nodes_exported": False,
            "skill_md_files_generated": False,
            "heldout_retrieval": "dense_top_k_nodes",
        },
        "output_dir": str(out),
        "node_count": len(nodes),
        "nodes": [asdict(node) for node in nodes],
        "item_to_node_ids": item_to_node_ids,
    }
    manifest_path = out / "node_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return SkillExportResult(
        output_dir=str(out),
        manifest_path=str(manifest_path),
        node_count=len(nodes),
        nodes=nodes,
    )


def affected_node_refs(changed_item_ids: Iterable[str], manifest_path: str | Path) -> dict[str, list[str]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    item_to_node_ids = manifest.get("item_to_node_ids", {})
    result: dict[str, list[str]] = {}
    for item_id in changed_item_ids:
        node_ids = list(item_to_node_ids.get(str(item_id), []))
        if node_ids:
            result[str(item_id)] = [f"{manifest_path}#{node_id}" for node_id in node_ids]
    return result


def _is_exportable_experience_card(item: Mapping[str, Any] | dict[str, Any]) -> bool:
    if item.get("kind") != ITEM_KIND_EXPERIENCE_CARD:
        return False
    metadata = dict(item.get("metadata", {}) or {})
    if metadata.get("llm_summary_skipped") or metadata.get("oversize_singleton"):
        return False
    if str(metadata.get("name", "")).strip() == "Oversize Trajectory Reference":
        return False
    return True


def _node_from_payload(item: Mapping[str, Any]) -> ExportedSkillNode:
    metadata = dict(item.get("metadata", {}) or {})
    name = _required_metadata_string(metadata, "name", item_id=str(item.get("item_id")))
    trigger = _required_metadata_string(metadata, "trigger", item_id=str(item.get("item_id")))
    content = _required_metadata_string(metadata, "content", item_id=str(item.get("item_id")))
    confidence = _required_confidence(metadata, item_id=str(item.get("item_id")))
    embedding_text = _render_node_embedding_text(name=name, trigger=trigger, content=content)
    prompt_text = _render_node_prompt_text(name=name, trigger=trigger, content=content)
    digest_payload = json.dumps(
        {"name": name, "trigger": trigger, "content": content},
        ensure_ascii=False,
        sort_keys=True,
    )
    return ExportedSkillNode(
        node_id=str(item.get("item_id")),
        item_id=str(item.get("item_id")),
        level=int(item.get("level", 0)),
        support_mass=float(item.get("support_mass", 0.0)),
        confidence=confidence,
        name=name,
        trigger=trigger,
        content=content,
        embedding_text=embedding_text,
        prompt_text=prompt_text,
        source_community_id=str(metadata.get("source_community_id", "") or ""),
        source_member_count=int(metadata.get("source_member_count", 0) or 0),
        analyst_mode=str(metadata.get("analyst_mode", "") or ""),
        sha256=hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
    )


def _sort_nodes_for_bank(nodes: list[ExportedSkillNode]) -> list[ExportedSkillNode]:
    return sorted(nodes, key=lambda n: (-n.support_mass, -n.level, n.item_id))


def _required_metadata_string(metadata: Mapping[str, Any], key: str, *, item_id: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ExperienceCard {item_id!r} missing required metadata string: {key}")
    return value.strip()


def _required_confidence(metadata: Mapping[str, Any], *, item_id: str) -> float:
    value = metadata.get("confidence")
    if value is None:
        raise ValueError(f"ExperienceCard {item_id!r} missing required metadata confidence")
    value = float(value)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"ExperienceCard {item_id!r} confidence must be positive and finite")
    return value


def _render_node_embedding_text(*, name: str, trigger: str, content: str) -> str:
    return "\n".join([
        f"name: {name.strip()}",
        f"trigger: {trigger.strip()}",
        f"content: {content.strip()}",
    ]).strip()


def _render_node_prompt_text(*, name: str, trigger: str, content: str) -> str:
    return "\n".join([
        f"### {name.strip()}",
        "",
        f"Trigger: {trigger.strip()}",
        "",
        "Guidance:",
        content.strip(),
    ]).strip()


__all__ = [
    "ExportedSkillNode",
    "SkillExportConfig",
    "SkillExportResult",
    "affected_node_refs",
    "export_skill_files",
    "export_skill_files_from_payload",
]

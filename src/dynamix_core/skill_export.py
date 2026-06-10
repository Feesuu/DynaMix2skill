from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

ITEM_KIND_EXPERIENCE_CARD = "experience_card"
_ALLOWED_PLACEMENTS = {"skill_md", "reference", "script"}
_REFERENCE_DIR_BY_KIND = {
    "procedure": "procedures",
    "procedures": "procedures",
    "edge_case": "edge-cases",
    "edge-case": "edge-cases",
    "edge_cases": "edge-cases",
    "example": "examples",
    "examples": "examples",
    "note": "notes",
    "notes": "notes",
}


class ExperienceHierarchyStateLike(Protocol):
    async def to_dict(self, *, include_embeddings: bool = False, validate: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SkillExportConfig:
    """Export DynaMix ExperienceCards as a skill folder bank.

    Required ExperienceCard metadata schema is intentionally minimal and strict:

    - name: experience name
    - trigger: when to use it
    - content: the reusable guidance body
    - placement: {target: skill_md|reference|script, reference_kind?}

    The LLM does not choose filenames or write placement rationales.  The
    exporter derives deterministic semantic filenames from experience name and
    node id.
    - confidence: positive float

    This project evolves as one versioned codebase: artifacts must use this
    schema and should be regenerated when the schema changes.
    """

    output_dir_name: str = "skills"
    max_skill_count: int | None = None
    include_metadata_comments: bool = True
    include_confidence: bool = True
    include_support_mass: bool = True
    root_title: str = "Experience Skill"
    references_dir_name: str = "references"
    scripts_dir_name: str = "scripts"
    write_reference_index: bool = True
    max_slug_len: int = 72


@dataclass(frozen=True)
class ExportedSkillNode:
    item_id: str
    level: int
    support_mass: float
    confidence: float
    name: str
    trigger: str
    content: str
    placement: dict[str, Any]


@dataclass(frozen=True)
class ExportedSkill:
    skill_id: str
    seed_item_id: str
    path: str
    node_ids: list[str]
    max_level: int
    min_level: int


@dataclass(frozen=True)
class SkillExportResult:
    output_dir: str
    manifest_path: str
    skill_count: int
    skills: list[ExportedSkill]
    item_to_skill_paths: dict[str, list[str]]


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
    communities = dict(payload.get("communities", {}))
    if not items:
        raise ValueError("cannot export skills from an empty hierarchy")

    seed_ids = _infer_top_level_skill_seed_ids(items)
    if config.max_skill_count is not None:
        seed_ids = seed_ids[: max(0, int(config.max_skill_count))]
    if not seed_ids:
        raise ValueError("no top-level ExperienceCard seed items found for skill export")

    skills: list[ExportedSkill] = []
    item_to_skill_paths: dict[str, list[str]] = {}
    item_positions: dict[str, list[dict[str, Any]]] = {}
    node_file_catalog: list[dict[str, Any]] = []
    placement_stats: dict[str, int] = {"skill_md": 0, "reference": 0, "script": 0}

    for index, seed_id in enumerate(seed_ids, start=1):
        nodes_payload = _collect_descendant_experience_nodes(seed_id, items, communities)
        if not nodes_payload:
            continue
        exported_nodes = _sort_nodes_for_skill([_node_from_payload(node, config=config) for node in nodes_payload])
        seed_node = _node_from_payload(items[seed_id], config=config)
        skill_id = f"skill_{index:03d}_{_slugify(seed_node.name or seed_id, max_len=config.max_slug_len)}"
        skill_dir = out / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        render_result = _render_skill_folder(skill_id, seed_node, exported_nodes, skill_dir, config=config)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(render_result["skill_markdown"], encoding="utf-8")

        for position in render_result["positions"]:
            item_id = str(position["item_id"])
            item_to_skill_paths.setdefault(item_id, []).append(str(position["path"]))
            item_positions.setdefault(item_id, []).append({**position, "skill_path": str(skill_path), "skill_id": skill_id})
        node_file_catalog.extend(render_result["node_file_catalog"])
        for key, value in render_result["placement_stats"].items():
            placement_stats[key] = placement_stats.get(key, 0) + int(value)
        levels = [node.level for node in exported_nodes]
        skills.append(ExportedSkill(
            skill_id=skill_id,
            seed_item_id=seed_id,
            path=str(skill_path),
            node_ids=[node.item_id for node in exported_nodes],
            max_level=max(levels),
            min_level=min(levels),
        ))

    manifest = {
        "format": "minimal_experience_skill_bank_v1",
        "export_policy": {
            "required_node_schema": ["name", "trigger", "content", "placement", "confidence"],
            "strict_minimal_schema": True,
            "seed_policy": "top_level_experience_cards_only",
            "root_fallback": False,
            "node_order": "descending_support_mass_then_descending_level",
            "trajectory_items_exported": False,
            "source_community_details_exported": False,
            "one_skill_folder_per_seed": True,
            "llm_controls_node_placement": True,
            "references_supported": True,
            "scripts_supported": True,
        },
        "output_dir": str(out),
        "skill_count": len(skills),
        "skills": [asdict(skill) for skill in skills],
        "item_to_skill_paths": item_to_skill_paths,
        "item_positions": item_positions,
        "node_file_catalog": node_file_catalog,
        "placement_stats": placement_stats,
    }
    manifest_path = out / "skill_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return SkillExportResult(str(out), str(manifest_path), len(skills), skills, item_to_skill_paths)


def affected_skill_paths(changed_item_ids: Iterable[str], manifest_path: str | Path) -> dict[str, list[str]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    item_to_paths = manifest.get("item_to_skill_paths", {})
    result: dict[str, list[str]] = {}
    for item_id in changed_item_ids:
        paths = list(item_to_paths.get(str(item_id), []))
        if paths:
            result[str(item_id)] = paths
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


def _infer_top_level_skill_seed_ids(items: dict[str, Any]) -> list[str]:
    cards = [item for item in items.values() if _is_exportable_experience_card(item)]
    if not cards:
        return []
    max_level = max(int(item.get("level", 0)) for item in cards)
    top_cards = [str(item["item_id"]) for item in cards if int(item.get("level", 0)) == max_level]
    return _sort_seed_ids(top_cards, items)


def _direct_experience_children(item_id: str, items: dict[str, Any], communities: dict[str, Any]) -> list[str]:
    item = items[item_id]
    child_ids: list[str] = []
    for cid in item.get("generated_from_community_ids", []) or []:
        community = communities.get(str(cid))
        if not community:
            continue
        for member_id in community.get("member_weights", {}) or {}:
            member = items.get(str(member_id))
            if member and _is_exportable_experience_card(member):
                child_ids.append(str(member_id))
    return _unique_preserve_order(child_ids)


def _collect_descendant_experience_nodes(seed_item_id: str, items: dict[str, Any], communities: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    def visit(item_id: str) -> None:
        if item_id in seen:
            return
        item = items.get(item_id)
        if not item:
            return
        seen.add(item_id)
        if _is_exportable_experience_card(item):
            result.append(item)
            for child_id in _direct_experience_children(item_id, items, communities):
                visit(child_id)

    visit(seed_item_id)
    return result


def _node_from_payload(item: Mapping[str, Any], *, config: SkillExportConfig) -> ExportedSkillNode:
    metadata = dict(item.get("metadata", {}) or {})
    name = _required_metadata_string(metadata, "name", item_id=str(item.get("item_id")))
    trigger = _required_metadata_string(metadata, "trigger", item_id=str(item.get("item_id")))
    content = _required_metadata_string(metadata, "content", item_id=str(item.get("item_id")))
    confidence = _required_confidence(metadata, item_id=str(item.get("item_id")))
    placement = _required_placement(metadata.get("placement"), name=name, item_id=str(item.get("item_id")), config=config)
    return ExportedSkillNode(
        item_id=str(item.get("item_id")),
        level=int(item.get("level", 0)),
        support_mass=float(item.get("support_mass", 0.0)),
        confidence=confidence,
        name=name,
        trigger=trigger,
        content=content,
        placement=placement,
    )


def _sort_nodes_for_skill(nodes: list[ExportedSkillNode]) -> list[ExportedSkillNode]:
    return sorted(nodes, key=lambda n: (-n.support_mass, -n.level, n.item_id))


def _sort_seed_ids(seed_ids: list[str], items: dict[str, Any]) -> list[str]:
    return sorted(_unique_preserve_order(seed_ids), key=lambda iid: (-float(items[iid].get("support_mass", 0.0)), -int(items[iid].get("level", 0)), iid))


def _render_skill_folder(skill_id: str, seed_node: ExportedSkillNode, nodes: list[ExportedSkillNode], skill_dir: Path, *, config: SkillExportConfig) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    node_file_catalog: list[dict[str, Any]] = []
    placement_stats: dict[str, int] = {"skill_md": 0, "reference": 0, "script": 0}
    references_dir = skill_dir / config.references_dir_name
    scripts_dir = skill_dir / config.scripts_dir_name
    references_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    main_nodes: list[ExportedSkillNode] = []
    support_entries: list[dict[str, Any]] = []

    for section_index, node in enumerate(nodes, start=1):
        target = str(node.placement["target"])
        placement_stats[target] = placement_stats.get(target, 0) + 1
        if target == "skill_md":
            main_nodes.append(node)
            positions.append(_position(node, path=skill_dir / "SKILL.md", section_index=section_index, placement=target))
            node_file_catalog.append(_catalog_entry(node, path=skill_dir / "SKILL.md", material_kind="skill_md"))
        elif target == "reference":
            path = _write_reference_node(node, references_dir, config=config)
            entry = _support_entry(node, path, kind="reference")
            support_entries.append(entry)
            positions.append(_position(node, path=path, section_index=section_index, placement=target))
            node_file_catalog.append(_catalog_entry(node, path=path, material_kind="reference"))
        elif target == "script":
            path = _write_script_node(node, scripts_dir, config=config)
            entry = _support_entry(node, path, kind="script")
            support_entries.append(entry)
            positions.append(_position(node, path=path, section_index=section_index, placement=target))
            node_file_catalog.append(_catalog_entry(node, path=path, material_kind="script"))

    if config.write_reference_index:
        _write_reference_index(references_dir / "index.md", skill_dir=skill_dir, seed_node=seed_node, entries=support_entries)

    skill_markdown = _render_skill_markdown(skill_id, seed_node, main_nodes, support_entries, config=config)
    return {
        "skill_markdown": skill_markdown,
        "positions": positions,
        "node_file_catalog": node_file_catalog,
        "placement_stats": placement_stats,
    }


def _render_skill_markdown(skill_id: str, seed_node: ExportedSkillNode, nodes: list[ExportedSkillNode], support_entries: list[dict[str, Any]], *, config: SkillExportConfig) -> str:
    lines: list[str] = [
        "---",
        f"name: {_safe_yaml_string(seed_node.name or skill_id)}",
        f"description: {_safe_yaml_string(_shorten(seed_node.content, 180) or config.root_title)}",
        "---",
        "",
        f"# {seed_node.name or config.root_title}",
        "",
        "## When to use",
        seed_node.trigger or "Use this skill when the current task matches the experience guidance below.",
        "",
        "## Core experience",
        "The guidance below is ordered by support mass. Raw trajectory records are intentionally omitted.",
        "",
    ]
    for node in nodes:
        if config.include_metadata_comments:
            lines.append(f"<!-- item_id={node.item_id} placement=skill_md -->")
        lines.append(f"### {node.name}")
        meta_bits: list[str] = []
        if config.include_support_mass:
            meta_bits.append(f"support_mass={node.support_mass:.6g}")
        if config.include_confidence:
            meta_bits.append(f"confidence={node.confidence:.4g}")
        if meta_bits:
            lines.extend(["", f"_Metadata: {', '.join(meta_bits)}_", ""])
        else:
            lines.append("")
        if node.trigger:
            lines.extend(["**Trigger.** " + node.trigger, ""])
        lines.extend([_clean_markdown_text(node.content), ""])

    if support_entries:
        lines.extend([
            "## Support files",
            "Some detailed procedures, examples, edge cases, or scripts are exported as support files. Use them when the current task matches their names or triggers.",
            "",
        ])
        for entry in sorted(support_entries, key=lambda e: (-float(e.get("support_mass", 0.0)), str(e.get("relative_path", "")))):
            rel = entry.get("relative_path")
            lines.append(f"- `{rel}` — {entry.get('name') or entry.get('node_id')} (support_mass={float(entry.get('support_mass', 0.0)):.6g})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_reference_node(node: ExportedSkillNode, references_dir: Path, *, config: SkillExportConfig) -> Path:
    kind_dir = _reference_kind_to_dir(str(node.placement.get("reference_kind", "note")))
    slug = _slugify(str(node.name or node.item_id), max_len=config.max_slug_len)
    path = references_dir / kind_dir / f"{slug}--{_slugify(node.item_id, max_len=24)}.md"
    lines = [
        f"# {node.name}",
        "",
        f"_Source node: `{node.item_id}`; level={node.level}; support_mass={node.support_mass:.6g}_",
        "",
        "## When to use",
        node.trigger,
        "",
        "## Content",
        _clean_markdown_text(node.content),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_script_node(node: ExportedSkillNode, scripts_dir: Path, *, config: SkillExportConfig) -> Path:
    slug = _slugify(str(node.name or node.item_id), max_len=config.max_slug_len)
    path = scripts_dir / _safe_script_filename(f"{slug}.py", default_suffix=".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_clean_markdown_text(node.content).rstrip() + "\n", encoding="utf-8")
    return path


def _write_reference_index(path: Path, *, skill_dir: Path, seed_node: ExportedSkillNode, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Support file index for {seed_node.name}",
        "",
        "Support files are sorted by support mass. They are ExperienceCard-derived support materials, not raw trajectories.",
        "",
    ]
    if not entries:
        lines.append("No support files were exported.")
    else:
        for entry in sorted(entries, key=lambda e: (-float(e.get("support_mass", 0.0)), str(e.get("relative_path", "")))):
            rel = entry.get("relative_path")
            lines.append(f"- `{rel}` — {entry.get('name') or entry.get('node_id')} [{entry.get('kind')}], support_mass={float(entry.get('support_mass', 0.0)):.6g}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _support_entry(node: ExportedSkillNode, path: Path, *, kind: str) -> dict[str, Any]:
    skill_dir = _skill_dir_from_path(path)
    try:
        relative_path = str(path.relative_to(skill_dir))
    except ValueError:
        relative_path = str(path)
    return {
        "node_id": node.item_id,
        "name": node.name,
        "level": node.level,
        "support_mass": node.support_mass,
        "confidence": node.confidence,
        "kind": kind,
        "relative_path": relative_path,
        "path": str(path),
    }


def _position(node: ExportedSkillNode, *, path: Path, section_index: int, placement: str) -> dict[str, Any]:
    return {
        "item_id": node.item_id,
        "level": node.level,
        "section_index": section_index,
        "anchor": "",
        "support_mass": node.support_mass,
        "confidence": node.confidence,
        "placement": placement,
        "path": str(path),
    }


def _catalog_entry(node: ExportedSkillNode, *, path: Path, material_kind: str) -> dict[str, Any]:
    return {
        "node_id": node.item_id,
        "level": node.level,
        "name": node.name,
        "support_mass": node.support_mass,
        "confidence": node.confidence,
        "material_kind": material_kind,
        "path": str(path),
        "placement": dict(node.placement),
    }


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
    if value <= 0.0:
        raise ValueError(f"ExperienceCard {item_id!r} confidence must be positive")
    return value


def _required_placement(value: Any, *, name: str, item_id: str, config: SkillExportConfig) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"ExperienceCard {item_id!r} missing required metadata placement")
    placement = dict(value)
    target = str(placement.get("target", "")).strip().lower()
    aliases = {"main": "skill_md", "skill": "skill_md", "skill.md": "skill_md", "references": "reference", "ref": "reference", "code": "script", "scripts": "script"}
    target = aliases.get(target, target)
    if target not in _ALLOWED_PLACEMENTS:
        raise ValueError(f"ExperienceCard {item_id!r} placement.target must be one of {sorted(_ALLOWED_PLACEMENTS)}")
    placement["target"] = target
    # Minimal placement schema.  Extra keys are intentionally ignored so new
    # version artifacts remain strict and easy to audit.
    return {"target": target, "reference_kind": str(placement.get("reference_kind", "note")).strip() or "note"}


def _reference_kind_to_dir(kind: str) -> str:
    return _REFERENCE_DIR_BY_KIND.get(kind.strip().lower(), "notes")


def _skill_dir_from_path(path: Path) -> Path:
    parts = list(path.parts)
    for marker in ("references", "scripts"):
        if marker in parts:
            idx = parts.index(marker)
            return Path(*parts[:idx]) if idx else Path(".")
    return path.parent


def _safe_script_filename(filename: str, *, default_suffix: str) -> str:
    filename = filename.replace("\\", "/").split("/")[-1]
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip(".-") or "helper"
    if not Path(filename).suffix:
        filename += default_suffix
    return filename


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _slugify(text: str, max_len: int = 64) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "skill")


def _safe_yaml_string(text: str) -> str:
    return '"' + text.replace("\n", " ").replace('"', "'").strip() + '"'


def _clean_markdown_text(text: str) -> str:
    return text.strip() or "No detailed text was provided for this experience node."


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "ExportedSkill",
    "ExportedSkillNode",
    "SkillExportConfig",
    "SkillExportResult",
    "affected_skill_paths",
    "export_skill_files",
    "export_skill_files_from_payload",
]

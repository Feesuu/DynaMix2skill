#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import traceback
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from dynamix_core.data_structures import (
    ExperienceCommunity,
    ExperienceHierarchyState,
    ExperienceItem,
    ExperienceLayer,
    ITEM_KIND_EXPERIENCE_CARD,
    ITEM_KIND_TRAJECTORY,
)
from dynamix_core.gmm_bic import compute_kmax
from dynamix_core.skill_export import SkillExportConfig, export_skill_files
from dynamix_core.tree_builder import LayerBuildResult, ProjectedGmmTreeBuilder
from dynamix_trace2skill.clients import EmbeddingClient, GenerationClient
from dynamix_trace2skill.pipeline import (
    DynaMixRunConfig,
    _layers_payload,
    _prepare_analyst_tokenizer_config,
    _write_runtime_artifacts,
    default_hierarchy_config,
)
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _field_payload(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in dict(payload).items() if key in allowed}


def _experience_item_from_payload(payload: dict[str, Any], *, embedding: list[float] | None = None) -> ExperienceItem:
    data = _field_payload(ExperienceItem, payload)
    data["embedding"] = list(embedding if embedding is not None else data.get("embedding", []))
    data.setdefault("generated_from_community_ids", [])
    data.setdefault("metadata", {})
    return ExperienceItem(**data)


def _community_from_payload(payload: dict[str, Any]) -> ExperienceCommunity:
    data = _field_payload(ExperienceCommunity, payload)
    data.setdefault("posterior_member_weights", dict(data.get("member_weights", {})))
    data.setdefault("generated_item_ids", [])
    data.setdefault("metadata", {})
    return ExperienceCommunity(**data)


def _layer_from_payload(payload: dict[str, Any]) -> ExperienceLayer:
    data = _field_payload(ExperienceLayer, payload)
    data.setdefault("metadata", {})
    return ExperienceLayer(**data)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_resume_runtime(
    *,
    config_path: Path,
    source_run: Path,
    out: Path,
    embedding_base_url: str | None,
    thinking: bool,
    max_levels: int | None,
) -> tuple[DynaMixRunConfig, dict[str, Any]]:
    config = DynaMixRunConfig.from_json(config_path)
    source_config_path = source_run / "build_config.json"
    source_config = DynaMixRunConfig.from_json(source_config_path) if source_config_path.exists() else None

    config.output_dir = str(out)
    config.scenario = "resume_l1_single_abstraction"
    if source_config is not None:
        config.records_path = source_config.records_path
    if max_levels is not None:
        config.max_levels = int(max_levels)

    config.generation.thinking_mode = bool(thinking)
    extra_body = dict(config.generation.extra_body or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
    chat_template_kwargs["enable_thinking"] = bool(thinking)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    config.generation.extra_body = extra_body
    config.generation.debug_dir = str(out / "analysis" / "generation_debug")

    if embedding_base_url:
        config.embedding.base_url = embedding_base_url
    if source_config is not None and source_config.embedding.tokenizer_model:
        source_tokenizer = Path(str(source_config.embedding.tokenizer_model))
        if source_tokenizer.exists():
            config.embedding.tokenizer_model = str(source_tokenizer)
    config.embedding.cache_path = str(out / "cache" / "qwen3_embedding_cache.sqlite")

    overrides = {
        "config_path": str(config_path),
        "source_config_path": str(source_config_path) if source_config_path.exists() else None,
        "records_path_source": "source_run.build_config.json" if source_config is not None else "config_path",
        "generation_thinking_mode": config.generation.thinking_mode,
        "generation_enable_thinking": config.generation.extra_body.get("chat_template_kwargs", {}).get("enable_thinking"),
        "embedding_base_url": config.embedding.base_url,
        "embedding_cache_path": config.embedding.cache_path,
        "embedding_tokenizer_model": config.embedding.tokenizer_model,
        "max_levels": config.max_levels,
    }
    return config, overrides


async def _embed_l1_cards(
    *,
    l1_payloads: list[dict[str, Any]],
    embedding_client: EmbeddingClient,
) -> list[ExperienceItem]:
    base_items = [_experience_item_from_payload(payload) for payload in l1_payloads]
    texts = [item.text for item in base_items]
    embeddings = await embedding_client.embed_texts(
        texts,
        cache_namespace="resume_l1_experience_card_embedding",
    )
    return [replace(item, embedding=list(vector)) for item, vector in zip(base_items, embeddings)]


def _restore_l0_l1_state(
    *,
    source_state: dict[str, Any],
    l1_items: list[ExperienceItem],
) -> tuple[ExperienceHierarchyState, dict[str, Any]]:
    source_items = dict(source_state.get("items", {}))
    source_communities = dict(source_state.get("communities", {}))
    source_layers = dict(source_state.get("layers", {}))

    l0_items = [
        _experience_item_from_payload(payload)
        for payload in source_items.values()
        if int(payload.get("level", -1)) == 0 and payload.get("kind") == ITEM_KIND_TRAJECTORY
    ]
    l0_communities = [
        _community_from_payload(payload)
        for payload in source_communities.values()
        if int(payload.get("level", -1)) == 0
    ]
    if "0" not in source_layers:
        raise ValueError("source hierarchy_state.json does not contain layer 0")
    layer0 = _layer_from_payload(source_layers["0"])

    state = ExperienceHierarchyState()
    state._items = {item.item_id: item for item in [*l0_items, *l1_items]}
    state._communities = {community.community_id: community for community in l0_communities}
    state._layers = {0: layer0}
    state._pending_reroute_item_ids = set()
    state._index = None

    restored = {
        "l0_trajectory_count": len(l0_items),
        "l0_community_count": len(l0_communities),
        "l1_card_count": len(l1_items),
        "layer0_generated_count": len(layer0.generated_item_ids),
        "kept_layers": [0],
        "dropped_source_layers": sorted(int(level) for level in source_layers if int(level) >= 1),
        "dropped_source_communities_by_level": _count_by_level(
            source_communities.values(),
            min_level=1,
        ),
        "dropped_source_items_by_level": _count_by_level(
            source_items.values(),
            min_level=2,
        ),
    }
    return state, restored


def _count_by_level(payloads, *, min_level: int = 0) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in payloads:
        level = int(payload.get("level", -1))
        if level < min_level:
            continue
        counts[str(level)] = counts.get(str(level), 0) + 1
    return counts


def _source_l1_payloads(source_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        payload
        for payload in source_state.get("items", {}).values()
        if int(payload.get("level", -1)) == 1 and payload.get("kind") == ITEM_KIND_EXPERIENCE_CARD
    ]


def _validate_source_reuse(source_state: dict[str, Any], source_run: Path) -> dict[str, Any]:
    items = dict(source_state.get("items", {}))
    communities = dict(source_state.get("communities", {}))
    layers = dict(source_state.get("layers", {}))
    l0_ids = {item_id for item_id, payload in items.items() if int(payload.get("level", -1)) == 0}
    l1_ids = {item_id for item_id, payload in items.items() if int(payload.get("level", -1)) == 1 and payload.get("kind") == ITEM_KIND_EXPERIENCE_CARD}
    l0_community_ids = {cid for cid, payload in communities.items() if int(payload.get("level", -1)) == 0}
    layer0_generated = set(layers.get("0", {}).get("generated_item_ids", []))

    missing_sources: list[dict[str, Any]] = []
    for item_id in sorted(l1_ids):
        for cid in items[item_id].get("generated_from_community_ids", []) or []:
            if cid not in l0_community_ids:
                missing_sources.append({"item_id": item_id, "missing_source_community_id": cid})

    return {
        "source_run": str(source_run),
        "source_state_exists": (source_run / "hierarchy_state.json").exists(),
        "source_layers_exists": (source_run / "hierarchy_layers.json").exists(),
        "source_summary_exists": (source_run / "summary.json").exists(),
        "source_build_config_exists": (source_run / "build_config.json").exists(),
        "source_validation_ok": bool((source_state.get("validation") or {}).get("ok")),
        "l0_trajectory_count": len(l0_ids),
        "l0_community_count": len(l0_community_ids),
        "l1_card_count": len(l1_ids),
        "layer0_generated_count": len(layer0_generated),
        "layer0_generated_matches_l1_set": layer0_generated == l1_ids,
        "missing_l1_source_communities": missing_sources[:20],
        "missing_l1_source_community_count": len(missing_sources),
        "ok": bool((source_state.get("validation") or {}).get("ok"))
        and len(l0_ids) > 0
        and len(l0_community_ids) > 0
        and len(l1_ids) > 0
        and layer0_generated == l1_ids
        and not missing_sources,
    }


async def _run_resume_build(
    *,
    config: DynaMixRunConfig,
    source_state: dict[str, Any],
    out: Path,
) -> tuple[ExperienceHierarchyState, list[LayerBuildResult], dict[str, Any], EmbeddingClient, ClusterAnalyst]:
    embedding_client = EmbeddingClient(config.embedding)
    generation_client = GenerationClient(config.generation)
    _prepare_analyst_tokenizer_config(config, out)

    l1_payloads = _source_l1_payloads(source_state)
    l1_items = await _embed_l1_cards(l1_payloads=l1_payloads, embedding_client=embedding_client)
    state, restored = _restore_l0_l1_state(source_state=source_state, l1_items=l1_items)
    validation = await state.validate_hierarchy(strict_layers=True)
    if not validation.get("ok"):
        raise ValueError(f"restored L0/L1 state is invalid: {validation}")

    analyst = ClusterAnalyst(generation_client, embedding_client, config.analyst)
    builder = ProjectedGmmTreeBuilder(default_hierarchy_config(config.hierarchy))
    layer_results: list[LayerBuildResult] = []

    start_level = 1
    for level in range(start_level, int(config.max_levels)):
        layer_items = await state.item_objects_at_level(level)
        if not layer_items:
            break
        print(
            json.dumps(
                {
                    "event": "resume_build_layer_start",
                    "level": level,
                    "input_count": len(layer_items),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        result = await builder.build_layer(state, level=level, items=layer_items, summary_fn=analyst.summarize)
        layer_results.append(result)
        await _write_checkpoint(config=config, state=state, out=out, layer_results=layer_results, skill_result=None)
        print(
            json.dumps(
                {
                    "event": "resume_build_layer_done",
                    "level": level,
                    "committed": result.committed,
                    "generated_count": len(result.generated_item_ids),
                    "community_count": len(result.clustering.communities),
                    "chosen_k": result.clustering.chosen_k,
                    "stop_reason": result.clustering.stop_reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not result.committed:
            break

    restored["l1_reembedded_count"] = len(l1_items)
    embedding_client.save_truncation_report(out / "analysis" / "embedding_truncation_report.json")
    analyst.save_prompt_token_report(out / "analysis" / "cluster_prompt_token_report.json")
    return state, layer_results, restored, embedding_client, analyst


async def _write_checkpoint(
    *,
    config: DynaMixRunConfig,
    state: ExperienceHierarchyState,
    out: Path,
    layer_results: list[LayerBuildResult],
    skill_result: Any | None,
) -> dict[str, Any]:
    payload = await state.to_dict(include_embeddings=False, validate=True)
    _json_dump(out / "hierarchy_state.json", payload)
    layers_payload = _resume_layers_payload(payload, layer_results)
    _json_dump(out / "hierarchy_layers.json", layers_payload)
    summary = _summary_payload(
        config=config,
        state_payload=payload,
        layers_payload=layers_payload,
        skill_result=skill_result,
    )
    _json_dump(out / "summary.json", summary)
    return payload


def _resume_layers_payload(state_payload: dict[str, Any], layer_results: list[LayerBuildResult]) -> list[dict[str, Any]]:
    layers = dict(state_payload.get("layers", {}))
    layer0 = dict(layers.get("0", {}))
    l0_payload = {
        "level": 0,
        "input_count": len(layer0.get("input_item_ids", [])),
        "committed": True,
        "generated_count": len(layer0.get("generated_item_ids", [])),
        "stop_reason": layer0.get("stop_reason", ""),
        "chosen_k": len(layer0.get("community_ids", [])),
        "tested_k": [],
        "bic_by_k": {},
        "bic_margin": None,
        "summary_budget": {
            "resume_reused": True,
            "note": "Layer 0 communities and L1 ExperienceCards were reused from the source run; L0 clustering and L0 analyst were not rerun.",
        },
    }
    return [l0_payload, *_layers_payload(layer_results)]


def _summary_payload(
    *,
    config: DynaMixRunConfig,
    state_payload: dict[str, Any],
    layers_payload: list[dict[str, Any]],
    skill_result: Any | None,
) -> dict[str, Any]:
    items = state_payload.get("items", {})
    communities = state_payload.get("communities", {})
    summary = {
        "scenario": "resume_l1_single_abstraction",
        "record_count": sum(1 for item in items.values() if item.get("kind") == ITEM_KIND_TRAJECTORY),
        "item_count": len(items),
        "community_count": len(communities),
        "layer_count": len(layers_payload),
        "skill_count": None,
        "skill_output_dir": None,
        "skill_manifest": None,
        "skillbank_index": None,
        "skillbank_index_skipped": True,
        "heldout_executed": False,
        "embedding_truncation_events": None,
        "layers": layers_payload,
        "runtime_flags": {
            "generation_thinking_mode": config.generation.thinking_mode,
            "generation_enable_thinking": config.generation.extra_body.get("chat_template_kwargs", {}).get("enable_thinking"),
            "embedding_base_url": config.embedding.base_url,
        },
    }
    if skill_result is not None:
        summary.update(
            {
                "skill_count": int(skill_result.skill_count),
                "skill_output_dir": skill_result.output_dir,
                "skill_manifest": skill_result.manifest_path,
            }
        )
    return summary


def _is_diagnostic_community(community: dict[str, Any]) -> bool:
    metadata = dict(community.get("metadata", {}) or {})
    return bool(metadata.get("llm_summary_skipped") or metadata.get("oversize_singleton"))


def _is_exportable_card(item: dict[str, Any]) -> bool:
    if item.get("kind") != ITEM_KIND_EXPERIENCE_CARD:
        return False
    metadata = dict(item.get("metadata", {}) or {})
    if metadata.get("llm_summary_skipped") or metadata.get("oversize_singleton"):
        return False
    if str(metadata.get("name", "")).strip() == "Oversize Trajectory Reference":
        return False
    return True


def _sample_weight_stats(input_ids: list[str], items: dict[str, Any]) -> dict[str, Any]:
    masses = [float(items[item_id].get("support_mass", 0.0)) for item_id in input_ids if item_id in items]
    total = float(sum(masses))
    n = len(masses)
    weights = [(float(n) * mass / total) for mass in masses] if total > 0 else []
    return {
        "input_n": n,
        "support_mass_sum": total,
        "sample_weight_sum": float(sum(weights)),
        "sample_weight_min": float(min(weights)) if weights else None,
        "sample_weight_max": float(max(weights)) if weights else None,
        "formula": "w_i = n * support_mass_i / sum(support_mass)",
    }


def _support_conservation(
    community_ids: list[str],
    *,
    items: dict[str, Any],
    communities: dict[str, Any],
) -> dict[str, Any]:
    errors: dict[str, float] = {}
    for cid in community_ids:
        community = communities.get(cid)
        if not community:
            continue
        expected = 0.0
        for item_id, weight in (community.get("member_weights", {}) or {}).items():
            item = items.get(str(item_id))
            if item is None:
                continue
            expected += float(item.get("support_mass", 0.0)) * float(weight)
        errors[cid] = float(community.get("support_mass", 0.0)) - expected
    return {
        "max_abs_error": max((abs(value) for value in errors.values()), default=0.0),
        "sum_abs_error": sum(abs(value) for value in errors.values()),
        "errors": errors,
    }


def _singleton_identity_stats(
    *,
    level: int,
    layer: dict[str, Any],
    items: dict[str, Any],
    communities: dict[str, Any],
) -> dict[str, Any]:
    singleton_count = 0
    same_name_count = 0
    same_content_count = 0
    examples: list[dict[str, Any]] = []
    for cid in layer.get("community_ids", []) or []:
        community = communities.get(str(cid), {})
        member_ids = list((community.get("member_weights", {}) or {}).keys())
        if len(member_ids) != 1:
            continue
        singleton_count += 1
        generated_ids = list(community.get("generated_item_ids", []) or [])
        if not generated_ids:
            continue
        parent = items.get(str(member_ids[0]), {})
        child = items.get(str(generated_ids[0]), {})
        parent_meta = dict(parent.get("metadata", {}) or {})
        child_meta = dict(child.get("metadata", {}) or {})
        same_name = str(parent_meta.get("name", "")).strip() == str(child_meta.get("name", "")).strip()
        same_content = str(parent_meta.get("content", "")).strip() == str(child_meta.get("content", "")).strip()
        same_name_count += int(same_name)
        same_content_count += int(same_content)
        if same_name or same_content:
            examples.append(
                {
                    "level": level,
                    "community_id": cid,
                    "member_id": member_ids[0],
                    "generated_id": generated_ids[0],
                    "same_name": same_name,
                    "same_content": same_content,
                }
            )
    return {
        "singleton_community_count": singleton_count,
        "singleton_same_name_count": same_name_count,
        "singleton_same_content_count": same_content_count,
        "examples": examples[:20],
    }


def _build_audit(
    *,
    config: DynaMixRunConfig,
    source_reuse: dict[str, Any],
    restored: dict[str, Any],
    state_payload: dict[str, Any],
    layer_results: list[LayerBuildResult],
    runtime_overrides: dict[str, Any],
    skill_export: dict[str, Any] | None,
) -> dict[str, Any]:
    hcfg = default_hierarchy_config(config.hierarchy)
    items = dict(state_payload.get("items", {}))
    communities = dict(state_payload.get("communities", {}))
    layers = dict(state_payload.get("layers", {}))
    result_by_level = {result.clustering.level: result for result in layer_results}

    layer_audits: list[dict[str, Any]] = []
    one_card_violations: list[dict[str, Any]] = []
    kmax_violations: list[dict[str, Any]] = []
    sample_weight_violations: list[dict[str, Any]] = []
    support_violations: list[dict[str, Any]] = []
    truncation_total = 0

    for level in sorted(result_by_level):
        result = result_by_level[level]
        layer = layers.get(str(level), {})
        input_ids = list(result.clustering.input_item_ids)
        community_ids = list(layer.get("community_ids", [])) if result.committed else []
        generated_ids = list(layer.get("generated_item_ids", [])) if result.committed else []
        generated_per_community = {
            cid: len((communities.get(str(cid), {}) or {}).get("generated_item_ids", []) or [])
            for cid in community_ids
        }
        generated_dist: dict[str, int] = {}
        for count in generated_per_community.values():
            generated_dist[str(count)] = generated_dist.get(str(count), 0) + 1
        extra_truncated = sum(
            int((items.get(str(item_id), {}).get("metadata", {}) or {}).get("higher_level_extra_cards_truncated", 0) or 0)
            for item_id in generated_ids
        )
        truncation_total += extra_truncated

        sw = _sample_weight_stats(input_ids, items)
        computed_kmax = compute_kmax(
            len(input_ids),
            hcfg.gmm_bic,
            total_weight=sw["support_mass_sum"],
            kmax_effective_n=float(len(input_ids)),
        )
        legacy_support_kmax = compute_kmax(
            len(input_ids),
            hcfg.gmm_bic,
            total_weight=sw["support_mass_sum"],
        )
        support = _support_conservation(community_ids, items=items, communities=communities)
        singleton = _singleton_identity_stats(level=level, layer=layer, items=items, communities=communities)
        layer_entry = {
            "level": level,
            "input_n": len(input_ids),
            "community_count": len(community_ids) if result.committed else len(result.clustering.communities),
            "committed": result.committed,
            "generated_card_count": len(generated_ids),
            "generated_card_count_distribution_per_community": generated_dist,
            "higher_level_extra_cards_truncated_count": extra_truncated,
            "chosen_k": int(result.clustering.chosen_k),
            "tested_k": list(result.clustering.tested_k),
            "computed_kmax": int(computed_kmax),
            "legacy_support_mass_kmax_without_effective_n": int(legacy_support_kmax),
            "sample_weight_sum": sw["sample_weight_sum"],
            "sample_weight_stats": sw,
            "support_mass_conservation_error": support["max_abs_error"],
            "support_mass_conservation_sum_abs_error": support["sum_abs_error"],
            "stop_reason": result.clustering.stop_reason,
            "singleton_identity_stats": singleton,
        }
        layer_audits.append(layer_entry)

        for cid, count in generated_per_community.items():
            community = communities.get(str(cid), {})
            if _is_diagnostic_community(community):
                continue
            if int(count) != 1:
                one_card_violations.append({"level": level, "community_id": cid, "generated_count": count})
        tested_k = list(result.clustering.tested_k or [])
        if tested_k and max(tested_k) > computed_kmax:
            kmax_violations.append({"level": level, "tested_k_max": max(tested_k), "computed_kmax": computed_kmax})
        if abs(float(sw["sample_weight_sum"]) - float(len(input_ids))) > 1.0e-6:
            sample_weight_violations.append({"level": level, "sample_weight_sum": sw["sample_weight_sum"], "input_n": len(input_ids)})
        if support["max_abs_error"] > 1.0e-6:
            support_violations.append({"level": level, "max_abs_error": support["max_abs_error"]})

    exportable_cards = {item_id: item for item_id, item in items.items() if _is_exportable_card(item)}
    max_exportable_level = max((int(item.get("level", 0)) for item in exportable_cards.values()), default=None)
    old_counts = _count_by_level(source_reuse.get("_source_items_for_counts", []), min_level=1)
    new_counts = _count_by_level(exportable_cards.values(), min_level=1)
    source_reuse.pop("_source_items_for_counts", None)

    final_result = layer_results[-1] if layer_results else None
    final_stop = {
        "level": final_result.clustering.level if final_result else None,
        "stop_reason": final_result.clustering.stop_reason if final_result else "no_l1_layers_built",
        "committed": final_result.committed if final_result else False,
    }

    audit_pass = bool(source_reuse.get("ok")) and not one_card_violations and not kmax_violations and not sample_weight_violations and not support_violations
    audit = {
        "audit_pass": audit_pass,
        "scenario": "resume_l1_single_abstraction",
        "source_reuse": source_reuse,
        "restored_state": restored,
        "runtime_overrides": runtime_overrides,
        "summary_py_cardinality_contract": {
            "multi_card_max_level": ClusterAnalystConfig.multi_card_max_level,
            "max_cards_higher": ClusterAnalystConfig.max_cards_higher,
            "truncate_higher_level_extra_cards": ClusterAnalystConfig.truncate_higher_level_extra_cards,
            "l0_raw_trajectory_communities_multi_card": True,
            "l1_plus_experience_card_communities_single_card": True,
        },
        "hierarchy_contract": {
            "kmax_effective_count": "len(items)",
            "sample_weight_formula": "w_i = n * support_mass_i / sum(support_mass)",
            "soft_membership_recursive_assignment": hcfg.soft_membership.recursive_assignment,
            "soft_membership_cumulative_mass_coverage": hcfg.soft_membership.cumulative_mass_coverage,
            "budget_refinement": asdict(hcfg.budget_refinement),
            "summary_budget": hcfg.summary_budget.to_dict(),
        },
        "layers": layer_audits,
        "violations": {
            "one_card_per_non_diagnostic_community": one_card_violations,
            "kmax": kmax_violations,
            "sample_weight_sum": sample_weight_violations,
            "support_mass_conservation": support_violations,
        },
        "higher_level_extra_cards_truncated_total": truncation_total,
        "stop_layer": final_stop,
        "skill_seed_layer": max_exportable_level,
        "card_count_reduction": {
            "old_exportable_card_counts_by_level": old_counts,
            "new_exportable_card_counts_by_level": new_counts,
            "old_exportable_total": sum(old_counts.values()),
            "new_exportable_total": sum(new_counts.values()),
            "delta_old_minus_new": sum(old_counts.values()) - sum(new_counts.values()),
        },
        "skill_export": skill_export or {"attempted": False, "ok": None},
        "heldout_executed": False,
        "answers_to_required_questions": _required_question_answers(
            source_reuse=source_reuse,
            restored=restored,
            layer_audits=layer_audits,
            final_stop=final_stop,
            max_exportable_level=max_exportable_level,
            card_reduction={
                "old_total": sum(old_counts.values()),
                "new_total": sum(new_counts.values()),
                "delta": sum(old_counts.values()) - sum(new_counts.values()),
            },
            truncation_total=truncation_total,
        ),
    }
    return audit


def _required_question_answers(
    *,
    source_reuse: dict[str, Any],
    restored: dict[str, Any],
    layer_audits: list[dict[str, Any]],
    final_stop: dict[str, Any],
    max_exportable_level: int | None,
    card_reduction: dict[str, int],
    truncation_total: int,
) -> list[dict[str, str]]:
    singleton_summary = {
        str(layer["level"]): layer["singleton_identity_stats"]["singleton_community_count"]
        for layer in layer_audits
    }
    return [
        {
            "question": "这次是否是真 resume，而不是重跑 raw trajectory embedding / L0 GMM / L0 analyst?",
            "answer": "是。脚本只读取旧 state 的 L0 trajectories、L0 communities、L1 cards；只对 L1 cards 重新计算 embedding；从 level=1 调 build_layer。L0 clustering 和 L0 analyst 没有调用。",
        },
        {
            "question": "复用的 L1 cards 有多少?",
            "answer": f"{restored.get('l1_card_count')} 张；layer0_generated_count={restored.get('layer0_generated_count')}，匹配旧 L0 输出。",
        },
        {
            "question": "L1+ 是否每个 community 只生成一张 higher-level ExperienceCard?",
            "answer": "审计字段 generated_card_count_distribution_per_community 按层列出；violations.one_card_per_non_diagnostic_community 为空才算通过。",
        },
        {
            "question": "Kmax 是否按 len(items) 而不是 support mass?",
            "answer": "是。每层 computed_kmax 调用 compute_kmax(..., kmax_effective_n=len(input_items))；审计同时列出 legacy_support_mass_kmax_without_effective_n 用于对照。",
        },
        {
            "question": "GMM sample weights 怎么归一化?",
            "answer": "每层 sample_weight_stats.formula 为 w_i = n * support_mass_i / sum(support_mass)，sample_weight_sum 应等于 input_n。",
        },
        {
            "question": "是否还有 singleton / identity chain?",
            "answer": f"singleton counts by level={singleton_summary}；same-name/same-content 的近似 identity 例子在 singleton_identity_stats.examples 中。",
        },
        {
            "question": "diagnostic oversize 是否会导出?",
            "answer": "不会。diagnostic community 允许 generated_count=0；skill_export 只导出非 diagnostic ExperienceCard。",
        },
        {
            "question": "树在哪里停止?",
            "answer": f"stop_layer={final_stop}",
        },
        {
            "question": "skill seed 来自哪一层?",
            "answer": f"来自最高层 exportable ExperienceCard：level={max_exportable_level}。",
        },
        {
            "question": "card count 是否减少?",
            "answer": f"old_total={card_reduction['old_total']}，new_total={card_reduction['new_total']}，delta_old_minus_new={card_reduction['delta']}；higher_level_extra_cards_truncated_total={truncation_total}。",
        },
    ]


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Resume L1 Single Abstraction Audit")
    lines.append("")
    lines.append(f"- audit_pass: `{audit['audit_pass']}`")
    lines.append(f"- scenario: `{audit['scenario']}`")
    lines.append(f"- heldout_executed: `{audit['heldout_executed']}`")
    lines.append("")
    lines.append("## Source Reuse")
    for key, value in audit["source_reuse"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Runtime Overrides")
    for key, value in audit["runtime_overrides"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Hierarchy Contract")
    contract = audit["hierarchy_contract"]
    lines.append(f"- kmax_effective_count: `{contract['kmax_effective_count']}`")
    lines.append(f"- sample_weight_formula: `{contract['sample_weight_formula']}`")
    lines.append(f"- recursive_assignment: `{contract['soft_membership_recursive_assignment']}`")
    lines.append(f"- cumulative_mass_coverage: `{contract['soft_membership_cumulative_mass_coverage']}`")
    lines.append(f"- summary_budget: `{contract['summary_budget']}`")
    lines.append("")
    lines.append("## Layers")
    for layer in audit["layers"]:
        lines.append(f"### Level {layer['level']}")
        lines.append(f"- input_n: `{layer['input_n']}`")
        lines.append(f"- committed: `{layer['committed']}`")
        lines.append(f"- community_count: `{layer['community_count']}`")
        lines.append(f"- generated_card_count: `{layer['generated_card_count']}`")
        lines.append(f"- generated_card_count_distribution_per_community: `{layer['generated_card_count_distribution_per_community']}`")
        lines.append(f"- higher_level_extra_cards_truncated_count: `{layer['higher_level_extra_cards_truncated_count']}`")
        lines.append(f"- chosen_k: `{layer['chosen_k']}`")
        lines.append(f"- computed_kmax: `{layer['computed_kmax']}`")
        lines.append(f"- legacy_support_mass_kmax_without_effective_n: `{layer['legacy_support_mass_kmax_without_effective_n']}`")
        lines.append(f"- sample_weight_sum: `{layer['sample_weight_sum']}`")
        lines.append(f"- support_mass_conservation_error: `{layer['support_mass_conservation_error']}`")
        lines.append(f"- stop_reason: `{layer['stop_reason']}`")
        lines.append(f"- singleton_identity_stats: `{layer['singleton_identity_stats']}`")
        lines.append("")
    lines.append("## Violations")
    for key, value in audit["violations"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Required Questions")
    for index, qa in enumerate(audit["answers_to_required_questions"], start=1):
        lines.append(f"{index}. {qa['question']}")
        lines.append(f"答案：{qa['answer']}")
    lines.append("")
    lines.append("## Skill Export")
    for key, value in audit.get("skill_export", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Card Count Reduction")
    for key, value in audit["card_count_reduction"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    source_run = Path(args.source_run).resolve()
    out = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output dir exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)

    source_state_path = source_run / "hierarchy_state.json"
    if not source_state_path.exists():
        raise FileNotFoundError(source_state_path)
    source_state = _load_json(source_state_path)
    source_reuse = _validate_source_reuse(source_state, source_run)
    source_reuse["_source_items_for_counts"] = list(source_state.get("items", {}).values())
    _json_dump(out / "analysis" / "resume_source_reuse_preflight.json", {k: v for k, v in source_reuse.items() if not k.startswith("_")})
    if not source_reuse["ok"]:
        raise ValueError(f"source run is not reusable: {source_reuse}")

    config, runtime_overrides = _configure_resume_runtime(
        config_path=config_path,
        source_run=source_run,
        out=out,
        embedding_base_url=args.embedding_base_url,
        thinking=args.thinking,
        max_levels=args.max_levels,
    )
    _write_runtime_artifacts(config, out)
    _json_dump(out / "build_config.json", asdict(config))
    _json_dump(out / "analysis" / "resume_runtime_overrides.json", runtime_overrides)

    state, layer_results, restored, embedding_client, analyst = await _run_resume_build(
        config=config,
        source_state=source_state,
        out=out,
    )
    skill_export_payload: dict[str, Any] | None = None
    state_payload = await _write_checkpoint(config=config, state=state, out=out, layer_results=layer_results, skill_result=None)
    audit = _build_audit(
        config=config,
        source_reuse=source_reuse,
        restored=restored,
        state_payload=state_payload,
        layer_results=layer_results,
        runtime_overrides=runtime_overrides,
        skill_export=skill_export_payload,
    )
    _json_dump(out / "analysis" / "resume_l1_single_abstraction_audit.json", audit)
    (out / "analysis" / "resume_l1_single_abstraction_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")

    if audit["audit_pass"] and not args.no_export:
        try:
            skill_result = await export_skill_files(
                state,
                out,
                config=SkillExportConfig(output_dir_name=config.skill_output_dir_name),
            )
            skill_export_payload = {
                "attempted": True,
                "ok": True,
                "skill_count": int(skill_result.skill_count),
                "output_dir": skill_result.output_dir,
                "manifest_path": skill_result.manifest_path,
                "skillbank_index_skipped": True,
            }
        except Exception as exc:
            skill_export_payload = {
                "attempted": True,
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            raise
        finally:
            state_payload = await _write_checkpoint(
                config=config,
                state=state,
                out=out,
                layer_results=layer_results,
                skill_result=locals().get("skill_result"),
            )
            audit = _build_audit(
                config=config,
                source_reuse=_validate_source_reuse(source_state, source_run) | {"_source_items_for_counts": list(source_state.get("items", {}).values())},
                restored=restored,
                state_payload=state_payload,
                layer_results=layer_results,
                runtime_overrides=runtime_overrides,
                skill_export=skill_export_payload,
            )
            _json_dump(out / "analysis" / "resume_l1_single_abstraction_audit.json", audit)
            (out / "analysis" / "resume_l1_single_abstraction_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")

    embedding_client.close()
    final_summary = _load_json(out / "summary.json")
    final_summary["embedding_truncation_events"] = len(embedding_client.truncation_events)
    final_summary["audit_pass"] = audit["audit_pass"]
    final_summary["audit_report"] = str(out / "analysis" / "resume_l1_single_abstraction_audit.md")
    _json_dump(out / "summary.json", final_summary)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume a DynaMix hierarchy from reused L1 ExperienceCards and rebuild L1+.")
    parser.add_argument("--source-run", required=True, help="Previous run dir containing hierarchy_state.json and hierarchy_layers.json.")
    parser.add_argument("--config", required=True, help="Latest DynaMix JSON config to use for hierarchy/analyst settings.")
    parser.add_argument("--output-dir", required=True, help="Fresh run directory for resumed artifacts.")
    parser.add_argument("--embedding-base-url", default=None, help="Runtime override for embedding API base_url.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable generation thinking mode. Default: false.")
    parser.add_argument("--max-levels", type=int, default=None, help="Override max_levels.")
    parser.add_argument("--no-export", action="store_true", help="Build and audit only; do not export skills.")
    args = parser.parse_args()
    summary = asyncio.run(_main_async(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

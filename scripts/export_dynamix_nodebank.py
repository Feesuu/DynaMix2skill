#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dynamix_core.skill_export import SkillExportConfig, export_skill_files_from_payload
from dynamix_trace2skill.pipeline import DynaMixRunConfig, _refresh_skillbank_index, default_hierarchy_config


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _skill_export_config(config: DynaMixRunConfig) -> SkillExportConfig:
    payload = dict(config.skill_export or {})
    return SkillExportConfig(
        output_dir_name=str(payload.get("output_dir_name") or config.skill_output_dir_name),
        max_node_count=payload.get("max_node_count"),
        min_level=payload.get("min_level"),
        max_level=payload.get("max_level"),
    )


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    return value


def _subset(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _normalize(payload.get(key)) for key in keys}


def _canonical_baseline_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    hierarchy = dict(payload.get("hierarchy", {}) or {})
    generation = dict(payload.get("generation", {}) or {})
    embedding = dict(payload.get("embedding", {}) or {})
    analyst = dict(payload.get("analyst", {}) or {})
    return {
        "scenario": str(payload.get("scenario", "")),
        "dataset_path": payload.get("dataset_path"),
        "train_start": payload.get("train_start"),
        "train_end": payload.get("train_end"),
        "max_levels": payload.get("max_levels"),
        "hierarchy": {
            "tree_policy": hierarchy.get("tree_policy"),
            "random_seed": hierarchy.get("random_seed"),
            "projection": _normalize(hierarchy.get("projection", {})),
            "gmm_bic": _normalize(hierarchy.get("gmm_bic", {})),
            "soft_membership": _normalize(hierarchy.get("soft_membership", {})),
            "budget_refinement": _normalize(hierarchy.get("budget_refinement", {})),
            "summary_budget": _normalize(hierarchy.get("summary_budget", {})),
        },
        "analyst": _subset(analyst, (
            "prompt_style",
            "confidence_floor",
            "tokenizer_model",
            "tokenizer_required",
            "allow_regex_tokenizer_fallback",
            "max_prompt_tokens",
            "max_output_tokens",
            "dynamic_max_output_tokens",
            "multi_card_max_level",
            "max_cards_l0",
            "max_cards_higher",
            "higher_level_mode",
            "truncate_higher_level_extra_cards",
            "analysis_bundle_max_chars",
            "analysis_bundle_max_steps",
            "analysis_bundle_max_step_chars",
            "analysis_bundle_max_final_response_chars",
        )),
        "generation": _subset(generation, (
            "base_url",
            "model",
            "temperature",
            "thinking_mode",
            "extra_body",
        )),
        "embedding": _subset(embedding, (
            "base_url",
            "model",
            "max_model_len",
            "max_input_tokens",
            "truncate_long_texts",
            "tokenizer_model",
            "tokenizer_required",
            "truncation_strategy",
        )),
        "chunked_embedding": _subset(dict(payload.get("chunked_embedding", {}) or {}), (
            "enabled",
            "chunk_tokens",
            "overlap_tokens",
            "pooling",
            "add_special_tokens",
            "normalize_after_pooling",
            "fail_if_chunk_exceeds_model_limit",
        )),
    }


def _expanded_target_payload(config: DynaMixRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["hierarchy"] = asdict(default_hierarchy_config(config.hierarchy))
    return payload


def _assert_baseline_protocol(protocol: dict[str, Any], *, label: str) -> None:
    observed = {
        "scenario": protocol["scenario"],
        "max_levels": protocol["max_levels"],
        "tree_policy": protocol["hierarchy"]["tree_policy"],
        "soft_recursive_assignment": protocol["hierarchy"]["soft_membership"].get("recursive_assignment"),
        "analyst_max_cards_l0": protocol["analyst"].get("max_cards_l0"),
    }
    expected = {
        "scenario": "static_build",
        "max_levels": 8,
        "tree_policy": "projected_gmm_bic",
        "soft_recursive_assignment": "cumulative_mass",
        "analyst_max_cards_l0": None,
    }
    if observed != expected:
        raise ValueError(f"reuse-tree {label} is not the expected full static baseline: expected={expected}, observed={observed}")


def _validate_source_tree_contract(source: Path, *, target_config: DynaMixRunConfig) -> dict[str, Any]:
    summary_path = source / "summary.json"
    config_path = source / "analysis" / "runtime_config.json"
    if not config_path.is_file():
        config_path = source.parent / "dynamix_config.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"reuse-tree source must contain summary.json: {summary_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"reuse-tree source must contain analysis/runtime_config.json or sibling dynamix_config.json: {source}")
    source_summary = _load_json(summary_path)
    source_config = _load_json(config_path)
    source_config["scenario"] = str(source_summary.get("scenario") or source_config.get("scenario") or "")
    source_run_config = DynaMixRunConfig.from_json(config_path)
    source_run_config.scenario = source_config["scenario"]
    source_protocol = _canonical_baseline_protocol(_expanded_target_payload(source_run_config))
    target_protocol = _canonical_baseline_protocol(_expanded_target_payload(target_config))
    _assert_baseline_protocol(source_protocol, label="source")
    _assert_baseline_protocol(target_protocol, label="target")
    if source_protocol != target_protocol:
        raise ValueError(f"reuse-tree source protocol does not match target baseline protocol: source={source_protocol}, target={target_protocol}")
    observed = {
        "scenario": source_protocol["scenario"],
        "tree_policy": source_protocol["hierarchy"]["tree_policy"],
        "soft_recursive_assignment": source_protocol["hierarchy"]["soft_membership"].get("recursive_assignment"),
        "max_levels": source_protocol["max_levels"],
        "analyst_max_cards_l0": source_protocol["analyst"].get("max_cards_l0"),
    }
    return {"summary": source_summary, "config": source_config, "observed": observed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fresh DynaMix nodebank from an existing hierarchy_state.json")
    parser.add_argument("--source-tree-dir", required=True)
    parser.add_argument("--output-tree-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    source = Path(args.source_tree_dir).resolve()
    output = Path(args.output_tree_dir).resolve()
    if source == output:
        raise ValueError("--source-tree-dir and --output-tree-dir must be different")
    state_path = source / "hierarchy_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing source hierarchy_state.json: {state_path}")

    config = DynaMixRunConfig.from_json(args.config)
    source_contract = _validate_source_tree_contract(source, target_config=config)
    output.mkdir(parents=True, exist_ok=True)
    state_payload = _load_json(state_path)
    _write_json(output / "hierarchy_state.json", state_payload)
    layers_path = source / "hierarchy_layers.json"
    if layers_path.is_file():
        shutil.copy2(layers_path, output / "hierarchy_layers.json")

    export_config = _skill_export_config(config)
    skill_result = export_skill_files_from_payload(state_payload, output, config=export_config)
    skillbank_index = _refresh_skillbank_index(skill_result.output_dir, config)
    source_summary = dict(source_contract["summary"])
    summary = {
        **source_summary,
        "scenario": "static_build",
        "reused_tree_dir": str(source),
        "reuse_tree_contract": {
            "source_scenario": source_contract["observed"]["scenario"],
            "source_tree_policy": source_contract["observed"]["tree_policy"],
            "source_soft_recursive_assignment": source_contract["observed"]["soft_recursive_assignment"],
            "source_max_levels": source_contract["observed"]["max_levels"],
            "source_analyst_max_cards_l0": source_contract["observed"]["analyst_max_cards_l0"],
        },
        "node_count": int(skill_result.node_count),
        "node_bank_dir": skill_result.output_dir,
        "node_bank_manifest": skill_result.manifest_path,
        "skillbank_index": skillbank_index,
        "skill_export": {
            "output_dir_name": export_config.output_dir_name,
            "max_node_count": export_config.max_node_count,
            "min_level": export_config.min_level,
            "max_level": export_config.max_level,
        },
    }
    _write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dynamix_core import GmmBicConfig, ProjectedGmmDynamicTreeConfig, ProjectionConfig, SoftMembershipConfig, SummaryBudgetConfig
from dynamix_core.data_structures import ExperienceCommunity, ExperienceHierarchyState, ExperienceItem, ExperienceLayer, ITEM_KIND_TRAJECTORY
from dynamix_core.skill_export import SkillExportConfig, export_skill_files
from dynamix_core.tree_builder import ProjectedGmmTreeBuilder
from dynamix_core.update import ExperienceHierarchyDynamicUpdater

from .clients import EmbeddingClient, EmbeddingConfig, GenerationClient, GenerationConfig
from .long_embeddings import ChunkedEmbeddingConfig, embed_records_chunked_mean, save_chunked_embedding_report
from .log_parser import load_records
from .schemas import RawTrajectoryRecord
from .summary import ClusterAnalyst, ClusterAnalystConfig
from .trace_views import render_analysis_bundle_text, render_embedding_trace
from .tokenization import get_tokenizer
from .skillbank import SkillBankSelector


@dataclass
class DynamicPipelineConfig:
    initial_count: int = 120
    arrival_count: int = 80
    update_batch_size: int = 8
    shuffle_seed: int | None = 42
    snapshot_include_embeddings: bool = True
    resume_from_snapshots: bool = False
    max_propagation_rounds: int = 16


@dataclass
class DynaMixRunConfig:
    output_dir: str
    records_path: str
    scenario: str = "static_build"
    dataset_path: str | None = None
    train_start: int = 0
    train_end: int | None = None
    enforce_dataset_order: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # Long trajectory embedding is handled here, above core clustering.
    # This keeps src/dynamix_core unchanged while avoiding tokenizer-level
    # head truncation for ReAct logs longer than the embedding model limit.
    chunked_embedding: dict[str, Any] = field(default_factory=dict)
    hierarchy: dict[str, Any] = field(default_factory=dict)
    analyst: ClusterAnalystConfig = field(default_factory=ClusterAnalystConfig)
    dynamic: DynamicPipelineConfig = field(default_factory=DynamicPipelineConfig)
    max_levels: int = 8
    skill_output_dir_name: str = "skills"

    @classmethod
    def from_json(cls, path: str | Path) -> "DynaMixRunConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            output_dir=payload["output_dir"],
            records_path=payload["records_path"],
            scenario=str(payload.get("scenario", "static_build")),
            dataset_path=payload.get("dataset_path"),
            train_start=int(payload.get("train_start", 0)),
            train_end=None if payload.get("train_end") is None else int(payload.get("train_end")),
            enforce_dataset_order=bool(payload.get("enforce_dataset_order", False)),
            generation=GenerationConfig(**payload.get("generation", {})),
            embedding=EmbeddingConfig(**payload.get("embedding", {})),
            chunked_embedding=dict(payload.get("chunked_embedding", {})),
            hierarchy=dict(payload.get("hierarchy", {})),
            analyst=ClusterAnalystConfig(**payload.get("analyst", {})),
            dynamic=DynamicPipelineConfig(**payload.get("dynamic", {})),
            max_levels=int(payload.get("max_levels", 8)),
            skill_output_dir_name=str(payload.get("skill_output_dir_name", "skills")),
        )


def default_hierarchy_config(payload: dict[str, Any] | None = None) -> ProjectedGmmDynamicTreeConfig:
    """Return the real experiment default hierarchy configuration.

    Smoke tests may pass explicit overrides, but the project default must be the
    main GMM-BIC setting described in the handoff, not a tiny synthetic config.
    """
    data = dict(payload or {})
    return ProjectedGmmDynamicTreeConfig.from_mapping({
        "projection": data.get("projection", {"variance_ratio": 0.90, "max_dim": 32, "min_dim": 2, "whiten": False}),
        "gmm_bic": data.get("gmm_bic", {
            "covariance_type": "spherical",
            "num_restarts": 5,
            "kmeans_init_iters": 15,
            "max_iter": 100,
            "tol": 1.0e-4,
            "min_covar": 1.0e-6,
            "min_split_size": 2,
            "min_effective_samples_per_component": 2,
            "abs_kmax": 64,
            "max_concurrent_candidates": 1,
            "max_concurrent_restarts": 1,
        }),
        "soft_membership": data.get("soft_membership", {
            "save_soft_edges": True,
            "recursive_assignment": "cumulative_mass",
            "cumulative_mass_coverage": 0.90,
        }),
        "budget_refinement": data.get("budget_refinement", {
            "enabled": True,
            "apply_to_level": 0,
            "selection_policy": "bic_best_with_token_progress",
            "min_token_reduction_fraction": 0.10,
            "fallback": "gmm_bic_recursive",
            "flatten_refinement_leaves_to_l0": True,
            "skip_oversize_singleton": True,
        }),
        "summary_budget": data.get("summary_budget", {
            "max_model_tokens": 100000,
            "budget_ratio": 0.85,
        }),
        "dynamic_update": data.get("dynamic_update", {}),
        "random_seed": data.get("random_seed", 42),
    })


async def build_tree_from_records(config: DynaMixRunConfig) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not config.generation.debug_dir:
        config.generation.debug_dir = str(out / "analysis" / "generation_debug")
    _write_runtime_artifacts(config, out)
    records = _load_records_for_protocol(config, out)
    embedding_client = EmbeddingClient(config.embedding)
    generation_client = GenerationClient(config.generation)
    _prepare_analyst_tokenizer_config(config, out)

    embedding_texts, embeddings = await _embed_records_for_build(
        records=records,
        embedding_client=embedding_client,
        config=config,
        out=out,
    )
    items, normalized = _records_to_items(records, embedding_texts, embeddings, config=config)
    (out / "normalized_records.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    analyst = ClusterAnalyst(generation_client, embedding_client, config.analyst)
    hierarchy_config = default_hierarchy_config(config.hierarchy)
    builder = ProjectedGmmTreeBuilder(hierarchy_config)
    result = await builder.build(
        items,
        summary_fn=analyst.summarize,
        prompt_token_estimator=analyst.estimate_static_prompt_tokens,
        prompt_token_budget=int(config.analyst.max_prompt_tokens or hierarchy_config.summary_budget.analyst_prompt_token_budget),
        max_levels=config.max_levels,
    )
    analyst.save_prompt_token_report(out / "analysis" / "cluster_prompt_token_report.json")
    state_payload = await result.state.to_dict(include_embeddings=False, validate=True)
    (out / "hierarchy_state.json").write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    layers_payload = _layers_payload(result.layers)
    (out / "hierarchy_layers.json").write_text(json.dumps(layers_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    skill_result = await export_skill_files(result.state, out, config=SkillExportConfig(output_dir_name=config.skill_output_dir_name))
    skillbank_index = _refresh_skillbank_index(skill_result.output_dir, config)
    summary = {
        "scenario": "static_build",
        "record_count": len(records),
        "item_count": len(state_payload.get("items", {})),
        "community_count": len(state_payload.get("communities", {})),
        "layer_count": len(result.layers),
        "node_count": skill_result.node_count,
        "node_bank_dir": skill_result.output_dir,
        "node_bank_manifest": skill_result.manifest_path,
        "skillbank_index": skillbank_index,
        "embedding_truncation_events": len(embedding_client.truncation_events),
        "layers": layers_payload,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    embedding_client.close()
    return summary


async def build_dynamic_tree_from_records(config: DynaMixRunConfig) -> dict[str, Any]:
    """Formal dynamic-update scenario using the online growing-K updater.

    Initial records build a normal static hierarchy. Later records are shuffled
    reproducibly when configured, admitted sequentially inside each update
    batch, and summarized concurrently by layer after the batch admission.
    L0 admission first tries routed candidate communities under the analyst
    token budget; if none fit, the trajectory forms a new dynamic L0 community
    and routing component. L0 raw trajectory communities may add independent
    cards, while L1+ ExperienceCard communities are updated in place.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not config.generation.debug_dir:
        config.generation.debug_dir = str(out / "analysis" / "generation_debug")
    _write_runtime_artifacts(config, out)
    records = _load_records_for_protocol(config, out)
    embedding_client = EmbeddingClient(config.embedding)
    generation_client = GenerationClient(config.generation)
    _prepare_analyst_tokenizer_config(config, out)
    embedding_texts, embeddings = await _embed_records_for_build(
        records=records,
        embedding_client=embedding_client,
        config=config,
        out=out,
    )
    items, normalized = _records_to_items(records, embedding_texts, embeddings, config=config)
    (out / "normalized_records.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    dyn = config.dynamic
    initial_count = min(max(1, int(dyn.initial_count)), len(items))
    initial_items = items[:initial_count]
    remaining = items[initial_count:]
    if dyn.arrival_count > 0:
        remaining = remaining[: int(dyn.arrival_count)]
    if dyn.shuffle_seed is not None:
        rng = random.Random(int(dyn.shuffle_seed))
        remaining = list(remaining)
        rng.shuffle(remaining)
    arrival_items = list(remaining)

    analyst = ClusterAnalyst(generation_client, embedding_client, config.analyst)
    hierarchy_config = default_hierarchy_config(config.hierarchy)
    builder = ProjectedGmmTreeBuilder(hierarchy_config)
    resume = await _load_dynamic_snapshot(
        out=out,
        all_items=items,
        embedding_client=embedding_client,
    ) if dyn.resume_from_snapshots else None
    if resume is not None:
        state, processed_count, update_summaries = resume
        remaining = arrival_items[processed_count:]
        excluded_input_item_ids = [
            item_id
            for entry in update_summaries
            for item_id in entry.get("excluded_item_ids", [])
        ]
        excluded_oversize_singletons = [
            payload
            for entry in update_summaries
            for payload in entry.get("excluded_oversize_singletons", [])
        ]
    else:
        build_result = await builder.build(
            initial_items,
            summary_fn=analyst.summarize,
            prompt_token_estimator=analyst.estimate_static_prompt_tokens,
            prompt_token_budget=int(config.analyst.max_prompt_tokens or hierarchy_config.summary_budget.analyst_prompt_token_budget),
            max_levels=config.max_levels,
        )
        analyst.save_prompt_token_report(out / "analysis" / "cluster_prompt_token_report.json")
        state = build_result.state
        processed_count = 0
        update_summaries = []
        excluded_input_item_ids: list[str] = []
        excluded_oversize_singletons: list[dict[str, Any]] = []
    updater = ExperienceHierarchyDynamicUpdater(hierarchy_config, max_propagation_rounds=int(dyn.max_propagation_rounds))

    async def dynamic_summary_fn(context):
        member_items = [ExperienceItem(**_item_payload_to_constructor_payload(payload)) for payload in context.member_items]
        previous = list(getattr(context, "previous_generated_experiences", []) or [])
        return await analyst.summarize_dynamic_update(context.community, member_items, previous)

    batch_size = max(1, int(dyn.update_batch_size))
    for batch_index, batch_items in enumerate(_chunks(remaining, batch_size), start=(processed_count // batch_size) + 1):
        processed_count += len(batch_items)
        update_result = await updater.update(
            state=state,
            new_trajectory_items=batch_items,
            dynamic_summary_fn=dynamic_summary_fn,
            dynamic_prompt_token_estimator=analyst.estimate_dynamic_prompt_tokens,
            dynamic_prompt_token_budget=int(config.analyst.max_prompt_tokens or hierarchy_config.summary_budget.analyst_prompt_token_budget),
        )
        snapshot = await state.to_dict(include_embeddings=bool(dyn.snapshot_include_embeddings), validate=True)
        snap_dir = out / "dynamic_snapshots" / f"batch_{batch_index:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "hierarchy_state.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        skill_result = await export_skill_files(state, snap_dir, config=SkillExportConfig(output_dir_name=config.skill_output_dir_name))
        skillbank_index = _refresh_skillbank_index(skill_result.output_dir, config)
        from dynamix_core.skill_export import affected_node_refs
        affected_refs = affected_node_refs(update_result.changed_item_ids, skill_result.manifest_path)
        excluded_input_item_ids.extend(update_result.excluded_item_ids)
        excluded_oversize_singletons.extend(update_result.excluded_oversize_singletons)
        update_summaries.append({
            "batch_index": batch_index,
            "processed_count": processed_count,
            "batch_item_ids": [item.item_id for item in batch_items],
            "inserted_item_ids": update_result.inserted_item_ids,
            "excluded_item_ids": update_result.excluded_item_ids,
            "excluded_oversize_singletons": update_result.excluded_oversize_singletons,
            "updated_community_ids": update_result.updated_community_ids,
            "changed_item_ids": update_result.changed_item_ids,
            "requires_skill_export": update_result.requires_skill_export,
            "node_bank_manifest": skill_result.manifest_path,
            "skillbank_index": skillbank_index,
            "affected_node_refs": affected_refs,
        })
        (snap_dir / "snapshot_meta.json").write_text(
            json.dumps({
                "batch_index": batch_index,
                "processed_count": processed_count,
                "batch_size": len(batch_items),
                "snapshot_include_embeddings": bool(dyn.snapshot_include_embeddings),
                "updates": update_summaries,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    analyst.save_prompt_token_report(out / "analysis" / "cluster_prompt_token_report.json")
    final_state = await state.to_dict(include_embeddings=False, validate=True)
    (out / "hierarchy_state.json").write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")
    final_skill = await export_skill_files(state, out, config=SkillExportConfig(output_dir_name=config.skill_output_dir_name))
    final_skillbank_index = _refresh_skillbank_index(final_skill.output_dir, config)
    summary = {
        "scenario": "dynamic_update",
        "record_count": len(records),
        "initial_count": len(initial_items),
        "arrival_count": len(arrival_items),
        "updated_count": sum(len(entry["inserted_item_ids"]) for entry in update_summaries),
        "insertion_count": sum(len(entry["inserted_item_ids"]) + len(entry["excluded_item_ids"]) for entry in update_summaries),
        "batch_count": len(update_summaries),
        "update_batch_size": batch_size,
        "shuffle_seed": dyn.shuffle_seed,
        "excluded_count": len(excluded_input_item_ids),
        "excluded_input_item_ids": excluded_input_item_ids,
        "excluded_oversize_singletons": excluded_oversize_singletons,
        "item_count": len(final_state.get("items", {})),
        "community_count": len(final_state.get("communities", {})),
        "node_count": final_skill.node_count,
        "node_bank_dir": final_skill.output_dir,
        "node_bank_manifest": final_skill.manifest_path,
        "skillbank_index": final_skillbank_index,
        "embedding_truncation_events": len(embedding_client.truncation_events),
        "updates": update_summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    embedding_client.close()
    return summary



async def _embed_records_for_build(
    *,
    records: list[RawTrajectoryRecord],
    embedding_client: EmbeddingClient,
    config: DynaMixRunConfig,
    out: Path,
) -> tuple[list[str], list[list[float]]]:
    """Render and embed trajectories for hierarchy construction.

    When ``config.chunked_embedding.enabled`` is false, this preserves the old
    one-text-per-trajectory path exactly.  When enabled, each rendered ReAct
    trace is split into overlapping token windows, each window is embedded with
    the same ``EmbeddingClient``, and chunk embeddings are mean-pooled to a
    single trajectory vector.  This is deliberately outside ``dynamix_core``.
    """

    payload = dict(config.chunked_embedding or {})
    enabled = bool(payload.get("enabled", False))

    if not enabled:
        embedding_texts = [render_embedding_trace(r) for r in records]
        embeddings = await embedding_client.embed_texts(
            embedding_texts,
            cache_namespace="trajectory_embedding",
        )
        embedding_client.save_truncation_report(
            out / "analysis" / "embedding_truncation_report.json"
        )
        return embedding_texts, embeddings

    tokenizer_model = (
        payload.get("tokenizer_model")
        or config.embedding.tokenizer_model
        or config.embedding.model
    )
    chunk_config = ChunkedEmbeddingConfig(
        tokenizer_model=str(tokenizer_model),
        chunk_tokens=int(payload.get("chunk_tokens", ChunkedEmbeddingConfig.chunk_tokens)),
        overlap_tokens=int(payload.get("overlap_tokens", ChunkedEmbeddingConfig.overlap_tokens)),
        pooling=str(payload.get("pooling", "mean")),
        add_special_tokens=bool(payload.get("add_special_tokens", False)),
        normalize_after_pooling=bool(payload.get("normalize_after_pooling", False)),
        fail_if_chunk_exceeds_model_limit=bool(
            payload.get("fail_if_chunk_exceeds_model_limit", True)
        ),
    )
    result = await embed_records_chunked_mean(
        records,
        embedding_client,
        chunk_config,
        cache_namespace="trajectory_embedding",
    )
    save_chunked_embedding_report(
        out / "analysis" / "chunked_embedding_report.json",
        result.report,
    )
    embedding_client.save_truncation_report(
        out / "analysis" / "embedding_truncation_report.json"
    )
    return result.embedding_texts, result.embeddings

def _refresh_skillbank_index(skillbank_root: str | Path, config: DynaMixRunConfig) -> str:
    """Build or refresh the nodebank embedding index after every export.

    Dynamic updates can change one or more exported nodes. Rebuilding the small
    JSON index is deterministic and avoids stale node retrieval.
    """
    root = Path(skillbank_root)
    index_path = root / ".dynamix_skillbank_index.json"
    selector = SkillBankSelector(
        skillbank_root=root,
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        api_key=config.embedding.api_key,
        cache_path=index_path,
    )
    selector._load_or_build_index()
    return str(index_path)


def _prepare_analyst_tokenizer_config(config: DynaMixRunConfig, out: Path) -> None:
    # Use one tokenizer policy for analysis-bundle token counts and actual
    # cluster analyst prompt preflight.  This makes core summary_budget splits
    # depend on the same tokenization regime used before the LLM call.
    mock_mode = config.generation.base_url.startswith("mock://") or config.embedding.base_url.startswith("mock://")
    if not config.analyst.tokenizer_model:
        config.analyst.tokenizer_model = None if mock_mode else (config.embedding.tokenizer_model or config.embedding.model)
    # Analyst prompt budget is the full prompt allowance.  The hierarchy builder
    # uses summary_budget.effective_token_budget for member evidence after
    # subtracting prompt overhead reserve.
    analyst_budget_was_overridden = config.analyst.max_prompt_tokens is not None
    if config.analyst.max_prompt_tokens is None:
        hierarchy_config = default_hierarchy_config(config.hierarchy)
        config.analyst.max_prompt_tokens = hierarchy_config.summary_budget.analyst_prompt_token_budget
    # Mock/local tests can use regex fallback; real Qwen runs should keep
    # tokenizer_required=true and fail fast if the tokenizer is unavailable.
    if mock_mode:
        config.analyst.allow_regex_tokenizer_fallback = True
        config.analyst.tokenizer_required = False
    if not config.analyst.prompt_token_report_path:
        config.analyst.prompt_token_report_path = str(out / "analysis" / "cluster_prompt_token_report.json")
    budget = {
        "analyst_max_prompt_tokens": config.analyst.max_prompt_tokens,
        "analyst_max_output_tokens": config.analyst.max_output_tokens,
        "analyst_dynamic_max_output_tokens": config.analyst.dynamic_max_output_tokens,
        "source": "analyst.max_prompt_tokens override" if analyst_budget_was_overridden else "hierarchy.summary_budget",
        "hierarchy_summary_budget": default_hierarchy_config(config.hierarchy).summary_budget.to_dict(),
    }
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "analyst_budget_config.json").write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_records_for_protocol(config: DynaMixRunConfig, out: Path) -> list[RawTrajectoryRecord]:
    records = load_records(config.records_path)
    if not config.enforce_dataset_order:
        return records
    if not config.dataset_path:
        raise ValueError("enforce_dataset_order=true requires dataset_path in DynaMix config")
    train_end = len(records) if config.train_end is None else int(config.train_end)
    ordered, manifest = _order_records_by_dataset_slice(
        records=records,
        dataset_path=Path(config.dataset_path),
        train_start=int(config.train_start),
        train_end=train_end,
        source_records_path=Path(config.records_path),
    )
    (out / "records_order_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ordered


def _order_records_by_dataset_slice(
    *,
    records: list[RawTrajectoryRecord],
    dataset_path: Path,
    train_start: int,
    train_end: int,
    source_records_path: Path,
) -> tuple[list[RawTrajectoryRecord], dict[str, Any]]:
    resolved_dataset = dataset_path / "dataset.json" if dataset_path.is_dir() else dataset_path
    payload = json.loads(resolved_dataset.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        dataset_rows = payload
    elif isinstance(payload, dict):
        dataset_rows = payload.get("results") or payload.get("data") or payload.get("instances") or []
    else:
        dataset_rows = []
    if not all(isinstance(row, dict) for row in dataset_rows):
        raise ValueError(f"unsupported dataset format: {resolved_dataset}")
    expected_ids = [
        str(row.get("id", row.get("task_id", row.get("instance_id", index))))
        for index, row in enumerate(dataset_rows[train_start:train_end], start=train_start)
    ]
    by_task_id: dict[str, RawTrajectoryRecord] = {}
    duplicates: list[str] = []
    for record in records:
        task_id = str(record.task_id)
        if task_id in by_task_id:
            duplicates.append(task_id)
        by_task_id[task_id] = record
    missing = [task_id for task_id in expected_ids if task_id not in by_task_id]
    expected_set = set(expected_ids)
    extra = [task_id for task_id in by_task_id if task_id not in expected_set]
    if duplicates or missing or extra:
        raise RuntimeError(
            "records do not match dataset train slice exactly: "
            f"duplicates={duplicates[:10]}, missing={missing[:10]}, extra={extra[:10]}"
        )
    source_ids = [str(record.task_id) for record in records]
    manifest = {
        "policy": "records are ordered by dataset.json train slice order; no filename sorting or random shuffling",
        "source_records": str(source_records_path),
        "source_dataset_json": str(resolved_dataset.resolve()),
        "train_range": [int(train_start), int(train_end)],
        "record_count": len(expected_ids),
        "source_order_equal_dataset_order": source_ids == expected_ids,
        "first_task_ids": expected_ids[:10],
        "last_task_ids": expected_ids[-10:],
    }
    return [by_task_id[task_id] for task_id in expected_ids], manifest


def _records_to_items(records: list[RawTrajectoryRecord], embedding_texts: list[str], embeddings: list[list[float]], *, config: DynaMixRunConfig) -> tuple[list[ExperienceItem], list[dict[str, Any]]]:
    items = []
    normalized = []
    tokenizer = get_tokenizer(config.analyst.tokenizer_model or config.embedding.tokenizer_model or config.embedding.model, allow_regex_fallback=config.analyst.allow_regex_tokenizer_fallback)
    per_member_overhead = 128
    for record, text, embedding in zip(records, embedding_texts, embeddings):
        analysis_bundle = render_analysis_bundle_text(record)
        analysis_count = tokenizer.count(analysis_bundle) + per_member_overhead
        embedding_count = tokenizer.count(text)
        metadata = {
            "success": record.success,
            "verifier_score": record.verifier_score,
            "instruction": record.instruction,
            "instruction_type": record.instruction_type,
            "answer_position": record.answer_position,
            "analysis_bundle": analysis_bundle,
            "analysis_token_count": analysis_count,
            "analysis_tokenizer": tokenizer.name,
            "analysis_per_member_prompt_overhead": per_member_overhead,
            "embedding_trace_token_count": embedding_count,
            "trajectory_id": record.trajectory_id,
            "task_id": record.task_id,
        }
        item = ExperienceItem(
            item_id=record.trajectory_id,
            level=0,
            kind=ITEM_KIND_TRAJECTORY,
            text=text,
            embedding=embedding,
            support_mass=1.0,
            metadata=metadata,
        )
        items.append(item)
        normalized.append({"record": record.to_dict(), "embedding_trace": text, "analysis_bundle": analysis_bundle, "experience_item": item.to_dict(include_embedding=False)})
    return items, normalized


def _layers_payload(layers) -> list[dict[str, Any]]:
    return [
        {
            "level": layer.clustering.level,
            "input_count": len(layer.clustering.input_item_ids),
            "committed": layer.committed,
            "generated_count": len(layer.generated_item_ids),
            "stop_reason": layer.clustering.stop_reason,
            "chosen_k": layer.clustering.chosen_k,
            "tested_k": layer.clustering.tested_k,
            "bic_by_k": layer.clustering.bic_by_k,
            "bic_margin": layer.clustering.bic_margin,
            "summary_budget": layer.clustering.summary_budget,
        }
        for layer in layers
    ]


def _write_runtime_artifacts(config: DynaMixRunConfig, out: Path) -> None:
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    for section in ("generation", "embedding"):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict):
            key = str(section_payload.get("api_key") or "")
            if key and key != "EMPTY":
                section_payload["api_key"] = f"sha256:{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
                section_payload["api_key_redacted"] = True
    (analysis / "runtime_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "records_path": str(Path(config.records_path).resolve()),
        "output_dir": str(out.resolve()),
        "scenario": config.scenario,
        "core_checksums": _core_checksums(Path(__file__).resolve().parents[1] / "dynamix_core"),
    }
    (analysis / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _core_checksums(core_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(core_dir.glob("*.py")):
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def _item_payload_to_constructor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {"item_id", "level", "kind", "text", "embedding", "support_mass", "generated_from_community_ids", "version", "metadata"}
    return {k: v for k, v in payload.items() if k in allowed}


def _chunks(items: list[ExperienceItem], size: int) -> list[list[ExperienceItem]]:
    size = max(1, int(size))
    return [items[index : index + size] for index in range(0, len(items), size)]


async def _load_dynamic_snapshot(
    *,
    out: Path,
    all_items: list[ExperienceItem],
    embedding_client: EmbeddingClient,
) -> tuple[ExperienceHierarchyState, int, list[dict[str, Any]]] | None:
    del embedding_client
    snapshot_root = out / "dynamic_snapshots"
    if not snapshot_root.exists():
        return None
    candidates = [
        path
        for path in sorted(snapshot_root.glob("batch_*"), key=_dynamic_batch_snapshot_sort_key)
        if (path / "hierarchy_state.json").is_file() and (path / "snapshot_meta.json").is_file()
    ]
    if not candidates:
        return None
    latest = candidates[-1]
    payload = json.loads((latest / "hierarchy_state.json").read_text(encoding="utf-8"))
    meta = json.loads((latest / "snapshot_meta.json").read_text(encoding="utf-8"))
    embedding_by_item = {item.item_id: list(item.embedding) for item in all_items if item.embedding}

    state = ExperienceHierarchyState()
    items: dict[str, ExperienceItem] = {}
    missing_embeddings: list[str] = []
    for item_id, item_payload in dict(payload.get("items", {})).items():
        data = _item_payload_to_constructor_payload(dict(item_payload))
        embedding = data.get("embedding")
        if not embedding:
            if str(item_id) in embedding_by_item:
                data["embedding"] = embedding_by_item[str(item_id)]
            else:
                missing_embeddings.append(str(item_id))
        items[str(item_id)] = ExperienceItem(**data)
    if missing_embeddings:
        raise RuntimeError(
            "cannot resume dynamic snapshot without embeddings; "
            f"missing item embeddings for {missing_embeddings[:10]}. "
            "Run with dynamic.snapshot_include_embeddings=true or rebuild from the initial static state."
        )

    state._items = items
    state._communities = {
        str(community_id): ExperienceCommunity(**dict(community_payload))
        for community_id, community_payload in dict(payload.get("communities", {})).items()
    }
    state._layers = {
        int(level): ExperienceLayer(**dict(layer_payload))
        for level, layer_payload in dict(payload.get("layers", {})).items()
    }
    state._pending_reroute_item_ids = set(str(item_id) for item_id in payload.get("pending_reroute_item_ids", []))
    state._index = None
    validation = await state.validate_hierarchy(require_no_pending_reroute=True, require_no_stale_layers=False)
    if not validation.get("ok", False):
        raise RuntimeError(f"dynamic snapshot {latest} is invalid: {validation}")
    return state, int(meta.get("processed_count", 0)), list(meta.get("updates", []))


def _dynamic_batch_snapshot_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("batch_")
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


async def run_config(config: DynaMixRunConfig) -> dict[str, Any]:
    if config.scenario == "static_build":
        return await build_tree_from_records(config)
    if config.scenario == "dynamic_update":
        return await build_dynamic_tree_from_records(config)
    raise ValueError(f"unknown scenario={config.scenario!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DynaMix hierarchy from Trace2Skill trajectories")
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()
    config = DynaMixRunConfig.from_json(args.config)
    summary = asyncio.run(run_config(config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

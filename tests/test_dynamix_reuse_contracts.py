from __future__ import annotations

import json
import asyncio
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dynamix_trace2skill.clients import EmbeddingClient, EmbeddingConfig, GenerationClient, GenerationConfig
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig
from dynamix_trace2skill.log_parser import parse_trace2skill_logs, _result_fields
from dynamix_trace2skill.pipeline import DynaMixRunConfig, default_hierarchy_config
from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep
from dynamix_trace2skill.trace_views import render_embedding_trace
from dynamix_core.data_structures import ExperienceCardPatch, ExperienceCommunity, ExperienceHierarchyState, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY
from dynamix_core.config import ProjectionConfig
from dynamix_core.projection import local_pca_project
from dynamix_core.update import ExperienceHierarchyDynamicUpdater


def _load_experiment_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamix_trace2skill_experiment.py"
    spec = importlib.util.spec_from_file_location("run_dynamix_trace2skill_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_full_soft_hard_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_spreadsheetbench_full_soft_hard_eval.py"
    spec = importlib.util.spec_from_file_location("run_spreadsheetbench_full_soft_hard_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_embedding_truncates_to_configured_32k_budget_with_tokenizer(tmp_path):
    cfg = EmbeddingConfig(
        base_url="mock://deterministic",
        max_model_len=32000,
        max_input_tokens=32000,
        tokenizer_required=False,
        cache_path=str(tmp_path / "cache.sqlite"),
    )
    client = EmbeddingClient(cfg)
    text = "x " * 33000
    vec = asyncio.run(async_embed(client, [text]))[0]
    assert len(vec) == cfg.deterministic_dim
    assert client.truncation_events
    assert client.truncation_events[0]["max_input_tokens"] == 32000
    assert "token_count" in client.truncation_events[0]
    report = tmp_path / "truncation.json"
    client.save_truncation_report(report)
    payload = json.loads(report.read_text())
    assert payload["event_count"] == 1
    assert payload["truncation_strategy"] == "head"


def test_embedding_trace_excludes_answer_position():
    record = RawTrajectoryRecord(
        trajectory_id="t0",
        task_id="task0",
        trial_index=0,
        instruction="Fill the result column.",
        instruction_type="Cell-Level Manipulation",
        answer_position="E6:E13",
        steps=[
            TrajectoryStep(
                step_id=1,
                raw_model_output="Thought: inspect the sheet",
                action="python inspect.py",
                observation="headers found",
            )
        ],
    )
    text = render_embedding_trace(record)
    assert "instruction: Fill the result column." in text
    assert "instruction_type: Cell-Level Manipulation" in text
    assert "raw_model_output" in text
    assert "answer_position" not in text
    assert "E6:E13" not in text


def test_local_pca_project_is_reproducible_for_randomized_solver_shape(monkeypatch):
    from dynamix_core import projection as projection_module

    values = np.random.default_rng(7).normal(size=(64, 600))
    config = ProjectionConfig(variance_ratio=0.9, max_dim=8, min_dim=2)
    solvers = []

    class TrackingPCA(projection_module.PCA):
        def fit_transform(self, values, y=None):
            projected = super().fit_transform(values, y)
            solvers.append(self._fit_svd_solver)
            return projected

    monkeypatch.setattr(projection_module, "PCA", TrackingPCA)

    first = local_pca_project(values, config, random_seed=42)
    second = local_pca_project(values, config, random_seed=42)

    np.testing.assert_array_equal(first.projected, second.projected)
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.mean, second.mean)
    assert solvers == ["randomized", "randomized"]

    wrapped = local_pca_project(values, config, random_seed=2**32 + 42)
    np.testing.assert_array_equal(first.projected, wrapped.projected)


def test_tree_builder_randomized_pca_is_reproducible_across_input_order():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    rng = np.random.default_rng(7)
    values = rng.normal(scale=0.05, size=(64, 600))
    values[:32, 0] -= 2.0
    values[32:, 0] += 2.0
    items = [
        ExperienceItem(
            item_id=f"e{index:03d}",
            level=1,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text=f"card {index}",
            embedding=row.tolist(),
        )
        for index, row in enumerate(values)
    ]
    config = default_hierarchy_config({
        "random_seed": 42,
        "projection": {"variance_ratio": 0.9, "max_dim": 8, "min_dim": 2},
        "gmm_bic": {
            "num_restarts": 2,
            "max_iter": 40,
            "min_split_size": 2,
            "min_effective_samples_per_component": 8,
            "abs_kmax": 4,
        },
    })

    first = asyncio.run(ProjectedGmmTreeBuilder(config).cluster_layer(items, level=1))
    second = asyncio.run(ProjectedGmmTreeBuilder(config).cluster_layer(list(reversed(items)), level=1))

    def snapshot(result):
        return {
            "chosen_k": result.chosen_k,
            "communities": [community.to_dict() for community in result.communities],
            "routing_model": result.routing_model.to_dict() if result.routing_model else None,
        }

    assert snapshot(first) == snapshot(second)


def test_budget_refinement_randomized_pca_is_reproducible():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    rng = np.random.default_rng(11)
    values = rng.normal(scale=0.05, size=(16, 600))
    values[:8, 0] -= 2.0
    values[8:, 0] += 2.0
    items = [
        ExperienceItem(
            item_id=f"t{index:03d}",
            level=0,
            kind=ITEM_KIND_TRAJECTORY,
            text=f"trace {index}",
            embedding=row.tolist(),
            metadata={"analysis_token_count": 20},
        )
        for index, row in enumerate(values)
    ]
    config = default_hierarchy_config({
        "random_seed": 2**32 + 42,
        "projection": {"variance_ratio": 0.9, "max_dim": 8, "min_dim": 2},
        "gmm_bic": {
            "num_restarts": 2,
            "max_iter": 40,
            "min_split_size": 2,
            "min_effective_samples_per_component": 8,
            "abs_kmax": 2,
        },
    })
    node = {
        "node_id": "L0_R0",
        "depth": 1,
        "path_weights": {item.item_id: 1.0 for item in items},
    }
    kwargs = {
        "node": node,
        "level": 0,
        "token_counts": {item.item_id: 20 for item in items},
        "parent_token_cost": 320,
        "token_budget": 100,
        "serial_start": 0,
    }

    first = asyncio.run(ProjectedGmmTreeBuilder(config)._budget_refinement_gmm_split(items, **kwargs))
    second = asyncio.run(ProjectedGmmTreeBuilder(config)._budget_refinement_gmm_split(items, **kwargs))

    assert first["accepted"] is True
    assert first == second



def test_chunked_embedding_uses_project_defaults_when_fields_omitted(tmp_path, monkeypatch):
    from dynamix_trace2skill import long_embeddings
    from dynamix_trace2skill.pipeline import DynaMixRunConfig, _embed_records_for_build

    class DummyTokenizer:
        def encode(self, text, *, add_special_tokens=False):
            return list(range(len(text.split())))

        def decode(self, ids, *, skip_special_tokens=True):
            return " ".join(f"tok{i}" for i in ids)

    monkeypatch.setattr(long_embeddings, "_load_hf_tokenizer", lambda tokenizer_model: DummyTokenizer())

    record = RawTrajectoryRecord(
        trajectory_id="t0",
        task_id="task0",
        trial_index=0,
        instruction="Do the task",
        instruction_type="Cell-Level Manipulation",
        steps=[TrajectoryStep(1, "raw", "action", "observation")],
    )
    embedding = EmbeddingClient(
        EmbeddingConfig(
            base_url="mock://deterministic",
            tokenizer_model="dummy-tokenizer",
            tokenizer_required=False,
            cache_path=str(tmp_path / "cache.sqlite"),
        )
    )
    config = DynaMixRunConfig(
        output_dir=str(tmp_path / "out"),
        records_path=str(tmp_path / "records.json"),
        embedding=embedding.config,
        chunked_embedding={"enabled": True},
    )

    asyncio.run(_embed_records_for_build(records=[record], embedding_client=embedding, config=config, out=tmp_path / "out"))
    report = json.loads((tmp_path / "out" / "analysis" / "chunked_embedding_report.json").read_text(encoding="utf-8"))
    assert report["chunk_tokens"] == 8000
    assert report["overlap_tokens"] == 1000


async def async_embed(client, texts):
    return await client.embed_texts(texts)


def _generation_debug_process_worker(debug_dir: str, label: str, barrier) -> None:
    client = GenerationClient(
        GenerationConfig(
            base_url="mock://deterministic",
            debug_dir=debug_dir,
            thinking_mode=False,
        )
    )
    barrier.wait()
    asyncio.run(
        client.chat_text(
            [{"role": "user", "content": f"Return only OK from {label}."}],
            debug_metadata={"worker": label},
        )
    )


def test_generation_debug_is_written_before_failed_request(tmp_path, monkeypatch):
    client = GenerationClient(GenerationConfig(base_url="http://example.invalid/v1", debug_dir=str(tmp_path)))

    def fail_request(*args, **kwargs):
        raise RuntimeError("simulated remote crash")

    monkeypatch.setattr(client, "_chat_text_sync", fail_request)
    with pytest.raises(RuntimeError, match="simulated remote crash"):
        asyncio.run(client.chat_text([{"role": "user", "content": "hello"}], debug_metadata={"community_id": "C0"}))

    debug_files = sorted(tmp_path.glob("generation_*.json"))
    assert len(debug_files) == 1
    payload = json.loads(debug_files[0].read_text())
    assert payload["status"] == "failed"
    assert payload["metadata"]["community_id"] == "C0"
    assert payload["messages"][0]["content"] == "hello"
    assert payload["error"]["type"] == "RuntimeError"


def test_generation_debug_write_failure_does_not_block_generation(tmp_path, monkeypatch, capsys):
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", debug_dir=str(tmp_path)))

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("dynamix_trace2skill.clients.Path.write_text", fail_write)
    text = asyncio.run(client.chat_text([{"role": "user", "content": "Return only OK."}]))

    assert text == "ACTION: TASK_COMPLETE"
    assert "dynamix-generation-debug-warning" in capsys.readouterr().err


def test_generation_debug_records_effective_timeout(tmp_path):
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", debug_dir=str(tmp_path), timeout_seconds=600.0))
    asyncio.run(client.chat_text([{"role": "user", "content": "Return only OK."}], timeout=12.5))

    payload = json.loads(next(tmp_path.glob("generation_*.json")).read_text())
    assert payload["request"]["timeout_seconds"] == 12.5


def test_chat_json_uses_response_format_json_schema():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic"))
    seen = {}
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}

    async def fake_chat_text(messages, **kwargs):
        seen.update(kwargs)
        return '{"cards": []}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            max_tokens=1234,
            retries=0,
        )
    )

    assert result == {"cards": []}
    assert seen["extra_body"] == {}
    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "MinimalClusterExperienceCards",
            "strict": True,
            "schema": schema,
        },
    }
    assert seen["max_tokens"] == 1234


def test_generation_debug_marks_outer_timeout(tmp_path, monkeypatch):
    client = GenerationClient(GenerationConfig(base_url="http://example.invalid/v1", debug_dir=str(tmp_path), timeout_seconds=30.0))

    def slow_request(*args, **kwargs):
        time.sleep(0.2)
        return "late response"

    monkeypatch.setattr(client, "_chat_text_sync", slow_request)
    with pytest.raises(TimeoutError, match="generation request exceeded timeout_seconds"):
        asyncio.run(client.chat_text([{"role": "user", "content": "hello"}], timeout=0.01))

    payload = json.loads(next(tmp_path.glob("generation_*.json")).read_text())
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "TimeoutError"
    assert payload["request"]["timeout_seconds"] == 0.01


def test_chat_json_rejects_embedded_json_when_guided_schema_is_requested():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic"))
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}

    async def fake_chat_text(messages, **kwargs):
        return 'Here is the JSON you requested:\n{"cards": []}'

    client.chat_text = fake_chat_text
    with pytest.raises(ValueError, match="failed to parse JSON"):
        asyncio.run(
            client.chat_json(
                [{"role": "user", "content": "return cards"}],
                schema_name="MinimalClusterExperienceCards",
                guided_json=schema,
                retries=0,
            )
        )


def test_generation_debug_reuses_succeeded_response_and_continues_numbering(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "summarize C0"}]
    cached_payload = {
        "status": "succeeded",
        "metadata": {"community_id": "C0"},
        "request": {
            "model": "Qwen3.5-9B",
            "base_url": "http://example.invalid/v1",
            "api_key": "EMPTY",
            "temperature": 0.6,
            "max_tokens": None,
            "timeout_seconds": 600.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "response_format": None,
        },
        "messages": messages,
        "response": "{\"cards\": []}",
    }
    (tmp_path / "generation_00007.json").write_text(json.dumps(cached_payload), encoding="utf-8")
    client = GenerationClient(
        GenerationConfig(
            base_url="http://example.invalid/v1",
            debug_dir=str(tmp_path),
            timeout_seconds=600.0,
            thinking_mode=False,
        )
    )

    def fail_if_called(*args, **kwargs):
        raise RuntimeError("remote should not be called for cached generation")

    monkeypatch.setattr(client, "_chat_text_sync", fail_if_called)
    text = asyncio.run(client.chat_text(messages, debug_metadata={"community_id": "C0"}))
    assert text == "{\"cards\": []}"
    assert not (tmp_path / "generation_00008.json").exists()

    monkeypatch.setattr(client, "_chat_text_sync", lambda *args, **kwargs: "fresh")
    fresh = asyncio.run(client.chat_text([{"role": "user", "content": "summarize C1"}], debug_metadata={"community_id": "C1"}))
    assert fresh == "fresh"
    fresh_payload = json.loads((tmp_path / "generation_00008.json").read_text())
    assert fresh_payload["status"] == "succeeded"
    assert fresh_payload["metadata"]["community_id"] == "C1"


def test_generation_debug_cache_identity_includes_api_key_fingerprint(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "summarize C0"}]
    cached_payload = {
        "status": "succeeded",
        "metadata": {"community_id": "C0"},
        "request": {
            "model": "Qwen3.5-9B",
            "base_url": "http://example.invalid/v1",
            "api_key": "sha256:old-key",
            "temperature": 0.6,
            "max_tokens": None,
            "timeout_seconds": 600.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "response_format": None,
        },
        "messages": messages,
        "response": "{\"cards\": []}",
    }
    (tmp_path / "generation_00001.json").write_text(json.dumps(cached_payload), encoding="utf-8")
    client = GenerationClient(
        GenerationConfig(
            base_url="http://example.invalid/v1",
            api_key="new-key",
            debug_dir=str(tmp_path),
            timeout_seconds=600.0,
            thinking_mode=False,
        )
    )

    monkeypatch.setattr(client, "_chat_text_sync", lambda *args, **kwargs: "fresh")
    text = asyncio.run(client.chat_text(messages, debug_metadata={"community_id": "C0"}))

    assert text == "fresh"
    payload = json.loads((tmp_path / "generation_00002.json").read_text())
    assert payload["request"]["api_key"].startswith("sha256:")
    assert "new-key" not in json.dumps(payload)


def test_generation_debug_numbering_is_cross_process_safe(tmp_path):
    (tmp_path / "generation_00007.json").write_text(
        json.dumps({"status": "succeeded", "metadata": {"seed": True}, "request": {}, "messages": [], "response": "cached"}),
        encoding="utf-8",
    )
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    barrier = ctx.Barrier(2, timeout=10)
    processes = [
        ctx.Process(target=_generation_debug_process_worker, args=(str(tmp_path), f"p{index}", barrier))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]
    payloads = {
        path.name: json.loads(path.read_text())
        for path in sorted(tmp_path.glob("generation_*.json"))
    }
    assert "generation_00008.json" in payloads
    assert "generation_00009.json" in payloads
    workers = sorted(
        payload["metadata"].get("worker")
        for name, payload in payloads.items()
        if name in {"generation_00008.json", "generation_00009.json"}
    )
    assert workers == ["p0", "p1"]


def test_cluster_analyst_passes_generation_debug_metadata(tmp_path):
    class DummyGeneration:
        def __init__(self):
            self.kwargs = None

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.kwargs = kwargs
            return {
                "cards": [{
                    "name": "Specific lesson",
                    "trigger": "When a task has this pattern.",
                    "content": "Use the observed procedure.",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.8,
                }],
            }

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[1.0] for _ in texts]

    generation = DummyGeneration()
    analyst = ClusterAnalyst(
        generation,
        DummyEmbedding(),
        ClusterAnalystConfig(
            tokenizer_required=False,
            allow_regex_tokenizer_fallback=True,
            max_prompt_tokens=100000,
            prompt_token_report_path=str(tmp_path / "tokens.json"),
        ),
    )
    community = ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t0": 1.0}, success_count=1, outcome_mode="success")
    member = ExperienceItem(
        item_id="t0",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_bundle": "short evidence", "task_id": "13-1", "success": True},
    )
    asyncio.run(analyst.summarize(community, [member]))

    metadata = generation.kwargs["debug_metadata"]
    assert metadata["community_id"] == "L0_C0"
    assert metadata["analyst_mode"] == "raw_extractor"
    assert metadata["prompt_token_event"]["prompt_tokens"] > 0
    assert metadata["members"][0]["task_id"] == "13-1"
    assert metadata["members"][0]["success"] is True


def test_embedding_raises_when_truncation_disabled(tmp_path):
    cfg = EmbeddingConfig(base_url="mock://deterministic", max_input_tokens=10, truncate_long_texts=False, tokenizer_required=False)
    client = EmbeddingClient(cfg)
    with pytest.raises(ValueError):
        asyncio.run(async_embed(client, ["x " * 20]))


def test_cluster_analyst_uses_all_members_not_member_cap():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C0", level=0, member_weights={f"t{i}": 1.0 for i in range(20)})
    members = [ExperienceItem(item_id=f"t{i}", level=0, kind=ITEM_KIND_TRAJECTORY, text=f"trace {i}", embedding=[1.0], metadata={"analysis_bundle": f"bundle {i}"}) for i in range(20)]
    prompt = analyst._build_prompt(community, members, "raw_extractor")
    payload = json.loads(prompt)
    assert len(payload["members"]) == 20
    assert "Use all provided members" in " ".join(payload["hard_constraints"])
    assert "success_user_template" in payload["template_user_prompt_adaptation"]


def test_default_hierarchy_config_is_real_not_tiny_smoke():
    from dynamix_core.config import ProjectedGmmDynamicTreeConfig

    cfg = default_hierarchy_config({})
    assert cfg.tree_policy == "projected_gmm_bic"
    assert cfg.gmm_bic.min_split_size == 2
    assert cfg.gmm_bic.min_effective_samples_per_component == 2
    assert cfg.gmm_bic.abs_kmax == 64
    assert cfg.gmm_bic.num_restarts == 5
    assert cfg.soft_membership.recursive_assignment == "cumulative_mass"
    assert cfg.soft_membership.cumulative_mass_coverage == pytest.approx(0.90)
    assert cfg.dynamic_update.mode == "budget_constrained_online_gmm"
    assert cfg.dynamic_update.assignment == "cumulative_mass"
    assert cfg.dynamic_update.cumulative_mass_coverage == pytest.approx(0.90)
    assert cfg.budget_refinement.fallback == "gmm_bic_recursive"
    direct_cfg = ProjectedGmmDynamicTreeConfig.from_mapping({})
    assert direct_cfg.tree_policy == "projected_gmm_bic"
    assert direct_cfg.soft_membership.recursive_assignment == "cumulative_mass"
    assert direct_cfg.soft_membership.cumulative_mass_coverage == pytest.approx(0.90)
    assert direct_cfg.dynamic_update.mode == "budget_constrained_online_gmm"
    with pytest.raises(ValueError, match="update_routing_model must be true"):
        ProjectedGmmDynamicTreeConfig.from_mapping({"dynamic_update": {"update_routing_model": False}})
    with pytest.raises(ValueError, match="budget_constrained_online_gmm"):
        ProjectedGmmDynamicTreeConfig.from_mapping({"dynamic_update": {"mode": "fixed_k_online_em"}})


def test_default_dynamic_protocol_is_train200_sixty_forty_batched_arrivals():
    cfg = DynaMixRunConfig(output_dir="out", records_path="records.json")
    assert cfg.dynamic.initial_count == 120
    assert cfg.dynamic.arrival_count == 80
    assert cfg.dynamic.initial_count + cfg.dynamic.arrival_count == 200
    assert cfg.dynamic.update_batch_size == 8
    assert cfg.dynamic.shuffle_seed == 42
    assert cfg.dynamic.snapshot_include_embeddings is True
    assert cfg.dynamic.resume_from_snapshots is False


def test_tracked_qwen_train200_config_uses_current_dynamic_arrival_schema():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "qwen_train200_tree_main_001.json"
    payload = json.loads(cfg_path.read_text())
    cfg = DynaMixRunConfig.from_json(cfg_path)
    assert cfg.dynamic.initial_count == 120
    assert cfg.dynamic.arrival_count == 80
    assert cfg.dynamic.update_batch_size == 8
    assert cfg.dynamic.shuffle_seed == 42
    assert cfg.dynamic.snapshot_include_embeddings is True
    assert cfg.dynamic.resume_from_snapshots is False


def _trajectory(item_id: str, embedding: list[float], tokens: int) -> ExperienceItem:
    return ExperienceItem(
        item_id=item_id,
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text=f"trace {item_id}",
        embedding=embedding,
        metadata={"analysis_token_count": tokens, "success": True},
    )


def _card(item_id: str, source_community_id: str, embedding: list[float]) -> ExperienceItem:
    return ExperienceItem(
        item_id=item_id,
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text=f"name: {item_id}\ntrigger: synthetic\ncontent: synthetic",
        embedding=embedding,
        generated_from_community_ids=[source_community_id],
        metadata={"confidence": 1.0, "name": item_id, "trigger": "synthetic", "content": "synthetic"},
    )


async def _build_two_l0_state(*, c1_tokens: int, c2_tokens: int) -> ExperienceHierarchyState:
    state = ExperienceHierarchyState()
    await state.initialize_trajectory_items([
        _trajectory("old_c1", [1.0, 0.0], c1_tokens),
        _trajectory("old_c2", [0.0, 1.0], c2_tokens),
    ])
    await state.commit_layer(
        level=0,
        communities=[
            ExperienceCommunity("L0_C1", 0, {"old_c1": 1.0}, posterior_member_weights={"old_c1": 1.0}),
            ExperienceCommunity("L0_C2", 0, {"old_c2": 1.0}, posterior_member_weights={"old_c2": 1.0}),
        ],
        generated_items=[
            _card("card_c1", "L0_C1", [1.0, 0.0]),
            _card("card_c2", "L0_C2", [0.0, 1.0]),
        ],
        metadata={
            "routing_model": {
                "routing_model_kind": "pca_gmm",
                "community_ids": ["L0_C1", "L0_C2"],
                "pca_mean": [0.0, 0.0],
                "pca_components": [[1.0, 0.0], [0.0, 1.0]],
                "pi": [0.5, 0.5],
                "means": [[1.0, 0.0], [0.0, 1.0]],
                "variances": [[1.0, 1.0], [1.0, 1.0]],
                "total_effective_count": 2.0,
                "component_effective_counts": [1.0, 1.0],
            }
        },
    )
    return state


def _budgeted_dynamic_config():
    return default_hierarchy_config({
        "summary_budget": {
            "max_model_tokens": 100,
            "budget_ratio": 1.0,
            "prompt_overhead_reserve_tokens": 50,
            "token_count_metadata_keys": ["analysis_token_count"],
        },
        "dynamic_update": {
            "assignment": "cumulative_mass",
            "top_r": 2,
            "max_membership_gap": 1.0,
            "cumulative_mass_coverage": 0.9,
        },
    })


async def _synthetic_l0_patch(context):
    previous = list(context.previous_generated_experiences or [])
    if previous:
        item_id = previous[0]["item_id"]
        return [
            ExperienceCardPatch(
                operation="update",
                item_id=item_id,
                text=f"name: updated {item_id}\ntrigger: synthetic\ncontent: updated",
                embedding=[0.5, 0.5],
                metadata={"confidence": 1.0, "name": f"updated {item_id}", "trigger": "synthetic", "content": "updated"},
            )
        ]
    item_id = f"card_{context.community.community_id}"
    return [
        ExperienceCardPatch(
            operation="add",
            item_id=item_id,
            text=f"name: {item_id}\ntrigger: synthetic\ncontent: new dynamic card",
            embedding=[0.5, 0.5],
            metadata={"confidence": 1.0, "name": item_id, "trigger": "synthetic", "content": "new dynamic card"},
        )
    ]


async def _support_only_l0_patch(context):
    previous = list(context.previous_generated_experiences or [])
    assert previous
    old = previous[0]
    return [
        ExperienceCardPatch(
            operation="update",
            item_id=old["item_id"],
            text=old["text"],
            embedding=[0.0, 1.0],
            metadata={**dict(old.get("metadata") or {}), "confidence": 1.0},
        )
    ]


def test_dynamic_l0_budget_gate_tries_next_candidate_before_growing_k():
    state = asyncio.run(_build_two_l0_state(c1_tokens=45, c2_tokens=20))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 10)],
            dynamic_summary_fn=_synthetic_l0_patch,
        )
    )
    assignments = asyncio.run(state.communities_for_item("new_t"))
    posterior = asyncio.run(state.posterior_communities_for_item("new_t"))
    metadata = asyncio.run(state.layer_metadata(0))

    assert assignments == {"L0_C2": pytest.approx(0.5)}
    assert posterior == {"L0_C1": pytest.approx(0.5), "L0_C2": pytest.approx(0.5)}
    assert not result.reroute_results[0].new_community_ids
    assert metadata["routing_model"]["community_ids"] == ["L0_C1", "L0_C2"]
    assert metadata["routing_model"]["component_effective_counts"] == [pytest.approx(1.5), pytest.approx(1.5)]


def test_dynamic_l0_budget_gate_uses_dynamic_prompt_token_estimator():
    state = asyncio.run(_build_two_l0_state(c1_tokens=1, c2_tokens=1))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    calls = []

    async def estimator(community, members, previous_generated_experiences):
        calls.append({
            "community_id": community.community_id,
            "member_ids": [item.item_id for item in members],
            "previous_ids": [card.get("item_id") for card in previous_generated_experiences],
        })
        if community.community_id == "L0_C1":
            return 101
        return 50

    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 1)],
            dynamic_summary_fn=_synthetic_l0_patch,
            dynamic_prompt_token_estimator=estimator,
        )
    )
    assignments = asyncio.run(state.communities_for_item("new_t"))

    assert assignments == {"L0_C2": pytest.approx(0.5)}
    assert not result.reroute_results[0].new_community_ids
    assert any(call["community_id"] == "L0_C1" and call["previous_ids"] == ["card_c1"] for call in calls)
    assert any(call["community_id"] == "L0_C2" and "new_t" in call["member_ids"] for call in calls)


def test_dynamic_l0_budget_gate_uses_explicit_prompt_budget_override():
    state = asyncio.run(_build_two_l0_state(c1_tokens=1, c2_tokens=1))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())

    async def estimator(community, members, previous_generated_experiences):
        if community.community_id == "L0_C1":
            return 60
        if community.community_id == "L0_C2":
            return 40
        return 40

    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 1)],
            dynamic_summary_fn=_synthetic_l0_patch,
            dynamic_prompt_token_estimator=estimator,
            dynamic_prompt_token_budget=50,
        )
    )
    assignments = asyncio.run(state.communities_for_item("new_t"))

    assert assignments == {"L0_C2": pytest.approx(0.5)}
    assert not result.reroute_results[0].new_community_ids


def test_dynamic_l0_budget_gate_keeps_all_fitting_static_soft_parents():
    state = asyncio.run(_build_two_l0_state(c1_tokens=20, c2_tokens=20))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 10)],
            dynamic_summary_fn=_synthetic_l0_patch,
        )
    )
    assignments = asyncio.run(state.communities_for_item("new_t"))

    assert assignments == {"L0_C1": pytest.approx(0.5), "L0_C2": pytest.approx(0.5)}
    assert not result.reroute_results[0].new_community_ids


def test_dynamic_support_only_update_clears_pending_reroute_for_sequential_item():
    state = asyncio.run(_build_two_l0_state(c1_tokens=45, c2_tokens=20))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 10)],
            dynamic_summary_fn=_support_only_l0_patch,
        )
    )

    assert result.validation["ok"]
    assert asyncio.run(state.validate_hierarchy(require_no_pending_reroute=True))["ok"]


def test_dynamic_l0_budget_gate_grows_new_component_when_all_candidates_overflow():
    state = asyncio.run(_build_two_l0_state(c1_tokens=45, c2_tokens=45))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 10)],
            dynamic_summary_fn=_synthetic_l0_patch,
        )
    )
    assignments = asyncio.run(state.communities_for_item("new_t"))
    new_ids = result.reroute_results[0].new_community_ids
    metadata = asyncio.run(state.layer_metadata(0))
    new_community = asyncio.run(state.community_objects(new_ids))[0]

    assert len(new_ids) == 1
    assert new_ids[0].startswith("L0_DYN_")
    assert assignments == {new_ids[0]: pytest.approx(1.0)}
    assert new_ids[0] in metadata["routing_model"]["community_ids"]
    assert metadata["routing_model"]["grow_k_components_added"] == 1
    assert new_community.generated_item_ids == [f"card_{new_ids[0]}"]
    assert new_community.metadata["split_reason"] == "dynamic_l0_budget_overflow_new_component"
    assert new_community.metadata["rejected_candidate_posterior_weights"] == {"L0_C1": pytest.approx(0.5), "L0_C2": pytest.approx(0.5)}
    snapshot = asyncio.run(state.to_dict(include_embeddings=False, validate=True))
    assert new_ids[0] in snapshot["layers"]["0"]["community_ids"]
    assert snapshot["validation"]["ok"]


def test_dynamic_grow_k_requires_saved_routing_model_not_bootstrap():
    state = asyncio.run(_build_two_l0_state(c1_tokens=45, c2_tokens=45))
    asyncio.run(state.update_layer_metadata(0, {}))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    community = ExperienceCommunity("L0_DYN_missing_model", 0, {"new_t": 1.0}, posterior_member_weights={"new_t": 1.0})

    with pytest.raises(ValueError, match="no routing_model is saved"):
        asyncio.run(
            updater._append_routing_component(
                state,
                level=0,
                community=community,
                seed_item=_trajectory("new_t", [0.5, 0.5], 10),
            )
        )

    metadata = asyncio.run(state.layer_metadata(0))
    assert "routing_model" not in metadata


def test_dynamic_update_requires_saved_routing_model_before_inserting_even_with_refinement_tree():
    state = asyncio.run(_build_two_l0_state(c1_tokens=45, c2_tokens=45))
    asyncio.run(
        state.update_layer_metadata(
            0,
            {
                "budget_refinement": {
                    "refinement_routing_tree": {
                        "coarse_roots": {"L0_C1": "root"},
                        "nodes": {"root": {"node_id": "root", "kind": "leaf", "community_id": "L0_C1"}},
                    }
                }
            },
        )
    )
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())

    with pytest.raises(ValueError, match="requires saved routing_model"):
        asyncio.run(
            updater.update(
                state=state,
                new_trajectory_items=[_trajectory("new_t", [0.5, 0.5], 10)],
                dynamic_summary_fn=_synthetic_l0_patch,
            )
        )

    snapshot = asyncio.run(state.to_dict(include_embeddings=False, validate=True))
    assert "new_t" not in snapshot["items"]


def test_dynamic_oversize_arrival_is_recorded_as_excluded_not_inserted():
    state = asyncio.run(_build_two_l0_state(c1_tokens=20, c2_tokens=20))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    result = asyncio.run(
        updater.update(
            state=state,
            new_trajectory_items=[_trajectory("too_long", [0.5, 0.5], 51)],
            dynamic_summary_fn=_synthetic_l0_patch,
        )
    )
    snapshot = asyncio.run(state.to_dict(include_embeddings=False, validate=True))

    assert result.inserted_item_ids == []
    assert result.excluded_item_ids == ["too_long"]
    assert result.excluded_oversize_singletons == [
        {
            "item_id": "too_long",
            "source_community_id": None,
            "token_cost": 51,
            "budget": 50,
            "reason": "oversize_singleton",
            "dynamic_arrival": True,
        }
    ]
    assert "too_long" not in snapshot["items"]


def test_dynamic_oversize_arrival_still_requires_valid_l0_routing_model():
    state = asyncio.run(_build_two_l0_state(c1_tokens=20, c2_tokens=20))
    asyncio.run(state.update_layer_metadata(0, {}))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())

    with pytest.raises(ValueError, match="requires saved routing_model"):
        asyncio.run(
            updater.update(
                state=state,
                new_trajectory_items=[_trajectory("too_long", [0.5, 0.5], 51)],
                dynamic_summary_fn=_synthetic_l0_patch,
            )
        )

    snapshot = asyncio.run(state.to_dict(include_embeddings=False, validate=True))
    assert "too_long" not in snapshot["items"]


def test_dynamic_contribution_cache_initialization_preserves_static_routing_parameters():
    state = asyncio.run(_build_two_l0_state(c1_tokens=20, c2_tokens=20))
    updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
    before = asyncio.run(state.layer_metadata(0))["routing_model"]

    asyncio.run(updater._ensure_layer_routing_contributions(state, 0))
    after = asyncio.run(state.layer_metadata(0))["routing_model"]

    for key in ["pi", "means", "variances", "component_effective_counts", "total_effective_count"]:
        assert after[key] == before[key]
    assert after["item_contributions_initialized"] is True
    assert after["item_contributions_source"] == "existing_state_preserve_routing_parameters"
    assert sorted(after["item_contributions"]) == ["old_c1", "old_c2"]


def test_dynamic_reroute_requires_model_for_non_terminal_upper_layer():
    async def run_case():
        state = await _build_two_l0_state(c1_tokens=20, c2_tokens=20)
        await state.commit_layer(
            level=1,
            communities=[ExperienceCommunity("L1_C0", 1, {"card_c1": 1.0, "card_c2": 1.0}, posterior_member_weights={"card_c1": 1.0, "card_c2": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="card_l2",
                    level=2,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="name: L2\ntrigger: synthetic\ncontent: synthetic",
                    embedding=[0.5, 0.5],
                    generated_from_community_ids=["L1_C0"],
                    metadata={"confidence": 1.0, "name": "L2", "trigger": "synthetic", "content": "synthetic"},
                )
            ],
            metadata={},
        )
        updater = ExperienceHierarchyDynamicUpdater(_budgeted_dynamic_config())
        await updater._propagate_reroute_items(state, ["card_c1"])

    with pytest.raises(ValueError, match="non-terminal level 1"):
        asyncio.run(run_case())


def test_experiment_runner_tree_resume_requires_matching_fingerprint(tmp_path):
    runner = _load_experiment_runner_module()
    marker = tmp_path / "04_build_tree.done"
    output = tmp_path / "summary.json"
    output.write_text("{}", encoding="utf-8")
    marker.write_text(json.dumps({"fingerprint": {"scenario": "dynamic_update"}}), encoding="utf-8")

    assert runner.stage_done(marker, [output], fingerprint={"scenario": "dynamic_update"})
    assert not runner.stage_done(marker, [output], fingerprint={"scenario": "static_build"})
    marker.write_text(json.dumps({"stage": "04_build_tree"}), encoding="utf-8")
    assert not runner.stage_done(marker, [output], fingerprint={"scenario": "dynamic_update"})


def test_experiment_runner_reuse_tree_requires_reused_train_artifacts(tmp_path, monkeypatch):
    runner = _load_experiment_runner_module()
    tree = tmp_path / "baseline_tree"
    tree.mkdir()
    (tree / "hierarchy_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "run_dynamix_trace2skill_experiment.py",
        "--data-path", str(tmp_path / "data"),
        "--run-dir", str(tmp_path / "run"),
        "--reuse-tree-dir", str(tree),
        "--model", "mock-model",
        "--openai-base-url", "mock://generation",
        "--embedding-base-url", "mock://embedding",
        "--embedding-model", "mock-embed",
        "--embedding-tokenizer", "mock-tokenizer",
        "--python-executable", sys.executable,
    ])
    with pytest.raises(RuntimeError, match="--reuse-tree-dir requires --records-path or --reuse-train-run-dir"):
        runner.main()


def test_experiment_runner_stage_report_aggregates_time_tokens_and_budget_pressure(tmp_path):
    runner = _load_experiment_runner_module()
    marker_dir = tmp_path / "stage_markers"
    marker_dir.mkdir()
    (marker_dir / "01_train_collect.done").write_text(
        json.dumps({
            "stage": "01_train_collect",
            "started_at": "2026-06-19T00:00:00Z",
            "ended_at": "2026-06-19T00:02:00Z",
            "elapsed_seconds": 120.0,
            "log": str(tmp_path / "logs" / "01_train_collect.log"),
            "outputs": [],
        }),
        encoding="utf-8",
    )
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    usage_log = usage_dir / "01_train_collect.react_usage.jsonl"
    usage_log.write_text(
        "\n".join([
            json.dumps({"cache_hit": False, "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}}),
            json.dumps({"cache_hit": False, "usage": {}}),
            json.dumps({"cache_hit": True, "usage": {}}),
        ]) + "\n",
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "dynamix_tree" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "cluster_prompt_token_report.json").write_text(
        json.dumps({"events": [{"community_id": "L0_C0", "level": 0, "member_count": 2, "prompt_tokens": 84, "max_prompt_tokens": 85, "over_budget": False}]}),
        encoding="utf-8",
    )
    (analysis_dir / "chunked_embedding_report.json").write_text(
        json.dumps({"chunk_tokens": 8000, "overlap_tokens": 1000, "pooling": "mean", "max_token_count": 90000, "over_limit_chunk_count": 0}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        summary_max_model_tokens=100000,
        summary_budget_ratio=0.85,
        summary_prompt_overhead_reserve_tokens=8000,
        analyst_max_prompt_tokens=-1,
        budget_refinement_apply_to_level=0,
        soft_recursive_assignment="cumulative_mass",
        soft_top_r_memberships=2,
        soft_cumulative_mass_coverage=0.90,
        soft_max_membership_gap=0.25,
        workers=8,
        thinking="true",
        generation_timeout_seconds=600,
        rollout_client_timeout_seconds=600,
        chunked_embedding_enabled=True,
        embedding_batch_size=8,
        chunked_embedding_chunk_tokens=8000,
        embedding_max_model_len=32000,
        train_start=0,
        train_end=200,
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
        tree_scenario="dynamic_update",
    )

    report = runner.write_experiment_stage_report(
        run_dir=tmp_path,
        marker_dir=marker_dir,
        stages=["01_train_collect"],
        usage_logs_by_stage={"01_train_collect": [usage_log]},
        runtime={"run_dir": str(tmp_path)},
        args=args,
    )

    assert report["stages"][0]["elapsed_seconds"] == 120.0
    assert report["stages"][0]["token_totals"]["prompt_tokens"] == 100
    assert report["stages"][0]["token_totals"]["completion_tokens"] == 25
    assert report["stages"][0]["usage_logs"][0]["provider_usage_status"] == "partial"
    assert report["stages"][0]["usage_logs"][0]["call_source_status"] == "mixed_cached_partial"
    assert report["stages"][0]["usage_summary"]["records_without_usage"] == 1
    assert report["prompt_token_stats"]["near_configured_limit_count"] == 1
    assert (tmp_path / "experiment_stage_report.json").exists()
    assert (tmp_path / "experiment_stage_report.md").exists()


def test_experiment_runner_rejects_wrong_tree_summary_before_heldout():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_scenario="dynamic_update",
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
    )
    runner.validate_tree_summary_for_heldout(
        {"scenario": "dynamic_update", "record_count": 200, "initial_count": 120, "arrival_count": 80, "updated_count": 80, "excluded_count": 0, "insertion_count": 80},
        args,
    )
    runner.validate_tree_summary_for_heldout(
        {"scenario": "dynamic_update", "record_count": 200, "initial_count": 120, "arrival_count": 80, "updated_count": 79, "excluded_count": 1, "insertion_count": 80},
        args,
    )
    with pytest.raises(RuntimeError, match="scenario mismatch"):
        runner.validate_tree_summary_for_heldout(
            {"scenario": "static_build", "record_count": 200},
            args,
        )
    with pytest.raises(RuntimeError, match="dynamic summary mismatch"):
        runner.validate_tree_summary_for_heldout(
            {"scenario": "dynamic_update", "record_count": 200, "initial_count": 160, "arrival_count": 40, "updated_count": 40, "excluded_count": 0, "insertion_count": 40},
            args,
        )
    with pytest.raises(RuntimeError, match="insertion accounting mismatch"):
        runner.validate_tree_summary_for_heldout(
            {"scenario": "dynamic_update", "record_count": 200, "initial_count": 120, "arrival_count": 80, "updated_count": 78, "excluded_count": 1, "insertion_count": 80},
            args,
        )


def test_experiment_runner_reorders_reused_records_to_dataset_train_order(tmp_path):
    runner = _load_experiment_runner_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(
        json.dumps([
            {"id": "b", "instruction": "task b"},
            {"id": "a", "instruction": "task a"},
            {"id": "c", "instruction": "task c"},
        ]),
        encoding="utf-8",
    )
    source_records = tmp_path / "records.json"
    source_records.write_text(
        json.dumps([
            {"task_id": "a", "trajectory_id": "a"},
            {"task_id": "b", "trajectory_id": "b"},
            {"task_id": "c", "trajectory_id": "c"},
        ]),
        encoding="utf-8",
    )

    manifest = runner.write_dataset_ordered_records(
        source_records=source_records,
        data_path=data_dir,
        output_path=tmp_path / "ordered_records.json",
        manifest_path=tmp_path / "records_order_manifest.json",
        train_start=0,
        train_end=3,
    )

    ordered = json.loads((tmp_path / "ordered_records.json").read_text(encoding="utf-8"))
    assert [row["task_id"] for row in ordered] == ["b", "a", "c"]
    assert manifest["source_order_equal_dataset_order"] is False
    assert manifest["first_task_ids"] == ["b", "a", "c"]


def test_pipeline_orders_records_before_dynamic_split(tmp_path):
    import dynamix_trace2skill.pipeline as pipeline
    from dynamix_trace2skill.schemas import RawTrajectoryRecord

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(
        json.dumps([{"id": "b"}, {"id": "a"}, {"id": "c"}]),
        encoding="utf-8",
    )
    records = [
        RawTrajectoryRecord(trajectory_id="a", task_id="a", trial_index=0, instruction="a"),
        RawTrajectoryRecord(trajectory_id="b", task_id="b", trial_index=0, instruction="b"),
        RawTrajectoryRecord(trajectory_id="c", task_id="c", trial_index=0, instruction="c"),
    ]

    ordered, manifest = pipeline._order_records_by_dataset_slice(
        records=records,
        dataset_path=data_dir,
        train_start=0,
        train_end=3,
        source_records_path=tmp_path / "records.json",
    )

    assert [record.task_id for record in ordered] == ["b", "a", "c"]
    assert [record.task_id for record in ordered[:2]] == ["b", "a"]
    assert [record.task_id for record in ordered[2:]] == ["c"]
    assert manifest["source_order_equal_dataset_order"] is False


def test_log_parser_recurses_and_loads_multiple_result_shapes(tmp_path):
    log_dir = tmp_path / "logs" / "seed_7"
    log_dir.mkdir(parents=True)
    (log_dir / "cli_only_agent_123__trial_2_seed_7.md").write_text(
        "## [1] SYSTEM\nSys\n---\n## [2] USER\n### instruction\nDo task\n### instruction_type\nCell-Level Manipulation\n### answer_position\nA1\n### spreadsheet_path\n/x/123/input.xlsx\n### output_path\n/y/out.xlsx\n---\n## [3] ASSISTANT\nThought: done\n\nACTION: TASK_COMPLETE\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "run_seed_7.json").write_text(json.dumps({"results": [{"id": "123", "trial_index": 2, "seed": 7, "success": True}]}), encoding="utf-8")
    records = parse_trace2skill_logs(tmp_path / "logs", results_file=results_dir)
    assert len(records) == 1
    assert records[0].task_id == "123"
    assert records[0].trial_index == 2
    assert "trial2" in records[0].trajectory_id
    assert "seed7" in records[0].trajectory_id
    assert records[0].success is True


def test_official_eval_passed_fields_are_parsed():
    result = {
        "id": "abc",
        "test_cases": [
            {"gt_file": "1_golden.xlsx", "output_file": "1_output.xlsx", "passed": True, "message": ""},
            {"gt_file": "2_golden.xlsx", "output_file": "2_output.xlsx", "passed": False, "message": "wrong cell"},
        ],
        "passed_count": 1,
        "total_count": 2,
        "soft_score": 0.5,
        "hard_score": 0,
    }
    success, score, feedback = _result_fields(result)
    assert success is False
    assert score == 0.5
    assert "wrong cell" in (feedback or "")


def test_prompt_budget_token_mass_is_unweighted():
    from dynamix_core.tree_builder import _community_token_mass
    assert _community_token_mass({"a": 100, "b": 200}, {"a": 0.01, "b": 0.02}) == 300.0


def test_cluster_prompt_preflight_records_tokens(tmp_path):
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, max_prompt_tokens=100000, prompt_token_report_path=str(tmp_path / "prompt_tokens.json")))  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_budget", level=0, member_weights={"t0": 1.0})
    members = [ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "bundle text"})]
    system = analyst._system_prompt("raw_extractor")
    user = analyst._build_prompt(community, members, "raw_extractor")
    analyst._preflight_prompt_budget(community, system, user, len(members))
    report = json.loads((tmp_path / "prompt_tokens.json").read_text())
    assert report["events"][0]["community_id"] == "C_budget"
    assert report["events"][0]["prompt_tokens"] > 0


def test_cluster_prompt_preflight_fails_when_over_budget():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, max_prompt_tokens=1))  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_too_big", level=0, member_weights={"t0": 1.0})
    members = [ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "many tokens here"})]
    with pytest.raises(ValueError):
        analyst._preflight_prompt_budget(
            community,
            analyst._system_prompt("raw_extractor"),
            analyst._build_prompt(community, members, "raw_extractor"),
            len(members),
        )


def test_budget_refinement_excludes_oversize_singleton_from_active_layer():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 10, "budget_ratio": 0.5},
        "budget_refinement": {
            "enabled": True,
            "apply_to_level": 0,
            "selection_policy": "bic_best_with_token_progress",
            "min_token_reduction_fraction": 0.10,
            "fallback": "gmm_bic_recursive",
            "flatten_refinement_leaves_to_l0": True,
            "skip_oversize_singleton": True,
        },
    })
    item = ExperienceItem(
        item_id="too_long",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_token_count": 10, "analysis_bundle": "oversize bundle"},
    )
    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer([item], level=0))
    assert clustering.should_stop
    assert clustering.stop_reason == "budget_refinement_no_active_communities"
    assert clustering.communities == []
    assert clustering.excluded_input_item_ids == ["too_long"]
    skipped = clustering.summary_budget["excluded_oversize_singletons"]
    assert len(skipped) == 1
    assert skipped[0]["item_id"] == "too_long"
    assert skipped[0]["reason"] == "oversize_singleton"


def test_budget_refinement_falls_back_to_token_packing_when_gmm_cannot_split():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5},
        "gmm_bic": {"min_split_size": 99, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0], metadata={"analysis_token_count": 30}),
        ExperienceItem(item_id="b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.9, 0.1], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="c", level=0, kind=ITEM_KIND_TRAJECTORY, text="c", embedding=[0.0, 1.0], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="d", level=0, kind=ITEM_KIND_TRAJECTORY, text="d", embedding=[0.1, 0.9], metadata={"analysis_token_count": 10}),
    ]
    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))
    assert clustering.excluded_input_item_ids == []
    assert len(clustering.communities) == 2
    assert sorted(sum(community.member_weights.values()) for community in clustering.communities) == [2.0, 2.0]
    assert {community.metadata["fallback_kind"] for community in clustering.communities} == {"token_packing_leaf"}
    assert {community.clustering_method for community in clustering.communities} == {"budget_fallback_token_packing_leaf"}
    assert max(community.metadata["prompt_token_cost"] for community in clustering.communities) <= 50
    tree = clustering.summary_budget["refinement_routing_tree"]
    router = next(node for node in tree["nodes"].values() if node["kind"] == "fallback_token_router")
    assert router["routing_model_kind"] == "fallback_centroid_softmax_v1"
    assert router["routing_temperature"] == pytest.approx(8.0)
    assert "singleton_budget" not in router


def test_static_tree_build_stops_before_gmm_when_effective_kmax_is_one():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "gmm_bic": {"min_split_size": 2, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0]),
        ExperienceItem(item_id="b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.0, 1.0]),
    ]

    async def summarize(*args, **kwargs):
        raise AssertionError("BIC selected one cluster; summary_fn should not run")

    async def run_build():
        return await asyncio.wait_for(
            ProjectedGmmTreeBuilder(cfg).build(items, summary_fn=summarize, max_levels=1),
            timeout=1.0,
        )

    result = asyncio.run(run_build())
    assert len(result.layers) == 1
    assert result.layers[0].clustering.stop_reason == "bic_selected_one"
    assert result.layers[0].committed is False


def test_primary_argmax_memberships_are_structural_one_hot():
    from dynamix_core.config import SoftMembershipConfig
    from dynamix_core.gmm_bic import membership_weight_dicts

    weights = membership_weight_dicts(
        ["item0"],
        ["c0", "c1"],
        np.asarray([[0.60, 0.40]], dtype=float),
        SoftMembershipConfig(recursive_assignment="primary_argmax"),
    )
    assert weights == {"item0": {"c0": 1.0}}


def test_cumulative_mass_memberships_preserve_soft_weights():
    from dynamix_core.config import SoftMembershipConfig
    from dynamix_core.gmm_bic import membership_weight_dicts

    weights = membership_weight_dicts(
        ["item0"],
        ["c0", "c1"],
        np.asarray([[0.60, 0.40]], dtype=float),
        SoftMembershipConfig(
            recursive_assignment="cumulative_mass",
            cumulative_mass_coverage=0.90,
            max_membership_gap=0.25,
            min_membership_weight=0.0,
        ),
    )
    assert weights == {"item0": {"c0": pytest.approx(0.60), "c1": pytest.approx(0.40)}}


def test_projected_kmeans_elbow_selects_hard_two_cluster_split():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "tree_policy": "projected_kmeans_elbow",
        "gmm_bic": {"min_split_size": 2, "min_effective_samples_per_component": 1, "abs_kmax": 4},
        "kmeans": {"min_k": 1, "num_restarts": 3, "max_iter": 50},
        "soft_membership": {"recursive_assignment": "primary_argmax"},
        "budget_refinement": {"enabled": False},
    })
    items = [
        ExperienceItem(item_id="a0", level=0, kind=ITEM_KIND_TRAJECTORY, text="a0", embedding=[1.0, 0.0]),
        ExperienceItem(item_id="a1", level=0, kind=ITEM_KIND_TRAJECTORY, text="a1", embedding=[0.98, 0.02]),
        ExperienceItem(item_id="b0", level=0, kind=ITEM_KIND_TRAJECTORY, text="b0", embedding=[-1.0, 0.0]),
        ExperienceItem(item_id="b1", level=0, kind=ITEM_KIND_TRAJECTORY, text="b1", embedding=[-0.98, -0.02]),
    ]
    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))
    assert not clustering.should_stop
    assert clustering.chosen_k == 2
    assert {community.clustering_method for community in clustering.communities} == {"weighted_kmeans_elbow"}
    for community in clustering.communities:
        assert set(community.member_weights) == set(community.posterior_member_weights)
        assert set(community.member_weights.values()) == {1.0}


def test_default_tree_policy_does_not_call_kmeans_selector(monkeypatch):
    import dynamix_core.tree_builder as tree_builder
    from dynamix_core.gmm_bic import GmmBicSelection, GmmCandidateFit
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("default projected_gmm_bic path must not call KMeans")

    async def fake_gmm_selector(*_args, **_kwargs):
        fit = GmmCandidateFit(
            k=1,
            valid=True,
            bic=0.0,
            log_likelihood=0.0,
            pi=np.asarray([1.0], dtype=float),
            means=np.zeros((1, 1), dtype=float),
            variances=np.ones((1, 1), dtype=float),
            responsibilities=np.ones((2, 1), dtype=float),
            primary_labels=np.zeros(2, dtype=int),
            component_masses=[2.0],
            child_sizes=[2],
        )
        return GmmBicSelection(chosen=fit, candidates=[fit], bic_margin=0.0)

    monkeypatch.setattr(tree_builder, "select_kmeans_split", fail_if_called)
    monkeypatch.setattr(tree_builder, "select_gmm_bic_async", fake_gmm_selector)
    cfg = default_hierarchy_config({})
    result = asyncio.run(ProjectedGmmTreeBuilder(cfg)._select_projected_split(
        np.zeros((2, 1), dtype=float),
        level=0,
        n_items=2,
        random_seed=42,
        sample_weights=np.ones(2, dtype=float),
        kmax_effective_n=2.0,
    ))
    assert result.chosen.k == 1


def test_budget_refinement_uses_static_prompt_token_estimator_before_analyst():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5, "prompt_overhead_reserve_tokens": 0},
        "gmm_bic": {"min_split_size": 4, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10}),
        ExperienceItem(item_id="b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.0, 1.0], metadata={"analysis_token_count": 10}),
    ]
    calls = []

    def estimator(community, members):
        calls.append((community.community_id, tuple(item.item_id for item in members)))
        return 90 if len(members) > 1 else 40

    clustering = asyncio.run(
        ProjectedGmmTreeBuilder(cfg).cluster_layer(
            items,
            level=0,
            prompt_token_estimator=estimator,
        )
    )

    assert clustering.excluded_input_item_ids == []
    assert len(clustering.communities) == 2
    assert {community.metadata["fallback_kind"] for community in clustering.communities} == {"singleton_leaf"}
    assert max(community.metadata["prompt_token_cost"] for community in clustering.communities) <= 50
    assert any(member_ids == ("a", "b") for _community_id, member_ids in calls)


def test_budget_refinement_validates_exact_final_community_prompt_tokens():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5, "prompt_overhead_reserve_tokens": 0},
        "gmm_bic": {"min_split_size": 4, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10}),
        ExperienceItem(item_id="b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.0, 1.0], metadata={"analysis_token_count": 10}),
    ]

    def estimator(community, members):
        if len(members) <= 1:
            return 30
        if community.metadata.get("budget_fallback_probe"):
            return 45
        if community.community_id.endswith("_R000"):
            return 51
        return 90

    clustering = asyncio.run(
        ProjectedGmmTreeBuilder(cfg).cluster_layer(
            items,
            level=0,
            prompt_token_estimator=estimator,
        )
    )

    assert clustering.excluded_input_item_ids == []
    assert len(clustering.communities) == 2
    assert all(len(clustering.member_item_ids_by_community[community.community_id]) == 1 for community in clustering.communities)
    assert any(event.get("split_reason") == "final_prompt_token_cost_exceeds_budget" for event in clustering.summary_budget["split_events"])


def test_budget_refinement_honors_explicit_prompt_token_budget():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5, "prompt_overhead_reserve_tokens": 0},
        "gmm_bic": {"min_split_size": 4, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10}),
        ExperienceItem(item_id="b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.0, 1.0], metadata={"analysis_token_count": 10}),
    ]

    def estimator(_community, members):
        return 45 if len(members) > 1 else 30

    clustering = asyncio.run(
        ProjectedGmmTreeBuilder(cfg).cluster_layer(
            items,
            level=0,
            prompt_token_estimator=estimator,
            prompt_token_budget=40,
        )
    )

    assert len(clustering.communities) == 2
    assert clustering.summary_budget["analyst_prompt_token_budget"] == 40
    assert clustering.summary_budget["effective_token_budget"] == 40


def test_static_pipeline_passes_explicit_analyst_prompt_budget_to_builder(tmp_path, monkeypatch):
    import dynamix_trace2skill.pipeline as pipeline
    from dynamix_core.skill_export import SkillExportResult

    captured: dict[str, Any] = {}

    class FakeState:
        async def to_dict(self, *, include_embeddings=False, validate=True):
            return {"items": {}, "communities": {}}

    class FakeBuilder:
        def __init__(self, hierarchy_config):
            captured["hierarchy_budget"] = hierarchy_config.summary_budget.analyst_prompt_token_budget

        async def build(self, items, **kwargs):
            captured["prompt_token_budget"] = kwargs.get("prompt_token_budget")
            captured["has_prompt_token_estimator"] = callable(kwargs.get("prompt_token_estimator"))
            return SimpleNamespace(state=FakeState(), layers=[])

    async def fake_embed_records_for_build(*, records, embedding_client, config, out):
        return ["embedding text"], [[1.0, 0.0]]

    async def fake_export_skill_files(state, out, *, config=None):
        return SkillExportResult(output_dir=str(tmp_path / "skills"), manifest_path=str(tmp_path / "manifest.json"), node_count=0)

    record = RawTrajectoryRecord(trajectory_id="t0", task_id="task0", trial_index=0, instruction="Do it")
    monkeypatch.setattr(pipeline, "_load_records_for_protocol", lambda config, out: [record])
    monkeypatch.setattr(pipeline, "_embed_records_for_build", fake_embed_records_for_build)
    monkeypatch.setattr(pipeline, "_records_to_items", lambda records, texts, embeddings, *, config: (
        [ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10})],
        [],
    ))
    monkeypatch.setattr(pipeline, "ProjectedGmmTreeBuilder", FakeBuilder)
    monkeypatch.setattr(pipeline, "export_skill_files", fake_export_skill_files)
    monkeypatch.setattr(pipeline, "_refresh_skillbank_index", lambda skillbank_root, config: str(tmp_path / "index.json"))

    config = DynaMixRunConfig(
        output_dir=str(tmp_path / "out"),
        records_path=str(tmp_path / "records.json"),
        generation=GenerationConfig(base_url="mock://chat"),
        embedding=EmbeddingConfig(base_url="mock://embedding", cache_path=str(tmp_path / "cache.sqlite")),
        hierarchy={"summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.8}},
        analyst=ClusterAnalystConfig(max_prompt_tokens=40),
    )

    asyncio.run(pipeline.build_tree_from_records(config))

    assert captured["hierarchy_budget"] == 80
    assert captured["prompt_token_budget"] == 40
    assert captured["has_prompt_token_estimator"] is True


def test_static_pipeline_writes_empty_nodebank_when_no_cards_are_exportable(tmp_path, monkeypatch):
    import dynamix_trace2skill.pipeline as pipeline

    class FakeState:
        async def to_dict(self, *, include_embeddings=False, validate=True):
            return {
                "items": {
                    "t0": {
                        "item_id": "t0",
                        "kind": ITEM_KIND_TRAJECTORY,
                        "level": 0,
                        "text": "trace",
                        "embedding": None,
                        "metadata": {},
                    },
                },
                "communities": {},
            }

    class FakeBuilder:
        def __init__(self, hierarchy_config):
            pass

        async def build(self, items, **kwargs):
            return SimpleNamespace(state=FakeState(), layers=[])

    async def fake_embed_records_for_build(*, records, embedding_client, config, out):
        return ["embedding text"], [[1.0, 0.0]]

    record = RawTrajectoryRecord(trajectory_id="t0", task_id="task0", trial_index=0, instruction="Do it")
    monkeypatch.setattr(pipeline, "_load_records_for_protocol", lambda config, out: [record])
    monkeypatch.setattr(pipeline, "_embed_records_for_build", fake_embed_records_for_build)
    monkeypatch.setattr(pipeline, "_records_to_items", lambda records, texts, embeddings, *, config: (
        [ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10})],
        [],
    ))
    monkeypatch.setattr(pipeline, "ProjectedGmmTreeBuilder", FakeBuilder)
    monkeypatch.setattr(pipeline, "_refresh_skillbank_index", lambda skillbank_root, config: pytest.fail("empty nodebank should not be indexed"))

    config = DynaMixRunConfig(
        output_dir=str(tmp_path / "out"),
        records_path=str(tmp_path / "records.json"),
        generation=GenerationConfig(base_url="mock://chat"),
        embedding=EmbeddingConfig(base_url="mock://embedding", cache_path=str(tmp_path / "cache.sqlite")),
        analyst=ClusterAnalystConfig(max_prompt_tokens=40),
    )

    summary = asyncio.run(pipeline.build_tree_from_records(config))

    assert summary["node_count"] == 0
    assert summary["skillbank_index"] == ""
    assert summary["empty_nodebank_reason"] == "no_exportable_experience_cards"
    manifest = json.loads((tmp_path / "out" / "skills" / "node_bank_manifest.json").read_text(encoding="utf-8"))
    assert manifest["node_count"] == 0
    assert manifest["nodes"] == []
    assert manifest["empty_reason"] == "no_exportable_experience_cards"


def test_budget_refinement_excludes_only_true_oversize_singleton_after_fallback():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.8, "prompt_overhead_reserve_tokens": 30},
        "gmm_bic": {"min_split_size": 99, "min_effective_samples_per_component": 2},
    })
    items = [
        ExperienceItem(item_id="too_big", level=0, kind=ITEM_KIND_TRAJECTORY, text="big", embedding=[1.0, 0.0], metadata={"analysis_token_count": 60}),
        ExperienceItem(item_id="keep_1", level=0, kind=ITEM_KIND_TRAJECTORY, text="k1", embedding=[0.0, 1.0], metadata={"analysis_token_count": 40}),
        ExperienceItem(item_id="keep_2", level=0, kind=ITEM_KIND_TRAJECTORY, text="k2", embedding=[0.1, 0.9], metadata={"analysis_token_count": 40}),
    ]
    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))
    assert clustering.excluded_input_item_ids == ["too_big"]
    assert sorted(item_id for community in clustering.communities for item_id in community.member_weights) == ["keep_1", "keep_2"]
    skipped = clustering.summary_budget["excluded_oversize_singletons"]
    assert len(skipped) == 1
    assert skipped[0]["item_id"] == "too_big"
    assert skipped[0]["reason"] == "oversize_singleton"
    assert skipped[0]["budget"] == 50
    assert "singleton_budget" not in skipped[0]


def test_cluster_analyst_skips_diagnostic_oversize_singleton():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True))  # type: ignore[arg-type]
    community = ExperienceCommunity(
        community_id="C_oversize",
        level=0,
        member_weights={"too_long": 1.0},
        metadata={"oversize_singleton": True, "llm_summary_skipped": True},
    )
    member = ExperienceItem(
        item_id="too_long",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[0.25, 0.75],
        metadata={"instruction": "Format the workbook.", "analysis_bundle": "very long bundle"},
    )
    cards = asyncio.run(analyst.summarize(community, [member]))
    assert cards == []
    assert analyst.config.token_report == []


def test_refinement_routing_masks_excluded_child_and_uses_next_active_leaf():
    from dynamix_core.config import SoftMembershipConfig
    from dynamix_core.update import _route_through_refinement_tree

    item = ExperienceItem(
        item_id="new_t",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="new trace",
        embedding=[0.0],
    )
    tree = {
        "coarse_roots": {"L0_C0": "root"},
        "nodes": {
            "root": {
                "node_id": "root",
                "kind": "gmm_split",
                "pca_mean": [0.0],
                "pca_components": [[1.0]],
                "pi": [0.7, 0.3],
                "means": [[0.0], [1.0]],
                "variances": [[1.0], [1.0]],
                "child_node_ids": ["excluded", "active_leaf"],
            },
            "excluded": {"node_id": "excluded", "kind": "excluded_oversize_singleton"},
            "active_leaf": {"node_id": "active_leaf", "kind": "leaf", "community_id": "L0_C0_R000"},
        },
    }
    selected = _route_through_refinement_tree(
        item=item,
        coarse_community_id="L0_C0",
        coarse_weight=1.0,
        tree=tree,
        soft_config=SoftMembershipConfig(recursive_assignment="cumulative_mass", cumulative_mass_coverage=0.9),
        selected_only=True,
    )
    posterior = _route_through_refinement_tree(
        item=item,
        coarse_community_id="L0_C0",
        coarse_weight=1.0,
        tree=tree,
        soft_config=SoftMembershipConfig(recursive_assignment="cumulative_mass", cumulative_mass_coverage=0.9),
        selected_only=False,
    )
    assert set(selected) == {"L0_C0_R000"}
    assert 0.15 < selected["L0_C0_R000"] < 0.30
    assert posterior == selected


def test_refinement_fallback_router_routes_to_token_packing_leaf():
    from dynamix_core.config import SoftMembershipConfig
    from dynamix_core.update import _route_through_refinement_tree

    item = ExperienceItem(
        item_id="new_t",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="new trace",
        embedding=[1.0, 0.0],
    )
    tree = {
        "coarse_roots": {"L0_C0": "router"},
        "nodes": {
            "router": {
                "node_id": "router",
                "kind": "fallback_token_router",
                "child_node_ids": ["leaf_a", "leaf_b"],
            },
            "leaf_a": {"node_id": "leaf_a", "kind": "token_packing_leaf", "community_id": "L0_C0_R000", "centroid_embedding": [1.0, 0.0]},
            "leaf_b": {"node_id": "leaf_b", "kind": "singleton_leaf", "community_id": "L0_C0_R001", "centroid_embedding": [0.0, 1.0]},
        },
    }
    selected = _route_through_refinement_tree(
        item=item,
        coarse_community_id="L0_C0",
        coarse_weight=1.0,
        tree=tree,
        soft_config=SoftMembershipConfig(recursive_assignment="cumulative_mass", cumulative_mass_coverage=0.9),
        selected_only=True,
    )
    posterior = _route_through_refinement_tree(
        item=item,
        coarse_community_id="L0_C0",
        coarse_weight=1.0,
        tree=tree,
        soft_config=SoftMembershipConfig(recursive_assignment="cumulative_mass", cumulative_mass_coverage=0.9),
        selected_only=False,
    )
    assert set(selected) == {"L0_C0_R000"}
    assert selected["L0_C0_R000"] > 0.99
    assert posterior["L0_C0_R000"] > posterior["L0_C0_R001"]


def test_dynamic_route_masks_removed_coarse_community_and_uses_next_active_cluster():
    from dynamix_core.data_structures import ExperienceHierarchyState
    from dynamix_core.update import ExperienceHierarchyDynamicUpdater

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="old_t", level=0, kind=ITEM_KIND_TRAJECTORY, text="old", embedding=[10.0]),
        ])
        await state.commit_layer(
            level=0,
            communities=[ExperienceCommunity(community_id="L0_C1", level=0, member_weights={"old_t": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="card",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="card",
                    embedding=[10.0],
                    generated_from_community_ids=["L0_C1"],
                    metadata={"name": "Card", "trigger": "active", "content": "active", "confidence": 1.0},
                )
            ],
            metadata={
                "routing_model": {
                    "routing_model_kind": "fixed_k_pca_gmm",
                    "level": 0,
                    "community_ids": ["L0_C0", "L0_C1"],
                    "pca_mean": [0.0],
                    "pca_components": [[1.0]],
                    "pi": [0.7, 0.3],
                    "means": [[0.0], [1.0]],
                    "variances": [[1.0], [1.0]],
                    "soft_assignment": {},
                }
            },
        )
        await state.insert_trajectory_items([
            ExperienceItem(item_id="new_t", level=0, kind=ITEM_KIND_TRAJECTORY, text="new", embedding=[0.0]),
        ])
        updater = ExperienceHierarchyDynamicUpdater(default_hierarchy_config({}))
        return await updater.route_existing_items(state, level=0, item_ids=["new_t"])

    routing = asyncio.run(run_case())
    assert set(routing.selected_assignments["new_t"]) == {"L0_C1"}
    assert 0.15 < routing.selected_assignments["new_t"]["L0_C1"] < 0.30
    assert routing.posterior_assignments == routing.selected_assignments


def test_routing_model_refresh_coarsens_refined_leaf_posterior_only_to_active_roots():
    from dynamix_core.update import _coarsen_posterior_for_routing_model

    tree = {
        "coarse_roots": {"L0_C0": "root0", "L0_C1": "root1"},
        "nodes": {
            "root0": {"node_id": "root0", "kind": "leaf", "community_id": "L0_C0_R000"},
            "root1": {"node_id": "root1", "kind": "excluded_oversize_singleton"},
        },
    }
    posterior = _coarsen_posterior_for_routing_model(
        {"L0_C0_R000": 0.7, "L0_C1_R000": 0.2, "L0_C2": 0.1},
        child_ids=["L0_C0", "L0_C1"],
        refinement_tree=tree,
    )
    assert posterior == {"L0_C0": pytest.approx(0.7)}


def test_adapted_trace2skill_templates_remove_obvious_single_trajectory_phrase():
    from dynamix_trace2skill.summary import _adapt_trace2skill_template
    text = _adapt_trace2skill_template("Analyze this single trajectory. This trajectory failed.")
    assert "single trajectory" not in text
    assert "This trajectory failed" not in text
    assert "trajectory cluster" in text


def test_analyst_budget_defaults_to_summary_budget(tmp_path):
    from dynamix_trace2skill.pipeline import DynaMixRunConfig, _prepare_analyst_tokenizer_config
    cfg = DynaMixRunConfig(
        output_dir=str(tmp_path / "out"),
        records_path=str(tmp_path / "records.json"),
        hierarchy={"summary_budget": {"max_model_tokens": 12345, "budget_ratio": 0.5}},
        analyst=ClusterAnalystConfig(max_prompt_tokens=None, tokenizer_required=False),
    )
    _prepare_analyst_tokenizer_config(cfg, tmp_path / "out")
    assert cfg.analyst.max_prompt_tokens == int(12345 * 0.5)
    payload = json.loads((tmp_path / "out" / "analysis" / "analyst_budget_config.json").read_text())
    assert payload["source"] == "hierarchy.summary_budget"


def test_nodebank_export_uses_only_name_trigger_content_for_embedding(tmp_path):
    from dynamix_core.skill_export import export_skill_files_from_payload
    payload = {
        "items": {
            "root": {
                "item_id": "root", "level": 1, "kind": "experience_card", "text": "Root guidance", "support_mass": 10.0,
                "generated_from_community_ids": ["c0"],
                "metadata": {
                    "name": "Cross Sheet Lookup",
                    "trigger": "When matching values across sheets.",
                    "content": "Use lookup keys and verify target ranges.",
                    "confidence": 0.9,
                    "placement": {"target": "script"},
                    "source_community_id": "c0",
                    "source_member_count": 7,
                    "analyst_mode": "raw_extractor",
                },
            },
            "raw": {"item_id": "raw", "level": 0, "kind": "trajectory", "text": "raw trace", "support_mass": 1.0, "metadata": {}},
            "skip": {"item_id": "skip", "level": 1, "kind": "experience_card", "text": "skip", "support_mass": 1.0, "metadata": {"name": "Skip", "trigger": "skip", "content": "skip", "confidence": 0.5, "oversize_singleton": True}},
        },
        "communities": {},
    }
    result = export_skill_files_from_payload(payload, tmp_path)
    assert result.node_count == 1
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["format"] == "dynamix_node_skill_bank_v1"
    node = manifest["nodes"][0]
    assert node["node_id"] == "root"
    assert node["embedding_text"] == "name: Cross Sheet Lookup\ntrigger: When matching values across sheets.\ncontent: Use lookup keys and verify target ranges."
    assert "placement" not in node
    assert "level:" not in node["embedding_text"]
    assert "support_mass" not in node["embedding_text"]
    assert not (tmp_path / "skills" / "SKILL.md").exists()


def test_nodebank_export_level_filter_selects_retrieval_layers(tmp_path):
    from dynamix_core.skill_export import SkillExportConfig, export_skill_files_from_payload

    def card(item_id: str, level: int) -> dict[str, Any]:
        return {
            "item_id": item_id,
            "level": level,
            "kind": "experience_card",
            "text": item_id,
            "support_mass": float(level),
            "generated_from_community_ids": [f"c{level}"],
            "metadata": {
                "name": f"Card {item_id}",
                "trigger": "When relevant.",
                "content": "Use the reusable experience.",
                "confidence": 0.9,
            },
        }

    payload = {"items": {"l1": card("l1", 1), "l2": card("l2", 2), "l3": card("l3", 3)}, "communities": {}}
    all_nodes = export_skill_files_from_payload(payload, tmp_path / "all")
    l1 = export_skill_files_from_payload(payload, tmp_path / "l1", config=SkillExportConfig(min_level=1, max_level=1))
    l2plus = export_skill_files_from_payload(payload, tmp_path / "l2plus", config=SkillExportConfig(min_level=2))
    assert {node.item_id for node in all_nodes.nodes} == {"l1", "l2", "l3"}
    assert [node.item_id for node in l1.nodes] == ["l1"]
    assert {node.item_id for node in l2plus.nodes} == {"l2", "l3"}
    manifest = json.loads(Path(l2plus.manifest_path).read_text())
    assert manifest["export_policy"]["level_filter"] == {"min_level": 2, "max_level": None}


def test_reuse_tree_contract_rejects_non_baseline_source(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_dynamix_nodebank.py"
    spec = importlib.util.spec_from_file_location("export_dynamix_nodebank", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    source = tmp_path / "source_tree"
    (source / "analysis").mkdir(parents=True)
    target = DynaMixRunConfig(
        output_dir=str(tmp_path / "target"),
        records_path=str(tmp_path / "records.json"),
        scenario="static_build",
        dataset_path="/data/spreadsheetbench",
        train_start=0,
        train_end=200,
        chunked_embedding={"enabled": True, "chunk_tokens": 28000, "overlap_tokens": 1000, "pooling": "mean"},
        max_levels=8,
    )

    def full_config_payload(config: DynaMixRunConfig) -> dict[str, Any]:
        payload = asdict(config)
        payload["hierarchy"] = asdict(default_hierarchy_config(config.hierarchy))
        return json.loads(json.dumps(payload))

    def write_source_config(payload: dict[str, Any]) -> None:
        (source / "summary.json").write_text(json.dumps({"scenario": "static_build"}), encoding="utf-8")
        (source / "analysis" / "runtime_config.json").write_text(json.dumps(payload), encoding="utf-8")

    baseline_config = full_config_payload(target)
    write_source_config(baseline_config)
    accepted = module._validate_source_tree_contract(source, target_config=target)
    assert accepted["observed"]["tree_policy"] == "projected_gmm_bic"

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["hierarchy"]["tree_policy"] = "projected_kmeans_elbow"
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target)

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["hierarchy"]["soft_membership"]["recursive_assignment"] = "primary_argmax"
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target)

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["analyst"]["max_cards_l0"] = 1
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target)

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["max_levels"] = 1
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target)

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["dataset_path"] = "/data/different_spreadsheetbench"
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="source protocol does not match"):
        module._validate_source_tree_contract(source, target_config=target)

    bad_config = json.loads(json.dumps(baseline_config))
    bad_config["embedding"]["base_url"] = "http://different-embedding/v1"
    write_source_config(bad_config)
    with pytest.raises(ValueError, match="source protocol does not match"):
        module._validate_source_tree_contract(source, target_config=target)

    write_source_config(baseline_config)
    target_kmeans = DynaMixRunConfig(
        output_dir=str(tmp_path / "target_kmeans"),
        records_path=str(tmp_path / "records.json"),
        scenario="static_build",
        dataset_path="/data/spreadsheetbench",
        train_start=0,
        train_end=200,
        chunked_embedding={"enabled": True, "chunk_tokens": 28000, "overlap_tokens": 1000, "pooling": "mean"},
        hierarchy={"tree_policy": "projected_kmeans_elbow"},
        max_levels=8,
    )
    with pytest.raises(ValueError, match="reuse-tree target is not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target_kmeans)

    target_l1_only = DynaMixRunConfig(
        output_dir=str(tmp_path / "target_l1_only"),
        records_path=str(tmp_path / "records.json"),
        scenario="static_build",
        dataset_path="/data/spreadsheetbench",
        train_start=0,
        train_end=200,
        chunked_embedding={"enabled": True, "chunk_tokens": 28000, "overlap_tokens": 1000, "pooling": "mean"},
        max_levels=1,
    )
    with pytest.raises(ValueError, match="reuse-tree target is not the expected full static baseline"):
        module._validate_source_tree_contract(source, target_config=target_l1_only)


def test_retrieval_only_variant_requires_reused_train_artifacts(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "experiments" / "ablations" / "static" / "common" / "run_variant.py"
    spec = importlib.util.spec_from_file_location("run_static_ablation_variant", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    variant = {
        "variant_name": "retrieve_l1_only",
        "reuse_full_tree": True,
        "tree": {"tree_policy": "projected_gmm_bic"},
        "skill_export": {"min_level": 1, "max_level": 1},
    }
    variant_path = tmp_path / "variant.json"
    variant_path.write_text(json.dumps(variant), encoding="utf-8")
    monkeypatch.setenv("REPO_ROOT", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BASELINE_TREE_DIR", str(tmp_path / "baseline_tree"))
    monkeypatch.delenv("RECORDS_PATH", raising=False)
    monkeypatch.delenv("REUSE_TRAIN_RUN_DIR", raising=False)
    monkeypatch.setattr("sys.argv", ["run_variant.py", "--variant-json", str(variant_path)])
    with pytest.raises(SystemExit, match="retrieval-only variants require RECORDS_PATH or REUSE_TRAIN_RUN_DIR"):
        module.main()


def test_cluster_prompt_uses_minimal_experience_schema():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_schema", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "bundle"})
    payload = json.loads(analyst._build_prompt(community, [member], "raw_extractor"))
    schema = payload["output_schema"]
    assert "task_profile" not in payload
    assert "officeqa_experience_policy" not in payload
    assert "spreadsheet_experience_policy" in payload
    assert "template_user_prompt_adaptation" in payload
    assert set(schema) == {"cards"}
    card_schema = schema["cards"][0]
    assert set(card_schema) == {"name", "trigger", "content", "placement", "confidence"}
    forbidden = {"shared_patterns", "success_motifs", "anti_patterns", "shared_patch_hints", "reference_materials", "script_files", "skill_placement"}
    assert not (forbidden & set(card_schema))
    constraints = " ".join(payload["hard_constraints"])
    assert "Do not output fields except cards and each card's name, trigger, content, placement, confidence" in constraints
    spreadsheet_policy = json.dumps(payload["spreadsheet_experience_policy"], ensure_ascii=False)
    assert "formula or API invariants" in spreadsheet_policy
    assert "recalculation-based verification" in spreadsheet_policy


def test_higher_level_prompt_requires_distinct_children_and_allows_no_parent():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    members = [
        ExperienceItem(item_id="e1", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="card one", embedding=[1.0]),
        ExperienceItem(item_id="e2", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="card two", embedding=[0.5]),
    ]
    payload = json.loads(analyst._build_prompt(community, members, "experience_abstractor"))
    combined = "\n".join([
        analyst._system_prompt("experience_abstractor"),
        payload["instruction"],
        " ".join(payload["hard_constraints"]),
    ])
    assert "at least two" in combined.lower()
    assert "shared invariant" in combined.lower()
    assert "empty cards list" in combined.lower() or "cards=[]" in combined


def test_higher_level_analyst_accepts_empty_cards_without_embedding():
    class DummyGeneration:
        async def chat_json(self, messages, *, schema_name, guided_json, **kwargs):
            assert schema_name == "HigherLevelExperienceCardOrEmpty"
            assert guided_json["properties"]["cards"]["maxItems"] == 1
            assert "minItems" not in guided_json["properties"]["cards"]
            return {"cards": []}

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            raise AssertionError("empty higher-level output must not be embedded")

    analyst = ClusterAnalyst(
        DummyGeneration(),
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    members = [
        ExperienceItem(item_id="e1", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="card one", embedding=[1.0]),
        ExperienceItem(item_id="e2", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="card two", embedding=[0.5]),
    ]
    assert asyncio.run(analyst.summarize(community, members)) == []
    assert community.metadata["higher_level_abstraction_skipped"] == "no_shared_invariant"


def test_static_cluster_analyst_uses_guided_json_without_forcing_thinking():
    class DummyGeneration:
        def __init__(self):
            self.kwargs = None

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.kwargs = kwargs
            return {
                "cards": [{
                    "name": "Static lesson",
                    "trigger": "When static clusters need one reusable card.",
                    "content": "Use the guided JSON card schema.",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.8,
                }]
            }

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[1.0] for _ in texts]

    generation = DummyGeneration()
    analyst = ClusterAnalyst(
        generation,
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, max_output_tokens=3333),
    )
    community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(
        item_id="t0",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_bundle": "trajectory text"},
    )

    items = asyncio.run(analyst.summarize(community, [member]))

    assert len(items) == 1
    assert generation.kwargs["guided_json"]["required"] == ["cards"]
    assert generation.kwargs["guided_json"]["properties"]["cards"]["minItems"] == 1
    placement_schema = generation.kwargs["guided_json"]["properties"]["cards"]["items"]["properties"]["placement"]
    assert placement_schema["required"] == ["target", "reference_kind"]
    assert generation.kwargs["max_tokens"] == 3333
    assert "extra_body" not in generation.kwargs


def test_render_card_text_minimal_schema_only():
    from dynamix_trace2skill.summary import _render_card_text
    text = _render_card_text({"name": "N", "trigger": "T", "content": "C"})
    assert "# N" in text
    assert "## Trigger" in text
    assert "## Content" in text
    assert "Shared patterns" not in text
    assert "Success motifs" not in text


def test_dynamic_analyst_adds_new_cards_without_position_matching_old_cards():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyGeneration:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return {
                "new_cards": [{
                    "name": "New lesson",
                    "trigger": "When a newly inserted trace shows a distinct procedure.",
                    "content": "Treat this as an independent reusable experience.",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.7,
                }],
            }

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[float(index + 1)] for index, _ in enumerate(texts)]

    generation = DummyGeneration()
    analyst = ClusterAnalyst(
        generation,
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, dynamic_max_output_tokens=7777),
    )
    community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(
        item_id="t0",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_bundle": "new trajectory evidence"},
    )
    previous = [{
        "item_id": "old_high_conf",
        "level": 1,
        "kind": ITEM_KIND_EXPERIENCE_CARD,
        "text": "old card",
        "support_mass": 1.0,
        "metadata": {"name": "Old", "trigger": "old", "content": "old", "confidence": 0.99},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

    assert len(patches) == 1
    assert patches[0].operation == "add"
    assert patches[0].item_id != "old_high_conf"
    assert patches[0].metadata["dynamic_patch_operation"] == "add"
    prompt = generation.messages[1]["content"]
    assert "old_high_conf" in prompt
    assert "new_cards" in prompt
    assert "Never infer update targets from output order or confidence rank" in prompt
    payload = json.loads(prompt)
    assert set(payload) == {
        "instruction",
        "analyst_mode",
        "dynamic_patch_policy",
        "hard_constraints",
        "members",
        "previous_generated_experiences",
        "spreadsheet_experience_policy",
    }
    assert "community" not in payload
    assert "output_schema" not in payload
    assert all("support_mass" not in member for member in payload["members"])
    assert all("support_mass" not in card for card in payload["previous_generated_experiences"])
    assert generation.kwargs["guided_json"]["required"] == ["updates", "new_cards"]
    assert generation.kwargs["max_tokens"] == 7777
    assert generation.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_dynamic_analyst_updates_only_explicit_previous_card_ids():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyGeneration:
        async def chat_json(self, messages, *, schema_name, **kwargs):
            return {
                "updates": [{
                    "item_id": "old_low_conf",
                    "name": "Revised old lesson",
                    "trigger": "When the same old lesson needs a confidence/content revision.",
                    "content": "Revise the old reusable experience by explicit id only.",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.8,
                }],
                "new_cards": [],
            }

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[2.0] for _ in texts]

    analyst = ClusterAnalyst(
        DummyGeneration(),
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0])
    previous = [
        {"item_id": "old_high_conf", "metadata": {"name": "High", "trigger": "h", "content": "h", "confidence": 0.99}},
        {"item_id": "old_low_conf", "metadata": {"name": "Low", "trigger": "l", "content": "l", "confidence": 0.1}},
    ]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

    assert len(patches) == 1
    assert patches[0].operation == "update"
    assert patches[0].item_id == "old_low_conf"
    assert patches[0].metadata["confidence"] == 0.8
    assert patches[0].metadata["dynamic_patch_operation"] == "update"


def test_dynamic_analyst_higher_level_prompt_is_updates_only_and_skips_unrepairable_legacy_card_output():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyGeneration:
        def __init__(self, payload):
            self.messages = []
            self.kwargs = []
            self.payload = payload

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.messages.append(messages)
            self.kwargs.append(kwargs)
            return self.payload

    class DummyEmbedding:
        def __init__(self):
            self.calls = 0

        async def embed_texts(self, texts, *, cache_namespace=None):
            self.calls += 1
            return [[1.0] for _ in texts]

    legacy_card = {
        "name": "New high-level abstraction",
        "trigger": "When lower-level cards suggest another abstraction.",
        "content": "This legacy-shaped output should not be accepted for L1+ dynamic updates.",
        "placement": {"target": "skill_md", "reference_kind": "procedure"},
        "confidence": 0.7,
    }
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    member2 = ExperienceItem(
        item_id="e2",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="another lower card",
        embedding=[0.5],
        metadata={"name": "Another", "trigger": "another", "content": "another", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    for payload in ({"new_cards": [legacy_card]}, {"cards": [legacy_card]}, legacy_card):
        generation = DummyGeneration(payload)
        embedding = DummyEmbedding()
        analyst = ClusterAnalyst(
            generation,
            embedding,
            ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
        )

        patches = asyncio.run(analyst.summarize_dynamic_update(community, [member, member2], previous))

        assert patches == []
        assert embedding.calls == 0
        assert len(generation.messages) == 3
        assert all(call["retries"] == 0 for call in generation.kwargs)
        assert analyst.config.token_report[-1]["event"] == "dynamic_schema_repair"
        assert analyst.config.token_report[-1]["status"] == "ignored_invalid_llm_output"
        assert analyst.config.token_report[-1]["action"] == "skip_invalid_dynamic_update"
        prompt = generation.messages[0][1]["content"]
        assert "new_cards" not in prompt
        assert "Return a top-level JSON object with only updates." in prompt
        system_prompt = generation.messages[0][0]["content"]
        assert "new_cards" not in system_prompt
        assert "top-level cards list" not in system_prompt
        assert all("new_cards" not in message["content"] for messages in generation.messages for message in messages)


def test_dynamic_analyst_higher_level_repairs_legacy_schema_and_accepts_update():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    valid_update = {
        "item_id": "old_l2",
        "name": "Repaired high-level abstraction",
        "trigger": "When lower-level cards share the repaired pattern.",
        "content": "Use the explicit old item_id and update only the existing abstraction.",
        "placement": {"target": "skill_md", "reference_kind": "procedure"},
        "confidence": 0.8,
    }

    class DummyGeneration:
        def __init__(self):
            self.messages = []
            self.kwargs = []
            self.payloads = [
                {
                    "cards": [{
                        "name": "Legacy shape",
                        "trigger": "legacy",
                        "content": "legacy",
                        "placement": {"target": "skill_md", "reference_kind": "procedure"},
                        "confidence": 0.5,
                    }]
                },
                {"updates": [valid_update]},
            ]

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.messages.append(messages)
            self.kwargs.append(kwargs)
            return self.payloads.pop(0)

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[4.0] for _ in texts]

    generation = DummyGeneration()
    analyst = ClusterAnalyst(
        generation,
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    member2 = ExperienceItem(
        item_id="e2",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="another lower card",
        embedding=[0.5],
        metadata={"name": "Another", "trigger": "another", "content": "another", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member, member2], previous))

    assert len(patches) == 1
    assert patches[0].operation == "update"
    assert patches[0].item_id == "old_l2"
    assert patches[0].metadata["dynamic_patch_operation"] == "update"
    assert len(generation.messages) == 2
    assert all(call["retries"] == 0 for call in generation.kwargs)
    assert generation.messages[1][-1]["role"] == "user"
    assert "required_top_level_schema" in generation.messages[1][-1]["content"]
    assert "new_cards" not in generation.messages[1][-1]["content"]
    assert analyst.config.token_report[-1]["event"] == "dynamic_schema_repair"
    assert analyst.config.token_report[-1]["status"] == "retry"


def test_dynamic_analyst_higher_level_skips_unrepairable_json_parse_failure():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyGeneration:
        def __init__(self):
            self.calls = 0
            self.messages = []
            self.kwargs = []

        async def chat_json(self, messages, *, schema_name, **kwargs):
            self.calls += 1
            self.messages.append(messages)
            self.kwargs.append(kwargs)
            raise ValueError("failed to parse JSON for DynamicExperienceCardPatchSet")

    class DummyEmbedding:
        def __init__(self):
            self.calls = 0

        async def embed_texts(self, texts, *, cache_namespace=None):
            self.calls += 1
            return [[1.0] for _ in texts]

    generation = DummyGeneration()
    embedding = DummyEmbedding()
    analyst = ClusterAnalyst(
        generation,
        embedding,
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, max_prompt_tokens=100000),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    member2 = ExperienceItem(
        item_id="e2",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="another lower card",
        embedding=[0.5],
        metadata={"name": "Another", "trigger": "another", "content": "another", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member, member2], previous))

    assert patches == []
    assert generation.calls == 3
    assert all(call["retries"] == 0 for call in generation.kwargs)
    assert embedding.calls == 0
    assert any(event.get("event") == "dynamic_schema_repair_prompt" for event in analyst.config.token_report)
    assert analyst.config.token_report[-1]["event"] == "dynamic_schema_repair"
    assert analyst.config.token_report[-1]["status"] == "ignored_invalid_llm_output"
    assert analyst.config.token_report[-1]["action"] == "skip_invalid_dynamic_update"


def test_dynamic_analyst_higher_level_accepts_explicit_updates():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyGeneration:
        async def chat_json(self, messages, *, schema_name, **kwargs):
            return {
                "updates": [{
                    "item_id": "old_l2",
                    "name": "Updated high-level abstraction",
                    "trigger": "When lower-level cards share a clearer high-level pattern.",
                    "content": "Revise the existing higher-level abstraction by explicit item_id.",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.75,
                }],
            }

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[3.0] for _ in texts]

    analyst = ClusterAnalyst(
        DummyGeneration(),
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    member2 = ExperienceItem(
        item_id="e2",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="another lower card",
        embedding=[0.5],
        metadata={"name": "Another", "trigger": "another", "content": "another", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member, member2], previous))

    assert len(patches) == 1
    assert patches[0].operation == "update"
    assert patches[0].item_id == "old_l2"
    assert patches[0].metadata["dynamic_patch_operation"] == "update"
    assert patches[0].metadata["higher_level_single_card_enforced"] is True


def test_dynamic_analyst_rejects_invalid_patch_ids():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[1.0] for _ in texts]

    async def run_payload(payload):
        class DummyGeneration:
            async def chat_json(self, messages, *, schema_name, **kwargs):
                return payload

        analyst = ClusterAnalyst(
            DummyGeneration(),
            DummyEmbedding(),
            ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
        )
        community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})
        member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0])
        previous = [{"item_id": "old_a", "metadata": {"name": "Old", "trigger": "old", "content": "old", "confidence": 0.9}}]
        return await analyst.summarize_dynamic_update(community, [member], previous)

    valid_card = {
        "name": "Card",
        "trigger": "trigger",
        "content": "content",
        "placement": {"target": "skill_md", "reference_kind": "procedure"},
        "confidence": 0.8,
    }
    with pytest.raises(ValueError, match="unknown previous ExperienceCard"):
        asyncio.run(run_payload({"updates": [{"item_id": "missing", **valid_card}], "new_cards": []}))
    with pytest.raises(ValueError, match="duplicate ExperienceCard"):
        asyncio.run(run_payload({"updates": [{"item_id": "old_a", **valid_card}, {"item_id": "old_a", **valid_card}], "new_cards": []}))
    with pytest.raises(ValueError, match="new_cards must not include item_id"):
        asyncio.run(run_payload({"updates": [], "new_cards": [{"item_id": "illegal", **valid_card}]}))


def test_dynamic_state_reallocates_support_mass_after_update_and_add():
    from dynamix_core.data_structures import ExperienceCardPatch, ExperienceHierarchyState

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 0", embedding=[1.0]),
            ExperienceItem(item_id="t1", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 1", embedding=[2.0]),
        ])
        community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0, "t1": 1.0})
        await state.commit_layer(
            level=0,
            communities=[community],
            generated_items=[
                ExperienceItem(
                    item_id="old_a",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old a",
                    embedding=[1.0],
                    generated_from_community_ids=["C0"],
                    metadata={"name": "Old A", "trigger": "a", "content": "a", "confidence": 0.8},
                ),
                ExperienceItem(
                    item_id="old_b",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old b",
                    embedding=[2.0],
                    generated_from_community_ids=["C0"],
                    metadata={"name": "Old B", "trigger": "b", "content": "b", "confidence": 0.2},
                ),
            ],
            stop_reason="split",
        )
        result = await state.apply_experience_card_patches(
            source_community_id="C0",
            patches=[
                ExperienceCardPatch(
                    operation="update",
                    item_id="old_a",
                    text="updated old a",
                    embedding=[3.0],
                    metadata={"name": "Old A revised", "trigger": "a", "content": "a2", "confidence": 0.5},
                ),
                ExperienceCardPatch(
                    operation="add",
                    item_id="new_c",
                    text="new c",
                    embedding=[4.0],
                    metadata={"name": "New C", "trigger": "c", "content": "c", "confidence": 0.5},
                ),
            ],
        )
        items = {item.item_id: item for item in await state.item_objects(["old_a", "old_b", "new_c"])}
        return result, items

    result, items = asyncio.run(run_case())

    assert result.updated_item_ids == ["old_a"]
    assert result.added_item_ids == ["new_c"]
    assert result.requires_reroute_item_ids == ["new_c", "old_a"]
    assert items["old_a"].support_mass == pytest.approx(2.0 * 0.5 / 1.2)
    assert items["old_b"].support_mass == pytest.approx(2.0 * 0.2 / 1.2)
    assert items["new_c"].support_mass == pytest.approx(2.0 * 0.5 / 1.2)


def test_dynamic_l0_add_updates_existing_next_layer_inputs():
    from dynamix_core.data_structures import ExperienceCardPatch, ExperienceHierarchyState

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0]),
        ])
        await state.commit_layer(
            level=0,
            communities=[ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="old_l1",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old l1",
                    embedding=[1.0],
                    generated_from_community_ids=["C0"],
                    metadata={"name": "Old L1", "trigger": "old", "content": "old", "confidence": 1.0},
                )
            ],
            stop_reason="split",
        )
        await state.commit_layer(
            level=1,
            communities=[ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"old_l1": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="old_l2",
                    level=2,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old l2",
                    embedding=[2.0],
                    generated_from_community_ids=["L1_C0"],
                    metadata={"name": "Old L2", "trigger": "old", "content": "old", "confidence": 1.0},
                )
            ],
            stop_reason="split",
        )
        before = await state.layer_input_item_ids(1)
        await state.apply_experience_card_patches(
            source_community_id="C0",
            patches=[
                ExperienceCardPatch(
                    operation="add",
                    item_id="new_l1",
                    text="new l1",
                    embedding=[3.0],
                    metadata={"name": "New L1", "trigger": "new", "content": "new", "confidence": 1.0},
                )
            ],
        )
        after = await state.layer_input_item_ids(1)
        return before, after

    before, after = asyncio.run(run_case())
    assert before == ["old_l1"]
    assert after == ["old_l1", "new_l1"]


def test_dynamic_state_rejects_l1_plus_add_patch():
    from dynamix_core.data_structures import ExperienceCardPatch, ExperienceHierarchyState

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0]),
        ])
        await state.commit_layer(
            level=0,
            communities=[ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="old_l1",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old l1",
                    embedding=[1.0],
                    generated_from_community_ids=["C0"],
                    metadata={"name": "Old L1", "trigger": "old", "content": "old", "confidence": 1.0},
                )
            ],
            stop_reason="split",
        )
        await state.commit_layer(
            level=1,
            communities=[ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"old_l1": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="old_l2",
                    level=2,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old l2",
                    embedding=[2.0],
                    generated_from_community_ids=["L1_C0"],
                    metadata={"name": "Old L2", "trigger": "old", "content": "old", "confidence": 1.0},
                )
            ],
            stop_reason="split",
        )
        return await state.apply_experience_card_patches(
            source_community_id="L1_C0",
            patches=[
                ExperienceCardPatch(
                    operation="add",
                    item_id="illegal_l2",
                    text="illegal",
                    embedding=[3.0],
                    metadata={"name": "Illegal", "trigger": "illegal", "content": "illegal", "confidence": 1.0},
                )
            ],
        )

    with pytest.raises(ValueError, match="allowed only for L0"):
        asyncio.run(run_case())


def test_dynamic_state_rejects_duplicate_patch_item_ids():
    from dynamix_core.data_structures import ExperienceCardPatch, ExperienceHierarchyState

    async def run_case(patches):
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0]),
        ])
        await state.commit_layer(
            level=0,
            communities=[ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="old_a",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="old",
                    embedding=[1.0],
                    generated_from_community_ids=["C0"],
                    metadata={"name": "Old", "trigger": "old", "content": "old", "confidence": 0.9},
                )
            ],
            stop_reason="split",
        )
        return await state.apply_experience_card_patches(source_community_id="C0", patches=patches)

    with pytest.raises(ValueError, match="duplicate update patch"):
        asyncio.run(run_case([
            ExperienceCardPatch(operation="update", item_id="old_a", text="u1", embedding=[2.0], metadata={"name": "U1", "trigger": "u", "content": "u", "confidence": 0.8}),
            ExperienceCardPatch(operation="update", item_id="old_a", text="u2", embedding=[3.0], metadata={"name": "U2", "trigger": "u", "content": "u", "confidence": 0.7}),
        ]))
    with pytest.raises(ValueError, match="add patch duplicates"):
        asyncio.run(run_case([
            ExperienceCardPatch(operation="add", item_id="new_a", text="a1", embedding=[2.0], metadata={"name": "A1", "trigger": "a", "content": "a", "confidence": 0.8}),
            ExperienceCardPatch(operation="add", item_id="new_a", text="a2", embedding=[3.0], metadata={"name": "A2", "trigger": "a", "content": "a", "confidence": 0.7}),
        ]))


def test_dynamic_prompt_payload_contract_distinguishes_l0_and_l1_plus():
    from dynamix_core.data_structures import ExperienceHierarchyState

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0]),
        ])
        await state.commit_layer(
            level=0,
            communities=[ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t0": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="e1",
                    level=1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="lower card",
                    embedding=[1.0],
                    generated_from_community_ids=["L0_C0"],
                    metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
                )
            ],
            stop_reason="split",
        )
        await state.commit_layer(
            level=1,
            communities=[ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})],
            generated_items=[
                ExperienceItem(
                    item_id="e2",
                    level=2,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text="higher card",
                    embedding=[2.0],
                    generated_from_community_ids=["L1_C0"],
                    metadata={"name": "Higher", "trigger": "higher", "content": "higher", "confidence": 0.9},
                )
            ],
            stop_reason="split",
        )
        l0_payload = await state.build_dynamic_prompt_payload((await state.community_objects(["L0_C0"]))[0])
        l1_payload = await state.build_dynamic_prompt_payload((await state.community_objects(["L1_C0"]))[0])
        return l0_payload, l1_payload

    l0_payload, l1_payload = asyncio.run(run_case())

    assert l0_payload["contract"]["analyst_mode"] == "raw_extractor"
    assert l0_payload["contract"]["allowed_patch_operations"] == ["update", "add"]
    assert "add patches" in l0_payload["contract"]["dynamic_summary_fn"]

    assert l1_payload["contract"]["analyst_mode"] == "experience_abstractor"
    assert l1_payload["contract"]["allowed_patch_operations"] == ["update"]
    assert "update patches only" in l1_payload["contract"]["dynamic_summary_fn"]
    assert "update/add" not in l1_payload["contract"]["dynamic_summary_fn"]
    assert "update/add" not in l1_payload["contract"]["confidence"]



def test_skill_export_rejects_old_alias_schema(tmp_path):
    from dynamix_core.skill_export import export_skill_files_from_payload
    payload = {
        "items": {
            "root": {
                "item_id": "root", "level": 1, "kind": "experience_card", "text": "old", "support_mass": 1.0,
                "generated_from_community_ids": ["c0"],
                "metadata": {"confidence": 0.9, "title": "Old Root", "skill_placement": {"target": "skill_md"}},
            },
        },
        "communities": {"c0": {"community_id": "c0", "level": 0, "member_weights": {}, "posterior_member_weights": {}, "generated_item_ids": ["root"], "support_mass": 1.0}},
    }
    with pytest.raises(ValueError):
        export_skill_files_from_payload(payload, tmp_path)


def test_nodebank_selector_selects_topk_nodes_without_skill_files(tmp_path):
    from dynamix_trace2skill.skillbank import SkillBankSelector
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "node_bank_manifest.json").write_text(json.dumps({
        "format": "dynamix_node_skill_bank_v1",
        "nodes": [
            {"node_id": "lookup", "item_id": "lookup", "name": "Lookup Keys", "trigger": "matching by key", "content": "Use lookup formulas and match keys.", "embedding_text": "name: Lookup Keys\ntrigger: matching by key\ncontent: Use lookup formulas and match keys.", "sha256": "a"},
            {"node_id": "format", "item_id": "format", "name": "Format Cells", "trigger": "styling", "content": "Use fonts and fills.", "embedding_text": "name: Format Cells\ntrigger: styling\ncontent: Use fonts and fills.", "sha256": "b"},
        ],
    }), encoding="utf-8")
    selector = SkillBankSelector(skillbank_root=bank, base_url="mock://deterministic", model="mock-embed", cache_path=tmp_path / "index.json")
    selected = selector.select("need vlookup matching by key", top_k=1)
    assert len(selected) == 1
    assert "lookup" in selected[0].skill.name.lower()
    assert not list(bank.rglob("SKILL.md"))


def test_spreadsheet_runner_validates_nodebank_not_skill_folders(tmp_path, monkeypatch):
    from run_spreadsheetbench import build_arg_parser, validate_args

    parser = build_arg_parser()
    args = parser.parse_args([
        "--data_path", str(tmp_path),
        "--agent", "cli_skill_preloaded",
        "--skills_dir", str(tmp_path),
    ])

    with pytest.raises(ValueError, match="DYNAMIX_SKILLBANK_TOP_K"):
        validate_args(parser, args)

    monkeypatch.setenv("DYNAMIX_SKILLBANK_TOP_K", "10")
    with pytest.raises(ValueError, match="Node bank manifest not found"):
        validate_args(parser, args)

    (tmp_path / "node_bank_manifest.json").write_text(json.dumps({"format": "dynamix_node_skill_bank_v1", "nodes": []}), encoding="utf-8")
    validate_args(parser, args)
    assert not list(tmp_path.rglob("SKILL.md"))


def test_nodebank_selection_injects_retrieved_experience_and_logs_nodes(tmp_path, monkeypatch):
    from spreadsheet_agent.agents.cli_skill_preloaded_agent import CLISkillPreloadedAgent

    class DummyClient:
        pass

    class DummySelector:
        def __init__(self):
            self.last_query = None

        def select(self, query, top_k=3):
            self.last_query = query
            from dynamix_trace2skill.skillbank import SkillNodeDocument, SkillSelection
            doc = SkillNodeDocument(
                node_id="node-1",
                item_id="node-1",
                name="Lookup Keys",
                trigger="When matching values by key.",
                content="Use lookup formulas and verify the key range.",
                embedding_text="name: Lookup Keys\ntrigger: When matching values by key.\ncontent: Use lookup formulas and verify the key range.",
                prompt_text="",
                sha256="abc",
            )
            return [SkillSelection(skill=doc, score=1.0)]

    skills_dir = tmp_path / "skills_root"
    skills_dir.mkdir()
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "node_bank_manifest.json").write_text(json.dumps({"format": "dynamix_node_skill_bank_v1", "nodes": []}), encoding="utf-8")

    selection_log = tmp_path / "raw" / "skill_selection_records.jsonl"
    monkeypatch.setenv("DYNAMIX_SKILLBANK_TOP_K", "1")
    monkeypatch.setenv("DYNAMIX_SKILLBANK_ROOT", str(bank))
    monkeypatch.setenv("DYNAMIX_SKILL_SELECTION_LOG", str(selection_log))
    agent = CLISkillPreloadedAgent(DummyClient(), skills_dir=str(skills_dir), verbose=False)
    selector = DummySelector()
    agent._skillbank_selector = selector
    class Context:
        instance_id = "task-1"
        instruction = "lookup values"
        instruction_type = "Cell-Level Manipulation"
        answer_position = "A1"
    agent._select_skills_for_context(Context())
    expected_query = "lookup values\n\nTask type: Cell-Level Manipulation"
    assert selector.last_query == expected_query
    assert agent._active_skill_selection
    selected = agent._active_skill_selection[0]
    assert selected["node_id"] == "node-1"
    prompt = agent.get_system_template()
    assert "Retrieved Experience" in prompt
    assert "Use lookup formulas" in prompt
    assert "SKILL.md" not in prompt
    record = json.loads(selection_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["instance_id"] == "task-1"
    assert record["query"] == expected_query
    assert record["selected_node_ids"] == ["node-1"]


def test_cli_agents_expose_only_task_relative_io_paths(tmp_path, monkeypatch):
    from spreadsheet_agent.agents.base import AgentContext
    from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
    from spreadsheet_agent.agents.cli_skill_preloaded_agent import CLISkillPreloadedAgent

    class DummyClient:
        pass

    work_dir = tmp_path / "work" / "2768_1_2768_init"
    work_dir.mkdir(parents=True)
    context = AgentContext(
        working_dir=str(work_dir),
        input_file=str(work_dir / "input.xlsx"),
        output_file=str(work_dir / "output.xlsx"),
        instruction="fill formulas",
        spreadsheet_content="('a', 'b')",
        instruction_type="Cell-Level Manipulation",
        answer_position="A1",
        instance_id="2768",
    )

    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "node_bank_manifest.json").write_text(
        json.dumps({"format": "dynamix_node_skill_bank_v1", "nodes": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DYNAMIX_SKILLBANK_TOP_K", "1")
    monkeypatch.setenv("DYNAMIX_SKILLBANK_ROOT", str(bank))

    agents = [
        CLIOnlyAgent(DummyClient(), verbose=False),
        CLISkillPreloadedAgent(DummyClient(), skills_dir=str(bank), verbose=False),
    ]
    for agent in agents:
        prompt = agent.build_task_prompt(context)
        assert "### working_directory\n.\n" in prompt
        assert "### spreadsheet_path\ninput.xlsx\n" in prompt
        assert "### output_path\noutput.xlsx\n" in prompt
        assert "update every required cell in that range" in prompt
        assert "verify representative target cells" in prompt
        assert str(work_dir) not in prompt
        assert str(context.input_file) not in prompt
        assert str(context.output_file) not in prompt


def test_spreadsheet_system_prompts_use_relative_io_examples():
    from spreadsheet_agent.system_prompts import load_full_system_prompt

    for filename in ("cli_only_full_system_v1.txt", "cli_skill_preloaded_full_system_v1.txt"):
        prompt = load_full_system_prompt(filename)
        assert "openpyxl.load_workbook('input.xlsx')" in prompt
        assert "wb.save('output.xlsx')" in prompt
        assert "Use `python -c` only for short read-only inspection" in prompt
        assert "Do NOT put compound Python logic inside `python -c`" in prompt
        assert "update every cell that the instruction requires within that range" in prompt
        assert "verify representative target cells" in prompt
        assert "cat <<'EOF' > solution.py" in prompt
        assert "for row in range(2, ws.max_row + 1):" in prompt
        assert "python solution.py" in prompt
        assert "python -c \"import openpyxl; wb = openpyxl.load_workbook('input.xlsx'); ws = wb.active; wb.save('output.xlsx')\"" not in prompt
        assert "/absolute/path/to/input.xlsx" not in prompt
        assert "/path/to/input.xlsx" not in prompt
        assert "Always use absolute paths" not in prompt


def test_bash_tool_recovers_from_invalid_python_c_syntax(tmp_path):
    from spreadsheet_agent.tools import create_bash_tool

    bash = create_bash_tool(str(tmp_path))
    result = bash.execute(command="python -c \"import sys; for row in range(2): print(row)\"")
    assert "SyntaxError" in result
    assert "[Recovery hint]" in result
    assert "solution.py" in result


def test_bash_tool_recovers_from_solution_py_syntax_error(tmp_path):
    from spreadsheet_agent.tools import create_bash_tool

    (tmp_path / "solution.py").write_text("formula = f'=\"broken\\n", encoding="utf-8")
    result = create_bash_tool(str(tmp_path)).execute(command="python solution.py")
    assert "SyntaxError" in result
    assert "[Recovery hint]" in result
    assert "final computed values directly" in result


def test_agent_runtime_env_uses_relative_io_names(tmp_path, monkeypatch):
    from spreadsheet_agent.agents.base import AgentContext
    from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent

    class DummyClient:
        pass

    work_dir = tmp_path / "work" / "task"
    work_dir.mkdir(parents=True)
    context = AgentContext(
        working_dir=str(work_dir),
        input_file=str(work_dir / "input.xlsx"),
        output_file=str(work_dir / "output.xlsx"),
        instruction="create output",
    )
    captured = {}

    class DummyReactAgent:
        def run(self, task_prompt):
            captured["task_prompt"] = task_prompt
            captured["input_file"] = os.environ["INPUT_FILE"]
            captured["output_file"] = os.environ["OUTPUT_FILE"]
            Path(context.output_file).write_text("placeholder", encoding="utf-8")
            return SimpleNamespace(success=True, total_turns=1, final_answer="", error="")

    agent = CLIOnlyAgent(DummyClient(), verbose=False)
    monkeypatch.setattr(agent, "_ensure_agent", lambda working_dir: DummyReactAgent())
    result = agent.run(context)
    assert result["success"] is True
    assert captured["input_file"] == "input.xlsx"
    assert captured["output_file"] == "output.xlsx"
    assert str(work_dir) not in captured["task_prompt"]


def test_l1_singleton_community_skips_analyst_and_records_reason():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder, LayerClusteringResult

    community = ExperienceCommunity(community_id="L1_C000", level=1, member_weights={"e1": 1.0})
    clustering = LayerClusteringResult(
        level=1,
        input_item_ids=["e1"],
        communities=[community],
        member_item_ids_by_community={"L1_C000": ["e1"]},
        stop_reason="",
    )
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower-level experience",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.9},
    )
    calls = []

    def summary_fn(comm, members, layer):
        calls.append((comm.community_id, [item.item_id for item in members]))
        return [ExperienceItem(
            item_id="e2",
            level=2,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text="higher-level experience",
            embedding=[1.0],
            generated_from_community_ids=[comm.community_id],
            metadata={"name": "Higher", "trigger": "higher", "content": "higher", "confidence": 0.9},
        )]

    generated = asyncio.run(ProjectedGmmTreeBuilder(default_hierarchy_config({}))._summarize_communities(clustering, items_by_id={"e1": member}, summary_fn=summary_fn))
    assert calls == []
    assert generated == []
    assert community.metadata["higher_level_distinct_child_count"] == 1
    assert community.metadata["higher_level_abstraction_skipped"] == "fewer_than_two_distinct_child_cards"


def test_dynamic_l1_singleton_skips_analyst_call():
    class DummyGeneration:
        async def chat_json(self, *args, **kwargs):
            raise AssertionError("L1+ singleton must not call the analyst LLM")

    analyst = ClusterAnalyst(
        DummyGeneration(),
        None,
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})
    member = ExperienceItem(item_id="e1", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="one lower card", embedding=[1.0])
    previous = [{"item_id": "old", "metadata": {"name": "Old", "trigger": "old", "content": "old", "confidence": 0.9}}]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))
    assert patches == []
    assert community.metadata["higher_level_abstraction_skipped"] == "fewer_than_two_distinct_child_cards"


def test_dynamic_l1_update_cannot_duplicate_child_card():
    class DummyGeneration:
        async def chat_json(self, *args, **kwargs):
            return {
                "updates": [{
                    "item_id": "old",
                    "name": "Lower",
                    "trigger": "lower",
                    "content": "lower",
                    "placement": {"target": "skill_md", "reference_kind": "procedure"},
                    "confidence": 0.9,
                }]
            }

    class DummyEmbedding:
        async def embed_texts(self, *args, **kwargs):
            raise AssertionError("duplicate higher-level update must not be embedded")

    analyst = ClusterAnalyst(
        DummyGeneration(),
        DummyEmbedding(),
        ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
    )
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    members = [
        ExperienceItem(
            item_id="e1",
            level=1,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text="# Lower\n\n## Trigger\nlower\n\n## Content\nlower\n",
            embedding=[1.0],
        ),
        ExperienceItem(item_id="e2", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="another child", embedding=[0.5]),
    ]
    previous = [{"item_id": "old", "metadata": {"name": "Old", "trigger": "old", "content": "old", "confidence": 0.9}}]

    assert asyncio.run(analyst.summarize_dynamic_update(community, members, previous)) == []
    assert community.metadata["higher_level_duplicate_output_count"] == 1
    assert community.metadata["higher_level_abstraction_skipped"] == "duplicate_existing_experience"


def test_l1_parent_duplicate_of_child_is_not_generated():
    from dynamix_core.tree_builder import LayerClusteringResult, ProjectedGmmTreeBuilder

    members = {
        "e1": ExperienceItem(
            item_id="e1",
            level=1,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text="name: Operand ledger\ntrigger: arithmetic\ncontent: Track each operand.",
            embedding=[1.0],
            metadata={"name": "Operand ledger", "trigger": "arithmetic", "content": "Track each operand.", "confidence": 0.9},
        ),
        "e2": ExperienceItem(
            item_id="e2",
            level=1,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text="name: Unit alignment\ntrigger: mixed units\ncontent: Normalize units before arithmetic.",
            embedding=[0.5],
            metadata={"name": "Unit alignment", "trigger": "mixed units", "content": "Normalize units before arithmetic.", "confidence": 0.9},
        ),
    }
    community = ExperienceCommunity(community_id="L1_C000", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    clustering = LayerClusteringResult(
        level=1,
        input_item_ids=list(members),
        communities=[community],
        member_item_ids_by_community={community.community_id: list(members)},
        stop_reason="",
    )

    def summary_fn(comm, child_cards, layer):
        return [ExperienceItem(
            item_id="e3",
            level=2,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text=members["e1"].text,
            embedding=[1.0],
            generated_from_community_ids=[comm.community_id],
            metadata={"name": "Operand ledger", "trigger": "arithmetic", "content": "Track each operand.", "confidence": 0.9},
        )]

    generated = asyncio.run(ProjectedGmmTreeBuilder(default_hierarchy_config({}))._summarize_communities(
        clustering,
        items_by_id=members,
        summary_fn=summary_fn,
    ))
    assert generated == []
    assert community.metadata["higher_level_abstraction_skipped"] == "duplicate_existing_experience"
    assert community.metadata["higher_level_duplicate_output_count"] == 1


def test_l1_distinct_children_can_generate_novel_parent():
    from dynamix_core.tree_builder import LayerClusteringResult, ProjectedGmmTreeBuilder

    members = {
        "e1": ExperienceItem(item_id="e1", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="operand provenance", embedding=[1.0]),
        "e2": ExperienceItem(item_id="e2", level=1, kind=ITEM_KIND_EXPERIENCE_CARD, text="unit alignment", embedding=[0.5]),
    }
    community = ExperienceCommunity(community_id="L1_C000", level=1, member_weights={"e1": 1.0, "e2": 1.0})
    clustering = LayerClusteringResult(
        level=1,
        input_item_ids=list(members),
        communities=[community],
        member_item_ids_by_community={community.community_id: list(members)},
        stop_reason="",
    )

    def summary_fn(comm, child_cards, layer):
        return [ExperienceItem(
            item_id="e3",
            level=2,
            kind=ITEM_KIND_EXPERIENCE_CARD,
            text="preserve semantic comparability before calculation",
            embedding=[0.75],
            generated_from_community_ids=[comm.community_id],
            metadata={"name": "Semantic comparability", "trigger": "derived values", "content": "Align provenance and units.", "confidence": 0.9},
        )]

    generated = asyncio.run(ProjectedGmmTreeBuilder(default_hierarchy_config({}))._summarize_communities(
        clustering,
        items_by_id=members,
        summary_fn=summary_fn,
    ))
    assert [item.item_id for item in generated] == ["e3"]
    assert community.metadata["higher_level_distinct_child_count"] == 2
    assert "higher_level_abstraction_skipped" not in community.metadata


def test_higher_level_skipped_community_can_commit_without_parent_card():
    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t1", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 1", embedding=[1.0]),
            ExperienceItem(item_id="t2", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 2", embedding=[0.5]),
        ])
        l0 = ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t1": 1.0, "t2": 1.0})
        l1_cards = [
            ExperienceItem(
                item_id=item_id,
                level=1,
                kind=ITEM_KIND_EXPERIENCE_CARD,
                text=text,
                embedding=[embedding],
                generated_from_community_ids=[l0.community_id],
                metadata={"confidence": 0.9},
            )
            for item_id, text, embedding in [("e1", "card one", 1.0), ("e2", "card two", 0.5)]
        ]
        await state.commit_layer(level=0, communities=[l0], generated_items=l1_cards)
        l1 = ExperienceCommunity(
            community_id="L1_C0",
            level=1,
            member_weights={"e1": 1.0, "e2": 1.0},
            metadata={"higher_level_abstraction_skipped": "no_shared_invariant"},
        )
        await state.commit_layer(
            level=1,
            communities=[l1],
            generated_items=[],
            stop_reason="no_new_abstractions",
        )
        return state

    state = asyncio.run(run_case())
    committed = asyncio.run(state.community_objects_at_level(1))
    assert len(committed) == 1
    assert committed[0].generated_item_ids == []
    payload = asyncio.run(state.to_dict())
    assert payload["layers"]["1"]["generated_item_ids"] == []
    assert payload["layers"]["1"]["stop_reason"] == "no_new_abstractions"


def test_build_layer_commits_higher_level_links_when_no_parent_is_generated():
    from dynamix_core.tree_builder import LayerClusteringResult, ProjectedGmmTreeBuilder

    async def run_case():
        state = ExperienceHierarchyState()
        await state.initialize_trajectory_items([
            ExperienceItem(item_id="t1", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 1", embedding=[1.0]),
            ExperienceItem(item_id="t2", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace 2", embedding=[0.5]),
        ])
        l0 = ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t1": 1.0, "t2": 1.0})
        cards = [
            ExperienceItem(
                item_id=item_id,
                level=1,
                kind=ITEM_KIND_EXPERIENCE_CARD,
                text=text,
                embedding=[embedding],
                generated_from_community_ids=[l0.community_id],
                metadata={"confidence": 0.9},
            )
            for item_id, text, embedding in [("e1", "card one", 1.0), ("e2", "card two", 0.5)]
        ]
        await state.commit_layer(level=0, communities=[l0], generated_items=cards)
        l1 = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0, "e2": 1.0})
        clustering = LayerClusteringResult(
            level=1,
            input_item_ids=["e1", "e2"],
            communities=[l1],
            member_item_ids_by_community={l1.community_id: ["e1", "e2"]},
            stop_reason="",
        )
        builder = ProjectedGmmTreeBuilder(default_hierarchy_config({}))

        async def cluster_layer(items, **kwargs):
            return clustering

        builder.cluster_layer = cluster_layer  # type: ignore[method-assign]
        result = await builder.build_layer(state, level=1, items=cards, summary_fn=lambda *args: [])
        return result, await state.to_dict()

    result, payload = asyncio.run(run_case())
    assert result.committed is True
    assert result.generated_item_ids == []
    assert payload["layers"]["1"]["community_ids"] == ["L1_C0"]
    assert payload["layers"]["1"]["generated_item_ids"] == []
    assert payload["communities"]["L1_C0"]["metadata"]["higher_level_abstraction_skipped"] == "no_shared_invariant"


def test_full_soft_hard_materializer_excludes_testcase_not_task_and_normalizes_input(tmp_path):
    full_eval = _load_full_soft_hard_module()
    full = tmp_path / "full"
    verified = tmp_path / "verified"

    dataset = [
        {"id": "A", "spreadsheet_path": "spreadsheet/A", "instruction": "do A", "answer_position": "A1"},
        {"id": "B", "spreadsheet_path": "spreadsheet/B", "instruction": "do B", "answer_position": "A1"},
    ]
    (full / "spreadsheet" / "A").mkdir(parents=True)
    (full / "spreadsheet" / "B").mkdir(parents=True)
    (verified / "spreadsheet" / "A").mkdir(parents=True)
    (full / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (verified / "dataset.json").write_text(json.dumps(dataset[:1]), encoding="utf-8")

    def write(path: Path, text: str) -> None:
        path.write_bytes(text.encode("utf-8"))

    write(full / "spreadsheet" / "A" / "1_A_input.xlsx", "verified input")
    write(full / "spreadsheet" / "A" / "1_A_answer.xlsx", "verified answer")
    write(full / "spreadsheet" / "A" / "2_A_input .xlsx", "input 2")
    write(full / "spreadsheet" / "A" / "2_A_answer.xlsx", "answer 2")
    write(full / "spreadsheet" / "A" / "3_A_input.xlsx", "input 3")
    write(full / "spreadsheet" / "A" / "3_A_answer.xlsx", "answer 3")
    for idx in (1, 2, 3):
        write(full / "spreadsheet" / "B" / f"{idx}_B_input.xlsx", f"input B{idx}")
        write(full / "spreadsheet" / "B" / f"{idx}_B_answer.xlsx", f"answer B{idx}")
    write(verified / "spreadsheet" / "A" / "1_A_init.xlsx", "verified input")
    write(verified / "spreadsheet" / "A" / "1_A_golden.xlsx", "verified answer")

    result = full_eval.materialize_paper_aligned_dataset(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=tmp_path / "prepared",
        train_start=0,
        train_end=1,
        expected_full_testcases=6,
        expected_excluded_testcases=1,
        expected_prepared_testcases=5,
        expected_normalized_input_files=1,
    )

    prepared_a = tmp_path / "prepared" / "spreadsheet" / "A"
    prepared_b = tmp_path / "prepared" / "spreadsheet" / "B"
    assert not (prepared_a / "1_A_input.xlsx").exists()
    assert not (prepared_a / "1_A_answer.xlsx").exists()
    assert (prepared_a / "2_A_input.xlsx").exists()
    assert not (prepared_a / "2_A_input .xlsx").exists()
    assert len(list(prepared_b.glob("*_answer.xlsx"))) == 3
    manifest = result["testcase_filter_manifest"]
    assert manifest["full_answer_testcases"] == 6
    assert manifest["excluded_testcases"] == 1
    assert manifest["prepared_answer_testcases"] == 5
    assert manifest["exclusions"][0]["match_method"] == "hash_input_and_answer"
    assert result["normalization_manifest"]["normalized_input_files"] == 1
    result_again = full_eval.materialize_paper_aligned_dataset(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=tmp_path / "prepared_again",
        train_start=0,
        train_end=1,
        expected_full_testcases=6,
        expected_excluded_testcases=1,
        expected_prepared_testcases=5,
        expected_normalized_input_files=1,
    )
    assert result_again["stable_materialization_hash"] == result["stable_materialization_hash"]
    write(full / "spreadsheet" / "A" / "2_A_answer.xlsx", "answer 2 changed")
    result_changed = full_eval.materialize_paper_aligned_dataset(
        full_data_path=full,
        verified_data_path=verified,
        output_dir=tmp_path / "prepared_changed",
        train_start=0,
        train_end=1,
        expected_full_testcases=6,
        expected_excluded_testcases=1,
        expected_prepared_testcases=5,
        expected_normalized_input_files=1,
    )
    assert result_changed["stable_materialization_hash"] != result["stable_materialization_hash"]


def test_full_soft_hard_materializer_rejects_output_overlap_with_sources(tmp_path):
    full_eval = _load_full_soft_hard_module()
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    full.mkdir()
    verified.mkdir()

    with pytest.raises(RuntimeError, match="must not overlap"):
        full_eval.materialize_paper_aligned_dataset(
            full_data_path=full,
            verified_data_path=verified,
            output_dir=full / "prepared",
            expected_full_testcases=None,
            expected_excluded_testcases=None,
            expected_prepared_testcases=None,
            expected_normalized_input_files=None,
        )
    assert full.exists()

    with pytest.raises(RuntimeError, match="must not overlap"):
        full_eval.materialize_paper_aligned_dataset(
            full_data_path=full,
            verified_data_path=verified,
            output_dir=tmp_path,
            expected_full_testcases=None,
            expected_excluded_testcases=None,
            expected_prepared_testcases=None,
            expected_normalized_input_files=None,
        )
    assert verified.exists()


def test_full_soft_hard_materializer_rejects_partial_hash_match_without_exact_fallback(tmp_path):
    full_eval = _load_full_soft_hard_module()
    full = tmp_path / "full"
    verified = tmp_path / "verified"
    dataset = [{"id": "C", "spreadsheet_path": "spreadsheet/C", "instruction": "do C", "answer_position": "A1"}]
    (full / "spreadsheet" / "C").mkdir(parents=True)
    (verified / "spreadsheet" / "C").mkdir(parents=True)
    (full / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (verified / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (full / "spreadsheet" / "C" / "2_C_input.xlsx").write_text("same input", encoding="utf-8")
    (full / "spreadsheet" / "C" / "2_C_answer.xlsx").write_text("different answer", encoding="utf-8")
    (verified / "spreadsheet" / "C" / "1_C_init.xlsx").write_text("same input", encoding="utf-8")
    (verified / "spreadsheet" / "C" / "1_C_golden.xlsx").write_text("verified answer", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not map verified train testcase"):
        full_eval.materialize_paper_aligned_dataset(
            full_data_path=full,
            verified_data_path=verified,
            output_dir=tmp_path / "prepared",
            train_start=0,
            train_end=1,
            expected_full_testcases=1,
            expected_excluded_testcases=1,
            expected_prepared_testcases=None,
            expected_normalized_input_files=None,
        )


def test_full_soft_hard_runtime_args_inherit_source_summary_and_reject_multi_seed():
    full_eval = _load_full_soft_hard_module()
    args = SimpleNamespace(
        model=None,
        openai_base_url=None,
        openai_api_key=None,
        embedding_base_url=None,
        embedding_model=None,
        skillbank_top_k=None,
        workers=None,
        max_turns=None,
        temperature=None,
        llm_client=None,
        llm_timeout_seconds=None,
        llm_retry_wait_seconds=None,
        num_random_seeds=1,
        repeat=1,
    )
    sources = full_eval.resolve_runtime_args(args, {
        "model": "Qwen3.5-9B-AWQ",
        "openai_base_url": "http://llm/v1",
        "embedding_base_url": "http://embed/v1",
        "embedding_model": "Qwen3-Embedding-8B",
        "skillbank_top_k": 10,
        "workers": 8,
        "max_turns": 100,
        "rollout_temperature": 0.0,
        "rollout_llm_client": "openai",
        "rollout_client_timeout_seconds": 1200.0,
        "rollout_client_retry_wait_seconds": [5.0, 10.0],
    })
    assert args.model == "Qwen3.5-9B-AWQ"
    assert args.skillbank_top_k == 10
    assert args.openai_base_url == "http://llm/v1"
    assert args.llm_retry_wait_seconds == "5.0,10.0"
    assert sources["model"].startswith("source_run_dir")

    args.repeat = 2
    with pytest.raises(ValueError, match="single-run eval"):
        full_eval.resolve_runtime_args(args, {})


def test_evaluator_reports_raw_soft_hard_alongside_recalc(tmp_path, monkeypatch):
    import evaluate_with_official as evaluator

    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    sheet = data / "spreadsheet" / "A"
    out_sheet = outputs / "spreadsheet" / "A"
    sheet.mkdir(parents=True)
    out_sheet.mkdir(parents=True)
    (data / "dataset.json").write_text(json.dumps([{
        "id": "A",
        "spreadsheet_path": "spreadsheet/A",
        "instruction": "fill",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "A1",
    }]), encoding="utf-8")
    for idx in (1, 2):
        (sheet / f"{idx}_A_answer.xlsx").write_text(f"answer {idx}", encoding="utf-8")
        (out_sheet / f"{idx}_A_output.xlsx").write_text(f"output {idx}", encoding="utf-8")

    monkeypatch.setattr(evaluator, "_preflight_libreoffice", lambda: "soffice")

    def fake_recalc(input_path, recalc_dir, instance_id, *, soffice=None):
        recalc_path = Path(recalc_dir) / f"{Path(input_path).stem}.recalc.xlsx"
        recalc_path.parent.mkdir(parents=True, exist_ok=True)
        recalc_path.write_text("recalc", encoding="utf-8")
        return str(recalc_path)

    def fake_compare(gt_path, output_path, instruction_type, answer_position):
        if output_path.endswith(".recalc.xlsx"):
            return True, "recalc ok"
        return output_path.endswith("1_A_output.xlsx"), "raw"

    monkeypatch.setattr(evaluator, "_recalculate_workbook", fake_recalc)
    monkeypatch.setattr(evaluator, "compare_workbooks", fake_compare)

    result = evaluator.evaluate(str(data), str(outputs), recalc_dir=str(tmp_path / "recalc"))
    summary = result["summary"]
    assert summary["avg_soft_score"] == 1.0
    assert summary["avg_hard_score"] == 1.0
    raw = summary["trace2skill_compatible_no_recalc"]
    assert raw["avg_soft_score"] == 0.5
    assert raw["avg_hard_score"] == 0.0
    assert raw["test_case_accuracy"] == 0.5
    assert result["results"][0]["raw_soft_score"] == 0.5
    assert result["results"][0]["raw_hard_score"] == 0


def test_full_soft_hard_wrapper_uses_trace2skill_runner_and_evaluator(tmp_path):
    full_eval = _load_full_soft_hard_module()
    args = SimpleNamespace(
        python_executable="/env/bin/python",
        model="Qwen3.5-9B-AWQ",
        llm_client="openai",
        temperature=0.0,
        llm_timeout_seconds=600.0,
        llm_retry_wait_seconds="5.0,10.0,30.0",
        num_random_seeds=1,
        repeat=1,
        max_turns=100,
        workers=8,
    )
    rollout = full_eval.build_rollout_command(
        args,
        tmp_path / "prepared_dataset",
        tmp_path / "trace2skill_full_outputs",
        tmp_path / "trace2skill_full_logs",
        tmp_path / "trace2skill_full_results.json",
        tmp_path / "skills",
        tmp_path / "trace2skill_generation_config.json",
    )
    eval_cmd = full_eval.build_eval_command(
        args,
        tmp_path / "prepared_dataset",
        tmp_path / "trace2skill_full_outputs",
        tmp_path / "full_soft_hard_eval.json",
        tmp_path / "recalc",
    )
    assert "run_spreadsheetbench.py" in rollout
    assert "evaluate_with_official.py" in eval_cmd
    assert "--agent" in rollout
    assert rollout[rollout.index("--agent") + 1] == "cli_skill_preloaded"
    assert "--recalc_dir" in eval_cmd

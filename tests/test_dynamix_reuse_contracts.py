from __future__ import annotations

import json
import asyncio
import importlib.util
import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynamix_trace2skill.clients import EmbeddingClient, EmbeddingConfig, GenerationClient, GenerationConfig
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig
from dynamix_trace2skill.log_parser import parse_trace2skill_logs, _result_fields
from dynamix_trace2skill.pipeline import DynaMixRunConfig, default_hierarchy_config
from dynamix_core.data_structures import ExperienceCardPatch, ExperienceCommunity, ExperienceHierarchyState, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY
from dynamix_core.update import ExperienceHierarchyDynamicUpdater


def _load_experiment_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamix_trace2skill_experiment.py"
    spec = importlib.util.spec_from_file_location("run_dynamix_trace2skill_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_officeqa_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_officeqa_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_officeqa_benchmark", path)
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


def test_chat_json_keeps_guided_schema_in_prompt_not_response_format():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=False))
    seen = {}
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}

    async def fake_chat_text(messages, **kwargs):
        seen.update(kwargs)
        seen["messages"] = messages
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
    assert seen["response_format"] is None
    assert seen["messages"][0]["role"] == "system"
    assert "JSON output contract for MinimalClusterExperienceCards" in seen["messages"][0]["content"]
    assert '"required": ["cards"]' in seen["messages"][0]["content"]
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


def test_chat_json_extracts_embedded_json_with_prompt_schema():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=True))
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}

    async def fake_chat_text(messages, **kwargs):
        return 'Here is the JSON you requested:\n{"cards": []}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            retries=0,
        )
    )
    assert result == {"cards": []}


def test_chat_json_retries_when_embedded_json_misses_required_key():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=True))
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}
    calls = []

    async def fake_chat_text(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return 'Reasoning with unrelated JSON:\n{"notes": []}'
        return '{"cards": []}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            retries=1,
        )
    )

    assert result == {"cards": []}
    assert "missing required top-level keys" in calls[1][-1]["content"]


def test_chat_json_strips_dangling_qwen_think_suffix_before_json():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=True))
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}

    async def fake_chat_text(messages, **kwargs):
        return 'Thinking Process with distracting JSON {"notes": []}\n</think>\n{"cards": []}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            retries=0,
        )
    )

    assert result == {"cards": []}


def test_chat_json_retries_when_nested_schema_is_invalid():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=True))
    schema = {
        "type": "object",
        "properties": {
            "cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["name", "confidence"],
                },
            }
        },
        "required": ["cards"],
    }
    calls = []

    async def fake_chat_text(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return '{"cards": [{"name": "missing confidence"}]}'
        if len(calls) == 2:
            return '{"cards": [{"name": "bad confidence", "confidence": 2.0}]}'
        return '{"cards": [{"name": "ok", "confidence": 0.7}]}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            retries=2,
        )
    )

    assert result == {"cards": [{"name": "ok", "confidence": 0.7}]}
    assert "missing required keys" in calls[1][-1]["content"]
    assert "above maximum" in calls[2][-1]["content"]


def test_chat_json_retries_local_parse_after_invalid_json():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic", thinking_mode=False))
    schema = {"type": "object", "properties": {"cards": {"type": "array"}}, "required": ["cards"]}
    calls = []

    async def fake_chat_text(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return "not json"
        return '{"cards": []}'

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json=schema,
            retries=1,
        )
    )
    assert result == {"cards": []}
    assert len(calls) == 2
    assert "Previous parse error" in calls[1][-1]["content"]


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
    assert cfg.gmm_bic.min_split_size == 4
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
        json.dumps({"chunk_tokens": 28000, "overlap_tokens": 1000, "pooling": "mean", "max_token_count": 90000, "over_limit_chunk_count": 0}),
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
        chunked_embedding_chunk_tokens=28000,
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


def test_cluster_prompt_uses_minimal_experience_schema():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_schema", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "bundle"})
    payload = json.loads(analyst._build_prompt(community, [member], "raw_extractor"))
    schema = payload["output_schema"]
    assert set(schema) == {"cards"}
    card_schema = schema["cards"][0]
    assert set(card_schema) == {"name", "trigger", "content", "placement", "confidence"}
    forbidden = {"shared_patterns", "success_motifs", "anti_patterns", "shared_patch_hints", "reference_materials", "script_files", "skill_placement"}
    assert not (forbidden & set(card_schema))
    constraints = " ".join(payload["hard_constraints"])
    assert "Do not output fields except cards and each card's name, trigger, content, placement, confidence" in constraints


def test_static_cluster_analyst_passes_guided_json_schema_without_forcing_thinking():
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
    assert set(payload) == {"instruction", "analyst_mode", "dynamic_patch_policy", "hard_constraints", "members", "previous_generated_experiences"}
    assert "community" not in payload
    assert "output_schema" not in payload
    assert all("support_mass" not in member for member in payload["members"])
    assert all("support_mass" not in card for card in payload["previous_generated_experiences"])
    assert generation.kwargs["guided_json"]["required"] == ["updates", "new_cards"]
    assert "new_cards" in generation.kwargs["guided_json"]["properties"]
    assert generation.kwargs["max_tokens"] == 7777
    assert "extra_body" not in generation.kwargs


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
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
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

        patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

        assert patches == []
        assert embedding.calls == 0
        assert len(generation.messages) == 3
        assert all(call["retries"] == 0 for call in generation.kwargs)
        assert all(call["guided_json"]["required"] == ["updates"] for call in generation.kwargs)
        assert all("new_cards" not in json.dumps(call["guided_json"]) for call in generation.kwargs)
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
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

    assert len(patches) == 1
    assert patches[0].operation == "update"
    assert patches[0].item_id == "old_l2"
    assert patches[0].metadata["dynamic_patch_operation"] == "update"
    assert len(generation.messages) == 2
    assert all(call["retries"] == 0 for call in generation.kwargs)
    assert all(call["guided_json"]["required"] == ["updates"] for call in generation.kwargs)
    assert all("new_cards" not in json.dumps(call["guided_json"]) for call in generation.kwargs)
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
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

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
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e1": 1.0})
    member = ExperienceItem(
        item_id="e1",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="lower card",
        embedding=[1.0],
        metadata={"name": "Lower", "trigger": "lower", "content": "lower", "confidence": 0.8},
    )
    previous = [{
        "item_id": "old_l2",
        "metadata": {"name": "Old L2", "trigger": "old", "content": "old", "confidence": 0.9},
    }]

    patches = asyncio.run(analyst.summarize_dynamic_update(community, [member], previous))

    assert len(patches) == 1
    assert patches[0].operation == "update"
    assert patches[0].item_id == "old_l2"
    assert patches[0].metadata["dynamic_patch_operation"] == "update"
    assert patches[0].metadata["higher_level_single_card_enforced"] is True


def test_dynamic_analyst_retries_then_ignores_invalid_patch_ids():
    from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig

    class DummyEmbedding:
        async def embed_texts(self, texts, *, cache_namespace=None):
            return [[1.0] for _ in texts]

    async def run_payload(payload):
        class DummyGeneration:
            def __init__(self):
                self.calls = 0

            async def chat_json(self, messages, *, schema_name, **kwargs):
                self.calls += 1
                return payload

        generation = DummyGeneration()
        analyst = ClusterAnalyst(
            generation,
            DummyEmbedding(),
            ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True),
        )
        community = ExperienceCommunity(community_id="C0", level=0, member_weights={"t0": 1.0})
        member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0])
        previous = [{"item_id": "old_a", "metadata": {"name": "Old", "trigger": "old", "content": "old", "confidence": 0.9}}]
        return await analyst.summarize_dynamic_update(community, [member], previous), generation.calls

    valid_card = {
        "name": "Card",
        "trigger": "trigger",
        "content": "content",
        "placement": {"target": "skill_md", "reference_kind": "procedure"},
        "confidence": 0.8,
    }
    result, calls = asyncio.run(run_payload({"updates": [{"item_id": "missing", **valid_card}], "new_cards": []}))
    assert result == []
    assert calls == 3
    result, calls = asyncio.run(run_payload({"updates": [{"item_id": "old_a", **valid_card}, {"item_id": "old_a", **valid_card}], "new_cards": []}))
    assert result == []
    assert calls == 3
    result, calls = asyncio.run(run_payload({"updates": [], "new_cards": [{"item_id": "illegal", **valid_card}]}))
    assert result == []
    assert calls == 3


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


def test_dynamic_l0_overflow_singleton_empty_patch_is_marked_skipped():
    async def run_case():
        state = ExperienceHierarchyState()
        item = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0])
        await state.initialize_trajectory_items([item])
        community = ExperienceCommunity(
            community_id="L0_DYN_t0",
            level=0,
            member_weights={"t0": 1.0},
            posterior_member_weights={"t0": 1.0},
            clustering_method="dynamic_budget_overflow_singleton",
            support_mass=1.0,
            failure_count=1,
            metadata={
                "created_by": "dynamic_budget_constrained_online_gmm",
                "seed_item_id": "t0",
                "prompt_token_cost": 100,
                "budget": 85000,
                "split_reason": "dynamic_l0_budget_overflow_new_component",
            },
        )
        state._communities[community.community_id] = community

        result = await state.commit_dynamic_community_update(community=community, patches=[])
        updated = (await state.community_objects([community.community_id]))[0]
        return result, updated

    result, updated = asyncio.run(run_case())

    assert result.changed_item_ids == []
    assert result.requires_reroute_item_ids == []
    assert updated.generated_item_ids == []
    assert updated.metadata["dynamic_llm_summary_skipped"] is True
    assert updated.metadata["dynamic_summary_skip_reason"] == "empty_dynamic_l0_patch"


def test_dynamic_plain_empty_patch_still_rejects_missing_generated_cards():
    async def run_case():
        state = ExperienceHierarchyState()
        item = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0])
        await state.initialize_trajectory_items([item])
        community = ExperienceCommunity(
            community_id="L0_C0",
            level=0,
            member_weights={"t0": 1.0},
            clustering_method="projected_gmm_bic",
            support_mass=1.0,
        )
        state._communities[community.community_id] = community
        return await state.commit_dynamic_community_update(community=community, patches=[])

    with pytest.raises(ValueError, match="no generated cards"):
        asyncio.run(run_case())


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


def test_l1_singleton_community_is_summarized_by_analyst():
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
    assert calls == [("L1_C000", ["e1"])]
    assert [item.item_id for item in generated] == ["e2"]


def test_default_hierarchy_config_passes_static_ablation_tree_policy():
    cfg = default_hierarchy_config({"tree_policy": "identity_singleton"})
    assert cfg.tree_policy == "identity_singleton"


def test_identity_singleton_tree_policy_creates_one_l0_community_per_fit_item():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "tree_policy": "identity_singleton",
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5},
    })
    items = [
        ExperienceItem(item_id="fit_a", level=0, kind=ITEM_KIND_TRAJECTORY, text="a", embedding=[1.0, 0.0], metadata={"analysis_token_count": 10}),
        ExperienceItem(item_id="fit_b", level=0, kind=ITEM_KIND_TRAJECTORY, text="b", embedding=[0.0, 1.0], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="too_big", level=0, kind=ITEM_KIND_TRAJECTORY, text="big", embedding=[0.5, 0.5], metadata={"analysis_token_count": 60}),
    ]

    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))

    assert clustering.stop_reason == ""
    assert [community.clustering_method for community in clustering.communities] == ["identity_singleton", "identity_singleton"]
    assert [list(community.member_weights) for community in clustering.communities] == [["fit_a"], ["fit_b"]]
    assert clustering.excluded_input_item_ids == ["too_big"]
    assert clustering.summary_budget["excluded_oversize_singletons"][0]["reason"] == "oversize_singleton"


def test_projected_kmeans_elbow_selects_hard_two_cluster_split():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "tree_policy": "projected_kmeans_elbow",
        "gmm_bic": {
            "min_split_size": 2,
            "min_effective_samples_per_component": 1,
            "abs_kmax": 4,
            "num_restarts": 3,
        },
        "summary_budget": {"max_model_tokens": 100000, "budget_ratio": 0.8},
    })
    items = [
        ExperienceItem(item_id="a0", level=0, kind=ITEM_KIND_TRAJECTORY, text="a0", embedding=[1.0, 0.0], metadata={"analysis_token_count": 1}),
        ExperienceItem(item_id="a1", level=0, kind=ITEM_KIND_TRAJECTORY, text="a1", embedding=[0.98, 0.02], metadata={"analysis_token_count": 1}),
        ExperienceItem(item_id="a2", level=0, kind=ITEM_KIND_TRAJECTORY, text="a2", embedding=[0.96, 0.04], metadata={"analysis_token_count": 1}),
        ExperienceItem(item_id="b0", level=0, kind=ITEM_KIND_TRAJECTORY, text="b0", embedding=[0.0, 1.0], metadata={"analysis_token_count": 1}),
        ExperienceItem(item_id="b1", level=0, kind=ITEM_KIND_TRAJECTORY, text="b1", embedding=[0.02, 0.98], metadata={"analysis_token_count": 1}),
        ExperienceItem(item_id="b2", level=0, kind=ITEM_KIND_TRAJECTORY, text="b2", embedding=[0.04, 0.96], metadata={"analysis_token_count": 1}),
    ]

    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))
    memberships = {item_id: 0 for item_id in [item.item_id for item in items]}
    for community in clustering.communities:
        for item_id in community.member_weights:
            memberships[item_id] += 1

    assert clustering.chosen_k == 2
    assert {community.clustering_method for community in clustering.communities} == {"kmeans_elbow"}
    assert set(memberships.values()) == {1}


def test_local_pca_project_async_returns_quickly_for_small_matrix():
    from dynamix_core.config import ProjectionConfig
    from dynamix_core.projection import local_pca_project_async

    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ],
        dtype=float,
    )

    projection = asyncio.run(asyncio.wait_for(local_pca_project_async(matrix, ProjectionConfig(max_dim=2)), timeout=5.0))

    assert projection.projected.shape[0] == 4
    assert projection.dim >= 1


def test_primary_argmax_gmm_uses_one_hot_member_weights():
    from dynamix_core.gmm_bic import GmmCandidateFit
    from dynamix_core.tree_builder import _candidate_layer_parts

    cfg = default_hierarchy_config({"soft_membership": {"recursive_assignment": "primary_argmax"}})
    item_ids = ["a", "b"]
    items_by_id = {
        item_id: ExperienceItem(
            item_id=item_id,
            level=0,
            kind=ITEM_KIND_TRAJECTORY,
            text=item_id,
            embedding=[1.0, 0.0],
            metadata={"analysis_token_count": 1},
        )
        for item_id in item_ids
    }
    fit = GmmCandidateFit(
        k=2,
        valid=True,
        bic=1.0,
        log_likelihood=-1.0,
        pi=np.asarray([0.5, 0.5], dtype=float),
        means=np.asarray([[0.0], [1.0]], dtype=float),
        variances=np.asarray([[1.0], [1.0]], dtype=float),
        responsibilities=np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float),
        primary_labels=np.asarray([0, 1], dtype=int),
        component_masses=[1.0, 1.0],
        child_sizes=[1, 1],
    )

    parts = _candidate_layer_parts(
        fit,
        level=0,
        input_ids=item_ids,
        items_by_id=items_by_id,
        token_counts={item_id: 1 for item_id in item_ids},
        soft_config=cfg.soft_membership,
    )

    assert parts is not None
    weights_by_child = {community.community_id: community.member_weights for community in parts["communities"]}
    assert weights_by_child == {"L0_C0": {"a": 1.0}, "L0_C1": {"b": 1.0}}
    posterior_by_child = {community.community_id: community.posterior_member_weights for community in parts["communities"]}
    assert posterior_by_child == {"L0_C0": {"a": 1.0}, "L0_C1": {"b": 1.0}}
    assert all("raw_posterior_member_weights" not in community.metadata for community in parts["communities"])


def test_projected_kmeans_elbow_budget_refinement_stays_kmeans_only():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "tree_policy": "projected_kmeans_elbow",
        "gmm_bic": {
            "min_split_size": 2,
            "min_effective_samples_per_component": 1,
            "abs_kmax": 4,
            "num_restarts": 3,
        },
        "summary_budget": {"max_model_tokens": 100, "budget_ratio": 0.5, "prompt_overhead_reserve_tokens": 0},
        "budget_refinement": {"enabled": True, "apply_to_level": 0, "min_token_reduction_fraction": 0.0},
    })
    items = [
        ExperienceItem(item_id="a0", level=0, kind=ITEM_KIND_TRAJECTORY, text="a0", embedding=[1.0, 0.0], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="a1", level=0, kind=ITEM_KIND_TRAJECTORY, text="a1", embedding=[0.99, 0.01], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="a2", level=0, kind=ITEM_KIND_TRAJECTORY, text="a2", embedding=[0.98, 0.02], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="b0", level=0, kind=ITEM_KIND_TRAJECTORY, text="b0", embedding=[0.0, 1.0], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="b1", level=0, kind=ITEM_KIND_TRAJECTORY, text="b1", embedding=[0.01, 0.99], metadata={"analysis_token_count": 20}),
        ExperienceItem(item_id="b2", level=0, kind=ITEM_KIND_TRAJECTORY, text="b2", embedding=[0.02, 0.98], metadata={"analysis_token_count": 20}),
    ]

    clustering = asyncio.run(ProjectedGmmTreeBuilder(cfg).cluster_layer(items, level=0))
    refinement = clustering.summary_budget["refinement_routing_tree"]
    node_kinds = {node["kind"] for node in refinement["nodes"].values()}
    split_events = clustering.summary_budget["split_events"]

    assert "gmm_split" not in node_kinds
    assert "kmeans_elbow_split" in node_kinds
    assert all("bic_by_k" not in event for event in split_events)
    assert any("inertia_by_k" in event for event in split_events)
    assert all(
        event.get("selected_k") == event.get("elbow_selected_k")
        for event in split_events
        if event.get("split_reason") == "budget_forced_kmeans_elbow_progress"
    )
    assert clustering.bic_by_k == {}
    assert "inertia_by_k" in clustering.summary_budget


def test_nodebank_export_level_filter_keeps_requested_experience_levels(tmp_path):
    from dynamix_core.skill_export import SkillExportConfig, export_skill_files_from_payload

    def card(item_id: str, level: int) -> dict:
        return {
            "item_id": item_id,
            "level": level,
            "kind": ITEM_KIND_EXPERIENCE_CARD,
            "text": "card",
            "support_mass": 1.0,
            "metadata": {
                "name": f"name {item_id}",
                "trigger": "trigger",
                "content": "content",
                "confidence": 0.9,
            },
        }

    payload = {
        "items": {
            "l1": card("l1", 1),
            "l2": card("l2", 2),
            "raw": {"item_id": "raw", "level": 0, "kind": ITEM_KIND_TRAJECTORY},
        }
    }

    l1 = export_skill_files_from_payload(payload, tmp_path / "l1", config=SkillExportConfig(min_level=1, max_level=1))
    l1_manifest = json.loads(Path(l1.manifest_path).read_text(encoding="utf-8"))
    l2plus = export_skill_files_from_payload(payload, tmp_path / "l2plus", config=SkillExportConfig(min_level=2))
    l2_manifest = json.loads(Path(l2plus.manifest_path).read_text(encoding="utf-8"))

    assert [node["item_id"] for node in l1_manifest["nodes"]] == ["l1"]
    assert l1_manifest["export_policy"]["level_filter"] == {"min_level": 1, "max_level": 1}
    assert [node["item_id"] for node in l2_manifest["nodes"]] == ["l2"]


def test_reuse_tree_protocol_rejects_filtered_source_tree(tmp_path):
    runner = _load_experiment_runner_module()
    reuse_tree = tmp_path / "source_tree"
    (reuse_tree / "analysis").mkdir(parents=True)
    records = tmp_path / "ordered_records.json"
    records.write_text('[{"task_id": "0-0"}]', encoding="utf-8")
    source_runtime = {
        "scenario": "static_build",
        "dataset_path": str(tmp_path / "data"),
        "train_start": 0,
        "train_end": 1,
        "enforce_dataset_order": True,
        "records_path": str(records),
        "generation": {"base_url": "mock://deterministic", "model": "mock", "temperature": 0.0, "thinking_mode": False, "extra_body": {}},
        "embedding": {"base_url": "mock://deterministic", "model": "embed", "max_model_len": 8192, "max_input_tokens": 8000, "truncate_long_texts": True, "tokenizer_model": "tok", "tokenizer_required": False, "truncation_strategy": "head"},
        "chunked_embedding": {
            "enabled": True,
            "tokenizer_model": "tok",
            "chunk_tokens": 7600,
            "overlap_tokens": 512,
            "pooling": "mean",
            "add_special_tokens": False,
            "normalize_after_pooling": True,
            "fail_if_chunk_exceeds_model_limit": True,
        },
        "hierarchy": {"tree_policy": "projected_gmm_bic", "dynamic_update": {"ignored_for_static": True}},
        "analyst": {"prompt_style": "default"},
        "max_levels": 8,
    }
    (reuse_tree / "analysis" / "runtime_config.json").write_text(json.dumps(source_runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    current_config = {
        **source_runtime,
        "records_path": str(tmp_path / "current_ordered_records.json"),
        "benchmark": "spreadsheetbench",
        "chunked_embedding": {
            "enabled": True,
            "chunk_tokens": 7600,
            "overlap_tokens": 512,
            "pooling": "mean",
            "add_special_tokens": False,
            "normalize_after_pooling": True,
            "fail_if_chunk_exceeds_model_limit": True,
        },
        "skill_export": {"min_level": 1, "max_level": 1, "max_node_count": None},
    }

    runner.validate_reused_tree_protocol(
        reuse_tree_dir=reuse_tree,
        current_config=current_config,
        source_summary={"skill_export": {"min_level": None, "max_level": None, "max_node_count": None}},
        current_records_sha256=runner.file_sha256(records),
    )
    runner.validate_reused_tree_protocol(
        reuse_tree_dir=reuse_tree,
        current_config=current_config,
        source_summary={},
        current_records_sha256=runner.file_sha256(records),
    )
    with pytest.raises(RuntimeError, match="already level/max-node filtered"):
        runner.validate_reused_tree_protocol(
            reuse_tree_dir=reuse_tree,
            current_config=current_config,
            source_summary={"skill_export": {"min_level": 1, "max_level": 1, "max_node_count": None}},
            current_records_sha256=runner.file_sha256(records),
        )


def test_materialize_reused_tree_rejects_overlapping_source_and_output_dir(tmp_path):
    runner = _load_experiment_runner_module()
    reuse_tree = tmp_path / "same_tree"
    reuse_tree.mkdir()
    (reuse_tree / "hierarchy_state.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap the output tree dir"):
        runner.materialize_reused_tree_nodebank(
            reuse_tree_dir=reuse_tree,
            tree_dir=reuse_tree,
            args=SimpleNamespace(),
            current_config={},
            fingerprint={},
            marker_dir=tmp_path / "markers",
            log_path=tmp_path / "log.txt",
        )

    nested_reuse_tree = tmp_path / "output_tree" / "nested_source"
    nested_reuse_tree.mkdir(parents=True)
    (nested_reuse_tree / "hierarchy_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap the output tree dir"):
        runner.materialize_reused_tree_nodebank(
            reuse_tree_dir=nested_reuse_tree,
            tree_dir=tmp_path / "output_tree",
            args=SimpleNamespace(),
            current_config={},
            fingerprint={},
            marker_dir=tmp_path / "markers",
            log_path=tmp_path / "log.txt",
        )


def test_spreadsheet_heldout_rollout_fingerprint_includes_skillbank_code():
    runner = _load_experiment_runner_module()
    repo = Path(__file__).resolve().parents[1]
    source_fp = runner.stage_source_fingerprints(repo)

    rollout_fp = runner.benchmark_source_fingerprints(source_fp, benchmark="spreadsheetbench", stage="rollout")

    assert "dynamix_trace2skill" in rollout_fp


def test_officeqa_loader_and_query_exclude_gold_answer(tmp_path):
    from dynamix_benchmarks.officeqa import load_officeqa_items, officeqa_skillbank_query

    split_dir = tmp_path / "officeqa" / "splits" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "items.json").write_text(
        json.dumps([
            {
                "uid": "q1",
                "question": "What was the 1984 balance?",
                "answer": "$123 million",
                "category": "numeric",
                "source_docs": ["https://example.test/doc?page=7"],
                "source_files": ["tb1984.pdf"],
            }
        ]),
        encoding="utf-8",
    )

    item = load_officeqa_items(tmp_path / "officeqa" / "splits", split="train")[0]
    query = officeqa_skillbank_query(item)

    assert item.uid == "q1"
    assert "What was the 1984 balance?" in query
    assert "Task type: numeric" in query
    assert "$123 million" not in query
    assert "page=7" not in query
    assert "tb1984" not in query


def test_officeqa_skillopt_evaluator_requires_answer_tag_but_parser_has_fallback():
    from dynamix_benchmarks.officeqa import evaluate_officeqa_prediction, extract_final_answer

    result = evaluate_officeqa_prediction(
        prediction_text="<think>hidden</think>\n<answer>$42 million</answer>",
        gold_answer="42",
    )

    assert result.hard == 1
    assert result.predicted_answer == "$42 million"
    assert result.evaluator == "skillopt_normalized_em_f1"
    assert extract_final_answer("<think>hidden</think>\n42") == "42"
    assert extract_final_answer("Thinking Process: hidden\n</think>\n<answer>42</answer>") == "42"

    missing_tag = evaluate_officeqa_prediction(
        prediction_text="<think>hidden</think>\n42",
        gold_answer="42",
    )
    assert missing_tag.hard == 0
    assert missing_tag.fail_reason == "missing_answer_tag"

    wrong_tag = evaluate_officeqa_prediction(
        prediction_text="<FINAL_ANSWER>42</FINAL_ANSWER>",
        gold_answer="42",
    )
    assert wrong_tag.hard == 0
    assert wrong_tag.fail_reason == "missing_answer_tag"


def test_officeqa_official_reward_wrapper_strips_thinking_when_explicit(tmp_path):
    from dynamix_benchmarks.officeqa import evaluate_officeqa_prediction

    reward = tmp_path / "reward.py"
    reward.write_text(
        "def score_answer(ground_truth, predicted, tolerance=0.0):\n"
        "    assert '<think>' not in predicted\n"
        "    return 1.0 if ground_truth == '42' and predicted == '<FINAL_ANSWER>42</FINAL_ANSWER>' else 0.0\n",
        encoding="utf-8",
    )

    result = evaluate_officeqa_prediction(
        prediction_text="<think>hidden</think>\n<FINAL_ANSWER>42</FINAL_ANSWER>",
        gold_answer="42",
        reward_path=reward,
        evaluator="official_reward",
    )

    assert result.hard == 1
    assert result.predicted_answer == "42"
    assert result.evaluator.startswith("official_reward:")

    skillopt_tag = evaluate_officeqa_prediction(
        prediction_text="<think>hidden</think>\n<answer>42</answer>",
        gold_answer="42",
        reward_path=reward,
        evaluator="official_reward",
    )
    assert skillopt_tag.hard == 1
    assert skillopt_tag.predicted_answer == "42"

    malformed = evaluate_officeqa_prediction(
        prediction_text="<think>hidden</think>\n42",
        gold_answer="42",
        reward_path=reward,
        evaluator="official_reward",
    )
    assert malformed.hard == 0


def test_officeqa_result_converts_to_raw_trajectory_record():
    from dynamix_benchmarks.officeqa import record_from_officeqa_result

    record = record_from_officeqa_result({
        "id": "q1",
        "trajectory_id": "officeqa_q1",
        "item": {
            "uid": "q1",
            "question": "Find the deficit.",
            "category": "numeric",
            "source_files": ["tb.txt"],
            "source_docs": ["https://example.test/doc?page=1"],
        },
        "final_response": "<answer>5</answer>",
        "success": True,
        "score": 1.0,
        "steps": [
            {
                "step_id": 1,
                "raw_model_output": "Action...",
                "action": "{\"name\":\"grep\",\"arguments\":{}}",
                "observation": "evidence",
                "tool_name": "grep",
                "action_valid": True,
            }
        ],
        "predicted_answer": "5",
        "evaluator": "skillopt_normalized_em_f1",
    })

    assert record.task_id == "q1"
    assert record.instruction == "Find the deficit."
    assert record.instruction_type == "numeric"
    assert record.answer_position == ""
    assert record.success is True
    assert record.extra["benchmark"] == "officeqa"


def test_officeqa_fallback_evaluator_requires_explicit_debug_flag():
    from dynamix_benchmarks.officeqa import evaluate_officeqa_prediction

    with pytest.raises(ValueError):
        evaluate_officeqa_prediction(prediction_text="<answer>42</answer>", gold_answer="42", evaluator="fallback")

    result = evaluate_officeqa_prediction(
        prediction_text="<answer>42</answer>",
        gold_answer="42",
        evaluator="fallback",
        allow_fallback=True,
    )
    assert result.hard == 1
    assert result.evaluator == "fallback_numeric_or_text"


def test_officeqa_official_reward_exception_fails_formal_eval(tmp_path):
    from dynamix_benchmarks.officeqa import OfficeQAOfficialRewardError, evaluate_officeqa_prediction

    reward = tmp_path / "reward.py"
    reward.write_text(
        "def score_answer(ground_truth, predicted, tolerance=0.0):\n"
        "    raise RuntimeError('broken evaluator')\n",
        encoding="utf-8",
    )

    with pytest.raises(OfficeQAOfficialRewardError):
        evaluate_officeqa_prediction(
            prediction_text="<FINAL_ANSWER>42</FINAL_ANSWER>",
            gold_answer="42",
            reward_path=reward,
            evaluator="official_reward",
        )

    debug = evaluate_officeqa_prediction(
        prediction_text="<FINAL_ANSWER>42</FINAL_ANSWER>",
        gold_answer="42",
        reward_path=reward,
        evaluator="official_reward",
        allow_fallback=True,
        raise_official_errors=False,
    )
    assert debug.hard == 0
    assert debug.fail_reason.startswith("official_reward_error:")


def test_officeqa_agent_execution_exceptions_are_infra_errors():
    from dynamix_benchmarks.officeqa import is_infrastructure_agent_error

    assert is_infrastructure_agent_error("Exception during execution: connection reset")
    assert not is_infrastructure_agent_error("Max turns exceeded")
    assert not is_infrastructure_agent_error("")


def test_officeqa_doc_tools_do_not_follow_symlink_outside_root(tmp_path):
    from dynamix_benchmarks.officeqa import OfficeQADocTools

    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("SECRET_VALUE", encoding="utf-8")
    os.symlink(outside, docs / "secret.txt")

    tools = OfficeQADocTools([docs], source_files=["secret.txt"])
    assert "path is required" in tools.grep("SECRET")
    assert "secret.txt" not in tools.glob("*")


def test_officeqa_doc_tools_expose_openai_function_schemas_and_execute(tmp_path):
    from dynamix_benchmarks.officeqa import OfficeQADocTools

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "tb.txt").write_text("1: header\n2: Answer is 42\n", encoding="utf-8")

    tools = OfficeQADocTools([docs])
    schemas = tools.to_openai_tool_schemas()

    assert [schema["function"]["name"] for schema in schemas] == ["glob", "read", "grep"]
    assert "path is required" in tools.execute_tool("grep", {"pattern": "answer"})
    assert "Answer is 42" in tools.execute_tool("grep", {"pattern": "answer", "path": "tb.txt"})
    assert "1: 1: header" in tools.execute_tool("read", {"path": "tb.txt", "start": 1, "limit": 1})


def test_officeqa_run_item_uses_function_calling_tools(tmp_path, monkeypatch):
    from dynamix_benchmarks import officeqa

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "tb.txt").write_text("Answer is 42\n", encoding="utf-8")

    class FakeToolCall:
        id = "call_1"
        function = SimpleNamespace(name="grep", arguments=json.dumps({"pattern": "answer", "path": "tb.txt"}))

        def model_dump(self, mode="json"):
            return {
                "id": self.id,
                "type": "function",
                "function": {
                    "name": self.function.name,
                    "arguments": self.function.arguments,
                },
            }

    class FakeOpenAIClient:
        instances: list["FakeOpenAIClient"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0
            self.seen_messages: list[list[dict]] = []
            self.seen_tools: list[list[dict]] = []
            FakeOpenAIClient.instances.append(self)

        def chat_with_tools(self, *, messages, tools, tool_choice="auto"):
            self.calls += 1
            self.seen_messages.append(list(messages))
            self.seen_tools.append(list(tools))
            assert tool_choice == "auto"
            if self.calls == 1:
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="<think>hidden</think>\n", tool_calls=[FakeToolCall()]),
                    )]
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="<answer>42</answer>", tool_calls=None),
                )]
            )

    monkeypatch.setattr(officeqa, "OpenAIClient", FakeOpenAIClient)
    item = officeqa.OfficeQAItem(
        id="q1",
        uid="q1",
        question="What is the answer?",
        ground_truth="42",
        category="numeric",
        source_files=["tb.txt"],
    )

    row = officeqa.run_officeqa_item(
        item,
        docs_dirs=[docs],
        model="mock-model",
        openai_base_url="mock://officeqa",
        openai_api_key="EMPTY",
        generation_config={"temperature": 0.0},
        max_turns=3,
        llm_timeout_seconds=30.0,
        llm_retry_wait_seconds=(0.0,),
        reward_path=None,
        reward_tolerance=0.0,
        evaluator="skillopt",
        allow_fallback_evaluator=False,
        output_dir=tmp_path / "out",
        use_oracle_context=False,
    )

    client = FakeOpenAIClient.instances[0]
    assert client.calls == 2
    assert row["success"] is True
    assert row["predicted_answer"] == "42"
    assert row["service_metadata"]["agent_mode"] == "function_calling"
    assert row["steps"][0]["tool_name"] == "grep"
    assert "<think>hidden</think>" in row["steps"][0]["raw_model_output"]
    assert "Answer is 42" in row["steps"][0]["observation"]
    assert client.seen_tools[0][0]["function"]["name"] == "glob"
    assert any(message["role"] == "tool" for message in client.seen_messages[1])
    assert "<think>" not in client.seen_messages[1][2]["content"]


def test_officeqa_run_item_requires_answer_tag_for_final_response(tmp_path, monkeypatch):
    from dynamix_benchmarks import officeqa

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "tb.txt").write_text("Answer is 42\n", encoding="utf-8")

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            pass

        def chat_with_tools(self, *, messages, tools, tool_choice="auto"):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="I should search first.", tool_calls=None),
                )]
            )

    monkeypatch.setattr(officeqa, "OpenAIClient", FakeOpenAIClient)
    item = officeqa.OfficeQAItem(
        id="q1",
        uid="q1",
        question="What is the answer?",
        ground_truth="42",
        category="numeric",
        source_files=["tb.txt"],
    )

    row = officeqa.run_officeqa_item(
        item,
        docs_dirs=[docs],
        model="mock-model",
        openai_base_url="mock://officeqa",
        openai_api_key="EMPTY",
        generation_config={"temperature": 0.0},
        max_turns=3,
        llm_timeout_seconds=30.0,
        llm_retry_wait_seconds=(0.0,),
        reward_path=None,
        reward_tolerance=0.0,
        evaluator="skillopt",
        allow_fallback_evaluator=False,
        output_dir=tmp_path / "out",
        use_oracle_context=False,
    )

    assert row["agent_success"] is False
    assert row["success"] is False
    assert row["predicted_answer"] == ""
    assert "neither produced a tool request nor a final answer" in row["fail_reason"]


def test_officeqa_oracle_context_does_not_follow_symlink_outside_root(tmp_path):
    from dynamix_benchmarks.officeqa import OfficeQAItem, build_oracle_context

    parsed_root = tmp_path / "treasury_bulletins_parsed"
    transformed = parsed_root / "transformed"
    jsons = parsed_root / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps({"document": {"elements": [{"bbox": [{"page_id": 7}], "content": "ORACLE_LEAK_MARKER_789"}]}}),
        encoding="utf-8",
    )
    os.symlink(outside, jsons / "tb1984.json")

    item = OfficeQAItem(
        id="q1",
        uid="q1",
        question="Q?",
        ground_truth="A",
        source_files=["tb1984.pdf"],
        source_docs=["https://example.test/doc?page=7"],
    )

    assert "ORACLE_LEAK_MARKER_789" not in build_oracle_context(item, [transformed])


def test_officeqa_oracle_context_preserves_multi_page_table_context(tmp_path):
    from dynamix_benchmarks.officeqa import OfficeQAItem, build_oracle_context

    parsed_root = tmp_path / "treasury_bulletins_parsed"
    transformed = parsed_root / "transformed"
    jsons = parsed_root / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir(parents=True)
    (jsons / "tb1984.json").write_text(
        json.dumps({
            "document": {
                "elements": [
                    {"bbox": [{"page_id": 7}], "content": "<table><tr><th>Year</th><th>Value</th></tr><tr><td>1984</td><td>42</td></tr></table>"},
                    {"bbox": [{"page_id": 8}], "content": "Second page evidence"},
                ]
            }
        }),
        encoding="utf-8",
    )
    item = OfficeQAItem(
        id="q1",
        uid="q1",
        question="Q?",
        ground_truth="A",
        source_files=["tb1984.pdf"],
        source_docs=["https://example.test/doc?page=7", "https://example.test/doc?page=8"],
    )

    context = build_oracle_context(item, [transformed])
    assert "tb1984.pdf page 7" in context
    assert "tb1984.pdf page 8" in context
    assert "| Year | Value |" in context
    assert "Second page evidence" in context


def test_officeqa_public_item_artifact_excludes_gold_answer():
    from dynamix_benchmarks.officeqa import normalize_item

    item = normalize_item({"uid": "q1", "question": "Q?", "answer": "SECRET_ANSWER", "source_docs": ["doc"]})
    public = item.to_public_dict()

    assert "SECRET_ANSWER" not in json.dumps(public)
    assert "ground_truth" not in public
    assert "answer" not in public
    assert "answers" not in public


def test_benchmark_adapter_orders_officeqa_records(tmp_path):
    from dynamix_benchmarks.adapters import BenchmarkSlice, OfficeQAAdapter

    split_root = tmp_path / "splits" / "train"
    split_root.mkdir(parents=True)
    (split_root / "items.json").write_text(
        json.dumps([
            {"uid": "a", "question": "A?", "answer": "1"},
            {"uid": "b", "question": "B?", "answer": "2"},
        ]),
        encoding="utf-8",
    )
    source_records = tmp_path / "records.json"
    source_records.write_text(
        json.dumps([
            {"trajectory_id": "officeqa_b", "task_id": "b", "trial_index": 0, "instruction": "B?"},
            {"trajectory_id": "officeqa_a", "task_id": "a", "trial_index": 0, "instruction": "A?"},
        ]),
        encoding="utf-8",
    )

    adapter = OfficeQAAdapter()
    manifest = adapter.write_ordered_records(
        source_records=source_records,
        output_path=tmp_path / "ordered.json",
        manifest_path=tmp_path / "manifest.json",
        data_path=tmp_path / "splits",
        train_slice=BenchmarkSlice(split="train", start=0, end=2),
    )
    ordered = json.loads((tmp_path / "ordered.json").read_text(encoding="utf-8"))

    assert [row["task_id"] for row in ordered] == ["a", "b"]
    assert manifest["record_count"] == 2


def test_benchmark_adapter_officeqa_builds_rollout_eval_extract_commands(tmp_path):
    from dynamix_benchmarks.adapters import BenchmarkSlice, EvalCommandSpec, ExtractCommandSpec, OfficeQAAdapter, RolloutCommandSpec

    adapter = OfficeQAAdapter()
    data_slice = BenchmarkSlice(split="train", start=0, end=2)
    rollout = adapter.run_rollout(RolloutCommandSpec(
        python_executable="/bin/python",
        data_path=tmp_path / "splits",
        data_slice=data_slice,
        output_dir=tmp_path / "out",
        results_file=tmp_path / "results.json",
        log_dir=tmp_path / "logs",
        generation_config=tmp_path / "gen.json",
        model="model",
        openai_base_url="http://127.0.0.1:18002/v1",
        rollout_llm_client="openai",
        rollout_temperature=0.0,
        llm_timeout_seconds=1200.0,
        llm_retry_wait_seconds=[5.0],
        officeqa_docs_dirs=[tmp_path / "docs"],
        officeqa_allow_fallback_evaluator=True,
    ))
    eval_cmd = adapter.evaluate_results(EvalCommandSpec(
        python_executable="/bin/python",
        data_path=tmp_path / "splits",
        data_slice=data_slice,
        output_dir=tmp_path / "out",
        results_file=tmp_path / "results.json",
        eval_file=tmp_path / "eval.json",
        officeqa_allow_fallback_evaluator=True,
    ))
    extract_cmd = adapter.extract_records(ExtractCommandSpec(
        python_executable="/bin/python",
        log_dir=tmp_path / "logs",
        eval_file=tmp_path / "eval.json",
        records_file=tmp_path / "records.json",
    ))

    assert "scripts/run_officeqa_benchmark.py" in rollout
    assert "--evaluator" in rollout
    assert rollout[rollout.index("--evaluator") + 1] == "skillopt"
    assert "--max-completion-tokens" in rollout
    assert rollout[rollout.index("--max-completion-tokens") + 1] == "16384"
    assert "--allow-fallback-evaluator" in rollout
    assert "scripts/evaluate_officeqa_results.py" in eval_cmd
    assert "--evaluator" in eval_cmd
    assert eval_cmd[eval_cmd.index("--evaluator") + 1] == "skillopt"
    assert "scripts/extract_officeqa_records.py" in extract_cmd


def test_benchmark_adapter_omits_none_end_idx_for_officeqa(tmp_path):
    from dynamix_benchmarks.adapters import BenchmarkSlice, OfficeQAAdapter, RolloutCommandSpec

    cmd = OfficeQAAdapter().run_rollout(RolloutCommandSpec(
        python_executable="/bin/python",
        data_path=tmp_path / "splits",
        data_slice=BenchmarkSlice(split="train", start=0, end=None),
        output_dir=tmp_path / "out",
        results_file=tmp_path / "results.json",
        log_dir=tmp_path / "logs",
        generation_config=tmp_path / "gen.json",
        model="model",
        openai_base_url="http://127.0.0.1:18002/v1",
        rollout_llm_client="openai",
        rollout_temperature=0.0,
        llm_timeout_seconds=1200.0,
        officeqa_docs_dirs=[tmp_path / "docs"],
    ))

    assert "--end_idx" not in cmd
    assert cmd[cmd.index("--llm_retry_wait_seconds") + 1] == "5.0,10.0,30.0"


def test_officeqa_default_ranges_use_named_split_sizes(tmp_path):
    runner = _load_experiment_runner_module()
    from dynamix_benchmarks.adapters import OfficeQAAdapter

    for split, count in (("train", 5), ("test", 3)):
        split_dir = tmp_path / "officeqa" / split
        split_dir.mkdir(parents=True)
        (split_dir / "items.json").write_text(
            json.dumps([
                {"uid": f"{split}_{idx}", "question": f"Q{idx}?", "answer": str(idx)}
                for idx in range(count)
            ]),
            encoding="utf-8",
        )

    args = SimpleNamespace(
        benchmark="officeqa",
        officeqa_train_split="train",
        officeqa_heldout_split="test",
        train_start=0,
        train_end=200,
        heldout_start=200,
        heldout_end=400,
        tree_scenario="dynamic_update",
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
        rollout_temperature=0.0,
        generation_temperature=0.6,
        thinking="true",
    )
    runner.apply_officeqa_default_ranges(args, OfficeQAAdapter(), tmp_path / "officeqa", [])

    assert (args.train_start, args.train_end) == (0, 5)
    assert (args.heldout_start, args.heldout_end) == (0, 3)
    assert args.dynamic_initial_count == 3
    assert args.dynamic_arrival_count == 2
    assert args.rollout_temperature == 0.7
    assert args.generation_temperature == 0.7
    assert args.thinking == "true"


def test_officeqa_skillopt_qwen_temperature_defaults_do_not_override_explicit_cli(tmp_path):
    runner = _load_experiment_runner_module()
    from dynamix_benchmarks.adapters import OfficeQAAdapter

    for split in ("train", "test"):
        split_dir = tmp_path / "officeqa" / split
        split_dir.mkdir(parents=True)
        (split_dir / "items.json").write_text(
            json.dumps([{"uid": f"{split}_0", "question": "Q?", "answer": "A"}]),
            encoding="utf-8",
        )

    args = SimpleNamespace(
        benchmark="officeqa",
        officeqa_train_split="train",
        officeqa_heldout_split="test",
        train_start=0,
        train_end=1,
        heldout_start=0,
        heldout_end=1,
        tree_scenario="static_build",
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
        rollout_temperature=0.0,
        generation_temperature=0.0,
        thinking="true",
    )
    runner.apply_officeqa_default_ranges(
        args,
        OfficeQAAdapter(),
        tmp_path / "officeqa",
        ["--rollout-temperature", "0.0", "--generation-temperature=0.0", "--thinking", "true"],
    )

    assert args.rollout_temperature == 0.0
    assert args.generation_temperature == 0.0
    assert args.thinking == "true"


def test_officeqa_default_ranges_reject_invalid_explicit_bounds(tmp_path):
    runner = _load_experiment_runner_module()
    from dynamix_benchmarks.adapters import OfficeQAAdapter

    for split in ("train", "test"):
        split_dir = tmp_path / "officeqa" / split
        split_dir.mkdir(parents=True)
        (split_dir / "items.json").write_text(
            json.dumps([{"uid": f"{split}_0", "question": "Q?", "answer": "A"}]),
            encoding="utf-8",
        )

    args = SimpleNamespace(
        benchmark="officeqa",
        officeqa_train_split="train",
        officeqa_heldout_split="test",
        train_start=0,
        train_end=1,
        heldout_start=0,
        heldout_end=2,
        tree_scenario="static_build",
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
        rollout_temperature=0.0,
        generation_temperature=0.6,
        thinking="true",
    )

    with pytest.raises(ValueError, match="Invalid OfficeQA heldout range"):
        runner.apply_officeqa_default_ranges(
            args,
            OfficeQAAdapter(),
            tmp_path / "officeqa",
            ["--heldout-end", "2"],
        )


def test_officeqa_runner_generation_config_follows_skillopt_qwen_defaults():
    runner = _load_officeqa_runner_module()

    config = runner.apply_skillopt_qwen_generation_defaults({}, max_completion_tokens=16384)

    assert config["max_tokens"] == 16384
    assert config["temperature"] == 0.7
    assert config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

    explicit = runner.apply_skillopt_qwen_generation_defaults(
        {"temperature": 0.0, "max_tokens": 128, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        max_completion_tokens=16384,
    )
    assert explicit["temperature"] == 0.0
    assert explicit["max_tokens"] == 128
    assert explicit["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

    partial_extra = runner.apply_skillopt_qwen_generation_defaults(
        {"extra_body": {"top_k": 20}},
        max_completion_tokens=16384,
    )
    assert partial_extra["extra_body"]["top_k"] == 20
    assert partial_extra["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_react_stop_settings_do_not_override_generation_temperature():
    from react_agent.models import ModelSettings

    request_config = {"temperature": 0.0, "max_tokens": 128}
    request_config.update(ModelSettings(stop=["Observation:"]).to_dict())

    assert request_config["temperature"] == 0.0
    assert request_config["max_tokens"] == 128
    assert request_config["stop"] == ["Observation:"]
    assert ModelSettings(temperature=0.7).to_dict()["temperature"] == 0.7


def test_react_usage_payload_preserves_non_empty_usage():
    from react_agent.models import _response_usage_payload

    assert _response_usage_payload({"usage": {"prompt_tokens": 1, "total_tokens": 2}}) == {
        "prompt_tokens": 1,
        "total_tokens": 2,
    }
    assert _response_usage_payload(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))) == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
    }


def test_openai_client_chat_with_tools_uses_disk_cache(tmp_path, monkeypatch):
    from react_agent.models import OpenAIClient

    class FakeToolCall:
        id = "call_1"
        function = SimpleNamespace(name="grep", arguments=json.dumps({"pattern": "answer", "path": "tb.txt"}))

        def model_dump(self, mode="json"):
            return {
                "id": self.id,
                "type": "function",
                "function": {
                    "name": self.function.name,
                    "arguments": self.function.arguments,
                },
            }

    client = OpenAIClient(
        model="mock-model",
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        cache_path=str(tmp_path / "officeqa_tools.diskcache"),
        use_cache=True,
        generation_config={"temperature": 0.0},
        retry_times=(0,),
        timeout=1.0,
    )
    calls = {"count": 0}

    def fake_send(messages, config):
        calls["count"] += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[FakeToolCall()]),
            )],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(client, "_send_request_with_retry", fake_send)
    messages = [{"role": "user", "content": "search"}]
    tools = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}]

    first = client.chat_with_tools(messages=messages, tools=tools)
    second = client.chat_with_tools(messages=messages, tools=tools)

    assert calls["count"] == 1
    assert first.choices[0].message.tool_calls[0].function.name == "grep"
    assert second.choices[0].message.tool_calls[0].function.name == "grep"
    assert second.choices[0].message.tool_calls[0].model_dump()["function"]["arguments"]
    assert second.usage == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def test_officeqa_tree_config_disables_dataset_json_ordering(tmp_path):
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        benchmark="officeqa",
        data_path=str(tmp_path / "officeqa"),
        train_start=0,
        train_end=5,
    )

    payload = runner.tree_dataset_order_payload(args)

    assert payload["dataset_path"] == str((tmp_path / "officeqa").resolve())
    assert payload["train_start"] == 0
    assert payload["train_end"] == 5
    assert payload["enforce_dataset_order"] is False


def test_officeqa_item_results_manifest_rotates_stale_jsonl(tmp_path):
    runner = _load_officeqa_runner_module()
    jsonl = tmp_path / "results.jsonl"
    manifest = tmp_path / "results.jsonl.manifest.json"
    selection_log = tmp_path / "selection.jsonl"
    jsonl.write_text(json.dumps({"id": "old"}) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({"fingerprint": {"old": True}}), encoding="utf-8")
    selection_log.write_text(json.dumps({"id": "old", "selected": []}) + "\n", encoding="utf-8")

    runner.rotate_stale_item_results(jsonl, manifest, reason="fingerprint mismatch", selection_log_path=selection_log)

    assert not jsonl.exists()
    assert not selection_log.exists()
    stale_jsonls = list(tmp_path.glob("results.jsonl.stale_*"))
    stale_selection_logs = list(tmp_path.glob("selection.jsonl.stale_*"))
    assert len(stale_jsonls) == 1
    assert len(stale_selection_logs) == 1
    assert json.loads(stale_jsonls[0].read_text(encoding="utf-8"))["id"] == "old"
    marker = json.loads(manifest.read_text(encoding="utf-8"))
    assert marker["reason"] == "fingerprint mismatch"


def test_officeqa_selection_log_resume_requires_matching_task_ids(tmp_path):
    runner = _load_officeqa_runner_module()
    selection_log = tmp_path / "selection.jsonl"
    selection_log.write_text(
        json.dumps({"task_id": "unrelated", "selected": []}) + "\n"
        + json.dumps({"task_id": "another", "selected": []}) + "\n",
        encoding="utf-8",
    )

    assert not runner.selection_log_covers(selection_log, {"expected"})

    with selection_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_id": "expected", "selected": []}) + "\n")

    assert runner.selection_log_covers(selection_log, {"expected"})


def test_officeqa_item_results_fingerprint_covers_docs_tree_files(tmp_path):
    runner = _load_officeqa_runner_module()
    docs = tmp_path / "docs"
    docs.mkdir()
    child = docs / "doc.txt"
    child.write_text("old", encoding="utf-8")

    before = runner.path_tree_identity(docs)
    child.write_text("newer content", encoding="utf-8")
    after = runner.path_tree_identity(docs)

    assert before["file_count"] == 1
    assert after["file_count"] == 1
    assert before["tree_content_sha256"] != after["tree_content_sha256"]


def test_officeqa_item_results_fingerprint_covers_skillbank_content_and_flags(tmp_path):
    runner = _load_officeqa_runner_module()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.txt").write_text("doc", encoding="utf-8")
    skillbank = tmp_path / "nodebank"
    skillbank.mkdir()
    manifest = skillbank / "node_bank_manifest.json"
    manifest.write_text("old", encoding="utf-8")

    def make_args(**overrides):
        payload = dict(
            split_dir=str(tmp_path / "splits"),
            split="test",
            start_idx=0,
            end_idx=1,
            model="model",
            openai_base_url="http://127.0.0.1:18002/v1",
            openai_api_key="EMPTY",
            max_turns=24,
            evaluator="skillopt",
            reward_tolerance=0.0,
            allow_fallback_evaluator=False,
            continue_on_infra_error=False,
            skillbank_root=str(skillbank),
            skillbank_top_k=4,
            use_oracle_context=True,
        )
        payload.update(overrides)
        return SimpleNamespace(**payload)

    before = runner.build_item_results_fingerprint(make_args(), [str(docs)], {"max_tokens": 16}, None)
    manifest.write_text("new", encoding="utf-8")
    after = runner.build_item_results_fingerprint(make_args(), [str(docs)], {"max_tokens": 16}, None)
    fallback = runner.build_item_results_fingerprint(make_args(allow_fallback_evaluator=True), [str(docs)], {"max_tokens": 16}, None)
    infra = runner.build_item_results_fingerprint(make_args(continue_on_infra_error=True), [str(docs)], {"max_tokens": 16}, None)

    assert before["skillbank_root"]["tree_content_sha256"] != after["skillbank_root"]["tree_content_sha256"]
    assert after != fallback
    assert after != infra


def test_no_proxy_helper_adds_internal_hosts_without_wildcard():
    runner = _load_experiment_runner_module()
    env = {"NO_PROXY": "localhost"}

    runner.append_no_proxy_hosts(env, ["http://asmiatbrqksz.10.27.127.9.nip.io/v1", "https://api.openai.com/v1"])

    assert "asmiatbrqksz.10.27.127.9.nip.io" in env["NO_PROXY"].split(",")
    assert "*" not in env["NO_PROXY"].split(",")

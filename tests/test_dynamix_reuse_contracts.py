from __future__ import annotations

import json
import asyncio
import hashlib
import importlib.util
import multiprocessing as mp
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dynamix_trace2skill.clients import (
    EmbeddingClient,
    EmbeddingConfig,
    GenerationClient,
    GenerationConfig,
    _SqliteEmbeddingCache,
    api_key_fingerprint,
    embedding_cache_namespace,
    embedding_protocol_payload,
    embedding_vector_sha256,
    normalized_embedding_vector_sha256,
    ordered_embedding_vectors,
    validate_embedding_cache_manifest,
    write_embedding_cache_manifest,
)
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig
from dynamix_trace2skill.log_parser import parse_trace2skill_logs, _result_fields
from dynamix_trace2skill.pipeline import (
    DynaMixRunConfig,
    _write_runtime_artifacts,
    default_hierarchy_config,
)
from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep
from dynamix_trace2skill.trace_views import render_embedding_trace
from dynamix_core.data_structures import ExperienceCardPatch, ExperienceCommunity, ExperienceHierarchyState, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY
from dynamix_core.update import ExperienceHierarchyDynamicUpdater


def test_react_model_settings_do_not_override_client_generation_defaults():
    from react_agent.models import ModelSettings

    assert "temperature" not in ModelSettings().to_dict()
    assert ModelSettings(temperature=0.0).to_dict()["temperature"] == 0.0


def test_react_openai_client_disables_hidden_sdk_retries(monkeypatch):
    from react_agent.models import OPENAI_SDK_MAX_RETRIES, OpenAIClient

    constructor_kwargs = {}
    async_constructor_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructor_kwargs.update(kwargs)

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            async_constructor_kwargs.update(kwargs)

    fake_openai = SimpleNamespace(
        OpenAI=FakeOpenAI,
        AsyncOpenAI=FakeAsyncOpenAI,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = OpenAIClient(
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        use_cache=False,
        timeout=600.0,
    )

    assert constructor_kwargs["max_retries"] == OPENAI_SDK_MAX_RETRIES
    assert (
        client._async_client_kwargs["max_retries"]
        == OPENAI_SDK_MAX_RETRIES
    )
    client._get_async_client()
    assert (
        async_constructor_kwargs["max_retries"]
        == OPENAI_SDK_MAX_RETRIES
    )


def test_react_openai_client_does_not_retry_runtime_timeout(monkeypatch):
    import react_agent.models as models

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise FakeAPITimeoutError("Request timed out.")

    class FakeAPITimeoutError(Exception):
        pass

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeOpenAI),
    )
    monkeypatch.setattr(
        models.time,
        "sleep",
        lambda _: pytest.fail("runtime timeout must not be retried"),
    )
    client = models.OpenAIClient(
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        use_cache=False,
    )

    with pytest.raises(
        models.RequestRuntimeTimeout,
        match="runtime_invalid_timeout",
    ):
        client._send_request_with_retry(
            [{"role": "user", "content": "test"}],
            {},
        )
    assert len(calls) == 1


def test_react_openai_client_keeps_retrying_non_timeout_errors(monkeypatch):
    import react_agent.models as models

    calls = []
    expected = object()

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary connection failure")
            return expected

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeOpenAI),
    )
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    client = models.OpenAIClient(
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        use_cache=False,
        retry_times=(0,),
    )

    assert client._send_request_with_retry([], {}) is expected
    assert len(calls) == 2


def test_react_async_openai_client_does_not_retry_runtime_timeout(
    monkeypatch,
):
    import react_agent.models as models

    calls = []

    class FakeAPITimeoutError(Exception):
        pass

    class FakeSyncOpenAI:
        def __init__(self, **kwargs):
            pass

    class FakeAsyncCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            raise FakeAPITimeoutError("Request timed out.")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeAsyncCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=FakeSyncOpenAI,
            AsyncOpenAI=FakeAsyncOpenAI,
        ),
    )
    client = models.OpenAIClient(
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        use_cache=False,
    )

    with pytest.raises(
        models.RequestRuntimeTimeout,
        match="runtime_invalid_timeout",
    ):
        asyncio.run(client._send_request_with_retry_async([], {}))
    assert len(calls) == 1


def test_react_response_cache_key_tracks_sdk_retry_policy(monkeypatch):
    import react_agent.models as models

    base_protocol = {
        "client": "openai",
        "base_url": "http://example.invalid/v1",
        "api_key": "EMPTY",
        "generation_config": {"temperature": 0.0},
        "retry_times": (5, 10, 30),
        "timeout": 600.0,
    }
    old_key = models._make_cache_key(
        "model",
        [{"role": "user", "content": "test"}],
        protocol=base_protocol,
    )
    new_key = models._make_cache_key(
        "model",
        [{"role": "user", "content": "test"}],
        protocol={
            **base_protocol,
            "sdk_max_retries": models.OPENAI_SDK_MAX_RETRIES,
        },
    )
    assert old_key != new_key

    captured_protocol = {}

    def capture_cache_key(model, messages, *, protocol=None):
        captured_protocol.update(protocol or {})
        return ("captured",)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    class FakeCache(dict):
        def close(self):
            pass

    monkeypatch.setattr(models, "_make_cache_key", capture_cache_key)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeOpenAI),
    )
    client = models.OpenAIClient(
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        use_cache=False,
    )
    client._cache = FakeCache(
        {("captured",): ("cached response", "")}
    )

    assert client.chat([models.Message(role="user", content="test")]) == (
        "cached response"
    )
    assert (
        captured_protocol["sdk_max_retries"]
        == models.OPENAI_SDK_MAX_RETRIES
    )


def test_spreadsheet_runner_temperature_is_written_to_client_config():
    import run_spreadsheetbench

    args = SimpleNamespace(
        generation_config=json.dumps({"temperature": 0.7}),
        temperature=0.2,
        run_seed=None,
    )
    assert run_spreadsheetbench._build_generation_config(args)[
        "temperature"
    ] == pytest.approx(0.2)


def test_spreadsheet_runner_can_disable_response_cache(monkeypatch):
    import run_spreadsheetbench

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        run_spreadsheetbench,
        "OpenAIClient",
        FakeClient,
    )
    args = SimpleNamespace(
        generation_config=None,
        temperature=0.0,
        run_seed=None,
        disable_response_cache=True,
        llm_client="openai",
        model="model",
        llm_retry_wait_seconds=(5.0,),
        llm_timeout_seconds=600.0,
    )

    run_spreadsheetbench._build_client(args)

    assert captured["use_cache"] is False


def test_spreadsheet_parallel_runner_uses_shared_task_queue_and_jsonl(
    monkeypatch,
    tmp_path,
):
    import run_spreadsheetbench
    from spreadsheet_agent.runner import InstanceResult

    task_two_started = threading.Event()
    task_zero_saw_task_two = []
    processed = []
    instances = [
        SimpleNamespace(id=str(index), instruction=f"task-{index}")
        for index in range(4)
    ]

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def load_data(self):
            return instances

        def run_instance(self, instance):
            processed.append(instance.id)
            if instance.id == "0":
                task_zero_saw_task_two.append(task_two_started.wait(1.0))
            elif instance.id == "2":
                task_two_started.set()
            return InstanceResult(
                id=instance.id,
                instruction=instance.instruction,
                success=True,
            )

    monkeypatch.setattr(
        run_spreadsheetbench,
        "SpreadsheetBenchRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        run_spreadsheetbench,
        "create_agent",
        lambda args: SimpleNamespace(name="fake-agent"),
    )
    monkeypatch.setattr(
        run_spreadsheetbench,
        "instance_has_outputs",
        lambda instance, output_dir, data_path: False,
    )
    args = SimpleNamespace(
        workers=2,
        data_path=str(tmp_path),
        output_dir=str(tmp_path / "outputs"),
        working_dir=str(tmp_path / "work"),
        start_idx=0,
        end_idx=4,
        shuffle_seed=None,
        sample=None,
        instance_ids=None,
        missing_only=False,
        results_file=str(tmp_path / "results.json"),
        agent="cli_only",
        skills_dir=str(tmp_path / "skills"),
        model="fake-model",
        llm_client="openai",
        generation_config=None,
        temperature=0.0,
        max_turns=30,
        llm_timeout_seconds=600.0,
        llm_retry_wait_seconds=(5.0, 10.0, 30.0),
        disable_response_cache=True,
        repeat=1,
        run_seed=None,
    )

    run_spreadsheetbench.run_parallel(args)

    assert task_zero_saw_task_two == [True]
    payload = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    assert [row["id"] for row in payload["results"]] == ["0", "1", "2", "3"]
    jsonl_path = Path(args.results_file).with_suffix(".jsonl")
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["id"] for row in rows} == {"0", "1", "2", "3"}

    Path(args.results_file).unlink()
    args.missing_only = True
    run_spreadsheetbench.run_parallel(args)
    assert len(processed) == 4
    restored = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
    assert [row["id"] for row in restored["results"]] == ["0", "1", "2", "3"]
    assert restored["response_cache_enabled"] is False
    assert restored["parallel_workers"] == 2
    assert restored["resume_only"] is True


def test_spreadsheet_result_ledger_repairs_only_partial_tail(tmp_path):
    import run_spreadsheetbench

    results_jsonl = tmp_path / "results.jsonl"
    valid_row = {
        "id": "task-1",
        "instruction": "first",
        "success": False,
        "error": "runtime_invalid_timeout",
        "test_cases": [],
    }
    results_jsonl.write_bytes(
        (
            json.dumps(valid_row, sort_keys=True)
            + "\n"
            + '{"id":"partial'
        ).encode("utf-8")
    )
    identity = {
        "format": "spreadsheetbench_result_ledger_v1",
        "fingerprint": "fingerprint",
        "protocol": {"task_ids": ["task-1", "task-2"]},
    }
    run_spreadsheetbench._write_json_atomic(
        run_spreadsheetbench._result_ledger_manifest_path(
            results_jsonl
        ),
        identity,
    )

    rows = run_spreadsheetbench._prepare_result_ledger(
        results_jsonl,
        identity=identity,
        valid_ids={"task-1", "task-2"},
        resume=True,
    )

    assert rows == {"task-1": valid_row}
    assert results_jsonl.read_bytes().endswith(b"\n")


def test_spreadsheet_result_ledger_rejects_identity_and_scope_drift(
    tmp_path,
):
    import run_spreadsheetbench

    results_jsonl = tmp_path / "results.jsonl"
    identity = {
        "format": "spreadsheetbench_result_ledger_v1",
        "fingerprint": "expected",
        "protocol": {"task_ids": ["task-1"]},
    }
    run_spreadsheetbench._prepare_result_ledger(
        results_jsonl,
        identity=identity,
        valid_ids={"task-1"},
        resume=False,
    )
    with pytest.raises(RuntimeError, match="identity"):
        run_spreadsheetbench._prepare_result_ledger(
            results_jsonl,
            identity={**identity, "fingerprint": "different"},
            valid_ids={"task-1"},
            resume=True,
        )

    out_of_scope_row = {
        "id": "task-2",
        "instruction": "stale",
        "success": True,
        "error": "",
        "test_cases": [],
    }
    results_jsonl.write_text(
        json.dumps(out_of_scope_row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="out-of-scope"):
        run_spreadsheetbench._prepare_result_ledger(
            results_jsonl,
            identity=identity,
            valid_ids={"task-1"},
            resume=True,
        )


def test_spreadsheet_result_ledger_publishes_manifest_after_empty_ledger(
    tmp_path,
    monkeypatch,
):
    import run_spreadsheetbench

    results_jsonl = tmp_path / "results.jsonl"
    results_jsonl.write_text('{"id":"old"}\n', encoding="utf-8")
    identity = {
        "format": "spreadsheetbench_result_ledger_v1",
        "fingerprint": "new",
        "protocol": {"task_ids": ["task-1"]},
    }
    original_write = run_spreadsheetbench._write_json_atomic

    def assert_empty_then_publish(path, payload):
        assert results_jsonl.read_bytes() == b""
        original_write(path, payload)

    monkeypatch.setattr(
        run_spreadsheetbench,
        "_write_json_atomic",
        assert_empty_then_publish,
    )

    rows = run_spreadsheetbench._prepare_result_ledger(
        results_jsonl,
        identity=identity,
        valid_ids={"task-1"},
        resume=False,
    )

    assert rows == {}


def test_spreadsheet_result_ledger_rejects_concurrent_process(tmp_path):
    import run_spreadsheetbench

    results_jsonl = tmp_path / "results.jsonl"
    first_lock = run_spreadsheetbench._acquire_result_ledger_lock(
        results_jsonl
    )
    try:
        with pytest.raises(RuntimeError, match="already running"):
            run_spreadsheetbench._acquire_result_ledger_lock(results_jsonl)
    finally:
        first_lock.close()


def test_spreadsheet_runner_preserves_agent_timeout_failure(
    tmp_path,
    monkeypatch,
):
    import spreadsheet_agent.runner as runner_module

    spreadsheet_dir = tmp_path / "data"
    spreadsheet_dir.mkdir()
    (spreadsheet_dir / "1_task_input.xlsx").write_bytes(b"input")
    monkeypatch.setattr(
        runner_module,
        "get_spreadsheet_content",
        lambda _: "preview",
    )

    class TimeoutAgent:
        def run(self, context):
            Path(context.output_file).write_bytes(b"partial")
            return {
                "success": False,
                "answer": "",
                "turns": 4,
                "error": "runtime_invalid_timeout: Request timed out",
            }

    runner = runner_module.SpreadsheetBenchRunner(
        agent=TimeoutAgent(),
        data_path=str(tmp_path),
        output_dir=str(tmp_path / "outputs"),
        working_dir=str(tmp_path / "work"),
    )
    instance = SimpleNamespace(
        id="task",
        spreadsheet_path="sheet",
        instruction="edit workbook",
        instruction_type="Cell-Level Manipulation",
        answer_position="A1",
    )
    stale_output = (
        tmp_path
        / "outputs"
        / "sheet"
        / "1_task_output.xlsx"
    )
    stale_output.parent.mkdir(parents=True)
    stale_output.write_bytes(b"stale-unrecorded-output")

    result = runner._run_test_case(
        instance,
        str(spreadsheet_dir),
        "1_task_input.xlsx",
    )

    assert result.success is False
    assert result.turns == 4
    assert result.error == "runtime_invalid_timeout: Request timed out"
    assert not stale_output.exists()


def test_tokenizer_initialization_is_serialized(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import dynamix_trace2skill.tokenization as tokenization

    tokenization._get_tokenizer_cached.cache_clear()
    constructor_calls = []

    class FakeTokenizer:
        def __init__(self, model_or_path):
            constructor_calls.append(model_or_path)
            time.sleep(0.05)

    monkeypatch.setattr(tokenization, "HuggingFaceTokenizer", FakeTokenizer)
    with ThreadPoolExecutor(max_workers=2) as executor:
        tokenizers = list(
            executor.map(
                lambda _: tokenization.get_tokenizer(
                    "test-tokenizer",
                    allow_regex_fallback=False,
                ),
                range(2),
            )
        )

    assert constructor_calls == ["test-tokenizer"]
    assert tokenizers[0] is tokenizers[1]
    tokenization._get_tokenizer_cached.cache_clear()


def test_embedding_response_is_strictly_reordered_by_index():
    rows = [
        SimpleNamespace(index=1, embedding=[2.0]),
        SimpleNamespace(index=0, embedding=[1.0]),
    ]
    assert ordered_embedding_vectors(rows, expected_count=2) == [
        [1.0],
        [2.0],
    ]
    with pytest.raises(RuntimeError, match="count"):
        ordered_embedding_vectors(rows[:1], expected_count=2)
    with pytest.raises(RuntimeError, match="unique range"):
        ordered_embedding_vectors(
            [
                SimpleNamespace(index=0, embedding=[1.0]),
                SimpleNamespace(index=0, embedding=[2.0]),
            ],
            expected_count=2,
        )


def test_embedding_cache_manifest_detects_required_row_mutation(
    tmp_path,
):
    import sqlite3

    cache_path = tmp_path / "vectors.sqlite"
    config = EmbeddingConfig(
        base_url="mock://deterministic",
        cache_path=str(cache_path),
        tokenizer_required=False,
    )
    client = EmbeddingClient(config)
    vector = asyncio.run(
        client.embed_texts(
            ["required text"],
            cache_namespace="manifest_test",
        )
    )[0]
    client.close()
    namespace = embedding_cache_namespace(
        config,
        "manifest_test",
        model_name=config.model,
    )
    manifest_path = tmp_path / "manifest.json"
    payload = write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=manifest_path,
        requirements=[
            {
                "namespace": namespace,
                "text": "required text",
                "vector": vector,
                "purpose": "test",
                "item_id": "item-1",
            }
        ],
    )
    assert payload["entry_count"] == 1
    validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=manifest_path,
    )

    connection = sqlite3.connect(cache_path)
    connection.execute(
        "INSERT OR REPLACE INTO embeddings(namespace,key,vector) "
        "VALUES(?,?,?)",
        (
            namespace,
            hashlib.sha256(b"unrelated text").hexdigest(),
            json.dumps([9.0]),
        ),
    )
    connection.commit()
    connection.close()
    validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=manifest_path,
    )

    connection = sqlite3.connect(cache_path)
    connection.execute(
        "UPDATE embeddings SET vector=? WHERE namespace=? AND key=?",
        (
            json.dumps([7.0]),
            namespace,
            hashlib.sha256(b"required text").hexdigest(),
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="changed after certification"):
        validate_embedding_cache_manifest(
            cache_path=cache_path,
            manifest_path=manifest_path,
        )


def test_embedding_cache_manifest_binds_actual_normalized_artifact_vector(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "vectors.sqlite"
    raw = np.linspace(0.001, 1.0, 4096, dtype=float)
    artifact = (
        raw.reshape(1, -1)
        / np.linalg.norm(raw.reshape(1, -1), axis=1, keepdims=True)
    )[0]
    cache = _SqliteEmbeddingCache(cache_path)
    cache.set("node-index", "node text", raw.tolist())
    cache.close()

    payload = write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=tmp_path / "manifest.json",
        requirements=[
            {
                "namespace": "node-index",
                "text": "node text",
                "normalized_vector": artifact.tolist(),
            }
        ],
    )

    entry = payload["entries"][0]
    assert entry["artifact_normalized_vector_sha256"] == (
        embedding_vector_sha256(artifact.tolist())
    )
    assert entry["artifact_cache_max_abs_error"] is not None
    assert entry["artifact_cache_max_abs_error"] <= (
        64.0 * np.finfo(float).eps
    )
    validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=tmp_path / "manifest.json",
    )


def test_embedding_cache_manifest_uses_exact_documented_absolute_tolerance(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "vectors.sqlite"
    cache = _SqliteEmbeddingCache(cache_path)
    cache.set("node-index", "node text", [1.0, 0.0])
    cache.close()
    epsilon = np.finfo(float).eps

    write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=tmp_path / "accepted.json",
        requirements=[
            {
                "namespace": "node-index",
                "text": "node text",
                "normalized_vector": [1.0 + 32.0 * epsilon, 0.0],
            }
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="normalized embedding cache vector differs",
    ):
        write_embedding_cache_manifest(
            cache_path=cache_path,
            output_path=tmp_path / "rejected.json",
            requirements=[
                {
                    "namespace": "node-index",
                    "text": "node text",
                    "normalized_vector": [1.0 + 100.0 * epsilon, 0.0],
                }
            ],
        )


def _load_experiment_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamix_trace2skill_experiment.py"
    spec = importlib.util.spec_from_file_location("run_dynamix_trace2skill_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_query_cache_audit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_cdost_query_vector_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_cdost_query_vector_cache_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_spreadsheet_evaluator_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "evaluate_with_official.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_with_official_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_official_evaluator_backend_fails_closed_when_unavailable(
    monkeypatch,
):
    evaluator = _load_spreadsheet_evaluator_module()
    monkeypatch.setattr(
        evaluator,
        "official_compare_workbooks",
        None,
    )
    with pytest.raises(
        RuntimeError,
        match="evaluation_official is unavailable",
    ):
        evaluator._resolve_comparator("official")


def test_evaluator_runtime_identity_binds_comparator_and_libreoffice(
    monkeypatch,
):
    evaluator = _load_spreadsheet_evaluator_module()
    monkeypatch.setattr(
        evaluator,
        "_preflight_libreoffice",
        lambda timeout_seconds=30: (
            "/usr/bin/soffice",
            "LibreOffice test-version",
        ),
    )
    identity = evaluator.evaluation_runtime_identity("local")
    assert identity["workbook_comparator"]["resolved_backend"] == "local"
    assert identity["workbook_comparator"]["source_sha256"]
    assert identity["libreoffice"] == {
        "executable": str(Path("/usr/bin/soffice").resolve()),
        "version": "LibreOffice test-version",
    }


def test_cdost_control_manifest_rejects_protocol_drift(tmp_path):
    runner = _load_experiment_runner_module()
    source = tmp_path / "source.json"
    current = tmp_path / "current.json"
    runner.write_cdost_control_manifest(
        source,
        {"format": "cdost_control_contract_v2", "top_k": 10},
    )
    runner.write_cdost_control_manifest(
        current,
        {"format": "cdost_control_contract_v2", "top_k": 9},
    )
    with pytest.raises(ValueError, match="control contract differs"):
        runner.validate_matching_cdost_control_manifest(
            current_manifest=current,
            source_manifest=source,
        )


def test_cdost_control_contract_covers_method_and_redacts_secrets(tmp_path):
    runner = _load_experiment_runner_module()
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps([{"trajectory_id": str(index)} for index in range(200)]),
        encoding="utf-8",
    )
    generation_config_path = tmp_path / "generation.json"
    generation_config_path.write_text(
        json.dumps({"temperature": 0.0}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        train_start=0,
        train_end=200,
        heldout_start=200,
        heldout_end=400,
        model="model-under-test",
        openai_base_url="https://generation.example/v1",
        openai_api_key="rollout-secret",
        thinking="true",
        max_turns=30,
        workers=16,
        rollout_client_timeout_seconds=1200.0,
        rollout_client_retry_wait_seconds=[5.0, 10.0],
        rollout_llm_client="openai",
        rollout_num_random_seeds=1,
        rollout_seeds="",
        rollout_instance_ids="",
        rollout_missing_only=False,
        rollout_repeat=1,
        rollout_shuffle_seed="",
        rollout_sample=0,
        skillbank_top_k=10,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        embedding_max_model_len=32000,
        embedding_max_input_tokens=32000,
        embedding_batch_size=8,
        embedding_tokenizer="/models/embedding",
        tree_policy="certified_dual_view_otd",
        tree_scenario="static_build",
        dynamic_initial_count=120,
        dynamic_arrival_count=80,
        dynamic_update_batch_size=8,
        dynamic_shuffle_seed=-1,
        dynamic_snapshot_include_embeddings=True,
        dynamic_resume_from_snapshots=False,
    )
    config = {
        "hierarchy": {
            "otd": {
                "dual_view_lambda": 0.5,
                "tie_epsilon": 0.0,
                "atom_cache_path": "/run/local/atoms.json",
            }
        },
        "generation": {
            "base_url": args.openai_base_url,
            "model": args.model,
            "api_key": "analyst-secret",
            "temperature": 0.0,
            "debug_dir": "/run/local/debug",
        },
        "embedding": {
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "api_key": "embedding-secret",
            "cache_path": "/run/local/cache.sqlite",
            "max_model_len": 32000,
        },
        "analyst": {
            "max_output_tokens": 4096,
            "prompt_token_report_path": "/run/local/prompt.json",
        },
    }
    source_keys = (
        "runner",
        "run_spreadsheetbench",
        "evaluate_with_official",
        "spreadsheetbench_support",
        "spreadsheet_agent",
        "react_agent",
        "skillbank",
        "antichain_retrieval",
        "dynamix_core",
        "dynamix_trace2skill",
    )
    contract = runner.cdost_control_contract(
        args=args,
        config=config,
        records_path=records_path,
        dataset_fingerprint={"sha256": "dataset-sha"},
        generation_config_path=generation_config_path,
        evaluator_identity={"libreoffice": {"version": "test"}},
        source_fingerprints={
            key: {"exists": True, "sha256": f"{key}-sha"}
            for key in source_keys
        },
    )

    encoded = json.dumps(contract, sort_keys=True)
    for secret in (
        "rollout-secret",
        "analyst-secret",
        "embedding-secret",
    ):
        assert secret not in encoded
    assert contract["train_split"] == [0, 200]
    assert contract["heldout_split"] == [200, 400]
    assert contract["paired_dynamic_schedule"] == {
        "initial_count": 120,
        "arrival_count": 80,
        "insertion_count": 80,
        "arrival_order": "dataset",
        "shuffle_seed": None,
        "snapshot_interval": 8,
        "snapshot_include_embeddings": True,
        "resume_from_snapshots": False,
    }
    assert contract["tree"]["otd"]["dual_view_lambda"] == 0.5
    assert "atom_cache_path" not in contract["tree"]["otd"]
    assert contract["retrieval"]["top_k"] == 10
    assert contract["rollout"]["generation_config"]["temperature"] == 0.0
    from react_agent.models import OPENAI_SDK_MAX_RETRIES

    assert (
        contract["rollout"]["sdk_max_retries"]
        == OPENAI_SDK_MAX_RETRIES
    )
    assert contract["evaluator"]["libreoffice"]["version"] == "test"
    assert set(contract["source"]) == set(source_keys)
    assert contract["tree"]["generation"]["api_key_fingerprint"].startswith(
        "sha256:"
    )
    assert contract["tree"]["embedding"]["api_key_fingerprint"].startswith(
        "sha256:"
    )
    assert contract["rollout"]["openai_api_key_fingerprint"].startswith(
        "sha256:"
    )


def test_cdost_control_manifest_rejects_dynamic_schedule_drift(tmp_path):
    runner = _load_experiment_runner_module()
    source = tmp_path / "source.json"
    current = tmp_path / "current.json"
    base = {
        "format": "cdost_control_contract_v2",
        "paired_dynamic_schedule": {
            "initial_count": 120,
            "arrival_count": 80,
            "insertion_count": 80,
            "arrival_order": "dataset",
            "shuffle_seed": None,
            "snapshot_interval": 8,
            "snapshot_include_embeddings": True,
            "resume_from_snapshots": False,
        },
    }
    changed = json.loads(json.dumps(base))
    changed["paired_dynamic_schedule"]["snapshot_interval"] = 10
    runner.write_cdost_control_manifest(source, base)
    runner.write_cdost_control_manifest(current, changed)

    with pytest.raises(ValueError, match="control contract differs"):
        runner.validate_matching_cdost_control_manifest(
            current_manifest=current,
            source_manifest=source,
        )


def test_heldout_query_cache_manifest_detects_mutation_and_reference_drift(
    tmp_path,
):
    import sqlite3

    module = _load_query_cache_audit_module()
    cache_path = tmp_path / "vectors.sqlite"
    config = EmbeddingConfig(
        base_url="mock://query-audit",
        cache_path=str(cache_path),
        tokenizer_required=False,
    )
    client = EmbeddingClient(config)
    query = "calculate the requested total\n\nTask type: arithmetic"
    vector = asyncio.run(
        client.embed_texts(
            [query],
            cache_namespace="query_protocol",
        )
    )[0]
    client.close()
    namespace = embedding_cache_namespace(
        config,
        "query_protocol",
        model_name=config.model,
    )
    selection_log = tmp_path / "selection.jsonl"
    selection_log.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "query": query,
                "query_embedding_audit": {
                    "namespace_sha256": namespace,
                    "text_sha256": [
                        hashlib.sha256(query.encode()).hexdigest()
                    ],
                    "vector_sha256": [
                        hashlib.sha256(
                            json.dumps(
                                vector,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                    ],
                    "scoring_vector_sha256": [
                        normalized_embedding_vector_sha256(vector)
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    static_manifest = tmp_path / "static-query-manifest.json"
    module.audit_query_vector_cache(
        selection_log=selection_log,
        cache_path=cache_path,
        output_path=static_manifest,
    )
    dynamic_manifest = tmp_path / "dynamic-query-manifest.json"
    module.audit_query_vector_cache(
        selection_log=selection_log,
        cache_path=cache_path,
        output_path=dynamic_manifest,
        reference_manifest=static_manifest,
    )

    connection = sqlite3.connect(cache_path)
    connection.execute(
        "UPDATE embeddings SET vector=? WHERE namespace=? AND key=?",
        (
            json.dumps([123.0]),
            namespace,
            hashlib.sha256(query.encode()).hexdigest(),
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="changed after certification"):
        validate_embedding_cache_manifest(
            cache_path=cache_path,
            manifest_path=static_manifest,
        )


def test_query_vector_manifest_is_bound_to_source_audit_marker(tmp_path):
    runner = _load_experiment_runner_module()
    manifest = tmp_path / "heldout_query_embedding_cache_manifest.json"
    manifest.write_text('{"logical_sha256":"stable"}', encoding="utf-8")
    marker = tmp_path / "06b_query_vector_audit.done"
    marker.write_text(
        json.dumps(
            {
                "output_identities": {
                    str(manifest): runner.path_fingerprint(manifest),
                }
            }
        ),
        encoding="utf-8",
    )

    runner.validate_source_build_output(
        marker_path=marker,
        output_path=manifest,
    )
    manifest.write_text('{"logical_sha256":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="completed stage marker"):
        runner.validate_source_build_output(
            marker_path=marker,
            output_path=manifest,
        )


def test_local_evaluator_backend_records_source_identity():
    evaluator = _load_spreadsheet_evaluator_module()
    comparator, identity = evaluator._resolve_comparator("local")

    assert comparator is evaluator.local_compare_workbooks
    assert identity["requested_backend"] == "local"
    assert identity["resolved_backend"] == "local"
    assert identity["module"] == "spreadsheetbench_support"
    assert identity["source_path"].endswith(
        "spreadsheetbench_support.py"
    )
    assert len(identity["source_sha256"]) == 64


def test_experiment_runner_builds_explicit_evaluator_command(tmp_path):
    runner = _load_experiment_runner_module()
    command = runner.build_evaluation_command(
        python_executable="/env/bin/python",
        data_path="/data/spreadsheetbench",
        output_dir=tmp_path / "outputs",
        recalc_dir=tmp_path / "recalc",
        start_idx=200,
        end_idx=400,
        results_file=tmp_path / "eval.json",
        evaluator_backend="local",
    )

    backend_index = command.index("--evaluator-backend")
    assert command[backend_index + 1] == "local"
    assert command[0:2] == [
        "/env/bin/python",
        "evaluate_with_official.py",
    ]


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


def test_runtime_manifest_records_resolved_service_identity(
    tmp_path,
    monkeypatch,
):
    records_path = tmp_path / "records.json"
    records_path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("TEST_GENERATION_KEY", "resolved-generation-secret")
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "resolved-embedding-secret")
    generation = GenerationConfig(
        base_url="http://generation.invalid/v1",
        model="test-generation",
        api_key="fallback-generation-secret",
        api_key_env_var="TEST_GENERATION_KEY",
    )
    embedding = EmbeddingConfig(
        base_url="http://embedding.invalid/v1",
        model="test-embedding",
        api_key="fallback-embedding-secret",
        api_key_env_var="TEST_EMBEDDING_KEY",
        max_input_tokens=28000,
        tokenizer_required=False,
    )
    config = DynaMixRunConfig(
        output_dir=str(tmp_path / "run"),
        records_path=str(records_path),
        generation=generation,
        embedding=embedding,
    )
    output_dir = Path(config.output_dir)
    _write_runtime_artifacts(config, output_dir)

    runtime_config = json.loads(
        (output_dir / "analysis" / "runtime_config.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(runtime_config, sort_keys=True)
    assert "fallback-generation-secret" not in serialized
    assert "fallback-embedding-secret" not in serialized
    assert "resolved-generation-secret" not in serialized
    assert "resolved-embedding-secret" not in serialized

    manifest = json.loads(
        (output_dir / "analysis" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["service_identity"]["generation"]["api_key"] == (
        api_key_fingerprint(generation.resolved_api_key)
    )
    assert manifest["service_identity"]["embedding"] == (
        embedding_protocol_payload(embedding)
    )
    manifest_serialized = json.dumps(manifest, sort_keys=True)
    assert "resolved-generation-secret" not in manifest_serialized
    assert "resolved-embedding-secret" not in manifest_serialized


def test_embedding_cache_namespace_preserves_legacy_protocol_identity():
    config = EmbeddingConfig(
        base_url="http://embedding.invalid/v1",
        model="test-embedding",
        api_key="EMPTY",
        max_model_len=32000,
        max_input_tokens=28000,
        truncate_long_texts=True,
        tokenizer_model="test-tokenizer",
        tokenizer_required=True,
        truncation_strategy="head",
        batch_size=8,
        max_concurrency=8,
        deterministic_dim=384,
    )
    legacy_payload = {
        "base_url": config.base_url,
        "model": config.model,
        "api_key": api_key_fingerprint(config.resolved_api_key),
        "max_model_len": config.max_model_len,
        "max_input_tokens": config.effective_max_input_tokens,
        "truncate_long_texts": config.truncate_long_texts,
        "tokenizer_model": config.tokenizer_model,
        "tokenizer_required": config.tokenizer_required,
        "truncation_strategy": config.truncation_strategy,
        "deterministic_dim": config.deterministic_dim,
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    client = EmbeddingClient(config)
    assert client._cache_namespace(
        config.model,
        model_name=config.model,
    ) == f"{config.model}::protocol::{expected_digest}"
    changed_execution = EmbeddingClient(
        EmbeddingConfig(
            **(asdict(config) | {"batch_size": 4, "max_concurrency": 4})
        )
    )
    assert changed_execution._cache_namespace(
        config.model,
        model_name=config.model,
    ) == client._cache_namespace(config.model, model_name=config.model)


def test_embedding_cache_default_replaces_legacy_rows(tmp_path):
    cache = _SqliteEmbeddingCache(tmp_path / "legacy.sqlite")
    assert cache.set("namespace", "text", [1.0]) == [1.0]
    assert cache.set("namespace", "text", [2.0]) == [2.0]
    assert cache.get("namespace", "text") == [2.0]
    cache.close()


def test_embedding_cache_first_write_policy_freezes_cdost_rows(tmp_path):
    cache = _SqliteEmbeddingCache(
        tmp_path / "cdost.sqlite",
        write_policy="first_write_wins",
    )
    assert cache.set("namespace", "text", [1.0]) == [1.0]
    assert cache.set("namespace", "text", [2.0]) == [1.0]
    assert cache.get("namespace", "text") == [1.0]
    cache.close()


def test_legacy_skillbank_protocol_records_actual_single_vector_behavior():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_policy="projected_gmm_bic",
        tree_scenario="static_build",
        skillbank_top_k=10,
        embedding_base_url="http://embedding.invalid/v1",
        embedding_model="Qwen3-Embedding-8B",
        embedding_max_model_len=32000,
        embedding_max_input_tokens=32000,
        embedding_batch_size=8,
        embedding_tokenizer="/models/Qwen3-Embedding-8B",
        chunked_embedding_enabled=True,
        chunked_embedding_chunk_tokens=28000,
        chunked_embedding_overlap_tokens=1000,
        chunked_embedding_pooling="mean",
        chunked_embedding_add_special_tokens=False,
        chunked_embedding_normalize_after_pooling=False,
        chunked_embedding_fail_if_chunk_exceeds_model_limit=True,
    )
    protocol = runner.skillbank_retrieval_protocol(
        args,
        cache_path=Path("/tmp/index.json"),
        vector_cache_path=Path("/tmp/vectors.sqlite"),
        selection_log=Path("/tmp/selection.jsonl"),
    )

    assert protocol["embedding_input_policy"] == "legacy_single_vector"
    assert protocol["chunked_embedding_active"] is False
    assert protocol["chunked_embedding_configured"] is True
    assert protocol["chunk_tokens"] == 28000
    assert protocol["chunk_overlap_tokens"] == 1000
    assert protocol["chunk_pooling"] == "mean"
    assert protocol["vector_cache_policy"] == "disabled"
    assert protocol["require_cache_match"] is False
    assert protocol["require_vector_cache_match"] is False


def test_embedding_client_reads_prefilled_legacy_namespace(
    tmp_path,
    monkeypatch,
):
    config = EmbeddingConfig(
        base_url="mock://deterministic",
        model="test-embedding",
        api_key="EMPTY",
        tokenizer_required=False,
        batch_size=4,
        max_concurrency=4,
        cache_path=str(tmp_path / "embedding.sqlite"),
    )
    client = EmbeddingClient(config)
    namespace = client._cache_namespace(
        config.model,
        model_name=config.model,
    )
    assert client._cache is not None
    client._cache.set(namespace, "hello", [0.25, 0.75])

    async def fail_uncached(*args, **kwargs):
        raise AssertionError("legacy cache entry must prevent backend embedding")

    monkeypatch.setattr(client, "_embed_uncached", fail_uncached)
    assert asyncio.run(client.embed_texts(["hello"])) == [[0.25, 0.75]]
    client.close()


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


def test_chat_json_repairs_malformed_json_once():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic"))
    responses = iter(('{"cards": [', '{"cards": []}'))
    calls = []

    async def fake_chat_text(messages, **kwargs):
        calls.append(list(messages))
        return next(responses)

    client.chat_text = fake_chat_text
    result = asyncio.run(
        client.chat_json(
            [{"role": "user", "content": "return cards"}],
            schema_name="MinimalClusterExperienceCards",
            guided_json={
                "type": "object",
                "properties": {"cards": {"type": "array"}},
                "required": ["cards"],
            },
            retries=1,
        )
    )

    assert result == {"cards": []}
    assert len(calls) == 2
    assert "Previous parse error" in calls[1][-1]["content"]
    assert "fresh, compact JSON object" in calls[1][-1]["content"]


def test_chat_json_stops_after_one_malformed_json_repair():
    client = GenerationClient(GenerationConfig(base_url="mock://deterministic"))
    calls = 0

    async def fake_chat_text(messages, **kwargs):
        nonlocal calls
        calls += 1
        return '{"cards": ['

    client.chat_text = fake_chat_text
    with pytest.raises(ValueError, match="failed to parse JSON"):
        asyncio.run(
            client.chat_json(
                [{"role": "user", "content": "return cards"}],
                schema_name="MinimalClusterExperienceCards",
                guided_json={
                    "type": "object",
                    "properties": {"cards": {"type": "array"}},
                    "required": ["cards"],
                },
                retries=1,
            )
        )
    assert calls == 2


def test_generation_client_disables_openai_sdk_retries(monkeypatch):
    constructor_kwargs = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructor_kwargs.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **request: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="ok")
                            )
                        ]
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = GenerationClient(
        GenerationConfig(
            base_url="http://example.invalid/v1",
            retry_wait_seconds=(),
        )
    )

    assert (
        client._chat_text_sync(
            [{"role": "user", "content": "hello"}],
            None,
            None,
            None,
            {},
            None,
        )
        == "ok"
    )
    assert constructor_kwargs["max_retries"] == 0


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
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    audit = output_dir / "events.jsonl"
    output.write_text("{}", encoding="utf-8")
    audit.write_text('{"event": 1}\n', encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "fingerprint": {"scenario": "dynamic_update"},
                "output_identities": {
                    str(output): runner.path_fingerprint(output),
                    str(output_dir): runner.path_fingerprint(output_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = [output, output_dir]
    assert runner.stage_done(marker, outputs, fingerprint={"scenario": "dynamic_update"})
    assert not runner.stage_done(marker, outputs, fingerprint={"scenario": "static_build"})
    audit.write_text('{"event": 2}\n', encoding="utf-8")
    assert not runner.stage_done(
        marker,
        outputs,
        fingerprint={"scenario": "dynamic_update"},
    )
    audit.write_text('{"event": 1}\n', encoding="utf-8")
    output.write_text('{"tampered": true}', encoding="utf-8")
    assert not runner.stage_done(
        marker,
        outputs,
        fingerprint={"scenario": "dynamic_update"},
    )
    marker.write_text(json.dumps({"stage": "04_build_tree"}), encoding="utf-8")
    assert not runner.stage_done(marker, outputs, fingerprint={"scenario": "dynamic_update"})


def test_experiment_runner_forced_rerun_removes_stage_owned_directories(
    tmp_path,
    monkeypatch,
):
    runner = _load_experiment_runner_module()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    old_outputs = tmp_path / "outputs"
    old_logs = tmp_path / "logs"
    old_outputs.mkdir()
    old_logs.mkdir()
    (old_outputs / "stale.xlsx").write_bytes(b"stale")
    (old_logs / "stale.md").write_text("stale", encoding="utf-8")
    result = tmp_path / "results.json"
    result.write_text("stale", encoding="utf-8")

    def fail_after_cleanup(*args, **kwargs):
        assert not old_outputs.exists()
        assert not old_logs.exists()
        assert not result.exists()
        raise RuntimeError("forced failure")

    monkeypatch.setattr(runner, "run", fail_after_cleanup)
    with pytest.raises(RuntimeError, match="forced failure"):
        runner.run_stage(
            "01_train_collect",
            ["false"],
            cwd=tmp_path,
            env={},
            log_path=tmp_path / "collect.log",
            marker_dir=marker_dir,
            outputs=[result, old_outputs, old_logs],
            resume=False,
            clear_outputs_before_run=[result, old_outputs, old_logs],
        )

    assert not old_outputs.exists()
    assert not old_logs.exists()
    assert not result.exists()


def test_experiment_runner_failed_forced_rerun_invalidates_old_done(
    tmp_path,
    monkeypatch,
):
    runner = _load_experiment_runner_module()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    marker = marker_dir / "04_build_tree.done"
    output = tmp_path / "summary.json"
    output.write_text("old", encoding="utf-8")
    fingerprint = {"scenario": "static_build"}
    marker.write_text(
        json.dumps({"fingerprint": fingerprint}),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(runner, "run", fail_run)
    with pytest.raises(RuntimeError, match="forced failure"):
        runner.run_stage(
            "04_build_tree",
            ["false"],
            cwd=tmp_path,
            env={},
            log_path=tmp_path / "build.log",
            marker_dir=marker_dir,
            outputs=[output],
            resume=False,
            fingerprint=fingerprint,
        )

    assert not marker.exists()
    assert (marker_dir / "04_build_tree.failed.json").is_file()
    assert not runner.stage_done(
        marker,
        [output],
        fingerprint=fingerprint,
    )


def test_experiment_runner_resumes_matching_partial_stage_without_cleanup(
    tmp_path,
    monkeypatch,
):
    runner = _load_experiment_runner_module()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    fingerprint = {"stage": "heldout", "version": 3}
    output = tmp_path / "results.jsonl"
    output.write_text('{"id":"task-1"}\n', encoding="utf-8")
    (marker_dir / "06_heldout_collect.running").write_text(
        json.dumps({"fingerprint": fingerprint}),
        encoding="utf-8",
    )

    def finish_run(*args, **kwargs):
        assert output.is_file()
        assert kwargs["append_log"] is True

    monkeypatch.setattr(runner, "run", finish_run)
    runner.run_stage(
        "06_heldout_collect",
        ["resume"],
        cwd=tmp_path,
        env={},
        log_path=tmp_path / "heldout.log",
        marker_dir=marker_dir,
        outputs=[output],
        resume=True,
        fingerprint=fingerprint,
        clear_outputs_before_run=[output],
        preserve_partial_outputs_on_resume=True,
    )

    assert output.read_text(encoding="utf-8") == '{"id":"task-1"}\n'
    assert (marker_dir / "06_heldout_collect.done").is_file()


def test_experiment_runner_split_manifest_is_disjoint_and_bounded(tmp_path):
    runner = _load_experiment_runner_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset.json").write_text(
        json.dumps(
            [
                {"id": f"task-{index}", "instruction_type": "test"}
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="non-overlapping"):
        runner.write_split_manifest(
            data_dir,
            run_dir,
            train_start=0,
            train_end=3,
            heldout_start=2,
            heldout_end=4,
        )
    with pytest.raises(ValueError, match="dataset size"):
        runner.write_split_manifest(
            data_dir,
            run_dir,
            train_start=0,
            train_end=2,
            heldout_start=2,
            heldout_end=5,
        )

    manifest = runner.write_split_manifest(
        data_dir,
        run_dir,
        train_start=0,
        train_end=2,
        heldout_start=2,
        heldout_end=4,
    )
    assert manifest["dataset_size"] == 4
    assert manifest["train_expected_count"] == 2
    assert manifest["heldout_expected_count"] == 2
    assert {row["id"] for row in manifest["train"]}.isdisjoint(
        row["id"] for row in manifest["heldout"]
    )


def test_experiment_runner_validates_heldout_eval_identity_and_denominator(
    tmp_path,
):
    runner = _load_experiment_runner_module()
    split_manifest = {
        "heldout": [{"id": "task-a"}, {"id": "task-b"}],
        "heldout_expected_count": 2,
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "summary": {"total_instances": 1},
                "results": [{"id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="denominator"):
        runner.validate_heldout_eval_coverage(eval_path, split_manifest)

    eval_path.write_text(
        json.dumps(
            {
                "summary": {"total_instances": 2},
                "results": [{"id": "task-b"}, {"id": "task-a"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity/order"):
        runner.validate_heldout_eval_coverage(eval_path, split_manifest)

    eval_path.write_text(
        json.dumps(
            {
                "summary": {"total_instances": 2},
                "results": [{"id": "task-a"}, {"id": "task-b"}],
            }
        ),
        encoding="utf-8",
    )
    runner.validate_heldout_eval_coverage(eval_path, split_manifest)


def test_experiment_runner_rejects_post_run_evaluator_identity_drift(
    tmp_path,
):
    runner = _load_experiment_runner_module()
    expected = {
        "workbook_comparator": {
            "resolved_backend": "local",
            "source_sha256": "comparator-sha",
        },
        "libreoffice": {
            "executable": "/usr/bin/soffice",
            "version": "LibreOffice test",
        },
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps({"summary": expected}),
        encoding="utf-8",
    )
    runner.validate_evaluation_runtime_identity(eval_path, expected)

    drifted = json.loads(eval_path.read_text(encoding="utf-8"))
    drifted["summary"]["libreoffice"]["version"] = "LibreOffice changed"
    eval_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        runner.validate_evaluation_runtime_identity(eval_path, expected)


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


def test_experiment_stage_report_marks_missing_optional_artifacts_unavailable():
    runner = _load_experiment_runner_module()

    rendered = runner.render_experiment_stage_report_md({
        "run_dir": "/tmp/run",
        "created_at": "2026-07-28T00:00:00Z",
        "stages": [],
        "prompt_token_stats": {
            "path": "/tmp/run/dynamix_tree/analysis/cluster_prompt_token_report.json",
            "exists": False,
        },
        "chunked_embedding_stats": {
            "path": "/tmp/run/dynamix_tree/analysis/chunked_embedding_report.json",
            "exists": False,
        },
        "runtime_dead_corner_findings": [],
    })

    assert rendered.count("Artifact status: unavailable; this report was not produced for the run.") == 2
    assert "Max observed prompt tokens:" not in rendered
    assert "chunk_tokens=`None`" not in rendered


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


def test_cdost_runtime_report_omits_gmm_only_findings():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_policy="certified_dual_view_otd",
        summary_max_model_tokens=100000,
        summary_budget_ratio=0.85,
        summary_prompt_overhead_reserve_tokens=8000,
        analyst_max_prompt_tokens=-1,
        budget_refinement_apply_to_level=0,
        soft_recursive_assignment="cumulative_mass",
        soft_top_r_memberships=2,
        soft_cumulative_mass_coverage=0.90,
        soft_max_membership_gap=0.25,
        workers=1,
        thinking="false",
        generation_timeout_seconds=1200,
        rollout_client_timeout_seconds=1200,
        chunked_embedding_enabled=True,
        embedding_batch_size=1,
        chunked_embedding_chunk_tokens=28000,
        embedding_max_model_len=32000,
        train_start=0,
        train_end=3,
        dynamic_initial_count=2,
        dynamic_arrival_count=1,
        tree_scenario="dynamic_update",
    )
    findings = runner.runtime_dead_corner_findings(args)
    areas = {finding["area"] for finding in findings}
    assert "budget_refinement" not in areas
    assert "soft_membership" not in areas
    assert "analyst_budget_override" not in areas


def test_cdost_hierarchy_fingerprint_uses_only_active_policy_fields():
    runner = _load_experiment_runner_module()
    payload = {
        "tree_policy": "certified_dual_view_otd",
        "otd": {"dual_view_lambda": 0.5},
        "summary_budget": {"max_model_tokens": 100000},
        "gmm_bic": {"min_split_size": 999},
        "soft_membership": {"recursive_assignment": "cumulative_mass"},
        "dynamic_update": {"mode": "budget_constrained_online_gmm"},
    }
    assert runner.active_hierarchy_payload(payload) == {
        "tree_policy": "certified_dual_view_otd",
        "otd": payload["otd"],
        "summary_budget": payload["summary_budget"],
    }

    gmm_payload = payload | {"tree_policy": "projected_gmm_bic"}
    active_gmm = runner.active_hierarchy_payload(gmm_payload)
    assert "otd" not in active_gmm
    assert active_gmm["gmm_bic"] == payload["gmm_bic"]
    assert active_gmm["soft_membership"] == payload["soft_membership"]


def test_cdost_dynamic_reuses_matching_static_embedding_vector_cache(
    tmp_path: Path,
) -> None:
    runner = _load_experiment_runner_module()
    static_dir = tmp_path / "static"
    tree_dir = static_dir / "dynamix_tree"
    cache_path = static_dir / "cache" / "embedding_cache.sqlite"
    tree_dir.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True)
    embedding_config = EmbeddingConfig(
        base_url="mock://deterministic",
        cache_path=str(cache_path),
        tokenizer_required=False,
    )
    embedding_client = EmbeddingClient(embedding_config)
    vector = asyncio.run(
        embedding_client.embed_texts(
            ["required vector"],
            cache_namespace="static_dynamic_test",
        )
    )[0]
    embedding_client.close()
    namespace = embedding_cache_namespace(
        embedding_config,
        "static_dynamic_test",
        model_name=embedding_config.model,
    )
    write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=(
            tree_dir / "embedding_vector_cache_manifest.json"
        ),
        requirements=[
            {
                "namespace": namespace,
                "text": "required vector",
                "vector": vector,
            }
        ],
    )
    vector_manifest = tree_dir / "embedding_vector_cache_manifest.json"
    atom_cache = tree_dir / "experience_atoms.json"
    atom_cache.write_text("{}", encoding="utf-8")
    marker = static_dir / "stage_markers" / "04_build_tree.done"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "output_identities": {
                    str(atom_cache): runner.path_fingerprint(atom_cache),
                    str(vector_manifest): runner.path_fingerprint(
                        vector_manifest
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    (static_dir / "dynamix_config.json").write_text(
        json.dumps(
            {
                "scenario": "static_build",
                "output_dir": str(tree_dir),
                "embedding": {"cache_path": str(cache_path)},
                "hierarchy": {
                    "tree_policy": "certified_dual_view_otd",
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = runner.resolve_embedding_cache_path(
        explicit_path="",
        scenario_dir=tmp_path / "dynamic",
        tree_policy="certified_dual_view_otd",
        tree_scenario="dynamic_update",
        atom_cache_path=str(atom_cache),
    )
    assert resolved == cache_path.resolve()
    original_vector_manifest = vector_manifest.read_text(encoding="utf-8")
    vector_manifest.write_text(
        original_vector_manifest + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed stage marker"):
        runner.resolve_embedding_cache_path(
            explicit_path="",
            scenario_dir=tmp_path / "dynamic",
            tree_policy="certified_dual_view_otd",
            tree_scenario="dynamic_update",
            atom_cache_path=str(atom_cache),
        )
    vector_manifest.write_text(
        original_vector_manifest,
        encoding="utf-8",
    )
    atom_cache.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="completed stage marker"):
        runner.resolve_embedding_cache_path(
            explicit_path="",
            scenario_dir=tmp_path / "dynamic",
            tree_policy="certified_dual_view_otd",
            tree_scenario="dynamic_update",
            atom_cache_path=str(atom_cache),
        )
    atom_cache.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must share"):
        runner.resolve_embedding_cache_path(
            explicit_path=str(tmp_path / "different.sqlite"),
            scenario_dir=tmp_path / "dynamic",
            tree_policy="certified_dual_view_otd",
            tree_scenario="dynamic_update",
            atom_cache_path=str(atom_cache),
        )
    import sqlite3

    connection = sqlite3.connect(cache_path)
    connection.execute(
        "UPDATE embeddings SET vector=? WHERE namespace=? AND key=?",
        (
            json.dumps([123.0]),
            namespace,
            hashlib.sha256(b"required vector").hexdigest(),
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="changed after certification"):
        runner.resolve_embedding_cache_path(
            explicit_path="",
            scenario_dir=tmp_path / "dynamic",
            tree_policy="certified_dual_view_otd",
            tree_scenario="dynamic_update",
            atom_cache_path=str(atom_cache),
        )


def test_cdost_dynamic_config_and_runtime_identity_are_method_specific():
    runner = _load_experiment_runner_module()
    payload = {
        "initial_count": 120,
        "arrival_count": 80,
        "update_batch_size": 8,
        "shuffle_seed": None,
        "snapshot_include_embeddings": True,
        "resume_from_snapshots": False,
        "max_propagation_rounds": 999,
    }
    assert runner.active_dynamic_payload(
        payload,
        tree_policy="certified_dual_view_otd",
    ) == {
        "initial_count": 120,
        "arrival_count": 80,
        "update_batch_size": 8,
        "shuffle_seed": None,
        "snapshot_include_embeddings": True,
        "resume_from_snapshots": False,
    }
    assert runner.active_dynamic_payload(
        payload,
        tree_policy="projected_gmm_bic",
    ) == payload

    args = SimpleNamespace(
        tree_policy="certified_dual_view_otd",
        tree_scenario="dynamic_update",
        dynamic_update_batch_size=8,
        graph_kind="overlapping_experience_hierarchy",
        allow_overlap=True,
        allow_multi_parent=True,
    )
    identity = runner.method_runtime_identity(args)
    assert identity == {
        "tree_policy": "certified_dual_view_otd",
        "structural_graph_kind": "single_parent_binary_tree",
        "allow_overlap": False,
        "allow_multi_parent": False,
        "arrival_update_semantics": "sequential_per_atom",
        "parent_refresh_semantics": "changed_path_bottom_up_per_atom",
        "snapshot_interval": 8,
        "nodebank_scope": "complete_tree",
        "retrieval_policy": "tree_antichain_knapsack",
        "embedding_vector_control": (
            "shared_content_addressed_cache_required"
        ),
    }
    args.tree_scenario = "static_build"
    static_identity = runner.method_runtime_identity(args)
    assert static_identity["arrival_update_semantics"] == "static_dataset_order"
    assert (
        static_identity["parent_refresh_semantics"]
        == "all_internal_nodes_bottom_up_after_build"
    )
    assert static_identity["snapshot_interval"] is None
    assert (
        static_identity["embedding_vector_control"]
        == "content_addressed_cache_populates_control"
    )


def test_cdost_runner_requires_nodebank_method_identity():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(tree_policy="certified_dual_view_otd")
    valid = {
        "tree_policy": "certified_dual_view_otd",
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
        },
    }
    runner.validate_nodebank_manifest_for_heldout(valid, args)
    with pytest.raises(RuntimeError, match="tree_policy identity"):
        runner.validate_nodebank_manifest_for_heldout(
            {"export_policy": valid["export_policy"]},
            args,
        )
    with pytest.raises(RuntimeError, match="antichain retrieval"):
        runner.validate_nodebank_manifest_for_heldout(
            {"tree_policy": "certified_dual_view_otd"},
            args,
        )


def test_experiment_runner_cdost_summary_gate_is_fail_closed():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_scenario="static_build",
        tree_policy="certified_dual_view_otd",
    )
    valid = {
        "scenario": "static_build",
        "tree_policy": "certified_dual_view_otd",
        "record_count": 3,
        "atom_count": 3,
        "excluded_count": 0,
        "parent_generation_error_count": 0,
        "atom_source": "generated",
    }
    runner.validate_tree_summary_for_heldout(valid, args)
    with pytest.raises(RuntimeError, match="parent skill generation"):
        runner.validate_tree_summary_for_heldout(
            valid | {"parent_generation_error_count": 1},
            args,
        )
    with pytest.raises(RuntimeError, match="excluded input"):
        runner.validate_tree_summary_for_heldout(
            valid | {"excluded_count": 1},
            args,
        )
    with pytest.raises(RuntimeError, match="atom_count"):
        runner.validate_tree_summary_for_heldout(
            valid | {"atom_count": 2},
            args,
        )

    dynamic_args = SimpleNamespace(
        tree_scenario="dynamic_update",
        tree_policy="certified_dual_view_otd",
        dynamic_initial_count=2,
        dynamic_arrival_count=1,
    )
    with pytest.raises(RuntimeError, match="frozen_cache"):
        runner.validate_tree_summary_for_heldout(
            valid
            | {
                "scenario": "dynamic_update",
                "initial_count": 2,
                "arrival_count": 1,
                "updated_count": 1,
                "insertion_count": 1,
            },
            dynamic_args,
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
    assert "template_user_prompt_adaptation" in payload
    assert set(schema) == {"cards"}
    card_schema = schema["cards"][0]
    assert set(card_schema) == {"name", "trigger", "content", "placement", "confidence"}
    forbidden = {"shared_patterns", "success_motifs", "anti_patterns", "shared_patch_hints", "reference_materials", "script_files", "skill_placement"}
    assert not (forbidden & set(card_schema))
    constraints = " ".join(payload["hard_constraints"])
    assert "Do not output fields except cards and each card's name, trigger, content, placement, confidence" in constraints


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
    assert set(payload) == {"instruction", "analyst_mode", "dynamic_patch_policy", "hard_constraints", "members", "previous_generated_experiences"}
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


def test_ebst_hierarchy_fingerprint_uses_only_active_policy_fields():
    runner = _load_experiment_runner_module()
    payload = {
        "tree_policy": "evidence_balanced_skill_tree",
        "ebst": {"max_entries": 8, "dual_view_lambda": 0.5},
        "summary_budget": {"max_model_tokens": 100000},
        "otd": {"dual_view_lambda": 0.1},
        "gmm_bic": {"min_split_size": 999},
    }
    assert runner.active_hierarchy_payload(payload) == {
        "tree_policy": "evidence_balanced_skill_tree",
        "ebst": payload["ebst"],
        "summary_budget": payload["summary_budget"],
    }


def test_ebst_runtime_identity_records_online_balanced_semantics():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_policy="evidence_balanced_skill_tree",
        tree_scenario="dynamic_update",
        dynamic_update_batch_size=8,
        graph_kind="overlapping_experience_hierarchy",
        allow_overlap=True,
        allow_multi_parent=True,
    )
    identity = runner.method_runtime_identity(args)
    assert identity["structural_graph_kind"] == (
        "single_parent_balanced_metric_tree"
    )
    assert identity["allow_overlap"] is False
    assert identity["arrival_update_semantics"] == (
        "sequential_structural_insert"
    )
    assert identity["parent_refresh_semantics"] == (
        "changed_path_batched_bottom_up"
    )


def test_ebst_runner_requires_capsule_only_nodebank():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_policy="evidence_balanced_skill_tree"
    )
    valid = {
        "tree_policy": "evidence_balanced_skill_tree",
        "root_node_id": "__root__",
        "node_count": 1,
        "nodes": [
            {
                "node_id": "capsule-1",
                "item_id": "capsule-1",
                "name": "Capsule",
                "trigger": "matching tasks",
                "content": "Apply the validated rule.",
                "embedding_text": (
                    "name: Capsule\n"
                    "trigger: matching tasks\n"
                    "content: Apply the validated rule."
                ),
                "prompt_text": "Validated capsule prompt.",
                "analyst_mode": "evidence_bucket_consolidation",
                "lifecycle_status": "active",
                "evidence_atom_ids": ["atom-1", "atom-2"],
                "parent_node_id": "__root__",
                "child_node_ids": [],
            }
        ],
        "tree_index": {
            "root_node_id": "__root__",
            "children_by_node": {
                "__root__": ["capsule-1"],
                "capsule-1": [],
            },
        },
        "export_policy": {
            "heldout_retrieval": "tree_antichain_knapsack",
            "experience_atoms_exported": False,
            "retrieval_unit": "validated_skill_capsule",
        },
    }
    runner.validate_nodebank_manifest_for_heldout(valid, args)
    archived = json.loads(json.dumps(valid))
    archived["nodes"][0]["lifecycle_status"] = "archived"
    with pytest.raises(ValueError, match="node entry is not retrievable"):
        runner.validate_nodebank_manifest_for_heldout(archived, args)
    with pytest.raises(RuntimeError, match="exclude experience atoms"):
        runner.validate_nodebank_manifest_for_heldout(
            {
                **valid,
                "export_policy": {
                    **valid["export_policy"],
                    "experience_atoms_exported": True,
                },
            },
            args,
        )


def test_ebst_summary_gate_blocks_retrievable_atoms():
    runner = _load_experiment_runner_module()
    args = SimpleNamespace(
        tree_scenario="static_build",
        tree_policy="evidence_balanced_skill_tree",
    )
    valid = {
        "scenario": "static_build",
        "tree_policy": "evidence_balanced_skill_tree",
        "record_count": 4,
        "atom_count": 4,
        "excluded_count": 0,
        "retrievable_atom_count": 0,
        "active_capsule_count": 2,
        "runtime_generation_error_count": 0,
        "prompt_budget_error_count": 0,
    }
    runner.validate_tree_summary_for_heldout(valid, args)
    with pytest.raises(RuntimeError, match="raw evidence atoms"):
        runner.validate_tree_summary_for_heldout(
            {**valid, "retrievable_atom_count": 1},
            args,
        )
    with pytest.raises(RuntimeError, match="runtime errors"):
        runner.validate_tree_summary_for_heldout(
            {**valid, "runtime_generation_error_count": 1},
            args,
        )
    with pytest.raises(RuntimeError, match="prompt budget"):
        runner.validate_tree_summary_for_heldout(
            {**valid, "prompt_budget_error_count": 1},
            args,
        )


def test_ebst_control_requires_cdost_baseline_protocol_match(
    tmp_path: Path,
):
    runner = _load_experiment_runner_module()
    shared = {
        "dataset": {"sha256": "dataset"},
        "records_sha256": "records",
        "train_split": [0, 200],
        "heldout_split": [200, 400],
        "paired_dynamic_schedule": {
            "initial_count": 120,
            "arrival_count": 80,
        },
        "retrieval": {
            "top_k": 10,
            "embedding_model": "Qwen3-Embedding-8B",
        },
        "rollout": {
            "model": "Qwen3.5-9B-AWQ",
            "thinking": "false",
            "max_turns": 30,
            "workers": 16,
            "generation_config": {"temperature": 0.0},
        },
        "evaluator": {"libreoffice": {"version": "test"}},
        "source": {
            "run_spreadsheetbench": "runner-sha",
            "evaluate_with_official": "evaluator-sha",
            "spreadsheetbench_support": "support-sha",
            "spreadsheet_agent": "agent-sha",
            "react_agent": "react-sha",
            "skillbank": "skillbank-sha",
            "antichain_retrieval": "antichain-sha",
            "dynamix_core": "baseline-core-sha",
        },
    }
    generation = {
        "model": "Qwen3.5-9B-AWQ",
        "temperature": 0.0,
        "thinking_mode": False,
    }
    embedding = {
        "model": "Qwen3-Embedding-8B",
        "max_model_len": 32000,
    }
    analyst = {"max_output_tokens": 4096}
    baseline_contract = {
        **shared,
        "tree": {
            "tree_policy": "certified_dual_view_otd",
            "generation": generation,
            "embedding": embedding,
            "analyst": analyst,
            "otd": {
                "dual_view_lambda": 0.5,
                "atom_temperature": 0.0,
                "parent_temperature": 0.0,
                "retrieval_token_budget": 24000,
                "retrieval_token_unit": 128,
                "retrieval_exact_search_max_states": 250000,
                "validation_mode": "structural_only",
            },
        },
    }
    baseline_manifest = tmp_path / "cdost_control_manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            {
                "format": "cdost_control_manifest_v1",
                "contract_sha256": runner._canonical_sha256(
                    baseline_contract
                ),
                "contract": baseline_contract,
            }
        ),
        encoding="utf-8",
    )
    current_contract = {
        **shared,
        "tree": {
            "tree_policy": "evidence_balanced_skill_tree",
            "generation": generation,
            "embedding": embedding,
            "analyst": analyst,
            "ebst": {
                "max_entries": 8,
                "dual_view_lambda": 0.5,
                "atom_temperature": 0.0,
                "capsule_temperature": 0.0,
                "validator_temperature": 0.0,
                "retrieval_token_budget": 24000,
                "retrieval_token_unit": 128,
                "retrieval_exact_search_max_states": 250000,
                "validation_mode": "structural_only",
            },
        },
    }
    report = runner.validate_ebst_against_cdost_baseline(
        current_contract=current_contract,
        baseline_manifest=baseline_manifest,
    )
    assert report["compatible"] is True

    mismatched = json.loads(json.dumps(current_contract))
    mismatched["rollout"]["workers"] = 8
    with pytest.raises(ValueError, match="differing sections.*rollout"):
        runner.validate_ebst_against_cdost_baseline(
            current_contract=mismatched,
            baseline_manifest=baseline_manifest,
        )

    mismatched_source = json.loads(json.dumps(current_contract))
    mismatched_source["source"]["react_agent"] = "changed-react-sha"
    with pytest.raises(
        ValueError,
        match="differing sections.*rollout_evaluator_source",
    ):
        runner.validate_ebst_against_cdost_baseline(
            current_contract=mismatched_source,
            baseline_manifest=baseline_manifest,
        )

    mismatched_retrieval = json.loads(json.dumps(current_contract))
    mismatched_retrieval["source"]["skillbank"] = "changed-skillbank-sha"
    with pytest.raises(
        ValueError,
        match="differing sections.*rollout_evaluator_source",
    ):
        runner.validate_ebst_against_cdost_baseline(
            current_contract=mismatched_retrieval,
            baseline_manifest=baseline_manifest,
        )

    treatment_only_source_change = json.loads(json.dumps(current_contract))
    treatment_only_source_change["source"]["dynamix_core"] = "ebst-core-sha"
    report = runner.validate_ebst_against_cdost_baseline(
        current_contract=treatment_only_source_change,
        baseline_manifest=baseline_manifest,
    )
    assert report["compatible"] is True

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import dynamix_benchmarks.officeqa.rollout as rollout_module
from dynamix_benchmarks.officeqa.data import OfficeQAItem, load_officeqa_split, load_officeqa_splits
from dynamix_benchmarks.officeqa.records import officeqa_result_to_record
from dynamix_benchmarks.officeqa.reward import evaluate_official_audit, evaluate_skillopt, extract_answer
from dynamix_benchmarks.officeqa.rollout import OfficeQARolloutConfig, build_system_prompt, build_user_prompt, run_officeqa_batch
from dynamix_benchmarks.officeqa.tools import build_oracle_parsed_pages_context, resolve_candidate_files, run_tool
from dynamix_trace2skill.pipeline import DynaMixRunConfig, _prepare_analyst_tokenizer_config, _records_to_items, _refresh_skillbank_index
from dynamix_trace2skill.schemas import RawTrajectoryRecord
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig
from dynamix_trace2skill.trace_views import render_analysis_bundle_text, render_embedding_trace
from dynamix_core.data_structures import ExperienceCommunity, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY


def test_officeqa_split_loads_items(tmp_path: Path) -> None:
    split = tmp_path / "splits" / "val"
    split.mkdir(parents=True)
    (split / "items.json").write_text(json.dumps([{
        "uid": "UIDX",
        "question": "What is the value?",
        "ground_truth": "42",
        "category": "easy",
        "source_files": ["doc.txt"],
        "source_docs": ["https://example.test?page=1"],
    }]), encoding="utf-8")
    items = load_officeqa_split(tmp_path / "splits", "val")
    assert items[0].uid == "UIDX"
    assert items[0].task_type == "easy"


def test_officeqa_combined_train_val_stream_preserves_order(tmp_path: Path) -> None:
    for split, uid in [("train", "UIDTRAIN"), ("val", "UIDVAL")]:
        split_dir = tmp_path / "splits" / split
        split_dir.mkdir(parents=True)
        (split_dir / "items.json").write_text(json.dumps([{
            "uid": uid,
            "question": f"Question for {split}?",
            "ground_truth": "42",
            "category": split,
            "source_files": ["doc.txt"],
            "source_docs": [],
        }]), encoding="utf-8")
    items = load_officeqa_splits(tmp_path / "splits", "train,val")
    assert [item.uid for item in items] == ["UIDTRAIN", "UIDVAL"]


def test_skillopt_reward_and_answer_extraction() -> None:
    assert extract_answer("<think>hidden</think><answer>2,602</answer>") == "2,602"
    assert extract_answer("<FINAL_ANSWER>2,602</FINAL_ANSWER>") == ""
    assert extract_answer("Use <answer>...</answer> as the format.\n<answer>73</answer>") == "73"
    result = evaluate_skillopt("2,602 million dollars", "2602")
    assert result["hard"] == 1


def test_official_audit_rejects_non_reward_py(tmp_path: Path) -> None:
    bad = tmp_path / "not_reward.py"
    bad.write_text("def score_answer(*args): return 1\n", encoding="utf-8")
    audit = evaluate_official_audit("1", "1", bad)
    assert audit is not None
    assert audit["available"] is False


def test_candidate_files_whitelist_for_read_and_grep(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    allowed = docs / "allowed.txt"
    denied = docs / "denied.txt"
    allowed.write_text("National defense 2602\n", encoding="utf-8")
    denied.write_text("secret\n", encoding="utf-8")
    roots = [str(docs)]
    resolved = resolve_candidate_files(["allowed.txt"], roots)
    assert resolved == [str(allowed.resolve())]
    _, obs = run_tool("grep", {"pattern": "defense", "path": str(allowed)}, allowed_roots=roots, allowed_paths=[str(allowed.resolve())])
    assert "2602" in obs
    _, denied_obs = run_tool("read", {"path": str(denied)}, allowed_roots=roots, allowed_paths=[str(allowed.resolve())])
    assert "path not allowed" in denied_obs
    _, denied_empty = run_tool("read", {"path": str(allowed)}, allowed_roots=roots, allowed_paths=[])
    assert "path not allowed" in denied_empty
    _, glob_empty = run_tool("glob", {"pattern": "*"}, allowed_roots=roots, allowed_paths=[])
    assert "allowlist is empty" in glob_empty


def test_candidate_whitelist_uses_exact_path_not_duplicate_basename(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    first = docs / "a"
    second = docs / "b"
    first.mkdir(parents=True)
    second.mkdir()
    allowed = first / "same.txt"
    denied = second / "same.txt"
    allowed.write_text("allowed\n", encoding="utf-8")
    denied.write_text("denied\n", encoding="utf-8")
    roots = [str(docs)]
    assert resolve_candidate_files(["same.txt"], roots) == []
    assert resolve_candidate_files(["a/same.txt"], roots) == [str(allowed.resolve())]
    _, denied_obs = run_tool("read", {"path": str(denied)}, allowed_roots=roots, allowed_paths=[str(allowed.resolve())])
    assert "path not allowed" in denied_obs


def test_user_prompt_has_question_candidates_and_hints_without_gold(tmp_path: Path) -> None:
    split = tmp_path / "val"
    split.mkdir()
    (split / "items.json").write_text(json.dumps([{
        "uid": "UIDX",
        "question": "What is the total?",
        "ground_truth": "999",
        "category": "hard",
        "source_files": ["doc.txt"],
        "source_docs": ["https://example.test?page=2"],
    }]), encoding="utf-8")
    item = load_officeqa_split(split, "val")[0]
    prompt = build_user_prompt(item, ["/tmp/doc.txt"])
    assert "What is the total?" in prompt
    assert "/tmp/doc.txt" in prompt
    assert "https://example.test?page=2" in prompt
    assert "999" not in prompt


def test_user_prompt_includes_oracle_pages_without_relaxing_absolute_allowlist(tmp_path: Path) -> None:
    docs = tmp_path / "treasury_bulletins_parsed"
    transformed = docs / "transformed"
    jsons = docs / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir()
    allowed = transformed / "treasury_bulletin_1941_01.txt"
    denied = transformed / "treasury_bulletin_1942_01.txt"
    allowed.write_text("allowed text\n", encoding="utf-8")
    denied.write_text("denied text\n", encoding="utf-8")
    (jsons / "treasury_bulletin_1941_01.json").write_text(json.dumps({
        "document": {
            "elements": [
                {
                    "bbox": [{"page_id": 15}],
                    "content": "<table><tr><th>Year</th><th>National defense</th></tr><tr><td>1940</td><td>2602</td></tr></table>",
                },
                {"bbox": [{"page_id": 16}], "content": "wrong page"},
            ]
        }
    }), encoding="utf-8")
    item = OfficeQAItem(
        uid="UIDX",
        question="What were national defense expenditures?",
        ground_truth="2602",
        task_type="hard",
        source_files=["treasury_bulletin_1941_01.txt"],
        source_docs=["https://fraser.test/title?page=15&deep=true"],
    )
    roots = [str(transformed.resolve())]
    candidates = resolve_candidate_files(item.source_files, roots)
    oracle_context = build_oracle_parsed_pages_context(item.source_files, item.source_docs, roots)
    prompt = build_user_prompt(item, candidates, oracle_context=oracle_context)
    assert "## Oracle Parsed Pages" in prompt
    assert "treasury_bulletin_1941_01.txt page 15" in prompt
    assert "National defense" in prompt
    assert "2602" in prompt
    assert str(allowed.resolve()) in prompt
    _, denied_obs = run_tool("read", {"path": str(denied)}, allowed_roots=roots, allowed_paths=candidates)
    assert "path not allowed" in denied_obs


def test_oracle_pages_map_each_source_doc_url_to_its_bulletin_file(tmp_path: Path) -> None:
    docs = tmp_path / "treasury_bulletins_parsed"
    transformed = docs / "transformed"
    jsons = docs / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir()
    for name in ["treasury_bulletin_1969_10.txt", "treasury_bulletin_1970_01.txt"]:
        (transformed / name).write_text(f"{name}\n", encoding="utf-8")
    (jsons / "treasury_bulletin_1969_10.json").write_text(json.dumps({
        "document": {"elements": [{"bbox": [{"page_id": 32}], "content": "October 1969 page evidence"}]}
    }), encoding="utf-8")
    (jsons / "treasury_bulletin_1970_01.json").write_text(json.dumps({
        "document": {"elements": [{"bbox": [{"page_id": 32}], "content": "January 1970 page evidence"}]}
    }), encoding="utf-8")
    roots = [str(transformed.resolve())]
    context = build_oracle_parsed_pages_context(
        ["treasury_bulletin_1969_10.txt"],
        [
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/october-1969-6874?page=32",
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/january-1970-6877?page=32",
        ],
        roots,
    )
    assert "treasury_bulletin_1969_10.txt page 32" in context
    assert "October 1969 page evidence" in context
    assert "treasury_bulletin_1970_01.txt page 32" in context
    assert "January 1970 page evidence" in context


def test_oracle_pages_handle_multiple_pages_per_derived_bulletin(tmp_path: Path) -> None:
    docs = tmp_path / "treasury_bulletins_parsed"
    transformed = docs / "transformed"
    jsons = docs / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir()
    (transformed / "treasury_bulletin_2011_03.txt").write_text("2011\n", encoding="utf-8")
    (transformed / "treasury_bulletin_2012_03.txt").write_text("2012\n", encoding="utf-8")
    for year in ["2011", "2012"]:
        (jsons / f"treasury_bulletin_{year}_03.json").write_text(json.dumps({
            "document": {
                "elements": [
                    {"bbox": [{"page_id": 20}], "content": f"{year} page 20"},
                    {"bbox": [{"page_id": 21}], "content": f"{year} page 21"},
                    {"bbox": [{"page_id": 22}], "content": f"{year} page 22"},
                ]
            }
        }), encoding="utf-8")
    context = build_oracle_parsed_pages_context(
        ["treasury_bulletin_2011_03.txt", "treasury_bulletin_2012_03.txt"],
        [
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/march-2011-7147?page=20",
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/march-2011-7147?page=21",
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/march-2012-7150?page=21",
            "https://fraser.stlouisfed.org/title/treasury-bulletin-407/march-2012-7150?page=22",
        ],
        [str(transformed.resolve())],
    )
    assert "2011 page 20" in context
    assert "2011 page 21" in context
    assert "2012 page 21" in context
    assert "2012 page 22" in context


def test_oracle_table_renderer_expands_rowspan_and_colspan(tmp_path: Path) -> None:
    docs = tmp_path / "treasury_bulletins_parsed"
    transformed = docs / "transformed"
    jsons = docs / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir()
    (transformed / "treasury_bulletin_1940_01.txt").write_text("1940\n", encoding="utf-8")
    (jsons / "treasury_bulletin_1940_01.json").write_text(json.dumps({
        "document": {
            "elements": [{
                "bbox": [{"page_id": 5}],
                "content": (
                    "<table>"
                    "<tr><th rowspan='2'>Fiscal year</th><th colspan='2'>Receipts</th></tr>"
                    "<tr><th>Gross</th><th>Net</th></tr>"
                    "<tr><td>1940</td><td>100</td><td>80</td></tr>"
                    "</table>"
                ),
            }]
        }
    }), encoding="utf-8")
    context = build_oracle_parsed_pages_context(
        ["treasury_bulletin_1940_01.txt"],
        ["https://fraser.stlouisfed.org/title/treasury-bulletin-407/january-1940-1?page=5"],
        [str(transformed.resolve())],
    )
    assert "| Fiscal year | Gross | Net |" in context
    assert "| 1940 | 100 | 80 |" in context


def test_bad_oracle_json_degrades_to_empty_context(tmp_path: Path) -> None:
    docs = tmp_path / "treasury_bulletins_parsed"
    transformed = docs / "transformed"
    jsons = docs / "jsons"
    transformed.mkdir(parents=True)
    jsons.mkdir()
    (transformed / "treasury_bulletin_1940_01.txt").write_text("1940\n", encoding="utf-8")
    (jsons / "treasury_bulletin_1940_01.json").write_text("{bad json", encoding="utf-8")
    context = build_oracle_parsed_pages_context(
        ["treasury_bulletin_1940_01.txt"],
        ["https://fraser.stlouisfed.org/title/treasury-bulletin-407/january-1940-1?page=5"],
        [str(transformed.resolve())],
    )
    assert context == ""


def test_officeqa_keeps_text_tool_fallback_prompt_available() -> None:
    prompt = build_system_prompt(text_tool_fallback=True)
    assert "<tool_call>" in prompt
    assert '"name": "grep"' in prompt


def test_officeqa_result_record_preserves_raw_trace_and_train_diagnostics() -> None:
    row = {
        "id": "UIDX",
        "question": "What is the total?",
        "task_type": "hard",
        "response": "I saw source evidence SOURCE_VALUE in doc.txt. <answer>1</answer>",
        "predicted_answer": "1",
        "ground_truth": "GOLD_VALUE",
        "hard": 0,
        "soft": 0.0,
        "source_files": ["doc.txt"],
        "source_docs": ["https://example.test"],
        "fail_reason": "predicted '1' but expected GOLD_VALUE",
        "primary_eval": {"gold_answer": "GOLD_VALUE", "predicted_answer": "1", "hard": 0, "f1": 0.0},
        "conversation": [
            {"type": "message", "content": "I will search evidence from doc.txt and mention SOURCE_VALUE.", "tool_calls": []},
            {"type": "tool_call", "cmd": "read(path='/tmp/private/doc.txt', start=1, limit=5)", "obs": "[SOURCE_VALUE] evidence source text"},
            {"type": "tool_call", "cmd": "grep(pattern='SOURCE_VALUE', path='/tmp/private/doc.txt')", "obs": "SOURCE_VALUE: source text"},
        ],
    }
    record = officeqa_result_to_record(row)
    text = render_embedding_trace(record)
    assert "What is the total?" in text
    assert "hard" in text
    assert "GOLD_VALUE" not in text
    assert "doc.txt" in text
    assert "/tmp/private" in text
    assert "SOURCE_VALUE" in text
    assert "evidence source text" in text
    assert "I will search" in text
    bundle = render_analysis_bundle_text(record)
    assert "GOLD_VALUE" in bundle
    assert "predicted_answer" in bundle
    assert "ground_truth" in bundle
    assert "doc.txt" in bundle
    assert "https://example.test" in bundle
    assert "/tmp/private" in bundle
    assert "expected GOLD_VALUE" in bundle
    assert "evidence source text" in bundle
    assert "I will search" in bundle


def test_officeqa_record_preserves_official_audit_for_train_diagnostics() -> None:
    row = {
        "id": "UIDX",
        "question": "What is the total?",
        "task_type": "hard",
        "response": "<answer>1</answer>",
        "ground_truth": "999",
        "hard": 0,
        "soft": 0.0,
        "primary_eval": {"gold_answer": "999", "predicted_answer": "1", "hard": 0, "f1": 0.0},
        "official_reward_audit": {"scorer": "official_reward_audit", "available": True, "error": "expected 999 got 1"},
        "conversation": [],
    }
    bundle = render_analysis_bundle_text(officeqa_result_to_record(row))
    assert "expected 999" in bundle
    assert "official_reward_error" not in bundle


def test_officeqa_text_tool_fallback_turn_is_one_step() -> None:
    row = {
        "id": "UIDX",
        "question": "What is the total?",
        "task_type": "hard",
        "response": "<answer>1</answer>",
        "ground_truth": "999",
        "hard": 0,
        "soft": 0.0,
        "conversation": [
            {"type": "message", "turn": 1, "content": '<tool_call>{"name":"grep","arguments":{"pattern":"total","path":"/tmp/doc.txt"}}</tool_call>', "tool_calls": []},
            {"type": "tool_call", "turn": 1, "tool_name": "grep", "cmd": "grep(pattern='total', path='/tmp/doc.txt')", "obs": "total 999", "tool_protocol": "text"},
        ],
    }
    record = officeqa_result_to_record(row)
    assert len(record.steps) == 1
    assert record.steps[0].tool_name == "grep"
    assert "<tool_call>" in record.steps[0].raw_model_output
    assert "total 999" in record.steps[0].observation


def test_officeqa_nodebank_audit_blocks_train_answer_leakage(tmp_path: Path) -> None:
    runner = _load_officeqa_runner()
    records = [{
        "task_id": "UIDX",
        "extra": {
            "officeqa_result": {"ground_truth": "GOLD_VALUE", "predicted_answer": "WRONG_VALUE"},
            "primary_eval": {"gold_answer": "GOLD_VALUE", "predicted_answer": "WRONG_VALUE"},
        },
    }]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    skillbank = tmp_path / "skills"
    skillbank.mkdir()
    (skillbank / "node_bank_manifest.json").write_text(json.dumps({
        "nodes": [{
            "node_id": "E1_bad",
            "name": "Bad card",
            "trigger": "When answer differs",
            "content": "The correct ground_truth was GOLD_VALUE.",
        }]
    }), encoding="utf-8")
    try:
        runner._audit_nodebank_for_train_diagnostic_leakage(skillbank, records_path, tmp_path)
    except ValueError as exc:
        assert "leaked train diagnostics" in str(exc)
    else:
        raise AssertionError("nodebank audit should reject copied train answer diagnostics")


def test_officeqa_runner_defaults_to_train_val_and_test() -> None:
    runner = _load_officeqa_runner()
    args = runner.parse_args(["--run-dir", "/tmp/x"])
    assert runner.OFFICEQA_PROTOCOL == "skillopt_compatible_officeqa_oracle_pages_v2"
    assert runner._resolve_train_splits(args) == ["train", "val"]
    assert args.heldout_split == "test"
    assert args.thinking == "true"
    assert args.max_tool_turns == 30
    assert args.skillbank_top_k == 10
    assert args.gmm_min_split_size == 4
    assert args.gmm_min_effective_samples_per_component == 4
    assert args.openai_base_url == "https://asmiatbrqksz.10.27.127.9.nip.io/v1"
    assert args.embedding_base_url == "http://10.26.1.184:18007/v1"
    assert args.embedding_tokenizer == runner.DEFAULT_EMBEDDING_TOKENIZER
    assert args.max_completion_tokens is None


def test_officeqa_build_config_uses_officeqa_analyst_profile(tmp_path: Path) -> None:
    runner = _load_officeqa_runner()
    args = runner.parse_args(["--run-dir", str(tmp_path / "run")])
    config = runner._build_tree_config(args, records_path=tmp_path / "records.json", output_dir=tmp_path / "tree")
    assert config["analyst"]["task_profile"] == "officeqa"
    assert config["analyst"]["prompt_style"] == "officeqa_cluster_level_v1"
    assert config["hierarchy"]["gmm_bic"]["min_split_size"] == 4
    assert config["hierarchy"]["gmm_bic"]["min_effective_samples_per_component"] == 4


OFFICEQA_ANALYST_FORBIDDEN_TOKENS = [
    "Trace2Skill",
    "Treasury bulletin",
    "glob/read/grep",
    "grep",
    "Candidate Files",
    "Source Hints",
    "Oracle Parsed Pages",
    "PDF page",
    "printed bulletin page",
    "<answer>",
    "iterative minimal-fix verifier loop",
    "xlsx",
    "workbook",
    "worksheet",
    "openpyxl",
    "spreadsheet manipulation tasks",
    "template_user_prompt_adaptation",
    "success_user_template",
    "error_user_template",
]


def test_officeqa_cluster_analyst_prompt_is_dataset_bound() -> None:
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(task_profile=" OfficeQA "))  # type: ignore[arg-type]
    system_prompt = analyst._system_prompt("raw_extractor")
    dynamic_prompt = analyst._dynamic_system_prompt("raw_extractor")
    community = ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(
        item_id="t0",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_bundle": "Selected relevant document evidence and computed operands."},
    )
    payload = json.loads(analyst._build_prompt(community, [member], "raw_extractor"))

    combined = "\n".join([system_prompt, dynamic_prompt, json.dumps(payload, ensure_ascii=False)])
    assert "OfficeQA" in combined
    assert "document question answering" in combined
    assert "Trajectory evidence discipline" in combined
    assert "officeqa_experience_policy" in payload
    assert "template_user_prompt_adaptation" not in payload
    assert "generic advice to use available actions" in json.dumps(payload["officeqa_experience_policy"], ensure_ascii=False)
    assert "predicted_answer" in payload["officeqa_experience_policy"]["train_diagnostic_use"]
    assert "ground_truth" in payload["officeqa_experience_policy"]["train_diagnostic_use"]
    for token in OFFICEQA_ANALYST_FORBIDDEN_TOKENS:
        assert token not in combined


def test_officeqa_raw_dynamic_prompt_is_generalized_and_dataset_bound() -> None:
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(task_profile="officeqa"))  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="L0_C0", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(
        item_id="t0",
        level=0,
        kind=ITEM_KIND_TRAJECTORY,
        text="trace",
        embedding=[1.0],
        metadata={"analysis_bundle": "Selected relevant document evidence and computed operands."},
    )
    prompt = analyst._build_dynamic_update_prompt(
        community,
        [member],
        previous_generated_experiences=[{"item_id": "old", "metadata": {"name": "Old", "trigger": "T", "content": "C", "confidence": 0.5}}],
        analyst_mode="raw_extractor",
    )
    payload = json.loads(prompt)
    combined = json.dumps(payload, ensure_ascii=False)
    assert payload["task_profile"] == "officeqa"
    assert "officeqa_experience_policy" in payload
    assert "template_user_prompt_adaptation" not in payload
    assert "new_cards" in payload["dynamic_patch_policy"]
    for token in OFFICEQA_ANALYST_FORBIDDEN_TOKENS:
        assert token not in combined


def test_officeqa_cluster_analyst_higher_level_dynamic_is_update_only() -> None:
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(task_profile="officeqa"))  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="L1_C0", level=1, member_weights={"e0": 1.0})
    member = ExperienceItem(
        item_id="e0",
        level=1,
        kind=ITEM_KIND_EXPERIENCE_CARD,
        text="name: Operand Ledger\ntrigger: financial table arithmetic\ncontent: track operands",
        embedding=[1.0],
        metadata={"name": "Operand Ledger", "trigger": "financial table arithmetic", "content": "track operands"},
    )
    prompt = analyst._build_dynamic_update_prompt(
        community,
        [member],
        previous_generated_experiences=[{"item_id": "old", "metadata": {"name": "Old", "trigger": "T", "content": "C", "confidence": 0.5}}],
        analyst_mode="experience_abstractor",
    )
    payload = json.loads(prompt)
    system_prompt = analyst._dynamic_system_prompt("experience_abstractor")
    assert "new_cards" not in payload
    assert "new_cards" not in system_prompt
    assert "updates" in payload["dynamic_patch_policy"]
    assert payload["task_profile"] == "officeqa"
    assert "officeqa_experience_policy" in payload


def test_officeqa_vanilla_runner_defaults_to_test_no_skill_baseline() -> None:
    runner = _load_officeqa_vanilla_runner()
    args = runner.parse_args(["--run-dir", "/tmp/x"])
    assert runner.VANILLA_PROTOCOL == "skillopt_compatible_officeqa_vanilla_oracle_pages_v1"
    assert args.split == "test"
    assert args.expected_count == 172
    assert args.model == "Qwen3.5-9B-AWQ"
    assert args.openai_base_url == "http://asmiatbrqksz.10.27.127.9.nip.io/v1"
    assert args.generation_temperature == 0.6
    assert args.generation_timeout == 1200.0
    assert args.thinking == "true"
    assert args.workers == 8
    assert args.max_tool_turns == 30
    assert args.max_completion_tokens is None


def test_officeqa_vanilla_resume_fingerprint_includes_workers() -> None:
    runner = _load_officeqa_vanilla_runner()
    args_8 = runner.parse_args(["--run-dir", "/tmp/x", "--workers", "8"])
    args_12 = runner.parse_args(["--run-dir", "/tmp/x", "--workers", "12"])
    assert runner._vanilla_fingerprint(args_8, ["/tmp/docs"]) != runner._vanilla_fingerprint(args_12, ["/tmp/docs"])


def test_officeqa_vanilla_runner_never_injects_retrieved_experience(tmp_path: Path, monkeypatch) -> None:
    runner = _load_officeqa_vanilla_runner()
    run_dir = tmp_path / "vanilla"
    items = [
        OfficeQAItem(uid="UIDA", question="Question A?", ground_truth="1", task_type="easy", source_files=[], source_docs=[]),
        OfficeQAItem(uid="UIDB", question="Question B?", ground_truth="2", task_type="hard", source_files=[], source_docs=[]),
    ]
    seen = {}

    def fake_run_officeqa_batch(batch_items, out_root, config, *, skill_content_fn=None):
        seen["items"] = batch_items
        seen["out_root"] = out_root
        seen["config"] = config
        seen["skill_content_fn"] = skill_content_fn
        return [{"id": item.uid, "hard": 1, "soft": 1.0, "agent_ok": True} for item in batch_items]

    monkeypatch.setattr(runner, "validate_generation_endpoint", lambda args: None)
    monkeypatch.setattr(runner, "resolve_docs_roots", lambda docs_dir: ["/tmp/docs"])
    monkeypatch.setattr(runner, "load_officeqa_split", lambda *args, **kwargs: items)
    monkeypatch.setattr(runner, "run_officeqa_batch", fake_run_officeqa_batch)
    monkeypatch.setattr(sys, "argv", [
        "run_officeqa_vanilla_test.py",
        "--run-dir", str(run_dir),
        "--expected-count", "2",
    ])

    runner.main()

    assert seen["items"] == items
    assert seen["out_root"] == run_dir / "vanilla_rollout"
    assert seen["skill_content_fn"] is None
    assert seen["config"].thinking is True
    assert seen["config"].workers == 8
    assert seen["config"].max_tool_turns == 30
    assert seen["config"].max_completion_tokens is None
    report = json.loads((run_dir / "officeqa_vanilla_report.json").read_text(encoding="utf-8"))
    assert report["preflight"]["nodebank_used"] is False
    assert report["preflight"]["retrieved_experience_injected"] is False
    assert report["preflight"]["expected_count"] == 2
    assert report["preflight"]["subset_debug"] is True
    assert report["vanilla"]["hard"] == 2


def test_officeqa_vanilla_runner_enforces_expected_count(tmp_path: Path, monkeypatch) -> None:
    runner = _load_officeqa_vanilla_runner()
    run_dir = tmp_path / "vanilla"
    item = OfficeQAItem(uid="UIDA", question="Question A?", ground_truth="1", task_type="easy", source_files=[], source_docs=[])

    monkeypatch.setattr(runner, "_validate_generation_endpoint", lambda args: None, raising=False)
    monkeypatch.setattr(runner, "validate_generation_endpoint", lambda args: None)
    monkeypatch.setattr(runner, "resolve_docs_roots", lambda docs_dir: ["/tmp/docs"])
    monkeypatch.setattr(runner, "load_officeqa_split", lambda *args, **kwargs: [item])
    monkeypatch.setattr(sys, "argv", [
        "run_officeqa_vanilla_test.py",
        "--run-dir", str(run_dir),
    ])

    try:
        runner.main()
    except ValueError as exc:
        assert "expected 172 items" in str(exc)
    else:
        raise AssertionError("expected count mismatch to fail")


def test_skillbank_index_refresh_uses_chunked_embedding_protocol(tmp_path: Path) -> None:
    bank = tmp_path / "skills"
    bank.mkdir()
    (bank / "node_bank_manifest.json").write_text(json.dumps({
        "format": "dynamix_node_skill_bank_v1",
        "nodes": [{
            "node_id": "node-1",
            "item_id": "node-1",
            "name": "Read targeted Treasury evidence",
            "trigger": "OfficeQA local document lookup",
            "content": "Use grep before read.",
        }],
    }), encoding="utf-8")
    cfg = DynaMixRunConfig(output_dir=str(tmp_path / "out"), records_path=str(tmp_path / "records.json"))
    cfg.embedding.base_url = "mock://deterministic"
    cfg.embedding.model = "Qwen3-Embedding-8B"
    cfg.embedding.max_model_len = 32000
    cfg.embedding.max_input_tokens = 32000
    cfg.embedding.batch_size = 8
    cfg.chunked_embedding = {
        "enabled": True,
        "chunk_tokens": 28000,
        "overlap_tokens": 1000,
    }
    index_path = Path(_refresh_skillbank_index(bank, cfg))
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["embedding_protocol"]["max_model_len"] == 32000
    assert payload["embedding_protocol"]["chunk_tokens"] == 28000
    assert payload["embedding_protocol"]["chunk_overlap_tokens"] == 1000


def test_mock_pipeline_records_to_items_uses_regex_tokenizer_without_hf(tmp_path: Path) -> None:
    record = RawTrajectoryRecord(
        trajectory_id="officeqa:UIDX",
        task_id="UIDX",
        trial_index=0,
        instruction="What is the total?",
        instruction_type="officeqa",
    )
    cfg = DynaMixRunConfig(output_dir=str(tmp_path / "out"), records_path=str(tmp_path / "records.json"))
    _prepare_analyst_tokenizer_config(cfg, tmp_path / "out")
    _, normalized = _records_to_items([record], ["trace"], [[1.0, 0.0]], config=cfg)
    assert normalized[0]["experience_item"]["metadata"]["analysis_tokenizer"] == "regex_fallback_test_only"


def test_officeqa_resume_requires_matching_fingerprint(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    item = OfficeQAItem(
        uid="UIDX",
        question="What is the total?",
        ground_truth="42",
        task_type="easy",
        source_files=[],
        source_docs=[],
    )
    out = tmp_path / "rollout"
    cfg1 = OfficeQARolloutConfig(docs_dirs=(str(docs),), resume=True, resume_fingerprint="fingerprint-a")
    cfg2 = OfficeQARolloutConfig(docs_dirs=(str(docs),), resume=True, resume_fingerprint="fingerprint-b")
    run_officeqa_batch([item], out, cfg1)
    run_officeqa_batch([item], out, cfg2)
    jsonl_rows = [json.loads(line) for line in (out / "officeqa_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["resume_fingerprint"] for row in jsonl_rows] == ["fingerprint-a", "fingerprint-b"]
    latest_rows = json.loads((out / "officeqa_results.json").read_text(encoding="utf-8"))
    assert [row["resume_fingerprint"] for row in latest_rows] == ["fingerprint-b"]


def test_officeqa_runner_key_config_does_not_persist_raw_fallback(monkeypatch) -> None:
    runner = _load_officeqa_runner()
    monkeypatch.delenv("MISSING_OFFICEQA_KEY", raising=False)
    monkeypatch.delenv("DYNAMIX_OFFICEQA_OPENAI_API_KEY", raising=False)
    args = SimpleNamespace(openai_api_key="fallback-secret", openai_api_key_env="MISSING_OFFICEQA_KEY")
    try:
        api_key, env_var = runner._generation_api_key_config(args)
        assert api_key == "EMPTY"
        assert env_var == "DYNAMIX_OFFICEQA_OPENAI_API_KEY"
        assert os.environ[env_var] == "fallback-secret"
    finally:
        monkeypatch.delenv("DYNAMIX_OFFICEQA_OPENAI_API_KEY", raising=False)


def test_officeqa_rollout_falls_back_to_text_tool_protocol(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "doc.txt"
    source.write_text("answer is 42\n", encoding="utf-8")
    item = OfficeQAItem(
        uid="UIDX",
        question="What is the answer?",
        ground_truth="42",
        task_type="easy",
        source_files=["doc.txt"],
        source_docs=[],
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("tool_choice") == "auto":
                raise RuntimeError('"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set')
            text_calls = [call for call in self.calls if "tools" not in call]
            if len(text_calls) == 1:
                content = json.dumps({"name": "read", "arguments": {"path": str(source), "start": 1, "limit": 1}})
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"<tool_call>{content}</tool_call>", tool_calls=[]))])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="<answer>42</answer>", tool_calls=[]))])

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(rollout_module, "_openai_client", lambda config: fake_client)

    cfg = OfficeQARolloutConfig(base_url="http://127.0.0.1:18002/v1", model="Qwen3.5-9B", docs_dirs=(str(docs),), max_tool_turns=3)
    row = rollout_module._process_one(item, tmp_path / "out", cfg, [str(docs)], "")

    assert row["hard"] == 1
    assert row["agent_ok"] is True
    assert any(event.get("type") == "protocol_fallback" for event in row["conversation"])
    assert any(event.get("tool_protocol") == "text" and "read(" in event.get("cmd", "") for event in row["conversation"])
    assert "tools" in completions.calls[0]
    assert all("tools" not in call for call in completions.calls[1:])
    assert all(call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True for call in completions.calls)
    assert all("max_tokens" not in call for call in completions.calls)


def test_officeqa_runner_heldout_retrieves_top10_per_task(tmp_path: Path, monkeypatch) -> None:
    runner = _load_officeqa_runner()
    run_dir = tmp_path / "run"
    skills = run_dir / "tree" / "skills"
    skills.mkdir(parents=True)
    (run_dir / "tree" / "summary.json").write_text(json.dumps({
        "node_bank_dir": str(skills),
        "node_count": 1,
    }), encoding="utf-8")
    (skills / "node_bank_manifest.json").write_text("{}", encoding="utf-8")
    (skills / ".dynamix_skillbank_index.json").write_text("{}", encoding="utf-8")
    records = tmp_path / "records.json"
    records.write_text("[]", encoding="utf-8")
    heldout_items = [
        OfficeQAItem(uid="UIDA", question="Question A?", ground_truth="1", task_type="easy", source_files=[], source_docs=[]),
        OfficeQAItem(uid="UIDB", question="Question B?", ground_truth="2", task_type="hard", source_files=[], source_docs=[]),
    ]
    select_calls: list[tuple[str, int]] = []
    rendered_skill_content: list[str] = []

    class FakeSelector:
        def __init__(self, **_: object) -> None:
            pass

        def select(self, query: str, *, top_k: int):
            select_calls.append((query, top_k))
            return [SimpleNamespace(skill=SimpleNamespace(name="Node", trigger="Trigger", content="Content"))]

    def fake_run_officeqa_batch(items, out_root, config, *, skill_content_fn=None):
        assert config.max_completion_tokens is None
        assert skill_content_fn is not None
        for item in items:
            rendered_skill_content.append(skill_content_fn(item))
        return [{"id": item.uid, "hard": 1, "soft": 1.0} for item in items]

    monkeypatch.setattr(runner, "_validate_generation_endpoint", lambda args: None)
    monkeypatch.setattr(runner, "resolve_docs_roots", lambda docs_dir: ["/tmp/docs"])
    monkeypatch.setattr(runner, "load_officeqa_splits", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "load_officeqa_split", lambda *args, **kwargs: heldout_items)
    monkeypatch.setattr(runner, "SkillBankSelector", FakeSelector)
    monkeypatch.setattr(runner, "run_officeqa_batch", fake_run_officeqa_batch)
    monkeypatch.setattr(sys, "argv", [
        "run_officeqa_dynamix_experiment.py",
        "--run-dir", str(run_dir),
        "--records-path", str(records),
        "--skip-train-rollout",
        "--skip-build-tree",
        "--run-heldout",
    ])

    runner.main()

    assert [top_k for _, top_k in select_calls] == [10, 10]
    assert "Question A?\n\nTask type: easy" == select_calls[0][0]
    assert "Question B?\n\nTask type: hard" == select_calls[1][0]
    assert len(rendered_skill_content) == 2
    assert all("# Retrieved Experience" in content for content in rendered_skill_content)
    assert all("## Experience 1: Node" in content for content in rendered_skill_content)


def test_officeqa_reused_records_allow_same_split_different_order(tmp_path: Path) -> None:
    runner = _load_officeqa_runner()
    records = tmp_path / "records.json"
    records.write_text(json.dumps([
        {"trajectory_id": "officeqa:UIDB", "task_id": "UIDB"},
        {"trajectory_id": "officeqa:UIDA", "task_id": "UIDA"},
    ]), encoding="utf-8")
    items = [
        OfficeQAItem(uid="UIDA", question="A?", ground_truth="1", task_type="easy", source_files=[], source_docs=[]),
        OfficeQAItem(uid="UIDB", question="B?", ground_truth="2", task_type="hard", source_files=[], source_docs=[]),
    ]

    validation = runner._validate_records_match_items(records, items)

    assert validation["order_matches"] is False
    assert validation["first_record_ids"] == ["UIDB", "UIDA"]
    assert validation["first_expected_ids"] == ["UIDA", "UIDB"]


def test_officeqa_reused_records_must_match_train_split_ids(tmp_path: Path) -> None:
    runner = _load_officeqa_runner()
    records = tmp_path / "records.json"
    records.write_text(json.dumps([
        {"trajectory_id": "officeqa:UIDA", "task_id": "UIDA"},
        {"trajectory_id": "officeqa:UIDB", "task_id": "UIDB"},
    ]), encoding="utf-8")
    items = [
        OfficeQAItem(uid="UIDA", question="A?", ground_truth="1", task_type="easy", source_files=[], source_docs=[]),
        OfficeQAItem(uid="UIDC", question="C?", ground_truth="3", task_type="hard", source_files=[], source_docs=[]),
    ]
    try:
        runner._validate_records_match_items(records, items)
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("mismatched OfficeQA records should be rejected")


def _load_officeqa_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_officeqa_dynamix_experiment.py"
    spec = importlib.util.spec_from_file_location("officeqa_dynamix_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_officeqa_vanilla_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_officeqa_vanilla_test.py"
    spec = importlib.util.spec_from_file_location("officeqa_vanilla_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

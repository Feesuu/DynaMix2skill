from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from dynamix_trace2skill.clients import EmbeddingClient, EmbeddingConfig
from dynamix_trace2skill.summary import ClusterAnalyst, ClusterAnalystConfig
from dynamix_trace2skill.log_parser import parse_trace2skill_logs, _result_fields
from dynamix_trace2skill.pipeline import default_hierarchy_config
from dynamix_core.data_structures import ExperienceCommunity, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY


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


def test_embedding_raises_when_truncation_disabled(tmp_path):
    cfg = EmbeddingConfig(base_url="mock://deterministic", max_input_tokens=10, truncate_long_texts=False, tokenizer_required=False)
    client = EmbeddingClient(cfg)
    with pytest.raises(ValueError):
        asyncio.run(async_embed(client, ["x " * 20]))


def test_cluster_analyst_uses_all_members_not_member_cap():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C0", level=0, member_weights={f"t{i}": 1.0 for i in range(20)})
    members = [ExperienceItem(item_id=f"t{i}", level=0, kind=ITEM_KIND_TRAJECTORY, text=f"trace {i}", embedding=[1.0], metadata={"analysis_bundle": f"bundle {i}"}) for i in range(20)]
    prompt = analyst._build_prompt(community, members)
    payload = json.loads(prompt)
    assert len(payload["members"]) == 20
    assert "Use all provided members" in " ".join(payload["hard_constraints"])
    assert "success_user_template" in payload["template_user_prompt_adaptation"]


def test_default_hierarchy_config_is_real_not_tiny_smoke():
    cfg = default_hierarchy_config({})
    assert cfg.gmm_bic.min_split_size == 16
    assert cfg.gmm_bic.min_effective_samples_per_component == 8
    assert cfg.gmm_bic.abs_kmax == 64
    assert cfg.gmm_bic.num_restarts == 5


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
    system = analyst._system_prompt()
    user = analyst._build_prompt(community, members)
    analyst._preflight_prompt_budget(community, system, user, len(members))
    report = json.loads((tmp_path / "prompt_tokens.json").read_text())
    assert report["events"][0]["community_id"] == "C_budget"
    assert report["events"][0]["prompt_tokens"] > 0


def test_cluster_prompt_preflight_fails_when_over_budget():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig(tokenizer_required=False, allow_regex_tokenizer_fallback=True, max_prompt_tokens=1))  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_too_big", level=0, member_weights={"t0": 1.0})
    members = [ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "many tokens here"})]
    with pytest.raises(ValueError):
        analyst._preflight_prompt_budget(community, analyst._system_prompt(), analyst._build_prompt(community, members), len(members))


def test_budget_refinement_retains_oversize_singleton_as_fallback_community():
    from dynamix_core.tree_builder import ProjectedGmmTreeBuilder

    cfg = default_hierarchy_config({
        "summary_budget": {"max_model_tokens": 10, "budget_ratio": 0.5},
        "budget_refinement": {
            "enabled": True,
            "apply_to_level": 0,
            "selection_policy": "bic_best_with_token_progress",
            "min_token_reduction_fraction": 0.10,
            "fallback": "pca_token_balanced_binary",
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
    assert not clustering.should_stop
    assert len(clustering.communities) == 1
    community = clustering.communities[0]
    assert community.member_weights == {"too_long": 1.0}
    assert community.metadata["oversize_singleton"] is True
    assert community.metadata["llm_summary_skipped"] is True
    assert clustering.summary_budget["oversize_singleton_fallback_count"] == 1


def test_cluster_analyst_uses_non_llm_fallback_for_oversize_singleton():
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
    assert len(cards) == 1
    card = cards[0]
    assert card.kind == ITEM_KIND_EXPERIENCE_CARD
    assert card.embedding == [0.25, 0.75]
    assert list(card.generated_from_community_ids) == ["C_oversize"]
    assert card.metadata["oversize_singleton_fallback"] is True
    assert card.metadata["llm_summary_skipped"] is True
    assert card.metadata["placement"]["target"] == "reference"
    assert analyst.config.token_report == []


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


def test_skill_export_respects_llm_placement_and_support_files(tmp_path):
    from dynamix_core.skill_export import export_skill_files_from_payload
    payload = {
        "items": {
            "root": {
                "item_id": "root", "level": 2, "kind": "experience_card", "text": "Root guidance", "support_mass": 10.0,
                "generated_from_community_ids": ["c1"],
                "metadata": {"name": "Root", "trigger": "Use for root guidance", "content": "Root guidance", "confidence": 0.9, "placement": {"target": "skill_md"}},
            },
            "ref": {
                "item_id": "ref", "level": 1, "kind": "experience_card", "text": "Detailed edge case", "support_mass": 7.0,
                "generated_from_community_ids": ["c0"],
                "metadata": {"name": "Edge Details", "trigger": "Use for edge cases", "content": "Detailed edge case", "confidence": 0.8, "placement": {"target": "reference", "reference_kind": "edge_case"}},
            },
            "script": {
                "item_id": "script", "level": 1, "kind": "experience_card", "text": "Helper script note", "support_mass": 5.0,
                "generated_from_community_ids": ["c0"],
                "metadata": {"name": "Helper", "trigger": "Use for helper script", "content": "print('ok')", "confidence": 0.7, "placement": {"target": "script", "reference_kind": "procedure"}},
            },
        },
        "communities": {
            "c1": {"community_id": "c1", "level": 1, "member_weights": {"ref": 1.0, "script": 1.0}, "posterior_member_weights": {"ref": 1.0, "script": 1.0}, "generated_item_ids": ["root"], "support_mass": 10.0},
            "c0": {"community_id": "c0", "level": 0, "member_weights": {}, "posterior_member_weights": {}, "generated_item_ids": ["ref", "script"], "support_mass": 12.0},
        },
    }
    result = export_skill_files_from_payload(payload, tmp_path)
    skill_dir = Path(result.skills[0].path).parent
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "references" / "index.md").exists()
    assert list((skill_dir / "references" / "edge-cases").glob("*.md"))
    assert list((skill_dir / "scripts").glob("helper*.py"))
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["placement_stats"]["reference"] >= 1
    assert any(entry["node_id"] == "ref" and entry["material_kind"] == "reference" for entry in manifest["node_file_catalog"])
    assert any(entry["node_id"] == "script" and entry["material_kind"] == "script" for entry in manifest["node_file_catalog"])


def test_skill_install_copies_support_files(tmp_path):
    from dynamix_trace2skill.skill_install import install_skill_for_trace2skill
    source = tmp_path / "source_skill"
    (source / "references").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n", encoding="utf-8")
    (source / "references" / "index.md").write_text("# index\n", encoding="utf-8")
    (source / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")
    manifest = install_skill_for_trace2skill(source, tmp_path / "installed")
    canonical_dir = Path(manifest["canonical_skill_dir"])
    assert (canonical_dir / "references" / "index.md").exists()
    assert (canonical_dir / "scripts" / "helper.py").exists()
    assert (tmp_path / "installed" / "xlsx" / "references" / "index.md").exists()


def test_cluster_prompt_uses_minimal_experience_schema():
    analyst = ClusterAnalyst(None, None, ClusterAnalystConfig())  # type: ignore[arg-type]
    community = ExperienceCommunity(community_id="C_schema", level=0, member_weights={"t0": 1.0})
    member = ExperienceItem(item_id="t0", level=0, kind=ITEM_KIND_TRAJECTORY, text="trace", embedding=[1.0], metadata={"analysis_bundle": "bundle"})
    payload = json.loads(analyst._build_prompt(community, [member]))
    schema = payload["output_schema"]
    assert set(schema) == {"name", "trigger", "content", "placement", "confidence"}
    forbidden = {"shared_patterns", "success_motifs", "anti_patterns", "shared_patch_hints", "reference_materials", "script_files", "skill_placement"}
    assert not (forbidden & set(schema))
    constraints = " ".join(payload["hard_constraints"])
    assert "Do not output fields except name, trigger, content, placement, confidence" in constraints


def test_render_card_text_minimal_schema_only():
    from dynamix_trace2skill.summary import _render_card_text
    text = _render_card_text({"name": "N", "trigger": "T", "content": "C"})
    assert "# N" in text
    assert "## Trigger" in text
    assert "## Content" in text
    assert "Shared patterns" not in text
    assert "Success motifs" not in text



def test_skill_export_names_skill_from_root_seed_name(tmp_path):
    from dynamix_core.skill_export import export_skill_files_from_payload
    payload = {
        "items": {
            "root": {
                "item_id": "root", "level": 1, "kind": "experience_card", "text": "Lookup guidance", "support_mass": 3.0,
                "generated_from_community_ids": ["c0"],
                "metadata": {"name": "Cross Sheet Lookup", "trigger": "lookup tasks", "content": "Use lookup guidance.", "confidence": 0.9, "placement": {"target": "skill_md"}},
            },
        },
        "communities": {"c0": {"community_id": "c0", "level": 0, "member_weights": {}, "posterior_member_weights": {}, "generated_item_ids": ["root"], "support_mass": 3.0}},
    }
    result = export_skill_files_from_payload(payload, tmp_path)
    skill = result.skills[0]
    assert skill.skill_id.startswith("skill_001_cross-sheet-lookup")
    text = Path(skill.path).read_text(encoding="utf-8")
    assert "name: \"Cross Sheet Lookup\"" in text


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


def test_skillbank_selects_topk_by_embedding_without_copying_skill_folders(tmp_path):
    from dynamix_trace2skill.skillbank import SkillBankSelector
    bank = tmp_path / "bank"
    (bank / "lookup" / "references").mkdir(parents=True)
    (bank / "formatting").mkdir(parents=True)
    (bank / "lookup" / "SKILL.md").write_text("---\nname: lookup skill\ndescription: vlookup and matching\n---\nUse lookup formulas and match keys.\n", encoding="utf-8")
    (bank / "lookup" / "references" / "index.md").write_text("# lookup references\n", encoding="utf-8")
    (bank / "formatting" / "SKILL.md").write_text("---\nname: formatting skill\ndescription: styles and colors\n---\nUse fonts, fills, and number formats.\n", encoding="utf-8")
    selector = SkillBankSelector(skillbank_root=bank, base_url="mock://deterministic", model="mock-embed", cache_path=tmp_path / "index.json")
    selected = selector.select("need vlookup matching by key", top_k=1)
    assert len(selected) == 1
    assert "lookup" in selected[0].skill.name.lower()
    assert Path(selected[0].skill.skill_dir) == bank / "lookup"
    assert Path(selected[0].skill.skill_path) == bank / "lookup" / "SKILL.md"
    assert (Path(selected[0].skill.skill_dir) / "references" / "index.md").exists()
    assert not (bank / ".dynamix_selected_skills").exists()
    assert not (bank / "selected_skills").exists()


def test_skillbank_selection_is_read_only_and_logs_selected_paths(tmp_path, monkeypatch):
    from spreadsheet_agent.agents.cli_skill_preloaded_agent import CLISkillPreloadedAgent

    class DummyClient:
        pass

    class DummySelector:
        def __init__(self, skill_dir):
            self.skill_dir = skill_dir
        def select(self, query, top_k=3):
            from dynamix_trace2skill.skillbank import SkillDocument, SkillSelection
            doc = SkillDocument(
                skill_id="selected-skill",
                name="Selected Skill",
                description="desc",
                skill_dir=str(self.skill_dir),
                skill_path=str(self.skill_dir / "SKILL.md"),
                content="Body",
                full_text="Selected Skill\ndesc\nBody",
                sha256="abc",
            )
            return [SkillSelection(skill=doc, score=1.0)]

    # Base skills_dir must still contain a Trace2Skill-compatible skill for constructor discovery.
    skills_dir = tmp_path / "skills_root"
    (skills_dir / "xlsx").mkdir(parents=True)
    (skills_dir / "xlsx" / "SKILL.md").write_text("---\nname: base\ndescription: base\n---\nbase\n", encoding="utf-8")

    source_skill = tmp_path / "source_skill"
    (source_skill / "references").mkdir(parents=True)
    (source_skill / "scripts").mkdir()
    (source_skill / "SKILL.md").write_text("---\nname: selected\ndescription: selected\n---\nRead references/index.md when needed.\n", encoding="utf-8")
    (source_skill / "references" / "index.md").write_text("# reference index\n", encoding="utf-8")
    (source_skill / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")

    selection_log = tmp_path / "raw" / "skill_selection_records.jsonl"
    monkeypatch.setenv("DYNAMIX_SKILLBANK_TOP_K", "1")
    monkeypatch.setenv("DYNAMIX_SKILL_SELECTION_LOG", str(selection_log))
    agent = CLISkillPreloadedAgent(DummyClient(), skills_dir=str(skills_dir), verbose=False)
    agent._skillbank_selector = DummySelector(source_skill)
    class Context:
        instance_id = "task-1"
        instruction = "lookup values"
        instruction_type = "Cell-Level Manipulation"
        answer_position = "A1"
    agent._select_skills_for_context(Context())
    assert agent._active_skill_selection
    selected = agent._active_skill_selection[0]
    assert selected["skill_dir"] == str(source_skill)
    assert selected["skill_path"] == str(source_skill / "SKILL.md")
    assert not (skills_dir / ".dynamix_selected_skills").exists()
    prompt = agent.get_system_template()
    assert str(source_skill) in prompt
    assert str(source_skill / "references") in prompt
    assert str(source_skill / "scripts") in prompt
    record = json.loads(selection_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["instance_id"] == "task-1"
    assert record["selected_skill_ids"] == ["selected-skill"]
    assert record["selected_skill_dirs"] == [str(source_skill)]


def test_trace2skill_bash_can_read_absolute_skillbank_support_file(tmp_path):
    from spreadsheet_agent.tools import create_bash_tool
    skill_dir = tmp_path / "skillbank" / "skill_a"
    (skill_dir / "references").mkdir(parents=True)
    target = skill_dir / "references" / "index.md"
    target.write_text("# Support Index\nreadable content\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    bash = create_bash_tool(str(work), timeout=10)
    output = bash.execute(command=f"cat {target}")
    assert "readable content" in output

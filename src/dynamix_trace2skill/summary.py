from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dynamix_core.data_structures import ExperienceCardPatch, ExperienceCommunity, ExperienceItem, ITEM_KIND_EXPERIENCE_CARD, ITEM_KIND_TRAJECTORY
from .clients import EmbeddingClient, GenerationClient
from .tokenization import get_tokenizer, TokenizerUnavailable


@dataclass
class ClusterAnalystConfig:
    """Configuration for cluster-level Trace2Skill-style abstraction.

    The ExperienceCard schema is intentionally minimal.  The LLM must produce
    only: name, trigger, content, placement, confidence.  Community size and
    prompt budget are controlled before the analyst is called, never by member
    truncation inside this prompt.
    """

    prompt_style: str = "trace2skill_cluster_minimal_experience_v6"
    confidence_floor: float = 0.05
    trace2skill_analysis_dir: str | None = None
    tokenizer_model: str | None = None
    tokenizer_required: bool = True
    allow_regex_tokenizer_fallback: bool = False
    # None means: derive from hierarchy.summary_budget.effective_token_budget in pipeline.
    max_prompt_tokens: int | None = None
    prompt_token_report_path: str | None = None
    token_report: list[dict[str, Any]] = field(default_factory=list)

    # Layer-aware card cardinality policy. L0 raw trajectory communities may
    # yield multiple cards, but higher-level ExperienceCard communities should
    # compress to exactly one higher-level card.
    multi_card_max_level: int = 0
    max_cards_l0: int | None = None
    max_cards_higher: int = 1
    higher_level_mode: str = "single_abstraction"
    truncate_higher_level_extra_cards: bool = True


class ClusterAnalyst:
    """Trace2Skill analyst templates rewritten for trajectory clusters.

    We use Trace2Skill's success/error analysis prompts as behavioral
    references, but the output schema is deliberately small.  A community may
    yield one or more cards, each with:

        name, trigger, content, placement, confidence

    No support_mass, no long structured guidance arrays, and no implicit
    per-trace patch proposal.
    """

    def __init__(self, generation: GenerationClient, embedding: EmbeddingClient, config: ClusterAnalystConfig | None = None):
        self.generation = generation
        self.embedding = embedding
        self.config = config or ClusterAnalystConfig()
        self._templates = _load_trace2skill_templates(self.config.trace2skill_analysis_dir)

    async def summarize(self, community: ExperienceCommunity, members: Sequence[ExperienceItem], clustering: Any | None = None) -> list[ExperienceItem]:
        if _is_diagnostic_community(community):
            return []
        analyst_mode = _infer_analyst_mode(community, members, self.config)
        system_prompt = self._system_prompt(analyst_mode)
        prompt = self._build_prompt(community, members, analyst_mode)
        token_event = self._preflight_prompt_budget(community, system_prompt, prompt, len(members))
        payload = await self.generation.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            schema_name="MinimalClusterExperienceCards",
            debug_metadata=self._generation_debug_metadata(
                community,
                members,
                analyst_mode,
                "MinimalClusterExperienceCards",
                token_event,
            ),
        )
        cards_payload = _extract_cards_payload(payload)
        llm_returned_card_count = len(cards_payload)
        cards_payload, extra_cards_truncated = _enforce_cardinality_policy(cards_payload, analyst_mode, self.config)
        items: list[ExperienceItem] = []
        rendered_texts: list[str] = []
        normalized_cards: list[dict[str, Any]] = []
        for card_index, card in enumerate(cards_payload, start=1):
            name = _required_string(card, "name")
            trigger = _required_string(card, "trigger")
            content = _required_string(card, "content")
            confidence = max(float(card.get("confidence", 0.5)), self.config.confidence_floor)
            placement = _normalize_placement(_required_mapping(card, "placement"), name=name)
            normalized = {
                "name": name,
                "trigger": trigger,
                "content": content,
                "confidence": confidence,
                "placement": placement,
                "card_index": card_index,
            }
            normalized_cards.append(normalized)
            rendered_texts.append(_render_card_text(normalized))

        if not normalized_cards:
            return []

        embeddings = await self.embedding.embed_texts(rendered_texts, cache_namespace="experience_card")
        total_cards = len(normalized_cards)
        for normalized, text, embedding in zip(normalized_cards, rendered_texts, embeddings):
            metadata = {
                "name": normalized["name"],
                "trigger": normalized["trigger"],
                "content": normalized["content"],
                "confidence": normalized["confidence"],
                "placement": normalized["placement"],
                "source_community_id": community.community_id,
                "source_member_count": len(members),
                "source_outcome_mode": community.outcome_mode,
                "cluster_level_analyst": True,
                "minimal_experience_schema": True,
                "multi_card_experience_schema": True,
                "source_card_index": normalized["card_index"],
                "source_generated_card_count": total_cards,
                "llm_returned_card_count": llm_returned_card_count,
                "analyst_mode": analyst_mode,
                "higher_level_single_card_enforced": analyst_mode == "experience_abstractor",
                "higher_level_extra_cards_truncated": extra_cards_truncated,
                "trace2skill_prompt_style_adapted": True,
                "trace2skill_template_inheritance": True,
                "iterative_rca_loop": False,
                "member_truncation_applied": False,
                "analysis_token_count": _safe_token_count(text, self.config.tokenizer_model, self.config.allow_regex_tokenizer_fallback),
            }
            item_id = f"E{community.level + 1}_{_short_hash(community.community_id + ':' + str(normalized['card_index']) + ':' + text)}"
            items.append(
                ExperienceItem(
                    item_id=item_id,
                    level=community.level + 1,
                    kind=ITEM_KIND_EXPERIENCE_CARD,
                    text=text,
                    embedding=embedding,
                    generated_from_community_ids=(community.community_id,),
                    metadata=metadata,
                )
            )
        return items

    async def summarize_dynamic_update(
        self,
        community: ExperienceCommunity,
        members: Sequence[ExperienceItem],
        previous_generated_experiences: Sequence[dict[str, Any]],
    ) -> list[ExperienceCardPatch]:
        if _is_diagnostic_community(community):
            return []
        analyst_mode = _infer_analyst_mode(community, members, self.config)
        system_prompt = self._dynamic_system_prompt(analyst_mode)
        prompt = self._build_dynamic_update_prompt(community, members, previous_generated_experiences, analyst_mode)
        token_event = self._preflight_prompt_budget(community, system_prompt, prompt, len(members))
        previous_by_id = {str(card.get("item_id")): dict(card) for card in previous_generated_experiences if card.get("item_id")}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        max_attempts = 3 if analyst_mode == "experience_abstractor" else 1
        last_error: Exception | None = None
        rendered_texts: list[str] = []
        patch_specs: list[tuple[str, str, dict[str, Any], str]] = []
        extra_patches_truncated = 0
        for attempt in range(1, max_attempts + 1):
            try:
                attempt_token_event = token_event
                if attempt > 1:
                    attempt_token_event = self._preflight_messages_budget(
                        community,
                        messages,
                        len(members),
                        extra_metadata={
                            "event": "dynamic_schema_repair_prompt",
                            "dynamic_schema_attempt": attempt,
                        },
                    )
                payload = await self.generation.chat_json(
                    messages,
                    schema_name="DynamicExperienceCardPatchSet",
                    retries=0 if analyst_mode == "experience_abstractor" else 2,
                    debug_metadata={
                        **self._generation_debug_metadata(
                            community,
                            members,
                            analyst_mode,
                            "DynamicExperienceCardPatchSet",
                            attempt_token_event,
                        ),
                        "dynamic_schema_attempt": attempt,
                    },
                )
                rendered_texts, patch_specs, extra_patches_truncated = self._dynamic_patch_specs_from_payload(
                    payload,
                    community,
                    analyst_mode,
                    previous_by_id,
                )
                last_error = None
                break
            except (KeyError, TypeError, ValueError) as exc:
                if analyst_mode != "experience_abstractor":
                    raise
                last_error = exc
                status = "retry" if attempt < max_attempts else "ignored_invalid_llm_output"
                self._record_dynamic_schema_repair_event(
                    community,
                    analyst_mode,
                    previous_by_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status=status,
                    error=exc,
                )
                if attempt >= max_attempts:
                    return []
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": _dynamic_update_repair_prompt(previous_by_id),
                    },
                ]
        if last_error is not None:
            return []

        if not patch_specs:
            return []

        patches: list[ExperienceCardPatch] = []
        embeddings = await self.embedding.embed_texts(rendered_texts, cache_namespace="experience_card")
        for (operation, item_id, normalized, rendered), embedding in zip(patch_specs, embeddings):
            patches.append(ExperienceCardPatch(
                operation="update" if operation == "update" else "add",
                item_id=item_id,
                text=rendered,
                embedding=embedding,
                metadata=self._dynamic_card_metadata(
                    normalized,
                    community,
                    members,
                    analyst_mode,
                    operation,
                    extra_patches_truncated=extra_patches_truncated,
                ),
            ))
        return patches

    def _dynamic_patch_specs_from_payload(
        self,
        payload: dict[str, Any],
        community: ExperienceCommunity,
        analyst_mode: str,
        previous_by_id: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[tuple[str, str, dict[str, Any], str]], int]:
        if analyst_mode == "experience_abstractor":
            updates_payload, extra_patches_truncated = _extract_dynamic_update_only_payload(payload)
            new_cards_payload: list[dict[str, Any]] = []
        else:
            updates_payload, new_cards_payload = _extract_dynamic_patch_payload(payload)
            extra_patches_truncated = 0
        updates_payload, new_cards_payload, extra_patches_truncated = _enforce_dynamic_patch_cardinality(
            updates_payload,
            new_cards_payload,
            analyst_mode,
            self.config,
            extra_patches_truncated=extra_patches_truncated,
        )

        rendered_texts: list[str] = []
        patch_specs: list[tuple[str, str, dict[str, Any], str]] = []
        seen_updates: set[str] = set()
        for update in updates_payload:
            item_id = _required_string(update, "item_id")
            if item_id not in previous_by_id:
                raise ValueError(f"dynamic update referenced unknown previous ExperienceCard item_id={item_id!r}")
            if item_id in seen_updates:
                raise ValueError(f"dynamic update referenced duplicate ExperienceCard item_id={item_id!r}")
            seen_updates.add(item_id)
            normalized = self._normalize_dynamic_card(update)
            rendered = _render_card_text(normalized)
            rendered_texts.append(rendered)
            patch_specs.append(("update", item_id, normalized, rendered))

        for index, card in enumerate(new_cards_payload, start=1):
            normalized = self._normalize_dynamic_card(card)
            rendered = _render_card_text(normalized)
            item_id = f"E{community.level + 1}_D{_short_hash(community.community_id + ':' + str(community.version) + ':' + str(index) + ':' + rendered)}"
            rendered_texts.append(rendered)
            patch_specs.append(("add", item_id, normalized, rendered))
        return rendered_texts, patch_specs, extra_patches_truncated

    def _record_dynamic_schema_repair_event(
        self,
        community: ExperienceCommunity,
        analyst_mode: str,
        previous_by_id: dict[str, dict[str, Any]],
        *,
        attempt: int,
        max_attempts: int,
        status: str,
        error: Exception,
    ) -> None:
        self.config.token_report.append({
            "event": "dynamic_schema_repair",
            "community_id": community.community_id,
            "community_level": community.level,
            "analyst_mode": analyst_mode,
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
            "action": "retry_with_update_only_prompt" if status == "retry" else "skip_invalid_dynamic_update",
            "previous_item_ids": sorted(previous_by_id),
        })

    def _normalize_dynamic_card(
        self,
        card: dict[str, Any],
    ) -> dict[str, Any]:
        name = _required_string(card, "name")
        trigger = _required_string(card, "trigger")
        content = _required_string(card, "content")
        confidence = max(float(card.get("confidence", 0.5)), self.config.confidence_floor)
        placement = _normalize_placement(_required_mapping(card, "placement"), name=name)
        return {
            "name": name,
            "trigger": trigger,
            "content": content,
            "confidence": confidence,
            "placement": placement,
            "analysis_token_count": _safe_token_count(
                _render_card_text({"name": name, "trigger": trigger, "content": content}),
                self.config.tokenizer_model,
                self.config.allow_regex_tokenizer_fallback,
            ),
        }

    def _dynamic_card_metadata(
        self,
        normalized: dict[str, Any],
        community: ExperienceCommunity,
        members: Sequence[ExperienceItem],
        analyst_mode: str,
        operation: str,
        *,
        extra_patches_truncated: int = 0,
    ) -> dict[str, Any]:
        return {
            "name": normalized["name"],
            "trigger": normalized["trigger"],
            "content": normalized["content"],
            "confidence": normalized["confidence"],
            "placement": normalized["placement"],
            "source_community_id": community.community_id,
            "source_member_count": len(members),
            "source_outcome_mode": community.outcome_mode,
            "cluster_level_analyst": True,
            "minimal_experience_schema": True,
            "dynamic_experience_patch_schema": True,
            "dynamic_patch_operation": operation,
            "analyst_mode": analyst_mode,
            "higher_level_single_card_enforced": analyst_mode == "experience_abstractor",
            "higher_level_extra_cards_truncated": int(extra_patches_truncated),
            "trace2skill_prompt_style_adapted": True,
            "trace2skill_template_inheritance": True,
            "iterative_rca_loop": False,
            "member_truncation_applied": False,
            "analysis_token_count": normalized["analysis_token_count"],
        }

    def _system_prompt(self, analyst_mode: str) -> str:
        if analyst_mode == "raw_extractor":
            mission = "Analyze one raw trajectory cluster and produce one or more reusable ExperienceCards."
            rewrite_policy = """- If the cluster contains successful trajectories, distill reusable success lessons into cards.
- If the cluster contains failed trajectories, distill reusable failure-prevention lessons into cards.
- If the cluster is mixed, analyze success and failure evidence together, and split the output into multiple cards when the raw trajectory cluster supports multiple distinct reusable lessons.
- If the cluster supports only one reusable lesson, return one card."""
        else:
            mission = "Analyze one cluster of lower-level ExperienceCards and produce exactly one higher-level ExperienceCard."
            rewrite_policy = """- The input members are already lower-level ExperienceCards, not raw trajectories.
- Your job is abstraction, not further extraction: identify the shared higher-level principle behind the lower-level cards.
- Return exactly one card. Do not split the cluster into multiple cards.
- If the community is somewhat mixed, write the broadest valid abstraction and lower the confidence instead of producing multiple cards."""
        return f"""# Role
You are an expert Trace2Skill-style AI agent trajectory analyst for spreadsheet manipulation tasks.

# Mission
{mission}

# Trace2Skill template inheritance
You must follow the evidence discipline of the Trace2Skill success/error analysis prompts below, but adapt their evidence unit from a single trajectory to a trajectory cluster.

<adapted_success_system_template>
{self._templates.get('success_system_llm', '')}
</adapted_success_system_template>

<adapted_error_system_template>
{self._templates.get('error_system_llm', '')}
</adapted_error_system_template>

# Cluster-level rewrite
{rewrite_policy}
- v1 does not run Trace2Skill's iterative minimal-fix verifier loop, so do not claim verified root causes.

# Minimal output schema
Return one JSON object with a top-level cards list. For higher-level abstraction mode, that list must contain exactly one card. Each card must contain only these fields:
- name: short name of the experience.
- trigger: when a future agent should use this experience.
- content: the concrete reusable experience/guidance. This is the main body.
- placement: where this experience should be exported. Keep it minimal: target plus optional reference_kind.
- confidence: float in (0, 1].

Do not output support_mass. Do not output extra structured fields such as shared_patterns, success_motifs, anti_patterns, or patch_hints. Put the useful guidance in content. Do not duplicate semantically identical cards.
"""

    def _dynamic_system_prompt(self, analyst_mode: str) -> str:
        if analyst_mode == "experience_abstractor":
            mission = "Update an existing higher-level ExperienceCard abstraction after its lower-level members changed."
            rewrite_policy = """- The input members are lower-level ExperienceCards, not raw trajectories.
- Your job is to revise existing higher-level abstractions by explicit item_id.
- Do not create new cards at this level. If the old abstraction remains valid, improve its name, trigger, content, placement, or confidence.
- If the community is somewhat mixed, write the broadest valid update and lower the confidence."""
            output_policy = """Return one JSON object with exactly one top-level field:
- updates: a list of revised existing cards. Each update must include item_id, name, trigger, content, placement, confidence.

Every updates[].item_id must be copied exactly from previous_generated_experiences."""
        else:
            mission = "Update the reusable ExperienceCards for one raw trajectory cluster after new trajectories were inserted."
            rewrite_policy = """- If the updated cluster contains successful trajectories, distill reusable success lessons.
- If the updated cluster contains failed trajectories, distill reusable failure-prevention lessons.
- If the updated cluster is mixed, analyze success and failure evidence together.
- Use updates only for revising the same old card by explicit item_id.
- Use new_cards for newly discovered independent raw-cluster lessons."""
            output_policy = """Return one JSON object with exactly these top-level fields:
- updates: a list of revised existing cards. Each update must include item_id and may only use an item_id from previous_generated_experiences.
- new_cards: a list of newly discovered independent cards. New cards must not include item_id."""
        return f"""
# Role
You are an expert Trace2Skill-style AI agent trajectory analyst for spreadsheet manipulation tasks.

# Mission
{mission}

# Trace2Skill template inheritance
You must follow the evidence discipline of the Trace2Skill success/error analysis prompts below, but adapt their evidence unit to the updated community.

<adapted_success_system_template>
{self._templates.get('success_system_llm', '')}
</adapted_success_system_template>

<adapted_error_system_template>
{self._templates.get('error_system_llm', '')}
</adapted_error_system_template>

# Dynamic community rewrite
{rewrite_policy}
- v1 does not run Trace2Skill's iterative minimal-fix verifier loop, so do not claim verified root causes.

# Dynamic output schema
{output_policy}

Do not match cards by list position. Do not overwrite an old card just because a revised card was generated.
Use updates only when you are revising the same reusable experience already represented by that exact old item_id.
Omit unchanged old cards. Do not delete old cards. Do not output support_mass.
Return valid JSON only.
"""

    def _preflight_prompt_budget(
        self,
        community: ExperienceCommunity,
        system_prompt: str,
        user_prompt: str,
        member_count: int,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.config.max_prompt_tokens is None or int(self.config.max_prompt_tokens) <= 0:
            return None
        text = system_prompt + "\n" + user_prompt
        try:
            tokenizer = get_tokenizer(self.config.tokenizer_model, allow_regex_fallback=self.config.allow_regex_tokenizer_fallback)
            token_count = tokenizer.count(text)
            tokenizer_name = tokenizer.name
        except Exception as exc:
            if self.config.tokenizer_required:
                raise TokenizerUnavailable(f"cluster analyst tokenizer unavailable for prompt preflight: {exc}") from exc
            token_count = max(1, (len(text) + 3) // 4)
            tokenizer_name = "char_estimate_fallback"
        event = {
            "community_id": community.community_id,
            "level": community.level,
            "member_count": member_count,
            "prompt_tokens": token_count,
            "max_prompt_tokens": int(self.config.max_prompt_tokens),
            "tokenizer": tokenizer_name,
            "over_budget": token_count > int(self.config.max_prompt_tokens),
        }
        if extra_metadata:
            event.update(extra_metadata)
        self.config.token_report.append(event)
        if self.config.prompt_token_report_path:
            path = Path(self.config.prompt_token_report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"events": self.config.token_report}, ensure_ascii=False, indent=2), encoding="utf-8")
        if event["over_budget"]:
            raise ValueError(
                f"Cluster analyst prompt exceeds token budget for community={community.community_id!r}: "
                f"prompt_tokens={token_count}, max_prompt_tokens={self.config.max_prompt_tokens}. "
                "The hierarchy summary_budget/token counts must force a finer split before analyst invocation."
            )
        return event

    def _preflight_messages_budget(
        self,
        community: ExperienceCommunity,
        messages: Sequence[dict[str, str]],
        member_count: int,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        system_prompt = "\n".join(message.get("content", "") for message in messages if message.get("role") == "system")
        user_prompt = "\n\n".join(
            f"[{message.get('role', 'user')}]\n{message.get('content', '')}"
            for message in messages
            if message.get("role") != "system"
        )
        return self._preflight_prompt_budget(
            community,
            system_prompt,
            user_prompt,
            member_count,
            extra_metadata=extra_metadata,
        )

    def save_prompt_token_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"events": self.config.token_report}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generation_debug_metadata(
        self,
        community: ExperienceCommunity,
        members: Sequence[ExperienceItem],
        analyst_mode: str,
        schema_name: str,
        token_event: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_name": schema_name,
            "community_id": community.community_id,
            "level": community.level,
            "analyst_mode": analyst_mode,
            "member_count": len(members),
            "member_item_ids": [item.item_id for item in members],
            "community": {
                "support_mass": community.support_mass,
                "outcome_mode": community.outcome_mode,
                "success_count": community.success_count,
                "failure_count": community.failure_count,
                "clustering_method": community.clustering_method,
                "metadata": dict(community.metadata or {}),
            },
            "prompt_token_event": dict(token_event or {}),
            "members": [_member_debug_metadata(item) for item in members],
        }

    def _build_prompt(self, community: ExperienceCommunity, members: Sequence[ExperienceItem], analyst_mode: str) -> str:
        member_payloads = []
        for item in members:
            member_payloads.append({
                "item_id": item.item_id,
                "membership_weight": community.member_weights.get(item.item_id),
                "success": item.metadata.get("success"),
                "analysis_bundle": item.metadata.get("analysis_bundle", item.text),
            })
        return json.dumps({
            "instruction": _mode_instruction(analyst_mode),
            "analyst_mode": analyst_mode,
            "cardinality_policy": {
                "raw_extractor": "L0 raw trajectory communities may return one or more cards.",
                "experience_abstractor": "L1+ ExperienceCard communities must return exactly one higher-level card.",
            },
            "template_user_prompt_adaptation": {
                "success_user_template": self._templates.get("success_user_llm", ""),
                "error_user_template": self._templates.get("error_user_llm", ""),
                "replace_agent_log_with": "cluster_member_analysis_bundles",
            },
            "hard_constraints": [
                "Use all provided members. Do not ignore later members or silently truncate the cluster.",
                "Do not output support_mass.",
                "Return a top-level cards list. Do not output fields except cards and each card's name, trigger, content, placement, confidence.",
                "Do not output file_slug or rationale; filenames are assigned automatically by the exporter.",
                "Do not copy absolute paths, output_path, xlsx filenames, raw long code blocks, XML/tool tags, or debug logs into reusable guidance.",
                "Never mention ground truth, gold answers, hidden labels, or verifier internals as something the target agent can access.",
                "Return valid JSON only.",
            ],
            "community": community.to_dict(),
            "members": member_payloads,
            "output_schema": {
                "cards": [
                    {
                        "name": "string: short reusable experience name",
                        "trigger": "string: when a future task should use this experience",
                        "content": "string: concrete reusable guidance / lesson / procedure",
                        "placement": {
                            "target": "skill_md | reference | script",
                            "reference_kind": "procedure | example | edge_case | note"
                        },
                        "confidence": "float in (0,1]"
                    }
                ]
            },
            "placement_rules": [
                "Use target=skill_md for concise high-support guidance that should be preloaded.",
                "Use target=reference for detailed examples, edge cases, narrow procedures, or lower-priority details.",
                "Use target=script only when content is a complete deterministic helper script; otherwise use reference.",
                "Do not put raw trajectory text, local paths, output paths, or ground-truth-specific content into any placement."
            ]
        }, ensure_ascii=False, indent=2)

    def _build_dynamic_update_prompt(
        self,
        community: ExperienceCommunity,
        members: Sequence[ExperienceItem],
        previous_generated_experiences: Sequence[dict[str, Any]],
        analyst_mode: str,
    ) -> str:
        member_payloads = []
        for item in members:
            member_payloads.append({
                "item_id": item.item_id,
                "membership_weight": community.member_weights.get(item.item_id),
                "success": item.metadata.get("success"),
                "analysis_bundle": item.metadata.get("analysis_bundle", item.text),
            })
        previous_payloads = []
        for card in previous_generated_experiences:
            metadata = dict(card.get("metadata", {}) or {})
            previous_payloads.append({
                "item_id": card.get("item_id"),
                "name": metadata.get("name"),
                "trigger": metadata.get("trigger"),
                "content": metadata.get("content"),
                "confidence": metadata.get("confidence"),
                "support_mass": card.get("support_mass"),
                "text": card.get("text"),
            })
        higher_level = analyst_mode == "experience_abstractor"
        if higher_level:
            instruction = (
                "Analyze the updated higher-level community after dynamic insertion. "
                "Revise existing ExperienceCards only by explicit previous item_id."
            )
            dynamic_patch_policy = {
                "updates": "Only revise old cards by explicitly naming a previous item_id.",
                "unchanged": "Omit unchanged previous cards.",
                "forbidden": "Never infer update targets from output order or confidence rank.",
            }
            hard_constraints = [
                "Use all provided members. Do not ignore later members or silently truncate the cluster.",
                "Do not output support_mass.",
                "Every updates[].item_id must be copied exactly from previous_generated_experiences.",
                "Return a top-level JSON object with only updates.",
                "Each update must contain item_id, name, trigger, content, placement, confidence.",
                "Do not copy absolute paths, output_path, xlsx filenames, raw long code blocks, XML/tool tags, or debug logs into reusable guidance.",
                "Never mention ground truth, gold answers, hidden labels, or verifier internals as something the target agent can access.",
                "Return valid JSON only.",
            ]
            output_schema = {
                "updates": [
                    {
                        "item_id": "string: must equal one previous_generated_experiences item_id",
                        "name": "string",
                        "trigger": "string",
                        "content": "string",
                        "placement": {
                            "target": "skill_md | reference | script",
                            "reference_kind": "procedure | example | edge_case | note"
                        },
                        "confidence": "float in (0,1]"
                    }
                ],
            }
        else:
            instruction = (
                "Analyze the updated raw trajectory community after dynamic insertion. "
                "Output explicit old-card updates by item_id and independent new cards separately."
            )
            dynamic_patch_policy = {
                "updates": "Only revise old cards by explicitly naming a previous item_id.",
                "new_cards": "Every newly discovered independent lesson goes here and receives a new item_id automatically.",
                "unchanged": "Omit unchanged previous cards.",
                "forbidden": "Never infer update targets from output order or confidence rank.",
            }
            hard_constraints = [
                "Use all provided members. Do not ignore later members or silently truncate the cluster.",
                "Do not output support_mass.",
                "Do not output item_id inside new_cards.",
                "Every updates[].item_id must be copied exactly from previous_generated_experiences.",
                "Return a top-level JSON object with only updates and new_cards.",
                "Each updated or new card must contain name, trigger, content, placement, confidence.",
                "Do not copy absolute paths, output_path, xlsx filenames, raw long code blocks, XML/tool tags, or debug logs into reusable guidance.",
                "Never mention ground truth, gold answers, hidden labels, or verifier internals as something the target agent can access.",
                "Return valid JSON only.",
            ]
            output_schema = {
                "updates": [
                    {
                        "item_id": "string: must equal one previous_generated_experiences item_id",
                        "name": "string",
                        "trigger": "string",
                        "content": "string",
                        "placement": {
                            "target": "skill_md | reference | script",
                            "reference_kind": "procedure | example | edge_case | note"
                        },
                        "confidence": "float in (0,1]"
                    }
                ],
                "new_cards": [
                    {
                        "name": "string",
                        "trigger": "string",
                        "content": "string",
                        "placement": {
                            "target": "skill_md | reference | script",
                            "reference_kind": "procedure | example | edge_case | note"
                        },
                        "confidence": "float in (0,1]"
                    }
                ],
            }
        return json.dumps({
            "instruction": instruction,
            "analyst_mode": analyst_mode,
            "dynamic_patch_policy": dynamic_patch_policy,
            "template_user_prompt_adaptation": {
                "success_user_template": self._templates.get("success_user_llm", ""),
                "error_user_template": self._templates.get("error_user_llm", ""),
                "replace_agent_log_with": "cluster_member_analysis_bundles",
            },
            "hard_constraints": hard_constraints,
            "community": community.to_dict(),
            "members": member_payloads,
            "previous_generated_experiences": previous_payloads,
            "output_schema": output_schema,
        }, ensure_ascii=False, indent=2)



def _infer_analyst_mode(community: ExperienceCommunity, members: Sequence[ExperienceItem], config: ClusterAnalystConfig) -> str:
    """Return the cardinality mode for this community.

    L0 raw trajectory clusters are extraction units and may produce multiple
    bottom-level cards. Once the members are ExperienceCards, each community is
    an abstraction unit and should compress to a single higher-level card.
    """
    if int(community.level) <= int(config.multi_card_max_level) and any(item.kind == ITEM_KIND_TRAJECTORY for item in members):
        return "raw_extractor"
    return "experience_abstractor"


def _mode_instruction(analyst_mode: str) -> str:
    if analyst_mode == "raw_extractor":
        return (
            "Analyze this raw trajectory cluster once and produce one or more reusable ExperienceCards. "
            "Each card should capture one distinct reusable lesson supported by the raw trajectories. "
            "Follow the Trace2Skill success/error analysis discipline inherited in the system prompt, "
            "but output the minimal cards-list schema only."
        )
    return (
        "Analyze this cluster of lower-level ExperienceCards once and produce exactly one higher-level ExperienceCard. "
        "Do not split the cluster into multiple cards. Your task is to identify the shared higher-level principle "
        "behind the lower-level cards. Follow the Trace2Skill evidence discipline inherited in the system prompt, "
        "but output the minimal cards-list schema only."
    )


def _enforce_cardinality_policy(
    cards: list[dict[str, Any]],
    analyst_mode: str,
    config: ClusterAnalystConfig,
) -> tuple[list[dict[str, Any]], int]:
    if analyst_mode == "raw_extractor":
        max_cards = config.max_cards_l0
        if max_cards is None or int(max_cards) <= 0:
            return cards, 0
        limit = int(max_cards)
    else:
        limit = max(1, int(config.max_cards_higher))

    if len(cards) <= limit:
        return cards, 0
    if analyst_mode == "experience_abstractor" and not config.truncate_higher_level_extra_cards:
        raise ValueError(
            f"higher-level ExperienceCard abstraction must return at most {limit} card(s), got {len(cards)}"
        )
    return cards[:limit], len(cards) - limit


def _extract_cards_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards = payload.get("cards")
    if cards is None:
        # Backward compatibility for older single-card responses.
        cards = [payload]
    if not isinstance(cards, list):
        raise ValueError("LLM ExperienceCard output must include a cards list")
    result: list[dict[str, Any]] = []
    for index, card in enumerate(cards, start=1):
        if not isinstance(card, dict):
            raise ValueError(f"cards[{index}] must be an object")
        result.append(dict(card))
    return result


def _extract_dynamic_patch_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "updates" not in payload and "new_cards" not in payload:
        # Backward-compatible LLM/mock shapes are safe only as additions. They
        # must never be position-matched onto existing cards.
        return [], _extract_cards_payload(payload)
    updates = payload.get("updates", [])
    new_cards = payload.get("new_cards", [])
    if not isinstance(updates, list):
        raise ValueError("dynamic ExperienceCard output field 'updates' must be a list")
    if not isinstance(new_cards, list):
        raise ValueError("dynamic ExperienceCard output field 'new_cards' must be a list")
    update_items: list[dict[str, Any]] = []
    for index, card in enumerate(updates, start=1):
        if not isinstance(card, dict):
            raise ValueError(f"updates[{index}] must be an object")
        update_items.append(dict(card))
    new_items: list[dict[str, Any]] = []
    for index, card in enumerate(new_cards, start=1):
        if not isinstance(card, dict):
            raise ValueError(f"new_cards[{index}] must be an object")
        if "item_id" in card:
            raise ValueError("new_cards must not include item_id; item ids are assigned by DynaMix")
        new_items.append(dict(card))
    return update_items, new_items


def _extract_dynamic_update_only_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    unexpected_fields = sorted(set(payload) - {"updates"})
    if unexpected_fields:
        joined = ", ".join(repr(field) for field in unexpected_fields)
        raise ValueError(f"L1+ dynamic ExperienceCard output must include only 'updates'; unexpected field(s): {joined}")
    if "updates" not in payload:
        raise ValueError("L1+ dynamic ExperienceCard output must include an 'updates' list")
    updates = payload["updates"]
    if not isinstance(updates, list):
        raise ValueError("dynamic ExperienceCard output field 'updates' must be a list")
    update_items: list[dict[str, Any]] = []
    for index, card in enumerate(updates, start=1):
        if not isinstance(card, dict):
            raise ValueError(f"updates[{index}] must be an object")
        update_items.append(dict(card))
    return update_items, 0


def _dynamic_update_repair_prompt(previous_by_id: dict[str, dict[str, Any]]) -> str:
    return json.dumps({
        "repair_instruction": (
            "Your previous JSON did not satisfy the L1+ dynamic update schema. "
            "Return valid JSON using only the update-only schema below. "
            "If no existing ExperienceCard should change, return an empty updates list."
        ),
        "allowed_previous_item_ids": sorted(previous_by_id),
        "required_top_level_schema": {
            "updates": [
                {
                    "item_id": "must exactly equal one allowed_previous_item_ids value",
                    "name": "string",
                    "trigger": "string",
                    "content": "string",
                    "placement": {
                        "target": "skill_md | reference | script",
                        "reference_kind": "procedure | example | edge_case | note"
                    },
                    "confidence": "float in (0,1]"
                }
            ]
        },
        "hard_constraints": [
            "Use no top-level field except updates.",
            "Every update must revise an existing ExperienceCard by explicit item_id.",
            "Do not infer update targets from list order, rank, or confidence.",
            "Do not output support_mass.",
            "Return valid JSON only.",
        ],
    }, ensure_ascii=False, indent=2)


def _enforce_dynamic_patch_cardinality(
    updates: list[dict[str, Any]],
    new_cards: list[dict[str, Any]],
    analyst_mode: str,
    config: ClusterAnalystConfig,
    *,
    extra_patches_truncated: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if analyst_mode == "raw_extractor":
        return updates, new_cards, extra_patches_truncated
    extra_patches_truncated += len(new_cards)
    new_cards = []
    limit = max(1, int(config.max_cards_higher))
    total = len(updates) + len(new_cards)
    if total <= limit:
        return updates, new_cards, extra_patches_truncated
    if not config.truncate_higher_level_extra_cards:
        raise ValueError(
            f"higher-level dynamic ExperienceCard update must return at most {limit} patch(es), got {total}"
        )
    kept_updates = updates[:limit]
    remaining = max(0, limit - len(kept_updates))
    kept_new_cards = new_cards[:remaining]
    kept_total = len(kept_updates) + len(kept_new_cards)
    return kept_updates, kept_new_cards, extra_patches_truncated + total - kept_total


def _is_diagnostic_community(community: ExperienceCommunity) -> bool:
    metadata = dict(community.metadata or {})
    return bool(metadata.get("llm_summary_skipped") or metadata.get("oversize_singleton"))


def _member_debug_metadata(item: ExperienceItem) -> dict[str, Any]:
    metadata = dict(item.metadata or {})
    record = metadata.get("record") if isinstance(metadata.get("record"), dict) else {}
    return {
        "item_id": item.item_id,
        "level": item.level,
        "kind": item.kind,
        "support_mass": item.support_mass,
        "success": metadata.get("success", record.get("success")),
        "task_id": metadata.get("task_id", record.get("task_id")),
        "trajectory_id": metadata.get("trajectory_id", record.get("trajectory_id")),
        "instruction_type": metadata.get("instruction_type", record.get("instruction_type")),
        "source_community_id": metadata.get("source_community_id"),
        "name": metadata.get("name"),
        "confidence": metadata.get("confidence"),
        "analysis_token_count": metadata.get("analysis_token_count"),
    }


def _normalize_placement(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    target = str(value.get("target", "")).strip().lower()
    aliases = {"main": "skill_md", "skill": "skill_md", "skill.md": "skill_md", "references": "reference", "ref": "reference", "code": "script", "scripts": "script"}
    target = aliases.get(target, target)
    if target not in {"skill_md", "reference", "script"}:
        raise ValueError("placement.target must be one of: skill_md, reference, script")
    # Minimal user-visible schema: LLM decides target and optional reference_kind only.
    # Filenames are derived deterministically by the exporter from experience name + node id.
    return {
        "target": target,
        "reference_kind": str(value.get("reference_kind", "note")).strip() or "note",
    }


def _load_trace2skill_templates(analysis_dir: str | None) -> dict[str, str]:
    if analysis_dir:
        root = Path(analysis_dir)
    else:
        root = Path(__file__).resolve().parents[2] / "analysis"
    names = {
        "success_system_llm": "success_analysis_system_llm.txt",
        "success_user_llm": "success_analysis_user_llm.txt",
        "error_system_llm": "error_analysis_system_llm.txt",
        "error_user_llm": "error_analysis_user_llm.txt",
    }
    out: dict[str, str] = {}
    for key, filename in names.items():
        path = root / filename
        try:
            out[key] = _adapt_trace2skill_template(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            out[key] = ""
    return out


def _adapt_trace2skill_template(text: str) -> str:
    """Rewrite obvious single-trajectory wording while preserving Trace2Skill structure."""
    replacements = {
        "single trajectory": "trajectory cluster",
        "Single trajectory": "Trajectory cluster",
        "one trajectory": "one trajectory cluster",
        "One trajectory": "One trajectory cluster",
        "the trajectory": "the trajectory cluster",
        "The trajectory": "The trajectory cluster",
        "this trajectory": "this trajectory cluster",
        "This trajectory": "This trajectory cluster",
        "agent trajectory": "agent trajectory cluster",
        "Agent trajectory": "Agent trajectory cluster",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _safe_token_count(text: str, tokenizer_model: str | None, allow_fallback: bool) -> int:
    try:
        return get_tokenizer(tokenizer_model, allow_regex_fallback=allow_fallback).count(text)
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _render_card_text(payload: dict[str, Any]) -> str:
    name = _required_string(payload, "name")
    trigger = _required_string(payload, "trigger")
    content = _required_string(payload, "content")
    lines = [f"# {name}", ""]
    if trigger:
        lines.extend(["## Trigger", trigger, ""])
    lines.extend(["## Content", content.strip() or "No detailed reusable experience content was provided.", ""])
    return "\n".join(lines).strip() + "\n"


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM ExperienceCard output must include non-empty string field: {key}")
    return value.strip()


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"LLM ExperienceCard output must include object field: {key}")
    return dict(value)


def _safe_token_list_count(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    return len([v for v in values if str(v).strip()])


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _slugify(text: str, max_len: int = 72) -> str:
    text = text.lower()
    text = "".join(ch if ch.isalnum() else "-" for ch in text)
    text = "-".join(part for part in text.split("-") if part)
    return (text[:max_len].strip("-") or "experience")

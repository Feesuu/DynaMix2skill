from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamix_core.balanced_metric_tree import (
    BalancedInsertionResult,
    BalancedMetricTreeState,
)
from dynamix_core.certified_otd import ExperienceAtom
from dynamix_core.skill_capsules import (
    SkillCapsule,
    SkillCapsuleRegistry,
    normalized_capsule_text,
    validate_capsule_candidate,
)

from .certified_otd_pipeline import (
    CertifiedOtdConfig,
    ExperienceAtomAnalyst,
    _atom_protocol_fingerprint,
    _load_atom_cache,
    _prepare_otd_analyst_config,
    _records_fingerprint,
    _require_complete_atom_identity,
    _require_prompt_budget,
    _require_unique_record_ids,
    _resolve_skill_output_dir,
    _tokenizer_for_config,
    _unsafe_reusable_text_reasons,
    _write_vector_cache_manifest,
    _write_jsonl,
)
from .clients import (
    EmbeddingClient,
    GenerationClient,
    validate_embedding_cache_manifest,
)
from .skillbank import (
    retrieved_experience_preamble,
)

__all__ = [
    "EvidenceBalancedSkillConfig",
    "build_evidence_balanced_dynamic_tree_from_records",
    "build_evidence_balanced_tree_from_records",
]


CAPSULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "promote": {"type": "boolean"},
        "name": {"type": "string", "maxLength": 256},
        "trigger": {"type": "string", "maxLength": 1200},
        "content": {"type": "string", "maxLength": 3600},
        "scope": {"type": "string", "maxLength": 1200},
        "verification": {"type": "string", "maxLength": 1200},
        "failure_modes": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 800},
            "maxItems": 8,
        },
        "rationale": {"type": "string", "maxLength": 1200},
    },
    "required": [
        "promote",
        "name",
        "trigger",
        "content",
        "scope",
        "verification",
        "failure_modes",
        "rationale",
    ],
    "additionalProperties": False,
}

PARENT_PROMOTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported_by_multiple_children": {"type": "boolean"},
        "adds_cross_child_abstraction": {"type": "boolean"},
        "copies_example_specific_content": {"type": "boolean"},
        "accept": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": [
        "supported_by_multiple_children",
        "adds_cross_child_abstraction",
        "copies_example_specific_content",
        "accept",
        "reason",
    ],
    "additionalProperties": False,
}

NEGATIVE_EVIDENCE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uses_negative_for_root_cause_boundary_or_guardrail": {
            "type": "boolean"
        },
        "recommends_observed_failed_action": {"type": "boolean"},
        "copies_example_specific_content": {"type": "boolean"},
        "accept": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": [
        "uses_negative_for_root_cause_boundary_or_guardrail",
        "recommends_observed_failed_action",
        "copies_example_specific_content",
        "accept",
        "reason",
    ],
    "additionalProperties": False,
}

_REJECTION_CLASSES = (
    "semantic_rejection",
    "validation_rejection",
    "runtime_generation_error",
    "prompt_budget_error",
)


def _generation_error_class(exc: Exception) -> str:
    if "prompt exceeds configured budget:" in str(exc).casefold():
        return "prompt_budget_error"
    return "runtime_generation_error"


@dataclass(frozen=True)
class EvidenceBalancedSkillConfig:
    max_entries: int = 8
    dual_view_lambda: float = 0.5
    atom_temperature: float = 0.0
    capsule_temperature: float = 0.0
    validator_temperature: float = 0.0
    retrieval_token_budget: int = 24000
    retrieval_token_unit: int = 128
    retrieval_exact_search_max_states: int = 250_000
    validation_mode: str = "structural_only"
    atom_cache_path: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "EvidenceBalancedSkillConfig":
        config = cls(**dict(payload or {}))
        if int(config.max_entries) < 4:
            raise ValueError("ebst.max_entries must be at least 4")
        if not 0.0 <= float(config.dual_view_lambda) <= 1.0:
            raise ValueError("ebst.dual_view_lambda must be in [0, 1]")
        if int(config.retrieval_token_budget) <= 0:
            raise ValueError("ebst.retrieval_token_budget must be positive")
        if int(config.retrieval_token_unit) <= 0:
            raise ValueError("ebst.retrieval_token_unit must be positive")
        if int(config.retrieval_exact_search_max_states) <= 0:
            raise ValueError(
                "ebst.retrieval_exact_search_max_states must be positive"
            )
        if config.validation_mode != "structural_only":
            raise ValueError(
                "only validation_mode='structural_only' is implemented; "
                "source-task replay must not be claimed before a benchmark "
                "replay adapter is connected"
            )
        return config


class SkillCapsuleAnalyst:
    def __init__(
        self,
        generation: GenerationClient,
        validator_generation: GenerationClient,
        *,
        tokenizer: Any,
        max_prompt_tokens: int,
        max_output_tokens: int | None,
        validation_mode: str,
    ) -> None:
        self.generation = generation
        self.validator_generation = validator_generation
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_output_tokens = max_output_tokens
        self.validation_mode = validation_mode

    async def generate_leaf(
        self,
        tree: BalancedMetricTreeState,
        registry: SkillCapsuleRegistry,
        tree_node_id: str,
    ) -> SkillCapsule | None:
        atom_ids = tree.descendant_atom_ids(tree_node_id)
        if len(atom_ids) < 2:
            return None
        atoms = [tree.atoms[atom_id] for atom_id in atom_ids]
        evidence = [
            {
                "atom_id": atom.atom_id,
                "trigger": atom.trigger,
                "scope": atom.scope,
                "decision": atom.decision,
                "invariant": atom.invariant,
                "verification": atom.verification,
                "failure_mode": atom.failure_mode,
                "reliability": atom.reliability,
            }
            for atom in atoms
        ]
        positive = [
            item
            for atom, item in zip(atoms, evidence)
            if atom.metadata.get("success") is True
        ]
        negative = [
            item
            for atom, item in zip(atoms, evidence)
            if atom.metadata.get("success") is False
        ]
        unknown = [
            item
            for atom, item in zip(atoms, evidence)
            if not isinstance(atom.metadata.get("success"), bool)
        ]
        return await self._generate(
            tree=tree,
            registry=registry,
            tree_node_id=tree_node_id,
            child_capsules=(),
            system_prompt=(
                "Create one transferable skill capsule from multiple "
                "trajectory-local evidence atoms. Preserve the common "
                "decision rule, applicability boundary, verification method, "
                "and failure conditions. Do not list or copy individual "
                "examples. Do not include task IDs, exact answers, file paths, "
                "URLs, coordinates, or accidental literal values. Reject the "
                "candidate when the evidence does not support one coherent "
                "cross-trajectory skill. Treat successful atoms as positive "
                "support. Treat failed atoms only as evidence for root causes, "
                "boundaries, guardrails, or corrected procedures; never turn "
                "an observed failed action into a recommendation."
            ),
            evidence_payload={
                "positive_evidence_atoms": positive,
                "negative_evidence_atoms": negative,
                "unknown_outcome_evidence_atoms": unknown,
            },
            analyst_mode="evidence_bucket_consolidation",
        )

    async def generate_parent(
        self,
        tree: BalancedMetricTreeState,
        registry: SkillCapsuleRegistry,
        tree_node_id: str,
        child_capsules: Sequence[SkillCapsule],
    ) -> SkillCapsule | None:
        distinct = {
            normalized_capsule_text(capsule)
            for capsule in child_capsules
        }
        if len(distinct) < 2:
            return None
        return await self._generate(
            tree=tree,
            registry=registry,
            tree_node_id=tree_node_id,
            child_capsules=child_capsules,
            system_prompt=(
                "Create one higher-level transferable skill capsule only when "
                "the child capsules support a genuinely new cross-child "
                "invariant. The parent must state when the shared rule applies, "
                "what decision or procedure transfers, how to verify it, and "
                "where it fails. Do not paraphrase one child and do not merge "
                "unrelated procedures. Reject when no useful new abstraction "
                "is supported. Never include task IDs, exact answers, paths, "
                "URLs, coordinates, or example-specific literal values."
            ),
            evidence_payload={
                "child_capsules": [
                    {
                        "capsule_id": capsule.capsule_id,
                        "name": capsule.name,
                        "trigger": capsule.trigger,
                        "content": capsule.content,
                        "scope": capsule.scope,
                        "verification": capsule.verification,
                        "failure_modes": list(capsule.failure_modes),
                        "evidence_count": len(capsule.evidence_atom_ids),
                        "positive_evidence_count": int(
                            capsule.metadata.get(
                                "positive_evidence_count",
                                0,
                            )
                        ),
                        "negative_evidence_count": int(
                            capsule.metadata.get(
                                "negative_evidence_count",
                                0,
                            )
                        ),
                    }
                    for capsule in child_capsules
                ]
            },
            analyst_mode="cross_child_abstraction",
        )

    async def _generate(
        self,
        *,
        tree: BalancedMetricTreeState,
        registry: SkillCapsuleRegistry,
        tree_node_id: str,
        child_capsules: Sequence[SkillCapsule],
        system_prompt: str,
        evidence_payload: Mapping[str, Any],
        analyst_mode: str,
    ) -> SkillCapsule:
        capsule_id, version = registry.next_identity(tree_node_id)
        if child_capsules:
            atom_ids = tuple(
                sorted(
                    {
                        atom_id
                        for child in child_capsules
                        for atom_id in child.evidence_atom_ids
                    }
                )
            )
        else:
            atom_ids = tree.descendant_atom_ids(tree_node_id)
        source_item_ids = tuple(
            tree.atoms[atom_id].source_item_id for atom_id in atom_ids
        )
        outcomes = [
            tree.atoms[atom_id].metadata.get("success")
            for atom_id in atom_ids
        ]
        child_ids = tuple(
            sorted(capsule.capsule_id for capsule in child_capsules)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "output_schema": CAPSULE_SCHEMA,
                        **dict(evidence_payload),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        try:
            _require_prompt_budget(
                self.tokenizer,
                messages,
                self.max_prompt_tokens,
                component="evidence-balanced skill capsule generation",
            )
            payload = await self.generation.chat_json(
                messages,
                schema_name="EvidenceBalancedSkillCapsule",
                guided_json=CAPSULE_SCHEMA,
                max_tokens=self.max_output_tokens,
                retries=1,
                debug_metadata={
                    "component": "ebst_capsule_analyst",
                    "tree_node_id": tree_node_id,
                    "capsule_id": capsule_id,
                    "analyst_mode": analyst_mode,
                    "evidence_atom_count": len(atom_ids),
                    "child_capsule_count": len(child_capsules),
                },
            )
            promote = bool(payload.get("promote"))
            capsule = SkillCapsule(
                capsule_id=capsule_id,
                tree_node_id=tree_node_id,
                version=version,
                level=_capsule_level(tree, tree_node_id),
                name=str(payload.get("name") or "").strip(),
                trigger=str(payload.get("trigger") or "").strip(),
                content=str(payload.get("content") or "").strip(),
                scope=str(payload.get("scope") or "").strip(),
                verification=str(payload.get("verification") or "").strip(),
                failure_modes=tuple(
                    str(value).strip()
                    for value in payload.get("failure_modes", [])
                    if str(value).strip()
                ),
                evidence_atom_ids=atom_ids,
                source_item_ids=source_item_ids,
                child_capsule_ids=child_ids,
                validation_mode=self.validation_mode,
                metadata={
                    "analyst_mode": analyst_mode,
                    "rationale": str(payload.get("rationale") or "").strip(),
                    "behavioral_replay_performed": False,
                    "positive_evidence_count": sum(
                        value is True for value in outcomes
                    ),
                    "negative_evidence_count": sum(
                        value is False for value in outcomes
                    ),
                    "unknown_outcome_evidence_count": sum(
                        not isinstance(value, bool) for value in outcomes
                    ),
                },
            )
            reasons = list(
                validate_capsule_candidate(
                    capsule,
                    child_capsules=child_capsules,
                )
            )
            reasons.extend(
                _unsafe_reusable_text_reasons(
                    "\n".join(
                        (
                            capsule.name,
                            capsule.trigger,
                            capsule.content,
                            capsule.scope,
                            capsule.verification,
                            *capsule.failure_modes,
                        )
                    )
                )
            )
            if not promote:
                reasons.append("analyst_rejected")
            negative_atoms = [
                tree.atoms[atom_id]
                for atom_id in atom_ids
                if tree.atoms[atom_id].metadata.get("success") is False
            ]
            if (
                promote
                and not child_capsules
                and negative_atoms
                and not reasons
            ):
                negative_audit = await self._validate_negative_evidence_use(
                    capsule,
                    negative_atoms,
                )
                capsule.metadata["negative_evidence_validation"] = (
                    negative_audit
                )
                if not bool(negative_audit.get("accept")):
                    reasons.append("negative_evidence_validation_rejected")
            if promote and child_capsules and not reasons:
                promotion = await self._validate_parent_promotion(
                    capsule,
                    child_capsules,
                )
                capsule.metadata["parent_promotion_validation"] = promotion
                if not bool(promotion.get("accept")):
                    reasons.append("parent_promotion_rejected")
            rejection_classes: list[str] = []
            if not promote:
                rejection_classes.append("semantic_rejection")
            if any(reason != "analyst_rejected" for reason in reasons):
                rejection_classes.append("validation_rejection")
            capsule.validation_reasons = tuple(sorted(set(reasons)))
            capsule.status = "active" if not reasons else "rejected"
            capsule.metadata["rejection_classes"] = sorted(
                set(rejection_classes)
            )
            return capsule
        except Exception as exc:
            rejection_class = _generation_error_class(exc)
            return SkillCapsule(
                capsule_id=capsule_id,
                tree_node_id=tree_node_id,
                version=version,
                level=_capsule_level(tree, tree_node_id),
                name="rejected capsule",
                trigger="not retrievable",
                content="generation failed",
                scope="none",
                verification="not performed",
                failure_modes=("generation error",),
                evidence_atom_ids=atom_ids,
                source_item_ids=source_item_ids,
                child_capsule_ids=child_ids,
                status="rejected",
                validation_mode=self.validation_mode,
                validation_reasons=(
                    f"{rejection_class}:{type(exc).__name__}",
                ),
                metadata={
                    "analyst_mode": analyst_mode,
                    "error": str(exc),
                    "behavioral_replay_performed": False,
                    "rejection_classes": [rejection_class],
                },
            )

    async def _validate_negative_evidence_use(
        self,
        capsule: SkillCapsule,
        negative_atoms: Sequence[ExperienceAtom],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Audit how a proposed skill uses failed-trajectory "
                    "evidence. Accept only when failures are used to identify "
                    "a root cause, applicability boundary, guardrail, or a "
                    "corrected procedure. Reject any recommendation of an "
                    "observed failed action and any copied task-specific "
                    "answer, path, coordinate, identifier, or literal."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "output_schema": NEGATIVE_EVIDENCE_AUDIT_SCHEMA,
                        "candidate_skill": {
                            "name": capsule.name,
                            "trigger": capsule.trigger,
                            "content": capsule.content,
                            "scope": capsule.scope,
                            "verification": capsule.verification,
                            "failure_modes": list(capsule.failure_modes),
                        },
                        "negative_evidence": [
                            {
                                "decision": atom.decision,
                                "invariant": atom.invariant,
                                "verification": atom.verification,
                                "failure_mode": atom.failure_mode,
                            }
                            for atom in negative_atoms
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        _require_prompt_budget(
            self.tokenizer,
            messages,
            self.max_prompt_tokens,
            component="negative evidence promotion audit",
        )
        payload = await self.validator_generation.chat_json(
            messages,
            schema_name="EvidenceBalancedNegativeEvidenceAudit",
            guided_json=NEGATIVE_EVIDENCE_AUDIT_SCHEMA,
            max_tokens=self.max_output_tokens,
            retries=1,
            debug_metadata={
                "component": "ebst_negative_evidence_audit",
                "capsule_id": capsule.capsule_id,
                "negative_evidence_count": len(negative_atoms),
            },
        )
        accept = (
            bool(
                payload.get(
                    "uses_negative_for_root_cause_boundary_or_guardrail"
                )
            )
            and not bool(payload.get("recommends_observed_failed_action"))
            and not bool(payload.get("copies_example_specific_content"))
            and bool(payload.get("accept"))
        )
        return {**dict(payload), "accept": accept}

    async def _validate_parent_promotion(
        self,
        capsule: SkillCapsule,
        child_capsules: Sequence[SkillCapsule],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Audit whether a proposed parent skill is justified by "
                    "multiple child skills. Accept only when at least two "
                    "distinct children support the same transferable rule and "
                    "the parent adds a useful cross-child abstraction rather "
                    "than paraphrasing one child. Reject copied task examples, "
                    "answers, paths, coordinates, or accidental literals."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "output_schema": PARENT_PROMOTION_SCHEMA,
                        "candidate_parent": {
                            "name": capsule.name,
                            "trigger": capsule.trigger,
                            "content": capsule.content,
                            "scope": capsule.scope,
                            "verification": capsule.verification,
                            "failure_modes": list(capsule.failure_modes),
                        },
                        "children": [
                            {
                                "capsule_id": child.capsule_id,
                                "name": child.name,
                                "trigger": child.trigger,
                                "content": child.content,
                                "scope": child.scope,
                                "verification": child.verification,
                                "failure_modes": list(
                                    child.failure_modes
                                ),
                            }
                            for child in child_capsules
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        _require_prompt_budget(
            self.tokenizer,
            messages,
            self.max_prompt_tokens,
            component="evidence-balanced parent promotion validation",
        )
        payload = await self.validator_generation.chat_json(
            messages,
            schema_name="EvidenceBalancedParentPromotion",
            guided_json=PARENT_PROMOTION_SCHEMA,
            max_tokens=self.max_output_tokens,
            retries=1,
            debug_metadata={
                "component": "ebst_parent_promotion_validator",
                "capsule_id": capsule.capsule_id,
                "tree_node_id": capsule.tree_node_id,
                "child_capsule_count": len(child_capsules),
            },
        )
        logically_consistent = (
            bool(payload.get("supported_by_multiple_children"))
            and bool(payload.get("adds_cross_child_abstraction"))
            and not bool(payload.get("copies_example_specific_content"))
        )
        payload["accept"] = bool(payload.get("accept")) and logically_consistent
        return payload


async def build_evidence_balanced_tree_from_records(
    config: Any,
) -> dict[str, Any]:
    return await _build(config, dynamic=False)


async def build_evidence_balanced_dynamic_tree_from_records(
    config: Any,
) -> dict[str, Any]:
    return await _build(config, dynamic=True)


async def _build(config: Any, *, dynamic: bool) -> dict[str, Any]:
    from .pipeline import (
        _load_records_for_protocol,
        _refresh_skillbank_index,
        _write_runtime_artifacts,
    )

    ebst = EvidenceBalancedSkillConfig.from_mapping(
        dict(config.hierarchy or {}).get("ebst", {})
    )
    if not bool(config.enforce_dataset_order):
        raise ValueError(
            "evidence_balanced_skill_tree requires enforce_dataset_order=true"
        )
    if config.dynamic.shuffle_seed is not None:
        raise ValueError(
            "evidence_balanced_skill_tree requires dataset-order arrivals; "
            "set dynamic.shuffle_seed=null"
        )
    if bool(config.dynamic.resume_from_snapshots):
        raise ValueError(
            "evidence_balanced_skill_tree does not yet support fingerprinted "
            "snapshot resume; set dynamic.resume_from_snapshots=false"
        )
    if not bool(config.dynamic.snapshot_include_embeddings):
        raise ValueError(
            "evidence_balanced_skill_tree snapshots its complete atom state "
            "and requires dynamic.snapshot_include_embeddings=true"
        )
    if not str(config.embedding.cache_path or "").strip():
        raise ValueError(
            "evidence_balanced_skill_tree requires embedding.cache_path"
        )
    if config.embedding.cache_write_policy != "first_write_wins":
        raise ValueError(
            "evidence_balanced_skill_tree requires "
            "embedding.cache_write_policy='first_write_wins'"
        )
    if dynamic and not ebst.atom_cache_path:
        raise ValueError(
            "controlled dynamic runs require ebst.atom_cache_path from the "
            "matching static atom extraction"
        )
    if dynamic:
        atom_cache_path = Path(str(ebst.atom_cache_path)).resolve()
        if not atom_cache_path.is_file():
            raise FileNotFoundError(
                f"frozen atom cache is missing: {atom_cache_path}"
            )
        source_tree_dir = atom_cache_path.parent
        source_config_path = source_tree_dir.parent / "dynamix_config.json"
        if not source_config_path.is_file():
            raise FileNotFoundError(
                "matching static dynamix_config.json is missing: "
                f"{source_config_path}"
            )
        source_config = json.loads(
            source_config_path.read_text(encoding="utf-8")
        )
        if (
            source_config.get("scenario") != "static_build"
            or source_config.get("hierarchy", {}).get("tree_policy")
            != "evidence_balanced_skill_tree"
            or Path(str(source_config.get("output_dir") or "")).resolve()
            != source_tree_dir
        ):
            raise ValueError(
                "frozen atom cache is not bound to a matching static "
                "evidence-balanced run"
            )
        source_cache = Path(
            str(source_config.get("embedding", {}).get("cache_path") or "")
        ).resolve()
        current_cache = Path(str(config.embedding.cache_path)).resolve()
        if source_cache != current_cache:
            raise ValueError(
                "paired static/dynamic evidence-balanced runs must share "
                "the same embedding cache"
            )
        validate_embedding_cache_manifest(
            cache_path=current_cache,
            manifest_path=(
                source_tree_dir / "embedding_vector_cache_manifest.json"
            ),
        )

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_bank_dir = _resolve_skill_output_dir(
        out,
        config.skill_output_dir_name,
    )
    if not config.generation.debug_dir:
        config.generation.debug_dir = str(
            out / "analysis" / "generation_debug"
        )
    _prepare_otd_analyst_config(config, out)
    _write_runtime_artifacts(config, out)
    records = _load_records_for_protocol(config, out)
    _require_unique_record_ids(records)
    tokenizer = _tokenizer_for_config(config)
    atoms, excluded, atom_source, atom_protocol_fingerprint = (
        await _prepare_atoms(
            config=config,
            ebst=ebst,
            records=records,
            out=out,
            tokenizer=tokenizer,
        )
    )
    if excluded:
        raise ValueError(
            "evidence_balanced_skill_tree requires one valid atom per record; "
            f"{len(excluded)} records were excluded"
        )

    capsule_generation = GenerationClient(
        replace(
            config.generation,
            temperature=float(ebst.capsule_temperature),
        )
    )
    validator_generation = GenerationClient(
        replace(
            config.generation,
            temperature=float(ebst.validator_temperature),
        )
    )
    capsule_analyst = SkillCapsuleAnalyst(
        capsule_generation,
        validator_generation,
        tokenizer=tokenizer,
        max_prompt_tokens=int(config.analyst.max_prompt_tokens),
        max_output_tokens=config.analyst.max_output_tokens,
        validation_mode=ebst.validation_mode,
    )
    tree = BalancedMetricTreeState(
        max_entries=ebst.max_entries,
        dual_view_lambda=ebst.dual_view_lambda,
    )
    registry = SkillCapsuleRegistry()
    insertions: list[dict[str, Any]] = []
    refresh_batches: list[dict[str, Any]] = []

    if dynamic:
        atom_by_source = {atom.source_item_id: atom for atom in atoms}
        initial_count = min(
            max(1, int(config.dynamic.initial_count)),
            len(records),
        )
        arrival_records = list(records[initial_count:])
        if int(config.dynamic.arrival_count) > 0:
            arrival_records = arrival_records[
                : int(config.dynamic.arrival_count)
            ]
        initial_atoms = [
            atom_by_source[record.trajectory_id]
            for record in records[:initial_count]
        ]
        arrival_atoms = [
            atom_by_source[record.trajectory_id]
            for record in arrival_records
        ]
        for atom in initial_atoms:
            result = tree.insert(atom)
            insertions.append(_insertion_payload(result))
        initial_refresh = await _refresh_capsules(
            tree=tree,
            registry=registry,
            analyst=capsule_analyst,
            dirty_node_ids=set(tree.nodes),
            retired_node_ids=set(),
            reason="initial_tree",
        )
        refresh_batches.append(
            {
                "batch_index": 0,
                "arrival_count": 0,
                **initial_refresh,
            }
        )

        batch_size = max(1, int(config.dynamic.update_batch_size))
        dirty: set[str] = set()
        retired: set[str] = set()
        for arrival_index, atom in enumerate(arrival_atoms, start=1):
            result = tree.insert(atom)
            insertions.append(_insertion_payload(result))
            dirty.update(result.capsule_refresh_node_ids)
            retired.update(result.retired_node_ids)
            flush = (
                arrival_index % batch_size == 0
                or arrival_index == len(arrival_atoms)
            )
            if not flush:
                continue
            refresh = await _refresh_capsules(
                tree=tree,
                registry=registry,
                analyst=capsule_analyst,
                dirty_node_ids=dirty,
                retired_node_ids=retired,
                reason=f"arrival_batch_{arrival_index}",
            )
            refresh_batches.append(
                {
                    "batch_index": len(refresh_batches),
                    "arrival_count": arrival_index,
                    **refresh,
                }
            )
            snapshot = (
                out
                / "dynamic_snapshots"
                / f"arrival_{arrival_index:04d}"
            )
            snapshot.mkdir(parents=True, exist_ok=True)
            _write_state_artifacts(tree, registry, snapshot)
            dirty = set()
            retired = set()
    else:
        initial_count = len(atoms)
        arrival_atoms = []
        for atom in atoms:
            result = tree.insert(atom)
            insertions.append(_insertion_payload(result))
        refresh_batches.append(
            {
                "batch_index": 0,
                "arrival_count": 0,
                **(
                    await _refresh_capsules(
                        tree=tree,
                        registry=registry,
                        analyst=capsule_analyst,
                        dirty_node_ids=set(tree.nodes),
                        retired_node_ids=set(),
                        reason="static_tree",
                    )
                ),
            }
        )

    registry.rebuild_active_links()
    registry.validate(tree)
    active_capsules = [
        registry.capsules[capsule_id]
        for capsule_id in registry.active_by_tree_node.values()
    ]
    if not active_capsules:
        raise RuntimeError(
            "evidence_balanced_skill_tree produced no active skill capsules"
        )

    _write_jsonl(
        out / "balanced_tree_insertions.jsonl",
        insertions,
    )
    _write_jsonl(
        out / "skill_lifecycle_events.jsonl",
        registry.events,
    )
    _write_state_artifacts(tree, registry, out)
    manifest = _write_nodebank_manifest(
        tree=tree,
        registry=registry,
        output_dir=node_bank_dir,
        config=config,
        ebst=ebst,
        tokenizer=tokenizer,
    )
    skillbank_index = _refresh_skillbank_index(node_bank_dir, config)
    vector_cache_manifest = _write_vector_cache_manifest(
        out=out,
        config=config,
        atoms=atoms,
        skillbank_index_path=skillbank_index,
    )
    audit = _quality_audit(
        tree,
        registry,
        insertion_events=insertions,
        nodebank_manifest=manifest,
    )
    rejection_class_counts = {
        rejection_class: sum(
            rejection_class
            in capsule.metadata.get("rejection_classes", ())
            for capsule in registry.capsules.values()
        )
        for rejection_class in _REJECTION_CLASSES
    }
    (out / "tree_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "scenario": "dynamic_update" if dynamic else "static_build",
        "tree_policy": "evidence_balanced_skill_tree",
        "record_count": len(records),
        "atom_count": len(atoms),
        "atom_source": atom_source,
        "atom_cache_path": ebst.atom_cache_path,
        "atom_protocol_fingerprint": atom_protocol_fingerprint,
        "initial_count": initial_count,
        "arrival_count": len(arrival_atoms),
        "insertion_count": len(arrival_atoms) if dynamic else 0,
        "updated_count": len(arrival_atoms) if dynamic else 0,
        "excluded_count": 0,
        "arrival_update_semantics": (
            "sequential_structural_insert_batched_local_capsule_refresh"
            if dynamic
            else "static_dataset_order_uses_same_insert_operation"
        ),
        "capsule_refresh_batch_size": (
            max(1, int(config.dynamic.update_batch_size))
            if dynamic
            else 0
        ),
        "tree_node_count": len(tree.nodes),
        "active_capsule_count": len(active_capsules),
        "rejected_capsule_count": sum(
            capsule.status == "rejected"
            for capsule in registry.capsules.values()
        ),
        "semantic_rejection_count": rejection_class_counts[
            "semantic_rejection"
        ],
        "validation_rejection_count": rejection_class_counts[
            "validation_rejection"
        ],
        "runtime_generation_error_count": rejection_class_counts[
            "runtime_generation_error"
        ],
        "prompt_budget_error_count": rejection_class_counts[
            "prompt_budget_error"
        ],
        "archived_capsule_count": sum(
            capsule.status == "archived"
            for capsule in registry.capsules.values()
        ),
        "retrievable_atom_count": 0,
        "node_count": int(manifest["node_count"]),
        "node_bank_dir": str(node_bank_dir),
        "node_bank_manifest": str(
            node_bank_dir / "node_bank_manifest.json"
        ),
        "skillbank_index": skillbank_index,
        "embedding_vector_cache_manifest": str(
            out / "embedding_vector_cache_manifest.json"
        ),
        "embedding_vector_cache_logical_sha256": (
            vector_cache_manifest["logical_sha256"]
        ),
        "embedding_vector_cache_entry_count": vector_cache_manifest[
            "entry_count"
        ],
        "refresh_batches": refresh_batches,
        "structural_audit": tree.structural_audit(),
        "quality_audit": audit,
        "guarantee_boundary": {
            "balanced_occupancy_verified": True,
            "equal_leaf_depth_verified": True,
            "cover_radius_certified_upper_bound": True,
            "unique_atom_placement": True,
            "height_within_reported_bound": bool(
                audit["structural"]["height_within_bound"]
            ),
            "metric_search_exactness_supported": True,
            "llm_semantics_guaranteed": False,
            "behavioral_replay_performed": False,
            "downstream_improvement_claimed": False,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


async def _prepare_atoms(
    *,
    config: Any,
    ebst: EvidenceBalancedSkillConfig,
    records: Sequence[Any],
    out: Path,
    tokenizer: Any,
) -> tuple[list[ExperienceAtom], list[dict[str, Any]], str, str]:
    atom_config = CertifiedOtdConfig(
        dual_view_lambda=ebst.dual_view_lambda,
        atom_temperature=ebst.atom_temperature,
        parent_temperature=ebst.capsule_temperature,
        atom_cache_path=ebst.atom_cache_path,
    )
    records_fingerprint = _records_fingerprint(records)
    protocol_fingerprint = _atom_protocol_fingerprint(
        config,
        atom_config,
    )
    if ebst.atom_cache_path:
        atoms, excluded = _load_atom_cache(
            Path(ebst.atom_cache_path),
            records,
            records_fingerprint=records_fingerprint,
            atom_protocol_fingerprint=protocol_fingerprint,
        )
        source = "frozen_cache"
    else:
        generation = GenerationClient(
            replace(
                config.generation,
                temperature=float(ebst.atom_temperature),
            )
        )
        embedding = EmbeddingClient(config.embedding)
        analyst = ExperienceAtomAnalyst(
            generation,
            embedding,
            config=atom_config,
            tokenizer=tokenizer,
            max_prompt_tokens=int(config.analyst.max_prompt_tokens),
            max_output_tokens=config.analyst.max_output_tokens,
            max_evidence_chars=int(
                config.analyst.analysis_bundle_max_chars or 60000
            ),
        )
        try:
            atoms, excluded = await analyst.extract_many(records)
        finally:
            embedding.close()
        source = "generated"
    _require_complete_atom_identity(records, atoms)
    payload = {
        "format": "certified_experience_atoms_v1",
        "record_count": len(records),
        "atom_count": len(atoms),
        "excluded_count": len(excluded),
        "excluded_records": excluded,
        "atom_source": source,
        "atom_cache_path": ebst.atom_cache_path,
        "records_fingerprint": records_fingerprint,
        "atom_protocol_fingerprint": protocol_fingerprint,
        "atoms": [atom.to_dict() for atom in atoms],
    }
    (out / "experience_atoms.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return atoms, excluded, source, protocol_fingerprint


async def _refresh_capsules(
    *,
    tree: BalancedMetricTreeState,
    registry: SkillCapsuleRegistry,
    analyst: SkillCapsuleAnalyst,
    dirty_node_ids: set[str],
    retired_node_ids: set[str],
    reason: str,
) -> dict[str, Any]:
    dirty = {
        node_id for node_id in dirty_node_ids if node_id in tree.nodes
    }
    for node_id in list(dirty):
        parent_id = tree.nodes[node_id].parent_id
        while parent_id is not None:
            dirty.add(parent_id)
            parent_id = tree.nodes[parent_id].parent_id
    registry.archive_tree_nodes(
        sorted(dirty | retired_node_ids),
        reason=reason,
    )
    depths = _tree_depths(tree)
    generated: list[str] = []
    rejected: list[str] = []
    skipped: list[str] = []
    for depth in sorted(
        {depths[node_id] for node_id in dirty},
        reverse=True,
    ):
        level_nodes = sorted(
            node_id
            for node_id in dirty
            if depths[node_id] == depth
        )
        coroutines = []
        coroutine_nodes = []
        for node_id in level_nodes:
            node = tree.nodes[node_id]
            if node.is_leaf:
                if len(node.atom_ids) < 2:
                    skipped.append(node_id)
                    continue
                coroutine = analyst.generate_leaf(
                    tree,
                    registry,
                    node_id,
                )
            else:
                frontier_by_child = [
                    registry.active_frontier(tree, child_id)
                    for child_id in node.child_ids
                ]
                supported_branches = [
                    capsules
                    for capsules in frontier_by_child
                    if capsules
                ]
                if len(supported_branches) < 2:
                    skipped.append(node_id)
                    continue
                frontier = [
                    capsule
                    for capsules in supported_branches
                    for capsule in capsules
                ]
                distinct = {
                    normalized_capsule_text(capsule)
                    for capsule in frontier
                }
                if len(distinct) < 2:
                    skipped.append(node_id)
                    continue
                coroutine = analyst.generate_parent(
                    tree,
                    registry,
                    node_id,
                    tuple(frontier),
                )
            coroutine_nodes.append(node_id)
            coroutines.append(coroutine)
        if not coroutines:
            continue
        results = await asyncio.gather(*coroutines)
        for node_id, capsule in zip(coroutine_nodes, results):
            if capsule is None:
                skipped.append(node_id)
                continue
            registry.register(capsule, event_reason=reason)
            if capsule.retrievable:
                generated.append(capsule.capsule_id)
            else:
                rejected.append(capsule.capsule_id)
    registry.rebuild_active_links()
    registry.validate(tree)
    return {
        "dirty_tree_node_ids": sorted(dirty),
        "retired_tree_node_ids": sorted(retired_node_ids),
        "generated_capsule_ids": generated,
        "rejected_capsule_ids": rejected,
        "skipped_tree_node_ids": sorted(set(skipped)),
    }


def _tree_depths(tree: BalancedMetricTreeState) -> dict[str, int]:
    if tree.root_id is None:
        return {}
    depths: dict[str, int] = {}
    stack = [(tree.root_id, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        stack.extend(
            (child_id, depth + 1)
            for child_id in tree.nodes[node_id].child_ids
        )
    return depths


def _capsule_level(
    tree: BalancedMetricTreeState,
    tree_node_id: str,
) -> int:
    depths = _tree_depths(tree)
    leaf_depth = max(tree.leaf_depths().values(), default=0)
    return leaf_depth - depths[tree_node_id] + 1


def _insertion_payload(
    result: BalancedInsertionResult,
) -> dict[str, Any]:
    return {
        "atom_id": result.atom_id,
        "path_before_split": list(result.path_before_split),
        "affected_node_ids": list(result.affected_node_ids),
        "capsule_refresh_node_ids": list(
            result.capsule_refresh_node_ids
        ),
        "created_node_ids": list(result.created_node_ids),
        "retired_node_ids": list(result.retired_node_ids),
        "split_count": result.split_count,
        "root_changed": result.root_changed,
    }


def _write_state_artifacts(
    tree: BalancedMetricTreeState,
    registry: SkillCapsuleRegistry,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "balanced_tree_state.json").write_text(
        json.dumps(tree.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "skill_capsules.json").write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(
        output_dir / "skill_capsules.jsonl",
        [
            capsule.to_dict()
            for capsule in sorted(
                registry.capsules.values(),
                key=lambda item: item.capsule_id,
            )
        ],
    )


def _render_capsule_prompt(capsule: SkillCapsule) -> str:
    failure_modes = "\n".join(
        f"- {failure_mode}" for failure_mode in capsule.failure_modes
    )
    return (
        f"### {capsule.name}\n"
        f"Trigger: {capsule.trigger}\n"
        f"Scope: {capsule.scope}\n\n"
        f"{capsule.content}\n\n"
        f"Verification: {capsule.verification}\n"
        f"Failure modes:\n{failure_modes}"
    )


def _write_nodebank_manifest(
    *,
    tree: BalancedMetricTreeState,
    registry: SkillCapsuleRegistry,
    output_dir: Path,
    config: Any,
    ebst: EvidenceBalancedSkillConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    active = [
        registry.capsules[capsule_id]
        for capsule_id in sorted(registry.active_by_tree_node.values())
    ]
    virtual_root = "__evidence_balanced_virtual_root__"
    children_by_node: dict[str, list[str]] = {
        virtual_root: sorted(
            capsule.capsule_id
            for capsule in active
            if capsule.parent_capsule_id is None
        )
    }
    nodes: list[dict[str, Any]] = []
    for capsule in active:
        children = list(
            capsule.metadata.get("active_child_capsule_ids", [])
        )
        children_by_node[capsule.capsule_id] = children
        embedding_text = (
            f"name: {capsule.name}\n"
            f"trigger: {capsule.trigger}\n"
            f"content: {capsule.content}"
        )
        prompt_text = _render_capsule_prompt(capsule)
        nodes.append(
            {
                "node_id": capsule.capsule_id,
                "item_id": capsule.capsule_id,
                "level": capsule.level,
                "support_mass": float(len(capsule.evidence_atom_ids)),
                "confidence": capsule.reliability,
                "name": capsule.name,
                "trigger": capsule.trigger,
                "content": capsule.content,
                "embedding_text": embedding_text,
                "prompt_text": prompt_text,
                "source_community_id": capsule.tree_node_id,
                "source_member_count": len(capsule.evidence_atom_ids),
                "analyst_mode": capsule.metadata.get(
                    "analyst_mode",
                    "",
                ),
                "sha256": hashlib.sha256(
                    embedding_text.encode("utf-8")
                ).hexdigest(),
                "parent_node_id": (
                    capsule.parent_capsule_id or virtual_root
                ),
                "child_node_ids": children,
                "token_cost": max(1, tokenizer.count(prompt_text)),
                "validation_mode": capsule.validation_mode,
                "validation_reasons": list(
                    capsule.validation_reasons
                ),
                "evidence_atom_ids": list(capsule.evidence_atom_ids),
                "source_item_ids": list(capsule.source_item_ids),
                "lifecycle_status": capsule.status,
                "behavioral_replay_performed": False,
            }
        )
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "evidence_balanced_skill_tree",
        "output_dir": str(output_dir),
        "root_node_id": virtual_root,
        "node_count": len(nodes),
        "nodes": nodes,
        "item_to_node_ids": {
            node["item_id"]: [node["node_id"]] for node in nodes
        },
        "tree_index": {
            "root_node_id": virtual_root,
            "children_by_node": children_by_node,
        },
        "export_policy": {
            "retrieval_unit": "validated_skill_capsule",
            "embedding_fields": ["name", "trigger", "content"],
            "trajectory_items_exported": False,
            "experience_atoms_exported": False,
            "heldout_retrieval": "tree_antichain_knapsack",
            "top_k_is_cardinality_cap": True,
            "token_budget": int(ebst.retrieval_token_budget),
            "token_unit": int(ebst.retrieval_token_unit),
            "exact_search_max_states": int(
                ebst.retrieval_exact_search_max_states
            ),
            "fixed_prompt_overhead_tokens": max(
                1,
                tokenizer.count(retrieved_experience_preamble()),
            ),
            "single_parent_structure": True,
            "ancestor_descendant_co_selection": False,
            "validation_mode": ebst.validation_mode,
            "behavioral_replay_performed": False,
            "relevance_transform": (
                "shifted_cosine_(1_plus_cosine)_over_2"
            ),
            "tokenizer": {
                "implementation": type(tokenizer).__name__,
                "model_or_path": (
                    getattr(config.analyst, "tokenizer_model", None)
                    or getattr(config.embedding, "tokenizer_model", None)
                ),
                "regex_fallback_allowed": (
                    not bool(config.analyst.tokenizer_required)
                    and bool(
                        config.analyst.allow_regex_tokenizer_fallback
                    )
                ),
            },
        },
    }
    (output_dir / "node_bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _quality_audit(
    tree: BalancedMetricTreeState,
    registry: SkillCapsuleRegistry,
    *,
    insertion_events: Sequence[Mapping[str, Any]],
    nodebank_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    active = [
        registry.capsules[capsule_id]
        for capsule_id in registry.active_by_tree_node.values()
    ]
    normalized: dict[str, list[str]] = {}
    for capsule in active:
        normalized.setdefault(
            normalized_capsule_text(capsule),
            [],
        ).append(capsule.capsule_id)
    duplicate_groups = [
        capsule_ids
        for capsule_ids in normalized.values()
        if len(capsule_ids) > 1
    ]
    affected_counts = [
        len(event.get("affected_node_ids", []))
        for event in insertion_events
    ]
    refresh_counts = [
        len(event.get("capsule_refresh_node_ids", []))
        for event in insertion_events
    ]
    level_counts: dict[str, int] = {}
    token_costs: list[int] = []
    parent_counts: dict[str, int] = {}
    for node in nodebank_manifest.get("nodes", []):
        level = str(int(node.get("level", 0)))
        level_counts[level] = level_counts.get(level, 0) + 1
        token_costs.append(int(node.get("token_cost", 0)))
        for child_id in node.get("child_node_ids", []):
            child_key = str(child_id)
            parent_counts[child_key] = parent_counts.get(child_key, 0) + 1
    return {
        "structural": tree.structural_audit(),
        "locality": {
            "insertion_event_count": len(insertion_events),
            "split_event_count": sum(
                int(event.get("split_count", 0)) > 0
                for event in insertion_events
            ),
            "max_structurally_affected_nodes": max(
                affected_counts,
                default=0,
            ),
            "mean_structurally_affected_nodes": (
                sum(affected_counts) / len(affected_counts)
                if affected_counts
                else 0.0
            ),
            "max_capsule_refresh_nodes": max(refresh_counts, default=0),
            "mean_capsule_refresh_nodes": (
                sum(refresh_counts) / len(refresh_counts)
                if refresh_counts
                else 0.0
            ),
        },
        "capsules": {
            "total_versions": len(registry.capsules),
            "active": len(active),
            "candidate": sum(
                capsule.status == "candidate"
                for capsule in registry.capsules.values()
            ),
            "candidate_event_count": sum(
                event.get("status") == "candidate"
                for event in registry.events
            ),
            "rejected": sum(
                capsule.status == "rejected"
                for capsule in registry.capsules.values()
            ),
            "archived": sum(
                capsule.status == "archived"
                for capsule in registry.capsules.values()
            ),
            "retrievable_atom_count": 0,
            "support_counts": [
                len(capsule.evidence_atom_ids) for capsule in active
            ],
            "exact_duplicate_groups": duplicate_groups,
            "exact_duplicate_capsule_count": sum(
                len(group) for group in duplicate_groups
            ),
            "behavioral_replay_performed": False,
            "rejection_class_counts": {
                rejection_class: sum(
                    rejection_class
                    in capsule.metadata.get("rejection_classes", ())
                    for capsule in registry.capsules.values()
                )
                for rejection_class in _REJECTION_CLASSES
            },
        },
        "nodebank_export": {
            "node_count": int(nodebank_manifest.get("node_count", 0)),
            "level_counts": dict(sorted(level_counts.items())),
            "token_cost_total": sum(token_costs),
            "token_cost_max": max(token_costs, default=0),
            "lineage_parent_conflict_count": sum(
                count > 1 for count in parent_counts.values()
            ),
            "retrieval_token_budget": nodebank_manifest.get(
                "export_policy",
                {},
            ).get("token_budget"),
        },
    }

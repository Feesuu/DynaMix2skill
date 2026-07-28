from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import dynamix_core.certified_otd as certified_otd_module
from dynamix_core.certified_otd import (
    ExperienceAtom,
    OtdInsertionResult,
    OtdTreeState,
    audit_observed_beta_separation,
)

from . import clients as clients_module
from . import openai_compat as openai_compat_module
from . import tokenization as tokenization_module
from . import trace_views as trace_views_module
from .clients import (
    EmbeddingClient,
    GenerationClient,
    embedding_cache_namespace,
    embedding_protocol_payload,
    generation_protocol_payload,
    validate_embedding_cache_manifest,
    write_embedding_cache_manifest,
)
from .schemas import RawTrajectoryRecord
from dynamix_trace2skill.skillbank import (
    render_cdost_node_prompt,
    retrieved_experience_preamble,
    skillbank_vector_cache_namespace,
)
from .tokenization import get_tokenizer
from .trace_views import render_compact_analysis_bundle_text


ATOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trigger": {"type": "string", "minLength": 1, "maxLength": 1200},
        "scope": {"type": "string", "minLength": 1, "maxLength": 1200},
        "decision": {"type": "string", "minLength": 1, "maxLength": 1200},
        "invariant": {"type": "string", "minLength": 1, "maxLength": 1200},
        "verification": {"type": "string", "minLength": 1, "maxLength": 1200},
        "failure_mode": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": [
        "trigger",
        "scope",
        "decision",
        "invariant",
        "verification",
        "failure_mode",
    ],
    "additionalProperties": False,
}


PARENT_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 256},
        "trigger": {"type": "string", "minLength": 1, "maxLength": 1200},
        "content": {"type": "string", "minLength": 1, "maxLength": 2400},
    },
    "required": ["name", "trigger", "content"],
    "additionalProperties": False,
}

FAILURE_ATOM_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported_root_cause", "cautious_diagnostic"],
        },
        "supporting_step_ids": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "maxItems": 16,
        },
        "rationale": {"type": "string", "maxLength": 1200},
        "atom": ATOM_SCHEMA,
    },
    "required": ["verdict", "supporting_step_ids", "rationale", "atom"],
    "additionalProperties": False,
}

FAILURE_ROOT_CAUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requested_semantics": {"type": "string", "maxLength": 1200},
        "performed_semantics": {"type": "string", "maxLength": 1200},
        "source_target_mapping": {"type": "string", "maxLength": 1200},
        "operation_audit": {"type": "string", "maxLength": 1200},
        "verifier_outcome": {"type": "string", "maxLength": 1200},
        "supported_root_cause": {"type": "string", "maxLength": 1200},
        "atom": ATOM_SCHEMA,
    },
    "required": [
        "requested_semantics",
        "performed_semantics",
        "source_target_mapping",
        "operation_audit",
        "verifier_outcome",
        "supported_root_cause",
        "atom",
    ],
    "additionalProperties": False,
}


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"exists": False}
    return {
        "exists": True,
        "kind": "file",
        "size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _validate_source_build_output(output_path: Path) -> None:
    resolved = output_path.resolve()
    marker_path = (
        resolved.parent.parent / "stage_markers" / "04_build_tree.done"
    )
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"source build marker is missing: {marker_path}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    identities = marker.get("output_identities")
    expected = (
        identities.get(str(resolved))
        if isinstance(identities, dict)
        else None
    )
    if not isinstance(expected, dict) or expected != _file_identity(resolved):
        raise ValueError(
            "source output no longer matches its completed stage marker: "
            f"{resolved}"
        )


def _resolve_skill_output_dir(output_dir: Path, value: str) -> Path:
    relative = Path(str(value or "").strip())
    if (
        not str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(
            "skill_output_dir_name must be a non-empty relative path "
            "without parent traversal"
        )
    resolved_output = output_dir.resolve()
    destination = (resolved_output / relative).resolve()
    if not destination.is_relative_to(resolved_output):
        raise ValueError("skill_output_dir_name escapes output_dir")
    return destination


def _atom_trigger_embedding_text(*, trigger: str, scope: str) -> str:
    return f"trigger: {trigger}\nscope: {scope}"


def _atom_procedure_embedding_text(
    *,
    decision: str,
    invariant: str,
    verification: str,
    failure_mode: str,
) -> str:
    return (
        f"decision: {decision}\n"
        f"invariant: {invariant}\n"
        f"verification: {verification}\n"
        f"failure_mode: {failure_mode}"
    )


@dataclass(frozen=True)
class CertifiedOtdConfig:
    dual_view_lambda: float = 0.5
    tie_epsilon: float = 0.0
    atom_temperature: float = 0.0
    parent_temperature: float = 0.0
    retrieval_token_budget: int = 24000
    retrieval_token_unit: int = 128
    retrieval_exact_search_max_states: int = 250_000
    validation_mode: str = "structural_only"
    atom_cache_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CertifiedOtdConfig":
        config = cls(**dict(payload or {}))
        if not 0.0 <= float(config.dual_view_lambda) <= 1.0:
            raise ValueError("otd.dual_view_lambda must be in [0, 1]")
        if float(config.tie_epsilon) < 0.0:
            raise ValueError("otd.tie_epsilon must be non-negative")
        if int(config.retrieval_token_budget) <= 0:
            raise ValueError("otd.retrieval_token_budget must be positive")
        if int(config.retrieval_token_unit) <= 0:
            raise ValueError("otd.retrieval_token_unit must be positive")
        if int(config.retrieval_exact_search_max_states) <= 0:
            raise ValueError(
                "otd.retrieval_exact_search_max_states must be positive"
            )
        if config.validation_mode != "structural_only":
            raise ValueError(
                "only validation_mode='structural_only' is implemented; "
                "witness replay must not be claimed without a benchmark replay adapter"
            )
        return config


class ExperienceAtomAnalyst:
    def __init__(
        self,
        generation: GenerationClient,
        embedding: EmbeddingClient,
        *,
        config: CertifiedOtdConfig,
        tokenizer: Any,
        max_prompt_tokens: int,
        max_output_tokens: int | None,
        max_evidence_chars: int,
    ) -> None:
        self.generation = generation
        self.embedding = embedding
        self.config = config
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_output_tokens = max_output_tokens
        self.max_evidence_chars = max(1000, int(max_evidence_chars))

    async def extract_many(
        self,
        records: Sequence[RawTrajectoryRecord],
    ) -> tuple[list[ExperienceAtom], list[dict[str, Any]]]:
        _require_unique_record_ids(records)
        results = await asyncio.gather(
            *(self._extract_one(record) for record in records),
            return_exceptions=True,
        )
        drafts: list[tuple[RawTrajectoryRecord, dict[str, str]]] = []
        excluded: list[dict[str, Any]] = []
        for record, result in zip(records, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                if record.success is False and isinstance(result, ValueError):
                    fallback = _cautious_failure_diagnostic(
                        _spreadsheet_reference_audit(record)
                    )
                    fallback_reasons = [
                        *_atom_leakage_reasons(record, fallback),
                        *_failure_atom_semantic_reasons(
                            fallback,
                            reference_audit=_spreadsheet_reference_audit(record),
                            verdict="cautious_diagnostic",
                        ),
                    ]
                    if not fallback_reasons:
                        fallback["_analysis_mode"] = (
                            "deterministic_cautious_fallback"
                        )
                        fallback["_review_attempts"] = "0"
                        fallback["_review_rejection_reasons"] = json.dumps(
                            [
                                "pre-review structured response rejected: "
                                + str(result)[:800]
                            ],
                            ensure_ascii=False,
                        )
                        fallback["_supporting_step_ids"] = "[]"
                        drafts.append((record, fallback))
                        continue
                excluded.append(
                    {
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "error_type": type(result).__name__,
                        "error": str(result),
                    }
                )
                continue
            drafts.append((record, result))
        if excluded:
            detail = "; ".join(
                f"{item['trajectory_id']}:{item['error_type']}:{item['error']}"
                for item in excluded
            )
            raise RuntimeError(
                "Experience Atom extraction must produce one clean atom per "
                f"trajectory; failures={len(excluded)} ({detail})"
            )

        trigger_texts = [
            _atom_trigger_embedding_text(
                trigger=draft["trigger"],
                scope=draft["scope"],
            )
            for _, draft in drafts
        ]
        procedure_texts = [
            _atom_procedure_embedding_text(
                decision=draft["decision"],
                invariant=draft["invariant"],
                verification=draft["verification"],
                failure_mode=draft["failure_mode"],
            )
            for _, draft in drafts
        ]
        trigger_embeddings, procedure_embeddings = await asyncio.gather(
            self.embedding.embed_texts(
                trigger_texts,
                cache_namespace="cdost_atom_trigger",
            ),
            self.embedding.embed_texts(
                procedure_texts,
                cache_namespace="cdost_atom_procedure",
            ),
        )
        atoms = [
            self._to_atom(record, draft, trigger_embedding, procedure_embedding)
            for (
                (record, draft),
                trigger_embedding,
                procedure_embedding,
            ) in zip(drafts, trigger_embeddings, procedure_embeddings)
        ]
        _require_complete_atom_identity(records, atoms)
        return atoms, excluded

    async def _extract_one(self, record: RawTrajectoryRecord) -> dict[str, str]:
        evidence = render_compact_analysis_bundle_text(
            record,
            max_chars=self.max_evidence_chars,
        )
        reference_audit = _spreadsheet_reference_audit(record)
        is_failure = not bool(record.success)
        response_schema = (
            FAILURE_ROOT_CAUSE_SCHEMA if is_failure else ATOM_SCHEMA
        )
        schema_name = (
            "CertifiedFailureRootCauseAtom"
            if is_failure
            else "CertifiedExperienceAtom"
        )
        if is_failure:
            system_prompt = (
                "You perform verifier-grounded root-cause analysis of one failed "
                "agent trajectory and produce one reusable Experience Atom. The "
                "LibreOffice-recalculated verifier result is authoritative; ignore "
                "the agent's claimed completion and never claim that the verifier "
                "or golden artifact is wrong. Explicitly contrast the requested "
                "semantics, the action actually performed, source and output roles, "
                "the operation type, and the verifier outcome before naming a root "
                "cause. A root cause must be supported by the trajectory; if evidence "
                "is insufficient, produce a cautious diagnostic procedure instead "
                "of guessing a fix. The Atom decision must state the corrected "
                "reusable policy and must never narrate or recommend the failed "
                "action. For spreadsheet tasks, explicitly "
                "audit whether every instruction-named source region and operation "
                "was used, whether source and output regions were confused, and "
                "whether required aggregation was replaced by row-wise mapping. "
                "Mention these causes only when supported by the trace. The "
                "reusable atom must not copy task "
                "IDs, paths, coordinates, filenames, or exact observed/expected "
                "values. Keep every field concise."
            )
        else:
            system_prompt = (
                "You extract one reusable Experience Atom from one successful "
                "agent trajectory and its verifier evidence. Separate observed "
                "evidence from reusable guidance. Produce one coherent atom, not "
                "a list. The atom must generalize beyond this task. Never copy task "
                "IDs, file paths, workbook coordinates, candidate document names, "
                "exact answer values, or other incidental identifiers into "
                "reusable fields. State when the guidance applies, how to verify "
                "it, and the failure mode outside scope. A single trajectory is "
                "local evidence, not proof of causality. Keep every field concise "
                "and never pad the response."
            )
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Return exactly one JSON object matching this schema:\n"
                    f"{json.dumps(response_schema, ensure_ascii=False)}\n\n"
                    "Trajectory and verifier evidence:\n"
                    f"{evidence}\n\nDeterministic spreadsheet reference audit:\n"
                    f"{json.dumps(reference_audit, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        base_messages = list(messages)
        last_reason_categories: list[str] = []
        candidate: dict[str, str] | None = None
        failure_analysis: Mapping[str, Any] | None = None
        for semantic_attempt in range(3):
            _require_prompt_budget(
                self.tokenizer,
                messages,
                self.max_prompt_tokens,
                component="Experience Atom extraction",
            )
            payload = await self.generation.chat_json(
                messages,
                schema_name=schema_name,
                guided_json=response_schema,
                max_tokens=self.max_output_tokens,
                retries=1,
                debug_metadata={
                    "component": "cdost_atom_extractor",
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.task_id,
                    "semantic_attempt": semantic_attempt + 1,
                },
            )
            atom_payload = payload.get("atom") if is_failure else payload
            if not isinstance(atom_payload, Mapping):
                raise ValueError("Experience Atom response is missing atom object")
            if is_failure:
                failure_analysis = payload
            draft = {
                name: _canonicalize_reusable_syntax(
                    _required_string(atom_payload, name, max_length=1200)
                )
                for name in (
                    "trigger",
                    "scope",
                    "decision",
                    "invariant",
                    "verification",
                    "failure_mode",
                )
            }
            leakage_reasons = _atom_leakage_reasons(record, draft)
            if not leakage_reasons:
                candidate = draft
                break
            if semantic_attempt == 2:
                redacted_draft = {
                    name: _redact_atom_source_literals(record, value)
                    for name, value in draft.items()
                }
                if not _atom_leakage_reasons(record, redacted_draft):
                    candidate = redacted_draft
                    break
            last_reason_categories = sorted(
                {
                    (
                        "copied source or verifier literal"
                        if reason.startswith("source literal ")
                        else reason
                    )
                    for reason in leakage_reasons
                }
            )
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        f"Revision {semantic_attempt + 1}: the previous draft was "
                        "rejected for these categories only: "
                        f"{'; '.join(last_reason_categories)}. Write a fresh, concise "
                        "atom. Do not quote or paraphrase any concrete example from "
                        "the trajectory, verifier, or previous draft. Use only "
                        "general guidance without identifiers, paths, coordinates, "
                        "URLs, or exact answer values."
                    ),
                },
            ]
        if candidate is None:
            raise ValueError(
                "failed to produce a leakage-free Experience Atom; categories="
                + ",".join(last_reason_categories)
            )
        if is_failure:
            candidate = await self._review_failed_atom(
                record,
                evidence=evidence,
                draft=candidate,
                analysis=failure_analysis or {},
                reference_audit=reference_audit,
            )
        else:
            candidate["_analysis_mode"] = "successful_trace_summary"
            candidate["_review_attempts"] = "0"
            candidate["_review_rejection_reasons"] = "[]"
            candidate["_supporting_step_ids"] = "[]"
        return candidate

    async def _review_failed_atom(
        self,
        record: RawTrajectoryRecord,
        *,
        evidence: str,
        draft: Mapping[str, str],
        analysis: Mapping[str, Any],
        reference_audit: Mapping[str, Any],
    ) -> dict[str, str]:
        base_messages = [
            {
                "role": "system",
                "content": (
                    "You are a verifier-grounded root-cause critic for a failed "
                    "agent trajectory. The LibreOffice-recalculated verifier "
                    "result is authoritative. Review the draft Experience Atom "
                    "and its proposed evidence chain against the requested task, "
                    "actions, observations, and final verifier result. Reject any "
                    "claimed cause that is not directly supported by the trace. "
                    "Never claim that the verifier or golden artifact is wrong. "
                    "For spreadsheet tasks, compare instruction-named source and "
                    "output regions, aggregation semantics, and the ranges actually "
                    "used by the action before accepting a cause. The deterministic "
                    "reference audit is only a lexical coverage signal: code may "
                    "access a range through loops, numeric bounds, variables, or "
                    "helper functions. Never treat a missing literal range string "
                    "as proof that data was omitted, and do not infer a hidden "
                    "formula. The decision field must state the corrected "
                    "reusable policy; it must never narrate what the failed agent "
                    "decided or repeat the failed action. "
                    "Use verdict supported_root_cause only when the claimed cause is "
                    "directly supported by specific recorded action or observation "
                    "steps, and list those integer step IDs in supporting_step_ids. "
                    "The verifier mismatch alone is not sufficient evidence. Use "
                    "cautious_diagnostic with an empty or limited support list when "
                    "the trace cannot establish a specific cause. "
                    "Return a corrected Atom that explains a "
                    "reusable cause or a cautious diagnostic procedure. Never "
                    "recommend the failed action, trust a claimed completion over "
                    "the recalculated result, invent an unsupported solution, or copy "
                    "task IDs, paths, coordinates, filenames, and exact observed "
                    "or expected values. The reusable Atom must not mention benchmark "
                    "evaluators, verifiers, golden artifacts, or ground truth. Keep "
                    "every field concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return one JSON object matching this schema:\n"
                    f"{json.dumps(FAILURE_ATOM_REVIEW_SCHEMA, ensure_ascii=False)}"
                    "\n\nFailed trajectory evidence:\n"
                    f"{evidence}\n\nDraft Atom to audit:\n"
                    f"{json.dumps(dict(draft), ensure_ascii=False, indent=2)}"
                    "\n\nProposed root-cause chain to audit:\n"
                    f"{json.dumps(dict(analysis), ensure_ascii=False, indent=2)}"
                    "\n\nDeterministic spreadsheet reference audit:\n"
                    f"{json.dumps(dict(reference_audit), ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        messages = list(base_messages)
        last_reasons: list[str] = []
        for review_attempt in range(2):
            _require_prompt_budget(
                self.tokenizer,
                messages,
                self.max_prompt_tokens,
                component="failed Experience Atom root-cause review",
            )
            try:
                payload = await self.generation.chat_json(
                    messages,
                    schema_name="CertifiedFailureAtomReview",
                    guided_json=FAILURE_ATOM_REVIEW_SCHEMA,
                    max_tokens=self.max_output_tokens,
                    retries=1,
                    debug_metadata={
                        "component": "cdost_failure_atom_reviewer",
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "review_attempt": review_attempt + 1,
                    },
                )
                reviewed_payload = payload.get("atom")
                if not isinstance(reviewed_payload, Mapping):
                    raise ValueError(
                        "failure Atom review did not return an atom object"
                    )
                verdict = _required_string(payload, "verdict", max_length=64)
                if verdict not in {
                    "supported_root_cause",
                    "cautious_diagnostic",
                }:
                    raise ValueError(
                        f"unsupported failed Atom review verdict: {verdict}"
                    )
                supporting_step_ids_payload = payload.get(
                    "supporting_step_ids"
                )
                if not isinstance(supporting_step_ids_payload, list):
                    raise ValueError(
                        "supporting_step_ids must be a JSON array"
                    )
                if any(
                    isinstance(step_id, bool) or not isinstance(step_id, int)
                    for step_id in supporting_step_ids_payload
                ):
                    raise ValueError(
                        "supporting_step_ids must contain only integers"
                    )
                supporting_step_ids = sorted(set(supporting_step_ids_payload))
                available_step_ids = _evidence_step_ids(evidence)
                unknown_step_ids = sorted(
                    set(supporting_step_ids) - available_step_ids
                )
                if unknown_step_ids:
                    raise ValueError(
                        "supporting_step_ids reference unknown trajectory steps"
                    )
                if verdict == "supported_root_cause" and not supporting_step_ids:
                    raise ValueError(
                        "supported_root_cause requires supporting_step_ids"
                    )
                _required_string(payload, "rationale", max_length=1200)
                reviewed = {
                    name: _canonicalize_reusable_syntax(
                        _required_string(
                            reviewed_payload,
                            name,
                            max_length=1200,
                        )
                    )
                    for name in (
                        "trigger",
                        "scope",
                        "decision",
                        "invariant",
                        "verification",
                        "failure_mode",
                    )
                }
            except ValueError:
                last_reasons = ["invalid structured critic response"]
                messages = [
                    *base_messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not a valid strict JSON "
                            "object matching the schema. Return only concise JSON; "
                            "do not include unescaped formulas, long whitespace, "
                            "markdown, or commentary outside the object."
                        ),
                    },
                ]
                continue
            reasons = [
                *_atom_leakage_reasons(record, reviewed),
                *_failure_atom_semantic_reasons(
                    reviewed,
                    reference_audit=reference_audit,
                    verdict=verdict,
                ),
            ]
            if not reasons:
                reviewed["_analysis_mode"] = verdict
                reviewed["_review_attempts"] = str(review_attempt + 1)
                reviewed["_review_rejection_reasons"] = json.dumps(
                    last_reasons,
                    ensure_ascii=False,
                )
                reviewed["_supporting_step_ids"] = json.dumps(
                    supporting_step_ids,
                    ensure_ascii=False,
                )
                return reviewed
            last_reasons = sorted(set(reasons))
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        "The reviewed Atom is still invalid for these reasons: "
                        f"{'; '.join(last_reasons)}. Return a fresh Atom that uses "
                        "only trace-supported, reusable guidance. Do not narrate "
                        "the failed action and do not mention benchmark evaluation "
                        "artifacts."
                    ),
                },
            ]
        fallback = _cautious_failure_diagnostic(reference_audit)
        fallback_reasons = [
            *_atom_leakage_reasons(record, fallback),
            *_failure_atom_semantic_reasons(
                fallback,
                reference_audit=reference_audit,
                verdict="cautious_diagnostic",
            ),
        ]
        if fallback_reasons:
            raise ValueError(
                "failed Experience Atom review remained unsafe after bounded "
                "diagnostic fallback: "
                + ",".join(sorted(set([*last_reasons, *fallback_reasons])))
            )
        fallback["_analysis_mode"] = "deterministic_cautious_fallback"
        fallback["_review_attempts"] = "2"
        fallback["_review_rejection_reasons"] = json.dumps(
            last_reasons,
            ensure_ascii=False,
        )
        fallback["_supporting_step_ids"] = "[]"
        return fallback

    @staticmethod
    def _to_atom(
        record: RawTrajectoryRecord,
        draft: Mapping[str, str],
        trigger_embedding: Sequence[float],
        procedure_embedding: Sequence[float],
    ) -> ExperienceAtom:
        verifier_score = record.verifier_score
        if verifier_score is None:
            reliability = 0.5
        else:
            numeric_score = float(verifier_score)
            if not math.isfinite(numeric_score):
                raise ValueError("verifier_score must be finite when provided")
            reliability = min(1.0, max(0.0, numeric_score))
        evidence_type = "single_trace"
        analysis_mode = str(
            draft.get(
                "_analysis_mode",
                (
                    "successful_trace_summary"
                    if record.success
                    else "failed_trace_unclassified"
                ),
            )
        )
        review_attempts = int(str(draft.get("_review_attempts", "0")))
        rejection_reasons_payload = json.loads(
            str(draft.get("_review_rejection_reasons", "[]"))
        )
        if not isinstance(rejection_reasons_payload, list):
            raise ValueError("review rejection reasons must be a JSON list")
        review_rejection_reasons = [
            str(reason) for reason in rejection_reasons_payload
        ]
        supporting_step_ids_payload = json.loads(
            str(draft.get("_supporting_step_ids", "[]"))
        )
        if not isinstance(supporting_step_ids_payload, list) or any(
            isinstance(step_id, bool) or not isinstance(step_id, int)
            for step_id in supporting_step_ids_payload
        ):
            raise ValueError("supporting step IDs must be a JSON list of integers")
        supporting_step_ids = sorted(set(supporting_step_ids_payload))
        identity = json.dumps(
            {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "trial_index": record.trial_index,
                "draft": dict(draft),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        atom_id = f"A_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"
        return ExperienceAtom(
            atom_id=atom_id,
            source_item_id=record.trajectory_id,
            trigger=draft["trigger"],
            scope=draft["scope"],
            decision=draft["decision"],
            invariant=draft["invariant"],
            verification=draft["verification"],
            failure_mode=draft["failure_mode"],
            evidence_type=evidence_type,
            reliability=reliability,
            trigger_embedding=tuple(trigger_embedding),
            procedure_embedding=tuple(procedure_embedding),
            metadata={
                "task_id": record.task_id,
                "trial_index": record.trial_index,
                "success": bool(record.success),
                "verifier_score": record.verifier_score,
                "verifier_feedback_present": bool(record.verifier_feedback),
                "analysis_mode": analysis_mode,
                "root_cause_review_attempts": review_attempts,
                "root_cause_review_rejection_reasons": (
                    review_rejection_reasons
                ),
                "root_cause_supporting_step_ids": supporting_step_ids,
                "reliability_semantics": (
                    "verifier_score_or_neutral_source_weight_not_probability"
                ),
            },
        )


class LocalParentSkillAnalyst:
    def __init__(
        self,
        generation: GenerationClient,
        *,
        validation_mode: str,
        tokenizer: Any,
        max_prompt_tokens: int,
        max_output_tokens: int | None,
    ) -> None:
        self.generation = generation
        self.validation_mode = validation_mode
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.max_output_tokens = max_output_tokens

    async def refresh(
        self,
        state: OtdTreeState,
        node_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ordered = sorted(
            {node_id for node_id in node_ids if not state.nodes[node_id].is_leaf},
            key=lambda node_id: (
                state.height(node_id),
                state.nodes[node_id].min_atom_id,
                node_id,
            ),
        )
        for node_id in ordered:
            node = state.nodes[node_id]
            left_id, right_id = sorted(
                node.child_ids,
                key=lambda child_id: (
                    state.nodes[child_id].min_atom_id,
                    child_id,
                ),
            )
            left = state.nodes[left_id]
            right = state.nodes[right_id]
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Create one higher-level transferable skill from two "
                            "child skills. Preserve only their shared invariant and "
                            "scope. Do not copy task-specific identifiers, exact "
                            "answers, paths, coordinates, or examples. The parent "
                            "must add a useful abstraction; do not merely paraphrase "
                            "one child. Return concise name, trigger, and content only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "schema": PARENT_SKILL_SCHEMA,
                                "descendant_atom_count": node.leaf_count,
                                "children": [
                                    _minimal_skill(left.skill),
                                    _minimal_skill(right.skill),
                                ],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ]
                skill = await self._generate_safe_parent(
                    messages,
                    debug_metadata={
                        "component": "cdost_parent_analyst",
                        "node_id": node_id,
                        "left_id": left_id,
                        "right_id": right_id,
                        "descendant_atom_count": node.leaf_count,
                    },
                )
                skill["confidence"] = min(
                    float(left.skill.get("confidence", 0.0) or 0.0),
                    float(right.skill.get("confidence", 0.0) or 0.0),
                )
                duplicate_child_ids = [
                    child.node_id
                    for child in (left, right)
                    if _normalized_skill_text(skill)
                    == _normalized_skill_text(child.skill)
                ]
                descendant_atom_ids = state.descendant_atom_ids(node_id)
                descendant_source_ids = [
                    state.atoms[atom_id].source_item_id
                    for atom_id in descendant_atom_ids
                    if atom_id in state.atoms
                ]
                provenance_complete = (
                    len(descendant_atom_ids) == node.leaf_count
                    and len(descendant_source_ids) == node.leaf_count
                    and len(set(descendant_source_ids)) == node.leaf_count
                    and all(descendant_source_ids)
                )
                passed = (
                    node.leaf_count >= 2
                    and not duplicate_child_ids
                    and provenance_complete
                    and all(skill.get(field_name) for field_name in ("name", "trigger", "content"))
                )
                node.skill = skill
                node.retrievable = bool(passed)
                node.validation_mode = self.validation_mode
                node.structural_certificate = {
                    "passed": bool(passed),
                    "descendant_atom_count": node.leaf_count,
                    "duplicate_child_ids": duplicate_child_ids,
                    "provenance_complete": provenance_complete,
                    "behavioral_replay_performed": False,
                }
                events.append(
                    {
                        "node_id": node_id,
                        "status": "accepted" if passed else "structural_reject",
                        "structural_certificate": dict(node.structural_certificate),
                    }
                )
            except Exception as exc:
                node.skill = {}
                node.retrievable = False
                node.validation_mode = self.validation_mode
                node.structural_certificate = {
                    "passed": False,
                    "parent_summary_error": type(exc).__name__,
                    "behavioral_replay_performed": False,
                }
                events.append(
                    {
                        "node_id": node_id,
                        "status": "generation_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return events

    async def _generate_safe_parent(
        self,
        messages: list[dict[str, str]],
        *,
        debug_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        base_messages = list(messages)
        current_messages = list(base_messages)
        for semantic_attempt in range(3):
            _require_prompt_budget(
                self.tokenizer,
                current_messages,
                self.max_prompt_tokens,
                component="OTD parent skill generation",
            )
            payload = await self.generation.chat_json(
                current_messages,
                schema_name="CertifiedOtdParentSkill",
                guided_json=PARENT_SKILL_SCHEMA,
                max_tokens=self.max_output_tokens,
                retries=1,
                debug_metadata={
                    **debug_metadata,
                    "semantic_attempt": semantic_attempt + 1,
                },
            )
            skill = {
                name: _canonicalize_reusable_syntax(
                    _required_string(
                        payload,
                        name,
                        max_length=int(
                            PARENT_SKILL_SCHEMA["properties"][name]["maxLength"]
                        ),
                    )
                )
                for name in ("name", "trigger", "content")
            }
            reasons = _unsafe_reusable_text_reasons(
                "\n".join(skill.values())
            )
            if not reasons:
                return skill
            current_messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        f"Revision {semantic_attempt + 1}: the previous parent was "
                        "rejected for these categories only: "
                        f"{'; '.join(sorted(set(reasons)))}. Write a fresh, concise "
                        "shared invariant without quoting the previous output or "
                        "using paths, URLs, coordinates, examples, or exact values."
                    ),
                },
            ]
        raise ValueError("failed to produce a leakage-free parent skill")

async def build_certified_otd_tree_from_records(config: Any) -> dict[str, Any]:
    return await _build(config, dynamic=False)


async def build_certified_otd_dynamic_tree_from_records(config: Any) -> dict[str, Any]:
    return await _build(config, dynamic=True)


def _write_vector_cache_manifest(
    *,
    out: Path,
    config: Any,
    atoms: Sequence[ExperienceAtom],
    skillbank_index_path: str | Path,
) -> dict[str, Any]:
    cache_path = str(config.embedding.cache_path or "").strip()
    if not cache_path:
        raise ValueError(
            "certified_dual_view_otd requires embedding.cache_path so "
            "static/dynamic vector identity can be audited"
        )
    trigger_namespace = embedding_cache_namespace(
        config.embedding,
        "cdost_atom_trigger",
        model_name=config.embedding.model,
    )
    procedure_namespace = embedding_cache_namespace(
        config.embedding,
        "cdost_atom_procedure",
        model_name=config.embedding.model,
    )
    requirements: list[dict[str, Any]] = []
    for atom in atoms:
        requirements.extend(
            [
                {
                    "namespace": trigger_namespace,
                    "text": _atom_trigger_embedding_text(
                        trigger=atom.trigger,
                        scope=atom.scope,
                    ),
                    "normalized_vector": atom.trigger_embedding,
                    "purpose": "experience_atom_trigger",
                    "item_id": atom.atom_id,
                },
                {
                    "namespace": procedure_namespace,
                    "text": _atom_procedure_embedding_text(
                        decision=atom.decision,
                        invariant=atom.invariant,
                        verification=atom.verification,
                        failure_mode=atom.failure_mode,
                    ),
                    "normalized_vector": atom.procedure_embedding,
                    "purpose": "experience_atom_procedure",
                    "item_id": atom.atom_id,
                },
            ]
        )

    index_path = Path(skillbank_index_path)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    documents = index_payload.get("documents")
    vectors = index_payload.get("embeddings")
    if (
        index_payload.get("format")
        != "dynamix_skillbank_embedding_index_v2"
        or not isinstance(documents, list)
        or not isinstance(vectors, list)
        or len(documents) != len(vectors)
    ):
        raise ValueError(f"invalid skillbank embedding index: {index_path}")
    node_namespace = skillbank_vector_cache_namespace(
        base_url=str(index_payload.get("base_url") or ""),
        model=str(index_payload.get("model") or ""),
        api_key_fingerprint_value=str(
            index_payload.get("api_key_fingerprint") or ""
        ),
        embedding_protocol=dict(
            index_payload.get("embedding_protocol") or {}
        ),
    )
    for document, vector in zip(documents, vectors):
        embedding_text = str(document.get("embedding_text") or "")
        node_id = str(document.get("node_id") or "")
        if not embedding_text or not node_id:
            raise ValueError(
                f"skillbank index contains an incomplete document: {index_path}"
            )
        requirements.append(
            {
                "namespace": node_namespace,
                "text": embedding_text,
                "normalized_vector": vector,
                "purpose": "retrievable_node",
                "item_id": node_id,
            }
        )

    manifest_path = out / "embedding_vector_cache_manifest.json"
    payload = write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=manifest_path,
        requirements=requirements,
    )
    validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=manifest_path,
    )
    return payload


async def _build(config: Any, *, dynamic: bool) -> dict[str, Any]:
    from .pipeline import (
        _load_records_for_protocol,
        _refresh_skillbank_index,
        _write_runtime_artifacts,
    )

    otd_config = CertifiedOtdConfig.from_mapping(
        dict(config.hierarchy or {}).get("otd", {})
    )
    if not bool(config.enforce_dataset_order):
        raise ValueError(
            "certified_dual_view_otd requires enforce_dataset_order=true"
        )
    if bool(config.dynamic.resume_from_snapshots):
        raise ValueError(
            "certified_dual_view_otd does not yet support fingerprinted snapshot "
            "resume; set dynamic.resume_from_snapshots=false"
        )
    if config.dynamic.shuffle_seed is not None:
        raise ValueError(
            "certified_dual_view_otd requires dataset-order arrivals for the "
            "static/dynamic prefix-consistency protocol; set "
            "dynamic.shuffle_seed=null (CLI: --dynamic-shuffle-seed -1)"
        )
    if not bool(config.dynamic.snapshot_include_embeddings):
        raise ValueError(
            "certified_dual_view_otd requires "
            "dynamic.snapshot_include_embeddings=true"
        )
    if dynamic and not otd_config.atom_cache_path:
        raise ValueError(
            "controlled certified_dual_view_otd dynamic runs require a frozen "
            "atom_cache_path from the matching static extraction"
        )
    if not str(config.embedding.cache_path or "").strip():
        raise ValueError(
            "certified_dual_view_otd requires embedding.cache_path"
        )
    if config.embedding.cache_write_policy != "first_write_wins":
        raise ValueError(
            "certified_dual_view_otd requires "
            "embedding.cache_write_policy='first_write_wins'"
        )
    if dynamic:
        source_tree = Path(str(otd_config.atom_cache_path)).resolve().parent
        _validate_source_build_output(
            Path(str(otd_config.atom_cache_path))
        )
        _validate_source_build_output(
            source_tree / "embedding_vector_cache_manifest.json"
        )
        validate_embedding_cache_manifest(
            cache_path=config.embedding.cache_path,
            manifest_path=(
                source_tree / "embedding_vector_cache_manifest.json"
            ),
        )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    node_bank_dir = _resolve_skill_output_dir(
        out,
        config.skill_output_dir_name,
    )
    if not config.generation.debug_dir:
        config.generation.debug_dir = str(out / "analysis" / "generation_debug")
    _prepare_otd_analyst_config(config, out)
    _write_runtime_artifacts(config, out)
    records = _load_records_for_protocol(config, out)
    if dynamic:
        selected_initial_count = min(
            max(1, int(config.dynamic.initial_count)),
            len(records),
        )
        selected_arrival_count = max(
            0,
            len(records) - selected_initial_count,
        )
        if int(config.dynamic.arrival_count) > 0:
            selected_arrival_count = min(
                selected_arrival_count,
                int(config.dynamic.arrival_count),
            )
        records = list(
            records[: selected_initial_count + selected_arrival_count]
        )
    _require_unique_record_ids(records)
    tokenizer = _tokenizer_for_config(config)

    atom_generation = GenerationClient(
        replace(config.generation, temperature=float(otd_config.atom_temperature))
    )
    parent_generation = GenerationClient(
        replace(config.generation, temperature=float(otd_config.parent_temperature))
    )
    embedding = EmbeddingClient(config.embedding)
    atom_analyst = ExperienceAtomAnalyst(
        atom_generation,
        embedding,
        config=otd_config,
        tokenizer=tokenizer,
        max_prompt_tokens=int(config.analyst.max_prompt_tokens),
        max_output_tokens=config.analyst.max_output_tokens,
        max_evidence_chars=int(
            config.analyst.analysis_bundle_max_chars or 60000
        ),
    )
    records_fingerprint = _records_fingerprint(records)
    atom_protocol_fingerprint = _atom_protocol_fingerprint(
        config,
        otd_config,
    )
    try:
        if otd_config.atom_cache_path:
            atoms, excluded = _load_atom_cache(
                Path(otd_config.atom_cache_path),
                records,
                records_fingerprint=records_fingerprint,
                atom_protocol_fingerprint=atom_protocol_fingerprint,
            )
            atom_source = "frozen_cache"
        else:
            atoms, excluded = await atom_analyst.extract_many(records)
            atom_source = "generated"
    finally:
        embedding_truncation_events = len(embedding.truncation_events)
        embedding.close()
    _require_complete_atom_identity(records, atoms)
    (out / "experience_atoms.json").write_text(
        json.dumps(
            {
                "format": "certified_experience_atoms_v1",
                "record_count": len(records),
                "atom_count": len(atoms),
                "excluded_count": len(excluded),
                "excluded_records": excluded,
                "atom_source": atom_source,
                "atom_cache_path": otd_config.atom_cache_path,
                "records_fingerprint": records_fingerprint,
                "atom_protocol_fingerprint": atom_protocol_fingerprint,
                "atoms": [atom.to_dict() for atom in atoms],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = OtdTreeState(
        dual_view_lambda=otd_config.dual_view_lambda,
        tie_epsilon=otd_config.tie_epsilon,
    )
    parent_analyst = LocalParentSkillAnalyst(
        parent_generation,
        validation_mode=otd_config.validation_mode,
        tokenizer=tokenizer,
        max_prompt_tokens=int(config.analyst.max_prompt_tokens),
        max_output_tokens=config.analyst.max_output_tokens,
    )
    insertion_events: list[dict[str, Any]] = []
    parent_events: list[dict[str, Any]] = []
    excluded_ids = {
        str(entry.get("trajectory_id", ""))
        for entry in excluded
        if entry.get("trajectory_id")
    }
    excluded_initial_count = 0
    excluded_arrival_count = 0
    insertion_count = 0
    updated_count = 0
    snapshot_count = 0
    initial_count = len(atoms)

    if dynamic:
        atom_by_source = {atom.source_item_id: atom for atom in atoms}
        initial_count = min(
            max(1, int(config.dynamic.initial_count)),
            len(records),
        )
        arrival_limit = int(config.dynamic.arrival_count)
        initial_records = list(records[:initial_count])
        arrival_records = list(records[initial_count:])
        if arrival_limit > 0:
            arrival_records = arrival_records[:arrival_limit]
        initial_atoms = [
            atom_by_source[record.trajectory_id]
            for record in initial_records
            if record.trajectory_id in atom_by_source
        ]
        arrival_atoms = [
            atom_by_source[record.trajectory_id]
            for record in arrival_records
            if record.trajectory_id in atom_by_source
        ]
        excluded_initial_count = sum(
            record.trajectory_id in excluded_ids
            for record in initial_records
        )
        excluded_arrival_count = sum(
            record.trajectory_id in excluded_ids
            for record in arrival_records
        )
        insertion_count = len(arrival_records)
        updated_count = len(arrival_atoms)
        for atom in initial_atoms:
            insertion_events.append(_insertion_payload(state.insert(atom)))
        parent_events.extend(
            await parent_analyst.refresh(
                state,
                [node_id for node_id, node in state.nodes.items() if not node.is_leaf],
            )
        )
        snapshot_interval = max(1, int(config.dynamic.update_batch_size))
        for arrival_index, atom in enumerate(arrival_atoms, start=1):
            result = state.insert(atom)
            insertion_events.append(_insertion_payload(result))
            parent_events.extend(
                await parent_analyst.refresh(
                    state,
                    result.changed_internal_node_ids,
                )
            )
            if (
                arrival_index % snapshot_interval == 0
                or arrival_index == len(arrival_atoms)
            ):
                snapshot_dir = (
                    out
                    / "dynamic_snapshots"
                    / f"arrival_{arrival_index:04d}"
                )
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                _write_tree_artifacts(
                    state,
                    snapshot_dir,
                    config=config,
                    otd_config=otd_config,
                )
                snapshot_count += 1
    else:
        initial_atoms = atoms
        arrival_atoms = []
        insertion_count = 0
        updated_count = 0
        for atom in atoms:
            insertion_events.append(_insertion_payload(state.insert(atom)))
        parent_events.extend(
            await parent_analyst.refresh(
                state,
                [node_id for node_id, node in state.nodes.items() if not node.is_leaf],
            )
        )

    _write_jsonl(out / "otd_insertions.jsonl", insertion_events)
    _write_jsonl(out / "otd_parent_updates.jsonl", parent_events)
    parent_generation_errors = [
        event
        for event in parent_events
        if event.get("status") == "generation_error"
    ]
    if parent_generation_errors:
        failure = {
            "stage": "parent_skill_generation",
            "error_count": len(parent_generation_errors),
            "errors": parent_generation_errors,
            "heldout_allowed": False,
        }
        (out / "build_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            "certified_dual_view_otd parent generation failed for "
            f"{len(parent_generation_errors)} nodes; heldout is blocked"
        )
    beta_audit = audit_observed_beta_separation(
        atoms,
        dual_view_lambda=otd_config.dual_view_lambda,
        tie_epsilon=otd_config.tie_epsilon,
    )
    (out / "otd_observed_beta_separation.json").write_text(
        json.dumps(beta_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = _write_tree_artifacts(
        state,
        out,
        config=config,
        otd_config=otd_config,
    )
    skillbank_index = _refresh_skillbank_index(node_bank_dir, config)
    vector_cache_manifest = _write_vector_cache_manifest(
        out=out,
        config=config,
        atoms=atoms,
        skillbank_index_path=skillbank_index,
    )
    summary = {
        "scenario": "dynamic_update" if dynamic else "static_build",
        "tree_policy": "certified_dual_view_otd",
        "record_count": len(records),
        "atom_count": len(atoms),
        "atom_source": atom_source,
        "atom_cache_path": otd_config.atom_cache_path,
        "excluded_count": (
            excluded_arrival_count if dynamic else len(excluded)
        ),
        "excluded_initial_count": excluded_initial_count,
        "initial_count": (
            initial_count if dynamic else len(initial_atoms)
        ),
        "arrival_count": (
            insertion_count if dynamic else len(arrival_atoms)
        ),
        "insertion_count": insertion_count,
        "updated_count": updated_count,
        "snapshot_count": snapshot_count,
        "snapshot_interval": (
            max(1, int(config.dynamic.update_batch_size))
            if dynamic
            else 0
        ),
        "arrival_update_semantics": (
            "sequential_per_atom" if dynamic else "static_dataset_order"
        ),
        "parent_refresh_semantics": (
            "changed_path_bottom_up_per_atom"
            if dynamic
            else "all_internal_nodes_bottom_up_after_build"
        ),
        "arrival_order": "dataset",
        "configured_shuffle_seed_ignored": (
            config.dynamic.shuffle_seed if dynamic else None
        ),
        "tree_node_count": len(state.nodes),
        "node_count": int(manifest["node_count"]),
        "retrievable_node_count": int(manifest["node_count"]),
        "node_bank_dir": str(node_bank_dir),
        "node_bank_manifest": str(
            node_bank_dir / "node_bank_manifest.json"
        ),
        "skillbank_index": skillbank_index,
        "embedding_vector_cache_manifest": str(
            out / "embedding_vector_cache_manifest.json"
        ),
        "embedding_vector_cache_logical_sha256": vector_cache_manifest[
            "logical_sha256"
        ],
        "embedding_vector_cache_entry_count": vector_cache_manifest[
            "entry_count"
        ],
        "embedding_truncation_events": embedding_truncation_events,
        "parent_update_count": len(parent_events),
        "parent_generation_error_count": sum(
            event.get("status") == "generation_error"
            for event in parent_events
        ),
        "structural_diagnostics": {
            **state.structural_diagnostics(),
            **_decision_margin_diagnostics(insertion_events),
            "observed_beta_separation": beta_audit,
        },
        "guarantee_boundary": {
            "theorem_claim_enabled": beta_audit["theorem_claim_enabled"],
            "structural_theorem": (
                "beta/3 Moseley-Wang revenue approximation under "
                "beta-well-separated fixed nonnegative similarities"
                if beta_audit["theorem_claim_enabled"]
                else None
            ),
            "theorem_disabled_reason": (
                (
                    "nonzero tie_epsilon changes exact Online Top-Down "
                    "comparisons"
                )
                if otd_config.tie_epsilon != 0.0
                else (
                    (
                        "the observed assumption was satisfied only "
                        "vacuously; no nonempty theorem constraint was audited"
                    )
                    if beta_audit["vacuous"]
                    else (
                        "the observed arrival stream has no positive beta "
                        "bound"
                        if not beta_audit["theorem_claim_enabled"]
                        else None
                    )
                )
            ),
            "well_separation_verified": beta_audit[
                "theorem_claim_enabled"
            ],
            "assumption_satisfied_on_observed_stream": beta_audit[
                "assumption_satisfied_on_observed_stream"
            ],
            "assumption_status": beta_audit["assumption_status"],
            "antecedent_count": beta_audit["antecedent_count"],
            "constraint_count": beta_audit["constraint_count"],
            "zero_rhs_constraint_count": beta_audit[
                "zero_rhs_constraint_count"
            ],
            "vacuous": beta_audit["vacuous"],
            "exact_otd_comparisons": beta_audit[
                "exact_otd_comparisons"
            ],
            "well_separation_scope": "observed_arrivals_only",
            "observed_beta": beta_audit["observed_beta"],
            "observed_beta_over_3": beta_audit["observed_beta_over_3"],
            "future_arrivals_certified": False,
            "llm_semantics_guaranteed": False,
            "behavioral_replay_performed": False,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _write_tree_artifacts(
    state: OtdTreeState,
    out: Path,
    *,
    config: Any,
    otd_config: CertifiedOtdConfig,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    state_payload = state.to_dict()
    state_path = out / "otd_tree_state.json"
    state_text = json.dumps(state_payload, ensure_ascii=False, indent=2)
    state_path.write_text(state_text, encoding="utf-8")
    structure_path = out / "otd_tree_structure.json"
    structure_text = json.dumps(
        state.to_structural_dict(),
        ensure_ascii=False,
        indent=2,
    )
    structure_path.write_text(structure_text, encoding="utf-8")
    diagnostics = state.structural_diagnostics()
    (out / "otd_structural_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    skill_dir = out / config.skill_output_dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = _nodebank_manifest(
        state,
        output_dir=skill_dir,
        config=config,
        otd_config=otd_config,
        authoritative_tree={
            "state_relative_path": os.path.relpath(state_path, skill_dir),
            "state_sha256": hashlib.sha256(
                state_text.encode("utf-8")
            ).hexdigest(),
            "structure_relative_path": os.path.relpath(
                structure_path,
                skill_dir,
            ),
            "structure_sha256": hashlib.sha256(
                structure_text.encode("utf-8")
            ).hexdigest(),
            "structural_node_count": len(state.nodes),
        },
    )
    (skill_dir / "node_bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _nodebank_manifest(
    state: OtdTreeState,
    *,
    output_dir: Path,
    config: Any,
    otd_config: CertifiedOtdConfig,
    authoritative_tree: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = _tokenizer_for_config(config)
    levels = {
        node_id: state.height(node_id) + 1
        for node_id in state.nodes
    }
    nodes: list[dict[str, Any]] = []
    for node_id, node in sorted(state.nodes.items()):
        if not node.retrievable or not node.skill:
            continue
        name = _required_string(node.skill, "name")
        trigger = _required_string(node.skill, "trigger")
        content = _required_string(node.skill, "content")
        embedding_text = (
            f"name: {name}\n"
            f"trigger: {trigger}\n"
            f"content: {content}"
        )
        prompt_text = render_cdost_node_prompt(
            name=name,
            trigger=trigger,
            content=content,
        )
        nodes.append(
            {
                "node_id": node_id,
                "item_id": node.atom_id or node_id,
                "level": levels[node_id],
                "support_mass": float(node.leaf_count),
                "confidence": float(node.skill.get("confidence", 0.0) or 0.0),
                "name": name,
                "trigger": trigger,
                "content": content,
                "embedding_text": embedding_text,
                "prompt_text": prompt_text,
                "source_community_id": "",
                "source_member_count": node.leaf_count,
                "analyst_mode": (
                    "cdost_experience_atom"
                    if node.is_leaf
                    else "cdost_parent_abstraction"
                ),
                "sha256": hashlib.sha256(
                    embedding_text.encode("utf-8")
                ).hexdigest(),
                "parent_node_id": node.parent_id,
                "child_node_ids": list(node.child_ids),
                "descendant_atom_count": node.leaf_count,
                "token_cost": max(1, tokenizer.count(prompt_text)),
                "validation_mode": node.validation_mode,
                "structural_certificate": dict(node.structural_certificate),
            }
        )
    tree_index = {
        "root_node_id": state.root_node_id,
        "children_by_node": {
            node_id: list(node.child_ids)
            for node_id, node in sorted(state.nodes.items())
        },
    }
    return {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "certified_dual_view_otd",
        "output_dir": str(output_dir),
        "root_node_id": state.root_node_id,
        "node_count": len(nodes),
        "authoritative_tree": {
            **dict(authoritative_tree),
            "retrievable_node_count": len(nodes),
        },
        "nodes": nodes,
        "item_to_node_ids": {
            str(node["item_id"]): [str(node["node_id"])]
            for node in nodes
        },
        "tree_index": tree_index,
        "export_policy": {
            "retrieval_unit": "experience_atom_or_validated_parent",
            "embedding_fields": ["name", "trigger", "content"],
            "trajectory_items_exported": False,
            "heldout_retrieval": "tree_antichain_knapsack",
            "top_k_is_cardinality_cap": True,
            "token_budget": int(otd_config.retrieval_token_budget),
            "token_unit": int(otd_config.retrieval_token_unit),
            "exact_search_max_states": int(
                otd_config.retrieval_exact_search_max_states
            ),
            "fixed_prompt_overhead_tokens": max(
                1,
                tokenizer.count(retrieved_experience_preamble()),
            ),
            "single_parent_structure": True,
            "ancestor_descendant_co_selection": False,
            "validation_mode": otd_config.validation_mode,
            "relevance_transform": "shifted_cosine_(1_plus_cosine)_over_2",
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


def _tokenizer_for_config(config: Any) -> Any:
    model = (
        getattr(config.analyst, "tokenizer_model", None)
        or getattr(config.embedding, "tokenizer_model", None)
    )
    allow_fallback = (
        not bool(config.analyst.tokenizer_required)
        and bool(
        getattr(config.analyst, "allow_regex_tokenizer_fallback", True)
        )
    )
    return get_tokenizer(model, allow_regex_fallback=allow_fallback)


def _prepare_otd_analyst_config(config: Any, out: Path) -> None:
    summary_budget = dict(config.hierarchy or {}).get("summary_budget", {})
    analyst_budget_was_overridden = config.analyst.max_prompt_tokens is not None
    if config.analyst.max_prompt_tokens is None:
        max_model_tokens = int(summary_budget.get("max_model_tokens", 100000))
        budget_ratio = float(summary_budget.get("budget_ratio", 0.85))
        config.analyst.max_prompt_tokens = int(max_model_tokens * budget_ratio)
    if not config.analyst.prompt_token_report_path:
        config.analyst.prompt_token_report_path = str(
            out / "analysis" / "cdost_prompt_token_report.json"
        )
    payload = {
        "tree_policy": "certified_dual_view_otd",
        "analyst_max_prompt_tokens": config.analyst.max_prompt_tokens,
        "analyst_max_output_tokens": config.analyst.max_output_tokens,
        "analyst_tokenizer_model": config.analyst.tokenizer_model,
        "analyst_tokenizer_required": config.analyst.tokenizer_required,
        "analyst_allow_regex_tokenizer_fallback": (
            config.analyst.allow_regex_tokenizer_fallback
        ),
        "summary_budget": summary_budget,
        "source": (
            "analyst.max_prompt_tokens override"
            if analyst_budget_was_overridden
            else "hierarchy.summary_budget"
        ),
    }
    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "analyst_budget_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _insertion_payload(result: OtdInsertionResult) -> dict[str, Any]:
    return {
        "atom_id": result.atom_id,
        "leaf_node_id": result.leaf_node_id,
        "root_node_id": result.root_node_id,
        "created_node_ids": list(result.created_node_ids),
        "changed_internal_node_ids": list(result.changed_internal_node_ids),
        "decisions": [
            {
                "node_id": decision.node_id,
                "action": decision.action,
                "within_similarity": decision.within_similarity,
                "cross_similarity": decision.cross_similarity,
                "within_similarity_sum": decision.within_similarity_sum,
                "cross_similarity_sum": decision.cross_similarity_sum,
                "left_cross_similarity": decision.left_cross_similarity,
                "right_cross_similarity": decision.right_cross_similarity,
                "selected_child_id": decision.selected_child_id,
            }
            for decision in result.decisions
        ],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def _decision_margin_diagnostics(
    insertion_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    margins = [
        abs(
            float(decision["within_similarity"])
            - float(decision["cross_similarity"])
        )
        for event in insertion_events
        for decision in event.get("decisions", [])
        if decision.get("action") != "merge"
        or float(decision.get("within_similarity_sum", 0.0)) > 0.0
    ]
    return {
        "observed_average_similarity_decision_count": len(margins),
        "minimum_observed_average_similarity_margin": (
            min(margins) if margins else None
        ),
        "observed_margin_is_not_a_beta_certificate": True,
    }


def _load_atom_cache(
    path: Path,
    records: Sequence[RawTrajectoryRecord],
    *,
    records_fingerprint: str,
    atom_protocol_fingerprint: str,
) -> tuple[list[ExperienceAtom], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "certified_experience_atoms_v1":
        raise ValueError(f"unsupported Experience Atom cache: {path}")
    if payload.get("records_fingerprint") != records_fingerprint:
        raise ValueError("Experience Atom cache record fingerprint mismatch")
    if payload.get("atom_protocol_fingerprint") != atom_protocol_fingerprint:
        raise ValueError("Experience Atom cache extraction/embedding protocol mismatch")
    atoms_payload = list(payload.get("atoms", []))
    if int(payload.get("record_count", -1)) != len(records):
        raise ValueError("Experience Atom cache record_count mismatch")
    if int(payload.get("atom_count", -1)) != len(atoms_payload):
        raise ValueError("Experience Atom cache atom_count mismatch")
    cached_atoms = [ExperienceAtom.from_dict(item) for item in atoms_payload]
    atom_ids = [atom.atom_id for atom in cached_atoms]
    source_ids = [atom.source_item_id for atom in cached_atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("Experience Atom cache contains duplicate atom_id values")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Experience Atom cache contains duplicate source_item_id values")
    expected_ids = [record.trajectory_id for record in records]
    if source_ids != expected_ids:
        raise ValueError(
            "Experience Atom cache order does not match record order"
        )
    for record, atom in zip(records, cached_atoms):
        if atom.evidence_type != "single_trace":
            raise ValueError(
                "Experience Atom cache must contain only single_trace atoms"
            )
        draft = {
            "trigger": atom.trigger,
            "scope": atom.scope,
            "decision": atom.decision,
            "invariant": atom.invariant,
            "verification": atom.verification,
            "failure_mode": atom.failure_mode,
        }
        leakage_reasons = _atom_leakage_reasons(record, draft)
        if leakage_reasons:
            raise ValueError(
                "Experience Atom cache leaked task-specific content for "
                f"{record.trajectory_id}: {', '.join(leakage_reasons)}"
            )
    excluded = list(payload.get("excluded_records", []))
    if excluded:
        raise ValueError(
            "Experience Atom cache contains excluded records and cannot provide "
            "a complete order-preserving build"
        )
    return cached_atoms, []


def _require_unique_record_ids(
    records: Sequence[RawTrajectoryRecord],
) -> None:
    source_ids = [str(record.trajectory_id).strip() for record in records]
    if any(not source_id for source_id in source_ids):
        raise ValueError("every trajectory must have a non-empty trajectory_id")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for source_id in source_ids:
        if source_id in seen:
            duplicates.add(source_id)
        seen.add(source_id)
    if duplicates:
        raise ValueError(
            "certified_dual_view_otd requires unique trajectory_id values: "
            f"{sorted(duplicates)[:10]}"
        )


def _require_complete_atom_identity(
    records: Sequence[RawTrajectoryRecord],
    atoms: Sequence[ExperienceAtom],
) -> None:
    _require_unique_record_ids(records)
    atom_ids = [atom.atom_id for atom in atoms]
    source_ids = [atom.source_item_id for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("Experience Atoms contain duplicate atom_id values")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Experience Atoms contain duplicate source_item_id values")
    expected_ids = [record.trajectory_id for record in records]
    if source_ids != expected_ids:
        raise ValueError(
            "Experience Atoms must preserve the complete ordered trajectory identity"
        )


def _records_fingerprint(records: Sequence[RawTrajectoryRecord]) -> str:
    payload = [record.to_dict() for record in records]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atom_protocol_fingerprint(
    config: Any,
    otd_config: CertifiedOtdConfig,
) -> str:
    pipeline_source = Path(__file__).read_text(encoding="utf-8")
    implementation_source = "\n\n".join(
        inspect.getsource(component)
        for component in (
            certified_otd_module,
            clients_module,
            openai_compat_module,
            tokenization_module,
            trace_views_module,
            ExperienceAtomAnalyst,
            ExperienceAtom,
            GenerationClient,
            EmbeddingClient,
            _atom_leakage_reasons,
            _answer_literals,
            _unsafe_reusable_text_reasons,
            render_compact_analysis_bundle_text,
        )
    )
    payload = {
        "format": "cdost_atom_protocol_v6",
        "schema": ATOM_SCHEMA,
        "prompt_version": "cdost_atom_v2_terminal_answer_guard",
        "dual_view_contract": "trigger_scope__procedure_invariant_verification_failure__unit_l2_v1",
        "implementation_source_sha256": hashlib.sha256(
            (pipeline_source + "\n\n" + implementation_source).encode("utf-8")
        ).hexdigest(),
        "generation": generation_protocol_payload(config.generation)
        | {
            "temperature": otd_config.atom_temperature,
            "max_output_tokens": config.analyst.max_output_tokens,
        },
        "embedding": embedding_protocol_payload(config.embedding),
        "analysis": {
            "max_prompt_tokens": config.analyst.max_prompt_tokens,
            "max_evidence_chars": (
                config.analyst.analysis_bundle_max_chars or 60000
            ),
            "tokenizer_model": config.analyst.tokenizer_model,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _minimal_skill(skill: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: _required_string(skill, name)
        for name in ("name", "trigger", "content")
    }


def _normalized_skill_text(skill: Mapping[str, Any]) -> str:
    return " ".join(
        " ".join(str(skill.get(name, "")).lower().split())
        for name in ("name", "trigger", "content")
    )


def _required_string(
    payload: Mapping[str, Any],
    name: str,
    *,
    max_length: int | None = None,
) -> str:
    value = str(payload.get(name, "") or "").strip()
    if not value:
        raise ValueError(f"missing required string field: {name}")
    if max_length is not None and len(value) > int(max_length):
        raise ValueError(
            f"string field {name} exceeds max length: "
            f"{len(value)} > {max_length}"
        )
    return value


def _require_prompt_budget(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    max_prompt_tokens: int,
    *,
    component: str,
) -> None:
    token_count = sum(
        tokenizer.count(
            f"{message.get('role', '')}\n{message.get('content', '')}"
        )
        for message in messages
    )
    if token_count > int(max_prompt_tokens):
        raise ValueError(
            f"{component} prompt exceeds configured budget: "
            f"{token_count} > {max_prompt_tokens}"
        )


def _atom_leakage_reasons(
    record: RawTrajectoryRecord,
    draft: Mapping[str, str],
) -> list[str]:
    text = "\n".join(str(value) for value in draft.values())
    reasons = _unsafe_reusable_text_reasons(text)
    for name, value in draft.items():
        without_redactions = re.sub(
            r"<redacted>",
            " ",
            str(value),
            flags=re.IGNORECASE,
        )
        if not any(character.isalnum() for character in without_redactions):
            reasons.append(f"insufficient reusable content in {name}")
    protected_literals = _atom_protected_literals(record)
    for literal in sorted(protected_literals, key=len, reverse=True):
        if _contains_source_literal(text, literal):
            reasons.append(f"source literal {literal!r}")
    return sorted(set(reasons))


def _failure_atom_semantic_reasons(
    draft: Mapping[str, str],
    *,
    reference_audit: Mapping[str, Any],
    verdict: str,
) -> list[str]:
    text = "\n".join(str(value) for value in draft.values())
    reasons: list[str] = []
    if re.search(
        r"\b(?:golden(?:\s+artifact)?|ground[\s_-]*truth|"
        r"reference\s+answer|benchmark\s+(?:evaluator|verifier)|verifier)\b",
        text,
        flags=re.IGNORECASE,
    ):
        reasons.append("benchmark evaluation artifact in reusable atom")
    decision = str(draft.get("decision", ""))
    if re.search(
        r"\b(?:the\s+)?agent\s+(?:decides?|uses?|chose|chooses|"
        r"applied|applies|generated|generates|wrote|writes)\b",
        decision,
        flags=re.IGNORECASE,
    ):
        reasons.append("decision narrates the failed agent action")
    if verdict == "cautious_diagnostic":
        if not re.match(
            r"^\s*(?:before\s+\w+[\s,]+)?(?:audit|check|compare|diagnose|"
            r"establish|identify|inspect|map|reconstruct|review|trace|"
            r"validate|verify)\b",
            decision,
            flags=re.IGNORECASE,
        ):
            reasons.append(
                "cautious diagnostic decision is prescriptive"
            )
    missing_references = reference_audit.get(
        "instruction_references_not_explicitly_used_by_actions"
    )
    if isinstance(missing_references, Sequence) and not isinstance(
        missing_references, (str, bytes)
    ) and missing_references:
        if not re.search(r"\b(?:source|input|upstream)\b", text, re.IGNORECASE):
            reasons.append("lexical source-coverage signal is not represented")
        if not re.search(r"\b(?:target|output|destination)\b", text, re.IGNORECASE):
            reasons.append("source and output roles are not distinguished")
    return sorted(set(reasons))


def _evidence_step_ids(evidence: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(
            r'(?m)^\s*"step_id"\s*:\s*(\d+),?\s*$',
            evidence,
        )
    }


def _cautious_failure_diagnostic(
    reference_audit: Mapping[str, Any],
) -> dict[str, str]:
    missing_references = reference_audit.get(
        "instruction_references_not_explicitly_used_by_actions"
    )
    if isinstance(missing_references, Sequence) and not isinstance(
        missing_references, (str, bytes)
    ) and missing_references:
        return {
            "trigger": (
                "A spreadsheet task names multiple regions whose roles must be "
                "distinguished."
            ),
            "scope": (
                "Transformations whose output depends on a separate input table."
            ),
            "decision": (
                "Map each named region as source, output, example, or helper area "
                "before editing; identify label and value fields, then determine "
                "whether the requested operation is row-wise or aggregate."
            ),
            "invariant": (
                "Every region confirmed as an input participates in the computation, "
                "while destination and example cells are not treated as source."
            ),
            "verification": (
                "Recalculate the workbook and independently check representative "
                "outputs against a manual computation from the input table."
            ),
            "failure_mode": (
                "Misclassifying a source, example, or destination region can produce "
                "plausible values from the wrong data."
            ),
        }
    return {
        "trigger": "A failed structured-artifact edit has an uncertain root cause.",
        "scope": "Tasks where the recorded evidence does not support a specific fix.",
        "decision": (
            "Reconstruct the requested transformation from the instruction and "
            "observed artifact schema, then identify the first action that diverges "
            "before reusing any implementation rule."
        ),
        "invariant": (
            "Reusable guidance distinguishes observed evidence from unsupported "
            "assumptions and does not repeat an unverified failed action."
        ),
        "verification": (
            "Recalculate or reopen the artifact and test the reconstructed operation "
            "on representative inputs before applying it broadly."
        ),
        "failure_mode": (
            "Guessing a specific repair from incomplete evidence can turn the "
            "original mistake into a misleading reusable rule."
        ),
    }


def _atom_protected_literals(record: RawTrajectoryRecord) -> set[str]:
    path_literals = {
        str(value)
        for value in (record.spreadsheet_path, record.output_path)
        if str(value or "").strip()
    }
    literals = {
        record.trajectory_id,
        record.task_id,
        record.answer_position,
    }
    literals.update(path_literals)
    literals.update(_answer_literals(record))
    for key in ("source_files", "source_docs", "source_hints"):
        value = record.extra.get(key)
        if isinstance(value, str):
            literals.add(value)
            if key == "source_files":
                path_literals.add(value)
        elif isinstance(value, Sequence):
            literals.update(str(item) for item in value)
            if key == "source_files":
                path_literals.update(str(item) for item in value)
    protected = {
        str(value).strip()
        for value in literals
        if str(value or "").strip()
    }
    protected.update(
        Path(value).name
        for value in path_literals
        if Path(value).name
    )
    return protected


def _redact_atom_source_literals(
    record: RawTrajectoryRecord,
    text: str,
) -> str:
    redacted = str(text)
    for literal in sorted(
        _atom_protected_literals(record),
        key=len,
        reverse=True,
    ):
        pattern, flags = _source_literal_pattern(literal)
        redacted = re.sub(pattern, "<redacted>", redacted, flags=flags)
    return redacted


def _contains_source_literal(text: str, candidate: str) -> bool:
    pattern, flags = _source_literal_pattern(candidate)
    return bool(re.search(pattern, text, flags=flags))


def _source_literal_pattern(candidate: str) -> tuple[str, int]:
    candidate = candidate.strip()
    if not candidate:
        return r"(?!x)x", 0
    if re.fullmatch(r"[A-Za-z]", candidate):
        escaped = re.escape(candidate)
        return (
            rf"\b(?:answer|choose|emit|label|output(?:\s+value)?|"
            rf"produce|return|select|task|trajectory|write)\s*"
            rf"(?:(?:is|as)\s*|[:=]\s*)?['\"]?(?i:{escaped})['\"]?"
            rf"(?=$|[\n)\]}}'\".,;!?])",
            0,
        )
    if len(candidate) <= 3 and re.fullmatch(
        r"[A-Za-z0-9]+[^\w\s]+",
        candidate,
    ):
        return (
            rf"(?<![\w.-]){re.escape(candidate)}(?![\w-])",
            re.IGNORECASE,
        )
    if re.fullmatch(r"[A-Za-z0-9]+", candidate):
        return (
            rf"(?<![\w-]){re.escape(candidate)}(?![\w-])",
            re.IGNORECASE,
        )
    return re.escape(candidate), re.IGNORECASE


def _answer_literals(record: RawTrajectoryRecord) -> set[str]:
    values: list[tuple[str, bool]] = []
    verifier_values: list[str] = []
    if record.final_response:
        values.append((str(record.final_response), False))
    if record.verifier_feedback:
        verifier_values.append(str(record.verifier_feedback))
    for key in (
        "answer",
        "expected_answer",
        "gold_answer",
        "ground_truth",
        "reference_answer",
        "source_answer",
    ):
        value = record.extra.get(key)
        if isinstance(value, (str, int, float)):
            values.append((str(value), True))
    for key in ("predicted_answer", "prediction"):
        value = record.extra.get(key)
        if isinstance(value, (str, int, float)):
            values.append((str(value), False))
    trace2skill_result = record.extra.get("trace2skill_result")
    if isinstance(trace2skill_result, Mapping):
        test_cases = trace2skill_result.get("test_cases")
        if isinstance(test_cases, Sequence) and not isinstance(
            test_cases, (str, bytes)
        ):
            for test_case in test_cases:
                if not isinstance(test_case, Mapping):
                    continue
                for key in ("message", "raw_message"):
                    value = test_case.get(key)
                    if isinstance(value, (str, int, float)):
                        verifier_values.append(str(value))
    values.extend((value, False) for value in verifier_values)

    literals: set[str] = set()
    always_protected: set[str] = set()

    def add_literal(value: str, *, protect_generic: bool) -> None:
        if not value or len(value) > 200:
            return
        literals.add(value)
        if protect_generic:
            always_protected.add(value)

    answer_pattern = re.compile(
        r"(?:the\s+)?(?:final\s+)?"
        r"(?:answer|prediction|expected|gold|ground[\s_-]*truth)"
        r"\s*(?:is|=|:)\s*(.+)$",
        flags=re.IGNORECASE,
    )
    for value, protect_generic in values:
        stripped = value.strip()
        if not stripped:
            continue
        for match in re.finditer(
            r"<\s*(?:final[_\s-]*answer|answer)\s*>\s*(.*?)\s*"
            r"<\s*/\s*(?:final[_\s-]*answer|answer)\s*>",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            answer = " ".join(match.group(1).split())
            if answer and len(answer) <= 200:
                add_literal(answer, protect_generic=protect_generic)
        for match in re.finditer(
            r"""["'](?:final[_\s-]*answer|answer|prediction|"""
            r"""expected_answer|gold_answer|ground_truth)["']\s*:\s*"""
            r"""["']([^"']+)["']""",
            stripped,
            flags=re.IGNORECASE,
        ):
            answer = " ".join(match.group(1).split())
            if answer and len(answer) <= 200:
                add_literal(answer, protect_generic=protect_generic)
        if len(stripped) <= 200:
            add_literal(stripped, protect_generic=protect_generic)
        without_tags = re.sub(r"<[^>]+>", " ", stripped)
        without_tags = " ".join(without_tags.split())
        if without_tags and len(without_tags) <= 200:
            add_literal(without_tags, protect_generic=protect_generic)
        for line in stripped.splitlines():
            line = line.strip()
            if line and len(line) <= 200:
                add_literal(line, protect_generic=protect_generic)
            match = answer_pattern.search(line)
            if match:
                answer = match.group(1).strip().strip("'\"")
                if answer and len(answer) <= 200:
                    add_literal(answer, protect_generic=protect_generic)
        normalized = " ".join(re.sub(r"<[^>]+>", " ", stripped).split())
        terminal_parts = [
            part.strip(" \t\r\n'\"")
            for part in re.split(r"(?<=[.!?])\s+|[;\n]+", normalized)
            if part.strip()
        ]
        if terminal_parts:
            terminal = terminal_parts[-1]
            if len(terminal) <= 200:
                add_literal(terminal, protect_generic=protect_generic)
    verifier_value_pattern = re.compile(
        r"\b(expected|got|actual|predicted)\b\s*"
        r"(?:(?:value\s*)?(?:is|=|:)\s*)?"
        r"(?:'([^']*)'|\"([^\"]*)\"|"
        r"([^,;\n]+?)(?=\s+(?:expected|got|actual|predicted)\b|[,;\n]|$))",
        flags=re.IGNORECASE,
    )
    for value in verifier_values:
        for match in verifier_value_pattern.finditer(value):
            label = str(match.group(1)).casefold()
            literal = next(
                (
                    group.strip()
                    for group in match.groups()[1:]
                    if group is not None and group.strip()
                ),
                "",
            )
            if literal and len(literal) <= 200:
                add_literal(
                    literal,
                    protect_generic=(label == "expected"),
                )
    generic_failure_literals = {
        "none",
        "null",
        "n/a",
        "#n/a",
        "#value!",
        "#ref!",
        "#name?",
        "#div/0!",
        "true",
        "false",
    }
    return {
        literal
        for literal in literals
        if re.search(r"\w", literal, flags=re.UNICODE)
        and (
            literal.casefold() not in generic_failure_literals
            or literal in always_protected
        )
    }


def _spreadsheet_reference_audit(
    record: RawTrajectoryRecord,
) -> dict[str, list[str]]:
    pattern = re.compile(
        r"(?<![\w])(?:"
        r"\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?"
        r"|\$?[A-Z]{1,3}:\$?[A-Z]{1,3}"
        r")(?![\w])",
        flags=re.IGNORECASE,
    )

    def references(text: str) -> set[str]:
        return {
            match.group(0).replace("$", "").upper()
            for match in pattern.finditer(text)
        }

    instruction_refs = references(str(record.instruction or ""))
    action_refs = references(
        "\n".join(str(step.action or "") for step in record.steps)
    )
    return {
        "instruction_references": sorted(instruction_refs),
        "action_references": sorted(action_refs),
        "instruction_references_not_explicitly_used_by_actions": sorted(
            instruction_refs - action_refs
        ),
        "audit_limitations": [
            "lexical_range_coverage_only",
            "absence_is_not_proof_of_omitted_access",
            "loops_numeric_bounds_variables_and_helpers_may_access_ranges",
        ],
    }


def _canonicalize_reusable_syntax(text: str) -> str:
    return re.sub(
        r"\b[A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?\b",
        "<cell-reference>",
        text,
    )


def _unsafe_reusable_text_reasons(text: str) -> list[str]:
    checks = (
        ("URL", r"https?://|www\."),
        ("file path", r"(?:^|[\s\"'])[/~][^\s\"']+|[A-Za-z]:\\"),
        (
            "artifact filename",
            r"\b[^\s/\\]+\.(?:xlsx|xls|csv|json|txt|pdf|docx?)\b",
        ),
        ("spreadsheet coordinate", r"\b[A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?\b"),
    )
    return [
        label
        for label, pattern in checks
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    ]


__all__ = [
    "ATOM_SCHEMA",
    "CertifiedOtdConfig",
    "ExperienceAtomAnalyst",
    "LocalParentSkillAnalyst",
    "PARENT_SKILL_SCHEMA",
    "build_certified_otd_dynamic_tree_from_records",
    "build_certified_otd_tree_from_records",
]

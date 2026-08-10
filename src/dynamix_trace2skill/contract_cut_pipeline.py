from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamix_core.contract_cut_ebst import (
    ContractAtom,
    ContractCutResult,
    ContractCutTreeState,
    ExactContractCutOptimizer,
)

from .clients import EmbeddingClient, GenerationClient
from .schemas import RawTrajectoryRecord

CONTRACT_ATOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "trigger",
        "scope",
        "decision",
        "invariant",
        "verification",
        "failure_mode",
    ],
    "properties": {
        name: {"type": "string", "minLength": 1}
        for name in (
            "trigger",
            "scope",
            "decision",
            "invariant",
            "verification",
            "failure_mode",
        )
    },
}

SKILL_COMPILER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reason", "contract"],
    "properties": {
        "status": {"type": "string", "enum": ["contract", "split_required"]},
        "reason": {"type": "string"},
        "contract": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "applicability",
                        "objective",
                        "conditional_rules",
                        "invariants",
                        "verification",
                        "recovery",
                        "source_atom_ids",
                    ],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "applicability": {"type": "string", "minLength": 1},
                        "objective": {"type": "string", "minLength": 1},
                        "conditional_rules": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "condition",
                                    "procedure",
                                    "invariant",
                                    "verification",
                                    "recovery",
                                    "source_atom_ids",
                                ],
                                "properties": {
                                    "condition": {"type": "string", "minLength": 1},
                                    "procedure": {"type": "string", "minLength": 1},
                                    "invariant": {"type": "string", "minLength": 1},
                                    "verification": {"type": "string", "minLength": 1},
                                    "recovery": {"type": "string", "minLength": 1},
                                    "source_atom_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "invariants": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "verification": {"type": "string", "minLength": 1},
                        "recovery": {"type": "string", "minLength": 1},
                        "source_atom_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                },
            ]
        },
    },
}

ATOM_PROMPT_VERSION = "contract_cut_atom_v1"
COMPILER_PROMPT_VERSION = "contract_cut_compiler_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _message_token_count(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> int:
    return sum(
        tokenizer.count(f"{message.get('role', '')}\n{message.get('content', '')}")
        for message in messages
    )


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name, "") or "").strip()
    if not value:
        raise ValueError(f"missing required text field: {field_name}")
    return value


def _required_string_list(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    raw_values = payload.get(field_name)
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings")
    values: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        values.append(raw_value.strip())
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(sorted(set(values)))


def _atom_id(record: RawTrajectoryRecord) -> str:
    identity = f"{record.trajectory_id}\n{record.task_id}\n{record.trial_index}"
    return "cc_atom_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _record_hash(record: RawTrajectoryRecord) -> str:
    return _sha256_json(record.to_dict())


def _safe_evaluator_payload(record: RawTrajectoryRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": bool(record.success),
        "verifier_score": record.verifier_score,
        "verifier_feedback": record.verifier_feedback,
    }
    for key in (
        "answer",
        "expected_answer",
        "gold_answer",
        "ground_truth",
        "reference_answer",
        "source_answer",
        "predicted_answer",
        "prediction",
    ):
        if key in record.extra:
            payload[key] = record.extra[key]
    result = record.extra.get("trace2skill_result")
    if isinstance(result, Mapping):
        payload["trace2skill_result"] = {
            key: result.get(key)
            for key in (
                "success",
                "passed_count",
                "total_count",
                "soft_score",
                "hard_score",
            )
            if key in result
        }
        test_cases = result.get("test_cases")
        if isinstance(test_cases, Sequence) and not isinstance(
            test_cases, (str, bytes)
        ):
            payload["trace2skill_result"]["test_cases"] = [
                {
                    key: test_case.get(key)
                    for key in (
                        "evaluation_mode",
                        "passed",
                        "message",
                        "recalc_error",
                    )
                    if key in test_case
                }
                for test_case in test_cases
                if isinstance(test_case, Mapping)
            ]
    return payload


def render_full_contract_evidence(record: RawTrajectoryRecord) -> str:
    """Render full semantic evidence without paths or fixed-length truncation."""

    payload = {
        "instruction": record.instruction,
        "instruction_type": record.instruction_type,
        "complete_rollout": [step.to_dict() for step in record.steps],
        "produced_answer": record.final_response,
        "evaluator_result": _safe_evaluator_payload(record),
        "success": bool(record.success),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _atom_system_prompt() -> str:
    return (
        "You analyze one complete agent trajectory and produce exactly one "
        "reusable Contract Atom. Use the same analysis procedure whether the "
        "trajectory succeeded or failed. Contrast the requested operation, the "
        "recorded actions and observations, the produced answer, and the "
        "authoritative evaluator result. For a success, preserve the transferable "
        "decision and verification principle. For a failure, identify only a "
        "root cause supported by recorded evidence and state a corrected reusable "
        "policy. If the specific cause is not supported, state a cautious "
        "diagnostic policy rather than guessing. Return one coherent Atom, not a "
        "list. Reusable fields must generalize beyond this task and must not copy "
        "task IDs, file paths, workbook coordinates, URLs, filenames, exact "
        "answers, expected values, incidental example numbers, or task-specific "
        "sheet, table, column, and header names. Replace such names with semantic "
        "roles such as source sheet, target region, key column, or summary row. "
        "Evaluator and "
        "gold evidence may guide diagnosis but must never be mentioned in the Atom. "
        "Keep each field concise and operational."
    )


def atom_messages(record: RawTrajectoryRecord) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _atom_system_prompt()},
        {
            "role": "user",
            "content": (
                "Return exactly one JSON object matching this schema:\n"
                f"{json.dumps(CONTRACT_ATOM_SCHEMA, ensure_ascii=False)}\n\n"
                "Complete trajectory evidence:\n"
                f"{render_full_contract_evidence(record)}"
            ),
        },
    ]


def _protected_literals(record: RawTrajectoryRecord) -> set[str]:
    literals = {
        record.trajectory_id,
        record.task_id,
        record.answer_position,
        record.spreadsheet_path,
        record.output_path,
    }
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
            literals.add(str(value))
    # Numbers stated by a task are examples, not reusable operating rules. Keep
    # exact source quantities out of public Atoms while still allowing general
    # numeric invariants that were not copied from the task.
    for match in re.finditer(
        r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d{2,})(?:\.\d+)?%?(?![\w.])",
        record.instruction,
    ):
        literal = match.group(0)
        literals.add(literal)
        literals.add(literal.replace(",", ""))
    return {
        str(value).strip()
        for value in literals
        if str(value or "").strip()
    }


def atom_leakage_reasons(
    record: RawTrajectoryRecord,
    draft: Mapping[str, str],
) -> tuple[str, ...]:
    text = "\n".join(str(value) for value in draft.values())
    checks = (
        ("URL", r"https?://|www\."),
        ("absolute path", r"(?:^|[\s\"'])[/~][^\s\"']+|[A-Za-z]:\\"),
        ("artifact filename", r"\b[^\s/\\]+\.(?:xlsx|xls|csv|json|txt|pdf|docx?)\b"),
        ("spreadsheet coordinate", r"\b[A-Z]{1,3}\d+(?::[A-Z]{1,3}\d+)?\b"),
        (
            "evaluation artifact",
            r"\b(?:golden|ground[\s_-]*truth|benchmark evaluator|verifier)\b",
        ),
    )
    reasons = [
        label
        for label, pattern in checks
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    ]
    for literal in sorted(_protected_literals(record), key=len, reverse=True):
        if len(literal) < 3:
            continue
        if re.search(re.escape(literal), text, flags=re.IGNORECASE):
            reasons.append("copied source literal")
            break
    quoted_literals = {
        match.group(2).strip()
        for match in re.finditer(r"(['\"])(.{2,80}?)\1", record.instruction)
        if match.group(2).strip()
    }
    for literal in sorted(quoted_literals, key=len, reverse=True):
        if re.search(
            rf"(?<!\w){re.escape(literal)}(?!\w)",
            text,
        ):
            reasons.append("copied source literal")
            break
    return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class AtomExclusion:
    trajectory_id: str
    task_id: str
    record_index: int
    reason: str
    prompt_tokens: int
    prompt_budget: int
    record_sha256: str


class ContractAtomAnalyst:
    def __init__(
        self,
        generation: GenerationClient,
        embedding: EmbeddingClient,
        *,
        tokenizer: Any,
        max_prompt_tokens: int,
        draft_cache_path: str | Path,
    ) -> None:
        self.generation = generation
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.draft_cache_path = Path(draft_cache_path)
        self._append_lock = asyncio.Lock()
        self.protocol_sha256 = _sha256_json(
            {
                "version": ATOM_PROMPT_VERSION,
                "system": _atom_system_prompt(),
                "schema": CONTRACT_ATOM_SCHEMA,
                "max_prompt_tokens": self.max_prompt_tokens,
                "thinking": False,
                "guided_json": False,
                "semantic_revision": "include_previous_draft_once_v1",
            }
        )

    async def extract_many(
        self,
        records: Sequence[RawTrajectoryRecord],
        *,
        record_index_offset: int = 0,
    ) -> tuple[list[ContractAtom], list[AtomExclusion]]:
        if record_index_offset < 0:
            raise ValueError("record_index_offset must be non-negative")
        cached = self._load_cache()

        async def extract(local_index: int, record: RawTrajectoryRecord) -> dict[str, Any]:
            index = record_index_offset + local_index
            record_sha = _record_hash(record)
            cache_key = (record.trajectory_id, record_sha, self.protocol_sha256)
            if cache_key in cached:
                cached_row = dict(cached[cache_key])
                if int(cached_row.get("record_index", -1)) != index:
                    raise ValueError(
                        "cached Contract Atom record index does not match dataset order"
                    )
                return cached_row
            messages = atom_messages(record)
            prompt_tokens = _message_token_count(self.tokenizer, messages)
            if prompt_tokens > self.max_prompt_tokens:
                result = {
                    "status": "excluded",
                    "trajectory_id": record.trajectory_id,
                    "task_id": record.task_id,
                    "record_index": index,
                    "record_sha256": record_sha,
                    "protocol_sha256": self.protocol_sha256,
                    "reason": "analyst_prompt_over_budget",
                    "prompt_tokens": prompt_tokens,
                    "prompt_budget": self.max_prompt_tokens,
                }
                await self._append_cache(result)
                return result

            base_messages = list(messages)
            last_reasons: tuple[str, ...] = ()
            for semantic_attempt in range(2):
                current_prompt_tokens = _message_token_count(self.tokenizer, messages)
                if current_prompt_tokens > self.max_prompt_tokens:
                    result = {
                        "status": "excluded",
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "record_index": index,
                        "record_sha256": record_sha,
                        "protocol_sha256": self.protocol_sha256,
                        "reason": "analyst_revision_prompt_over_budget",
                        "prompt_tokens": current_prompt_tokens,
                        "prompt_budget": self.max_prompt_tokens,
                    }
                    await self._append_cache(result)
                    return result
                payload = await self.generation.chat_json(
                    messages,
                    schema_name="ContractAtom",
                    guided_json=None,
                    max_tokens=None,
                    retries=1,
                    debug_metadata={
                        "component": "contract_cut_atom_analyst",
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "semantic_attempt": semantic_attempt + 1,
                    },
                )
                draft = {
                    field_name: _required_text(payload, field_name)
                    for field_name in (
                        "trigger",
                        "scope",
                        "decision",
                        "invariant",
                        "verification",
                        "failure_mode",
                    )
                }
                last_reasons = atom_leakage_reasons(record, draft)
                if not last_reasons:
                    result = {
                        "status": "accepted",
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "record_index": index,
                        "record_sha256": record_sha,
                        "protocol_sha256": self.protocol_sha256,
                        "prompt_tokens": current_prompt_tokens,
                        "draft": draft,
                    }
                    await self._append_cache(result)
                    return result
                messages = [
                    *base_messages,
                    {
                        "role": "assistant",
                        "content": json.dumps(draft, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Return only a revised Contract Atom JSON object, not the "
                            "schema or an explanation. The preceding draft "
                            "violated reusable-content constraints: "
                            f"{', '.join(last_reasons)}. Remove all task-specific "
                            "literals while preserving only the general decision, "
                            "invariant, verification, and recovery principle."
                        ),
                    },
                ]
            raise ValueError(
                "Contract Atom contains non-reusable literals after one revision: "
                + ", ".join(last_reasons)
            )

        results = await asyncio.gather(
            *(extract(index, record) for index, record in enumerate(records)),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            summary = "; ".join(
                f"{type(error).__name__}: {error}" for error in failures[:10]
            )
            raise RuntimeError(
                f"Contract Atom extraction had {len(failures)} runtime failures: {summary}"
            )

        accepted_rows = [
            dict(result)
            for result in results
            if isinstance(result, Mapping) and result.get("status") == "accepted"
        ]
        accepted_rows.sort(key=lambda row: int(row["record_index"]))
        boundary_texts = [
            "\n".join(
                (
                    f"Applicability: {row['draft']['trigger']}",
                    f"Scope: {row['draft']['scope']}",
                    f"Success condition: {row['draft']['verification']}",
                )
            )
            for row in accepted_rows
        ]
        embeddings = await self.embedding.embed_texts(
            boundary_texts,
            cache_namespace="contract_cut_boundary_v1",
        )
        records_by_index = {
            record_index_offset + index: record
            for index, record in enumerate(records)
        }
        atoms: list[ContractAtom] = []
        for row, vector in zip(accepted_rows, embeddings):
            index = int(row["record_index"])
            record = records_by_index[index]
            draft = dict(row["draft"])
            atoms.append(
                ContractAtom(
                    atom_id=_atom_id(record),
                    source_item_id=record.task_id,
                    trigger=draft["trigger"],
                    scope=draft["scope"],
                    decision=draft["decision"],
                    invariant=draft["invariant"],
                    verification=draft["verification"],
                    failure_mode=draft["failure_mode"],
                    boundary_embedding=tuple(vector),
                    provenance={
                        "trajectory_id": record.trajectory_id,
                        "task_id": record.task_id,
                        "trial_index": record.trial_index,
                        "record_index": index,
                        "record_sha256": row["record_sha256"],
                        "success": bool(record.success),
                        "verifier_score": record.verifier_score,
                        "verifier_feedback": record.verifier_feedback,
                        "atom_prompt_tokens": row["prompt_tokens"],
                        "atom_protocol_sha256": self.protocol_sha256,
                    },
                )
            )
        exclusions = [
            AtomExclusion(
                trajectory_id=str(result["trajectory_id"]),
                task_id=str(result["task_id"]),
                record_index=int(result["record_index"]),
                reason=str(result["reason"]),
                prompt_tokens=int(result["prompt_tokens"]),
                prompt_budget=int(result["prompt_budget"]),
                record_sha256=str(result["record_sha256"]),
            )
            for result in results
            if isinstance(result, Mapping) and result.get("status") == "excluded"
        ]
        return atoms, exclusions

    def _load_cache(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        if not self.draft_cache_path.exists():
            return {}
        cached: dict[tuple[str, str, str], dict[str, Any]] = {}
        for line in self.draft_cache_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            key = (
                str(row.get("trajectory_id") or ""),
                str(row.get("record_sha256") or ""),
                str(row.get("protocol_sha256") or ""),
            )
            if all(key):
                cached[key] = dict(row)
        return cached

    async def _append_cache(self, payload: Mapping[str, Any]) -> None:
        async with self._append_lock:
            self.draft_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.draft_cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


@dataclass(frozen=True)
class ConditionalRule:
    condition: str
    procedure: str
    invariant: str
    verification: str
    recovery: str
    source_atom_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_atom_ids"] = list(self.source_atom_ids)
        return payload


@dataclass(frozen=True)
class SkillContract:
    name: str
    applicability: str
    objective: str
    conditional_rules: tuple[ConditionalRule, ...]
    invariants: tuple[str, ...]
    verification: str
    recovery: str
    source_atom_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applicability": self.applicability,
            "objective": self.objective,
            "conditional_rules": [rule.to_dict() for rule in self.conditional_rules],
            "invariants": list(self.invariants),
            "verification": self.verification,
            "recovery": self.recovery,
            "source_atom_ids": list(self.source_atom_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillContract":
        return cls(
            name=str(payload["name"]),
            applicability=str(payload["applicability"]),
            objective=str(payload["objective"]),
            conditional_rules=tuple(
                ConditionalRule(
                    condition=str(rule["condition"]),
                    procedure=str(rule["procedure"]),
                    invariant=str(rule["invariant"]),
                    verification=str(rule["verification"]),
                    recovery=str(rule["recovery"]),
                    source_atom_ids=tuple(str(value) for value in rule["source_atom_ids"]),
                )
                for rule in payload["conditional_rules"]
            ),
            invariants=tuple(str(value) for value in payload["invariants"]),
            verification=str(payload["verification"]),
            recovery=str(payload["recovery"]),
            source_atom_ids=tuple(str(value) for value in payload["source_atom_ids"]),
        )


@dataclass(frozen=True)
class CompilerResult:
    status: str
    input_sha256: str
    contract: SkillContract | None
    reason: str
    prompt_tokens: int
    cache_hit: bool = False


def _compiler_system_prompt() -> str:
    return (
        "You compile a complete reusable agent skill from a set of structured "
        "Contract Atoms that a metric tree placed in one candidate region. Do not "
        "reanalyze raw trajectories. Return status contract only when the Atoms "
        "can be organized as one coherent skill with explicit conditional rules, "
        "shared invariants, verification, and recovery. Use status split_required "
        "when the region contains incompatible objectives or cannot form one "
        "complete executable skill without hiding contradictions. Do not create an "
        "arbitrary DAG. Every rule must cite real source_atom_ids, and the top-level "
        "source_atom_ids must cover every provided Atom exactly once as a set. Do "
        "not introduce task IDs, paths, coordinates, exact answers, evaluator "
        "language, or unsupported facts."
    )


class ContractCompiler:
    def __init__(
        self,
        generation: GenerationClient,
        *,
        tokenizer: Any,
        max_prompt_tokens: int,
        cache_path: str | Path,
    ) -> None:
        self.generation = generation
        self.tokenizer = tokenizer
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()
        self._cache_lock = asyncio.Lock()

    def input_payload(
        self,
        state: ContractCutTreeState,
        node_id: str,
    ) -> dict[str, Any]:
        atom_ids = state.candidate_atom_ids(node_id)
        return {
            "compiler_prompt_version": COMPILER_PROMPT_VERSION,
            "atoms": [
                {
                    "atom_id": atom_id,
                    "trigger": state.atoms[atom_id].trigger,
                    "scope": state.atoms[atom_id].scope,
                    "decision": state.atoms[atom_id].decision,
                    "invariant": state.atoms[atom_id].invariant,
                    "verification": state.atoms[atom_id].verification,
                    "failure_mode": state.atoms[atom_id].failure_mode,
                }
                for atom_id in atom_ids
            ],
        }

    def input_sha256(self, state: ContractCutTreeState, node_id: str) -> str:
        return _sha256_json(self.input_payload(state, node_id))

    def messages(
        self,
        state: ContractCutTreeState,
        node_id: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _compiler_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Return exactly one JSON object matching this schema:\n"
                    f"{json.dumps(SKILL_COMPILER_SCHEMA, ensure_ascii=False)}\n\n"
                    "Candidate Contract Atoms:\n"
                    f"{json.dumps(self.input_payload(state, node_id), ensure_ascii=False, indent=2)}"
                ),
            },
        ]

    async def compile(
        self,
        state: ContractCutTreeState,
        node_id: str,
    ) -> CompilerResult:
        input_sha = self.input_sha256(state, node_id)
        cached = self._cache.get(input_sha)
        if cached is not None:
            result = self._result_from_dict(cached)
            return CompilerResult(
                status=result.status,
                input_sha256=result.input_sha256,
                contract=result.contract,
                reason=result.reason,
                prompt_tokens=result.prompt_tokens,
                cache_hit=True,
            )
        atom_ids = state.candidate_atom_ids(node_id)
        if state.is_atom_entry(node_id):
            contract = self._leaf_contract(state.atoms[atom_ids[0]])
            result = CompilerResult(
                status="contract",
                input_sha256=input_sha,
                contract=contract,
                reason="deterministic_single_atom_leaf",
                prompt_tokens=0,
            )
            await self._save_result(result)
            return result

        messages = self.messages(state, node_id)
        last_error: Exception | None = None
        base_messages = list(messages)
        previous_payload: Mapping[str, Any] | None = None
        for attempt in range(2):
            prompt_tokens = _message_token_count(self.tokenizer, messages)
            if prompt_tokens > self.max_prompt_tokens:
                reason = (
                    "compiler_prompt_over_budget"
                    if attempt == 0
                    else "compiler_repair_prompt_over_budget"
                )
                result = CompilerResult(
                    status="split_required",
                    input_sha256=input_sha,
                    contract=None,
                    reason=reason,
                    prompt_tokens=prompt_tokens,
                )
                await self._save_result(result)
                return result
            try:
                payload = await self.generation.chat_json(
                    messages,
                    schema_name="ContractCutSkillCompiler",
                    guided_json=None,
                    max_tokens=None,
                    retries=0,
                    debug_metadata={
                        "component": "contract_cut_skill_compiler",
                        "node_id": node_id,
                        "input_sha256": input_sha,
                        "format_attempt": attempt + 1,
                    },
                )
                previous_payload = payload
                result = self._validate_result(
                    payload,
                    member_atom_ids=atom_ids,
                    input_sha=input_sha,
                    prompt_tokens=prompt_tokens,
                )
                await self._save_result(result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    prior_output = (
                        [
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    previous_payload,
                                    ensure_ascii=False,
                                ),
                            }
                        ]
                        if previous_payload is not None
                        else []
                    )
                    messages = [
                        *base_messages,
                        *prior_output,
                        {
                            "role": "user",
                            "content": (
                                "Repair the output once. Return a complete fresh JSON "
                                "object matching the schema and source Atom IDs. "
                                f"Previous integrity error: {exc}"
                            ),
                        },
                    ]
        result = CompilerResult(
            status="split_required",
            input_sha256=input_sha,
            contract=None,
            reason=(
                "compiler_invalid_after_repair:"
                f"{type(last_error).__name__ if last_error else 'unknown'}"
            ),
            prompt_tokens=_message_token_count(self.tokenizer, messages),
        )
        await self._save_result(result)
        return result

    def _validate_result(
        self,
        payload: Mapping[str, Any],
        *,
        member_atom_ids: Sequence[str],
        input_sha: str,
        prompt_tokens: int,
    ) -> CompilerResult:
        status = str(payload.get("status") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if status == "split_required":
            if not reason:
                raise ValueError("split_required result must include a reason")
            if payload.get("contract") not in (None, {}):
                raise ValueError("split_required result must not include a contract")
            return CompilerResult(
                status=status,
                input_sha256=input_sha,
                contract=None,
                reason=reason,
                prompt_tokens=prompt_tokens,
            )
        if status != "contract":
            raise ValueError("compiler status must be contract or split_required")
        raw_contract = payload.get("contract")
        if not isinstance(raw_contract, Mapping):
            raise ValueError("contract result is missing the contract object")
        contract = self._contract_from_payload(raw_contract)
        expected = set(member_atom_ids)
        if set(contract.source_atom_ids) != expected:
            raise ValueError("top-level source_atom_ids must equal candidate Atom IDs")
        covered_by_rules: set[str] = set()
        for rule in contract.conditional_rules:
            rule_ids = set(rule.source_atom_ids)
            if not rule_ids or not rule_ids <= expected:
                raise ValueError("conditional rule has invalid source_atom_ids")
            covered_by_rules.update(rule_ids)
        if covered_by_rules != expected:
            raise ValueError("conditional rules do not map every candidate Atom")
        return CompilerResult(
            status=status,
            input_sha256=input_sha,
            contract=contract,
            reason=reason or "compiled_contract",
            prompt_tokens=prompt_tokens,
        )

    def _contract_from_payload(self, payload: Mapping[str, Any]) -> SkillContract:
        rules_value = payload.get("conditional_rules")
        if not isinstance(rules_value, Sequence) or isinstance(
            rules_value, (str, bytes)
        ) or not rules_value:
            raise ValueError("conditional_rules must be a non-empty list")
        rules: list[ConditionalRule] = []
        for raw_rule in rules_value:
            if not isinstance(raw_rule, Mapping):
                raise ValueError("conditional rule must be an object")
            ids = _required_string_list(raw_rule, "source_atom_ids")
            rules.append(
                ConditionalRule(
                    condition=_required_text(raw_rule, "condition"),
                    procedure=_required_text(raw_rule, "procedure"),
                    invariant=_required_text(raw_rule, "invariant"),
                    verification=_required_text(raw_rule, "verification"),
                    recovery=_required_text(raw_rule, "recovery"),
                    source_atom_ids=ids,
                )
            )
        invariants_value = payload.get("invariants")
        if not isinstance(invariants_value, Sequence) or isinstance(
            invariants_value, (str, bytes)
        ):
            raise ValueError("invariants must be a list")
        invariants = tuple(
            value
            for value in (str(item).strip() for item in invariants_value)
            if value
        )
        if not invariants:
            raise ValueError("invariants must not be empty")
        source_atom_ids = _required_string_list(payload, "source_atom_ids")
        return SkillContract(
            name=_required_text(payload, "name"),
            applicability=_required_text(payload, "applicability"),
            objective=_required_text(payload, "objective"),
            conditional_rules=tuple(rules),
            invariants=invariants,
            verification=_required_text(payload, "verification"),
            recovery=_required_text(payload, "recovery"),
            source_atom_ids=source_atom_ids,
        )

    def _leaf_contract(self, atom: ContractAtom) -> SkillContract:
        trigger_words = " ".join(atom.trigger.split()).split()
        name = "Evidence-guided " + " ".join(trigger_words[:12]).rstrip(",.;:")
        recovery = (
            "Diagnose and correct this violated condition before retrying: "
            + atom.failure_mode
        )
        return SkillContract(
            name=name,
            applicability=atom.trigger + " " + atom.scope,
            objective=atom.decision,
            conditional_rules=(
                ConditionalRule(
                    condition=atom.trigger,
                    procedure=atom.decision,
                    invariant=atom.invariant,
                    verification=atom.verification,
                    recovery=recovery,
                    source_atom_ids=(atom.atom_id,),
                ),
            ),
            invariants=(atom.invariant,),
            verification=atom.verification,
            recovery=recovery,
            source_atom_ids=(atom.atom_id,),
        )

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("format") != "contract_cut_compiler_cache_v1":
            raise ValueError("unsupported Contract-Cut compiler cache")
        return {
            str(key): dict(value)
            for key, value in dict(payload.get("results", {})).items()
        }

    async def _save_result(self, result: CompilerResult) -> None:
        async with self._cache_lock:
            self._cache[result.input_sha256] = self._result_to_dict(result)
            _write_json_atomic(
                self.cache_path,
                {
                    "format": "contract_cut_compiler_cache_v1",
                    "results": self._cache,
                },
            )

    @staticmethod
    def _result_to_dict(result: CompilerResult) -> dict[str, Any]:
        return {
            "status": result.status,
            "input_sha256": result.input_sha256,
            "contract": result.contract.to_dict() if result.contract else None,
            "reason": result.reason,
            "prompt_tokens": result.prompt_tokens,
        }

    @staticmethod
    def _result_from_dict(payload: Mapping[str, Any]) -> CompilerResult:
        raw_contract = payload.get("contract")
        return CompilerResult(
            status=str(payload["status"]),
            input_sha256=str(payload["input_sha256"]),
            contract=(
                SkillContract.from_dict(raw_contract)
                if isinstance(raw_contract, Mapping)
                else None
            ),
            reason=str(payload.get("reason") or ""),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
        )


@dataclass(frozen=True)
class CompiledRegion:
    tree_node_id: str
    member_atom_ids: tuple[str, ...]
    source_item_ids: tuple[str, ...]
    input_sha256: str
    contract: SkillContract
    compiler_cache_hit: bool


@dataclass
class SkillVersion:
    skill_id: str
    version: int
    status: str
    tree_node_id: str
    member_atom_ids: tuple[str, ...]
    source_item_ids: tuple[str, ...]
    compiler_input_sha256: str
    contract: SkillContract
    created_at_batch: int
    derived_from: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()
    skill_md_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status,
            "tree_node_id": self.tree_node_id,
            "member_atom_ids": list(self.member_atom_ids),
            "source_item_ids": list(self.source_item_ids),
            "compiler_input_sha256": self.compiler_input_sha256,
            "contract": self.contract.to_dict(),
            "created_at_batch": self.created_at_batch,
            "derived_from": list(self.derived_from),
            "superseded_by": list(self.superseded_by),
            "skill_md_sha256": self.skill_md_sha256,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Return lineage metadata that is safe for agent-readable skill files."""
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status,
            "member_atom_ids": list(self.member_atom_ids),
            "created_at_batch": self.created_at_batch,
            "derived_from": list(self.derived_from),
            "superseded_by": list(self.superseded_by),
            "skill_md_sha256": self.skill_md_sha256,
        }


class StableSkillRegistry:
    def __init__(self) -> None:
        self.next_skill_sequence = 0
        self.active: dict[str, SkillVersion] = {}
        self.history: list[SkillVersion] = []

    def update(
        self,
        regions: Sequence[CompiledRegion],
        *,
        batch_index: int,
    ) -> tuple[SkillVersion, ...]:
        old_active = dict(self.active)
        old_members = {
            skill_id: set(version.member_atom_ids)
            for skill_id, version in old_active.items()
        }
        region_members = [set(region.member_atom_ids) for region in regions]
        overlaps = [
            tuple(
                sorted(
                    skill_id
                    for skill_id, members in old_members.items()
                    if members & region_members[index]
                )
            )
            for index in range(len(regions))
        ]
        old_overlap_counts = {
            skill_id: sum(skill_id in region_overlaps for region_overlaps in overlaps)
            for skill_id in old_active
        }
        assignments: dict[int, str] = {}
        used_old: set[str] = set()

        for index, members in enumerate(region_members):
            exact = [
                skill_id
                for skill_id, old in old_members.items()
                if old == members
            ]
            if exact:
                skill_id = min(exact)
                assignments[index] = skill_id
                used_old.add(skill_id)
        for index, members in enumerate(region_members):
            if index in assignments:
                continue
            candidates = [
                skill_id
                for skill_id, old in old_members.items()
                if skill_id not in used_old
                and old < members
                and old_overlap_counts[skill_id] == 1
                and len(overlaps[index]) == 1
            ]
            if candidates:
                skill_id = min(
                    candidates,
                    key=lambda value: (-len(old_members[value]), value),
                )
                assignments[index] = skill_id
                used_old.add(skill_id)

        for index in range(len(regions)):
            if index not in assignments:
                assignments[index] = self._new_skill_id()

        new_active: dict[str, SkillVersion] = {}
        for index, region in enumerate(regions):
            skill_id = assignments[index]
            prior = old_active.get(skill_id)
            predecessors = overlaps[index]
            unchanged = (
                prior is not None
                and prior.member_atom_ids == region.member_atom_ids
                and prior.compiler_input_sha256 == region.input_sha256
                and prior.contract == region.contract
            )
            if unchanged:
                # Physical EBST node IDs may change after a deterministic split.
                # That relocation is geometry metadata, not a new Skill version.
                prior.tree_node_id = region.tree_node_id
                new_active[skill_id] = prior
                continue
            if prior is not None:
                prior.status = "archived"
            version = SkillVersion(
                skill_id=skill_id,
                version=(prior.version + 1 if prior is not None else 1),
                status="active",
                tree_node_id=region.tree_node_id,
                member_atom_ids=region.member_atom_ids,
                source_item_ids=region.source_item_ids,
                compiler_input_sha256=region.input_sha256,
                contract=region.contract,
                created_at_batch=batch_index,
                derived_from=(
                    tuple(value for value in predecessors if value != skill_id)
                    if prior is not None
                    else predecessors
                ),
                skill_md_sha256=hashlib.sha256(
                    render_skill_markdown(region.contract).encode("utf-8")
                ).hexdigest(),
            )
            self.history.append(version)
            new_active[skill_id] = version

        for skill_id, prior in old_active.items():
            if skill_id in new_active:
                continue
            prior.status = "superseded"
            prior.superseded_by = tuple(
                sorted(
                    assignments[index]
                    for index, region_overlap in enumerate(overlaps)
                    if skill_id in region_overlap
                )
            )
        self.active = new_active
        return tuple(self.active[key] for key in sorted(self.active))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "contract_cut_skill_registry_v1",
            "next_skill_sequence": self.next_skill_sequence,
            "active_skill_ids": sorted(self.active),
            "versions": [version.to_dict() for version in self.history],
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "format": "contract_cut_public_skill_registry_v1",
            "active_skill_ids": sorted(self.active),
            "versions": [version.to_public_dict() for version in self.history],
        }

    def _new_skill_id(self) -> str:
        skill_id = f"cc_skill_{self.next_skill_sequence:04d}"
        self.next_skill_sequence += 1
        return skill_id


@dataclass(frozen=True)
class ContractCutBuildConfig:
    batch_size: int = 8
    max_entries: int = 8
    beta: float = 1.0
    analyst_max_prompt_tokens: int = 92000
    compiler_max_prompt_tokens: int = 92000

    def __post_init__(self) -> None:
        if self.batch_size != 8:
            raise ValueError("Contract-Cut EBST v1 fixes batch_size=8")
        if self.max_entries != 8:
            raise ValueError("Contract-Cut EBST v1 fixes max_entries=8")
        if self.analyst_max_prompt_tokens <= 0 or self.compiler_max_prompt_tokens <= 0:
            raise ValueError("prompt budgets must be positive")


@dataclass(frozen=True)
class ContractCutBuildResult:
    tree: ContractCutTreeState
    atoms: tuple[ContractAtom, ...]
    exclusions: tuple[AtomExclusion, ...]
    final_cut: ContractCutResult
    active_skills: tuple[SkillVersion, ...]
    run_dir: str
    skills_dir: str
    batch_events: tuple[dict[str, Any], ...]


async def _lazy_contract_cut(
    state: ContractCutTreeState,
    compiler: ContractCompiler,
    *,
    beta: float,
) -> tuple[ContractCutResult, tuple[CompiledRegion, ...], list[dict[str, Any]]]:
    blocked: set[str] = set()
    attempts: list[dict[str, Any]] = []
    optimizer = ExactContractCutOptimizer(state, beta=beta)
    for iteration in range(len(state.nodes) + 1):
        cut = optimizer.solve(infeasible_node_ids=tuple(sorted(blocked)))
        results = await asyncio.gather(
            *(compiler.compile(state, node_id) for node_id in cut.selected_node_ids)
        )
        newly_blocked = {
            node_id
            for node_id, result in zip(cut.selected_node_ids, results)
            if result.status == "split_required"
        }
        attempts.append(
            {
                "iteration": iteration + 1,
                "selected_node_ids": list(cut.selected_node_ids),
                "objective": cut.objective,
                "compiler_results": [
                    {
                        "node_id": node_id,
                        "status": result.status,
                        "reason": result.reason,
                        "input_sha256": result.input_sha256,
                        "prompt_tokens": result.prompt_tokens,
                        "cache_hit": result.cache_hit,
                    }
                    for node_id, result in zip(cut.selected_node_ids, results)
                ],
            }
        )
        if not newly_blocked:
            regions = []
            for node_id, result in zip(cut.selected_node_ids, results):
                if result.contract is None:
                    raise RuntimeError("feasible cut node has no compiled contract")
                atom_ids = state.candidate_atom_ids(node_id)
                regions.append(
                    CompiledRegion(
                        tree_node_id=node_id,
                        member_atom_ids=atom_ids,
                        source_item_ids=tuple(
                            sorted(state.atoms[atom_id].source_item_id for atom_id in atom_ids)
                        ),
                        input_sha256=result.input_sha256,
                        contract=result.contract,
                        compiler_cache_hit=result.cache_hit,
                    )
                )
            return cut, tuple(regions), attempts
        if newly_blocked <= blocked:
            raise RuntimeError("lazy Contract feasibility made no progress")
        for node_id in newly_blocked:
            if state.is_atom_entry(node_id):
                raise RuntimeError("an Atom terminal cannot be split_required")
        blocked.update(newly_blocked)
    raise RuntimeError("lazy Contract feasibility exceeded the number of tree nodes")


async def build_contract_cut_skills(
    records: Sequence[RawTrajectoryRecord],
    *,
    generation: GenerationClient,
    embedding: EmbeddingClient,
    tokenizer: Any,
    run_dir: str | Path,
    config: ContractCutBuildConfig,
) -> ContractCutBuildResult:
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_root / "contract_cut"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    analyst = ContractAtomAnalyst(
        generation,
        embedding,
        tokenizer=tokenizer,
        max_prompt_tokens=config.analyst_max_prompt_tokens,
        draft_cache_path=artifact_dir / "atom_drafts.jsonl",
    )
    atoms: list[ContractAtom] = []
    exclusions: list[AtomExclusion] = []
    tree = ContractCutTreeState(max_entries=config.max_entries)
    compiler = ContractCompiler(
        generation,
        tokenizer=tokenizer,
        max_prompt_tokens=config.compiler_max_prompt_tokens,
        cache_path=artifact_dir / "compiler_cache.json",
    )
    registry = StableSkillRegistry()
    batch_events: list[dict[str, Any]] = []
    final_cut = ContractCutResult((), 0.0, 0.0, 0.0, ())
    active_skills: tuple[SkillVersion, ...] = ()
    for batch_start in range(0, len(records), config.batch_size):
        batch_stop = min(len(records), batch_start + config.batch_size)
        batch_atoms, batch_exclusions = await analyst.extract_many(
            records[batch_start:batch_stop],
            record_index_offset=batch_start,
        )
        atoms.extend(batch_atoms)
        exclusions.extend(batch_exclusions)
        _write_json_atomic(
            artifact_dir / "contract_atoms.json",
            {
                "format": "contract_cut_atoms_v1",
                "input_record_count": len(records),
                "processed_record_count": batch_stop,
                "accepted_atom_count": len(atoms),
                "excluded_count": len(exclusions),
                "analyst_protocol_sha256": analyst.protocol_sha256,
                "atoms": [atom.to_dict() for atom in atoms],
                "exclusions": [asdict(exclusion) for exclusion in exclusions],
            },
        )
        insertions = []
        for atom in sorted(
            batch_atoms,
            key=lambda value: int(value.provenance["record_index"]),
        ):
            record_index = int(atom.provenance["record_index"])
            result = tree.insert(atom)
            insertions.append(
                {
                    "record_index": record_index,
                    "atom_id": atom.atom_id,
                    "path_before_split": list(result.path_before_split),
                    "affected_node_ids": list(result.affected_node_ids),
                    "created_node_ids": list(result.created_node_ids),
                    "retired_node_ids": list(result.retired_node_ids),
                    "split_count": result.split_count,
                }
            )
        if tree.root_id is None:
            batch_index = batch_start // config.batch_size + 1
            batch_event = {
                "batch_index": batch_index,
                "record_range": [batch_start, batch_stop],
                "accepted_atom_ids": [atom.atom_id for atom in batch_atoms],
                "excluded_trajectory_ids": [
                    exclusion.trajectory_id for exclusion in batch_exclusions
                ],
                "insertions": insertions,
                "selected_skill_count": 0,
            }
            batch_events.append(batch_event)
            _write_json_atomic(
                artifact_dir / "batches" / f"batch_{batch_index:04d}.json",
                batch_event,
            )
            continue
        final_cut, regions, feasibility_attempts = await _lazy_contract_cut(
            tree,
            compiler,
            beta=config.beta,
        )
        batch_index = batch_start // config.batch_size + 1
        active_skills = registry.update(regions, batch_index=batch_index)
        batch_event = {
            "batch_index": batch_index,
            "record_range": [batch_start, batch_stop],
            "accepted_atom_ids": [atom.atom_id for atom in batch_atoms],
            "excluded_trajectory_ids": [
                exclusion.trajectory_id for exclusion in batch_exclusions
            ],
            "insertions": insertions,
            "tree_node_count": len(tree.nodes),
            "tree_height": max(tree.leaf_depths().values(), default=0),
            "cut_objective": final_cut.objective,
            "cut_distortion": final_cut.distortion,
            "selected_node_ids": list(final_cut.selected_node_ids),
            "active_skill_ids": [skill.skill_id for skill in active_skills],
            "feasibility_attempts": feasibility_attempts,
        }
        batch_events.append(batch_event)
        _write_json_atomic(
            artifact_dir / "batches" / f"batch_{batch_index:04d}.json",
            batch_event,
        )

    tree.validate()
    _write_json_atomic(artifact_dir / "tree_state.json", tree.to_dict())
    _write_json_atomic(artifact_dir / "tree_audit.json", tree.structural_audit())
    _write_json_atomic(
        artifact_dir / "build_manifest.json",
        {
            "format": "contract_cut_build_manifest_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": asdict(config),
            "input_record_count": len(records),
            "accepted_atom_count": len(atoms),
            "excluded_count": len(exclusions),
            "batch_count": len(batch_events),
            "final_cut": asdict(final_cut),
            "active_skill_ids": [skill.skill_id for skill in active_skills],
        },
    )
    skills_dir = export_skill_folders(
        tree,
        active_skills,
        run_root / "skills",
        tokenizer=tokenizer,
        registry=registry,
    )
    _write_json_atomic(artifact_dir / "skill_registry.json", registry.to_dict())
    return ContractCutBuildResult(
        tree=tree,
        atoms=tuple(atoms),
        exclusions=tuple(exclusions),
        final_cut=final_cut,
        active_skills=active_skills,
        run_dir=str(run_root),
        skills_dir=str(skills_dir),
        batch_events=tuple(batch_events),
    )


def render_skill_markdown(contract: SkillContract) -> str:
    decision_rules = []
    procedures = []
    for index, rule in enumerate(contract.conditional_rules, start=1):
        decision_rules.extend(
            (
                f"### Rule {index}",
                f"- **Condition:** {rule.condition}",
                f"- **Decision:** {rule.procedure}",
                f"- **Rule invariant:** {rule.invariant}",
                f"- **Rule verification:** {rule.verification}",
                f"- **Rule recovery:** {rule.recovery}",
                "",
            )
        )
        procedures.append(f"{index}. {rule.procedure}")
    invariants = [f"- {value}" for value in contract.invariants]
    return "\n".join(
        (
            f"# {contract.name}",
            "",
            "## When To Use",
            contract.applicability,
            "",
            "## Objective",
            contract.objective,
            "",
            "## Decision Rules",
            *decision_rules,
            "## Procedure",
            *procedures,
            "",
            "## Invariants",
            *invariants,
            "",
            "## Verification",
            contract.verification,
            "",
            "## Recovery",
            contract.recovery,
            "",
            "## References",
            "- [Structured source atoms](references/atoms.json)",
            "- [Skill provenance](provenance.json)",
            "",
        )
    )


def _public_atom_reference(atom: ContractAtom) -> dict[str, Any]:
    return {
        "atom_id": atom.atom_id,
        "trigger": atom.trigger,
        "scope": atom.scope,
        "decision": atom.decision,
        "invariant": atom.invariant,
        "verification": atom.verification,
        "failure_mode": atom.failure_mode,
    }


def export_skill_folders(
    tree: ContractCutTreeState,
    skills: Sequence[SkillVersion],
    output_dir: str | Path,
    *,
    tokenizer: Any,
    registry: StableSkillRegistry,
) -> Path:
    requested_root = Path(output_dir)
    if requested_root.is_symlink():
        raise ValueError("refusing to export skills through a symlink")
    root = requested_root.resolve()
    if root.name != "skills":
        raise ValueError("Contract-Cut skill export directory must be named 'skills'")
    owner_marker = ".contract_cut_owned.json"
    if root.exists():
        marker_path = root / owner_marker
        if not marker_path.is_file():
            raise ValueError(
                f"refusing to replace an unowned skill directory: {root}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("format") != "contract_cut_skill_export_owner_v1":
            raise ValueError(f"invalid Contract-Cut ownership marker: {marker_path}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.contract-cut-",
            dir=root.parent,
        )
    )
    _write_json_atomic(
        temporary_root / owner_marker,
        {"format": "contract_cut_skill_export_owner_v1"},
    )
    nodes = []
    for skill in sorted(skills, key=lambda value: value.skill_id):
        final_skill_dir = root / f"skill_{skill.skill_id}"
        skill_dir = temporary_root / f"skill_{skill.skill_id}"
        references_dir = skill_dir / "references"
        final_references_dir = final_skill_dir / "references"
        references_dir.mkdir(parents=True, exist_ok=True)
        markdown = render_skill_markdown(skill.contract)
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(markdown, encoding="utf-8")
        rendered_sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        if skill.skill_md_sha256 != rendered_sha256:
            raise RuntimeError("registry SKILL.md hash does not match rendered content")
        atoms_payload = {
            "format": "contract_cut_skill_atoms_v1",
            "skill_id": skill.skill_id,
            "atoms": [
                _public_atom_reference(tree.atoms[atom_id])
                for atom_id in skill.member_atom_ids
            ],
        }
        _write_json_atomic(references_dir / "atoms.json", atoms_payload)
        provenance_payload = skill.to_public_dict()
        provenance_payload["format"] = "contract_cut_skill_provenance_v1"
        provenance_payload["references"] = ["references/atoms.json"]
        _write_json_atomic(skill_dir / "provenance.json", provenance_payload)
        reference_notice = (
            "\nOptional reference directory for this retrieved skill: "
            f"{final_references_dir}. Only inspect a file explicitly cited in SKILL.md "
            "when the current task needs that detail."
        )
        prompt_text = markdown.rstrip() + "\n" + reference_notice + "\n"
        embedding_text = "\n".join(
            (
                f"name: {skill.contract.name}",
                f"applicability: {skill.contract.applicability}",
                f"objective: {skill.contract.objective}",
            )
        )
        nodes.append(
            {
                "node_id": skill.skill_id,
                "item_id": skill.skill_id,
                "name": skill.contract.name,
                "trigger": skill.contract.applicability,
                "content": skill.contract.objective,
                "embedding_text": embedding_text,
                "prompt_text": prompt_text,
                "sha256": hashlib.sha256(embedding_text.encode("utf-8")).hexdigest(),
                "level": max(tree.leaf_depths().values(), default=0)
                - _node_depth(tree, skill.tree_node_id),
                "support_mass": float(len(skill.member_atom_ids)),
                "confidence": 1.0,
                "source_community_id": skill.tree_node_id,
                "source_member_count": len(skill.member_atom_ids),
                "analyst_mode": "contract_cut_skill",
                "parent_node_id": None,
                "child_node_ids": [],
                "token_cost": max(1, tokenizer.count(prompt_text)),
                "skill_directory": str(final_skill_dir),
                "skill_version": skill.version,
            }
        )
    manifest = {
        "format": "dynamix_node_skill_bank_v1",
        "tree_policy": "contract_cut_ebst",
        "export_policy": {
            "heldout_retrieval": "dense_top_k",
            "default_top_k": 1,
            "retrieval_corpus": "contract_feasible_optimal_cut",
            "prompt_injection": "full_skill_md",
            "retrieval_embedding_fields": ["name", "applicability", "objective"],
        },
        "node_count": len(nodes),
        "nodes": nodes,
    }
    _write_json_atomic(temporary_root / "node_bank_manifest.json", manifest)
    _write_json_atomic(
        temporary_root / "skill_registry.json",
        registry.to_public_dict(),
    )
    backup_root: Path | None = None
    if root.exists():
        backup_root = Path(
            tempfile.mkdtemp(
                prefix=f".{root.name}.contract-cut-backup-",
                dir=root.parent,
            )
        )
        backup_root.rmdir()
        os.replace(root, backup_root)
    try:
        os.replace(temporary_root, root)
    except Exception:
        if backup_root is not None and backup_root.exists() and not root.exists():
            os.replace(backup_root, root)
        raise
    if backup_root is not None and backup_root.exists():
        shutil.rmtree(backup_root)
    return root


def _node_depth(tree: ContractCutTreeState, node_id: str) -> int:
    depth = 0
    if tree.is_atom_entry(node_id):
        depth = 1
        parent_id = tree.candidate_parent_id(node_id)
        if parent_id is None:
            return depth
        node_id = parent_id
    current = tree.nodes[node_id]
    while current.parent_id is not None:
        depth += 1
        current = tree.nodes[current.parent_id]
    return depth


def load_raw_trajectory_records(path: str | Path) -> list[RawTrajectoryRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("records file must contain a JSON list")
    records = [RawTrajectoryRecord.from_dict(dict(row)) for row in payload]
    trajectory_ids = [record.trajectory_id for record in records]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("records file contains duplicate trajectory IDs")
    return records

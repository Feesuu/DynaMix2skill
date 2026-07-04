from __future__ import annotations

from typing import Any

from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep


def officeqa_results_to_records(results: list[dict[str, Any]]) -> list[RawTrajectoryRecord]:
    return [officeqa_result_to_record(row) for row in results]


def officeqa_result_to_record(row: dict[str, Any]) -> RawTrajectoryRecord:
    task_id = str(row.get("id") or row.get("uid") or "")
    steps = _conversation_to_steps(row.get("conversation", []))
    return RawTrajectoryRecord(
        trajectory_id=f"officeqa:{task_id}",
        task_id=task_id,
        trial_index=0,
        instruction=str(row.get("question") or ""),
        instruction_type=str(row.get("task_type") or row.get("category") or "officeqa"),
        final_response=str(row.get("response") or ""),
        success=bool(int(row.get("hard", 0) or 0)),
        verifier_score=float(row.get("soft", 0.0) or 0.0),
        verifier_feedback="correct" if int(row.get("hard", 0) or 0) else _safe_failure_reason(row.get("fail_reason")),
        steps=steps,
        runtime_metadata={
            "benchmark": "officeqa",
            "n_turns": int(row.get("n_turns", len(steps)) or len(steps)),
            "agent_ok": bool(row.get("agent_ok", False)),
        },
        service_metadata={
            "model": row.get("model", ""),
            "base_url": row.get("base_url", ""),
        },
        extra={
            "officeqa_result": _officeqa_result_diagnostic(row),
            "primary_eval": _copy_mapping(row.get("primary_eval", {})),
            "official_reward_audit": _copy_optional_mapping(row.get("official_reward_audit")),
        },
    )


def _safe_failure_reason(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "answer_mismatch"
    if "expected" in text.lower():
        return "answer_mismatch"
    return text


def _copy_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _copy_optional_mapping(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _officeqa_result_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "predicted_answer",
        "ground_truth",
        "fail_reason",
        "source_files",
        "source_docs",
        "resolved_source_paths",
        "oracle_parsed_pages_included",
        "oracle_parsed_pages_chars",
        "agent_ok",
        "n_turns",
        "resume_fingerprint",
    )
    return {key: row[key] for key in keys if key in row}


def _conversation_to_steps(conversation: object) -> list[TrajectoryStep]:
    if not isinstance(conversation, list):
        return []
    steps: list[TrajectoryStep] = []
    last_model_output = ""
    for event in conversation:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type") or event.get("role")
        if event_type == "message":
            last_model_output = str(event.get("content") or "")
            if not event.get("tool_calls") and not _looks_like_text_tool_call(last_model_output):
                steps.append(TrajectoryStep(
                    step_id=len(steps),
                    raw_model_output=last_model_output,
                    action="",
                    observation="",
                    tool_name=None,
                    action_valid=True,
                ))
        elif event_type == "tool_call":
            raw_obs = str(event.get("obs") or "")
            tool_name = _tool_name_from_event(event)
            if steps and steps[-1].raw_model_output == last_model_output and not steps[-1].action and not steps[-1].observation:
                steps.pop()
            steps.append(TrajectoryStep(
                step_id=len(steps),
                raw_model_output=last_model_output,
                action=str(event.get("cmd") or ""),
                observation=raw_obs,
                tool_name=tool_name,
                action_valid=not raw_obs.startswith("[tool error"),
            ))
    return steps


def _looks_like_text_tool_call(text: str) -> bool:
    lowered = str(text or "").lower()
    return "<tool_call" in lowered or '"name"' in lowered and '"arguments"' in lowered


def _tool_name_from_event(event: dict[str, Any]) -> str:
    explicit = str(event.get("tool_name") or "").strip()
    if explicit:
        return explicit
    cmd = str(event.get("cmd") or "").strip()
    if "(" in cmd:
        candidate = cmd.split("(", 1)[0].strip()
        if candidate:
            return candidate
    return "officeqa_tool"

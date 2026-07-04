from __future__ import annotations

import json
from .schemas import RawTrajectoryRecord


def render_embedding_trace(record: RawTrajectoryRecord) -> str:
    """Trace2Skill-aligned trajectory text used for embedding/clustering.

    It deliberately excludes local paths, verifier outputs, task files, labels,
    and ground truth.  This matches the project requirement that clustering is
    driven by q_i and (r_k, a_k, o_k), not by post-hoc answers or local paths.
    """
    lines = [
        f"instruction: {record.instruction}",
        f"instruction_type: {record.instruction_type}",
        "trajectory_steps:",
    ]
    for step in record.steps:
        lines.extend([
            f"[step {step.step_id}] raw_model_output:",
            step.raw_model_output.strip(),
            "action:",
            step.action.strip(),
            "observation:",
            step.observation.strip(),
        ])
    return "\n".join(lines).strip()


def render_analysis_bundle_text(record: RawTrajectoryRecord) -> str:
    """Full evidence bundle for cluster-level analyst prompts.

    This view can include verifier and provenance fields because it is used for
    abstraction, not for similarity computation.  Downstream prompts still tell
    the LLM not to copy absolute paths or raw debug logs into reusable skills.
    """
    payload = {
        "trajectory_id": record.trajectory_id,
        "task_id": record.task_id,
        "trial_index": record.trial_index,
        "q_i": {
            "instruction": record.instruction,
            "instruction_type": record.instruction_type,
            "answer_position": record.answer_position,
            "spreadsheet_path": record.spreadsheet_path,
            "output_path": record.output_path,
        },
        "y_i": {
            "success": record.success,
            "verifier_score": record.verifier_score,
            "verifier_feedback": record.verifier_feedback,
            "final_response": record.final_response,
        },
        "trajectory_steps": [s.to_dict() for s in record.steps],
        "runtime_metadata": record.runtime_metadata,
        "service_metadata": record.service_metadata,
        "extra": record.extra,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_compact_analysis_bundle_text(
    record: RawTrajectoryRecord,
    *,
    max_chars: int = 60000,
    max_steps: int = 12,
    max_step_chars: int = 6000,
    max_final_response_chars: int = 12000,
) -> str:
    """Compact evidence bundle for analyst prompts.

    ReAct traces can include long document snippets, workbook dumps, or repeated
    tool observations.  The analyst still needs train diagnostics and enough
    trajectory evidence to infer reusable causes, but it does not need every raw
    byte from the trace in the prompt.  Raw records remain unchanged on disk.
    """
    max_chars = max(1000, int(max_chars))
    max_steps = max(1, int(max_steps))
    max_step_chars = max(200, int(max_step_chars))
    max_final_response_chars = max(200, int(max_final_response_chars))
    payload = {
        "trajectory_id": record.trajectory_id,
        "task_id": record.task_id,
        "trial_index": record.trial_index,
        "q_i": {
            "instruction": record.instruction,
            "instruction_type": record.instruction_type,
            "answer_position": record.answer_position,
            "spreadsheet_path": record.spreadsheet_path,
            "output_path": record.output_path,
        },
        "y_i": {
            "success": record.success,
            "verifier_score": record.verifier_score,
            "verifier_feedback": _clip_text(record.verifier_feedback, 2000),
            "final_response": _clip_text(record.final_response, max_final_response_chars),
        },
        "trajectory_steps": _compact_steps(record, max_steps=max_steps, max_step_chars=max_step_chars, preserve_all=True),
        "runtime_metadata": _compact_json_value(record.runtime_metadata, max_chars=4000),
        "service_metadata": _compact_json_value(record.service_metadata, max_chars=2000),
        "extra": _compact_json_value(record.extra, max_chars=max_chars // 3),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(rendered) <= max_chars:
        return rendered
    payload["trajectory_steps"] = _compact_steps(record, max_steps=max_steps, max_step_chars=max_step_chars, preserve_all=False)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(rendered) <= max_chars:
        return rendered
    payload["trajectory_steps"] = _compact_steps(record, max_steps=min(max_steps, 6), max_step_chars=max(800, max_step_chars // 2), preserve_all=False)
    payload["extra"] = _compact_json_value(record.extra, max_chars=max(1000, max_chars // 6))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(rendered) <= max_chars:
        return rendered
    return _clip_text(rendered, max_chars)


def _compact_steps(record: RawTrajectoryRecord, *, max_steps: int, max_step_chars: int, preserve_all: bool = False) -> list[dict[str, object]]:
    steps = record.steps
    if preserve_all or len(steps) <= max_steps:
        selected = steps
        omitted = 0
    else:
        head_count = max_steps // 2
        tail_count = max_steps - head_count
        selected = [*steps[:head_count], *steps[-tail_count:]]
        omitted = len(steps) - len(selected)
    compacted = []
    for step in selected:
        compacted.append({
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "action_valid": step.action_valid,
            "raw_model_output": _clip_text(step.raw_model_output, max_step_chars),
            "action": _clip_text(step.action, max_step_chars),
            "observation": _clip_text(step.observation, max_step_chars),
        })
    if omitted:
        compacted.insert(max(1, len(compacted) // 2), {"omitted_middle_step_count": omitted})
    return compacted


def _compact_json_value(value: object, *, max_chars: int) -> object:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_char_count": len(rendered),
        "preview": _clip_text(rendered, max_chars),
    }


def _clip_text(value: object, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    max_chars = max(20, int(max_chars))
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f"\n...[truncated {omitted} chars]...\n"
    available = max_chars - len(marker)
    if available <= 0:
        return text[:max_chars]
    head = available // 2
    tail = available - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)

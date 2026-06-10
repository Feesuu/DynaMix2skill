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
        f"answer_position: {record.answer_position}",
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


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .data import OfficeQAItem
from .reward import evaluate_official_audit, evaluate_skillopt, extract_answer, extract_audit_answer, has_answer, strip_thinking
from .tools import TOOL_SCHEMAS, build_oracle_parsed_pages_context, resolve_candidate_files, resolve_docs_roots, run_tool

ROLLOUT_SYSTEM_TEMPLATE = """You are an expert OfficeQA agent working over local Treasury bulletin text files.

{skill_section}## Rules
1. Use only the provided local document tools to inspect candidate files.
2. Narrow to the most relevant file before reading long passages.
3. Prefer short targeted searches, then small reads around matching evidence.
4. Do not invent values that are not grounded in the retrieved text.
5. When the question requires arithmetic, compute only after extracting the exact operands.
6. If you have enough evidence, return the final answer inside <answer>...</answer>.

## Tool Use
{tool_instructions}

## Final Answer Format
When you are ready to answer, emit the final answer inside <answer>...</answer> and do not request another tool.
"""

FUNCTION_TOOL_INSTRUCTIONS = (
    "Use the provided function tools directly when you need them. Prefer searching and small reads before answering. "
    "Do not ask the user for permission to use tools; just call the tools."
)

TEXT_TOOL_INSTRUCTIONS = """The backend does not support native function tools in this run. Request one local document tool by writing exactly one JSON object inside <tool_call>...</tool_call>, then wait for the tool result.

Available tools:
- {"name": "grep", "arguments": {"pattern": "literal text", "path": "/absolute/candidate/file.txt"}}
- {"name": "read", "arguments": {"path": "/absolute/candidate/file.txt", "start": 1, "limit": 80}}
- {"name": "glob", "arguments": {"pattern": "*.txt"}}

When you have enough evidence, stop requesting tools and return the final answer inside <answer>...</answer>."""


class ToolProtocolUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficeQARolloutConfig:
    base_url: str = "mock://deterministic"
    model: str = "Qwen3.5-9B-AWQ"
    api_key: str = "EMPTY"
    temperature: float = 0.6
    timeout_seconds: float = 1200.0
    max_tool_turns: int = 30
    max_completion_tokens: int | None = None
    workers: int = 8
    thinking: bool = True
    docs_dirs: tuple[str, ...] = ()
    reward_path: str = ""
    resume: bool = True
    resume_fingerprint: str = ""


SkillContentFn = Callable[[OfficeQAItem], str]


def build_system_prompt(skill_content: str = "", *, text_tool_fallback: bool = False) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    tool_instructions = TEXT_TOOL_INSTRUCTIONS if text_tool_fallback else FUNCTION_TOOL_INSTRUCTIONS
    return ROLLOUT_SYSTEM_TEMPLATE.format(skill_section=skill_section, tool_instructions=tool_instructions)


def build_user_prompt(item: OfficeQAItem, candidate_files: list[str], *, oracle_context: str = "") -> str:
    parts = [f"## Question\n{item.question}"]
    if oracle_context.strip():
        parts.append(f"## Oracle Parsed Pages\n{oracle_context.strip()}")
    file_block = "\n".join(f"- {path}" for path in candidate_files[:20]) or "- none resolved"
    parts.append(f"## Candidate Files\n{file_block}")
    if item.source_docs:
        parts.append("## Source Hints\n" + "\n".join(f"- {hint}" for hint in item.source_docs))
    return "\n\n".join(parts)


def run_officeqa_batch(
    items: list[OfficeQAItem],
    out_root: str | Path,
    config: OfficeQARolloutConfig,
    *,
    skill_content_fn: SkillContentFn | None = None,
) -> list[dict[str, Any]]:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "officeqa_results.jsonl"
    existing = _load_existing_results(results_path, resume_fingerprint=config.resume_fingerprint) if config.resume else []
    done_ids = {str(row.get("id")) for row in existing}
    pending = [item for item in items if item.uid not in done_ids]
    results = list(existing)
    if not pending:
        _write_results_json(out, results)
        return results

    docs_roots = resolve_docs_roots(list(config.docs_dirs))
    mode = "a" if config.resume else "w"
    with results_path.open(mode, encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, int(config.workers))) as pool:
        futures = {
            pool.submit(_process_one, item, out, config, docs_roots, skill_content_fn(item) if skill_content_fn else ""): item
            for item in pending
        }
        for future in as_completed(futures):
            row = future.result()
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            results.append(row)
            total = len(items)
            correct = sum(1 for item_row in results if int(item_row.get("hard", 0) or 0))
            print(f"[officeqa] {len(results)}/{total} hard={correct}/{len(results)} latest={row.get('id')}", flush=True)
    results.sort(key=lambda row: str(row.get("id")))
    _write_results_json(out, results)
    return results


def _process_one(
    item: OfficeQAItem,
    out_root: Path,
    config: OfficeQARolloutConfig,
    docs_roots: list[str],
    skill_content: str,
) -> dict[str, Any]:
    pred_dir = out_root / "predictions" / item.uid
    pred_dir.mkdir(parents=True, exist_ok=True)
    candidate_files = resolve_candidate_files(item.source_files, docs_roots)
    allowed_paths = candidate_files
    oracle_context = build_oracle_parsed_pages_context(item.source_files, item.source_docs, docs_roots)
    system = build_system_prompt(skill_content)
    user = build_user_prompt(item, candidate_files, oracle_context=oracle_context)
    conversation: list[dict[str, Any]] = [{"role": "user", "content": user}]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    final_response = ""
    fail_reason = ""
    use_function_tools = True

    try:
        client = _openai_client(config)
        for turn in range(1, max(1, int(config.max_tool_turns)) + 1):
            try:
                response = _chat_once(client, config, messages, use_function_tools=use_function_tools)
            except ToolProtocolUnsupported as exc:
                if not use_function_tools:
                    raise
                use_function_tools = False
                system = build_system_prompt(skill_content, text_tool_fallback=True)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                conversation.append({
                    "type": "protocol_fallback",
                    "turn": turn,
                    "from": "openai_function_tools",
                    "to": "text_tool_call_tags",
                    "reason": str(exc),
                })
                response = _chat_once(client, config, messages, use_function_tools=False)
            message = response.choices[0].message
            content = getattr(message, "content", None) or ""
            final_response = content
            assistant_message: dict[str, Any] = {"role": "assistant", "content": strip_thinking(content)}
            tool_calls = list(getattr(message, "tool_calls", None) or []) if use_function_tools else []
            text_tool_call = _parse_text_tool_call(content) if not use_function_tools else None
            if tool_calls:
                assistant_message["tool_calls"] = [_tool_call_dump(call) for call in tool_calls]
            messages.append(assistant_message)
            conversation.append({"type": "message", "turn": turn, "content": content, "tool_calls": assistant_message.get("tool_calls", [])})
            if tool_calls:
                for call in tool_calls:
                    tool_name = call.function.name
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    cmd, obs = run_tool(tool_name, arguments, allowed_roots=docs_roots, allowed_paths=allowed_paths)
                    conversation.append({"type": "tool_call", "turn": turn, "tool_name": tool_name, "cmd": cmd, "obs": obs})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
                continue
            if text_tool_call is not None:
                tool_name, arguments = text_tool_call
                cmd, obs = run_tool(tool_name, arguments, allowed_roots=docs_roots, allowed_paths=allowed_paths)
                conversation.append({"type": "tool_call", "turn": turn, "tool_name": tool_name, "cmd": cmd, "obs": obs, "tool_protocol": "text"})
                messages.append({"role": "user", "content": _render_text_tool_observation(cmd, obs)})
                continue
            if has_answer(content):
                break
            fail_reason = f"Exceeded tool-turn budget ({config.max_tool_turns})" if turn == config.max_tool_turns else "Model neither produced a tool request nor a final answer"
            break
    except Exception as exc:  # noqa: BLE001
        fail_reason = f"error: {exc}"

    final_answer = extract_answer(final_response) if final_response else ""
    audit_answer = extract_audit_answer(final_response) if final_response else ""
    primary_eval = evaluate_skillopt(final_answer, item.ground_truth) if final_answer else {
        "scorer": "skillopt_em_f1",
        "em": 0.0,
        "f1": 0.0,
        "hard": 0,
        "score": 0.0,
        "predicted_answer": "",
        "gold_answer": item.ground_truth,
    }
    official_audit = evaluate_official_audit(audit_answer, item.ground_truth, config.reward_path)
    row = {
        "id": item.uid,
        "uid": item.uid,
        "question": item.question,
        "task_type": item.task_type,
        "category": item.task_type,
        "predicted_answer": primary_eval["predicted_answer"],
        "response": final_response,
        "ground_truth": item.ground_truth,
        "source_files": list(item.source_files),
        "source_docs": list(item.source_docs),
        "resolved_source_paths": candidate_files,
        "oracle_parsed_pages_included": bool(oracle_context),
        "oracle_parsed_pages_chars": len(oracle_context),
        "hard": int(primary_eval["hard"]),
        "soft": float(primary_eval["f1"]),
        "primary_eval": primary_eval,
        "official_reward_audit": official_audit,
        "fail_reason": fail_reason or ("" if primary_eval["hard"] else "answer_mismatch"),
        "agent_ok": not bool(fail_reason),
        "n_turns": len(conversation),
        "conversation": conversation,
        "target_system_prompt": system,
        "target_user_prompt": user,
        "model": config.model,
        "base_url": config.base_url,
        "resume_fingerprint": config.resume_fingerprint,
    }
    (pred_dir / "target_system_prompt.txt").write_text(system, encoding="utf-8")
    (pred_dir / "target_user_prompt.txt").write_text(user, encoding="utf-8")
    (pred_dir / "conversation.json").write_text(json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _openai_client(config: OfficeQARolloutConfig):
    if config.base_url.startswith("mock://"):
        return _MockClient()
    try:
        from openai import OpenAI
    except ImportError:
        from dynamix_trace2skill.openai_compat import OpenAI
    from dynamix_trace2skill.clients import _openai_http_client_kwargs
    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_seconds, **_openai_http_client_kwargs())


def _chat_once(
    client: Any,
    config: OfficeQARolloutConfig,
    messages: list[dict[str, Any]],
    *,
    use_function_tools: bool = True,
) -> Any:
    if config.base_url.startswith("mock://"):
        return client.chat.completions.create(messages=messages)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": float(config.temperature),
    }
    if config.max_completion_tokens is not None:
        kwargs["max_tokens"] = int(config.max_completion_tokens)
    if use_function_tools:
        kwargs["tools"] = TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(config.thinking)}}
    wait_schedule = (0.0, 2.0, 5.0, 15.0)
    for attempt, wait in enumerate(wait_schedule):
        if wait:
            time.sleep(wait)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if use_function_tools and _is_auto_tool_choice_unsupported(exc):
                raise ToolProtocolUnsupported(str(exc)) from exc
            if attempt >= len(wait_schedule) - 1:
                raise
    raise RuntimeError("OfficeQA chat request failed without exception")


def _is_auto_tool_choice_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "auto" in text and "tool choice" in text and "enable-auto-tool-choice" in text


def _parse_text_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    text = strip_thinking(content or "")
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        match = re.search(r"<tool>\s*(.*?)\s*</tool>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or payload.get("tool") or payload.get("tool_name") or "").strip()
    if name not in {"glob", "read", "grep"}:
        return None
    raw_args = payload.get("arguments", payload.get("args", {}))
    arguments = dict(raw_args) if isinstance(raw_args, dict) else {}
    return name, arguments


def _render_text_tool_observation(cmd: str, obs: str) -> str:
    return (
        f"Tool result for {cmd}:\n"
        f"{obs}\n\n"
        "Use another <tool_call>...</tool_call> if more evidence is needed, otherwise return the final answer inside <answer>...</answer>."
    )


def _tool_call_dump(call: Any) -> dict[str, Any]:
    if hasattr(call, "model_dump"):
        return call.model_dump(mode="json")
    return {
        "id": getattr(call, "id", ""),
        "type": "function",
        "function": {
            "name": getattr(getattr(call, "function", None), "name", ""),
            "arguments": getattr(getattr(call, "function", None), "arguments", ""),
        },
    }


def _load_existing_results(path: Path, *, resume_fingerprint: str = "") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                if resume_fingerprint and row.get("resume_fingerprint") != resume_fingerprint:
                    continue
                by_id[str(row.get("id"))] = row
    return list(by_id.values())


def _write_results_json(out: Path, results: list[dict[str, Any]]) -> None:
    (out / "officeqa_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


class _MockClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self

    def create(self, **_: Any) -> Any:
        from types import SimpleNamespace
        message = SimpleNamespace(content="<answer>mock</answer>", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

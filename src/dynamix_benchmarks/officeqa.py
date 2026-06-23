from __future__ import annotations

import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from dynamix_trace2skill.schemas import RawTrajectoryRecord, TrajectoryStep
from dynamix_trace2skill.skillbank import SkillBankSelector, selected_experience_to_system_content
from react_agent import AgentConfig, OpenAIClient, ReActAgent
from react_agent.tools import Tool, ToolParameter


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
OFFICIAL_FINAL_ANSWER_RE = re.compile(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
OFFICIAL_REWARD_SHA256 = "0d91698c87df6d889339aac36f63ae0966607f169890b0bf8b472b26bfe8138f"
_NUMERIC_CHARS = set("0123456789.-")


@dataclass(frozen=True)
class OfficeQAItem:
    id: str
    uid: str
    question: str
    ground_truth: str
    category: str = "officeqa"
    source_files: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "uid": self.uid,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.ground_truth,
            "answers": [self.ground_truth] if self.ground_truth else [],
            "task_type": self.category,
            "category": self.category,
            "source_files": list(self.source_files),
            "source_docs": list(self.source_docs),
            "split": self.split,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "uid": self.uid,
            "question": self.question,
            "task_type": self.category,
            "category": self.category,
            "source_files": list(self.source_files),
            "source_docs": list(self.source_docs),
            "split": self.split,
        }


@dataclass(frozen=True)
class OfficeQAEvalResult:
    score: float
    hard: int
    predicted_answer: str
    gold_answer: str
    evaluator: str
    fail_reason: str = ""
    f1: float = 0.0


class OfficeQAOfficialRewardError(RuntimeError):
    pass


class OfficeQAInfrastructureError(RuntimeError):
    pass


def is_infrastructure_agent_error(error: str | None) -> bool:
    return bool(error and str(error).startswith("Exception during execution:"))


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    if "," in text and not text.lower().endswith((".txt", ".pdf", ".json")):
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def normalize_item(row: dict[str, Any], *, split: str = "") -> OfficeQAItem:
    uid = str(row.get("uid") or row.get("id") or "").strip()
    if not uid:
        raise ValueError(f"OfficeQA row is missing uid/id: {row}")
    return OfficeQAItem(
        id=uid,
        uid=uid,
        question=str(row.get("question") or "").strip(),
        ground_truth=str(row.get("ground_truth") or row.get("answer") or "").strip(),
        category=str(row.get("category") or row.get("difficulty") or row.get("task_type") or "officeqa").strip() or "officeqa",
        source_files=parse_list_field(row.get("source_files")),
        source_docs=parse_list_field(row.get("source_docs")),
        split=str(row.get("split") or split or "").strip(),
    )


def _load_json_or_csv(path: Path, *, split: str = "") -> list[OfficeQAItem]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [normalize_item(dict(row), split=split) for row in csv.DictReader(handle)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"OfficeQA split file must contain a JSON array or items array: {path}")
    return [normalize_item(row, split=split) for row in rows if isinstance(row, dict)]


def load_officeqa_items(split_dir: str | Path, *, split: str, start: int = 0, end: int | None = None) -> list[OfficeQAItem]:
    root = Path(split_dir).expanduser()
    candidates = [
        root / split / "items.json",
        root / "splits" / split / "items.json",
        root / f"{split}.json",
        root / f"{split}.csv",
    ]
    for path in candidates:
        if path.is_file():
            items = _load_json_or_csv(path, split=split)
            break
    else:
        raise FileNotFoundError(
            f"OfficeQA split {split!r} not found under {root}. "
            "Expected <split>/items.json, splits/<split>/items.json, <split>.json, or <split>.csv."
        )
    end_idx = len(items) if end is None else int(end)
    return items[int(start):end_idx]


def resolve_reward_path(
    split_dir: str | Path,
    reward_path: str | Path | None = None,
    *,
    required: bool = False,
) -> Path | None:
    def verified(path: Path) -> Path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != OFFICIAL_REWARD_SHA256:
            raise RuntimeError(
                "OfficeQA reward.py hash mismatch. "
                f"Expected official sha256={OFFICIAL_REWARD_SHA256}, got {digest} for {path}."
            )
        return path

    if reward_path:
        path = Path(reward_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"explicit OfficeQA reward.py not found: {path}")
        return verified(path)
    root = Path(split_dir).expanduser()
    for path in (root / "reward.py", root.parent / "reward.py"):
        if path.is_file():
            return verified(path)
    if required:
        raise FileNotFoundError(
            f"Official OfficeQA reward.py not found under {root} or {root.parent}. "
            "Pass --reward-path for formal evaluation, or enable fallback only for debug."
        )
    return None


def strip_think_blocks(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def extract_final_answer(text: str) -> str:
    stripped = strip_think_blocks(text)
    match = ANSWER_RE.search(stripped)
    return match.group(1).strip() if match else ""


def extract_official_or_answer(text: str) -> str:
    stripped = strip_think_blocks(text)
    for pattern in (OFFICIAL_FINAL_ANSWER_RE, ANSWER_RE):
        match = pattern.search(stripped)
        if match:
            return match.group(1).strip()
    return ""


def _load_reward_module(path: Path):
    spec = importlib.util.spec_from_file_location("officeqa_official_reward", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import OfficeQA reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _normalize_text_answer(text: str) -> str:
    return re.sub(r"\s+", " ", strip_think_blocks(text).lower().strip().strip("\"'"))


def normalize_answer(text: str) -> str:
    text = strip_think_blocks(text).lower().strip()
    text = text.replace(",", "")
    text = "".join(ch for ch in text if ch not in string.punctuation or ch in _NUMERIC_CHARS or ch == "%")
    text = re.sub(r"\b(million|millions|billion|billions|dollars|dollar|nominal)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _parse_float(text: str) -> float | None:
    cleaned = str(text).strip().lower().replace("$", "").replace(",", "")
    scale = 1.0
    if "billion" in cleaned:
        scale = 1_000_000_000.0
    elif "million" in cleaned:
        scale = 1_000_000.0
    elif "thousand" in cleaned:
        scale = 1_000.0
    cleaned = re.sub(r"\(([-+]?[0-9.]+)\)", r"-\1", cleaned)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0)) * scale
    except ValueError:
        return None


def evaluate_officeqa_prediction(
    *,
    prediction_text: str,
    gold_answer: str,
    reward_path: str | Path | None = None,
    tolerance: float = 0.0,
    evaluator: str = "skillopt",
    allow_fallback: bool = False,
    raise_official_errors: bool = True,
) -> OfficeQAEvalResult:
    reward = Path(reward_path).expanduser() if reward_path else None
    evaluator_mode = str(evaluator or "skillopt").strip().lower()
    answer = extract_official_or_answer(prediction_text) if evaluator_mode in {"official", "official_reward"} else extract_final_answer(prediction_text)
    if evaluator_mode in {"official", "official_reward"}:
        if reward is None or not reward.is_file():
            raise FileNotFoundError("OfficeQA official_reward evaluator requires --reward-path")
        module = _load_reward_module(reward)
        official_prediction = f"<FINAL_ANSWER>{answer}</FINAL_ANSWER>" if answer else strip_think_blocks(prediction_text)
        try:
            score = float(module.score_answer(gold_answer, official_prediction, tolerance=float(tolerance)))
            return OfficeQAEvalResult(
                score=score,
                hard=int(score >= 1.0),
                predicted_answer=answer,
                gold_answer=gold_answer,
                evaluator=f"official_reward:{reward}",
                fail_reason="" if score >= 1.0 else "official_reward_mismatch",
                f1=score,
            )
        except Exception as exc:  # noqa: BLE001
            if raise_official_errors:
                raise OfficeQAOfficialRewardError(f"OfficeQA official reward failed for {reward}: {exc}") from exc
            return OfficeQAEvalResult(
                score=0.0,
                hard=0,
                predicted_answer=answer,
                gold_answer=gold_answer,
                evaluator=f"official_reward:{reward}",
                fail_reason=f"official_reward_error:{exc}",
                f1=0.0,
            )

    if evaluator_mode == "skillopt":
        if not answer:
            return OfficeQAEvalResult(
                score=0.0,
                hard=0,
                predicted_answer="",
                gold_answer=gold_answer,
                evaluator="skillopt_normalized_em_f1",
                fail_reason="missing_answer_tag",
                f1=0.0,
            )
        em = exact_match(answer, gold_answer)
        f1 = token_f1(answer, gold_answer)
        return OfficeQAEvalResult(
            score=em,
            hard=int(em),
            predicted_answer=answer,
            gold_answer=gold_answer,
            evaluator="skillopt_normalized_em_f1",
            fail_reason="" if em else f"skillopt_mismatch:f1={f1:.3f}",
            f1=f1,
        )

    if evaluator_mode not in {"fallback", "fallback_numeric_or_text"}:
        raise ValueError(f"unsupported OfficeQA evaluator: {evaluator!r}")
    if not allow_fallback:
        raise ValueError("fallback OfficeQA evaluator is debug-only; pass allow_fallback=True")

    pred_num = _parse_float(answer)
    gold_num = _parse_float(gold_answer)
    if pred_num is not None and gold_num is not None:
        ok = math.isclose(pred_num, gold_num, abs_tol=float(tolerance)) if gold_num == 0 else abs(pred_num - gold_num) / abs(gold_num) <= float(tolerance)
        return OfficeQAEvalResult(1.0 if ok else 0.0, int(ok), answer, gold_answer, "fallback_numeric_or_text", "" if ok else "numeric_mismatch", 1.0 if ok else 0.0)
    ok = _normalize_text_answer(answer) == _normalize_text_answer(gold_answer)
    return OfficeQAEvalResult(1.0 if ok else 0.0, int(ok), answer, gold_answer, "fallback_numeric_or_text", "" if ok else "text_mismatch", 1.0 if ok else 0.0)


def _candidate_names(source_file: str) -> set[str]:
    path = Path(str(source_file).strip())
    names = {path.name}
    if path.suffix:
        names.add(path.stem + ".txt")
        names.add(path.stem + ".json")
    return {name for name in names if name}


def _relative_to_any(path: Path, roots: list[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _is_regular_file_under_roots(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        return False
    if not path.is_file() or path.is_symlink():
        return False
    for root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


class OfficeQADocTools:
    def __init__(self, docs_dirs: Iterable[str | Path], *, source_files: Iterable[str] = ()):
        roots: list[Path] = []
        for value in docs_dirs:
            path = Path(value).expanduser()
            if not path.is_dir():
                continue
            transformed = path / "transformed"
            roots.append(transformed if transformed.is_dir() else path)
        if not roots:
            raise FileNotFoundError("No OfficeQA docs directory found. Pass --docs-dir or set OFFICEQA_DOCS_DIR.")
        self.roots = roots
        self.allowed_names: set[str] = set()
        for source_file in source_files:
            self.allowed_names.update(_candidate_names(source_file))

    def resolve_candidate_files(self) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()
        for root in self.roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.is_symlink() or not self._is_allowed(path):
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
        return candidates

    def _is_allowed(self, path: Path) -> bool:
        if not _is_regular_file_under_roots(path, self.roots):
            return False
        return not self.allowed_names or path.resolve().name in self.allowed_names

    def _resolve_user_path(self, value: str) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
        if candidate.is_absolute() and candidate.is_file() and self._is_allowed(candidate):
            return candidate
        for root in self.roots:
            direct = root / text
            if direct.is_file() and self._is_allowed(direct):
                return direct
            for path in root.rglob("*"):
                if path.is_file() and not path.is_symlink() and (path.name == text or path.as_posix().endswith(text)) and self._is_allowed(path):
                    return path
        return None

    def glob(self, pattern: str = "*") -> str:
        matches: list[str] = []
        for root in self.roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.is_symlink() or not self._is_allowed(path):
                    continue
                rel = path.relative_to(root).as_posix()
                if Path(rel).match(pattern) or path.name == pattern:
                    matches.append(rel)
                if len(matches) >= 50:
                    break
        return "\n".join(matches) if matches else "[no matches]"

    def read(self, path: str, start: int = 1, limit: int = 80) -> str:
        resolved = self._resolve_user_path(path)
        if resolved is None:
            return f"[read error: path not found or not allowed: {path}]"
        start = max(int(start or 1), 1)
        limit = min(max(int(limit or 80), 1), 200)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        excerpt = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start - 1:start - 1 + limit], start=start))
        return excerpt[:6000] if excerpt else "[empty file]"

    def grep(self, pattern: str, path: str = "") -> str:
        query = str(pattern or "").strip().lower()
        if not query:
            return "[grep error: empty pattern]"
        targets = [self._resolve_user_path(path)] if str(path or "").strip() else self.resolve_candidate_files()
        matches: list[str] = []
        for resolved in [target for target in targets if target is not None]:
            rel = _relative_to_any(resolved, self.roots)
            for idx, line in enumerate(resolved.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if query in line.lower():
                    matches.append(f"{rel}:{idx}: {line}")
                if len(matches) >= 30:
                    break
            if len(matches) >= 30:
                break
        return "\n".join(matches) if matches else "[no matches]"

    def to_tools(self) -> list[Tool]:
        return [
            Tool(
                name="glob",
                description="Find candidate OfficeQA local document files by relative-path glob pattern.",
                func=self.glob,
                parameters=[ToolParameter("pattern", "string", "Glob pattern such as *.txt or *1987*.txt", required=False, default="*")],
            ),
            Tool(
                name="grep",
                description="Search OfficeQA local text files for a literal case-insensitive pattern.",
                func=self.grep,
                parameters=[
                    ToolParameter("pattern", "string", "Literal text to search for"),
                    ToolParameter("path", "string", "Optional relative path. Omit to search candidate source files.", required=False, default=""),
                ],
            ),
            Tool(
                name="read",
                description="Read a line-window from an OfficeQA local text file.",
                func=self.read,
                parameters=[
                    ToolParameter("path", "string", "Relative file path returned by glob/grep"),
                    ToolParameter("start", "integer", "1-based start line", required=False, default=1),
                    ToolParameter("limit", "integer", "Maximum number of lines", required=False, default=80),
                ],
            ),
        ]


def _page_number_from_source_doc(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in (r"(?:[?&]|^)page=(\d+)", r"(?:[?&]|^)pagenum=(\d+)", r"(?:[?&]|^)page_id=(\d+)"):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _iter_oracle_refs(source_files: list[str], source_docs: list[str]) -> list[tuple[str, int, str]]:
    refs: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, source_doc in enumerate(source_docs):
        page_number = _page_number_from_source_doc(source_doc)
        if page_number is None:
            continue
        if index < len(source_files):
            source_file = source_files[index]
        elif len(source_files) == 1:
            source_file = source_files[0]
        else:
            continue
        key = (source_file, page_number, source_doc)
        if key in seen:
            continue
        seen.add(key)
        refs.append(key)
    return refs


def _parsed_root_candidates(docs_dirs: Iterable[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for value in docs_dirs:
        root = Path(value).expanduser()
        for candidate in (
            root,
            root.parent,
            root / "treasury_bulletins_parsed",
            root.parent / "treasury_bulletins_parsed",
        ):
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _locate_parsed_json(source_file: str, docs_dirs: Iterable[str | Path]) -> Path | None:
    source_path = Path(str(source_file).strip())
    stem = source_path.stem if source_path.suffix else source_path.name
    if not stem:
        return None
    candidate_names = [stem + ".json"]
    if source_path.suffix == ".json":
        candidate_names.insert(0, source_path.name)
    allowed_roots: list[Path] = []
    candidates: list[Path] = []
    for root in _parsed_root_candidates(docs_dirs):
        json_root = root / "jsons"
        if json_root.is_dir():
            allowed_roots.append(json_root)
        for name in candidate_names:
            candidates.append(json_root / name)
    return next((path for path in candidates if _is_regular_file_under_roots(path, allowed_roots)), None)


class _TableMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"td", "th"} and self._cell is not None and self._row is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(cell)
            self._cell = None
        elif normalized_tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _html_table_to_markdown(raw_html: str) -> str:
    parser = _TableMarkdownParser()
    try:
        parser.feed(raw_html)
    except Exception:  # noqa: BLE001
        parser.rows = []
    if not parser.rows:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?is)<[^>]+>", " ", raw_html))).strip()
    width = max(len(row) for row in parser.rows)
    rows = [row + [""] * (width - len(row)) for row in parser.rows]
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(cell.replace("|", "\\|").strip() for cell in header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|").strip() for cell in row) + " |" for row in body)
    return "\n".join(lines)


def _render_parsed_content(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    if "<table" in text.lower():
        return _html_table_to_markdown(text)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _element_page_ids(element: dict[str, Any]) -> set[int]:
    page_ids: set[int] = set()
    bbox = element.get("bbox")
    for box in bbox if isinstance(bbox, list) else []:
        if not isinstance(box, dict):
            continue
        try:
            page_ids.add(int(box.get("page_id")))
        except (TypeError, ValueError):
            continue
    return page_ids


@lru_cache(maxsize=256)
def _load_parsed_elements(json_path: str) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8", errors="replace"))
    document = payload.get("document") if isinstance(payload, dict) else {}
    elements = document.get("elements") if isinstance(document, dict) else []
    if not isinstance(elements, list):
        return ()
    return tuple(element for element in elements if isinstance(element, dict))


@lru_cache(maxsize=2048)
def _render_json_page(json_path: str, page_number: int) -> str:
    blocks: list[str] = []
    for element in _load_parsed_elements(json_path):
        if page_number not in _element_page_ids(element):
            continue
        content = element.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        rendered = _render_parsed_content(content)
        if rendered:
            blocks.append(rendered)
    return "\n\n".join(blocks).strip()


def build_oracle_context(item: OfficeQAItem, docs_dirs: Iterable[str | Path], *, max_chars: int = 80000, max_page_chars: int = 24000) -> str:
    refs = _iter_oracle_refs(item.source_files, item.source_docs)
    if not refs:
        return ""
    blocks: list[str] = []
    total_chars = 0
    seen_pages: set[tuple[str, int]] = set()
    for source_file, page_number, source_doc in refs:
        json_path = _locate_parsed_json(source_file, docs_dirs)
        if json_path is None:
            continue
        page_key = (str(json_path), page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        page_text = _render_json_page(str(json_path), page_number)
        if not page_text:
            continue
        if len(page_text) > max_page_chars:
            omitted = len(page_text) - max_page_chars
            page_text = page_text[:max_page_chars].rstrip() + f"\n\n[... {omitted} characters omitted from this parsed page ...]"
        block = f"### {source_file} page {page_number}\nSource URL: {source_doc}\n\n{page_text}"
        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            block = block[:remaining].rstrip() + "\n\n[... oracle parsed page context truncated ...]"
            blocks.append(block)
            break
        blocks.append(block)
        total_chars += len(block)
    if not blocks:
        return ""
    return (
        "The following content is pre-parsed from the oracle OfficeQA source page(s). "
        "Treat it as primary document evidence and combine it with local document tool evidence when useful.\n\n"
        + "\n\n".join(blocks)
    )


def build_officeqa_system_prompt(*, retrieved_experience: str = "") -> str:
    parts = [
        "You are an expert OfficeQA agent working over local Treasury bulletin text files.",
        "Use the provided local document tools to inspect evidence before answering.",
        "Do not invent values that are not grounded in the retrieved text.",
        "When arithmetic is required, extract exact operands first and then compute.",
        "When ready, return the direct answer inside <answer>...</answer>, then finish with ACTION: TASK_COMPLETE.",
    ]
    if retrieved_experience.strip():
        parts.append(retrieved_experience.strip())
    return "\n\n".join(parts)


def build_officeqa_task_prompt(item: OfficeQAItem, *, candidate_files: list[Path], docs_tools: OfficeQADocTools, oracle_context: str = "") -> str:
    rel_files = [_relative_to_any(path, docs_tools.roots) for path in candidate_files[:20]]
    parts = [f"Question:\n{item.question}", f"Task type: {item.category}"]
    if oracle_context.strip():
        parts.append(f"Oracle parsed page context:\n{oracle_context.strip()}")
    parts.append("Candidate files:\n" + ("\n".join(f"- {path}" for path in rel_files) if rel_files else "- none resolved; use glob/grep to inspect the local corpus"))
    if item.source_docs:
        parts.append("Source hints:\n" + "\n".join(f"- {hint}" for hint in item.source_docs))
    parts.append(
        "Answer format:\n"
        "Return only the concise answer inside <answer>...</answer> before ACTION: TASK_COMPLETE. "
        "Do not include citations or explanation inside the final answer tag."
    )
    return "\n\n".join(parts)


def officeqa_skillbank_query(item: OfficeQAItem) -> str:
    return f"{item.question}\n\nTask type: {item.category}".strip()


def select_retrieved_experience(
    item: OfficeQAItem,
    *,
    skillbank_root: str | Path | None,
    top_k: int,
    selection_log: str | Path | None = None,
) -> str:
    if not skillbank_root or int(top_k) <= 0:
        if selection_log:
            Path(selection_log).parent.mkdir(parents=True, exist_ok=True)
            with Path(selection_log).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "task_id": item.uid,
                    "query_policy": "question + Task type/category; answer/source_docs excluded",
                    "query": officeqa_skillbank_query(item),
                    "top_k": int(top_k),
                    "selected": [],
                }, ensure_ascii=False) + "\n")
        return ""
    selector = SkillBankSelector.from_env(default_skillbank_root=skillbank_root)
    query = officeqa_skillbank_query(item)
    selections = selector.select(query, top_k=int(top_k))
    if selection_log:
        Path(selection_log).parent.mkdir(parents=True, exist_ok=True)
        with Path(selection_log).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "task_id": item.uid,
                "query_policy": "question + Task type/category; answer/source_docs excluded",
                "query": query,
                "top_k": int(top_k),
                "selected": [
                    {"rank": rank, "node_id": sel.skill.node_id, "name": sel.skill.name, "score": sel.score}
                    for rank, sel in enumerate(selections, start=1)
                ],
            }, ensure_ascii=False) + "\n")
    return selected_experience_to_system_content(selections).replace(
        "matches the spreadsheet operation",
        "matches the OfficeQA document-question task",
    )


def _steps_to_record_steps(steps: Iterable[Any]) -> list[TrajectoryStep]:
    out: list[TrajectoryStep] = []
    for index, step in enumerate(steps, start=1):
        action = getattr(step, "action", None)
        action_payload = ""
        tool_name = None
        action_valid = None
        if action is not None:
            tool_name = getattr(action, "name", None)
            action_payload = json.dumps({"name": tool_name or "", "arguments": getattr(action, "arguments", {})}, ensure_ascii=False)
            action_valid = True
        elif getattr(step, "is_format_error", False):
            action_valid = False
        out.append(TrajectoryStep(
            step_id=index,
            raw_model_output=str(getattr(step, "thought", "") or ""),
            action=action_payload,
            observation=str(getattr(step, "observation", "") or ""),
            tool_name=tool_name,
            action_valid=action_valid,
        ))
    return out


def record_from_officeqa_result(row: dict[str, Any]) -> RawTrajectoryRecord:
    item = row.get("item") if isinstance(row.get("item"), dict) else {}
    return RawTrajectoryRecord(
        trajectory_id=str(row.get("trajectory_id") or f"officeqa_{row.get('id')}"),
        task_id=str(row.get("id") or item.get("uid") or item.get("id")),
        trial_index=int(row.get("trial_index", 0) or 0),
        instruction=str(item.get("question") or row.get("question") or ""),
        instruction_type=str(item.get("category") or row.get("category") or "officeqa"),
        answer_position="",
        spreadsheet_path="",
        output_path=str(row.get("output_path") or ""),
        final_response=str(row.get("final_response") or ""),
        success=bool(row.get("success")),
        verifier_score=float(row.get("score", 0.0) or 0.0),
        verifier_feedback=str(row.get("fail_reason") or ("correct" if row.get("success") else "incorrect")),
        steps=[TrajectoryStep(**step) if isinstance(step, dict) else step for step in row.get("steps", [])],
        runtime_metadata={"benchmark": "officeqa", "split": item.get("split", row.get("split", "")), "elapsed_seconds": row.get("elapsed_seconds"), "log_file": row.get("log_file", "")},
        service_metadata=dict(row.get("service_metadata") or {}),
        extra={"benchmark": "officeqa", "predicted_answer": row.get("predicted_answer", ""), "source_files": item.get("source_files", []), "source_docs": item.get("source_docs", []), "evaluator": row.get("evaluator", "")},
    )


def run_officeqa_item(
    item: OfficeQAItem,
    *,
    docs_dirs: list[str | Path],
    model: str,
    openai_base_url: str,
    openai_api_key: str,
    generation_config: dict[str, Any],
    max_turns: int,
    llm_timeout_seconds: float,
    llm_retry_wait_seconds: tuple[float, ...],
    reward_path: str | Path | None,
    reward_tolerance: float,
    evaluator: str = "skillopt",
    allow_fallback_evaluator: bool = False,
    output_dir: str | Path,
    log_dir: str | Path | None = None,
    skillbank_root: str | Path | None = None,
    skillbank_top_k: int = 0,
    selection_log: str | Path | None = None,
    use_oracle_context: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    docs_tools = OfficeQADocTools(docs_dirs, source_files=item.source_files)
    candidate_files = docs_tools.resolve_candidate_files()
    retrieved = select_retrieved_experience(item, skillbank_root=skillbank_root, top_k=skillbank_top_k, selection_log=selection_log)
    oracle_context = build_oracle_context(item, docs_dirs) if use_oracle_context else ""
    system_prompt = build_officeqa_system_prompt(retrieved_experience=retrieved)
    task_prompt = build_officeqa_task_prompt(item, candidate_files=candidate_files, docs_tools=docs_tools, oracle_context=oracle_context)
    client = OpenAIClient(
        model=model,
        api_key=openai_api_key,
        base_url=openai_base_url,
        cache_path=str(Path(output_dir) / "cache" / "officeqa_react.diskcache"),
        use_cache=True,
        generation_config=dict(generation_config),
        retry_times=tuple(float(value) for value in llm_retry_wait_seconds),
        timeout=float(llm_timeout_seconds),
    )
    agent = ReActAgent(client=client, tools=docs_tools.to_tools(), config=AgentConfig(max_turns=max_turns, verbose=verbose, system_instructions=system_prompt))
    result = agent.run(task_prompt)
    if is_infrastructure_agent_error(result.error):
        raise OfficeQAInfrastructureError(result.error)
    evaluation = evaluate_officeqa_prediction(
        prediction_text=result.final_answer,
        gold_answer=item.ground_truth,
        reward_path=reward_path,
        tolerance=reward_tolerance,
        evaluator=evaluator,
        allow_fallback=allow_fallback_evaluator,
        raise_official_errors=not allow_fallback_evaluator,
    )
    row: dict[str, Any] = {
        "id": item.uid,
        "trajectory_id": f"officeqa_{item.uid}",
        "trial_index": 0,
        "benchmark": "officeqa",
        "item": item.to_public_dict(),
        "question": item.question,
        "category": item.category,
        "candidate_files": [_relative_to_any(path, docs_tools.roots) for path in candidate_files],
        "final_response": result.final_answer,
        "predicted_answer": evaluation.predicted_answer,
        "score": evaluation.score,
        "f1": evaluation.f1,
        "hard": evaluation.hard,
        "success": bool(evaluation.hard),
        "fail_reason": result.error or evaluation.fail_reason,
        "evaluator": evaluation.evaluator,
        "agent_success": bool(result.success),
        "total_turns": result.total_turns,
        "elapsed_seconds": time.monotonic() - started,
        "steps": [step.to_dict() for step in _steps_to_record_steps(result.steps)],
        "service_metadata": {
            "model": model,
            "base_url": openai_base_url,
            "generation_config": generation_config,
            "evaluator": evaluator,
            "officeqa_context_variant": "skillopt_oracle_source_pages" if use_oracle_context else "local_docs_without_oracle_context",
            "use_oracle_context": bool(use_oracle_context),
        },
    }
    if log_dir:
        log_path = Path(log_dir) / f"officeqa_agent_{item.uid}.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(render_officeqa_log(row, system_prompt=system_prompt, task_prompt=task_prompt), encoding="utf-8")
        row["log_file"] = str(log_path)
    return row


def render_officeqa_log(row: dict[str, Any], *, system_prompt: str, task_prompt: str) -> str:
    lines = [f"# OfficeQA Trajectory {row.get('id')}", "", "## [0] SYSTEM", system_prompt, "", "## [1] USER", task_prompt, ""]
    for step in row.get("steps", []):
        lines.extend([
            f"## Step {step.get('step_id')}",
            "",
            "### Model Output",
            str(step.get("raw_model_output", "")),
            "",
            "### Action",
            str(step.get("action", "")),
            "",
            "### Observation",
            str(step.get("observation", "")),
            "",
        ])
    lines.extend([
        "## Final",
        "",
        f"Predicted: `{row.get('predicted_answer')}`",
        f"Success: `{row.get('success')}`",
        f"Score: `{row.get('score')}`",
        f"Fail reason: `{row.get('fail_reason')}`",
        "",
        str(row.get("final_response") or ""),
        "",
    ])
    return "\n".join(lines)

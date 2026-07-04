from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficeQAItem:
    uid: str
    question: str
    ground_truth: str
    task_type: str
    source_files: list[str]
    source_docs: list[str]
    split: str = ""


def load_officeqa_split(split_dir: str | Path, split: str, *, start: int = 0, end: int | None = None) -> list[OfficeQAItem]:
    """Load a materialized OfficeQA split from items.json/csv.

    The expected local layout is ``<split_dir>/<split>/items.json``.  A direct
    file path is also accepted for small smoke fixtures.
    """
    path = Path(split_dir)
    if path.is_dir() and (path / split / "items.json").is_file():
        path = path / split / "items.json"
    elif path.is_dir() and (path / split / "items.csv").is_file():
        path = path / split / "items.csv"
    elif path.is_dir() and (path / "items.json").is_file():
        path = path / "items.json"
    elif path.is_dir() and (path / "items.csv").is_file():
        path = path / "items.csv"
    if not path.is_file():
        raise FileNotFoundError(f"OfficeQA split file not found for split={split!r}: {path}")

    rows = _read_rows(path)
    items = [_normalize_row(row, default_split=split) for row in rows]
    stop = None if end is None else int(end)
    return items[int(start):stop]


def load_officeqa_splits(
    split_dir: str | Path,
    splits: list[str] | tuple[str, ...] | str,
    *,
    start: int = 0,
    end: int | None = None,
) -> list[OfficeQAItem]:
    """Load multiple materialized splits as one ordered stream."""
    if isinstance(splits, str):
        split_names = [part.strip() for part in splits.split(",") if part.strip()]
    else:
        split_names = [str(part).strip() for part in splits if str(part).strip()]
    if not split_names:
        raise ValueError("at least one OfficeQA split is required")
    items: list[OfficeQAItem] = []
    for split in split_names:
        items.extend(load_officeqa_split(split_dir, split))
    stop = None if end is None else int(end)
    return items[int(start):stop]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return [dict(row) for row in payload if isinstance(row, dict)]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported OfficeQA split file: {path}")


def _normalize_row(row: dict[str, Any], *, default_split: str) -> OfficeQAItem:
    uid = str(row.get("uid") or row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    ground_truth = str(row.get("ground_truth") or row.get("answer") or "").strip()
    task_type = str(row.get("task_type") or row.get("category") or row.get("difficulty") or "officeqa").strip() or "officeqa"
    if not uid or not question:
        raise ValueError(f"Invalid OfficeQA row: missing uid/question in {row}")
    return OfficeQAItem(
        uid=uid,
        question=question,
        ground_truth=ground_truth,
        task_type=task_type,
        source_files=_parse_list_field(row.get("source_files")),
        source_docs=_parse_list_field(row.get("source_docs")),
        split=str(row.get("split") or default_split).strip(),
    )


def _parse_list_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if "\n" in text:
        return [part.strip() for part in text.splitlines() if part.strip()]
    if "," in text and not text.lower().endswith(".txt"):
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]

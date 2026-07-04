from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_api_key(value: str, env_var: str = "") -> str:
    if env_var:
        return os.environ.get(env_var, value or "EMPTY")
    return value or "EMPTY"


def redact_args(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    for field in ("openai_api_key", "embedding_api_key"):
        key = str(data.get(field) or "")
        if key and key != "EMPTY":
            data[field] = "sha256:redacted"
    return data


def timed(report: dict[str, Any], stage: str, fn):
    start = time.time()
    result = fn()
    report.setdefault("stages", {})[stage] = {"seconds": round(time.time() - start, 3)}
    return result


def score_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    hard = sum(1 for row in results if int(row.get("hard", 0) or 0))
    mean_f1 = sum(float(row.get("soft", 0.0) or 0.0) for row in results) / count if count else 0.0
    return {
        "count": count,
        "hard": hard,
        "skillopt_em": hard / count if count else 0.0,
        "skillopt_f1": mean_f1,
        "accuracy": hard / count if count else 0.0,
        "mean_f1": mean_f1,
    }


def validate_generation_endpoint(args: argparse.Namespace) -> None:
    base_url = str(args.openai_base_url or "").rstrip("/")
    forbidden_local = {"http://127.0.0.1:18002/v1", "http://localhost:18002/v1"}
    if base_url in forbidden_local and os.environ.get("ALLOW_OFFICEQA_LOCAL_18002") != "1":
        raise ValueError(
            "OfficeQA runner refuses local A5000 18002 by default. "
            "Use the external AWQ endpoint/tunnel, or set ALLOW_OFFICEQA_LOCAL_18002=1 for an explicit debug run."
        )


def split_fingerprints(split_dir: str | Path, splits: list[str]) -> list[dict[str, str]]:
    return [
        {"split": split, "path": str(path), "sha256": file_sha256(path)}
        for split in splits
        for path in [resolve_split_file(split_dir, split)]
    ]


def resolve_split_file(split_dir: str | Path, split: str) -> Path:
    root = Path(split_dir).expanduser()
    candidates = [
        root / split / "items.json",
        root / split / "items.csv",
        root / "items.json",
        root / "items.csv",
        root,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (root / split / "items.json").resolve()


def file_sha256(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

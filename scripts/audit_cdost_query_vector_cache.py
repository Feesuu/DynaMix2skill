#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dynamix_trace2skill.clients import (
    validate_embedding_cache_manifest,
    write_embedding_cache_manifest,
)


def _load_selection_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"selection log is empty: {path}")
    return records


def audit_query_vector_cache(
    *,
    selection_log: Path,
    cache_path: Path,
    output_path: Path,
    reference_manifest: Path | None = None,
) -> dict[str, Any]:
    records = _load_selection_records(selection_log)
    requirements: list[dict[str, Any]] = []
    expected_vectors: dict[tuple[str, str], str] = {}
    expected_scoring_vectors: dict[tuple[str, str], str] = {}
    for record in records:
        query = str(record.get("query") or "")
        audit = dict(record.get("query_embedding_audit") or {})
        namespace = str(audit.get("namespace_sha256") or "")
        text_hashes = list(audit.get("text_sha256") or [])
        vector_hashes = list(audit.get("vector_sha256") or [])
        scoring_vector_hashes = list(
            audit.get("scoring_vector_sha256") or []
        )
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if (
            not query
            or not namespace
            or text_hashes != [query_sha256]
            or len(vector_hashes) != 1
            or len(scoring_vector_hashes) != 1
        ):
            raise ValueError(
                "selection record lacks one exact query embedding audit"
            )
        key = (namespace, query_sha256)
        vector_sha256 = str(vector_hashes[0])
        if key in expected_vectors and expected_vectors[key] != vector_sha256:
            raise ValueError(
                "one heldout query used multiple cached embedding vectors"
            )
        expected_vectors[key] = vector_sha256
        scoring_vector_sha256 = str(scoring_vector_hashes[0])
        if (
            key in expected_scoring_vectors
            and expected_scoring_vectors[key] != scoring_vector_sha256
        ):
            raise ValueError(
                "one heldout query used multiple scoring vectors"
            )
        expected_scoring_vectors[key] = scoring_vector_sha256
        requirements.append(
            {
                "namespace": namespace,
                "text": query,
                "purpose": "heldout_query",
                "item_id": str(record.get("instance_id") or ""),
            }
        )

    payload = write_embedding_cache_manifest(
        cache_path=cache_path,
        output_path=output_path,
        requirements=requirements,
    )
    for entry in payload["entries"]:
        key = (entry["namespace"], entry["text_sha256"])
        if entry["vector_sha256"] != expected_vectors[key]:
            raise RuntimeError(
                "query cache vector differs from the raw vector used by selection"
            )
        if (
            entry["normalized_vector_sha256"]
            != expected_scoring_vectors[key]
        ):
            raise RuntimeError(
                "query scoring vector differs from the normalized cache vector"
            )
    payload["selection_record_count"] = len(records)
    payload["selection_log_sha256"] = hashlib.sha256(
        selection_log.read_bytes()
    ).hexdigest()

    if reference_manifest is not None:
        reference = validate_embedding_cache_manifest(
            cache_path=cache_path,
            manifest_path=reference_manifest,
        )
        if payload["entries"] != reference["entries"]:
            raise ValueError(
                "dynamic heldout query vectors differ from the source static run"
            )
        payload["reference_manifest"] = str(reference_manifest.resolve())
        payload["reference_logical_sha256"] = reference["logical_sha256"]

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validate_embedding_cache_manifest(
        cache_path=cache_path,
        manifest_path=output_path,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-log", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    payload = audit_query_vector_cache(
        selection_log=args.selection_log,
        cache_path=args.cache_path,
        output_path=args.output_path,
        reference_manifest=args.reference_manifest,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

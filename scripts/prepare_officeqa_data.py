#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_benchmarks.officeqa import OFFICIAL_REWARD_SHA256, normalize_item  # noqa: E402


OFFICEQA_REPO_ID = "databricks/officeqa"
OFFICEQA_REVISION = "8ecbf18d3833daf4750a903d14963e4c4c1d4cd8"
OFFICIAL_REWARD_COMMIT = "86753108d69e149cc28abd346bb8c3ca1cbfc7cf"
OFFICIAL_REWARD_URL = f"https://raw.githubusercontent.com/databricks/officeqa/{OFFICIAL_REWARD_COMMIT}/reward.py"


def _download_hf_snapshot(output_root: Path, *, include_json_pages: bool, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it or materialize OfficeQA manually.") from exc
    allow_patterns = ["officeqa_full.csv", "treasury_bulletins_parsed/transformed/*.txt"]
    if include_json_pages:
        allow_patterns.append("treasury_bulletins_parsed/jsons/*.json")
    return Path(snapshot_download(
        repo_id=OFFICEQA_REPO_ID,
        repo_type="dataset",
        revision=OFFICEQA_REVISION,
        local_dir=str(output_root / "hf"),
        token=token,
        allow_patterns=allow_patterns,
    ))


def _load_csv_by_uid(csv_path: Path) -> dict[str, dict]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {str(row.get("uid") or row.get("id") or "").strip(): row for row in rows if str(row.get("uid") or row.get("id") or "").strip()}


def _load_split_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"SkillOpt OfficeQA split file must be a list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def materialize_splits(*, split_source: Path, full_csv: Path, output_split_dir: Path) -> dict:
    full_by_uid = _load_csv_by_uid(full_csv)
    summary = {}
    for split in ("train", "val", "test"):
        source_path = split_source / split / "items.json"
        source_rows = _load_split_items(source_path)
        items = []
        missing = []
        for row in source_rows:
            uid = str(row.get("uid") or row.get("id") or "").strip()
            full_row = full_by_uid.get(uid)
            if full_row is None:
                missing.append(uid)
                continue
            merged = {**full_row, **row, "split": split}
            items.append(normalize_item(merged, split=split).to_dict())
        if missing:
            raise RuntimeError(f"{len(missing)} OfficeQA split ids were not found in officeqa_full.csv for split={split}: {missing[:10]}")
        out_dir = output_split_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[split] = {"count": len(items), "path": str(out_dir / "items.json")}
    return summary


def download_reward_py(output_root: Path) -> Path:
    path = output_root / "reward.py"
    with urllib.request.urlopen(OFFICIAL_REWARD_URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != OFFICIAL_REWARD_SHA256:
        raise RuntimeError(
            "Downloaded OfficeQA reward.py hash mismatch: "
            f"expected {OFFICIAL_REWARD_SHA256}, got {digest}. "
            "Do not run evaluation until the official reward provenance is reviewed."
        )
    path.write_bytes(content)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize the gated OfficeQA dataset for DynaMix.")
    parser.add_argument("--output-root", default="/mnt/data/yaodong/officeqa")
    parser.add_argument("--skillopt-split-dir", default="/mnt/data/yaodong/codes/SkillOpt/data/officeqa_id_split")
    parser.add_argument("--include-json-pages", action=argparse.BooleanOptionalAction, default=True, help="Download parsed JSON pages so SkillOpt-style oracle page context can be rendered.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing files under output-root/hf instead of calling HuggingFace.")
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not args.skip_download and not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required to download gated databricks/officeqa. "
            "Request access on HuggingFace, export HF_TOKEN, or use --skip-download with existing files."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    hf_root = output_root / "hf" if args.skip_download else _download_hf_snapshot(output_root, include_json_pages=args.include_json_pages, token=hf_token)
    full_csv = next((path for path in [hf_root / "officeqa_full.csv", *hf_root.rglob("officeqa_full.csv")] if path.is_file()), None)
    if full_csv is None:
        raise FileNotFoundError(
            "officeqa_full.csv not found. OfficeQA is gated on HuggingFace; make sure your HF account has access "
            "and set HF_TOKEN, or use --skip-download with existing files."
        )
    split_summary = materialize_splits(
        split_source=Path(args.skillopt_split_dir).expanduser(),
        full_csv=full_csv,
        output_split_dir=output_root / "splits",
    )
    reward_path = download_reward_py(output_root)
    docs_root = hf_root / "treasury_bulletins_parsed"
    manifest = {
        "format": "dynamix_officeqa_materialized_v1",
        "source_repo": OFFICEQA_REPO_ID,
        "source_revision": OFFICEQA_REVISION,
        "reward_url": OFFICIAL_REWARD_URL,
        "reward_commit": OFFICIAL_REWARD_COMMIT,
        "reward_sha256": OFFICIAL_REWARD_SHA256,
        "full_csv": str(full_csv),
        "split_dir": str(output_root / "splits"),
        "docs_dir": str(docs_root),
        "reward_path": str(reward_path),
        "splits": split_summary,
        "include_json_pages": bool(args.include_json_pages),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

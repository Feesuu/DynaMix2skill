#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamix_trace2skill.skillbank import SkillBankSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DynaMix nodebank embedding index")
    parser.add_argument("--skillbank-root", required=True)
    parser.add_argument("--output", default=None, help="Index JSON path; default: <skillbank-root>/.dynamix_skillbank_index.json")
    parser.add_argument("--embedding-base-url", default="mock://deterministic")
    parser.add_argument("--embedding-model", default="Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", default="EMPTY")
    args = parser.parse_args()
    selector = SkillBankSelector(
        skillbank_root=args.skillbank_root,
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        cache_path=args.output,
    )
    docs, embeddings = selector._load_or_build_index()
    path = selector.cache_path
    print(json.dumps({"index_path": str(path), "node_count": len(docs), "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

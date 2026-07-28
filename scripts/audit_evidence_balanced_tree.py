#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from dynamix_core.balanced_metric_tree import BalancedMetricTreeState
from dynamix_core.certified_otd import ExperienceAtom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v4 structure on a frozen Experience Atom artifact."
    )
    parser.add_argument("--atoms", required=True)
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument("--dual-view-lambda", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    atom_path = Path(args.atoms).resolve()
    raw = atom_path.read_bytes()
    payload = json.loads(raw)
    atoms = [
        ExperienceAtom.from_dict(atom_payload)
        for atom_payload in payload.get("atoms", [])
    ]
    if not atoms:
        raise ValueError(f"no Experience Atoms found in {atom_path}")

    tree = BalancedMetricTreeState(
        max_entries=args.max_entries,
        dual_view_lambda=args.dual_view_lambda,
    )
    started = time.perf_counter()
    for atom in atoms:
        tree.insert(atom)
    insertion_seconds = time.perf_counter() - started

    started = time.perf_counter()
    audit = tree.structural_audit()
    validation_seconds = time.perf_counter() - started
    report = {
        "format": "evidence_balanced_structure_audit_v1",
        "source": str(atom_path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_atom_count": len(atoms),
        "max_entries": int(args.max_entries),
        "dual_view_lambda": float(args.dual_view_lambda),
        "insertion_seconds": insertion_seconds,
        "validation_seconds": validation_seconds,
        "audit": audit,
        "claim_boundary": (
            "structural audit only; no capsule semantics, retrieval, "
            "behavioral replay, or benchmark improvement claim"
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

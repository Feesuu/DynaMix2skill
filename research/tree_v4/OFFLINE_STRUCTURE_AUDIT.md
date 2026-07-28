# Offline Structure Audit

## Scope

This audit does not call an LLM, generate Skill Capsules, run heldout tasks, or
measure benchmark quality. It asks one falsifiable structural question:

> Does the v4 insertion rule avoid the 399-node, height-97 chain observed in
> CDOST v3 when both receive the same 200 Experience Atoms?

## Controlled Input

```text
runs/spreadsheet_cdost_static_runtimefixed_20260728_002122/
  dynamix_tree/experience_atoms.json
```

- Atom count: `200`
- SHA-256:
  `f566aa1980576e3d33727945255d2b011a0eedccd6fb40cb990dd768caa18519`
- Embedding model recorded by the source run: `Qwen3-Embedding-8B`
- Embedding dimension/control: source artifact reused without regeneration
- Dual-view weight: `lambda=0.5`

V4 parameters:

```text
max_entries B = 8
min non-root occupancy = ceil(B / 2) = 4
insertion order = source artifact order
```

## Result

| Structure | CDOST v3 | Evidence-Balanced v4 |
|---|---:|---:|
| Atoms | 200 | 200 |
| Total structural nodes | 399 | 41 |
| Internal nodes | 199 | 6 |
| Leaves | 200 atom leaves | 35 bucket leaves |
| Height | 97 | 2 |
| Leaf occupancy | singleton | 4–8 |
| All leaves same depth | no balanced-tree claim | yes |
| Every Atom placed once | yes | yes |
| Certified cover-radius upper bounds | n/a for this comparison | verified |
| Reported height bound | separation guarantee inapplicable | 4; observed 2 |

The v4 leaf occupancies were all within `[4, 8]`; internal occupancies were all
within `[4, 8]`; the tree contained 38 local split events. All 35 leaf radii
were exact. The six internal radii were conservative certified upper bounds,
as required for safe metric pruning.

On the current A5000 host, pure-Python insertion of the 200 real 4096-dimensional
Atoms took `8.45s`; one final full structural validation took `1.46s`. These
are engineering timings, not benchmark metrics. They confirm that validation
is now caller-scheduled rather than repeated after every insert.

## Reproduction

```bash
cd /mnt/data/yaodong/codes/DynaMix2skill_tree_v2
env PYTHONPATH=src \
  /home/yaodong/miniconda3/envs/stableskill-skillrl/bin/python \
  scripts/audit_evidence_balanced_tree.py \
  --atoms runs/spreadsheet_cdost_static_runtimefixed_20260728_002122/dynamix_tree/experience_atoms.json \
  --max-entries 8 \
  --dual-view-lambda 0.5
```

The executable source of truth is
`tests/test_evidence_balanced_skill_tree.py`, which additionally verifies
randomized insertion orders, exact nearest-search equivalence to brute force,
occupancy, equal leaf depth, certified radius upper bounds, unique placement,
and local mutation.

## Supported Conclusion

On this fixed Atom set, v4 eliminates the pathological binary chain and
produces a shallow balanced bucket hierarchy while preserving unique Atom
placement and certified metric cover-radius upper bounds.

## Unsupported Conclusions

This audit does not establish:

- better Skill Capsule semantics;
- better retrieval;
- improved SpreadsheetBench or OfficeQA scores;
- a semantic clustering optimum;
- a Skill-SP reproduction;
- behavioral validity without replay.

Those require the paired static/dynamic generation and heldout experiments
described in `experiments/tree_v4/README.md`.

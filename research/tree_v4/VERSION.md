# Version Record

- Date: 2026-07-28
- Branch: `research/evidence-balanced-skill-tree-v4`
- Implementation commit: `0d4a3e5`
- Remote: `https://github.com/Feesuu/DynaMix2skill.git`
- Method: Evidence-Balanced Skill Tree v4
- Frozen Atom audit source:
  `runs/spreadsheet_cdost_static_runtimefixed_20260728_002122/dynamix_tree/experience_atoms.json`
- Frozen Atom SHA256:
  `f566aa1980576e3d33727945255d2b011a0eedccd6fb40cb990dd768caa18519`

## Verification

- `305 passed`, one pre-existing NumPy deprecation warning.
- Python compilation passed with `PYTHONPYCACHEPREFIX` redirected to `/tmp`.
- Shell syntax passed for the common runner and control/static/dynamic wrappers.
- `git diff --check` passed.
- Independent spec/research and regression/fairness/security/Ponytail reviewers
  completed second-pass review with no remaining actionable findings.
- Offline structure audit: 200 atoms, 41 structural nodes, 35 leaves,
  6 internal nodes, height 2, occupancy 4-8.

## Result Status

No live LLM or LibreOffice benchmark result is attached to this version.
The first formal comparison must rerun the CDOST control from this same commit,
then run EBST static and dynamic under the matched control manifest.

# Version History

| Date | Version / Purpose | Commit SHA | Branch | Remote | Push Status | Changed Artifacts | Validation | Linked Idea/Contract/Run | Residual Risk |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-10 | research-loop bootstrap | uncommitted; base `06bb7803` | main | origin | not pushed | `research/skill_tree_refinement/` | manifest hashes match; workspace initialized | IDEA-001/002/003 | dirty worktree must be isolated before any checkpoint commit |
| 2026-07-14 | conditional L1+ hierarchy checkpoint | `c2444dfef4c9fe8ed3c9c738f233863dd6c79302` | main | origin | pushed 2026-07-14T01:00:39+08:00 | conditional parent generation, cross-level deduplication, reproducible PCA seed, full Soft/Hard evaluator, runtime HTTP controls, tests and research evidence | 149 targeted tests passed; `py_compile`, `bash -n`, `git diff --check`, and secret scan passed | Spreadsheet conditional run `spreadsheet_conditional_l1plus_static_reuse_l0_w16_20260713_194849` | diagnostic run underperformed the preserved 97/200 static checkpoint and did not isolate Conditional L1+ because L0 cards were regenerated; do not promote it as the headline method |

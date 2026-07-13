# Run Queue

| Priority | Run | Purpose | Status | Promotion Gate | Artifact |
|---:|---|---|---|---|---|
| 0 | current minsplit4 reference | freeze current 48.5% anchor and tree stats | complete | reference only | `runs/spreadsheet_awq_retrain200_recalc_tree_v1_20260709_191319/` |
| 1 | IDEA-001 minsplit8 build-only | test global split granularity | complete | passed safety and structural signal; one level shallower, no redundancy repair | `runs/research_minsplit8_build_only_20260710_121345/` |
| 2 | IDEA-001 minsplit8 heldout | measure selected global-granularity variant | complete: 88/200 LibreOffice | final result does not promote another depth sweep | `runs/research_minsplit8_heldout_20260710_191426/` |
| 3 | IDEA-001 minsplit10 build-only | conditional follow-up | cancelled by outcome gate | minsplit8 lost 9 tasks vs historical minsplit4 and did not repair redundancy | not run |
| 4 | IDEA-002 Phase A retrieval preflight | L2 routing with L1-only advice | complete | passed 200/200; exact top10; zero backfill; cache/index parity | `/tmp/phase_a_live_{dense,routed}_preflight.json` |
| 5 | IDEA-002 Phase A heldout | fresh matched dense control then routed role separation | waiting at before-dense drain; no arm request emitted | exact current 200-task protocol, 16 workers, zero cache/retry/timeout, stable arm drains, strict pair audit | `runs/research_phase_a_pair_workers16_20260711_143515/` |
| 6 | deterministic frozen-L1 K18 build/preflight | create non-generative router control without new summaries | complete: 239 L1, K18, 200/200 retrieval preflight | externally pinned builder/index/membership hashes and full reachability | `runs/research_frozen_l1_k18_router_20260711_161021/` |
| 7 | frozen-L1 K18 matched heldout | compare dense L1 with K18-routed L1 on the same bank | queued behind successful Phase A | 16 workers, exact prepared bank, full 200-task LibreOffice pair audit | tmux `research_frozen_l1_k18_pair_queue_20260711` |
| 8 | IDEA-003 community-conditioned utility | compare true K18 evidence with equal-budget matched non-community evidence | train-only contract frozen; implementation pending pair evidence | source-excluded paired utility gate; no heldout feedback | `contracts/IDEA-003-community-conditioned-utility.md` |
| 9 | current-endpoint vanilla fairness anchor | remove endpoint/worker/runtime confounders | queued before comparative claim | exact protocol match; execution failures audited | pending |

Config-only diagnostics are not gated by independent code review. The Phase-A
implementation completed independent review rounds. The K18 control fixed every
valid P1 from spec/regression/research review; the current verification suite is
130 tests plus real build/export and 200-query preflight evidence.

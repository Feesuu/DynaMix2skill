# Heartbeat

- Updated: 2026-07-11T21:15:55+08:00
- Goal backend status: `active`; endpoint occupancy is not treated as a goal
  blocker while offline implementation and audit work can continue.
- The old Phase-A idle watcher emitted no arm requests and is being retired;
  global vLLM occupancy was continuously nonzero because the endpoint is
  shared, not because local DynaMix work was active.
- Reviewed Phase-A protocol: dense and routed run concurrently with eight
  workers each, total client concurrency sixteen, cache disabled, no retry
  after ambiguous timeout, full 200-task LibreOffice-recalc denominator, and
  fail-closed Stage-06/request-window/load-sampling audit.
- New control identity:
  `deterministically_reprojected_frozen_l1_k18_router`. It freezes 239 L1
  items and both stored embedding views, then refits PCA/GMM with the current
  deterministic implementation. It is not a min-effective-only replay of the
  historical source tree, whose L1 selected K=64 with older PCA code.
- Formal prepared bank:
  `runs/research_frozen_l1_k18_router_20260711_161021/skills`.
  It contains 239 L1 nodes, 18 virtual routers, one singleton and two selected
  multi-parent items. No LLM generation or source-card re-embedding occurred.
- Formal hashes: index
  `1395142c97f8161333d1077228f826a74d1493ffd2312cf007ed0f28b65b4225`,
  build audit
  `2f1604f26608686b849d8c7390ed935c0668a8b5fb7ef1032733d543ea954222`,
  stable membership
  `1946e7f87eecd29adc92d0a342b7d54ee7f09d54f5e30fe36f05855c6d8893ad`.
- The first shared 8+8 launch failed safely in Stage 05 before any heldout LLM
  request. It exposed dynamic-batching drift in live query embeddings and the
  older preflight's use of train queries instead of heldout queries.
- Frozen heldout query cache:
  `runs/research_query_embedding_freeze_heldout200_20260711_192028/heldout_query_embeddings.json`,
  SHA `60d77ea4ba6a7b0f062bdf5f1bab919f06d747c48572cd31ff58dccdf1e41571`.
  It contains 200 unique `[200,400)` query hashes, 4096-dimensional vectors,
  and no answer position, label, output, or evaluator result.
- Updated full-node dense heldout selection SHA:
  `ec58751f918f10d94d308457b1c65a705e7645f5a459ed6b2c191649c4c363dc`.
  Updated prepared-L1 dense heldout selection SHA:
  `9fbb4fd428fac4c5e9f21b2e3f471704db6b45c556d0b0ae6c04b1a630c7f896`.
  Real dense and routed 200-query preflights both pass with all query vectors
  sourced from that cache; routed uses zero global backfill.
- The old K18 queue is being retired. The reviewed K18 launcher now requires a
  hash-bound, manually reviewed Phase-A continuation gate before it can start.
- The corrected frozen-query Phase-A attempt reached Stage 06 in both arms,
  but request-attempt telemetry exposed repeated Istio/Envoy failures at
  exactly 600 seconds: four dense and five routed failures were the stop-
  decision snapshot; in-flight cleanup finalized the ledgers at eight per arm.
  No evaluation score was read or produced. This attempt is invalid
  infrastructure history and is not mixed with later runs.
- The reviewed Envoy-header attempt also failed at the 600-second boundary:
  three dense and four routed timeouts were recorded before stop. The ingress
  accepts but ignores/overrides the per-request timeout header. No evaluation
  was produced, and the ineffective header code has been removed.
- The next protocol preserves `thinking=true`, 1200-second client timeout and
  no ambiguous retry, but fixes `max_tokens=16384` for every ReAct request in
  both arms. This is over 2.7x the largest successful uncapped completion
  observed (5901). Runtime, generation config, Stage 06 and usage audit must
  match exactly; cap hits are reported. Contract SHA is
  `44a944790b1ee38363601ee9d0a28a2e42855c701d4858c483c91d8e3ce7c454`.
- IDEA-003 authoritative train-only construction preflight is
  `preflight_final.json`; it passed 18/18 after strict posterior, source/hash,
  exact prompt-token and deterministic blinding repairs. Earlier preflight
  artifacts are rejected audit history. It accessed no heldout data and made
  no LLM generation or utility claim.
- Current capped-protocol implementation is awaiting targeted/full regression
  and independent review. Earlier 192-test/header review evidence applies only
  to the rejected header attempt.
- Next action: validate and review the capped protocol, then relaunch Phase A
  in a fresh run root. Evaluate the frozen decision gate before constructing a
  K18 continuation artifact.

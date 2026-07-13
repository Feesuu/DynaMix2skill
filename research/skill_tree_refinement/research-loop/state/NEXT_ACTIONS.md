# Next Actions

1. Minsplit8 is complete at LibreOffice 88/200. Do not run minsplit10.
2. Complete regression and independent review of the explicit 16384-token
   completion cap, then relaunch Phase A concurrently with eight dense and
   eight routed workers, total sixteen. The failed `20260711_185737` preflight
   emitted zero heldout requests; `20260711_194716` exposed the 600-second
   ingress timeout; `20260711_204946` proved the Envoy override is ignored.
   None produced evaluation, and all are audit history only.
3. Require both arms to use the exact 200-query frozen embedding cache SHA
   `60d77ea4ba6a7b0f062bdf5f1bab919f06d747c48572cd31ff58dccdf1e41571`.
   Stage 05 must audit dataset rows `[200,400)`, never train records. Also
   require `audit_phase_a_pair.py` to validate exact tree/index/settings,
   actual temperature 0.0, all 200 tasks, LibreOffice outputs, Stage-06 and
   request-window overlap, request-attempt integrity and shared-load coverage.
   Both arms must also record `max_tokens=16384` in generation config, runtime,
   Stage 06 and every usage row; any timeout still invalidates the pair.
4. Apply the frozen gate: continue if routed is non-inferior or reduces negative
   flips by at least 25% while losing no more than two total passes; otherwise
   stop the routing branch.
5. If and only if Phase A passes, write a hash-bound reviewed continuation gate
   and launch dense-L1 versus K18-routed-L1 concurrently over the same 239-node
   prepared bank. This is a deterministic re-projection control, not a
   min-effective-only replay of the historical K64 layer.
6. IDEA-003 construction preflight passed independent review with 18/18 strict
   pairs at `runs/IDEA-003/20260711_165829_preflight/preflight_final.json`.
   Treat earlier files in that directory as rejected audit history. After the
   frozen-L1 matched retrieval pair completes, implement only the registered 36
   candidate generations plus at most 108 source-excluded train rollouts. Do
   not relax matching/blinding rules, implement the old 239-card admission loop,
   or use heldout feedback.
7. Admit IDEA-003 to a full heldout run only if its train-side paired utility
   gate passes exactly as registered.
8. Rerun matched vanilla before any comparative or paper-facing method claim.

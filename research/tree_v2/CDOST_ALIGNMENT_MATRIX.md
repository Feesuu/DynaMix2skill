# CDOST Proposal-to-Code Alignment Matrix

Status: active audit; no formal benchmark; no commit or push

Repository:
`/mnt/data/yaodong/codes/DynaMix2skill_tree_v2`

Branch:
`research/tree-v2-constrained-gmm`

Base commit:
`06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a`

## Method Identity

| Requirement | Code | Test / evidence | Runtime artifact | Status |
|---|---|---|---|---|
| New method is explicit and old GMM tree-building core remains default | `pipeline.build_tree_from_records`; `pipeline.build_dynamic_tree_from_records` | Full reuse-contract suite; zero diff in `gmm_bic.py`, `tree_builder.py`, `update.py`, `data_structures.py`; shared rollout/evaluator corrections are separately audited | `analysis/runtime_config.json` | PASS |
| One atom per trajectory; no partial certified build | `ExperienceAtomAnalyst.extract_many` | `test_atom_extraction_fails_closed_after_leakage_retries` | `experience_atoms.json` | PASS |
| Unique, complete, ordered record-to-atom identity | `_require_unique_record_ids`; `_require_complete_atom_identity`; `_load_atom_cache` | `test_record_and_atom_identity_must_be_unique_and_order_complete`; cache order tests | records/atom fingerprints | PASS |
| Static and dynamic consume dataset order | `_build`; `_load_records_for_protocol` | `test_cdost_library_entry_requires_dataset_order_and_dynamic_cache`; prefix consistency test | `records_order_manifest.json`; `summary.arrival_order` | PASS |
| Static and dynamic bind the same planned prefix/arrival schedule | `cdost_control_contract.paired_dynamic_schedule` | schedule field and mutation-rejection tests | `analysis/cdost_control_manifest.json` | PASS |
| Dynamic arrivals are sequential, not batched | `_build` inserts and refreshes each arrival before the next; `active_dynamic_payload` | active-config and summary semantics tests; static/dynamic structure smoke | `summary.snapshot_interval`; `experiment_runtime_config.method_identity` | PASS |
| Controlled dynamic run uses frozen static atoms | `_build`; experiment CLI preflight | cache fingerprint/order/leakage/evidence-type tests | `summary.atom_source=frozen_cache` | PASS |
| Frozen atoms, vector sidecar, and control contract are bound to their source static build | `resolve_embedding_cache_path`; library and runner `validate_source_build_output` calls | atom, vector-manifest, and control-manifest source-marker mutation tests | source `04_build_tree.done`; dynamic build fingerprint | PASS |
| Static/dynamic runs use the same complete experiment contract | `cdost_control_contract`; `validate_matching_cdost_control_manifest` | protocol-drift, field-coverage, secret-redaction tests; full-runner smoke | `analysis/cdost_control_manifest.json` | PASS |
| Atom cache identity covers the complete semantic embedding/generation protocol | canonical payload helpers; atom protocol v6 hashes complete pipeline/client/OpenAI-fallback/tokenization/trace/structural modules and SDK identity | module-source/config/protocol mutation and resolved-key tests, including timeout/retry/concurrency/batch | atom protocol fingerprint; `analysis/run_manifest.json` | PASS |
| Cache storage location is audited but not treated as semantic identity | run manifest file identity; fingerprint intentionally omits `cache_path` | static/frozen-dynamic full-runner smoke with different paths | atom cache path and SHA-256 | PASS |
| Required embedding vectors have immutable logical identity | `write_embedding_cache_manifest`; `validate_embedding_cache_manifest`; runner build/heldout fingerprints | required-row mutation and unrelated-row-addition tests; static/dynamic smoke | `embedding_vector_cache_manifest.json` with raw/normalized SHA-256 | PASS |
| Frozen atom/node/query sidecars are bound to their producer stage | `validate_source_build_output` before accepting static atom/vector manifests or query-vector references | atom, node-vector and query-vector marker mutation tests | source `04_build_tree.done`; `06b_query_vector_audit.done` | PASS |
| CDOST does not invalidate unrelated legacy GMM embedding caches | cache namespace retains pre-CDOST payload; atom fingerprint separately includes execution controls | exact legacy namespace regression test | legacy cache namespace | PASS |
| Cache write semantics are policy-enforced at library entry | CDOST requires `first_write_wins`; legacy static/dynamic require `replace` | direct `build_tree_from_records` policy tests | resolved runtime config | PASS |

## Structural Algorithm

| Requirement | Code | Test / evidence | Runtime artifact | Status |
|---|---|---|---|---|
| Fixed normalized trigger/procedure views | `ExperienceAtom.__post_init__`; atom analyst view rendering | canonical-kernel and extraction tests | `experience_atoms.json` | PASS |
| Admitted atom state is deeply immutable | `ExperienceAtom.__post_init__`; recursively frozen metadata | nested-metadata mutation test | `experience_atoms.json` | PASS |
| One shifted-cosine kernel for pair and aggregate similarities | `_canonical_similarity_average`; pair/cross/within methods | brute-force sufficient-statistic and near-unit tests | `otd_tree_state.json` | PASS |
| Exact single-parent binary OTD insertion | `OtdTreeState.insert`; `_insert_at` | binary tree, average routing, deterministic tie tests | `otd_insertions.jsonl` | PASS |
| Online insertion path is `O(hd)` rather than full-tree validation | `insert` has no `validate()` call; path-only aggregate updates | `test_insert_does_not_run_full_tree_validation`; unaffected-subtree test | insertion decision trace | PASS |
| Structural parameters cannot be reassigned through public API | read-only `dual_view_lambda`; `tie_epsilon` | `test_structural_configuration_is_immutable_after_construction` | serialized tree config | PASS |
| Static full insertion equals prefix plus sequential arrivals | same `OtdTreeState.insert` entry point | `test_static_and_prefix_dynamic_builds_are_structurally_identical` | structural snapshots | PASS |
| Declared exactness is formula-exact binary64, not formal real arithmetic | diagnostics record arithmetic and comparison semantics; nonzero epsilon disables claim | diagnostics assertions; fake and real smoke artifacts | `comparison_arithmetic`; `exact_comparison_semantics` | PASS |

## Theory Boundary

| Requirement | Code | Test / evidence | Runtime artifact | Status |
|---|---|---|---|---|
| MW revenue uses fixed nonnegative similarities | `pair_similarity`; `moseley_wang_revenue` | nonnegative and brute-force tests | `otd_structural_diagnostics.json` | PASS |
| Nonzero tie epsilon disables exact theorem claim | `structural_diagnostics`; summary guarantee boundary | `test_nonzero_tie_epsilon_disables_theorem_claim` | `summary.guarantee_boundary` | PASS |
| Observed beta is computed from the exact Menon subtree-and-sibling antecedent | `audit_observed_beta_separation` | exact-assumption test plus non-qualifying-subtree counterexample | `otd_observed_beta_separation.json` | PASS |
| Active zero-RHS beta constraints are non-vacuous | separate antecedent, binding and zero-RHS counts | explicit 0/1 child-similarity counterexample | beta diagnostic v3 | PASS |
| Observed beta is not claimed for future arrivals or semantic truth | diagnostic scope/flags and plan wording | assertions on `future_arrivals_certified` | beta diagnostic | PASS |
| Offline beta/revenue diagnostics are not counted as online insertion cost | separate post-build audit functions | source/path test plus plan complexity statement | separate diagnostic files | PASS |

## Semantic and Retrieval Protocol

| Requirement | Code | Test / evidence | Runtime artifact | Status |
|---|---|---|---|---|
| Atom reusable text rejects IDs, paths, coordinates, explicit answer fields, labeled verifier expected/got values, and tagged/untagged answers without blanket-rejecting incidental record numbers | `_atom_leakage_reasons`; `_answer_literals`; `_contains_source_literal` | leakage/repair tests, verifier-value tests, one-character source-literal test, generalized numeric-rule regression, long tagged/untagged answer tests | atom debug records | PASS |
| Non-finite verifier evidence is not converted into reliability | `ExperienceAtomAnalyst._to_atom` | `test_atom_rejects_nonfinite_verifier_evidence` | extraction failure evidence | PASS |
| Parent update sees only two child skills and changed path | `LocalParentSkillAnalyst.refresh` | local parent contract and path stability tests | `otd_parent_updates.jsonl` | PASS |
| Parent provenance is checked, not asserted | descendant atom/source identities are recomputed before admission | missing-descendant provenance test | `structural_certificate.provenance_complete` | PASS |
| Parent generation error blocks nodebank and heldout | `_build`; `validate_tree_summary_for_heldout` | fail-closed summary gate test | `build_failure.json` | PASS |
| Retrieval returns an exact antichain or fails closed | additive discretized tree DP; exact complete-cost branch-and-bound with explicit state cap; no approximate fallback | exhaustive additive/random non-additive tests, tied-optimum cost test, state-limit test and reviewer state-collision counterexample | nodebank tree index; `export_policy.exact_search_max_states` | PASS |
| Retrieval budget matches the complete injected CDOST prompt | selector supplies exact full-render token callback including preamble/separators; callback cost is compared to the exact budget without `token_unit` rounding; final rendered-context check | exact 129/129 callback-budget test, non-additive render-cost tests and final budget test | selection token cost; `export_policy.retrieval_token_budget` | PASS |
| Retrieval representation is distinct from structural dual views | node manifest embeds `name + trigger + content`; selector applies shifted single-vector cosine | manifest and selector tests | `export_policy.relevance_transform` | PASS |
| Nodebank/query embedding is one complete vector, never silent chunk pooling | `SkillBankSelector._embed` token-counts the complete text and fails over limit | single-vector protocol and overflow tests | index `embedding_protocol.input_policy` | PASS |
| Batch embedding responses cannot be silently misassigned | `ordered_embedding_vectors` validates count, unique indices, and exact index set before reordering | reversed, missing, and duplicate-index tests | embedding request/usage log | PASS |
| Controlled static/dynamic comparison freezes exact atom/node vectors | CDOST-only first-write-wins SQLite cache; dynamic strict-match preflight; vector sidecar records raw-cache, cache-normalized, and exact artifact/index hashes | first-vector-wins, missing-vector, final-bit normalization, wrong-index-vector and matching-static-cache tests; full-runner vector/score equality | build vector manifest; runtime cache path | PASS |
| Heldout query vectors have immutable logical identity | query-vector audit stage; dynamic reference-manifest comparison | query cache mutation/reference-drift test; fake and real full-runner audits | `raw/heldout_query_embedding_cache_manifest.json`; `06b_query_vector_audit.done` | PASS |
| Heldout uses the exact prebuilt embedding protocol/cache | `SkillBankSelector.require_cache_match`; runner env/fingerprint | strict mismatch test | stage fingerprint and index | PASS |
| Rollout temperature is explicit and not provider-default dependent | `run_spreadsheetbench._build_generation_config`; nullable `ModelSettings.temperature` | CLI generation-config test and real emitted-request audit | generation config and usage JSONL | PASS |
| Nodebank identity is bound to the authoritative built tree | `_write_tree_artifacts`; `_authoritative_cdost_tree`; `_validate_cdost_tree_index` | coordinated manifest rewire, leaf-only omission, state digest mutation, topology/count/lineage tests | manifest `authoritative_tree`; state/structure SHA-256 | PASS |
| CDOST retrieval cannot silently become dense top-k or a partial tree | runner/selector expected-policy lock; `_validate_cdost_tree_index` | simultaneous identity/policy removal, root/count/lineage mismatch, partial-map, duplicate-edge, multi-parent, unreachable-node and omitted-leaf rejection tests | expected tree policy; complete tree and atom-leaf coverage | PASS |

## Reproducibility and Remaining Gates

| Requirement | Evidence | Status |
|---|---|---|
| Source, records, atom cache, Git state are auditable | Full-runner `analysis/run_manifest.json` records file/cache SHA-256, source checksums, Git HEAD/branch/dirty paths | PASS |
| Split and resume identities fail closed | ordered/disjoint split validation; stage file/directory SHA-256 markers; stage-owned output cleanup before rerun; invalid split, directory mutation, and stale-workbook/log cleanup tests | PASS |
| Evaluator identity is explicit, reproducible, resume-bound, and checked after execution | runner passes `local` or `official`; official fails closed when unavailable; comparator/LibreOffice identity is in the control contract and eval fingerprints; train/heldout output identity must equal preflight identity | PASS |
| Legacy GMM protocol remains isolated from CDOST cache controls | Default cache writes remain replace; trajectory/tree chunking remains separate; legacy nodebank declares `legacy_single_vector` and `chunking_active=false`; pre-CDOST index protocol is accepted only by exact legacy-field match; covered by cache-replacement, namespace, old-index reuse and protocol regression tests. | PASS |
| Final scoring vectors are artifact-bound | CDOST reload uses the validated persisted unit node vector without second normalization; query selection logs raw cache and final normalized scoring hashes; covered by non-idempotent floating-point normalization regressions plus node/query sidecars. | PASS |
| Build completion binds every required CDOST audit artifact | `build_outputs` and `04_build_tree.done` include state, structure, insertions, parent updates, diagnostics, runtime config/manifest, summary, nodebank and vector sidecar; checked by full-runner marker audit. | PASS |
| Runtime audit is method-specific | `runtime_dead_corner_findings` filters GMM-only checks for CDOST; covered by the policy-specific report test and experiment-stage report. | PASS |
| Targeted tests | CDOST/protocol/cache-identity tests passed; one unrelated NumPy warning | PASS |
| Full repository tests | 229 passed; one unrelated NumPy deprecation warning | PASS |
| Full experiment-runner smoke with mock/fake services | `/tmp/cdost_full_runner_smoke_v25`: static and frozen-cache dynamic runs completed build, paired-schedule control binding, authoritative state/structure binding, nodebank, heldout raw/final query-vector audit and post-run evaluator-identity checking; the third insertion exercised recurse, a changed path deeper than one internal node, and a nonempty/non-vacuous beta antecedent; contract, structure, atoms, vectors and selection matched; claim-boundary sidecar forbids benchmark interpretation | PASS |
| Real embedding + generation smoke | `/tmp/cdost_real_full_runner_smoke_v12`: real Qwen3.5-9B-AWQ/Qwen3-Embedding-8B services; static/dynamic paired-schedule control contract, atoms, five-node tree, 11-entry logical vector-cache manifest, nodebank, selection IDs and raw/final query-vector manifests matched. Six analyst/generation calls, build embeddings, and the static heldout query embedding were uncached real requests at `temperature=0.0`; the final heldout ReAct response itself hit the pre-existing global response cache. Comparator SHA and LibreOffice 24.2.7.2 matched. One-turn heldout produced no output workbook, so this validates wiring/protocol only. | PASS |
| Independent post-fix spec/theory/protocol/Ponytail review | Independent spec reviewer PASS; independent regression/protocol/Ponytail reviewer found three P2 issues, verified their fixes, reran 229 tests and fake v25 audit, then returned PASS with no remaining P0/P1/P2 | PASS |

## Claim Boundary

Passing this matrix establishes implementation/protocol alignment, not
SpreadsheetBench or OfficeQA performance. No smoke result is a benchmark
score. The beta audit is conditional on fixed admitted atoms, fixed
similarities, formula-exact binary64 OTD comparisons, the disclosed `1e-10`
shifted-cosine boundary clamp, and only the observed arrival sequence. It is
not an exact-real-arithmetic certificate. Insertion, validation, height,
ancestry, and antichain traversal are recursive. A 200-atom binary tree has at
most 399 structural nodes, but no claim is made for arbitrarily deep streams
or every adversarial shape. The local
full-runner smoke used deterministic fake generation/embedding services and
therefore validates orchestration only, not model semantics or benchmark
quality.

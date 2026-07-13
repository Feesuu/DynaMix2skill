# Repo Onboarding

## Repo Identity

| Field | Value |
|---|---|
| Concrete repo root | `/mnt/data/yaodong/codes/DynaMix2skill` |
| Repo instructions read | `/home/yaodong/codes/AGENTS.md`; no repo-local `AGENTS.md` found |
| Primary branch / commit | `main` / `06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a` |
| Config schema path | `src/dynamix_core/config.py`, `src/dynamix_trace2skill/pipeline.py` |
| Metric / evaluator path | `evaluate_with_official.py` |
| Dataset loader path | `run_spreadsheetbench.py`, pipeline record loader |
| Baseline command | existing fresh run script/config; exact command to be frozen in contract |
| Verifier path | LibreOffice recalc path in `evaluate_with_official.py` |
| Run scripts | `scripts/run_dynamix_trace2skill_experiment.py`, `scripts/build_dynamix_tree.py` |
| Cache / resume behavior | stage markers and source/config fingerprints; must use new run dirs |
| Protocol invariants | split, evaluator, model/endpoint, embedding, tools, thinking/decoding, max turns, top-k/query/index, retry/cache, denominator |

## Codebase-Memory Graph

| Field | Value |
|---|---|
| Available? | yes |
| Project ID / name | `mnt-data-yaodong-codes-DynaMix2skill` |
| Index status | ready; 1,595 nodes and 6,454 edges |
| Graph freshness | core tree paths current; dirty-file impact detected separately |
| Architecture summary | `dynamix_core` tree/GMM/export, `dynamix_trace2skill` pipeline/clients, ReAct runner/evaluator scripts |
| Entry points | `pipeline.main`, experiment runner `main`, SpreadsheetBench runner/evaluator |
| Hotspots | `ProjectedGmmTreeBuilder.cluster_layer`, budget refinement, GMM-BIC selector, pipeline build/export |
| Queries run | architecture; min-split search; build-tree and clustering symbol search; dirty impact from HEAD |
| Fallback | live Serena/`rg`/run artifacts override stale graph data |

## Serena Semantic Context

| Field | Value |
|---|---|
| Available? | yes |
| Activated project | `DynaMix2skill` at the exact `/mnt/data` repo |
| Memories read | `core`, `memory_maintenance` |
| Symbol overview | `tree_builder.py` and `pipeline.py` |
| Relevant symbols | `ProjectedGmmTreeBuilder.cluster_layer`, `build_tree_from_records`, `default_hierarchy_config` |
| Key live finding | `cluster_layer` stops when `n_items < min_split_size` and `compute_kmax` also uses this value; build passes config through unchanged |
| Edit/refactor targets | none until literature/contract gate passes |

## Live Artifact Overrides

- Source default currently uses `min_split_size=2`, but the fresh formal run's
  persisted `dynamix_config.json` explicitly uses 4. Formal experiment analysis
  therefore follows the run artifact, not the source default.
- Codebase-Memory detected seven dirty tracked files; none are silently treated
  as part of the research method.

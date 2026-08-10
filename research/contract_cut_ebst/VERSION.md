# Contract-Cut EBST v1 Checkpoint

- Initial implementation commit: `b43fd3f`
- Formal run code commit: `1ebcbd1`
- Branch: `research/contract-cut-ebst-v1`
- Remote: `https://github.com/Feesuu/DynaMix2skill.git`
- Base commit: `5aea7dc`
- Method contract: `research/contract_cut_ebst/ALGORITHM_CONTRACT.md`
- Frozen records SHA256: `f07519085195e8fa77d036e4cea5cc3654c57722d8cd9476fffc93da51075db1`

## Frozen Protocol

- SpreadsheetBench train `0:200`, heldout `200:400`.
- Batch size 8 and asynchronous LLM concurrency 8.
- `Qwen3.5-9B-AWQ`, 100,000-token context, no thinking, temperature 0,
  and no request-level `max_tokens` limit.
- `Qwen3-Embedding-8B`, 32,000-token context, embedding batch/concurrency 8.
- Final-cut skill folders only, dense top-1 retrieval, complete `SKILL.md`
  injection, and LibreOffice-recalculated heldout evaluation.

## Verification

- Contract-Cut tests: `31 passed`.
- Existing DynaMix regression tests: `159 passed` with one warning.
- `py_compile`, `bash -n`, `git diff --check`, and repository secret scan pass.
- Two independent post-fix reviewers returned `PASS` with no remaining P0/P1.
- An 8-record real-service smoke completed Atom extraction, EBST insertion,
  exact cut, skill export, and top-1 complete-skill retrieval. This smoke is a
  pipeline validation only, not benchmark evidence.

## Formal Run

- Run: `runs/contract_cut_ebst_v1_20260811_023336_r3_compiler_v2`.
- Accepted/excluded Atoms: `180/20`.
- Physical EBST: 38 nodes, 32 leaves, 6 internal nodes, height 2, all
  structural audits passing.
- Final cut: 153 active skills, consisting of 6 multi-Atom skills and 147
  singleton skills.
- Heldout: LibreOffice recalc `72/200 = 36.0%`; raw cached-value audit
  `67/200 = 33.5%`.
- Detailed report:
  `research/contract_cut_ebst/EXPERIMENT_REPORT_20260811.md`.

## Claim Boundary

This checkpoint establishes the implementation and structural contracts and
contains a complete 200-record build plus 200-task heldout run. The engineering
pipeline is complete, but the benchmark does not support a quality-improvement
claim: the final skill cut is dominated by singleton skills and scores below
the historical vanilla artifact. Preserve this branch as a reproducible
negative-result and diagnostic checkpoint rather than promoting it as the best
method.

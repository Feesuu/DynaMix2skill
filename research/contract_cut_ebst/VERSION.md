# Contract-Cut EBST v1 Checkpoint

- Implementation commit: `b43fd3f`
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

- Contract-Cut tests: `27 passed`.
- Existing DynaMix regression tests: `159 passed` with three pre-existing
  warnings.
- `py_compile`, `bash -n`, `git diff --check`, and repository secret scan pass.
- Two independent post-fix reviewers returned `PASS` with no remaining P0/P1.
- An 8-record real-service smoke completed Atom extraction, EBST insertion,
  exact cut, skill export, and top-1 complete-skill retrieval. This smoke is a
  pipeline validation only, not benchmark evidence.

## Claim Boundary

This checkpoint establishes the implementation and structural contracts. It
does not yet contain a full 200-record tree result or a 200-task heldout score.
Those results must be attached to a later run-specific checkpoint after tree
structure audit and LibreOffice evaluation.

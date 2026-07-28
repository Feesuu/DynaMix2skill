# Certified Dual-View Online Skill Tree: Coding Plan

Status: proposal/code alignment gate passed; local current-code and prior
real-service full-runner artifacts passed their declared checks; no benchmark
run

Target worktree:
`/mnt/data/yaodong/codes/DynaMix2skill_tree_v2`

Target branch:
`research/tree-v2-constrained-gmm`

Stable checkpoint:
`06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a`

## 1. Objective

Implement one skill-tree state that supports both static construction and
sequential dynamic insertion. The new method is an explicit experimental
policy named `certified_dual_view_otd`. Existing GMM-BIC tree-building
policies remain the default and their structural core is unchanged. Shared
rollout-temperature, cache-audit, and evaluator-identity guards are explicit
protocol corrections, not changes to the GMM clustering algorithm.

The implementation separates three objects:

1. immutable task/trajectory/verifier evidence;
2. one reusable Experience Atom extracted from each trajectory;
3. a single-parent binary skill tree maintained by exact Online Top-Down
   insertion.

The tree is structural. LLM-generated skill text is attached to nodes but is
not allowed to change routing or the structural theorem.

## 2. Research Contract

### 2.1 Experience Atom

Each trajectory produces exactly one atom with:

```text
trigger
scope
decision
invariant
verification
failure_mode
evidence_type
reliability
```

The source trajectory remains immutable provenance. The reusable fields must
not contain task IDs, local paths, answer coordinates, or incidental answer
values.

The extractor receives only the current trajectory and verifier evidence. It
does not read the current tree, future trajectories, heldout data, or gold
answers from another task.

The reusable fields pass a deterministic semantic guard before admission.
Task IDs, source paths, URLs, artifact filenames, spreadsheet coordinates,
answer positions, explicit answer fields, and labeled verifier
`expected`/`got` values are rejected. Incidental numbers elsewhere in the
source record are not blanket-rejected, so reusable numeric rules are not
discarded merely because a task happened to contain the same number. The
analyst gets at most three schema-constrained semantic attempts, with one
bounded JSON parse repair per attempt; if any trajectory still lacks one clean
atom, the certified build fails instead of silently building from a partial
dataset.

### 2.2 Dual-view embedding

Each atom has two fixed, normalized embeddings:

```text
trigger view   = trigger + scope
procedure view = decision + invariant + verification + failure_mode
```

For normalized vectors `g_i` and `p_i`, define:

```text
w(i,j) =
    lambda * (1 + dot(g_i, g_j)) / 2
  + (1-lambda) * (1 + dot(p_i, p_j)) / 2
```

`lambda` is in `[0,1]`. Similarities are therefore nonnegative, as required by
the Online Top-Down revenue guarantee.

Structural routing and heldout retrieval use different representations.
Routing uses the lambda-weighted trigger/procedure views above. Retrieval
embeds each node's `name + trigger + content` as one vector, takes ordinary
query-node cosine, and applies the same nonnegative
`(1 + cosine) / 2` transformation recorded in the nodebank manifest. This
makes the antichain objective inspectable; neither score is claimed to be a
calibrated probability or downstream utility.

Embeddings are immutable after atom admission. Mutable parent skill text is
never used as the structural routing vector.

### 2.3 Exact Online Top-Down insertion

For the current subtree `T` and a new atom `x`:

1. If `T` is a leaf, merge `T` and `x`.
2. If `avg_w(T) >= avg_w(T,x)`, merge `T` and `x`.
3. Otherwise recurse into the child with larger `avg_w(child,x)`.
4. Break exact ties by a stable subtree key.

Here:

```text
avg_w(T)   = sum of pairwise leaf similarities / choose(|T|, 2)
avg_w(T,x) = sum of similarities from leaves in T to x / |T|
```

The implementation stores binary64 sufficient-statistic sums and divides them
only at comparison time. It does not use a centroid approximation:

```text
leaf_count
sum_trigger
sum_procedure
sum_trigger_squared_norms
sum_procedure_squared_norms
within_similarity
```

This gives `O(hd)` arithmetic work per insertion for tree height `h` and
embedding dimension `d`. The current Python implementation uses recursive
traversal for insertion, validation, height calculation, ancestry checks, and
antichain retrieval. It therefore does not claim stack safety for arbitrarily
deep adversarial trees. A binary tree over 200 atoms has at most 399 structural
nodes, which is below the default Python recursion limit, but the suite does
not certify every adversarial 200-atom shape. Substantially larger streams
require iterative traversals before they can inherit the same engineering
claim.

### 2.4 Theoretical boundary

The structural claim is the theorem of Menon et al. (2019): for nonnegative
similarities and beta-well-separated arrivals, exact Online Top-Down obtains a
`beta/3` approximation for the Moseley-Wang revenue objective.

For each observed arrival `x` and every internal subtree `S` in the tree before
that arrival, let `A` and `B` be the children of `S`. The audited implication
from Menon et al. Assumption 1 is:

```text
avg_w(S,x) > avg_w(S)
and avg_w(A,x) <= avg_w(B,x)
implies avg_w(A) >= beta * avg_w(A,x)
```

Exact ties impose the condition in both orientations. A singleton child has
`avg_w(A)=0`. The implementation replays the observed arrival sequence and
reports the largest beta in `[0,1]` satisfying every observed constraint.
This is an offline `O(n^2 d)` diagnostic, separate from the `O(hd)` insertion
path. It certifies neither future arrivals nor semantic atom quality.
If no antecedent is active, the implication is recorded as
`satisfied_vacuously_on_observed_stream`, but the implementation does not
enable a theorem claim because no nonempty beta constraint was audited.

The claim applies only to:

- the single-parent binary tree;
- the fixed atom embeddings;
- the exact insertion rule.

Here, `exact` means that the implementation follows the declared OTD formulas
without a centroid surrogate, pruning heuristic, or nonzero engineering tie
epsilon, using the represented IEEE-754 binary64 values. It is not a claim of
formal exact real arithmetic or numerically certified interval arithmetic.
Artifacts therefore record
`comparison_arithmetic=ieee754_binary64` and
`exact_comparison_semantics=formula_exact_without_engineering_tie_epsilon`.
Shifted-cosine values are canonically clamped to `[0,1]` only when binary64
rounding places them within `1e-10` of a boundary; values farther outside the
range are rejected. Artifacts disclose this as
`similarity_boundary_tolerance=1e-10`. The theoretical statement is therefore
about the represented, canonically bounded binary64 similarities, not exact
real-valued cosine arithmetic.

It does not prove:

- semantic truth of LLM-extracted atoms;
- correctness of generated parent text;
- replay success;
- heldout task accuracy;
- beta-well-separation on the benchmark.

The implementation must include diagnostics that measure separation and
revenue but must not label the assumption as satisfied without evidence.
The audit reports antecedent count separately from positive-right-hand-side
constraints. An active antecedent with `avg_w(A,x)=0` is a non-vacuous,
trivially satisfied zero-RHS constraint; only a stream with no active
antecedent is labeled vacuous.

### 2.5 Static/dynamic unification

Static construction calls `insert(atom)` in dataset order for every atom.
Dynamic construction calls the same `insert(atom)` method for the initial
prefix and each later arrival.

Given identical atoms, insertion order, `lambda`, and tie policy, the two modes
must produce byte-equivalent structural snapshots. This is tested as prefix
consistency. LLM text equality is a separate conditional claim and requires
deterministic generation or a shared cache.

The certified dynamic protocol therefore preserves dataset order, rejects
shuffle configuration, and rejects snapshot resume until snapshots carry a
validated state/config/record fingerprint. The frozen atom cache records
ordered-record and atom-protocol SHA-256 fingerprints and rejects missing,
extra, duplicate, reordered, protocol-mismatched, non-single-trace, or
task-specific leaked atoms. The experiment runner also validates ordered,
non-overlapping split identities and hashes every declared stage output before
allowing resume.

### 2.6 Local skill evolution

Each leaf skill is the atom itself. After insertion, only internal nodes on the
changed root-to-leaf path are regenerated, bottom-up.

A parent analyst receives only:

- the two current child skill payloads;
- aggregate support/provenance counts;
- a fixed output schema.

It does not receive all descendant raw trajectories. The input size is bounded
by the two child skill payloads plus fixed prompt overhead.

Parent skills are retrievable only when a structural certificate passes:

- at least two descendant atoms;
- required semantic fields are present;
- the parent is not an exact duplicate of either child;
- every descendant leaf resolves to one distinct admitted source trajectory.

Optional source-task replay may strengthen this certificate. Without replay,
the artifact must say `validation_mode=structural_only`; no behavioral
non-regression claim is allowed.

### 2.7 Antichain retrieval

Retrieval must never inject both an ancestor and its descendant. Given
query-node relevance, node token costs, top-k, and a token budget, select an
antichain maximizing total relevance.

Additive per-node costs use polynomial tree dynamic programming with token
costs rounded up to a configured token unit. The production objective instead
uses the complete final `Retrieved Experience` prompt, including the fixed
preamble and separators, whose token cost is non-additive. It first solves the
cardinality-only relevance objective. A unique relevance optimum that fits may
return immediately; tied optima still enter complete-cost search because the
tie-break prefers the lower rendered prompt cost. Otherwise it runs an exact
branch-and-bound search over antichains using only sound cardinality, conflict,
and optimistic relevance bounds. No partial set is pruned merely because its
non-additive rendered cost is currently over budget. The callback path compares
the exact rendered token count to the exact budget; `retrieval_token_unit`
applies only to the additive DP path.

Complete-cost search is exponential in the worst case. A declared state limit
therefore stops the run without returning an approximate selection. Every
successful result is exact for the declared objective, but the method does not
claim guaranteed completion or a polynomial-time bound for arbitrary rendered
prompt costs. Exactness is with respect to the represented binary64 relevance
scores and integer tokenizer costs, not symbolic real arithmetic. The final
winner is rendered and checked again before injection. Optimality is for this
retrieval objective, not downstream usefulness.

## 3. Minimal Code Boundary

### New files

`src/dynamix_core/certified_otd.py`

- `ExperienceAtom`
- `OtdNode`
- `OtdTreeState`
- exact sufficient-statistic similarity
- exact insertion and validation
- serialization
- structural diagnostics
- additive-cost antichain dynamic programming
- exact non-additive full-prompt branch-and-bound

`src/dynamix_trace2skill/certified_otd_pipeline.py`

- atom JSON schema and extraction prompt
- atom extraction from `RawTrajectoryRecord`
- dual-view embedding
- local parent skill generation
- static and dynamic build functions
- OTD state and compatible nodebank artifacts

`tests/test_certified_otd.py`

- exact sufficient-statistic checks against brute force
- insertion rule and deterministic tie checks
- single-parent/tree invariants
- unaffected-subtree stability
- static/dynamic prefix consistency
- serialization round-trip
- antichain and budget optimality on exhaustive small trees
- nodebank lineage contract

### Existing files with minimal edits

`src/dynamix_trace2skill/pipeline.py`

- dispatch only when
  `hierarchy.tree_policy == "certified_dual_view_otd"`;
- old GMM static/dynamic tree-building code remains the default and unchanged.

`src/dynamix_trace2skill/skillbank.py`

- read optional `parent_node_id`, `child_node_ids`, and `token_cost`;
- use tree-antichain optimization only for manifests that explicitly request
  `tree_antichain_knapsack`;
- preserve dense top-k behavior for existing manifests.

No changes are planned for:

- `gmm_bic.py`;
- `tree_builder.py`;
- `update.py`;
- `data_structures.py`;
- existing GMM config validation;
- SpreadsheetBench or OfficeQA evaluator semantics.

## 4. Configuration

The new policy reads a small `hierarchy.otd` mapping:

```json
{
  "tree_policy": "certified_dual_view_otd",
  "otd": {
    "dual_view_lambda": 0.5,
    "tie_epsilon": 0.0,
    "atom_temperature": 0.0,
    "parent_temperature": 0.0,
    "atom_cache_path": null,
    "retrieval_token_budget": 24000,
    "retrieval_token_unit": 128,
    "retrieval_exact_search_max_states": 250000,
    "validation_mode": "structural_only"
  }
}
```

Inherited settings:

- generation model/base URL/API key/thinking/timeout/concurrency;
- embedding model/base URL/tokenizer/cache/max length/batch/concurrency;
- analyst max prompt and output tokens;
- analyst evidence bundle size and tokenizer;
- dynamic initial/arrival counts, order, snapshot cadence;
- nodebank top-k and benchmark evaluator.

Rollout temperature is explicit at the request boundary:
`run_spreadsheetbench.py` writes the CLI `--temperature` value into the client
generation config, while `ModelSettings` has no hidden non-null default. Runs
created before this correction are comparable only when their emitted usage
logs prove the same actual request temperature.

The shared `DynamicPipelineConfig.update_batch_size` transport field is used
by CDOST only as the snapshot interval. Static construction and the dynamic
prefix insert their atoms through the same structural `insert` operation, then
synthesize all internal parent skills once bottom-up. Each post-prefix dynamic
arrival is subsequently inserted immediately, updates the exact sufficient
statistics, and refreshes only its changed path bottom-up before the next
arrival. CDOST summaries and experiment reports therefore distinguish
`all_internal_nodes_bottom_up_after_build` from
`changed_path_bottom_up_per_atom`; neither is a batch-update algorithm.

No GMM parameter is silently reinterpreted as an OTD parameter. A non-zero
`tie_epsilon` is allowed only as an engineering variant and disables the exact
OTD theorem claim.

The serialized and fingerprinted CDOST hierarchy contains only
`tree_policy`, `otd`, and the inherited `summary_budget`. Inactive GMM
configuration is neither interpreted nor included in CDOST method identity.
Both members of a paired static/dynamic experiment serialize and bind the same
planned dynamic schedule: normalized initial/arrival counts, dataset arrival
order, disabled shuffle, snapshot cadence, snapshot embedding inclusion, and
disabled snapshot resume. The static build does not execute dynamic updates,
but it must preserve this comparison contract. This prevents a static source
run from being paired with a dynamic run that evaluates a different prefix or
arrival stream. Legacy propagation controls are excluded from CDOST method
identity.

For a static/dynamic consistency experiment, the static run first freezes
`experience_atoms.json`; the dynamic run sets `atom_cache_path` to that file.
This prevents independent stochastic analyst calls from changing the objects
being compared. Controlled dynamic runs fail preflight when the cache is
missing, reordered, incomplete, or fingerprint-incompatible.

This is a controlled structural experiment, not an online-latency claim: all
per-trajectory atoms are precomputed by the static source run, although the
dynamic tree does not consume an atom until that trajectory's dataset-order
arrival. Because atom extraction is trajectory-local and does not read the
tree or future trajectories, the cache controls semantic input and generation
noise. A paper claim about end-to-end online extraction cost still requires a
separate arrival-time extraction run and must not reuse this controlled result
as latency evidence.

The atom semantic fingerprint includes resolved generation and embedding
service identity, thinking mode, extra request body, timeout, retry schedule,
generation concurrency, embedding tokenization/truncation, batch size and
embedding concurrency. `atom_cache_path` is deliberately excluded because it
is only the storage location: static and dynamic controlled runs necessarily
use different paths for the same frozen payload. The path and file SHA-256 are
still recorded in the run manifest.

The broader generation/embedding implementation is part of atom protocol v6.
The fingerprint hashes the complete CDOST pipeline, client, tokenization,
trace-view, OpenAI fallback, and deterministic structural modules rather than
only selected classes. It also records the selected OpenAI SDK/fallback client
identity and version. Changes to request construction, JSON parsing,
truncation, cache use, trace rendering, or backend embedding therefore
invalidate frozen atoms. The
shared legacy GMM embedding-cache namespace intentionally retains its
pre-CDOST identity; CDOST-only execution controls are added to the atom
fingerprint without forcing unrelated legacy cache misses.

Controlled static/dynamic comparisons also share one content-addressed
embedding-vector cache. CDOST explicitly selects a `first_write_wins` policy:
the static run freezes the first successful vector for each exact text under a
namespace derived from the resolved embedding service and active embedding
protocol. The dynamic run must resolve the matching static config and cache
path; it fails before build or heldout if any required atom, node, or query text
is absent. `INSERT OR IGNORE` plus reread makes the first committed CDOST vector
canonical even if concurrent callers race. The shared legacy embedding client
retains its pre-CDOST `INSERT OR REPLACE` behavior and legacy cache namespace.
Selection logs record text/vector SHA-256 values and hit/miss counts so
equality is auditable rather than inferred from equal selected IDs.

The cache file path alone is not accepted as identity. Every completed CDOST
build writes `embedding_vector_cache_manifest.json`, which binds the exact
required atom and retrievable-node texts to their cache namespace, raw vector
SHA-256, cache-normalized vector SHA-256, exact artifact/index-vector SHA-256,
item IDs, and purposes. Because NumPy row-wise and one-vector normalization can
differ in the final binary64 bit, creation verifies numerical equivalence at
`64 * machine epsilon` and records the maximum absolute error; it does not
silently equate semantically different vectors. The runner hashes this logical
sidecar into build/heldout resume identity and revalidates it against the SQLite
cache before reuse. The build stage marker independently binds the node index
that contains the exact vectors used for scoring. Unrelated cache rows may be
added, but a missing or mutated required row fails closed.

The frozen atom file, source vector manifest, and source control manifest are
accepted only when the source static `04_build_tree.done` marker still binds
their exact digests. Static and dynamic runs additionally write and strictly
compare `analysis/cdost_control_manifest.json`. That contract binds the
dataset and records digests, train/heldout splits, active OTD configuration,
generation/embedding/analyst protocols, retrieval, rollout, evaluator runtime
identity, and relevant source digests. Run-local paths are excluded, while
API-key identities are stored only as fingerprints. A mismatch fails before
dynamic build.

Heldout query vectors receive a separate logical sidecar,
`raw/heldout_query_embedding_cache_manifest.json`. The audit stage binds each
exact query text and namespace to both the raw cached vector digest and the
L2-normalized vector digest actually used by the dot product.
Dynamic heldout requires an entry-for-entry match with the source static query
manifest and fails if a certified cache row changes.

Embedding batch responses are interpreted by the OpenAI `index` field, not by
transport order. The client requires exactly one unique row for every expected
index `0..N-1`, reorders rows by index, and rejects missing, duplicate, or
out-of-range responses before any vector is assigned to text.

Nodebank embedding is deliberately single-vector: each complete
`name + trigger + content` record, and each heldout query, must fit within
`min(max_model_len, max_input_tokens)`. Nodebank text is not silently chunked
or pooled because that would implement a different retrieval representation.
An over-limit node or query fails closed.

This statement applies to both CDOST and the current legacy nodebank selector:
the legacy index may record inherited chunk-size configuration for audit, but
its node/query embedding input policy is explicitly
`legacy_single_vector` with `chunking_active=false`. Chunked mean pooling is
used for trajectory/tree embedding stages that invoke the chunked embedding
pipeline, not for nodebank retrieval.

For CDOST, node vectors are normalized once when the index is written. A
reloaded index must contain finite unit vectors within `64 * machine epsilon`;
those persisted values are then used bit-for-bit for scoring without a second
normalization. The node-vector sidecar binds that exact index value. Legacy
nodebanks retain their previous reload normalization behavior, and indexes
written before the additional truthful legacy protocol metadata remain
read-compatible only when every original protocol field matches exactly.

Prompt admission and exported node token costs use the configured analyst or
embedding tokenizer. Regex fallback is allowed only when the inherited
configuration explicitly disables `tokenizer_required` and enables fallback;
otherwise tokenizer load failure stops the run.

## 5. Artifacts

Every CDOST tree build writes under its tree output directory:

```text
experience_atoms.json
otd_tree_state.json
otd_tree_structure.json
otd_insertions.jsonl
otd_parent_updates.jsonl
otd_structural_diagnostics.json
otd_observed_beta_separation.json
skills/node_bank_manifest.json
summary.json
```

The full experiment runner additionally writes under the scenario run
directory:

```text
analysis/cdost_control_manifest.json
raw/heldout_query_embedding_cache_manifest.json
stage_markers/06b_query_vector_audit.done
```

The query manifest and its stage marker exist only when heldout retrieval is
run; the tree-building library entry point does not fabricate heldout
artifacts.

If parent generation is incomplete, the build writes `build_failure.json` and
stops before nodebank indexing or heldout. Runtime identity is recorded under
`analysis/runtime_config.json` and `analysis/run_manifest.json`, including
records/cache hashes, source checksums, Git HEAD/branch/dirty state, and
redacted API-key identity.

The manifest keeps `dynamix_node_skill_bank_v1` compatibility and adds:

```text
tree_policy
root_node_id
parent_node_id
child_node_ids
descendant_atom_count
token_cost
fixed_prompt_overhead_tokens
validation_mode
structural_certificate
```

Existing readers ignore these optional fields. The updated selector uses them
only when the manifest requests antichain retrieval.

For a CDOST manifest, `root_node_id`, `node_count`, every node's
`parent_node_id` / `child_node_ids`, and the exact retrievable-node set must
match the authoritative `otd_tree_state.json` and
`otd_tree_structure.json` before index loading or query embedding. The
manifest stores relative paths plus SHA-256 digests for both artifacts; the
selector confines both paths to the same build directory, verifies both
digests and formats, and requires state/structure topology equality. This
prevents a self-consistent but rewired or leaf-only manifest from being
treated as the built tree. Atom metadata is recursively frozen at admission
so a caller cannot mutate provenance after fingerprinting.

Collect-stage resume markers bind the result JSON, output workbook directory,
and trajectory-log directory. A rerun clears those stage-owned paths before
launch, so a failed rerun cannot leave a stale workbook or trace that is later
evaluated or converted into an atom. LibreOffice recalculation copies live in
separate stage-owned directories and are cleared before an evaluation rerun;
the evaluation JSON remains the required stage result even when no output
workbook exists and therefore no recalc directory is created. The build marker
covers every OTD state, event, diagnostic, runtime identity, nodebank, and
summary artifact listed above.

The experiment runner passes an explicit SpreadsheetBench comparator backend;
it never relies on environment-dependent module discovery. Evaluation output
records the requested/resolved backend, comparator module/function, source
path and SHA-256, plus the resolved LibreOffice executable and version.
Requesting the official backend fails closed when `evaluation_official` is
unavailable. This provenance changes no workbook comparison formula, split, or
metric; it makes the implementation identity inspectable.
The same evaluator identity is included in the static/dynamic control contract
and train/heldout evaluation resume fingerprints, so a comparator or
LibreOffice change cannot silently reuse an old evaluation.

## 6. Verification

### Pure deterministic tests

1. Compare every aggregate `w(T)` and `w(T,x)` with brute-force pair sums.
2. Validate parent pointers, binary arity, reachability, unique leaves, counts,
   and aggregate vectors after every insertion.
3. Compare full static insertion with prefix build plus sequential arrivals.
4. Verify only insertion-path node aggregates/skills are changed.
5. Exhaustively enumerate small antichains and compare both additive DP and
   non-additive full-render branch-and-bound optima, including a state-collision
   counterexample that the old DP could not solve.
6. Verify old dense top-k nodebank selection is unchanged.
7. Reject task-specific numeric and textual answer leakage, including long
   tagged responses.
8. Reject reordered or protocol-mismatched frozen atom caches.
9. Reorder cached embedding rows by node ID when a compatible manifest changes
   document order.
10. Disable the theorem claim whenever nonzero `tie_epsilon` changes exact OTD
    comparisons.
11. Replay the observed arrival stream and verify the reported beta against
    the exact sibling condition.
12. Reject non-finite verifier evidence and duplicate/reordered trajectory or
    atom identities.
13. Require heldout to consume the exact prebuilt embedding index; a protocol
    mismatch must not silently rebuild it.
14. Revalidate frozen atoms for `single_trace` method identity and
    task-specific leakage when loading the cache.
15. Reject a CDOST nodebank that omits the antichain policy, authoritative
    state/structure identity, or a complete tree index; validate both
    authoritative SHA-256 digests, state/structure topology equality, binary
    arity, edge closure, unique parentage, reachability, coordinated manifest
    rewires, and exact retrievable-node coverage before any embedding request;
    never fall back to dense top-k.
16. Reject invalid or overlapping train/heldout ranges and stale resume
    markers whose declared output hashes no longer match.
17. Keep experiment diagnostics policy-aware so CDOST reports do not claim
    that GMM soft-membership or budget-refinement settings control this tree.
18. Require the runner and selector to agree on the CDOST method identity, so
    deleting both manifest identity and retrieval policy cannot activate
    legacy dense top-k.
19. Distinguish active zero-RHS beta constraints from a stream with no active
    theorem antecedent.
20. Compute parent provenance completeness from the admitted descendant atoms
    and distinct source trajectories instead of asserting it.
21. Reserve fixed retrieval-prompt overhead, derive every node cost from the
    exact injected CDOST block, and fail closed if the final rendered context
    exceeds the declared budget.
22. Bind every required atom/node vector to an immutable logical cache sidecar;
    detect mutation while allowing unrelated cache rows.
23. Reject missing/duplicate embedding response indices and restore reversed
    transport rows to request order.
24. Verify client-level rollout temperature is not overwritten by a per-call
    model default; inspect the real emitted request rather than only CLI config.
25. Reject a frozen atom cache when its source static build marker no longer
    binds the atom-file digest.
26. Strictly compare the complete static/dynamic control contract and ensure
    that no raw API key is persisted.
27. Bind every heldout query vector to a logical cache manifest; reject cache
    mutation or static/dynamic query-vector drift.
28. Use exact complete rendered prompt cost, including preamble and separators,
    in a separate branch-and-bound path rather than pretending the cost has
    additive tree-DP optimal substructure.
29. Preserve the legacy GMM trajectory/tree chunked-embedding pipeline while
    truthfully declaring legacy nodebank retrieval as single-vector, and avoid
    enabling the CDOST persistent-vector protocol for legacy policies.
30. Bind the normalized 120/80-style paired dynamic schedule in both static and
    dynamic control manifests and reject any prefix/order/snapshot drift.
31. Keep legacy embedding-cache replacement semantics while making immutable
    first-write behavior an explicit CDOST-only policy.
32. Bind frozen atom, node-vector, and heldout query-vector sidecars to the
    producing stage marker before accepting their cache contents.
33. Fail closed if exact non-additive antichain search exceeds its declared
    state limit; never return a heuristic selection under the exact policy.
34. Bind the exact node-index vector used for scoring as well as the
    cache-normalized vector, while tolerating only final-bit binary64
    normalization differences.
35. Require source static build markers to bind the frozen control manifest and
    node-vector manifest before a dynamic run accepts them.
36. Make SpreadsheetBench CLI temperature an explicit client configuration
    value rather than relying on a hidden model/provider default.
37. Reject CDOST skill output paths that are absolute or escape the tree output
    directory.

### Regression tests

Run existing DynaMix reuse-contract tests relevant to:

- default GMM policy;
- static build dispatch;
- dynamic build dispatch;
- nodebank export/discovery;
- retrieval prompt injection.

No benchmark score claim is made from these tests.

### Real-service smoke

Use:

- embedding: `http://10.26.1.184:18007/v1`,
  `Qwen3-Embedding-8B`, max length 32000;
- generation: `http://10.26.1.184:18080/v1` or `:18084/v1`,
  `Qwen3.5-9B-AWQ`, max model length 100000.

Run 1-5 records only:

1. extract one atom per trajectory;
2. obtain two real embeddings per atom;
3. insert atoms and generate changed parent skills;
4. export and reload the nodebank;
5. run one antichain retrieval query.

This smoke is an execution check only. It is not a benchmark score or evidence
that the beta-separation assumption will hold on the full dataset.

This is pipeline validation only, not benchmark evidence.

## 7. Independent Review Gate

After implementation and local verification, start independent reviewers:

1. spec/research reviewer: theorem boundary, exact OTD rule, static/dynamic
   identity, no unsupported claims;
2. regression/Ponytail reviewer: old GMM behavior, nodebank compatibility,
   unnecessary abstractions, tests and failure paths.

Reviewers receive the original request, this plan, changed-file list, git diff,
and test/smoke outputs. They do not edit files. Valid P0/P1 findings are fixed
before completion. Rejected findings require file/test evidence.

## 8. Non-goals

- no multi-parent structural edges;
- no online GMM or conformal router in this version;
- no automatic local split/merge heuristic outside exact OTD;
- no full benchmark run before the structural implementation passes review;
- no commit or push without explicit user approval;
- no modification of the stable worktree.

## 9. Implemented Verification Record

- Branch: `research/tree-v2-constrained-gmm`
- Base commit: `06bb7803c3e4262f6e2a6f5c0dc2859c6dae2c8a`
- Automated suite: `229 passed`, with one unrelated NumPy deprecation warning
  in the legacy KMeans-elbow test.
- A post-fix local full-runner smoke at
  `/tmp/cdost_full_runner_smoke_v25` completed static and frozen-cache dynamic
  build, nodebank export, heldout retrieval, and evaluator wiring. The two
  runs produced identical control contracts, structural snapshots, atom
  payloads, normalized nodebank manifests, node embedding vectors, exact
  full-render prompt costs, query-vector manifests, selected IDs, and selected
  scores for the same three atoms. The static query populated the
  shared vector cache with zero hits and one miss; dynamic retrieval recorded
  one hit and zero misses. Both evaluator outputs resolved the same local
  comparator source SHA-256 and LibreOffice version.
  `SMOKE_CLAIM_BOUNDARY.json` records that deterministic fake services were
  used and that evaluator scores are not benchmark evidence. Its experiment
  reports contain no GMM-only runtime findings.
  Unlike the earlier smoke, the third insertion necessarily takes a recursive
  branch, updates a path deeper than one internal node, and yields a nonempty,
  non-vacuous beta antecedent.
- A post-fix real-service full-runner smoke at
  `/tmp/cdost_real_full_runner_smoke_v12` used
  `Qwen3.5-9B-AWQ` at `:18084` and `Qwen3-Embedding-8B` at `:18007`.
  Static and frozen-cache dynamic runs produced the same three atoms, five-node
  tree with root `otd_00000002`, normalized nodebank, 11-entry logical vector
  cache manifest, control contract, selected node IDs, and query-vector
  manifest SHA-256. The static query embedding was a vector-cache miss and the
  dynamic query embedding was a hit. Real usage logs show six uncached
  analyst/generation calls at `temperature=0.0`, uncached build embeddings,
  and an uncached static heldout query embedding. The final heldout ReAct text
  response was not re-emitted in either arm because it hit the existing global
  response cache.
  Both eval files bind the same local comparator source SHA-256 and LibreOffice
  24.2.7.2. The one-turn heldout smoke did not produce an output workbook, so
  `SMOKE_CLAIM_BOUNDARY.json` explicitly treats it as wiring/protocol evidence,
  not workbook-correctness or benchmark evidence.
  After the final guard/evaluator-tolerance fixes, `:18080`, `:18084`, and
  `:18007` were unavailable, so no new real request is claimed for that final
  diff. The immutable v12 artifacts were re-audited successfully, while the
  changed code paths were re-executed by unit tests and fake v25 full-runner
  smoke.
- The earlier v11 real-service run exposed a final-bit difference between NumPy
  row-wise normalization and one-vector normalization. The manifest now binds
  the exact index vector and the canonical cache-normalized vector separately,
  verifies their numerical equivalence within `64 * machine epsilon`, and still
  rejects a genuinely different index vector. A targeted regression test and
  the repeated real-service smoke pass.
- A final regression/protocol review found three additional P2 gaps. The atom
  guard no longer rejects every number indiscriminately and instead rejects
  exact source-record literals; one-character literals are checked as complete
  tokens. Artifact/cache normalized-vector equivalence now compares the
  recorded maximum absolute error directly to `64 * machine epsilon`.
  Train and heldout evaluator outputs are checked after execution against the
  comparator and LibreOffice identity locked at preflight, closing the runtime
  identity TOCTOU gap. Boundary tests and fake v25 full-runner validation cover
  all three fixes.
- The preceding post-fix review found that node indexes and heldout query sidecars
  still bound pre-scoring vectors while the selector performed one additional
  normalization before the dot product. CDOST now uses the persisted unit node
  vector without reload renormalization, records the final normalized query
  scoring hash, and preserves pre-CDOST legacy index compatibility. Fake v25
  and real v12 smoke audits pass.
- Earlier independent reviews found cache-row misalignment, textual answer
  leakage, similarity inconsistencies, incomplete cache fingerprints, and
  unsupported runtime/theorem claims. The latest independent review additionally
  found an unbound shared vector cache, non-strict embedding response ordering,
  and insufficient recursive smoke coverage. These issues were reproduced and
  remediated with the logical cache sidecar, strict indexed response validation,
  and the v23 recursive smoke. Atom cache identity and run manifests use the
  same resolved generation/embedding protocol identity, including timeout,
  retries, concurrency, batching, truncation behavior, and API-key
  fingerprints. Frozen atoms are revalidated on load, and CDOST retrieval
  cannot silently fall back from the declared antichain objective. The latest
  remediation additionally binds the source build marker, complete
  static/dynamic control contract, evaluator identity, and query vectors;
  antichain optimization now uses the exact complete rendered prompt cost, and
  the CDOST vector protocol is isolated from legacy GMM retrieval. A subsequent
  independent review found that non-additive costs invalidated DP state
  compression, the paired dynamic schedule was absent from the shared control
  contract, and shared cache writes had changed legacy replacement semantics.
  Those findings are now addressed by exact branch-and-bound with exhaustive
  counterexamples, an explicit paired schedule contract, and policy-specific
  cache/retrieval protocols. A later review found false legacy chunk-pooling
  claims, exact-cost budget rounding, tied-optimum cost ordering, missing
  library-level cache-policy guards, incomplete source-marker binding, and
  potentially unbounded non-additive search. Those findings are now addressed
  by truthful single-vector protocol metadata, exact callback budgets,
  cost-aware tie handling, policy guards, stage-bound sidecars, and an explicit
  fail-closed exact-search state limit. The most recent review additionally
  found unbound source control/vector artifacts, incomplete node-index vector
  identity, implicit standalone rollout temperature, skill-output path
  traversal, and an unused dynamic parameter. These are now addressed by
  stage-marker validation, dual cache/artifact vector identities, explicit
  request temperature, confined CDOST output paths, and removal of the unused
  parameter. Recursive descent for adversarially deep trees remains a
  documented P2 engineering boundary rather than a theorem claim. The final
  independent spec review passed, and the independent regression/protocol/
  Ponytail reviewer passed after the three final P2 fixes above. No P0, P1,
  or P2 remains open.

These checks establish implementation and protocol correctness only. They do
not establish beta-well-separation, semantic truth, heldout improvement, or a
paper-level benchmark claim.

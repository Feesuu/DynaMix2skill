# Retrieval and Outcome Flip Diagnostic

> Diagnostic only: the vanilla comparison is not fully protocol-matched, so these flips locate hypotheses but do not establish causal gain.

## Outcome Alignment

- Shared tasks: 200
- DynaMix pass: 97/200
- Vanilla pass: 82/200
- positive_flip: 33
- negative_flip: 18
- both_pass: 64
- both_fail: 85

## Retrieval Structure

- Total selections: 2000
- Unique selected nodes: 218/346 (63.0%)
- Top-10 node concentration: 30.9%
- Level distribution: L1=1563 (78.1%), L2=326 (16.3%), L3=76 (3.8%), L4=28 (1.4%), L5=7 (0.4%)

## Retrieval by Outcome Category

- positive_flip (n=33): mean score=0.5203; L1=75.2%, L2=17.6%, L3=4.8%, L4=2.1%, L5=0.3%
- negative_flip (n=18): mean score=0.5081; L1=73.9%, L2=20.0%, L3=5.6%, L4=0.6%
- both_pass (n=64): mean score=0.5190; L1=77.0%, L2=17.0%, L3=3.6%, L4=2.0%, L5=0.3%
- both_fail (n=85): mean score=0.5199; L1=81.1%, L2=14.5%, L3=3.2%, L4=0.8%, L5=0.5%

## Most Frequently Retrieved Nodes

- E1_89d4bba92177 (L1, n=115): Verify formula logic with test data before finalizing
- E1_e3299e93ede4 (L1, n=103): Verify column mapping before writing formulas
- E1_c3be17ab5330 (L1, n=67): Dynamic Formula Construction with Row-Specific References
- E1_591eef15eb4a (L1, n=63): Verify Spreadsheet Cell Values Before Applying Logic
- E1_5cdc8450bd48 (L1, n=56): Mapping Values to Cell Coordinates for Targeted Formatting
- E1_f3031f1e37cf (L1, n=47): Correctly map spreadsheet column names to data fields
- E1_c24544670880 (L1, n=47): Correctly Identify Date Range Column Indices
- E1_05fb3c486c6a (L1, n=43): Extract and map structured text patterns across sheets
- E1_1ea00738f206 (L1, n=41): SUMIFS Formula Reference Consistency
- E1_5af1b91045b3 (L1, n=37): Handle empty cells in conditional counting
- E1_189bcbd42bc2 (L1, n=34): Build Dynamic Lookup from Source Sheet
- E2_d5b98dd66f4b (L2, n=34): Dynamic Cross-Sheet Lookup and Reference
- E1_b1e8488c2a3b (L1, n=31): Dynamic Date Range Calculation in SUMIFS
- E1_21372f6d2c36 (L1, n=29): Iterative Cell-by-Cell Filtering for Sheet Manipulation
- E1_dfad7e787baf (L1, n=28): Dynamic Range Calculation for Shifting Windows

## Interpretation Boundary

- Node-level selection frequency can reveal concentration, redundancy, and whether high levels are actually used.
- Positive/negative flip association is hypothesis-generating only because the vanilla run is not fully matched.
- A causal conclusion requires a matched no-skill rerun or a controlled retrieval-level intervention on the same runtime protocol.

## Minsplit8 Running Snapshot (Not A Final Score)

This snapshot was taken while the controlled `min_split_size=8` heldout run was
still active. It is evidence about retrieval structure only; it must not be
reported as a completed benchmark result.

- Queries that had started retrieval: 141/200.
- Selections observed: 1,410, exactly 10 per query.
- Level mix: L1 81.06%, L2 12.84%, L3 3.90%, L4 1.63%, L5 0.57%.
- Unique selected nodes: 191.
- Top-10-node concentration: 37.02%, compared with 30.95% in the completed
  minsplit4 tree. This is an interim comparison and may move before all 200
  queries finish.

The most frequently selected high-level node at this snapshot was
`E2_6fa6d973698d` (63 selections), named *Cross-sheet Value Lookup via
Coordinate Mapping*. Its source community `L1_C40` contains exactly two direct
L1 members:

- `E1_46c77c9f6eba`: *Extract structured data from spreadsheet for cross-sheet
  lookup*.
- `E1_7854a4c3f57d`: *Extract cell-to-value mapping for formula construction*.

The L2 content reproduces the first child's cross-sheet lookup procedure and
then appends the second child's value-to-coordinate mapping procedure. It does
not add a distinct executable decision rule. Because dense all-node retrieval
can return the L2 node together with either child, one underlying lesson can
consume multiple top-10 slots. This is direct artifact evidence for testing a
separation between hierarchy-level routing and L1 executable advice; it is not
yet evidence that the proposed replacement improves task accuracy.

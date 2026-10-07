# Rubik's Snake research handover

## Objective and stopping condition

Continue the existing research and Lean formalization; do not restart from the
original baseline or wait for another confirmation. The user has authorized
continued work. An exact formula for the number of snakes would be welcome, but
the required outcome is:

1. Prove `L_mu <= mu <= U_mu` with `U_mu - L_mu < 0.1`.
2. Prove uniform pointwise bounds
   `c * L_S^(n - 1) <= S_n <= C * U_S^(n - 1)` for every positive `n`,
   with explicit positive constants independent of `n`, and `U_S - L_S < 0.1`.
3. Audit the Lean proofs, then update and review the paper against those proofs.

Both gaps concern the numerical exponential **bases**, not their logarithms.
A numerical experiment, a conditional theorem with an unchecked certificate, or
an improvement that leaves either gap at least 0.1 does not finish the task.
The user specifically expects new mathematical methods: simply increasing the
old slab cutoffs or collision windows is unlikely to suffice.

The most recent turn requested documentation and this handover, rather than
another research run. All 957 declarations across the 57 project-owned Lean
sources now have docstrings. The documentation audit found no changes to
non-comment code tokens, and all 57 files passed Lean's parser. Their statements
and proof code were preserved; expensive native certificates were not rerun.

## Verified results

The five public summary theorems are in
[FinalResults.lean](lean/RubiksSnake/FinalResults.lean), namespace
`RubiksSnake.FinalResults`:

| Theorem | Statement |
| --- | --- |
| `mu_exists` | The root growth rate of valid rotation-word counts exists and is positive. |
| `mu_lower_bound` | `3.400034903 <= snakeGrowthConstant` |
| `mu_upper_bound` | `snakeGrowthConstant <= 3.661786723` |
| `Sn_lower_bound` | `3.400034903^(n - 1) <= S_n` |
| `Sn_upper_bound` | `S_n <= 3 * 3.661786723^(n - 1)` |

The last two statements hold for every positive wedge length. Both current
base gaps are **0.261751820**. Keep exactly these five summary theorems, with
exact decimal real numerals and proofs reusing the underlying results.

The master paper's lower endpoint, 3.400034903, has been formalized. Its upper
endpoint, 3.667542939, has also been formalized and improved to 3.661786723.
Do not revert to the earlier 3.193 lower bound.

With the present upper endpoint, a proved lower endpoint of **3.562** would
suffice: `3.661786723 - 3.562 = 0.099786723 < 0.1`.

The former `sorry` in [Loops.lean](lean/RubiksSnake/Loops.lean) is already
resolved. Validity is preserved by arbitrary integer cyclic shifts, including
negative shifts.

## Model and indexing

Read [Definitions.lean](lean/RubiksSnake/Definitions.lean) before extending a
construction.

- `Formula k` is a word of `k` quarter-turn choices and describes `k + 1` wedges.
  `countValidFormulas k` counts those valid formulas; `S n` is
  `countValidFormulas (n - 1)` for positive `n`.
- The initial directions are `ey` and `ex`. Successive directions are
  perpendicular cardinal vectors.
- A wedge at cube center `p`, with incoming direction `d` and outgoing direction
  `e`, has face pair `{-d, e}`. At different integer centers interiors cannot
  overlap. At the same center only complementary pairs can coexist.
- Boundary contact is allowed. The count retains chain order and distinguishes
  reflection and reversal. It does not require a collision-free motion from
  another configuration.
- The final rotation fixes the last wedge's orientation without adding a
  further center displacement. Watch this offset in every encoding.

[SnAsymptotic_MuExistence.lean](lean/RubiksSnake/SnAsymptotic_MuExistence.lean)
proves submultiplicativity and the useful consequence
`snakeGrowthConstant^k <= countValidFormulas k`. Thus any new lower bound on
`mu` immediately gives a same-base pointwise lower bound with prefactor one.
There is no proved ratio limit or asymptotic formula `S_n ~ C * mu^n`.

## Build and validation

Repository root on the current machine: `/home/dima/projects/math`.
Always run Python through the existing virtual environment.

```bash
cd /home/dima/projects/math
.venv/bin/python -m pytest rubiks-snake/rubiks_snake_test.py -q
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd \
  rubiks-snake/paper-draft/paper.tex

cd rubiks-snake/lean
lake build
lake build RubiksSnake.PrunedSlabEnumeration RubiksSnake.ExtendedSlabCertificate
```

Start by validating these commands and reading
[paper.tex](paper-draft/paper.tex), the certificate notebooks in
[asymptotic-analysis/](asymptotic-analysis/), and the relevant Lean modules.
The last completed full root build passed with 3,062 jobs; the Python suite
previously passed all 36 tests, and the six-page paper built successfully.
These are historical validations, not a claim that every experimental Lean
file currently builds.
During handover, the five theorem statements and their dependency audit were
checked again against the existing compiled results. This did not rerun the
native certificates or constitute a fresh full project build.

**Change into the Lean project before invoking Lake.** The pinned toolchain
and mathlib version are 4.33.1, while Elan outside the project has selected
4.34.1. A previous `lake -d ...` invocation from outside the project used the
wrong compiler and disturbed dependency artifacts. The pinned root build was
subsequently restored. Do not change toolchains, dependencies, or manifests
to work around this.

The default [RubiksSnake.lean](lean/RubiksSnake.lean) imports the verified
results but does not import the pruning or extended-slab experimental modules.
A successful root build therefore does not validate those modules.

Build one Lean dependency graph at a time. The machine is shared and has about
15 GiB RAM. Applying a 6 GiB virtual-memory limit to the entire Lake coordinator
previously caused thread-creation failure; use bounded individual jobs instead.
Comment changes can invalidate native certificates, so a fresh build may repeat
their expensive checks.

## Proof map

### Existing lower bound

[RubiksSnakeComputation.lean](lean/RubiksSnakeComputation.lean) is a small
separately precompiled library for restoring byte-array enumeration.
[SlabEnumeration.lean](lean/RubiksSnake/SlabEnumeration.lean) proves its counting
semantics. [SlabBoard.lean](lean/RubiksSnake/SlabBoard.lean),
[SlabGeometry.lean](lean/RubiksSnake/SlabGeometry.lean), and
[SlabBlockValidity.lean](lean/RubiksSnake/SlabBlockValidity.lean) connect the
representation to collision-free geometry.

[SlabCountCertificate.lean](lean/RubiksSnake/SlabCountCertificate.lean) checks
widths 0, 1, and 2 at internal-edge cutoffs 28, 20, and 18. That native check
took about 325 seconds. [SlabLowerBound.lean](lean/RubiksSnake/SlabLowerBound.lean)
derives the currently certified endpoint.

For a slab, `width = forward progress - 1`; internal-edge length `k` gives block
length `k + 1`. Irreducibility requires a backward crossing of every internal
cut. Mixed-width blocks are **not prefix-free**. Use the primitive-bridge unique
decoding proof in [BridgeWords.lean](lean/RubiksSnake/BridgeWords.lean).

[BridgeCode.lean](lean/RubiksSnake/BridgeCode.lean) now provides reusable code
concatenation, geometric validity, unique decoding, renewal counting, and
`BridgeCode.le_growthConstant`. Its coefficients may be **undercounts**.
`BridgeCode.ofSlabs` builds a code from a list of distinct widths and cutoffs;
`ofSlabs_blocks_length` expresses its block counts through the original counter.
[SlabLanguage.lean](lean/RubiksSnake/SlabLanguage.lean) reuses these proofs.

### Existing upper bound

[ForbiddenPrefixSixteenUpperBound.lean](lean/RubiksSnake/ForbiddenPrefixSixteenUpperBound.lean)
uses the prefixes of collision factors through length 16. Its certificate has
4,748,260 states and 14,133,721 labeled edges. Forty-eight integer potential
updates and a six-step terminal estimate give base 3.661786723 and prefactor 3.
The finite check took 36 minutes 19 seconds with 2.30 GiB peak RSS. Zero weights
at dead ends are intentional; making them artificially positive worsens the
prefactor. The length-14 certificate takes about four minutes.

Preserve the existing public bound APIs in
[SnAsymptotic_LowerBound.lean](lean/RubiksSnake/SnAsymptotic_LowerBound.lean),
[SnAsymptotic_UpperBound.lean](lean/RubiksSnake/SnAsymptotic_UpperBound.lean),
[SnAsymptotitc_MuLowerBound.lean](lean/RubiksSnake/SnAsymptotitc_MuLowerBound.lean),
and [SnAsymptotitc_MuUpperBound.lean](lean/RubiksSnake/SnAsymptotitc_MuUpperBound.lean).
The `SnAsymptotitc` spelling is the actual filename.

### Tools for a new construction

[CardinalDirections.lean](lean/RubiksSnake/CardinalDirections.lean) normalizes
arbitrary cardinal initial frames.
[BoundedComponents.lean](lean/RubiksSnake/BoundedComponents.lean) bounds families
of a fixed or bounded number of independent snake components. Polynomial
placement factors do not change their exponential rate.

[FiniteTransfer.lean](lean/RubiksSnake/FiniteTransfer.lean) proves:

- A nonzero nonnegative subeigenvector forces divergence of a finite transfer
  series, without an irreducibility assumption.
- A nonnegative renewal supersolution proves convergence.
- A rank drift inequality bounds construction steps by wedge count, even when
  some transitions add zero wedges, and excludes zero-cost cycles.
- A geometric comparison with bounded-component counts can turn transfer
  divergence into a lower bound on `mu`.

These theorems do not supply a geometric construction by themselves. Prove the
trace encoding, multiplicity bounds, and counting comparison for any application.

## Immediate unfinished work: the extended slab candidate

The candidate lower endpoint is **3.429771044**, not a proved numerical bound.
The coefficient rows are already in
[ExtendedSlabCertificate.lean](lean/RubiksSnake/ExtendedSlabCertificate.lean).
They use widths 0, 1, 2, and 3 with internal cutoffs **28, 23, 22, and 23**.
The exact degree-29 polynomial comparison passed Lean. Independent external
enumeration brackets its renewal root between 3.429771044 and 3.429771045.
Even if certified, it leaves gap **0.232015679**.

Completed supporting work:

- [RubiksSnakePrunedComputation.lean](lean/RubiksSnakePrunedComputation.lean)
  implements arbitrary pruning, a cached remaining-edge budget, and an optimized
  worker passing cursor fields separately.
- [PrunedSlabEnumeration.lean](lean/RubiksSnake/PrunedSlabEnumeration.lean)
  proves restoration and coefficient domination for any Boolean predicate.
  It also proves the optimized worker equals the reference traversal.
  The previously failing `budgetSearch_eq_prunedCountSearch` proof was fixed
  by simplifying the outer lets before `congr`; its targeted build passed.
- Medium native coefficient controls passed. Passing cursor fields separately
  reduced their time from 44.15 seconds to 35.88 seconds.
- `budgetCounts_getD_le_counts` handles every natural index by reading
  out-of-range entries as zero.

Remaining work:

1. Fix the resource problem in `coefficient_le` in
   [ExtendedSlabLowerBound.lean](lean/RubiksSnake/ExtendedSlabLowerBound.lean).
   A fresh bounded check during handover failed there with
   **`(kernel) excessive memory consumption detected`** at a 4096 MiB Lean limit.
   The earlier target build also did not complete. The root cause has not been
   isolated; investigate reduction of concrete code/count definitions and proof
   term size before increasing limits. This is not a failure in the verified
   current endpoint.
2. Actually check the three new native rows: widths/cutoffs `(1,23)`, `(2,22)`,
   `(3,23)`. Width zero reuses the existing certificate. `RowsVerified` is only
   a proposition, and `rows_verified_of_check` is an implication: no theorem
   currently proves `checkRows = true`. Its three searches run in parallel;
   benchmark resource use before launching it. Large Lean checks may take hours.
3. Finish and validate `growthConstant_lower_bound_of_rows`, then supply its
   verified hypothesis to obtain an unconditional theorem. Transfer the result
   pointwise using Fekete, wire imports, and update the five-result facade only
   after the complete proof chain passes.

Pruning correctness only needs an undercount, so do not introduce an unproved
completeness assumption or an unchecked compiler replacement. An earlier
external heuristic based only on the extreme missing cuts discarded valid
width-three blocks. The corrected budget uses the lowest missing cut and the
number of missing cuts at or above the cursor. Use artifacts labeled
`cutwise`, not the old extrema-pruned row, when comparing complete enumerations.

## Research already explored

| Direction | What is established or observed |
| --- | --- |
| Plane and occupied-interface blocks | Formal lower bases 3.1, 3.16, and 3.193. Preserve as reusable examples, not the best current result. |
| Multi-strand whole-column transfer | One restricted numerical model suggests about 3.448788965. Its global geometric/counting proof is unfinished. |
| Adaptive helical graphs | Exact supersolutions at 3.562 and complete SCC coverage exclude six saved finite graphs at that rate. This excludes neither the full family nor `mu`. |
| Seven-port column screen | Period 7, cap 7, frontier energy 7, column cutoff 12: 3,716 states and numerical weighted radius 0.865455055615 at 3.562. It did not reach the target. |
| Diagonal bridge slicing | Independent small-length checks passed. At cutoff 22, coordinate-height renewal root was about 3.420429041 and `x+y+z` height about 3.399911778; the tested diagonal slices did not help. |
| Chronological floor with retained occupancy | Completed numerical batch, not a pending agent. Best tested graph: `D=1`, retained-wedge cap `N=10`, 285,443 states, radius about 3.183473828. All its states/transitions passed an independent oracle; SCCs and radius were cross-checked. Larger cases hitting the one-million-state cap were truncated graphs, not complete-family exclusions. No new exact lower certificate or Lean geometric theorem resulted. |
| Literature/FCC comparison | No directly applicable theorem reaching 3.562 was found in the bounded search. A single fixed midpoint per two-step FCC edge has ceiling `sqrt(11)`; allowing both midpoint realizations is not excluded by that argument. |

For the chronological-floor idea, retain all occupied cubes at
`x >= runningMaxX - D`, forbid future visits below that floor, and delete only
the unreachable past. Cap retained **wedges**, counting a full complementary
cube twice. Canonicalization uses the four proper rotations about x. This is
different from dropping arbitrary old history, which would admit collisions.
The small-cap results are poor; they do not disprove the unrestricted idea.

Prefer a denser, rigorously encodable family or a genuinely stronger upper
comparison to blind cutoff escalation. Before expensive enumeration, check
the indexing and geometry independently at small sizes, analyze every relevant
strongly connected component, and distinguish a frozen finite subgraph from a
complete state space. Floating-point spectral radii only screen candidates.

## Local research archive

Prior work has persistent artifacts outside Git at:

```text
/home/dima/.copilot/session-state/096bfb93-6e0b-4499-b389-28b0e7ffcd93/
```

This archive may be unavailable on another machine. The checked proof sources
and proposed coefficient rows are in the repository; do not make formal results
depend on private session files. Useful paths relative to that archive:

- `checkpoints/005-bridge-codes-and-pruned-certif.md`: detailed earlier handover.
  Its statement that the optimized-worker proof is still failing is stale;
  see the corrected status above.
- `files/master_bounds_audit.lean`: exact five-theorem shapes, combined limit,
  and dependency audit.
- `files/pinned-root-revalidation.log`: last successful full root build.
- `files/handover-extended-slab-check.log`: reproduced experimental-module
  kernel-memory failure.
- `files/lean_documentation_audit.py`, `files/lean-documentation-baseline.json`,
  and `files/lean_documentation_parse.lean`: documentation coverage, comparison
  of non-comment code tokens, and parser-only validation. These checks do not
  replace a full Lean build.
- `files/pruned-profile-small.lean`, `files/pruned-profile-unpacked.log`,
  `files/pruned_slab_audit.lean`, and `files/pruned-cutwise-audit.log`:
  pruning controls and timings.
- `files/experiments/extended-slab-candidate.json`: combined candidate rows,
  exact root bracket, and unsuccessful target sign at 3.562.
- `files/experiments/diagonal_bridges.cpp` and `diagonal_bridges_check.py`:
  external enumerator and independent geometric/separator checker.
- `files/experiments/diagonal-bridges-a1-w1-n24-cutwise.json`,
  `diagonal-bridges-a1-w2-n23-cutwise.json`, and
  `diagonal-bridges-a1-w3-n24-cutwise.json`: corrected candidate data.
  These JSON rows use **wedge length**; Lean rows use **internal-edge length**.
- `files/experiments/chronological_floor_20261006_172155/`: completed floor
  experiment, including `batch_results.json`, `engine.cpp`, `oracle.py`, and
  `D1_N10_cap1000000/graph.full_audit.json`.

No old agent needs to be kept running. Resume from the saved sources and
results, not an assumed live computation.

## Finish the research, then the paper

Once both strict gap inequalities have been proved:

1. Build all relevant Lean targets, including new modules outside the default
   root. Check the exact five public theorem statements and dependencies.
   There must be no `sorry`, unproved custom axiom, external-count assumption,
   or unchecked computational substitution. Existing `native_decide` checks
   use Lean's native evaluation trust boundary; disclose that accurately.
2. Audit for indexing errors, duplicate encodings, unjustified concatenation,
   omitted reachable components, and invalid finite-to-asymptotic implications.
   Preserve the current results while simplifying proof structure.
3. Read [HUMANIZER_SKILL.md](agents/HUMANIZER_SKILL.md), then update
   [paper.tex](paper-draft/paper.tex) and [README.md](README.md). Explain the
   actual geometric and counting arguments. Prefer short human-readable proofs;
   if the result still depends on computations, say so rather than hiding them.
4. Keep the formalization link
   <https://github.com/fedimser/math/tree/master/rubiks-snake/lean>.
   Add any newly downloaded reference in [papers/](papers/) to
   [refs.bib](paper-draft/refs.bib), the draft's current bibliography.
5. Review the mathematics and exposition as a peer reviewer, address substantive
   findings, rebuild the PDF with TeX Live, rerun the Python tests, and verify
   that the paper, summary theorems, and measured gaps agree.

Do not commit or push. Preserve unrelated work in the dirty worktree. Use the
existing [rubiks_snake.py](rubiks_snake.py) model and tests; the exploratory
notebooks are not automatically reliable proofs. Record only significant
verified research milestones in [ai-log.txt](ai-log.txt), one timestamped line
per entry. Use subagents for bounded independent tasks when useful, with no
overlapping heavy Lean builds or unbounded memory use.

## Important note on research direction
Try different direction from what we currently use for the best prrof.
Current proof requires about 36 minutes to build (espectialy RubiksSnake.ForbiddenPrefixSixteenComputation).
You should find proofs that do not require this much computation and use some
genuinely different approach.
After finding such proof, as long as you can make sure that these expenseive 
computations are not needed anymore for best bound, you should remove these expensive computations.
The proof should be more mathematical and creative, rathern then relying on precomputing numbers of large snakes.
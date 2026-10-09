# Rubik's Snake research handover

## Objective and stopping condition

**Current request (2026-10-08): optimize upper-proof build time, keeping the
bounds fixed.** The user is satisfied with the current bounds and has stopped
numerical-bound research. The length-16 certificate now passes a fresh serial
check in **218.69 seconds**, versus the recorded **36m19s**, with maximum
child RSS **2.51 GiB**. Both of its theorems remain necessary. See
[the optimization record](#upper-certificate-optimization-2026-10-08).
The historical research goals and unfinished candidates below are archived,
not instructions to resume numerical searches.

**Previous focused request (completed 2026-10-08): achieved.** The transverse-cap
coefficient connection is proved, giving unconditional bounds
**`3.4505674 <= mu`** and **`3.4505674^(n-1) <= S_n`**.
A fresh serial rebuild of all **40 project-owned lower-proof modules** and
all three native libraries took **172.40 seconds**, with maximum child RSS
**3.17 GiB**. This meets the requested five-minute and 10 GB limits.
The public facade uses the new endpoint. See the completion record below.

**Previous focused request (2026-10-07 evening): achieved.** The user asked for
a lower bound strictly above 3.4 with a Lean check below two minutes.
That session's unconditional bound was **3.4003**. A fresh serial rebuild of its
entire 24-module project-owned dependency chain took **117.61 seconds**,
including both native computation libraries and the coefficient check.
See [the fast lower proof](#fast-fourfold-lower-proof-2026-10-07-evening).
The historical strict-gap objectives below were not reached.

The earlier research request sought the following outcomes:

1. Prove `L_mu <= mu <= U_mu` with `U_mu - L_mu < 0.1`.
2. Prove uniform pointwise bounds
   `c * L_S^(n - 1) <= S_n <= C * U_S^(n - 1)` for every positive `n`,
   with explicit positive constants independent of `n`, and `U_S - L_S < 0.1`.
3. Audit the Lean proofs, then update and review the paper against those proofs.

Both gaps concern the numerical exponential **bases**, not their logarithms.
The earlier request called for proved, unconditional improvements using new
mathematical methods, rather than larger slab cutoffs or collision windows.
These goals have been superseded by the user's decision to keep the bounds.

The earlier documentation audit covered all 957 declarations across the 57
project-owned Lean sources: all have docstrings, non-comment code tokens were
unchanged, and all files passed Lean's parser. A subsequent research session
revalidated the baseline builds but found no new certified bound. Its
unsuccessful directions are recorded below so they are not repeated. A later
2026-10-07 request resumed research, focusing on a lower bound of at least 3.5.
The [earlier research session](#new-methods-tested-2026-10-07) produced a new
exact-arithmetic candidate and a kernel-checked FCC obstruction, but did not
reach 3.5. The later fast-proof session improved the certified lower endpoint.

## Verified results

The five public summary theorems are in
[FinalResults.lean](lean/RubiksSnake/FinalResults.lean), namespace
`RubiksSnake.FinalResults`:

| Theorem | Statement |
| --- | --- |
| `mu_exists` | The root growth rate of valid rotation-word counts exists and is positive. |
| `mu_lower_bound` | `3.4505674 <= snakeGrowthConstant` |
| `mu_upper_bound` | `snakeGrowthConstant <= 3.661786723` |
| `Sn_lower_bound` | `3.4505674^(n - 1) <= S_n` |
| `Sn_upper_bound` | `S_n <= 3 * 3.661786723^(n - 1)` |

The last two statements hold for every positive wedge length. Both current
base gaps are **0.211219323**. Keep exactly these five summary theorems, with
exact decimal real numerals and proofs reusing the underlying results.

The original lower endpoint, 3.400034903, has been formalized and improved to
3.4003 and then 3.4505674. The original upper endpoint, 3.667542939, has also been formalized and
improved to 3.661786723.
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
The subsequent research session reran the baseline commands successfully:
the Python suite passed all 36 tests, `latexmk` reported the paper PDF up to
date, and the root Lean build passed with 3,062 jobs. Separate builds of
`RubiksSnake.PrunedSlabEnumeration` and `RubiksSnake.ExtendedSlabCertificate`
each passed with 3,013 jobs. These were incremental builds using existing
artifacts, not fresh executions of the expensive native certificates.
The earlier five-statement/dependency audit remains separate from these builds.
In particular, building `ExtendedSlabCertificate` does not discharge
`RowsVerified`, and `ExtendedSlabLowerBound` was not revalidated in this session.

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

The current default proof is
[CapLowerBound.lean](lean/RubiksSnake/CapLowerBound.lean). Its new geometric
and recurrence connection is described in the completion record below.
It retains the earlier [FastLowerBound.lean](lean/RubiksSnake/FastLowerBound.lean),
supported by
[BridgeSymmetry.lean](lean/RubiksSnake/BridgeSymmetry.lean),
[QuarterSlab.lean](lean/RubiksSnake/QuarterSlab.lean), and
[FastLowerCertificate.lean](lean/RubiksSnake/FastLowerCertificate.lean).
It does not import the older expensive coefficient certificate described next.

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
derives the historical endpoint. These older modules remain available for
experiments, but the default root no longer imports them.

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
The finite check now takes 218.69 seconds including its native library build,
with 2.51 GiB peak child RSS; the historical run took 36 minutes 19 seconds.
Zero weights
at dead ends are intentional; making them artificially positive worsens the
prefactor. The length-14 certificate module rebuilt in 44 seconds after the
same optimization.

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

## Archived unfinished work: the extended slab candidate

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

## Additional dead-end screens (recorded 2026-10-07)

**Do not restart the following searches or merely increase their cutoffs.**
They produced no new Lean theorem, unconditional certificate, or numerical
endpoint. The paper and five-result facade were left unchanged. A new
mathematical ingredient is needed before revisiting them; larger enumeration
alone repeats the approach rejected in the final research-direction section.

Here "dead end" describes the tested approach, not a proof that every member
of an infinite family fails. In particular, finite count ratios, floating-point
eigenvalues, particle estimates, and low rates of truncated subgraphs are
**not rigorous upper bounds** on the full family or on `mu`. The previous
session's closing summary overstated some of these as exclusions; use the
precise scopes below instead.

### Low-entropy and finite-count screens

| Direction | Parameters, observation, and reason not to repeat |
| --- | --- |
| Longer center-simple plane walks | The cutoff-30 renewal root was approximately `3.133235933906813`. This is a truncated, stricter subclass, not the infinite plane constant: the script forbids all repeated centers and counts only 64 length-seven blocks, whereas the wedge-aware plane code has 72. It did not improve the existing bound. |
| Direction-only ranks | Allow `d -> e` when `d` and `e` are perpendicular and `s*d_x + rank(e) - rank(d) >= 1`. Scanning rank ranges 1 through 6 and scales 1 through 12 gave maximum numerical radius about `2.455291193596027`, even before collision exclusions. The initial scan incorrectly fixed `rank(+x)=0` and returned zero; the corrected scan normalizes the minimum rank to zero. Do not reuse the initial conclusion. |
| Two spatial credits | Independent x/y credits with range 1 through 10 and scales 1 through 9 gave maximum numerical radius `2.9315639937465936`. The tested finite automata lose too much entropy before adding collision checks; larger ranges were not excluded. |
| Prudent turning walks | Every step is perpendicular to its predecessor and cannot point along a ray containing an earlier center. Python and C++ counts agreed through 11 steps. The successive ratio fell from `3.579510961` at step 11 to `3.558721117` at step 13 and `3.525827838` at step 18; the last count was `14331246192`. The early near-target ratios were misleading, but their decline is not a proved asymptotic ceiling. |
| Outward bounding-box walks | Requiring every move to extend a global box face was much too restrictive. Dynamic programming through 80 steps matched `2^(n+3)-16` for `n >= 2`, suggesting growth 2, not a target candidate. This was not formalized. |
| One exact count plus submultiplicativity | The stored value `S_28 = 2377810831870022` gives root `S_28^(1/27) = 3.7109751672587388`, worse than the existing upper bound. None of the stored finite roots closes the gap. These stored counts were not newly certified in Lean. |
| Fixed-length raw slab codes | Equal-length blocks avoid variable-length decoding ambiguity, but no useful new coefficient was obtained. Width zero through internal length 28 gave a best coefficient root of 2. The unpruned width-one/cutoff-23 Python run was stopped after 600 seconds without a row; the planned wider runs were not reached. This is a computational dead end for that implementation, not an exclusion of all fixed-length codes. |

### Bounded potential with a direction suffix

The proposed invariant uses a credit `0 <= r <= R` and update
`r' = max(0, r + 1 - s*delta_x)`, rejecting `r' > R`. Thus `s*x + r`
increases by at least one per step. A return to the same center can span at
most `R` steps, so retaining and checking the last `R` directions is enough
to forbid every center repetition. This targets a center-simple subclass;
it does not exploit complementary wedges.

The optimistic credit/direction automaton, which ignores collisions, first
crossed the target in the scan at `R=15, s=7`, with numerical radius
`3.577409667479997`. That headroom did not survive the attempted construction:

- Direct Python history expansion was unbounded and failed or was interrupted.
  **Do not rerun that prototype as written.**
- C++ expansions at ranges 15 and 16 hit a 30-million-state guard before
  closure. Quotienting by all eight signed transverse-axis symmetries still
  hit the guard at range 15. These are size failures, not complete graphs.
- A five-million-state breadth-first subset seeded with empty histories
  contained only transient states and was acyclic. Its zero rate says nothing
  about the full recurrent language; power iteration also lacked a zero-norm
  guard in that discarded prototype.
- Seeding full histories from the period-seven word
  `(+x,+y,+x,+y,+z,+x,+y)` and its credit cycle produced a capped subgraph with
  numerical rate about `1.3478528`. There was no all-SCC certificate or
  complete-family exclusion.
- A fixed population of 500,000 particles over 500 steps, with random seed
  `20261006`, sampled the range-15/scale-7 language at a rate near `3.342`.
  This is a stochastic screen, not a certified Perron value, lower bound,
  or upper bound.

The method neither supplied the target nor avoided large state spaces.
Do not repeat capped breadth-first discovery or treat the optimistic
`3.5774` as a collision-free rate.

### Corner-to-corner box blocks: reject the raw-mass false positive

This was the last attempted new separator construction. A center-simple path
starts at its coordinatewise minimum and ends at its coordinatewise maximum.
Translated consecutive boxes can meet only at their shared endpoint, offering
a simple geometric separation argument. The script counted paths with first
step `+x` through **18 center steps**; conversion to rotation/wedge indexing,
boundary frames, and unique decoding was not formalized.

Let `A_n` count those paths ending in direction `+x` and `B_n` those ending
in `+y`. The proposed switching-frame row totals were `C_n = 2*A_n + 4*B_n`.
Their raw sum at `q=3.562` reached `1.512475819987` through length 18, and
already exceeded one through length three. **This is not a lower certificate:**
variable-length blocks admit multiple decompositions of the same path.

The candidate scalar primitive extraction was
`I_n = C_n - sum_{1 <= k < n} I_k*C_(n-k)`. It returned
`I_1=2`, `I_2=I_3=I_4=0`, `I_5=8`, and `I_18=5969804`.
The resulting sum `sum_{n=1}^{18} I_n / 3.562^n` was only
**`0.606474456896`**, far below one. No tail bound or independent proof that
this scalar extraction describes the intended geometric code was supplied.
Do not present these as certified primitive counts or resurrect the raw sum
as a proof. The finite proposal failed; excluding the infinite box family
would require an additional argument.

### Loop erasure and the two-midpoint FCC obstruction

Ordinary chronological loop erasure does not preserve perpendicularity:
erasing the middle loop in
`(+x,+y,+z,-y,-z,+x)` leaves `(+x,+x)`. The literature search did not supply
a rigorous adaptation of the loop-erasure lower bounds to this constraint.

Pairing cubic steps gives FCC edges with two possible midpoints. The tempting
claim that every self-avoiding FCC path has a center-simple perpendicular
cubic lift is false. An explicit FCC path, all in `z=0`, is

```text
(0,0,0), (1,1,0), (2,0,0), (1,-1,0), (2,-2,0), (3,-1,0).
```

Every edge uses the x/y axes. Perpendicularity forces all five edges to use
the same first axis. In the x-first lift, midpoint `(1,0,0)` repeats; in the
y-first lift, midpoint `(2,-1,0)` repeats. This refutes the universal
**center-simple** lifting claim, not all valid snake lifts: complementary
wedges may share a center. It does not exclude every two-midpoint FCC method.

### Artifacts and resource lessons

Exploratory sources remain outside Git at:

```text
/home/dima/.copilot/session-state/6a3e2014-c618-4809-b2dc-607520fcdf78/files/
```

They include `credit_search.py`, `direction_rank_search.py`,
`multi_credit_search.py`, `plane_walk_search.py`, `potential_suffix.cpp`,
`prudent_count.py`, `prudent_count.cpp`, `box_outward_count.py`, and
`corner_blocks.cpp`. Scripts were overwritten during experimentation; in
particular, `potential_suffix.cpp` now contains the particle screen, not the
earlier graph builders. The numbers above are recorded session outputs, not a
checked certificate archive. No research process was left running, and the
temporary executables were removed.

Later runs used explicit per-process address-space caps of at most 8 GiB;
the smaller samplers and depth-first searches used 1 or 2 GiB caps. Early
Python graph prototypes had no safe state guard. Do not assume that any
archived script is safe to rerun or scale up: budget all retained states and
temporary copies, enforce limits, and keep aggregate usage below **10 GB**.
No new verified mathematical milestone from this session was added to
[ai-log.txt](ai-log.txt).

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

## Historical research completion checklist

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

## Historical note on research direction

The earlier request sought more geometric proofs and removal of expensive
certificates if superseded. The length-16 upper certificate remains necessary
for the best upper bound. It has now been accelerated without weakening or
removing any checked obligations; no further numerical search is requested.

## Memory usage
You are running on system with 16GBof memory
Whatever you do, don't exceed 10GB of memory usage.

## New methods tested 2026-10-07

**Historical outcome of the morning session:** the certified lower bound
remained 3.400034903. None of the
constructions below reached 3.5. In particular, the new number 3.437532869 is
an external exact-arithmetic **candidate**, not a new Lean theorem about `mu`.
The paper and five-result facade were deliberately left unchanged.

### Transverse-bridge resummation: a new candidate, not a longer slab search

Instead of enumerating entire long x-slabs, build them from short pieces
separated in y. A head enters by +x and exits by +y; middle pieces enter and
exit by +y; a tail enters by +y and exits by +x at the last x layer. Each
piece stays in `0 <= x <= w`, has y-width 0, 1, or 2, and crosses every
internal y cut backwards. Enumerating pieces of at most **18 wedges** gives
a finite library; arbitrarily many middle pieces are then summed by a
transfer matrix. This resums unbounded total lengths rather than raising
the old whole-slab enumeration cutoff.

The state is the current x coordinate and the mask of x cuts already crossed
backwards. Mask updates are unions. Requiring the full mask at the tail makes
the resulting x block irreducible. There are `(w + 1) * 2^w` states at width
`w`, hence **769 states** over widths 0 through 6. The equations are
`F = T + M F`, with head mass `H F`. Their diagonal mask blocks have size
only `w + 1`.

The intended geometric and decoding argument is:

- Consecutive pieces occupy disjoint y layers; their connecting +y edge
  checks the boundary orientations without sharing an occupied cube.
- Cutting at every y separator recovers the head, middle pieces, and tail.
  Their internal backward-crossing condition prevents alternative cuts.
- Reflecting y gives a second, disjoint family, since every block has at
  least one separator and hence strictly positive or negative y progress.
- Retain only total lengths above 29, 21, and 19 at x widths 0, 1, and 2,
  respectively, before adjoining the existing certified coefficient rows.
  This makes the old and new sets disjoint by length. No raw-mass addition
  or unproved subtraction of overlapping languages is used.
- Finally use the existing primitive-x-bridge decoding principle to
  concatenate blocks of different x widths.

The short-piece counter and the transfer counting/geometry argument have
**not** been formalized in Lean. The external checks establish:

| Check | Result |
| --- | --- |
| Independent coordinate/face-set oracle, x widths 0 through 6, y widths 0 through 2, through 8 wedges | All 1,236 nonzero piece coefficients agree, including the zero-entry support check. |
| Complete short-grammar word checks | 151, 1,287, and 966 distinct words at x widths 0, 1, and 2; piece cutoff 6 and total cutoffs 10, 10, and 14. Geometry, x irreducibility, coefficient counts, and exact y-separator decoding all pass. |
| Combined polynomial through total length 96 | Exact rational root bracket **(3.437532869, 3.437532870)**. |
| Infinite grammar at `q = 7/2` | Integer supersolutions check **all 769 states** and give combined renewal mass **< 891/1000 < 1**. |

The last check is stronger than observing that the degree-96 polynomial
fails at 3.5: it covers arbitrarily many middle pieces from these saved
libraries. It is an exact algebraic exclusion of **this supplied grammar**,
not an upper bound on the full x-slab families or on `mu`. The floating
combined mass is about 0.890313131142 at 3.5. Increasing only the number of
transfer iterations or the final polynomial degree cannot make this grammar
reach the target.

The seven width-0-through-6 short-piece enumerations ran serially in about
120.55 seconds total, using about 4.3 MiB peak RSS. The degree-96 integer
coefficient/root computation took about two seconds and 34 MiB. This is a
potentially reusable counting method, but a richer construction is needed
before investing in its Lean geometric formalization.

### Two-midpoint FCC lifting: the complementary-wedge loophole is closed

The earlier five-edge planar example has **two valid wedge-aware lifts**.
It remains only a counterexample to center-simple lifting.

A different, simple eight-edge FCC path has **no valid lift**, even with
complementary wedges allowed:

```text
(0,0,0), (-1,0,-1), (-2,0,0), (-1,-1,0), (-2,-1,-1),
(-2,-2,0), (-3,-2,1), (-3,-1,0), (-2,-1,1).
```

An exact 2-SAT model uses one bit for each of the two midpoint choices.
Adjacent-edge clauses forbid nonperpendicular joins; nonadjacent-edge
clauses forbid incompatible wedges at a shared midpoint. A separate
coordinate/face-set oracle checked all 256 assignments to this path and
found zero lifts. Small random-path controls also checked SAT against
exhaustive geometry; their sampling is not used as proof.

There is a short contradiction behind the finite check. Number edges
0 through 7, with midpoint bit 0 choosing the smaller first-axis index.
Perpendicularity forces `b1 = b0`. If `b2 = 0`, a collision forces `b0 = 1`
while a turn constraint forces `b1 = 0`, so `b2 = 1`. Two more collision
constraints then force `b4 = 0` and `b7 = 1`. The intervening turn
constraints force `b5 = 0`, then `b6 = 0`, then `b7 = 0`, a contradiction.

[FCCLiftObstruction.lean](lean/RubiksSnake/FCCLiftObstruction.lean) now
formalizes this example, including distinct backbone vertices, FCC edge
geometry, exhaustiveness and realizability of both midpoints, and
`no_valid_lift`. It tests only internal wedges, so changing endpoint faces
cannot repair the obstruction. All five theorems use **kernel `decide`**,
not `native_decide`; the dependency audit reports only `propext`,
`Classical.choice`, and `Quot.sound`. The targeted module build took about
7.43 seconds, with 2.33 GiB peak RSS.

This rules out lifting *every* FCC self-avoiding path. It does not rule out
restricted FCC families, weighted lifting estimates, or an FCC-based lower
bound with an additional geometric ingredient. The module is not imported
by the default root; validate it explicitly with
`lake build RubiksSnake.FCCLiftObstruction`.

### Self-avoiding planar projections: an analytic ceiling

Another proposed family deletes z steps and requires the resulting planar
path to be self-avoiding. There can be at most one z step between planar
steps. At wedge/step fugacity `t`, a projected straight continuation has
weight `s = 2*t^2`, and either projected left or right turn has weight
`a = t + 2*t^2`. Endpoint choices change only a fixed prefactor.

Even the relaxation that forbids only three consecutive left turns or
three consecutive right turns (the four-edge projected square) is too
small. Its three-state weighted matrix is

```text
M = [[s, 2a, 0], [s, a, a], [s, a, 0]].
```

The threshold satisfies
`q^6 - q^5 - 5*q^4 - 6*q^3 - 10*q^2 - 8*q - 8 = 0`;
an exact bracket is **(3.362208006, 3.362208007)**. In particular, at
`q = 17/5`, the positive vector
`(1, 21200/26281, 14450/26281)` satisfies
`v - M*v = (557159/7595209, 0, 0)`. Every row reaches the first state, so
the weighted series converges. This gives a small analytic reason not to
enumerate more projected self-avoiding paths: this entire subclass cannot
improve even the current lower endpoint. This argument is not Lean-formalized.

### Coarse-box first-exit codes: inexpensive optimistic screening

Partition space into fixed cubic boxes. A local block starts at any incoming
face port and first exits through a positive-coordinate face; negative
boundary exits are rejected, but backward steps inside the box are allowed.
Box coordinates then increase, so boxes cannot be revisited and first-exit
decoding is unambiguous. Unlike the earlier corner-to-corner proposal,
individual block endpoints need not be coordinatewise extrema.

There is also a simple whole-family optimistic ceiling for small boxes.
For side `b`, the one-axis residue walk has positive steps with wraparound
and negative steps killed at residue zero. Its positive eigenvector is
`u_i = sin((i+1)*pi/(b+2))`, with eigenvalue `2*cos(pi/(b+2))`.
The product of three such vectors, independent of the incoming axis,
gives the perpendicular three-dimensional walk eigenvalue
`4*cos(pi/(b+2))`: each step has two allowed outgoing axes. Thus sides
at most 4 cannot reach 3.5 even with unbounded block lengths and no
collision checks; at side 4 the ceiling is `2*sqrt(3) < 3.5`.

Before enumerating valid blocks, dynamic programming counted the relaxation
that ignores **all** within-box collisions. At `q = 3.5`, sides
3, 4, 5, 6, 8, 10, and 12, and cutoffs through 24 wedges, its largest tested
weighted radius was only about **0.756995235254** (side 5, cutoff 24).
Longer optimistic cutoffs sometimes exceed one, but include colliding
walks and give no lower certificate. An all-face, uniform-direction-weight
variant was also screened, without finding useful short-block headroom.
These are numerical parameter screens, not complete-family exclusions.
No expensive collision-aware enumeration was launched.

### Reproduction, validation, and resource limits

Sources and data for this session are preserved outside Git in:

```text
/home/dima/.copilot/session-state/81f1076b-6f9f-4016-af64-9fac1441b9f9/files/
```

Important files are `transverse_bridges.cpp`, `transverse_resum.py`,
`transverse_oracle.py`, `transverse_upper_certificate.py`,
`exact_research_checks.py`, `transverse-w*-r2-j18.json`,
`transverse-exact-candidate.json`, `transverse-upper-certificate.json`,
`transverse-piece-oracles.jsonl`, `transverse-whole-grammar-oracle.jsonl`, `fcc_lifts.py`,
`fcc-lifts-result.jsonl`, `fcc-lift-audit.log`, `coarse_boxes.py`, and
`coarse-boxes-results.jsonl`. The C++ executable was removed after use;
rebuild it with `g++ -std=c++20 -O3 -Wall -Wextra -Werror`.

Run Python from the repository root with `.venv/bin/python`. The resummation
script accepts the seven width-specific JSON files; the exact-check script
also accepts `--length 96 --output OUTPUT.json`; the upper-certificate
script requires `--output OUTPUT.json`. Running the oracle without a file
checks complete short grammar words; with a coefficient file it checks
short individual pieces. These scripts are research artifacts, not trusted
inputs to a Lean lower-bound theorem.

All numerical experiments ran **one at a time**, without subagents or
parallel numerical workers, under address-space limits of at most
2,000,000,000 bytes. Lean builds/checks also ran serially; the largest
measured Lean RSS in the session was about 3.65 GiB during the trial of the
kernel proof, below the 10 GB budget. The baseline Python suite passed all
36 tests, the paper build was up to date, and the root and existing
experimental Lean targets passed their incremental builds. The new FCC
module and its five-theorem dependency audit passed separately.

## Fast fourfold lower proof: 2026-10-07 evening

**Verified result: `3.4003 <= mu` and `3.4003^(n-1) <= S_n` for every
positive `n`.** This is stronger than the previous 3.400034903 endpoint.
The mathematical saving is a free fourfold rotation action on the code,
proved at the geometric-word level rather than assumed by the counter.

### Argument and proof dependencies

1. A seed x-bridge starts with incoming +x and mandatory first step +y.
   Its remaining suffix is counted by the existing restoring traversal,
   with budget pruning used only to obtain an undercount.
2. Rotate by `(x,y,z) -> (x,-z,y)`. This preserves perpendicularity,
   complementary-wedge disjointness, all x-prefix heights, and hence
   irreducibility. The four copies start in +y, +z, -y, and -z, so they
   are disjoint. Each seed length coefficient can be multiplied by four.
3. Combine widths 0, 1, 2, and 3. Total x height separates the four
   seed lists; the existing primitive-bridge argument uniquely decodes
   mixed-width concatenations.
4. The internal-edge cutoffs are **20, 20, 18, and 21**. The four quarter
   rows are checked in a single sequential `native_decide` conjunction.
   `Elab.async` is disabled in that certificate. There is no unproved
   symmetry assumption, external-count hypothesis, or pruning-completeness
   requirement.
5. The degree-22 renewal inequality at **34003/10000** is checked by
   kernel arithmetic. The generic bridge-code theorem gives the lower
   bound on `mu`; Fekete gives the uniform pointwise bound with prefactor one.

The dependency audit confirms that the symmetry, undercount, and polynomial
theorems use only standard logical axioms. The final bounds additionally use
the new native row check and the pre-existing small native checks in the
growth-existence development. No `sorryAx`, old slab-row check, external-count
assumption, or unchecked replacement occurs in the new lower-bound dependency
chain.

The supporting modules are
[BridgeSymmetry.lean](lean/RubiksSnake/BridgeSymmetry.lean),
[QuarterSlab.lean](lean/RubiksSnake/QuarterSlab.lean),
[FastLowerCertificate.lean](lean/RubiksSnake/FastLowerCertificate.lean), and
[FastLowerBound.lean](lean/RubiksSnake/FastLowerBound.lean).
The old public rational-bound APIs remain available and now follow from
the stronger result. The five-theorem facade has been updated without
changing its size or the upper endpoint.

This still uses finite enumeration. It is not a purely analytic proof:
fourfold geometric symmetry replaces three quarters of the search.
The new coefficient check takes about **50--51 seconds**, instead of
the historical 325-second check. The 36-minute upper certificate remains
necessary for the best upper endpoint and was not removed.

### Exact timing requirement

Run from the repository root:

```bash
.venv/bin/python rubiks-snake/lean/check_fast_lower.py \
  --output /tmp/fast-lower-build.json
```

The [fresh-check script](lean/check_fast_lower.py) copies the 24 required
project-owned sources and the unchanged pinned dependency configuration to
a temporary directory. It reuses cached **third-party** dependencies, but
copies no project-owned proof or native-library artifacts. It rebuilds the
two native libraries and checks every source in dependency order, one job
at a time, with one Lean thread and a 4096 MiB Lean heap limit.
The temporary directory is removed on success or failure, and a timed-out
process group is terminated before cleanup.

The completed run took **117.609593850 seconds**. Maximum child RSS was
**3,324,488 KiB**, about **3.17 GiB**. The final native coefficient check
took 51.113 seconds in that run. This is a fresh check of the lower proof's
entire project-owned chain, not an incremental build, and not a clean build
of mathlib or the whole repository. Timing is machine-dependent and the
margin below two minutes is small.

The report is saved as `files/fast-lower-fresh-build.json` in this session's
artifact directory listed above. The exact statements, numerical gap, and
dependencies are checked by `files/fast_lower_audit.lean`, with output in
`files/fast-lower-audit.log`. The final default Lean build passed, all
36 Python regression tests passed, and the updated seven-page paper built
successfully with resolved cross-references.

### Lessons from the timing probes

- Use `lake lean FILE.lean` or a normal Lake target when checking native
  computations. `lake env lean FILE.lean` does not automatically initialize
  the precompiled computation plugins and can be much slower. Several early
  probes were stopped at 120 seconds for this reason; they are not valid
  measurements of the compiled algorithm.
- When checking sources directly, initialize both freshly built computation
  libraries with Lean's `--plugin` flag. Merely opening one shared library
  does not reproduce the normal Lake setup.
- An address-space limit on the Lake coordinator caused thread-creation
  failures even with single-threaded child Lean arguments. The successful
  fresh check bounds each Lean job's heap instead; all jobs are serial and
  measured memory is well below 10 GB.
- Repeated Lake startup for every single module added enough overhead to miss
  120 seconds. The successful script obtains the environment once, uses Lake
  for native library builds, and invokes the pinned Lean checker directly for
  the remaining sources. It does not skip any project-owned proof.

## Transverse caps: historical 2026-10-07 discovery record

The following records the state before the completed proof described below.
At the end of this session, the 3.45/five-minute target was not yet achieved.
The finite exact candidate had root in
**(3.450567417452112, 3.450567417455022)**, using only **14-wedge pieces**.
The integer renewal inequality is checked in Lean, but the assembled
coefficients have not yet been proved to undercount distinct valid x bridges.
The five public results and paper therefore remain unchanged.

### New geometric ingredient

The old transverse grammar unnecessarily required its head to start at its
lowest y level, and its tail to end at its highest y level. Allow an overhanging
head and tail instead:

- A head's exit is above every occupied y level; it may dip below its entry.
- A tail stays above its entry, but may rise above its final y level.
- Intermediate pieces are the earlier irreducible y bridges.
- Head, successive middle pieces, and tail occupy disjoint y slabs.
- A head has a backward crossing of each positive internal cut; a tail has
  one of each positive cut up to and including its final height.
- These conditions identify all separating cuts uniquely. Positive total y
  displacement also separates the construction from its y-reflected copy.

Head enumeration does not need a new geometric board theorem: run the old
full-cut-mask slab search from an arbitrary initial y layer instead of zero.
Its full backward mask implies both the needed positive-cut crossings and a
visit to the bottom layer. An additional secondary coordinate bounds x.
Tail coefficients are obtained by reversing the head and reflecting both x
and y. The corresponding cut mask reverses its `width` bits.

### Exact candidate

Adjoin cap families at x widths **2, 3, 4**, y spans **0, 1, 2**, and piece
cutoff **14 wedges** to the existing fast fourfold rows. Retain only new
whole-block lengths greater than **19** and **22** at widths 2 and 3;
width 4 has no old-row overlap. Truncate assembled blocks at **64 wedges**.
The exact renewal mass at 69/20 is greater than one (approximately 1.00119).
All overlap removal is by total word length, not heuristic subtraction.

The independent C++ enumeration took 0.30, 0.50, and 0.80 seconds for these
three widths, at less than 8 MiB RSS. The Lean native count/assembly took
**6.51 seconds** including Lake startup, at about **801 MiB** maximum RSS.
The standalone candidate certificate target took **8.51 seconds**, at about
**3.14 GiB** maximum RSS. These are not timings of a completed fresh lower
proof; no five-minute end-to-end claim has been established.

Validation so far:

- Independent coordinate/face-set oracle agrees with all 547 nonzero piece
  coefficients through eight wedges, including zero-entry support.
- Complete small grammars contain 279, 1,573, and 1,124 distinct valid words
  at widths 0, 1, and 2; the oracle checks geometry, x irreducibility, and
  exact recovery of every y separator.
- All **10,320** native Lean head/middle slots agree with the C++ implementation.
- All **195** assembled coefficients through degree 64 agree with the
  independent Python recurrence, including head-to-tail mask reflection.
- Lean checks the exact integer polynomial directly from its own executable
  counts, with no imported count table.

### Formalization already checked

- [BridgeCaps.lean](lean/RubiksSnake/BridgeCaps.lean): overhanging head/tail
  definitions, backward-crossing criteria, and injectivity of the entire
  head-middle-tail factorization.
- [CapGeometry.lean](lean/RubiksSnake/CapGeometry.lean): collision-free cap
  concatenation in any coordinate and initial frame; the existing slab
  traversal from an arbitrary initial layer produces canonical heads.
- [RubiksSnakeCapComputation.lean](lean/RubiksSnakeCapComputation.lean):
  separately precompiled short-piece counter and finite assembly recurrence.
- [CapEnumeration.lean](lean/RubiksSnake/CapEnumeration.lean): board
  restoration, exact tagged-histogram semantics, sublist relation to the
  previously verified geometric traversal, and absence of duplicate words.
- [CapCrossing.lean](lean/RubiksSnake/CapCrossing.lean): terminal-tag meaning,
  secondary-coordinate bounds, mask bounds, and actual backward-crossing
  witnesses for all set bits.
- [CapCatalogue.lean](lean/RubiksSnake/CapCatalogue.lean): validity of every
  enumerated piece, canonical head/middle semantics, recovery of the starting
  layer and span, duplicate-free unions, and equality of native array entries
  with the tagged geometric catalogue counts.
- [CapReversal.lean](lean/RubiksSnake/CapReversal.lean): injective head-to-tail
  transformation, preservation of geometric validity, and canonical tail
  semantics. It reuses the earlier rigid-frame and reversal theorems.
- [CapCandidateCertificate.lean](lean/RubiksSnake/CapCandidateCertificate.lean):
  exact integer renewal inequality at 69/20. Its header explicitly states
  that this is not yet a lower-bound theorem.

These modules are outside the default root. Relevant targets have been built
serially. The default root also passed after the shared compatibility lemma
was made public; that incremental rebuild took 68.26 seconds, including a
fresh 49-second check of the existing lower certificate, with maximum child
RSS 3,403,976 KiB (about 3.25 GiB).

The new theorem dependency audit reports only `propext`, `Classical.choice`
where applicable, and `Quot.sound` for the geometric/counting lemmas. The
candidate polynomial additionally uses its own `native_decide` check.
There is no `sorryAx`, external count assumption, or user-added axiom.
The mathlib ingredients reused so
far are list sublists, duplicate-free concatenation, finite sums, natural
bitwise lemmas, and the existing project's Fekete/renewal framework; no
ready-made self-avoiding-walk lower theorem was identified.

### Proof obligations at that time (all now discharged)

1. Finish the aggregate-array/catalogue histogram identities and transition
   metadata bookkeeping. Single-search array equality and disjointness across
   starting layers and transverse spans are already proved.
2. Complete the tail's secondary-coordinate displacement/mask reflection
   bookkeeping and direction compatibility. Its geometric validity and
   primary-coordinate canonical-tail property are already proved.
3. Construct the complete capped x-bridge code, using the checked canonical
   factorization theorem and secondary backward-crossing witnesses.
4. Prove the assembly coefficient lower comparison. It is sufficient to
   check the computed dynamic-programming table against local recurrence
   inequalities and use induction; proving the imperative implementation
   complete is unnecessary.
5. Combine the length-disjoint old/new codes, apply the existing renewal
   lower theorem, wire public results, and measure a fresh serial dependency
   rebuild under 300 seconds. Only then update the paper and claim 3.45.

### Reproducible external artifacts

The same session artifact directory listed earlier contains
`caps-w{2,3,4}-j14.json`, `caps-exact-candidate.json`, `caps_exact.py`,
`caps_profile.lean`, `check_cap_profile.py`, `caps-profile-output.txt`,
`caps_assembled_profile.lean`, `check_cap_assembly.py`, and
`caps-assembled-output.txt`, plus `caps_audit.lean`. `transverse_bridges.cpp` and
`transverse_oracle.py` now support caps while preserving their original mode.

Two cheaper geometric variants were screened first. Choosing the growth
axis from the first turn gave renewal mass about 0.97105 at 3.45; adding a
disjoint perpendicular-axis family with negative head excursions raised it
only to about 0.97611. These variants did not meet the target. The useful
improvement is the overhanging-cap construction, not longer enumeration.

## Completed transverse-cap proof: 2026-10-08

The full coefficient-to-bridge connection now compiles without assumptions
about externally computed counts. The exact endpoint is
`17252837 / 5000000 = 3.4505674`; both public lower bounds use it with
pointwise prefactor one.

- [CapPieces.lean](lean/RubiksSnake/CapPieces.lean) proves longitudinal
  confinement, actual backward-crossing witnesses, reflected tail masks,
  collision-free concatenation, and irreducibility of a full-mask assembly.
- [CapCounts.lean](lean/RubiksSnake/CapCounts.lean) proves slot decoding and
  exact aggregate head/middle histogram identities.
- [CapAssembly.lean](lean/RubiksSnake/CapAssembly.lean) defines semantic
  traces, proves validity and length, and proves both duplicate-free
  enumeration and injectivity of flattening by canonical cap factorization.
- [CapRecurrence.lean](lean/RubiksSnake/CapRecurrence.lean) groups catalogue
  sums by tags and proves that any checked local recurrence subsolution
  undercounts the semantic traces.
- [CapAssemblyCertificate.lean](lean/RubiksSnake/CapAssemblyCertificate.lean)
  checks every row of independently computed backward tables and compares
  the forward assembler with their head step. The tables have 65 length
  layers and `(width+1)*2^width` states per layer; long words are never
  enumerated. The final comparison is `assembledCounts_le_words`.
- [CapCode.lean](lean/RubiksSnake/CapCode.lean) exchanges axes by a proper
  rigid frame change and adds a disjoint half-turned copy, separated by
  the sign of the transverse displacement.
- [CapCombinedCode.lean](lean/RubiksSnake/CapCombinedCode.lean) proves
  width/length disjointness from the old code and the exact combined
  length-class formula.
- [CapCandidateCertificate.lean](lean/RubiksSnake/CapCandidateCertificate.lean)
  now checks the integer polynomial at **3.4505674**, not merely 3.45.
- [CapLowerBound.lean](lean/RubiksSnake/CapLowerBound.lean) supplies these
  coefficients to `BridgeCode.le_growthConstant` and transfers the result
  pointwise using the existing Fekete theorem.

The new native checker deliberately shares its counted arrays across all
row checks. A first formulation using a dependent proposition directly
repeated expensive work and was stopped after 180 seconds; the shared
Boolean checker takes about 21 seconds. Its Boolean result is converted
back to the exact logical recurrence contract by a proved theorem.

Fresh validation used [check_fast_lower.py](lean/check_fast_lower.py),
with target `RubiksSnake.CapLowerBound`, a 300-second deadline, serial
module checks, and `-j1 -M4096`. Native plugins load in dependency order.
All **40** project-owned modules were rebuilt with no reused project-owned
artifacts. Only the pinned compiler and cached third-party dependencies
were reused. Total elapsed time was **172.395182865 seconds**, with maximum
child RSS **3,325,184 KiB** (about **3.17 GiB**).
The older row check took 49.95 seconds, the recurrence certificate 21.31
seconds, and the cap polynomial module 7.60 seconds.

The machine-readable report is in the session artifact
`cap-lower-fresh-build.json`. The earlier 3.4003 check remains reproducible
with `--target RubiksSnake.FastLowerBound --seconds 120`.
At this milestone, the five-minute requirement applied to the lower-proof
chain; the length-16 upper certificate still took about 36 minutes.
The later optimization below reduces that cost. The broader base-gap target
below 0.1 was not reached and is no longer an active request.

The complete default library builds with the new five-theorem facade.
The dependency audit of the connection, growth bound, and both public lower
results contains no `sorryAx` or external-count assumptions. It includes
standard logical axioms, the existing native geometric/counting checks,
and the new recurrence and integer-polynomial `native_decide` certificates.
These compiled-evaluation checks remain part of the explicit trust boundary.
The updated seven-page paper also passed a forced `latexmk -g -pdf` rebuild,
with no LaTeX warnings, undefined references, or overfull boxes in its log.
The public Lean APIs rebuild without new warnings, and `git diff --check`
passes. No commit or push was made.

## Upper certificate optimization (2026-10-08)

The user stopped numerical-bound research at **3.4505674 <= mu <= 3.661786723**
and requested faster verification of
[ForbiddenPrefixSixteenComputation.lean](lean/RubiksSnake/ForbiddenPrefixSixteenComputation.lean).
The module has two theorems: private `array_checked` performs the finite
verification, and public `checked` converts its array result to statewise
propositions. Both feed both final upper bounds. The graph metadata is not
needed for the numerical inequality alone, but costs little and remains
checked to preserve the public API. No theorem was removed.

Two changes reduce the cost:

1. The executable prefix and window routines now live in
   [RubiksSnakePrefixComputation.lean](lean/RubiksSnakePrefixComputation.lean),
   a separately precompiled Lake library.
   [PrefixAutomatonData.lean](lean/RubiksSnake/PrefixAutomatonData.lean) remains
   a compatibility import, preserving declaration names and existing imports.
   Native precompilation alone reduced the fresh certificate check to
   **307.34 seconds**.
2. Each encoded state's four candidate extensions share its wedge list,
   last wedge, and existing pairwise-validity check. Only the new wedge is
   tested separately for each rotation.
   [WindowUpperComputation.lean](lean/RubiksSnake/WindowUpperComputation.lean)
   proves `extensionAllowed_eq_valid` for every prefix, including invalid
   ones; [EncodedPrefixAutomaton.lean](lean/RubiksSnake/EncodedPrefixAutomaton.lean)
   uses this equality in its transition-correctness proof.
   A length-12 control retained all counts and weights while graph generation
   fell from 972 ms to 414 ms.

The full, unchanged length-16 certificate then passed from fresh project-owned
artifacts in **218.693560426 seconds**, about **10x** faster than the recorded
36m19s run. This includes the native library (2.534 s), compatibility import
(0.524 s), finite-check module (213.441 s), and setup overhead. Maximum child
RSS was **2,637,076 KiB** (**2.51 GiB**). The check was serial with `-j1 -M4096`;
only the pinned compiler and cached third-party dependencies were reused.
All 4,748,260 states, 14,133,721 edges, root potential, and integer certificate
inequalities remain unchanged. The report is the session artifact
`prefix-optimized-fresh-build.json`.

Reproduce from the repository root:

```bash
.venv/bin/python rubiks-snake/lean/check_fast_lower.py \
  --target RubiksSnake.ForbiddenPrefixSixteenComputation
```

The benchmark discovers each native library's plugin dependencies from Lake's
setup file, loading Batteries before the prefix library. Its default lower
target also passed a fresh regression check: all 40 modules in **190.62 seconds**
and **3.17 GiB** peak child RSS, within the existing five-minute budget.

Normal Lake builds of the affected upper certificates were run serially,
followed by the public facade and default root library. All passed, producing
persistent build artifacts. In that build the length-16 module took 207 seconds.
The optimization changes neither numerical bound, the pointwise prefactors,
nor the existing `native_decide` trust boundary.
An audit of all five public statements and the new equivalence theorem found
no `sorryAx` or new trust assumptions; the equivalence itself uses only
`propext` and `Quot.sound`. The updated seven-page paper passed a forced
rebuild without warnings, undefined references, or overfull boxes.
No commit or push was made.
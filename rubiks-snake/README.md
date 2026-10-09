# Rubik's Snake

Here you can find several Jupyter Notebooks solving
problems related to [Rubik's Snake](https://en.wikipedia.org/wiki/Rubik%27s_Snake):

1. [Counting Rubik's Snake Shapes](count-shapes.ipynb)
2. [Counting Rubik's Snake shapes, up to reversal symmetry](count-shapes-with-reversal.ipynb)
3. [Asymptotic behavior of Rubik's Snake Shapes Count](asymptotic.ipynb)
4. [Counting Rubik's Snake Loops](count-loops.ipynb)
5. [Counting Rubik's Snake shapes with restricted rotations](rotation-restricted-shapes.ipynb)

The sequence of numbers of Rubik's Snake shapes is also [published on OEIS](https://oeis.org/A375865). 

## Certified asymptotic bounds

The [paper](paper-draft/paper.tex) proves that the exponential growth constant
exists and gives computer-assisted lower and upper bounds. Four notebooks
reproduce the finite computations; each can be run on its own:

* [Lower bound for the growth constant](asymptotic-analysis/mu-lower.ipynb)
* [Upper bound for the growth constant](asymptotic-analysis/mu-upper.ipynb)
* [Pointwise lower bounds](asymptotic-analysis/pointwise-lower.ipynb)
* [Pointwise upper bounds](asymptotic-analysis/pointwise-upper.ipynb)

Each notebook defines a reusable bound function and calls it first with small
inputs, then with the published parameters. Larger cutoffs or window sizes can improve the
bounds. The largest upper-bound run needs several GiB of memory. Its saved
output is labeled archival.

The notebooks use [rubiks_snake.py](rubiks_snake.py), NumPy, Numba, and SciPy.
Run them from this directory, their own directory, or the repository root.
Compile the paper from the repository root with
`latexmk -pdf -cd rubiks-snake/paper-draft/paper.tex`.

## Lean formalization

Build the formal development with:

```bash
cd rubiks-snake/lean
lake build
```

The five summary theorems are in
[FinalResults.lean](lean/RubiksSnake/SnAsymptotic/FinalResults.lean), in the namespace
`RubiksSnake.FinalResults`. They state existence of the growth constant and
the following bounds using exact decimal constants, reusing the existing proofs:

$$
3.4505674 \leq \mu \leq 3.661786723,
\qquad
(3.4505674)^{n-1} \leq S_n \leq 3(3.661786723)^{n-1}
\quad(n\geq1).
$$

The [current lower proof](lean/RubiksSnake/SnAsymptotic/CapLowerBound.lean) assembles
14-wedge transverse pieces between overhanging head and tail caps.
[Unique geometric decoding](lean/RubiksSnake/SnAsymptotic/CapAssembly.lean) and
[checked local recurrence inequalities](lean/RubiksSnake/SnAsymptotic/CapAssemblyCertificate.lean)
prove that the degree-64 coefficients undercount distinct valid irreducible
bridges. Opposite transverse orientations give two disjoint copies.
Length cutoffs exclude overlap with the older slab code.

That [older fast lower proof](lean/RubiksSnake/SnAsymptotic/FastLowerBound.lean) uses
[fourfold geometric symmetry](lean/RubiksSnake/SnAsymptotic/BridgeSymmetry.lean): count
only blocks whose first step is `+y`, then rotate them about x. The four
copies are disjoint because their first steps differ. This multiplies every
coefficient by four without repeating the search. The checked seed widths
are 0, 1, 2, and 3, with internal-edge cutoffs 20, 20, 18, and 21.
Budget pruning only needs to give undercounts.

The old row check takes about 50 seconds, the new recurrence check about
21 seconds, and the cap polynomial module about 8 seconds. A fresh,
serial rebuild of **all 40 project-owned lower-proof modules**, including
all three native computation libraries and all certificates, passed in
**172.40 seconds**, with maximum child RSS about **3.17 GiB**. This timing
reuses the pinned Lean compiler and cached third-party dependencies, but
no project-owned build artifacts. Reproduce it from the repository root:

```bash
.venv/bin/python rubiks-snake/lean/check_fast_lower.py
```

The fresh check has a five-minute deadline and a 4096 MiB Lean heap limit.
The older 117.61-second proof can still be benchmarked with
`--target RubiksSnake.FastLowerBound --seconds 120`.

For an ordinary incremental build, use `lake build RubiksSnake.CapLowerBound`
from the Lean directory. For fresh checks of individual files, use
`lake lean FILE.lean`, which loads the native libraries; bare
`lake env lean FILE.lean` can fall back to much slower interpreted execution.

The original five-minute
[slab certificate](lean/RubiksSnake/SnAsymptotic/SlabCountCertificate.lean) remains available
for historical and experimental modules, but is no longer imported by the
default build or required by the best lower bound. Its public numerical
APIs are preserved as corollaries of the stronger result.
[Enumeration correctness](lean/RubiksSnake/SnAsymptotic/SlabEnumeration.lean),
[collision freedom](lean/RubiksSnake/SnAsymptotic/SlabBlockValidity.lean), and
[unique decoding](lean/RubiksSnake/SnAsymptotic/BridgeCode.lean) are proved separately.
A block is selected by requiring a backward crossing of every internal slab
boundary, avoiding the subtraction recurrence used by the Python certificate.

The earlier [plane-block construction](lean/RubiksSnake/SnAsymptotic/SlabBlocks.lean) gives a base
of 3.1. The [occupied-interface construction](lean/RubiksSnake/SnAsymptotic/RecordBlocks.lean)
allows a block to revisit the preceding plane. It checks compatibility
against the preceding block, proves that the entire concatenation is
collision-free, and proves unique decoding. One
[weighted continuation step](lean/RubiksSnake/SnAsymptotic/WeightedRecordLowerBound.lean)
gives the base 3.193 on the same 428-block language; using only uniform
continuation counts gives 3.16.
The [upper bound](lean/RubiksSnake/SnAsymptotic/ForbiddenPrefixSixteenUpperBound.lean) uses the
proper prefixes of collision factors through length 16. Its automaton has
4,748,260 states; 48 sparse integer iterations generate the potential. A six-step terminal
estimate handles states with no long continuation, without an artificial
positive weight that would inflate the prefactor.
The smaller [seven-symbol window](lean/RubiksSnake/SnAsymptotic/WindowSevenUpperBound.lean)
also gives $S_n\leq(8/3)(3.704)^{n-1}$.
The length-16 finite check now takes **3m39s**, down from the recorded **36m19s**.
Its executable routines are in the separately precompiled
[prefix computation library](lean/RubiksSnake/SnAsymptotic/PrefixComputation.lean).
For each state, the graph builder computes the existing wedge geometry and
validity once, then tests each of the four possible new wedges against it.
[The equivalence proof](lean/RubiksSnake/SnAsymptotic/WindowUpperComputation.lean)
covers all prefixes, including invalid ones.
Both theorems in
[the finite-check module](lean/RubiksSnake/SnAsymptotic/ForbiddenPrefixSixteenComputation.lean)
remain needed by the final upper bounds; every original certificate obligation
and graph count is retained.

The fresh serial measurement includes rebuilding the native library, its
compatibility import, and the complete finite check: **218.69 seconds**,
with maximum child RSS **2.51 GiB** and a **4096 MiB** Lean heap limit.
It reuses only the pinned compiler and cached third-party dependencies.
Reproduce it from the repository root:

```bash
.venv/bin/python rubiks-snake/lean/check_fast_lower.py \
  --target RubiksSnake.ForbiddenPrefixSixteenComputation
```

This measures the finite certificate, not a fresh build of the entire library.
The smaller
[length-14 certificate](lean/RubiksSnake/SnAsymptotic/ForbiddenPrefixFourteenUpperBound.lean)
matches the master paper's upper endpoint 3.667542939; its module rebuilt in
44 seconds with the new shared computation library.
The strongest formal bounds improve both endpoints of the original notebook
certificate. Their remaining base gap is 0.211219323. Numerical-bound research
has concluded at the user's request.
The lower notebooks retain the historical 3.400034903 computation; they do
not reproduce the fourfold or transverse-cap certificates.

[Submultiplicativity](lean/RubiksSnake/SnAsymptotic/SnAsymptotic_MuExistence.lean) also gives
$\mu^{n-1}\leq S_n$ at every positive length. Consequently, any lower bound on
$\mu$ gives a pointwise lower bound with prefactor one.

The [bounded-component comparison](lean/RubiksSnake/SnAsymptotic/BoundedComponents.lean)
proves that counting a fixed number of independent snake components, even
with a polynomial number of placements, does not increase the exponential
rate. The comparison also covers a variable number of components within a
fixed bound, indexed by total wedges. A
[frame-normalization lemma](lean/RubiksSnake/SnAsymptotic/CardinalDirections.lean)
connects paths in any cardinal initial frame to the original rotation formulas.
The [finite-transfer criterion](lean/RubiksSnake/SnAsymptotic/FiniteTransfer.lean) proves
divergence of a weighted transfer series from a nonzero nonnegative
subeigenvector, without assuming irreducibility.
A rank inequality bounds construction steps by wedge count, including
zero-wedge steps.
These are counting tools; they do not certify an additional geometric
construction or improve the displayed numerical bounds.

The [loop development](lean/RubiksSnake/OtherSequences/Loops.lean) proves preservation of
validity under every integer cyclic shift, including negative shifts.
The default library build imports all these results and contains no `sorry`.
The finite computational checks use Lean's `native_decide`, as do the
existing small-count proofs.

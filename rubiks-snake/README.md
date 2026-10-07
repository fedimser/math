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
[FinalResults.lean](lean/RubiksSnake/FinalResults.lean), in the namespace
`RubiksSnake.FinalResults`. They state existence of the growth constant and
the following bounds using exact decimal constants, reusing the existing proofs:

$$
3.400034903 \leq \mu \leq 3.661786723,
\qquad
(3.400034903)^{n-1} \leq S_n \leq 3(3.661786723)^{n-1}
\quad(n\geq1).
$$

The [irreducible-slab construction](lean/RubiksSnake/SlabLowerBound.lean)
verifies the master paper's lower endpoint. It uses slab widths 0, 1, and 2,
with internal-edge cutoffs 28, 20, and 18. A restoring byte-array traversal
checks the exact coefficients in about five minutes. The small
[computation library](lean/RubiksSnakeComputation.lean) is compiled to native
code automatically by Lake; the mathematical proofs do not require native
compilation of mathlib.
[Enumeration correctness](lean/RubiksSnake/SlabEnumeration.lean),
[collision freedom](lean/RubiksSnake/SlabBlockValidity.lean), and
[unique decoding](lean/RubiksSnake/SlabLanguage.lean) are proved separately.
A block is selected by requiring a backward crossing of every internal slab
boundary, avoiding the subtraction recurrence used by the Python certificate.

The earlier [plane-block construction](lean/RubiksSnake/SlabBlocks.lean) gives a base
of 3.1. The [occupied-interface construction](lean/RubiksSnake/RecordBlocks.lean)
allows a block to revisit the preceding plane. It checks compatibility
against the preceding block, proves that the entire concatenation is
collision-free, and proves unique decoding. One
[weighted continuation step](lean/RubiksSnake/WeightedRecordLowerBound.lean)
gives the base 3.193 on the same 428-block language; using only uniform
continuation counts gives 3.16.
The [upper bound](lean/RubiksSnake/ForbiddenPrefixSixteenUpperBound.lean) uses the
proper prefixes of collision factors through length 16. Its automaton has
4,748,260 states; 48 sparse integer iterations generate the potential. A six-step terminal
estimate handles states with no long continuation, without an artificial
positive weight that would inflate the prefactor.
The smaller [seven-symbol window](lean/RubiksSnake/WindowSevenUpperBound.lean)
also gives $S_n\leq(8/3)(3.704)^{n-1}$.
The length-16 finite check took about 36 minutes, with peak RSS of 2.30 GiB under a
6 GiB address-space cap. The smaller
[length-14 certificate](lean/RubiksSnake/ForbiddenPrefixFourteenUpperBound.lean)
matches the master paper's upper endpoint 3.667542939 and takes about four minutes.
The strongest formal bounds match the master paper's lower endpoint and improve
its upper endpoint. Their remaining base gap is 0.261751820, not yet below 0.1.

[Submultiplicativity](lean/RubiksSnake/SnAsymptotic_MuExistence.lean) also gives
$\mu^{n-1}\leq S_n$ at every positive length. Consequently, any lower bound on
$\mu$ gives a pointwise lower bound with prefactor one.

The [bounded-component comparison](lean/RubiksSnake/BoundedComponents.lean)
proves that counting a fixed number of independent snake components, even
with a polynomial number of placements, does not increase the exponential
rate. The comparison also covers a variable number of components within a
fixed bound, indexed by total wedges. A
[frame-normalization lemma](lean/RubiksSnake/CardinalDirections.lean)
connects paths in any cardinal initial frame to the original rotation formulas.
The [finite-transfer criterion](lean/RubiksSnake/FiniteTransfer.lean) proves
divergence of a weighted transfer series from a nonzero nonnegative
subeigenvector, without assuming irreducibility.
A rank inequality bounds construction steps by wedge count, including
zero-wedge steps.
These are counting tools; they do not certify an additional geometric
construction or improve the displayed numerical bounds.

The [loop development](lean/RubiksSnake/Loops.lean) proves preservation of
validity under every integer cyclic shift, including negative shifts.
The default library build imports all these results and contains no `sorry`.
The finite computational checks use Lean's `native_decide`, as do the
existing small-count proofs.

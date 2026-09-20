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

import RubiksSnakeComputation
import Mathlib.Tactic

/-!
# Exact irreducible slab counts

Rows are indexed by the number of internal edges. A block has one more
wedge. Irreducibility is checked during enumeration by recording a backward
crossing of every internal slab boundary.
Thus internal length `k` gives block-word length `k + 1`; a full formula with
`n` rotations has `n + 1` wedges, including its starting wedge.
-/

namespace RubiksSnake.SlabEnumeration

/-- Retained enumeration row for displacement one (width zero), indexed by
internal-edge lengths zero through 28. -/
def rowOne : Array Nat :=
  #[0, 4, 8, 16, 24, 40, 72, 136, 224, 392, 712, 1272, 2168, 3840,
    6832, 12112, 20904, 36856, 65192, 115096, 199368, 350696, 618032,
    1087696, 1887888, 3314376, 5825784, 10230736, 17775440]

/-- Retained irreducible enumeration row for displacement two (width one),
indexed by internal-edge lengths zero through 20. -/
def rowTwo : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 136, 976, 4592, 17072, 55640, 168560, 492640,
    1420080, 4053840, 11473488, 32221168, 90091856, 250832920, 696851968]

/-- Retained irreducible enumeration row for displacement three (width two),
indexed by internal-edge lengths zero through 18. -/
def rowThree : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10088, 129840, 983752,
    5554512, 26141688, 109326232]

/-- Sum the three rows at block-word length `j`, using internal-edge index
`j - 1`; length zero and entries beyond each row's cutoff contribute zero. -/
def retainedCount (j : Nat) : Nat :=
  if j = 0 then 0 else
    rowOne[j - 1]?.getD 0 + rowTwo[j - 1]?.getD 0 + rowThree[j - 1]?.getD 0

/-- Native evaluation certifies the three retained rows against the
irreducible traversal at widths `0, 1, 2` and internal cutoffs `28, 20, 18`. -/
theorem rows_checked :
    counts 0 28 true = rowOne ∧
    counts 1 20 true = rowTwo ∧
    counts 2 18 true = rowThree := by
  native_decide

end RubiksSnake.SlabEnumeration

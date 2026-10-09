import RubiksSnake.SnAsymptotic.PrunedComputation
import RubiksSnake.SnAsymptotic.SlabCountCertificate

/-! Lower coefficient rows from four slab widths and a pruned traversal. -/

namespace RubiksSnake.ExtendedSlab

open SlabEnumeration

/-- Proposed width-one coefficients through 23 internal edges; these still require a native check. -/
def rowTwo : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 136, 976, 4592, 17072, 55640, 168560, 492640,
    1420080, 4053840, 11473488, 32221168, 90091856, 250832920, 696851968,
    1931311224, 5347647232, 14784171080]

/-- Proposed width-two coefficients through 22 internal edges, indexed from length zero. -/
def rowThree : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10088, 129840, 983752,
    5554512, 26141688, 109326232, 423261992, 1558968496, 5549341816, 19278469184]

/-- Proposed width-three coefficients through 23 internal edges, with zeros below the first block. -/
def rowFour : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    817336, 15211408, 159061880, 1205491776, 7421527784]

/-- Candidate number of length-`j` blocks, summing the four rows at internal length `j - 1`.
Out-of-range coefficients and the empty block contribute zero. -/
def retainedCount (j : Nat) : Nat :=
  if j = 0 then 0 else
    SlabEnumeration.rowOne[j - 1]?.getD 0 + rowTwo[j - 1]?.getD 0 +
      rowThree[j - 1]?.getD 0 + rowFour[j - 1]?.getD 0

/-- The three enumeration equalities needed to use the proposed rows as lower coefficients.
This proposition is an outstanding certificate obligation, not an assumed axiom. -/
def RowsVerified : Prop :=
  budgetCounts 1 23 true = rowTwo ∧
    budgetCounts 2 22 true = rowThree ∧ budgetCounts 3 23 true = rowFour

/-- Check the three proposed rows in parallel using the budget-pruned native traversal. -/
def checkRows : Bool :=
  let second := Task.spawn fun _ => budgetCounts 1 23 true == rowTwo
  let third := Task.spawn fun _ => budgetCounts 2 22 true == rowThree
  let fourth := Task.spawn fun _ => budgetCounts 3 23 true == rowFour
  second.get && third.get && fourth.get

attribute [local irreducible] budgetCounts

/-- A successful Boolean row check discharges all three enumeration obligations. -/
theorem rows_verified_of_check (h : checkRows = true) : RowsVerified := by
  simpa only [checkRows, Task.spawn, Bool.and_eq_true, beq_iff_eq, RowsVerified,
    and_assoc] using h

/-- Exact arithmetic verifies the degree-29 renewal inequality at the candidate base 3.429771044.
This numerical comparison alone does not verify the proposed coefficient rows. -/
theorem polynomial :
    (3429771044 / 1000000000 : ℝ) ^ 29 ≤
      ∑ j : Fin 29, (retainedCount (j.val + 1) : ℝ) *
        (3429771044 / 1000000000 : ℝ) ^ (29 - (j.val + 1)) := by
  norm_num [Fin.sum_univ_succ, retainedCount, SlabEnumeration.rowOne,
    rowTwo, rowThree, rowFour]

end RubiksSnake.ExtendedSlab

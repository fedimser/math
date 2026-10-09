import RubiksSnake.SnAsymptotic.QuarterSlab

/-!
# Short fourfold-symmetry lower certificate

The searches run sequentially and count only the first `+y` branch.
No expensive slab or upper-bound certificate is imported.
-/

set_option Elab.async false

namespace RubiksSnake.FastLower

/-- Quarter-orbit row at width zero and twenty internal edges. -/
def rowZero : Array Nat :=
  #[0, 1, 2, 4, 6, 10, 18, 34, 56, 98, 178, 318, 542, 960, 1708,
    3028, 5226, 9214, 16298, 28774, 49842]

/-- Quarter-orbit row at width one and twenty internal edges. -/
def rowOne : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 34, 244, 1148, 4268, 13910, 42140, 123160,
    355020, 1013460, 2868372, 8055292, 22522964, 62708230, 174212992]

/-- Quarter-orbit row at width two and eighteen internal edges. -/
def rowTwo : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2522, 32460, 245938,
    1388628, 6535422, 27331558]

/-- Quarter-orbit row at width three and twenty-one internal edges. -/
def rowThree : Array Nat :=
  #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    204334, 3802852, 39765470]

/-- Native checking of all four small rows, in a single sequential conjunction. -/
theorem rows_checked :
    QuarterSlab.counts 0 19 = rowZero ∧
    QuarterSlab.counts 1 19 = rowOne ∧
    QuarterSlab.counts 2 17 = rowTwo ∧
    QuarterSlab.counts 3 20 = rowThree := by
  native_decide

/-- Full-orbit lower coefficients indexed by internal edges, extended by zero. -/
def coefficient (n : Nat) : Nat :=
  4 * (rowZero[n]?.getD 0 + rowOne[n]?.getD 0 +
    rowTwo[n]?.getD 0 + rowThree[n]?.getD 0)

/-- Exact rational renewal inequality at the improved lower endpoint. -/
theorem polynomial :
    (34003 / 10000 : ℝ) ^ 22 ≤
      ∑ j : Fin 22, (coefficient j.val : ℝ) *
        (34003 / 10000 : ℝ) ^ (22 - (j.val + 1)) := by
  norm_num [Fin.sum_univ_succ, coefficient, rowZero, rowOne, rowTwo, rowThree]

end RubiksSnake.FastLower

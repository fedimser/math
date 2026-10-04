import RubiksSnake.Definitions

/-! Definitions and simple facts about rotation-restricted sequences. -/

namespace RubiksSnake

noncomputable section

/-
Rubik's Snake has 4 possible rotations, encoded in formula by numbers 0,1,2,3.

Define SR_r(n) - number of n-wedge snakes when only rotations from set r
are allowed.
-/

/-- Shapes using only rotation symbols from `allowed`. -/
def SR (allowed : Finset Rotation) (n : ℕ+) : ℕ :=
  countWords ((n : ℕ) - 1) fun w => Valid w ∧ ∀ i, w i ∈ allowed

/-- Unrestricted rotations. -/
lemma SR_0123 (n : ℕ+) :
    SR ({0, 1, 2, 3} : Finset Rotation) n = S n := by
  sorry

/-- Counts when only one rotation is allowed. -/
lemma SR_0 (n : ℕ+) : SR {0} n = 1 := by
  sorry

lemma SR_1 (n : ℕ+) : SR {1} n = 1 := by
  sorry

lemma SR_2 (n : ℕ+) : SR {2} n = if n ≤ 4 then 1 else 0 := by
  sorry

lemma SR_3 (n : ℕ+) : SR {3} n = 1 := by
  sorry

end

end RubiksSnake

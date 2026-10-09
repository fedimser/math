import RubiksSnake.SnAsymptotic.FiniteWindowUpperBound
import RubiksSnake.SnAsymptotic.WindowSevenCertificate
import Mathlib.Tactic.IntervalCases

/-!
# The seven-symbol pointwise bound

The exact certificate gives base `463/125 = 3.704` and prefactor `8/3`,
uniformly in the formula length. The weighted argument starts at seven
symbols and reserves two terminal symbols; the shorter lengths are bounded
by the elementary count `4^k`.
-/

namespace RubiksSnake
namespace WindowSevenUpper

/-- Seven-window potential summed over actual valid `k`-rotation words,
retaining multiplicity when several words have the same suffix. -/
def totalWeight (k : ℕ) : ℕ :=
  FiniteWindowUpper.totalWeight 7 weight k

/-- At seven rotations, the weighted total is the certified initial sum over
valid full-window states. -/
lemma totalWeight_seven : totalWeight 7 = 83514793291 := by
  rw [totalWeight, FiniteWindowUpper.totalWeight_at_width]
  exact initialWeight_value

/-- Actual weighted totals after `t` further rotations grow by at most the
exact factor `(463 / 125)^t` from the certified seven-rotation initial total. -/
lemma totalWeight_bound (t : ℕ) :
    (totalWeight (7 + t) : ℝ) ≤ 83514793291 * (463 / 125 : ℝ) ^ t := by
  have h := FiniteWindowUpper.totalWeight_bound 7 (by decide) weight 463 125
    (by decide) (fun rs hlen hvalid => (certificate rs hlen hvalid).2) t
  change (totalWeight (7 + t) : ℝ) ≤ (totalWeight 7 : ℝ) * _ at h
  simpa only [totalWeight_seven, Nat.cast_ofNat] using h

/-- Combines the seven-rotation starting window and two terminal steps to bound
valid `(9 + t)`-rotation formulas with prefactor `8 / 3` and exact base `463 / 125`. -/
lemma count_bound_from_nine (t : ℕ) :
    (countValidFormulas (9 + t) : ℝ) ≤
      (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ (9 + t) := by
  have hcount :
      242294 * (countValidFormulas (9 + t) : ℝ) ≤
        (totalWeight (7 + t) : ℝ) := by
    have h := FiniteWindowUpper.count_terminal_le 7 (by decide) weight 242294 2
      (7 + t) (by omega) (fun rs hlen hvalid => (certificate rs hlen hvalid).1)
    rw [show 7 + t + 2 = 9 + t by omega] at h
    exact_mod_cast h
  calc
    (countValidFormulas (9 + t) : ℝ) ≤ (totalWeight (7 + t) : ℝ) / 242294 := by
      linarith
    _ ≤ (83514793291 * (463 / 125 : ℝ) ^ t) / 242294 :=
      div_le_div_of_nonneg_right (totalWeight_bound t) (by norm_num)
    _ = (83514793291 / 242294 : ℝ) * (463 / 125 : ℝ) ^ t := by ring
    _ ≤ ((8 / 3 : ℝ) * (463 / 125 : ℝ) ^ 9) * (463 / 125 : ℝ) ^ t := by
      apply mul_le_mul_of_nonneg_right
      · norm_num
      · positivity
    _ = (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ (9 + t) := by
      rw [pow_add]
      ring

end WindowSevenUpper

/-- A uniform seven-symbol upper bound, valid for every formula length. -/
theorem countValidFormulas_le_seven_window_upper (k : ℕ) :
    (countValidFormulas k : ℝ) ≤ (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ k := by
  by_cases hk : 9 ≤ k
  · have h := WindowSevenUpper.count_bound_from_nine (k - 9)
    simpa [Nat.add_sub_of_le hk] using h
  · have hsmall : k < 9 := by omega
    calc
      (countValidFormulas k : ℝ) ≤ (4 : ℝ) ^ k := by
        exact_mod_cast countFormulas_upper_bound k
      _ ≤ (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ k := by
        interval_cases k <;> norm_num

end RubiksSnake

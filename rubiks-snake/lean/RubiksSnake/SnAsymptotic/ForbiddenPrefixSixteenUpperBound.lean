import RubiksSnake.SnAsymptotic.ForbiddenPrefixSixteenCertificate
import Mathlib.Tactic.IntervalCases

namespace RubiksSnake
namespace ForbiddenPrefixSixteenUpper

/-- Uses six terminal rotations to bound the count of valid `(6 + k)`-rotation
formulas with prefactor `2.95621` and exact base `3661786723 / 1000000000`. -/
lemma count_bound_from_six (k : ℕ) :
    (countValidFormulas (6 + k) : ℝ) ≤
      2.95621 * (3661786723 / 1000000000 : ℝ) ^ (6 + k) := by
  apply PrefixAutomaton.count_bound_of_certificate dictionary dictionary_zero
    dictionary_closed weight terminalWeight 3661786723 1000000000 2990000000000 6
    (by norm_num) (by norm_num) ?_ ?_ ?_ ?_ 2.95621 ?_ k
  · intro rs hrs hv
    exact (certificate rs hrs hv).1
  · intro t ht rs hrs hv
    exact (certificate rs hrs hv).2.1 ⟨t, ht⟩
  · intro rs hrs hv
    exact (certificate rs hrs hv).2.2.1
  · intro rs hrs hv
    exact (certificate rs hrs hv).2.2.2
  · rw [initialWeight_value]
    norm_num

end ForbiddenPrefixSixteenUpper

/-- Uniform certified bound `2.95621 * (3661786723 / 1000000000)^k` for the count of
valid formulas with `k` rotations and `k + 1` wedges. The base is exactly
`3.661786723`, and the prefactor applies pointwise, including the short lengths. -/
theorem countValidFormulas_le_forbidden_prefix_sixteen_upper (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      2.95621 * (3661786723 / 1000000000 : ℝ) ^ k := by
  by_cases hk : 6 ≤ k
  · simpa [Nat.add_sub_of_le hk] using
      ForbiddenPrefixSixteenUpper.count_bound_from_six (k - 6)
  · calc
      (countValidFormulas k : ℝ) ≤ (4 : ℝ) ^ k := by
        exact_mod_cast countFormulas_upper_bound k
      _ ≤ 2.95621 * (3661786723 / 1000000000 : ℝ) ^ k := by
        interval_cases k <;> norm_num

end RubiksSnake

import RubiksSnake.ForbiddenPrefixFourteenCertificate
import Mathlib.Tactic.IntervalCases

namespace RubiksSnake
namespace ForbiddenPrefixFourteenUpper

/-- Uses five terminal rotations to bound the count of valid `(5 + k)`-rotation
formulas with prefactor `3` and exact base `3667542939 / 1000000000`. -/
lemma count_bound_from_five (k : ℕ) :
    (countValidFormulas (5 + k) : ℝ) ≤
      3 * (3667542939 / 1000000000 : ℝ) ^ (5 + k) := by
  apply PrefixAutomaton.count_bound_of_certificate dictionary dictionary_zero
    dictionary_closed weight terminalWeight 3667542939 1000000000 11500000000000 5
    (by norm_num) (by norm_num) ?_ ?_ ?_ ?_ 3 ?_ k
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

end ForbiddenPrefixFourteenUpper

/-- Uniform certified bound with prefactor `3` and exact base
`3667542939 / 1000000000` for the count of valid formulas with `k` rotations
(`k + 1` wedges), including lengths below the five-step terminal horizon. -/
theorem countValidFormulas_le_forbidden_prefix_fourteen_upper (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      3 * (3667542939 / 1000000000 : ℝ) ^ k := by
  by_cases hk : 5 ≤ k
  · simpa [Nat.add_sub_of_le hk] using
      ForbiddenPrefixFourteenUpper.count_bound_from_five (k - 5)
  · calc
      (countValidFormulas k : ℝ) ≤ (4 : ℝ) ^ k := by
        exact_mod_cast countFormulas_upper_bound k
      _ ≤ 3 * (3667542939 / 1000000000 : ℝ) ^ k := by
        interval_cases k <;> norm_num

end RubiksSnake

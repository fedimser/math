import RubiksSnake.ForbiddenPrefixCertificate
import Mathlib.Tactic.IntervalCases

namespace RubiksSnake
namespace ForbiddenPrefixUpper

/-- Reserves four terminal rotations to obtain the exact pointwise bound with
prefactor `9 / 2` and base `147 / 40` for the count of valid `(4 + k)`-rotation formulas. -/
lemma count_bound_from_four (k : ℕ) :
    (countValidFormulas (4 + k) : ℝ) ≤
      (9 / 2 : ℝ) * (147 / 40 : ℝ) ^ (4 + k) := by
  apply PrefixAutomaton.count_bound_of_certificate dictionary dictionary_zero
    dictionary_closed weight terminalWeight 147 40 361538007 4
    (by norm_num) (by norm_num) ?_ ?_ ?_ ?_ (9 / 2) ?_ k
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

end ForbiddenPrefixUpper

/-- Uniform certified bound with prefactor `9 / 2` and exact rational base
`147 / 40` for the count of valid formulas with `k` rotations and `k + 1` wedges. -/
theorem countValidFormulas_le_forbidden_prefix_upper (k : ℕ) :
    (countValidFormulas k : ℝ) ≤ (9 / 2 : ℝ) * (147 / 40 : ℝ) ^ k := by
  by_cases hk : 4 ≤ k
  · simpa [Nat.add_sub_of_le hk] using ForbiddenPrefixUpper.count_bound_from_four (k - 4)
  · calc
      (countValidFormulas k : ℝ) ≤ (4 : ℝ) ^ k := by
        exact_mod_cast countFormulas_upper_bound k
      _ ≤ (9 / 2 : ℝ) * (147 / 40 : ℝ) ^ k := by
        interval_cases k <;> norm_num

end RubiksSnake

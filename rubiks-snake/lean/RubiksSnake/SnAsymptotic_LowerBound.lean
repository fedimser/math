import RubiksSnake.SnAsymptoticEasy

/-!
# A simple pointwise lower bound

Choosing only positive coordinate directions gives two independent choices at
every joint and never repeats a cube. This proves the baseline exponential
lower bound with base `2`.

To improve the base later, replace this positive-direction family with a larger
concatenable family. The final theorem can retain the same shape, with only its
base and constant changed.
-/

namespace RubiksSnake

/-- The positive-direction construction, stated over the reals with explicit
constant `1`. -/
theorem Sn_lower_bound_two_real (n : ℕ+) :
    (2 : ℝ) ^ ((n : ℕ) - 1) ≤ S n := by
  exact_mod_cast Sn_lower_bound_2n n

/-- Main result: `S_n ≥ 1 * 2^(n-1)` for every positive `n`. -/
theorem SnAsymptotic_LowerBound (n : ℕ+) :
    1 * (2 : ℝ) ^ ((n : ℕ) - 1) ≤ S n := by
  simpa using Sn_lower_bound_two_real n

end RubiksSnake

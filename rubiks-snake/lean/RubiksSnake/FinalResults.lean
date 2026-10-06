import RubiksSnake.SnAsymptotitc_MuLowerBound
import RubiksSnake.SnAsymptotic_UpperBound

/-!
# Final asymptotic results

These five statements reuse the existing proofs. The growth constant is
`snakeGrowthConstant`; a formula of length `n` describes `n + 1` wedges.
All decimal constants below are exact real numbers, not floating-point values.
-/

open Filter Topology

namespace RubiksSnake.FinalResults

theorem mu_exists :
    ∃ μ : ℝ, 0 < μ ∧
      Tendsto
        (fun n : ℕ => (countValidFormulas n : ℝ) ^ (1 / (n : ℝ)))
        atTop (𝓝 μ) :=
  SnAsymptotic_MuExistence

theorem mu_lower_bound : (3.193 : ℝ) ≤ snakeGrowthConstant := by
  convert snakeGrowthConstant_ge_3193_div_1000 using 1
  norm_num

theorem mu_upper_bound : snakeGrowthConstant ≤ (3.675 : ℝ) := by
  convert snakeGrowthConstant_le_147_div_40 using 1
  norm_num

theorem Sn_lower_bound (n : ℕ+) :
    (3.193 : ℝ) ^ ((n : ℕ) - 1) ≤ (S n : ℝ) := by
  convert Sn_lower_bound_3193_div_1000 n using 1
  norm_num

theorem Sn_upper_bound (n : ℕ+) :
    (S n : ℝ) ≤ 4.5 * (3.675 : ℝ) ^ ((n : ℕ) - 1) := by
  convert Sn_upper_bound_147_div_40 n using 1
  norm_num

end RubiksSnake.FinalResults

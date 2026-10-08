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

/-- The root growth rate of valid rotation-word counts converges to a positive real number. -/
theorem mu_exists :
    ∃ μ : ℝ, 0 < μ ∧
      Tendsto
        (fun n : ℕ => (countValidFormulas n : ℝ) ^ (1 / (n : ℝ)))
        atTop (𝓝 μ) :=
  SnAsymptotic_MuExistence

/-- Geometrically separated overhanging caps give this exact lower bound on `mu`. -/
theorem mu_lower_bound : (3.4505674 : ℝ) ≤ snakeGrowthConstant :=
  CapLower.growthConstant_lower_bound

/-- The length-sixteen collision-prefix certificate bounds `mu` by this exact decimal. -/
theorem mu_upper_bound : snakeGrowthConstant ≤ (3.661786723 : ℝ) := by
  convert snakeGrowthConstant_le_3661786723_div_1000000000 using 1
  norm_num

/-- Every positive wedge length has the certified lower base, with prefactor one. -/
theorem Sn_lower_bound (n : ℕ+) :
    (3.4505674 : ℝ) ^ ((n : ℕ) - 1) ≤ (S n : ℝ) :=
  CapLower.Sn_lower_bound n

/-- Every positive wedge length has the certified upper base and uniform prefactor three. -/
theorem Sn_upper_bound (n : ℕ+) :
    (S n : ℝ) ≤ 3 * (3.661786723 : ℝ) ^ ((n : ℕ) - 1) := by
  convert Sn_upper_bound_3661786723_div_1000000000 n using 1
  norm_num

end RubiksSnake.FinalResults

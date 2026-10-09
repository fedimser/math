import RubiksSnake.SnAsymptotic.SnAsymptotic_LowerBound
import RubiksSnake.SnAsymptotic.SnAsymptotic_MuExistence

/-!
# Lower bounds on the growth constant

A positive exponential pointwise lower bound passes to the logarithmic
growth limit. Plane blocks give `mu >= 3.1`; blocks with an occupied
interface give `mu >= 3.16`. Weighting the same interfaces strengthens the
bound to `mu >= 3.193`. The fourfold-symmetry slab certificate strengthens
this to `mu >= 3.4003`. Overhanging transverse caps now give
`mu >= 3.4505674`, while preserving the earlier numerical APIs.
-/

namespace RubiksSnake

/-- The plane-block pointwise estimate yields growth constant at least 3.1. -/
theorem snakeGrowthConstant_ge_31_div_10 :
    (31 / 10 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise (1 / 4) (31 / 10)
    (by norm_num) (by norm_num) countValidFormulas_lower_bound_31_div_10_with_prefactor

/-- Allowing compatible occupied interfaces between blocks raises the certified rate to 3.16. -/
theorem snakeGrowthConstant_ge_79_div_25 :
    (79 / 25 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise (1 / 4) (79 / 25)
    (by norm_num) (by norm_num) countValidFormulas_lower_bound_79_div_25_with_prefactor

/-- Weighting those occupied interfaces gives the stronger certified rate 3.193. -/
theorem snakeGrowthConstant_ge_3193_div_1000 :
    (3193 / 1000 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise 1 (3193 / 1000)
    (by norm_num) (by norm_num)
    (fun k => by simpa only [one_mul] using countValidFormulas_lower_bound_3193_div_1000 k)

/-- The earlier lower endpoint follows from the stronger symmetry-based certificate. -/
theorem snakeGrowthConstant_ge_3400034903_div_1000000000 :
    (3400034903 / 1000000000 : ℝ) ≤ snakeGrowthConstant :=
  (by norm_num : (3400034903 / 1000000000 : ℝ) ≤ 3.4003).trans
    FastLower.growthConstant_lower_bound

/-- The fast fourfold-symmetry certificate gives the improved lower endpoint. -/
theorem snakeGrowthConstant_ge_34003_div_10000 :
    (34003 / 10000 : ℝ) ≤ snakeGrowthConstant := by
  convert FastLower.growthConstant_lower_bound using 1
  norm_num

/-- The fully connected transverse-cap certificate gives the current lower endpoint. -/
theorem snakeGrowthConstant_ge_17252837_div_5000000 :
    (17252837 / 5000000 : ℝ) ≤ snakeGrowthConstant := by
  convert CapLower.growthConstant_lower_bound using 1
  norm_num

end RubiksSnake
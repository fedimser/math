import RubiksSnake.SnAsymptotic_LowerBound
import RubiksSnake.SnAsymptotic_MuExistence

/-!
# Lower bounds on the growth constant

A positive exponential pointwise lower bound passes to the logarithmic
growth limit. Plane blocks give `mu >= 3.1`; blocks with an occupied
interface give `mu >= 3.16`. Weighting the same interfaces strengthens the
bound to `mu >= 3.193`.
-/

namespace RubiksSnake

theorem snakeGrowthConstant_ge_31_div_10 :
    (31 / 10 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise (1 / 4) (31 / 10)
    (by norm_num) (by norm_num) countValidFormulas_lower_bound_31_div_10_with_prefactor

theorem snakeGrowthConstant_ge_79_div_25 :
    (79 / 25 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise (1 / 4) (79 / 25)
    (by norm_num) (by norm_num) countValidFormulas_lower_bound_79_div_25_with_prefactor

theorem snakeGrowthConstant_ge_3193_div_1000 :
    (3193 / 1000 : ℝ) ≤ snakeGrowthConstant := by
  exact snakeGrowthConstant_ge_of_pointwise 1 (3193 / 1000)
    (by norm_num) (by norm_num)
    (fun k => by simpa only [one_mul] using countValidFormulas_lower_bound_3193_div_1000 k)

end RubiksSnake
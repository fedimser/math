import RubiksSnake.SnAsymptotic.CapLowerBound
import RubiksSnake.SnAsymptotic.ForbiddenPrefixSixteenUpperBound

/-!
# Pointwise lower and upper bounds for S(n).
-/

open Filter Topology

namespace RubiksSnake.FinalResults

/-- Geometrically separated cap-and-bridge codes give this lower bound at
every positive wedge length. -/
theorem Sn_lower_bound (n : ℕ+) :
    (3.4505674 : ℝ) ^ ((n : ℕ) - 1) ≤ (S n : ℝ) :=
  CapLower.Sn_lower_bound n

/-- A length-sixteen collision-prefix certificate gives this upper bound at
every positive wedge length. -/
theorem Sn_upper_bound (n : ℕ+) :
    (S n : ℝ) ≤ 2.95621 * (3.661786723 : ℝ) ^ ((n : ℕ) - 1) := by
  convert countValidFormulas_le_forbidden_prefix_sixteen_upper ((n : ℕ) - 1) using 1
  all_goals norm_num [S]

end RubiksSnake.FinalResults

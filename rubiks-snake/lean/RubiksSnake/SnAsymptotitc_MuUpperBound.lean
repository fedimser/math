import RubiksSnake.SmallCounts
import RubiksSnake.SnAsymptotic_MuExistence

/-!
# Upper bounds on the Rubik's Snake growth constant

A single finite count bounds the Fekete limit from above. Keeping the block
length and comparison base as parameters means that improving the upper bound
only requires a new exact count and a new finite power inequality.
-/

open Filter Set Topology

namespace RubiksSnake

noncomputable section

/-- A finite count bounded by `q ^ k` certifies that the growth constant is at
most `q`.  This is the parameter-adjustment point for future sharper bounds. -/
theorem snakeGrowthConstant_le_of_countValidFormulas_le_pow
    (k : ℕ) (hk : k ≠ 0) (q : ℝ) (hq : 0 < q)
    (hcount : (countValidFormulas k : ℝ) ≤ q ^ k) :
    snakeGrowthConstant ≤ q := by
  rw [snakeGrowthConstant, ← Real.exp_log hq]
  apply Real.exp_le_exp.mpr
  calc
    logValidFormulaCount_subadditive.lim ≤
        logValidFormulaCount k / (k : ℝ) :=
      logValidFormulaCount_subadditive.lim_le_div
        logValidFormulaCount_div_bddBelow hk
    _ ≤ Real.log (q ^ k) / (k : ℝ) := by
      apply div_le_div_of_nonneg_right
      · exact Real.strictMonoOn_log.monotoneOn
          (by
            change (0 : ℝ) < countValidFormulas k
            exact_mod_cast countValidFormulas_pos k)
          (pow_pos hq k) hcount
      · positivity
    _ = Real.log q := by
      rw [Real.log_pow]
      field_simp

/-- A version of the finite certificate in which the exact integer count is
separated from the elementary real power comparison. -/
theorem snakeGrowthConstant_le_of_exact_count
    (k count : ℕ) (hk : k ≠ 0) (q : ℝ) (hq : 0 < q)
    (hexact : countValidFormulas k = count)
    (hpower : (count : ℝ) ≤ q ^ k) :
    snakeGrowthConstant ≤ q := by
  apply snakeGrowthConstant_le_of_countValidFormulas_le_pow k hk q hq
  simpa [hexact] using hpower

end

def boundHelper (n Sn : Nat) : Float :=
  Float.pow Sn.toFloat (1.0 / (n.toFloat - 1.0))
#eval boundHelper 7 3384

/-- The Rubik's Snake growth constant is at most `3.9`. -/
theorem snakeGrowthConstant_le_39_div_10 :
    snakeGrowthConstant ≤ 3.875:= by
  apply snakeGrowthConstant_le_of_exact_count 6 3384 (by norm_num) 3.875 (by norm_num)
  · simpa [S] using S7_value
  · norm_num

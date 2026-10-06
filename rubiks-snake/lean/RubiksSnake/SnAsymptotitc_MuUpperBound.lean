import RubiksSnake.WindowUpperBound
import RubiksSnake.WindowSevenUpperBound
import RubiksSnake.ForbiddenPrefixUpperBound

/-!
# Upper bounds on the Rubik's Snake growth constant

A compressed collision-prefix certificate gives `mu <= 3.675`.
Its pointwise bound passes to the logarithmic growth limit. The seven-symbol
`3.704` and five-symbol `3.7202` window certificates remain available.
-/

open Filter Set Topology

namespace RubiksSnake

noncomputable section

/-- A positive-prefactor pointwise upper bound passes to the growth limit. -/
theorem snakeGrowthConstant_le_of_pointwise
    (C q : ℝ) (hC : 0 < C) (hq : 0 < q)
    (hcount : ∀ k : ℕ, (countValidFormulas k : ℝ) ≤ C * q ^ k) :
    snakeGrowthConstant ≤ q := by
  have hzero : Tendsto (fun k : ℕ => Real.log C / (k : ℝ)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  have hupper :
      Tendsto (fun k : ℕ => Real.log C / (k : ℝ) + Real.log q)
        atTop (𝓝 (Real.log q)) := by
    simpa using hzero.add_const (Real.log q)
  have hlog : Real.log snakeGrowthConstant ≤ Real.log q := by
    apply le_of_tendsto_of_tendsto tendsto_logValidFormulaCount_div hupper
    filter_upwards [eventually_ge_atTop 1] with k hk
    have hk0 : (k : ℝ) ≠ 0 := by
      exact_mod_cast (show k ≠ 0 by omega)
    have hcompare : Real.log (countValidFormulas k) ≤ Real.log (C * q ^ k) :=
      Real.strictMonoOn_log.monotoneOn
        (by
          change (0 : ℝ) < countValidFormulas k
          exact_mod_cast countValidFormulas_pos k)
        (mul_pos hC (pow_pos hq k)) (hcount k)
    rw [Real.log_mul hC.ne' (pow_pos hq k).ne', Real.log_pow] at hcompare
    have hdiv := div_le_div_of_nonneg_right hcompare (Nat.cast_nonneg k : (0 : ℝ) ≤ k)
    have heq :
        (Real.log C + (k : ℝ) * Real.log q) / (k : ℝ) =
          Real.log C / (k : ℝ) + Real.log q := by
      field_simp
    simpa only [heq, logValidFormulaCount] using hdiv
  rw [← Real.exp_log snakeGrowthConstant_pos, ← Real.exp_log hq]
  exact Real.exp_le_exp.mpr hlog

/-- The compressed collision-prefix certificate gives `mu <= 3.675`. -/
theorem snakeGrowthConstant_le_147_div_40 :
    snakeGrowthConstant ≤ (147 / 40 : ℝ) := by
  exact snakeGrowthConstant_le_of_pointwise (9 / 2) (147 / 40)
    (by norm_num) (by norm_num) countValidFormulas_le_forbidden_prefix_upper

/-- The exact seven-symbol window certificate gives `mu <= 3.704`. -/
theorem snakeGrowthConstant_le_463_div_125 :
    snakeGrowthConstant ≤ (463 / 125 : ℝ) := by
  exact snakeGrowthConstant_le_of_pointwise (8 / 3) (463 / 125)
    (by norm_num) (by norm_num) countValidFormulas_le_seven_window_upper

/-- The exact five-symbol window certificate gives `mu <= 3.7202`. -/
theorem snakeGrowthConstant_le_18601_div_5000 :
    snakeGrowthConstant ≤ (18601 / 5000 : ℝ) := by
  exact snakeGrowthConstant_le_of_pointwise (9 / 4) (18601 / 5000)
    (by norm_num) (by norm_num) countValidFormulas_le_window_upper

/-- In particular, the growth constant is at most `3.73`. -/
theorem snakeGrowthConstant_le_373_div_100 :
    snakeGrowthConstant ≤ (373 / 100 : ℝ) := by
  exact snakeGrowthConstant_le_18601_div_5000.trans (by norm_num)

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

/-- Legacy theorem, with its historical name retained for compatibility.
The statement is the old `3.875` bound, not `3.9`; the sharper window bound
is `snakeGrowthConstant_le_18601_div_5000`. -/
theorem snakeGrowthConstant_le_39_div_10 :
    snakeGrowthConstant ≤ 3.875 := by
  apply snakeGrowthConstant_le_of_exact_count 6 3384 (by norm_num) 3.875 (by norm_num)
  · simpa [S] using S7_value
  · norm_num

import RubiksSnake.SmallCounts
import RubiksSnake.ReversalTransform
import RubiksSnake.SnAsymptoticEasy
import Mathlib.Analysis.Subadditive
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Existence of the Rubik's Snake growth constant

This file proves that the exponential growth rate of the number of valid
Rubik's Snake formulas exists.  The key combinatorial fact is that validity is
hereditary for prefixes and suffixes.  Consequently, the counting sequence is
submultiplicative, so its logarithm is subadditive and Fekete's lemma applies.
-/

open Filter Set Topology

namespace RubiksSnake

noncomputable section

lemma mem_validRotationLists_take {k : ℕ} {rs : List Rotation}
    (hrs : rs ∈ validRotationLists k) (m : ℕ) :
    rs.take m ∈ validRotationLists (min m k) := by
  induction k generalizing rs with
  | zero =>
      have hrs_nil : rs = [] := by simpa [validRotationLists] using hrs
      subst rs
      simp [validRotationLists]
  | succ k ih =>
      simp only [validRotationLists, List.mem_flatMap, List.mem_filterMap] at hrs
      obtain ⟨pre, hpre, r, hr, hchild⟩ := hrs
      split at hchild
      · rename_i hcan
        cases hchild
        by_cases hm : m ≤ k
        · rw [Nat.min_eq_left (hm.trans (Nat.le_succ k))]
          rw [List.take_append_of_le_length]
          · simpa [Nat.min_eq_left hm] using ih hpre
          · simpa [validRotationLists_length k pre hpre] using hm
        · have hkm : k + 1 ≤ m := by omega
          rw [Nat.min_eq_right hkm]
          have htake : (pre ++ [r]).take m = pre ++ [r] :=
            (List.take_eq_self_iff _).mpr (by
              simp [validRotationLists_length k pre hpre, hkm])
          rw [htake]
          simp only [validRotationLists, List.mem_flatMap, List.mem_filterMap]
          exact ⟨pre, hpre, r, hr, by simp [hcan]⟩
      · simp at hchild

lemma validList_take (rs : List Rotation) (m : ℕ) (hrs : ValidList rs) :
    ValidList (rs.take m) := by
  have hmem : rs ∈ validRotationLists rs.length :=
    (mem_validRotationLists rs).mpr hrs
  have htake := mem_validRotationLists_take hmem m
  exact validRotationLists_valid _ _ htake

lemma validList_drop (rs : List Rotation) (m : ℕ) (hrs : ValidList rs) :
    ValidList (rs.drop m) := by
  by_cases hm : m ≤ rs.length
  · have hreverse : ValidList rs.reverse := by
      simpa [ValidList] using (collisionFree_reverse rs).mpr hrs
    have htake : ValidList (rs.reverse.take (rs.length - m)) :=
      validList_take rs.reverse (rs.length - m) hreverse
    have htake_reverse :
        ValidList (rs.reverse.take (rs.length - m)).reverse := by
      simpa [ValidList] using
        (collisionFree_reverse (rs.reverse.take (rs.length - m))).mpr htake
    simpa [List.take_reverse, Nat.sub_sub_self hm] using htake_reverse
  · have : rs.drop m = [] := List.drop_eq_nil_iff.mpr (Nat.le_of_not_ge hm)
    rw [this]
    native_decide

def splitValidRotationList (m n : ℕ) :
    {rs : List Rotation // rs ∈ validRotationLists (m + n)} →
      {rs : List Rotation // rs ∈ validRotationLists m} ×
        {rs : List Rotation // rs ∈ validRotationLists n} :=
  fun rs =>
    let hlen := validRotationLists_length (m + n) rs.1 rs.2
    ⟨⟨rs.1.take m, by
        simpa [Nat.min_eq_left (Nat.le_add_right m n)] using
          mem_validRotationLists_take rs.2 m⟩,
      ⟨rs.1.drop m, by
        have hmem := (mem_validRotationLists (rs.1.drop m)).mpr
          (validList_drop rs.1 m (validRotationLists_valid _ _ rs.2))
        simpa [List.length_drop, hlen] using hmem⟩⟩

lemma splitValidRotationList_injective (m n : ℕ) :
    Function.Injective (splitValidRotationList m n) := by
  intro a b hab
  apply Subtype.ext
  have htake : a.1.take m = b.1.take m :=
    congrArg (fun p => p.1.1) hab
  have hdrop : a.1.drop m = b.1.drop m :=
    congrArg (fun p => p.2.1) hab
  calc
    a.1 = a.1.take m ++ a.1.drop m := (List.take_append_drop m a.1).symm
    _ = b.1.take m ++ b.1.drop m := by rw [htake, hdrop]
    _ = b.1 := List.take_append_drop m b.1

/-- Valid rotation formulas are submultiplicative in their word length. -/
theorem countValidFormulas_submultiplicative (m n : ℕ) :
    countValidFormulas (m + n) ≤
      countValidFormulas m * countValidFormulas n := by
  unfold countValidFormulas countFormulas
  rw [Nat.card_congr (validFormulaEquiv (m + n))]
  rw [Nat.card_congr (validFormulaEquiv m)]
  rw [Nat.card_congr (validFormulaEquiv n)]
  rw [← Nat.card_prod]
  exact Nat.card_le_card_of_injective
    (splitValidRotationList m n) (splitValidRotationList_injective m n)

lemma countValidFormulas_pos (k : ℕ) : 0 < countValidFormulas k := by
  have hle : 2 ^ k ≤ countValidFormulas k := by
    unfold countValidFormulas countFormulas
    calc
      2 ^ k = Nat.card (Fin k → Bool) := by simp
      _ ≤ Nat.card {w : Formula k // Valid w} :=
        Nat.card_le_card_of_injective
          (increasingValidFormula k) (increasingValidFormula_injective k)
  exact lt_of_lt_of_le (by positivity) hle

/-- The logarithm of the valid-formula count. -/
def logValidFormulaCount (k : ℕ) : ℝ :=
  Real.log (countValidFormulas k)

lemma logValidFormulaCount_nonneg (k : ℕ) :
    0 ≤ logValidFormulaCount k := by
  apply Real.log_nonneg
  exact_mod_cast (countValidFormulas_pos k)

/-- Subadditivity of the logarithmic counting sequence. -/
theorem logValidFormulaCount_subadditive :
    Subadditive logValidFormulaCount := by
  intro m n
  unfold logValidFormulaCount
  calc
    Real.log (countValidFormulas (m + n)) ≤
        Real.log (countValidFormulas m * countValidFormulas n) := by
      exact Real.strictMonoOn_log.monotoneOn
        (by
          change (0 : ℝ) < countValidFormulas (m + n)
          exact_mod_cast countValidFormulas_pos (m + n))
        (by
          change (0 : ℝ) < countValidFormulas m * countValidFormulas n
          exact mul_pos
            (by exact_mod_cast countValidFormulas_pos m)
            (by exact_mod_cast countValidFormulas_pos n))
        (by exact_mod_cast countValidFormulas_submultiplicative m n)
    _ = Real.log (countValidFormulas m) +
        Real.log (countValidFormulas n) := by
      rw [Real.log_mul]
      · exact_mod_cast (countValidFormulas_pos m).ne'
      · exact_mod_cast (countValidFormulas_pos n).ne'

lemma logValidFormulaCount_div_bddBelow :
    BddBelow (range fun k : ℕ => logValidFormulaCount k / k) := by
  refine ⟨0, ?_⟩
  rintro _ ⟨k, rfl⟩
  exact div_nonneg (logValidFormulaCount_nonneg k) (Nat.cast_nonneg k)

/-- The exponential growth constant for valid Rubik's Snake formulas. -/
def snakeGrowthConstant : ℝ :=
  Real.exp logValidFormulaCount_subadditive.lim

lemma tendsto_logValidFormulaCount_div :
    Tendsto (fun k : ℕ => logValidFormulaCount k / k) atTop
      (𝓝 (Real.log snakeGrowthConstant)) := by
  have ht := logValidFormulaCount_subadditive.tendsto_lim
    logValidFormulaCount_div_bddBelow
  simpa [snakeGrowthConstant] using ht

lemma snakeGrowthConstant_pos : 0 < snakeGrowthConstant :=
  Real.exp_pos _

/-- Submultiplicativity makes the limiting exponential rate a pointwise
lower bound, with no prefactor or exceptional lengths. -/
theorem snakeGrowthConstant_pow_le_countValidFormulas (k : ℕ) :
    snakeGrowthConstant ^ k ≤ (countValidFormulas k : ℝ) := by
  by_cases hk : k = 0
  · subst k
    have h0 : countValidFormulas 0 = 1 := by simpa [S] using S1_value
    norm_num [h0]
  · have hkpos : (0 : ℝ) < k := by exact_mod_cast Nat.pos_of_ne_zero hk
    have hlim := logValidFormulaCount_subadditive.lim_le_div
      logValidFormulaCount_div_bddBelow hk
    have hmul :
        (k : ℝ) * logValidFormulaCount_subadditive.lim ≤ logValidFormulaCount k := by
      simpa [mul_comm] using (le_div_iff₀ hkpos).mp hlim
    rw [snakeGrowthConstant, ← Real.exp_nat_mul]
    calc
      Real.exp ((k : ℝ) * logValidFormulaCount_subadditive.lim) ≤
          Real.exp (logValidFormulaCount k) := Real.exp_le_exp.mpr hmul
      _ = countValidFormulas k := by
        rw [logValidFormulaCount, Real.exp_log]
        exact_mod_cast countValidFormulas_pos k

theorem snakeGrowthConstant_ge_of_pointwise
    (C q : ℝ) (hC : 0 < C) (hq : 0 < q)
    (hcount : ∀ k : ℕ, C * q ^ k ≤ (countValidFormulas k : ℝ)) :
    q ≤ snakeGrowthConstant := by
  have hzero : Tendsto (fun k : ℕ => Real.log C / (k : ℝ)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  have hlower :
      Tendsto (fun k : ℕ => Real.log C / (k : ℝ) + Real.log q)
        atTop (𝓝 (Real.log q)) := by
    simpa using hzero.add_const (Real.log q)
  have hlog : Real.log q ≤ Real.log snakeGrowthConstant := by
    apply le_of_tendsto_of_tendsto hlower tendsto_logValidFormulaCount_div
    filter_upwards [eventually_ge_atTop 1] with k hk
    have hk0 : (k : ℝ) ≠ 0 := by
      exact_mod_cast (show k ≠ 0 by omega)
    have hcompare : Real.log (C * q ^ k) ≤ Real.log (countValidFormulas k) :=
      Real.strictMonoOn_log.monotoneOn
        (mul_pos hC (pow_pos hq k))
        (by
          change (0 : ℝ) < countValidFormulas k
          exact_mod_cast countValidFormulas_pos k) (hcount k)
    rw [Real.log_mul hC.ne' (pow_pos hq k).ne', Real.log_pow] at hcompare
    have hdiv := div_le_div_of_nonneg_right hcompare (Nat.cast_nonneg k : (0 : ℝ) ≤ k)
    have heq :
        (Real.log C + (k : ℝ) * Real.log q) / (k : ℝ) =
          Real.log C / (k : ℝ) + Real.log q := by
      field_simp
    simpa only [heq, logValidFormulaCount] using hdiv
  rw [← Real.exp_log hq, ← Real.exp_log snakeGrowthConstant_pos]
  exact Real.exp_le_exp.mpr hlog

lemma tendsto_countValidFormulas_rpow :
    Tendsto
      (fun k : ℕ => (countValidFormulas k : ℝ) ^ (1 / (k : ℝ)))
      atTop (𝓝 snakeGrowthConstant) := by
  have ht := (Real.continuous_exp.tendsto _).comp
    (logValidFormulaCount_subadditive.tendsto_lim
      logValidFormulaCount_div_bddBelow)
  change Tendsto _ atTop
    (𝓝 (Real.exp logValidFormulaCount_subadditive.lim))
  refine ht.congr' (Filter.Eventually.of_forall fun k => ?_)
  change Real.exp (logValidFormulaCount k / (k : ℝ)) =
    (countValidFormulas k : ℝ) ^ (1 / (k : ℝ))
  rw [Real.rpow_def_of_pos]
  · congr 2
    simp [logValidFormulaCount, div_eq_mul_inv]
  · exact_mod_cast countValidFormulas_pos k

/-- The paper's growth constant `μ` exists: the `n`th roots of the number of
valid length-`n` rotation formulas converge to a positive real number. -/
theorem SnAsymptotic_MuExistence :
    ∃ μ : ℝ, 0 < μ ∧
      Tendsto
        (fun n : ℕ => (countValidFormulas n : ℝ) ^ (1 / (n : ℝ)))
        atTop (𝓝 μ) :=
  ⟨snakeGrowthConstant, snakeGrowthConstant_pos,
    tendsto_countValidFormulas_rpow⟩

end

end RubiksSnake

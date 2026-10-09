import RubiksSnake.SnAsymptotic.SmallCounts
import RubiksSnake.OtherSequences.ReversalTransform
import RubiksSnake.SnAsymptotic.SnAsymptoticEasy
import Mathlib.Analysis.Subadditive
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Submultiplicative Rubik's Snake counts

Validity is hereditary for prefixes and suffixes, so valid-formula counts are
submultiplicative. Fekete's inequality then removes a fixed positive prefactor
from any pointwise exponential lower bound. No growth constant is defined here.
-/

open Filter Set Topology

namespace RubiksSnake

noncomputable section

/-- Truncating an enumerated valid word gives an enumerated valid prefix of the truncated length. -/
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

/-- A prefix of a collision-free rotation word is again valid. -/
lemma validList_take (rs : List Rotation) (m : ℕ) (hrs : ValidList rs) :
    ValidList (rs.take m) := by
  have hmem : rs ∈ validRotationLists rs.length :=
    (mem_validRotationLists rs).mpr hrs
  have htake := mem_validRotationLists_take hmem m
  exact validRotationLists_valid _ _ htake

/-- A suffix remains valid when interpreted in the standard initial frame; reversal reduces
this to prefix validity. -/
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

/-- Split a valid word of length `m + n` into separately valid prefix and suffix words. -/
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

/-- The prefix and suffix determine the original word, giving the injection behind submultiplicativity. -/
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

/-- There is at least one valid formula at every length; the increasing construction gives `2^k`. -/
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

/-- The logarithmic count is nonnegative because every integer formula count is at least one. -/
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

/-- Normalized logarithmic counts are bounded below by zero, as required by Fekete's lemma. -/
lemma logValidFormulaCount_div_bddBelow :
    BddBelow (range fun k : ℕ => logValidFormulaCount k / k) := by
  refine ⟨0, ?_⟩
  rintro _ ⟨k, rfl⟩
  exact div_nonneg (logValidFormulaCount_nonneg k) (Nat.cast_nonneg k)

/-- For a submultiplicative count, any uniform exponential lower bound with a
positive prefactor implies the same pointwise base with prefactor one. -/
theorem pow_le_countValidFormulas_of_pointwise
    (C q : ℝ) (hC : 0 < C) (hq : 0 < q)
    (hcount : ∀ k : ℕ, C * q ^ k ≤ (countValidFormulas k : ℝ))
    (k : ℕ) : q ^ k ≤ (countValidFormulas k : ℝ) := by
  have hzero : Tendsto (fun k : ℕ => Real.log C / (k : ℝ)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  have hlower :
      Tendsto (fun k : ℕ => Real.log C / (k : ℝ) + Real.log q)
        atTop (𝓝 (Real.log q)) := by
    simpa using hzero.add_const (Real.log q)
  have hlog : Real.log q ≤ logValidFormulaCount_subadditive.lim := by
    apply le_of_tendsto_of_tendsto hlower
      (logValidFormulaCount_subadditive.tendsto_lim
        logValidFormulaCount_div_bddBelow)
    filter_upwards [eventually_ge_atTop 1] with k hk
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
  by_cases hk : k = 0
  · subst k
    have h0 : countValidFormulas 0 = 1 := by
      simpa [S] using (show S 1 = 1 by snake_decide)
    norm_num [h0]
  · have hkpos : (0 : ℝ) < k := by exact_mod_cast Nat.pos_of_ne_zero hk
    have hlim := logValidFormulaCount_subadditive.lim_le_div
      logValidFormulaCount_div_bddBelow hk
    have hmul : (k : ℝ) * Real.log q ≤ logValidFormulaCount k := by
      apply (mul_le_mul_of_nonneg_left (hlog.trans hlim) hkpos.le).trans_eq
      field_simp
    rw [← Real.exp_log hq, ← Real.exp_nat_mul]
    calc
      Real.exp ((k : ℝ) * Real.log q) ≤ Real.exp (logValidFormulaCount k) :=
        Real.exp_le_exp.mpr hmul
      _ = countValidFormulas k := by
        rw [logValidFormulaCount, Real.exp_log]
        exact_mod_cast countValidFormulas_pos k

end

end RubiksSnake

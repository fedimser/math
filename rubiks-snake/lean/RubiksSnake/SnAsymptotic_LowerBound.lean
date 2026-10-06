import RubiksSnake.SnAsymptoticEasy
import RubiksSnake.SlabBlocks
import RubiksSnake.RecordBlocks
import RubiksSnake.WeightedRecordLowerBound
import RubiksSnake.SnAsymptotic_MuExistence

import Mathlib.Tactic

/-!
# Certified pointwise lower bounds

The plane-block construction in the paper gives `4, 8, 16, 24, 40, 72`
blocks of lengths `2, ..., 7`. Concatenating these blocks gives the renewal
sequence below. `SlabBlocks` verifies the geometry and unique decoding of
these blocks, so the renewal counts inject into valid rotation formulas.
The finite checks through index eight, together with the renewal polynomial
inequality, certify a lower exponential rate of `3.1`.

Allowing adjacent blocks to share a plane gives the stronger rate `3.16`.
`RecordBlocks` checks the occupied interfaces and establishes uniform lower
counts for legal continuations. Weighting those same interfaces improves
the rate to `3.193`. Submultiplicativity removes the finite induction
prefactor from the final bounds.
-/

namespace RubiksSnake

lemma renewal_ge_of_finite_certificate
    (f : ℕ → ℕ) (L start : ℕ) (coeff : Fin L → ℕ)
    (a q : ℝ) (ha : 0 ≤ a) (hq : 0 ≤ q)
    (hpoly : q ^ L ≤ ∑ j : Fin L, (coeff j : ℝ) * q ^ (L - (j.val + 1)))
    (hbase : ∀ k, start ≤ k → k < start + L → a * q ^ k ≤ f k)
    (hrec : ∀ k, start + L ≤ k →
      (∑ j : Fin L, (coeff j : ℝ) * (f (k - (j.val + 1)) : ℝ)) ≤ f k)
    (k : ℕ) (hk : start ≤ k) :
    a * q ^ k ≤ f k := by
  induction k using Nat.strong_induction_on with
  | h k ih =>
      by_cases hsmall : k < start + L
      · exact hbase k hk hsmall
      · have hlarge : start + L ≤ k := by omega
        have hpow (j : Fin L) :
            q ^ (k - L) * q ^ (L - (j.val + 1)) = q ^ (k - (j.val + 1)) := by
          rw [← pow_add]
          congr 1
          omega
        calc
          a * q ^ k = (a * q ^ (k - L)) * q ^ L := by
            rw [mul_assoc, ← pow_add, Nat.sub_add_cancel (by omega : L ≤ k)]
          _ ≤ (a * q ^ (k - L)) *
              (∑ j : Fin L, (coeff j : ℝ) * q ^ (L - (j.val + 1))) :=
            mul_le_mul_of_nonneg_left hpoly (mul_nonneg ha (pow_nonneg hq _))
          _ = ∑ j : Fin L, (coeff j : ℝ) * (a * q ^ (k - (j.val + 1))) := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro j _
            rw [show (a * q ^ (k - L)) * ((coeff j : ℝ) * q ^ (L - (j.val + 1))) =
                (coeff j : ℝ) * (a * (q ^ (k - L) * q ^ (L - (j.val + 1)))) by ring,
              hpow]
          _ ≤ ∑ j : Fin L, (coeff j : ℝ) * (f (k - (j.val + 1)) : ℝ) := by
            apply Finset.sum_le_sum
            intro j _
            exact mul_le_mul_of_nonneg_left
              (ih (k - (j.val + 1)) (by omega) (by omega)) (Nat.cast_nonneg _)
          _ ≤ f k := hrec k hlarge

/-- Number of words in the renewal language with total block length `n`. -/
def planeRenewal (n : ℕ) : ℕ :=
  if n = 0 then 1
  else
    (if 2 ≤ n then planeBlockCount 2 * planeRenewal (n - 2) else 0) +
    (if 3 ≤ n then planeBlockCount 3 * planeRenewal (n - 3) else 0) +
    (if 4 ≤ n then planeBlockCount 4 * planeRenewal (n - 4) else 0) +
    (if 5 ≤ n then planeBlockCount 5 * planeRenewal (n - 5) else 0) +
    (if 6 ≤ n then planeBlockCount 6 * planeRenewal (n - 6) else 0) +
    (if 7 ≤ n then planeBlockCount 7 * planeRenewal (n - 7) else 0)
termination_by n
decreasing_by all_goals omega

lemma planeRenewal_zero : planeRenewal 0 = 1 := by
  rw [planeRenewal]
  simp

lemma planeRenewal_rec (n : ℕ) (hn : 7 ≤ n) :
    planeRenewal n =
      4 * planeRenewal (n - 2) +
      8 * planeRenewal (n - 3) +
      16 * planeRenewal (n - 4) +
      24 * planeRenewal (n - 5) +
      40 * planeRenewal (n - 6) +
      72 * planeRenewal (n - 7) := by
  have h2 : 2 ≤ n := by omega
  have h3 : 3 ≤ n := by omega
  have h4 : 4 ≤ n := by omega
  have h5 : 5 ≤ n := by omega
  have h6 : 6 ≤ n := by omega
  have hn0 : n ≠ 0 := by omega
  rw [planeRenewal]
  simp [planeBlockCount, hn0, h2, h3, h4, h5, h6, hn]

private lemma planeRenewal_two : planeRenewal 2 = 4 := by native_decide
private lemma planeRenewal_three : planeRenewal 3 = 8 := by native_decide
private lemma planeRenewal_four : planeRenewal 4 = 32 := by native_decide
private lemma planeRenewal_five : planeRenewal 5 = 88 := by native_decide
private lemma planeRenewal_six : planeRenewal 6 = 296 := by native_decide
private lemma planeRenewal_seven : planeRenewal 7 = 904 := by native_decide
private lemma planeRenewal_eight : planeRenewal 8 = 2752 := by native_decide

private lemma planeRenewal_ge_aux (k : ℕ) (hk : 2 ≤ k) :
    (1 / 4 : ℝ) * ((31 / 10 : ℝ) ^ k) ≤ planeRenewal k := by
  apply renewal_ge_of_finite_certificate planeRenewal 7 2
    (fun j => planeBlockCount (j.val + 1)) (1 / 4) (31 / 10)
    (by norm_num) (by norm_num) ?_ ?_ ?_ k hk
  · norm_num [Fin.sum_univ_succ, planeBlockCount]
  · intro k hk hlo
    interval_cases k <;>
      norm_num [planeRenewal_two, planeRenewal_three, planeRenewal_four,
        planeRenewal_five, planeRenewal_six, planeRenewal_seven, planeRenewal_eight]
  · intro k hk
    rw [planeRenewal_rec k (by omega)]
    norm_num [Fin.sum_univ_succ, planeBlockCount]
    linarith

/-- The six certified block counts imply the renewal lower bound with base
`3.1`; only indices `2, ..., 8` are checked computationally. -/
theorem planeRenewal_ge (k : ℕ) (hk : 2 ≤ k) :
    (1 / 4 : ℝ) * ((31 / 10 : ℝ) ^ k) ≤ planeRenewal k :=
  planeRenewal_ge_aux k hk

lemma planeLanguage_length_eq_planeRenewal (n : ℕ) :
    (planeLanguage n).length = planeRenewal n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hn : n = 0
      · subst n
        rw [planeLanguage_zero, planeRenewal_zero]
        rfl
      · rw [planeLanguage_length_rec n hn, planeRenewal, if_neg hn]
        simp only [planeBlockCount,
          ih (n - 2) (by omega), ih (n - 3) (by omega),
          ih (n - 4) (by omega), ih (n - 5) (by omega),
          ih (n - 6) (by omega), ih (n - 7) (by omega)]

/-- The final block's outgoing face already specifies the last wedge;
no additional terminal wedge is needed. -/
theorem planeRenewal_le_countValidFormulas (k : ℕ) :
    planeRenewal k ≤ countValidFormulas k := by
  rw [← planeLanguage_length_eq_planeRenewal]
  exact planeLanguage_length_le k

theorem countValidFormulas_lower_bound_31_div_10_with_prefactor (k : ℕ) :
    (1 / 4 : ℝ) * (31 / 10 : ℝ) ^ k ≤ countValidFormulas k := by
  by_cases hk : 2 ≤ k
  · exact (planeRenewal_ge k hk).trans
      (by exact_mod_cast planeRenewal_le_countValidFormulas k)
  · obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    · have h0 : countValidFormulas 0 = 1 := by simpa [S] using S1_value
      norm_num [h0]
    · have h1 : countValidFormulas 1 = 4 := by simpa [S] using S2_value
      norm_num [h1]

/-- Submultiplicativity removes the prefactor from the constructive estimate. -/
theorem countValidFormulas_lower_bound_31_div_10 (k : ℕ) :
   (31 / 10 : ℝ) ^ k ≤ countValidFormulas k := by
 have hmu : (31 / 10 : ℝ) ≤ snakeGrowthConstant :=
   snakeGrowthConstant_ge_of_pointwise (1 / 4) (31 / 10)
     (by norm_num) (by norm_num) countValidFormulas_lower_bound_31_div_10_with_prefactor
 exact (pow_le_pow_left₀ (by norm_num) hmu k).trans
   (snakeGrowthConstant_pow_le_countValidFormulas k)

/-- A uniform pointwise lower bound for every positive snake length. -/
theorem Sn_lower_bound_31_div_10 (n : ℕ+) :
   (31 / 10 : ℝ) ^ ((n : ℕ) - 1) ≤ S n :=
 countValidFormulas_lower_bound_31_div_10 ((n : ℕ) - 1)

private lemma recordRenewal_base :
   ∀ j : Fin 7, 79 ^ (j.val + 2) ≤ 4 * 25 ^ (j.val + 2) * recordRenewal (j.val + 2) := by
 native_decide

lemma recordRenewal_ge (k : ℕ) (hk : 2 ≤ k) :
   (1 / 4 : ℝ) * (79 / 25 : ℝ) ^ k ≤ recordRenewal k := by
 apply renewal_ge_of_finite_certificate recordRenewal 7 2
   (fun j => recordLowerCount (j.val + 1)) (1 / 4) (79 / 25)
   (by norm_num) (by norm_num) ?_ ?_ ?_ k hk
 · norm_num [Fin.sum_univ_succ, recordLowerCount]
 · intro k hk hlo
   have hnat := recordRenewal_base ⟨k - 2, by omega⟩
   have hk2 : k - 2 + 2 = k := by omega
   simp only [hk2] at hnat
   have hreal : (79 : ℝ) ^ k ≤ 4 * (25 : ℝ) ^ k * (recordRenewal k : ℝ) := by
     exact_mod_cast hnat
   calc
     (1 / 4 : ℝ) * (79 / 25 : ℝ) ^ k = (79 : ℝ) ^ k / (4 * (25 : ℝ) ^ k) := by
       rw [div_pow]
       ring
     _ ≤ recordRenewal k := (div_le_iff₀ (by positivity)).mpr (by nlinarith)
 · intro k hk
   have h2 : 2 ≤ k := by omega
   have h3 : 3 ≤ k := by omega
   have h4 : 4 ≤ k := by omega
   have h5 : 5 ≤ k := by omega
   have h6 : 6 ≤ k := by omega
   have h7 : 7 ≤ k := by omega
   rw [recordRenewal, if_neg (by omega : k ≠ 0)]
   norm_num [Fin.sum_univ_succ, recordLowerCount, h2, h3, h4, h5, h6, h7]
   linarith

theorem countValidFormulas_lower_bound_79_div_25_with_prefactor (k : ℕ) :
   (1 / 4 : ℝ) * (79 / 25 : ℝ) ^ k ≤ countValidFormulas k := by
 by_cases hk : 2 ≤ k
 · exact (recordRenewal_ge k hk).trans
     (by exact_mod_cast recordRenewal_le_countValidFormulas k)
 · obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
   · have h0 : countValidFormulas 0 = 1 := by simpa [S] using S1_value
     norm_num [h0]
   · have h1 : countValidFormulas 1 = 4 := by simpa [S] using S2_value
     norm_num [h1]

theorem countValidFormulas_lower_bound_79_div_25 (k : ℕ) :
   (79 / 25 : ℝ) ^ k ≤ countValidFormulas k := by
 have hmu : (79 / 25 : ℝ) ≤ snakeGrowthConstant :=
   snakeGrowthConstant_ge_of_pointwise (1 / 4) (79 / 25)
     (by norm_num) (by norm_num) countValidFormulas_lower_bound_79_div_25_with_prefactor
 exact (pow_le_pow_left₀ (by norm_num) hmu k).trans
   (snakeGrowthConstant_pow_le_countValidFormulas k)

theorem Sn_lower_bound_79_div_25 (n : ℕ+) :
   (79 / 25 : ℝ) ^ ((n : ℕ) - 1) ≤ S n :=
 countValidFormulas_lower_bound_79_div_25 ((n : ℕ) - 1)

theorem countValidFormulas_lower_bound_3193_div_1000 (k : ℕ) :
   (3193 / 1000 : ℝ) ^ k ≤ countValidFormulas k := by
 have hc : 0 < (1 / (RecordWeighting.baseScale : ℝ)) :=
   one_div_pos.mpr (Nat.cast_pos.mpr RecordWeighting.baseScale_pos)
 have hmu : (3193 / 1000 : ℝ) ≤ snakeGrowthConstant :=
   snakeGrowthConstant_ge_of_pointwise (1 / (RecordWeighting.baseScale : ℝ))
     (3193 / 1000) hc (by norm_num)
     countValidFormulas_lower_bound_3193_div_1000_with_prefactor
 exact (pow_le_pow_left₀ (by norm_num) hmu k).trans
   (snakeGrowthConstant_pow_le_countValidFormulas k)

theorem Sn_lower_bound_3193_div_1000 (n : ℕ+) :
   (3193 / 1000 : ℝ) ^ ((n : ℕ) - 1) ≤ S n :=
 countValidFormulas_lower_bound_3193_div_1000 ((n : ℕ) - 1)

end RubiksSnake

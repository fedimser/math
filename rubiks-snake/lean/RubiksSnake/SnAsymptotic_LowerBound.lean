import RubiksSnake.SnAsymptoticEasy

import Mathlib.Tactic

/-!
# A certified pointwise lower bound

The plane-block construction in the paper gives `4, 8, 16, 24, 40, 72`
blocks of lengths `2, ..., 7`. Concatenating these blocks gives the renewal
sequence below. The finite checks through index eight, together with the
renewal polynomial inequality, certify a lower exponential rate of `3.1`.

To improve the bound, extend `planeBlockCount` with more certified block counts
and enlarge the finite base window in `planeRenewal_ge`.
-/

namespace RubiksSnake

/-- Certified numbers of width-zero plane blocks, indexed by block length. -/
def planeBlockCount : ℕ → ℕ
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 24
  | 6 => 40
  | 7 => 72
  | _ => 0

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
  induction k using Nat.strong_induction_on with
  | h k ih =>
      by_cases hk₉ : k < 9
      · interval_cases k <;>
          norm_num [planeRenewal_two, planeRenewal_three, planeRenewal_four,
            planeRenewal_five, planeRenewal_six, planeRenewal_seven,
            planeRenewal_eight]
      · have hk9 : 9 ≤ k := by omega
        have h2 := ih (k - 2) (by omega) (by omega)
        have h3 := ih (k - 3) (by omega) (by omega)
        have h4 := ih (k - 4) (by omega) (by omega)
        have h5 := ih (k - 5) (by omega) (by omega)
        have h6 := ih (k - 6) (by omega) (by omega)
        have h7 := ih (k - 7) (by omega) (by omega)
        rw [planeRenewal_rec k (by omega)]
        norm_num only [Nat.cast_add, Nat.cast_mul]
        have hpow (j : ℕ) (hj : j ≤ 7) :
            (31 / 10 : ℝ) ^ (k - j) =
              (31 / 10 : ℝ) ^ (k - 7) * (31 / 10 : ℝ) ^ (7 - j) := by
          rw [← pow_add]
          congr 1
          omega
        rw [hpow 2 (by omega)] at h2
        rw [hpow 3 (by omega)] at h3
        rw [hpow 4 (by omega)] at h4
        rw [hpow 5 (by omega)] at h5
        rw [hpow 6 (by omega)] at h6
        rw [hpow 7 (by omega)] at h7
        norm_num at h2 h3 h4 h5 h6 h7
        have hpowk :
            (31 / 10 : ℝ) ^ k =
              (31 / 10 : ℝ) ^ (k - 7) * (31 / 10 : ℝ) ^ 7 := by
          simpa using hpow 0 (by omega)
        rw [hpowk]
        have hnonneg : 0 ≤ (31 / 10 : ℝ) ^ (k - 7) := by positivity
        calc
          (1 / 4 : ℝ) *
                ((31 / 10 : ℝ) ^ (k - 7) * (31 / 10 : ℝ) ^ 7) =
              (31 / 10 : ℝ) ^ (k - 7) *
                ((1 / 4 : ℝ) * (31 / 10 : ℝ) ^ 7) := by ring
          _ ≤ (31 / 10 : ℝ) ^ (k - 7) *
                (4 * ((1 / 4 : ℝ) * (31 / 10 : ℝ) ^ 5) +
                 8 * ((1 / 4 : ℝ) * (31 / 10 : ℝ) ^ 4) +
                 16 * ((1 / 4 : ℝ) * (31 / 10 : ℝ) ^ 3) +
                 24 * ((1 / 4 : ℝ) * (31 / 10 : ℝ) ^ 2) +
                 40 * ((1 / 4 : ℝ) * (31 / 10 : ℝ)) +
                 72 * (1 / 4 : ℝ)) := by
              apply mul_le_mul_of_nonneg_left _ hnonneg
              norm_num
          _ ≤ 4 * planeRenewal (k - 2) +
                8 * planeRenewal (k - 3) +
                16 * planeRenewal (k - 4) +
                24 * planeRenewal (k - 5) +
                40 * planeRenewal (k - 6) +
                72 * planeRenewal (k - 7) := by
              norm_num only [Nat.cast_add, Nat.cast_mul]
              nlinarith

/-- The six certified block counts imply the renewal lower bound with base
`3.1`; only indices `2, ..., 8` are checked computationally. -/
theorem planeRenewal_ge (k : ℕ) (hk : 2 ≤ k) :
    (1 / 4 : ℝ) * ((31 / 10 : ℝ) ^ k) ≤ planeRenewal k :=
  planeRenewal_ge_aux k hk

end RubiksSnake

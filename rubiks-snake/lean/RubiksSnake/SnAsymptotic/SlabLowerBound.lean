import RubiksSnake.SnAsymptotic.SlabLanguage
import RubiksSnake.SnAsymptotic.SlabRenewal
import RubiksSnake.SnAsymptotic.SnAsymptotic_MuExistence

/-! The master slab certificate gives the same base for both lower bounds. -/

namespace RubiksSnake.SlabEnumeration

/-- There are four retained blocks of direction-word length two. -/
private lemma retainedCount_two : retainedCount 2 = 4 := by decide
/-- There are eight retained blocks of direction-word length three. -/
private lemma retainedCount_three : retainedCount 3 = 8 := by decide

/-- Any allowed first-block length `j + 1` contributes its coefficient times
the number of remaining tails to the total language count. -/
lemma language_length_ge_term (n : Nat) (j : Fin 29) (hj : j.val + 1 ≤ n) :
    retainedCount (j.val + 1) * (language (n - (j.val + 1))).length ≤
      (language n).length := by
  rw [language_length_rec n (by omega)]
  have h := Finset.single_le_sum
    (f := fun i : Fin 29 =>
      if i.val + 1 ≤ n then retainedCount (i.val + 1) *
        (language (n - (i.val + 1))).length else 0)
    (fun i _ => Nat.zero_le _) (Finset.mem_univ j)
  simpa only [if_pos hj] using h

/-- Length-two and length-three retained blocks yield at least one
concatenation at every total direction length `n >= 2`. -/
lemma language_length_pos (n : Nat) (hn : 2 ≤ n) : 1 ≤ (language n).length := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    by_cases htwo : n = 2
    · subst n
      have h := language_length_ge_term 2 1 (by decide)
      norm_num [retainedCount_two, language_zero] at h
      omega
    · by_cases hthree : n = 3
      · subst n
        have h := language_length_ge_term 3 2 (by decide)
        norm_num [retainedCount_three, language_zero] at h
        omega
      · have h := language_length_ge_term n 1 (by simp; omega)
        norm_num [retainedCount_two] at h
        have := ih (n - 2) (by omega) (by omega)
        omega

/-- For `n >= 31`, all block lengths up to 29 fit, so the full retained
renewal sum is bounded by the language count as required by the certificate. -/
lemma language_length_recurrence (n : Nat) (hn : 31 ≤ n) :
    (∑ j : Fin 29, retainedCount (j.val + 1) *
      (language (n - (j.val + 1))).length) ≤ (language n).length := by
  rw [language_length_rec n (by omega)]
  apply le_of_eq
  apply Finset.sum_congr rfl
  intro j _
  rw [if_pos (by omega : j.val + 1 ≤ n)]

/-- The slab renewal certificate gives at least `4^(-31) * 3.400034903^k`
valid formulas for every rotation count `k`, corresponding to `k + 1` wedges. -/
theorem count_lower_bound_with_prefactor (k : Nat) :
    (1 / 4 ^ 31 : ℝ) * (3400034903 / 1000000000 : ℝ) ^ k ≤
      countValidFormulas k := by
  by_cases hk : 2 ≤ k
  · exact (slabRenewal_ge_of_recurrence (fun n => (language n).length)
      language_length_pos language_length_recurrence k hk).trans
        (by exact_mod_cast language_length_le k)
  · obtain rfl | rfl : k = 0 ∨ k = 1 := by omega
    · have hzero : countValidFormulas 0 = 1 := by simpa [S] using S1_value
      norm_num [hzero]
    · have hone : countValidFormulas 1 = 4 := by simpa [S] using S2_value
      norm_num [hone]

/-- The retained slab construction certifies the growth-constant lower bound
`3.400034903`. -/
theorem growthConstant_lower_bound :
    (3400034903 / 1000000000 : ℝ) ≤ snakeGrowthConstant :=
  snakeGrowthConstant_ge_of_pointwise (1 / 4 ^ 31) (3400034903 / 1000000000)
    (by positivity) (by norm_num) count_lower_bound_with_prefactor

/-- The growth-constant inequality removes the renewal prefactor, giving
`3.400034903^k` as a lower bound for every `k`-rotation formula count. -/
theorem count_lower_bound (k : Nat) :
    (3400034903 / 1000000000 : ℝ) ^ k ≤ countValidFormulas k :=
  (pow_le_pow_left₀ (by norm_num) growthConstant_lower_bound k).trans
    (snakeGrowthConstant_pow_le_countValidFormulas k)

/-- For every positive wedge count `n`, the snake count is at least
`3.400034903^(n - 1)`, since there are `n - 1` rotations. -/
theorem Sn_lower_bound (n : ℕ+) :
    (3400034903 / 1000000000 : ℝ) ^ ((n : Nat) - 1) ≤ S n :=
  count_lower_bound ((n : Nat) - 1)

end RubiksSnake.SlabEnumeration

import Mathlib.Tactic

/-! # Finite certificates for renewal lower bounds -/

namespace RubiksSnake

/-- A nonnegative renewal inequality propagates an exponential lower bound from `L` initial
indices to every index at least `start`, provided the finite coefficient polynomial dominates
`q ^ L`. The recurrence may undercount rather than exactly equal `f`. -/
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

end RubiksSnake

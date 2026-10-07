import RubiksSnake.SlabCountCertificate
import RubiksSnake.RenewalBounds

/-!
# Numerical slab renewal bound

The retained slab coefficients certify the base `3.400034903`. The resulting
bound applies to any positive sequence satisfying the renewal inequality;
no geometric injection is assumed or proved here.
-/

namespace RubiksSnake

/-- The exact retained coefficients satisfy the real renewal polynomial inequality. -/
theorem slabRenewal_polynomial :
    (3400034903 / 1000000000 : ℝ) ^ 29 ≤
      ∑ j : Fin 29, (SlabEnumeration.retainedCount (j.val + 1) : ℝ) *
        (3400034903 / 1000000000 : ℝ) ^ (29 - (j.val + 1)) := by
  norm_num [Fin.sum_univ_succ, SlabEnumeration.retainedCount,
    SlabEnumeration.rowOne, SlabEnumeration.rowTwo, SlabEnumeration.rowThree]

/-- Positivity from index two and the retained-coefficient recurrence from
index 31 give a uniform exponential bound, independently of geometry. -/
theorem slabRenewal_ge_of_recurrence
    (f : ℕ → ℕ)
    (hpos : ∀ k, 2 ≤ k → 1 ≤ f k)
    (hrec : ∀ k, 31 ≤ k →
      (∑ j : Fin 29, SlabEnumeration.retainedCount (j.val + 1) *
        f (k - (j.val + 1))) ≤ f k)
    (k : ℕ) (hk : 2 ≤ k) :
    (1 / 4 ^ 31 : ℝ) * (3400034903 / 1000000000 : ℝ) ^ k ≤ f k := by
  apply renewal_ge_of_finite_certificate f 29 2
    (fun j => SlabEnumeration.retainedCount (j.val + 1))
    (1 / 4 ^ 31) (3400034903 / 1000000000)
    (by positivity) (by norm_num) slabRenewal_polynomial ?_ ?_ k hk
  · intro k hk hsmall
    have hpow : (3400034903 / 1000000000 : ℝ) ^ k ≤ (4 : ℝ) ^ 31 := by
      calc
        (3400034903 / 1000000000 : ℝ) ^ k ≤ (4 : ℝ) ^ k :=
          pow_le_pow_left₀ (by norm_num) (by norm_num) k
        _ ≤ (4 : ℝ) ^ 31 := pow_le_pow_right₀ (by norm_num) (by omega)
    calc
      (1 / 4 ^ 31 : ℝ) * (3400034903 / 1000000000 : ℝ) ^ k ≤
          (1 / 4 ^ 31 : ℝ) * 4 ^ 31 :=
        mul_le_mul_of_nonneg_left hpow (by positivity)
      _ = 1 := by norm_num
      _ ≤ (f k : ℝ) := by exact_mod_cast hpos k hk
  · intro k hk
    exact_mod_cast hrec k hk

end RubiksSnake

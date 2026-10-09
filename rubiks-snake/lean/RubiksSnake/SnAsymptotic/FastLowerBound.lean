import RubiksSnake.SnAsymptotic.FastLowerCertificate

/-!
# A fast-checked lower bound above 3.4

The geometric fourfold orbit argument replaces three quarters of the native
enumeration. Arbitrary budget pruning gives undercounts; primitive bridge
decoding and the existing renewal theorem give the asymptotic lower bound.
-/

namespace RubiksSnake.FastLower

attribute [local irreducible] QuarterSlab.counts QuarterSlab.ofSlabs
  QuarterSlab.seedWords BridgeSymmetry.fourfold BridgeCode.blocks

/-- The fourfold geometric code for the four checked seed widths. -/
def code : BridgeCode.Code :=
  BridgeSymmetry.fourfold
    (QuarterSlab.ofSlabs [(0, 19), (1, 19), (2, 17), (3, 20)] (by decide))
    (QuarterSlab.ofSlabs_heading _ _)

/-- The checked orbit coefficients undercount this valid, uniquely decoded code. -/
lemma coefficient_le (n : Nat) :
    coefficient n ≤ (BridgeCode.blocks code (n + 1)).length := by
  have h := QuarterSlab.fourfold_counts_le
    [(0, 19), (1, 19), (2, 17), (3, 20)] (by decide) n
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    rows_checked.1, rows_checked.2.1, rows_checked.2.2.1, rows_checked.2.2.2,
    Nat.add_zero] at h
  simpa only [code, coefficient, Nat.add_assoc] using h

/-- The new lower bound, with no dependency on the earlier expensive certificate. -/
theorem growthConstant_lower_bound : (3.4003 : ℝ) ≤ snakeGrowthConstant := by
  rw [show (3.4003 : ℝ) = 34003 / 10000 by norm_num]
  apply BridgeCode.le_growthConstant code 22 (by decide)
    (fun j => coefficient j.val) (fun j => coefficient_le j.val)
    ?_ ?_ (34003 / 10000) (by norm_num) (by norm_num) polynomial
  · exact (by decide : 1 ≤ coefficient 1).trans (coefficient_le 1)
  · exact (by decide : 1 ≤ coefficient 2).trans (coefficient_le 2)

/-- Fekete's inequality removes the renewal prefactor at every rotation length. -/
theorem count_lower_bound (k : Nat) :
    (3.4003 : ℝ) ^ k ≤ countValidFormulas k :=
  (pow_le_pow_left₀ (by norm_num) growthConstant_lower_bound k).trans
    (snakeGrowthConstant_pow_le_countValidFormulas k)

/-- Uniform pointwise lower bound for every positive number of wedges. -/
theorem Sn_lower_bound (n : ℕ+) : (3.4003 : ℝ) ^ ((n : Nat) - 1) ≤ S n :=
  count_lower_bound ((n : Nat) - 1)

end RubiksSnake.FastLower

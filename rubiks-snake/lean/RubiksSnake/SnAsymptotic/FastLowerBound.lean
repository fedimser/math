import RubiksSnake.SnAsymptotic.FastLowerCertificate

/-!
# Fourfold bridge code

The geometric fourfold orbit argument replaces three quarters of the native
enumeration. The resulting checked coefficients supply the legacy bridge
family used by the cap construction.
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

end RubiksSnake.FastLower

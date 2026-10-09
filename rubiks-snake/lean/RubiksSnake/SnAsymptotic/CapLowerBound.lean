import RubiksSnake.SnAsymptotic.CapCombinedCode
import RubiksSnake.SnAsymptotic.CapCandidateCertificate

/-! Pointwise lower bounds from geometrically separated overhanging caps. -/

namespace RubiksSnake.CapLower

open CapAssembly CapEnumeration CapCandidate

/-- The certified coefficients undercount a finite code of distinct valid irreducible bridges. -/
theorem coefficient_le (j : Fin 64) :
    coefficients[j.val + 1]! ≤ (BridgeCode.blocks combinedCode (j.val + 1)).length := by
  have hn : j.val + 1 ≤ 64 := by omega
  have htwo : (assembledCounts 2 2 14 64)[j.val + 1]! ≤ (traces 2 14 (j.val + 1)).length :=
    assembledCounts_le_traces (0 : Fin 3) (j.val + 1) hn
  have hthree : (assembledCounts 3 2 14 64)[j.val + 1]! ≤ (traces 3 14 (j.val + 1)).length :=
    assembledCounts_le_traces (1 : Fin 3) (j.val + 1) hn
  have hfour : (assembledCounts 4 2 14 64)[j.val + 1]! ≤ (traces 4 14 (j.val + 1)).length :=
    assembledCounts_le_traces (2 : Fin 3) (j.val + 1) hn
  have ht :
      (if 19 < j.val + 1 then (assembledCounts 2 2 14 64)[j.val + 1]! else 0) ≤
        (if 19 < j.val + 1 then (traces 2 14 (j.val + 1)).length else 0) := by
    split_ifs <;> omega
  have hh :
      (if 22 < j.val + 1 then (assembledCounts 3 2 14 64)[j.val + 1]! else 0) ≤
        (if 22 < j.val + 1 then (traces 3 14 (j.val + 1)).length else 0) := by
    split_ifs <;> omega
  have hold := FastLower.coefficient_le j.val
  rw [coefficients_get _ (by omega) (by omega),
    combinedCode_blocks_length _ (by omega) (by omega)]
  simp only [Nat.add_sub_cancel]
  omega

/-- Clearing the common positive denominator translates the integer check to real arithmetic. -/
theorem polynomial :
    (17252837 / 5000000 : ℝ) ^ 64 ≤
      ∑ j : Fin 64, (coefficients[j.val + 1]! : ℝ) *
        (17252837 / 5000000 : ℝ) ^ (64 - (j.val + 1)) := by
  have h : (17252837 : ℝ) ^ 64 ≤ ∑ j : Fin 64,
      (coefficients[j.val + 1]! : ℝ) * 17252837 ^ (64 - (j.val + 1)) *
        5000000 ^ (j.val + 1) := by
    exact_mod_cast integer_polynomial
  rw [div_pow]
  apply (div_le_iff₀ (by positivity : (0 : ℝ) < 5000000 ^ 64)).mpr
  rw [Finset.sum_mul]
  apply h.trans_eq
  symm
  apply Finset.sum_congr rfl
  intro j _
  have he : (5000000 : ℝ) ^ 64 =
      5000000 ^ (64 - (j.val + 1)) * 5000000 ^ (j.val + 1) := by
    rw [← pow_add]
    congr 1
    omega
  rw [div_pow, he]
  field_simp

/-- The cap certificate proves the exponential lower bound at every rotation length. -/
theorem count_lower_bound (k : Nat) :
    (3.4505674 : ℝ) ^ k ≤ countValidFormulas k := by
  rw [show (3.4505674 : ℝ) = 17252837 / 5000000 by norm_num]
  apply BridgeCode.pow_le_countValidFormulas combinedCode 64 (by decide)
    (fun j => coefficients[j.val + 1]!) coefficient_le ?_ ?_
    (17252837 / 5000000) (by norm_num) (by norm_num) polynomial k
  · have hold : 1 ≤ (BridgeCode.blocks FastLower.code 2).length :=
      (by decide : 1 ≤ FastLower.coefficient 1).trans (FastLower.coefficient_le 1)
    rw [combinedCode, appendCode_blocks_length]
    omega
  · have hold : 1 ≤ (BridgeCode.blocks FastLower.code 3).length :=
      (by decide : 1 ≤ FastLower.coefficient 2).trans (FastLower.coefficient_le 2)
    rw [combinedCode, appendCode_blocks_length]
    omega

/-- Uniform lower bound at every positive wedge length. -/
theorem Sn_lower_bound (n : ℕ+) : (3.4505674 : ℝ) ^ ((n : Nat) - 1) ≤ S n :=
  count_lower_bound ((n : Nat) - 1)

end RubiksSnake.CapLower

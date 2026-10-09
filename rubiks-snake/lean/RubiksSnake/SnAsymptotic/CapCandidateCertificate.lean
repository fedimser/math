import RubiksSnake.SnAsymptotic.CapComputation
import RubiksSnake.SnAsymptotic.FastLowerCertificate

/-!
# Exact arithmetic for the transverse-cap lower certificate

This module checks the finite coefficient calculation at 3.4505674.
`CapAssemblyCertificate` proves the coefficient-to-word connection, and
`CapLowerBound` supplies the complete bridge code to the renewal theorem.
-/

set_option Elab.async false

namespace RubiksSnake.CapCandidate

/-- Candidate coefficients by wedge length, with short words removed from
the new families to avoid overlap with the existing fourfold certificate. -/
def coefficients : Array Nat :=
  let two := CapEnumeration.assembledCounts 2 2 14 64
  let three := CapEnumeration.assembledCounts 3 2 14 64
  let four := CapEnumeration.assembledCounts 4 2 14 64
  (Array.range 65).map fun n =>
    (if n = 0 then 0 else FastLower.coefficient (n - 1)) +
      2 * ((if 19 < n then two[n]! else 0) +
        (if 22 < n then three[n]! else 0) + four[n]!)

/-- The coefficient layout at every positive length in the certificate. -/
lemma coefficients_get (n : Nat) (hn : n < 65) (hpos : 0 < n) :
    coefficients[n]! =
      FastLower.coefficient (n - 1) +
        2 * ((if 19 < n then (CapEnumeration.assembledCounts 2 2 14 64)[n]! else 0) +
          (if 22 < n then (CapEnumeration.assembledCounts 3 2 14 64)[n]! else 0) +
          (CapEnumeration.assembledCounts 4 2 14 64)[n]!) := by
  simp [coefficients, hn, Nat.ne_of_gt hpos]

/-- Clearing denominators gives the exact renewal inequality at `17252837/5000000`. -/
theorem integer_polynomial :
    17252837 ^ 64 ≤ ∑ j : Fin 64,
      coefficients[j.val + 1]! * 17252837 ^ (64 - (j.val + 1)) *
        5000000 ^ (j.val + 1) := by
  native_decide

end RubiksSnake.CapCandidate

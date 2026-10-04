import RubiksSnake.Definitions

/-! Asymptotic analysis for sequence S_n. -/

namespace RubiksSnake

/-- Show that for n<=4, S_n is power of 4 (because all formulas are valid). -/
lemma Sn_is_power_of_4 (n : ℕ) : 1 ≤ n → n ≤ 4 → S n = 4 ^ (n - 1) := by
  intro hn hfour
  have allValid : ∀ w : Word (n - 1), Valid w := by
    obtain rfl | rfl | rfl | rfl : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
    all_goals native_decide
  let validEquiv : {w : Word (n - 1) // Valid w} ≃ Word (n - 1) :=
    { toFun := Subtype.val
      invFun := fun w => ⟨w, allValid w⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hcount : countWords (n - 1) Valid = 4 ^ (n - 1) := by
    rw [countWords, Nat.card_congr validEquiv]
    simp [Word, Rotation]
  rw [S, wordCount, hcount, max_eq_right]
  exact Nat.one_le_pow (n - 1) 4 (by omega)

example : S 1 = 1 := by simpa using Sn_is_power_of_4 1
example : S 2 = 4 := by simpa using Sn_is_power_of_4 2
example : S 3 = 16 := by simpa using Sn_is_power_of_4 3
example : S 4 = 64 := by simpa using Sn_is_power_of_4 4

end RubiksSnake

import RubiksSnake.Geometry

/-!
  Proof that 2^(n-1) <= S_n <= 4^(n-1).
-/

namespace RubiksSnake

/-- The valid formulas are a subset of all four-symbol words of the same length. -/
lemma countFormulas_upper_bound (k : ℕ) : countFormulas k Valid ≤ 4 ^ k := by
  unfold countFormulas
  calc
    Nat.card {w : Formula k // Valid w} ≤ Nat.card (Formula k) :=
      Nat.card_le_card_of_injective Subtype.val Subtype.val_injective
    _ = 4 ^ k := by simp [Formula, Rotation]

/-- For every positive `n`, the number of valid formulas is at most `4^(n-1)`. -/
lemma Sn_upper_bound_4n (n : ℕ+) : S n ≤ 4 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  exact countFormulas_upper_bound (n - 1)

def increasingFormula {k : ℕ} (choices : Fin k → Bool) : Formula k :=
  fun i =>
    (increasingRotationList ey ex (List.ofFn choices)).get
      (Fin.cast (by simp) i)

lemma increasingFormula_toList {k : ℕ} (choices : Fin k → Bool) :
    List.ofFn (increasingFormula choices) =
      increasingRotationList ey ex (List.ofFn choices) := by
  apply List.ext_get
  · simp
  · intro i hi₁ hi₂
    simp [increasingFormula]

lemma increasingFormula_valid {k : ℕ} (choices : Fin k → Bool) :
    Valid (increasingFormula choices) := by
  unfold Valid
  rw [increasingFormula_toList]
  exact increasingRotationList_valid (List.ofFn choices)

lemma increasingFormula_injective (k : ℕ) :
    Function.Injective (@increasingFormula k) := by
  intro a b hab
  apply List.ofFn_injective
  apply increasingRotationList_injective ey ex
  rw [← increasingFormula_toList a, ← increasingFormula_toList b, hab]

def increasingValidFormula (k : ℕ) :
    (Fin k → Bool) → {w : Formula k // Valid w} :=
  fun choices => ⟨increasingFormula choices, increasingFormula_valid choices⟩

lemma increasingValidFormula_injective (k : ℕ) :
    Function.Injective (increasingValidFormula k) := by
  intro a b hab
  apply increasingFormula_injective k
  exact congrArg Subtype.val hab

lemma Sn_lower_bound_2n (n : ℕ+) : S n ≥ 2 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  calc
    2 ^ (n - 1) = Nat.card (Fin (n - 1) → Bool) := by simp
    _ ≤ Nat.card {w : Formula (n - 1) // Valid w} :=
      Nat.card_le_card_of_injective
        (increasingValidFormula (n - 1)) (increasingValidFormula_injective (n - 1))
    _ = countFormulas (n - 1) Valid := rfl

end RubiksSnake

import RubiksSnake.Definitions

/-! Asymptotic analysis for sequence S_n. -/

namespace RubiksSnake

/-- Show that for `n ≤ 4`, `S_n` is a power of four because all formulas are valid. -/
lemma Sn_is_power_of_4 (n : ℕ+) (hfour : n ≤ 4) :
    S n = 4 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  change n ≤ 4 at hfour
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
  change wordCount (n - 1) = 4 ^ (n - 1)
  rw [wordCount, hcount, max_eq_right]
  exact Nat.one_le_pow (n - 1) 4 (by omega)

example : S 1 = 1 := by simpa using Sn_is_power_of_4 1
example : S 2 = 4 := by simpa using Sn_is_power_of_4 2
example : S 3 = 16 := by simpa using Sn_is_power_of_4 3
example : S 4 = 64 := by simpa using Sn_is_power_of_4 4

/-- The valid formulas are a subset of all four-symbol words of the same length. -/
lemma countWords_upper_bound (k : ℕ) : countWords k Valid ≤ 4 ^ k := by
  unfold countWords
  calc
    Nat.card {w : Word k // Valid w} ≤ Nat.card (Word k) :=
      Nat.card_le_card_of_injective Subtype.val Subtype.val_injective
    _ = 4 ^ k := by simp [Word, Rotation]

/-- For every positive `n`, the number of valid formulas is at most `4^(n-1)`. -/
lemma Sn_upper_bound_4n (n : ℕ+) : S n ≤ 4 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  change max 1 (countWords (n - 1) Valid) ≤ 4 ^ (n - 1)
  apply max_le
  · exact Nat.one_le_pow (n - 1) 4 (by omega)
  · exact countWords_upper_bound (n - 1)

/-- A positive coordinate direction. -/
def PositiveDirection (v : Vec3) : Prop :=
  v = ex ∨ v = ey ∨ v = ez

instance (v : Vec3) : Decidable (PositiveDirection v) := by
  unfold PositiveDirection
  infer_instance

def increasingRotation (previous axis : Vec3) (choice : Bool) : Rotation :=
  if choice then
    if PositiveDirection (cross axis previous) then 1 else 3
  else 0

lemma increasingRotation_injective (previous axis : Vec3) :
    Function.Injective (increasingRotation previous axis) := by
  intro a b hab
  cases a <;> cases b
  · rfl
  · exfalso
    have := congrArg Fin.val hab
    simp [increasingRotation] at this
    split at this <;> omega
  · exfalso
    have := congrArg Fin.val hab
    simp [increasingRotation] at this
    split at this <;> omega
  · rfl

lemma increasingRotation_next {previous axis : Vec3}
    (hprevious : PositiveDirection previous) (haxis : PositiveDirection axis)
    (hne : previous ≠ axis) (choice : Bool) :
    let next := rotateQuarter axis (increasingRotation previous axis choice) previous
    PositiveDirection next ∧ next ≠ axis := by
  rcases hprevious with rfl | rfl | rfl <;>
    rcases haxis with rfl | rfl | rfl <;>
    cases choice <;>
    simp_all [increasingRotation, PositiveDirection, rotateQuarter, cross, ex, ey, ez]
  all_goals native_decide

def increasingFormula : Vec3 → Vec3 → List Bool → List Rotation
  | _, _, [] => []
  | previous, axis, choice :: choices =>
      let rotation := increasingRotation previous axis choice
      let next := rotateQuarter axis rotation previous
      rotation :: increasingFormula axis next choices

@[simp] lemma increasingFormula_length (previous axis : Vec3) (choices : List Bool) :
    (increasingFormula previous axis choices).length = choices.length := by
  induction choices generalizing previous axis with
  | nil => rfl
  | cons choice choices ih =>
      simp [increasingFormula, ih]

lemma increasingFormula_injective (previous axis : Vec3) :
    Function.Injective (increasingFormula previous axis) := by
  intro xs
  induction xs generalizing previous axis with
  | nil =>
      intro ys h
      cases ys <;> simp_all [increasingFormula]
  | cons choice choices ih =>
      intro ys h
      cases ys with
      | nil => simp [increasingFormula] at h
      | cons choice' choices' =>
          simp only [increasingFormula, List.cons.injEq] at h
          have hchoice : choice = choice' :=
            increasingRotation_injective previous axis h.1
          subst choice'
          have hchoices : choices = choices' := ih axis _ h.2
          subst choices'
          rfl

lemma directionTail_increasingFormula {previous axis : Vec3}
    (hprevious : PositiveDirection previous) (haxis : PositiveDirection axis)
    (hne : previous ≠ axis) (choices : List Bool) :
    ∀ v ∈ directionTail previous axis (increasingFormula previous axis choices),
      PositiveDirection v := by
  induction choices generalizing previous axis with
  | nil => simp [increasingFormula, directionTail]
  | cons choice choices ih =>
      let rotation := increasingRotation previous axis choice
      let next := rotateQuarter axis rotation previous
      have hnext : PositiveDirection next ∧ next ≠ axis := by
        simpa [rotation, next] using
          increasingRotation_next hprevious haxis hne choice
      intro v hv
      simp only [increasingFormula, directionTail, List.mem_cons] at hv
      rcases hv with rfl | hv
      · exact hnext.1
      · exact ih haxis hnext.1 hnext.2.symm v hv

def coordinateSum (v : Vec3) : ℤ :=
  v 0 + v 1 + v 2

lemma coordinateSum_addVec (u v : Vec3) :
    coordinateSum (addVec u v) = coordinateSum u + coordinateSum v := by
  simp [coordinateSum, addVec]
  omega

lemma coordinateSum_positiveDirection {v : Vec3} (hv : PositiveDirection v) :
    coordinateSum v = 1 := by
  rcases hv with rfl | rfl | rfl <;>
    native_decide

lemma coordinateSum_lt_scanl_tail (steps : List Vec3) (start : Vec3)
    (hsteps : ∀ v ∈ steps, PositiveDirection v) :
    ∀ p ∈ (steps.scanl addVec start).tail, coordinateSum start < coordinateSum p := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      have hstep := coordinateSum_positiveDirection (hsteps step (by simp))
      have hrest : ∀ v ∈ steps, PositiveDirection v :=
        fun v hv => hsteps v (by simp [hv])
      intro p hp
      rw [List.scanl_cons] at hp
      simp only [List.tail_cons] at hp
      have hfirst : coordinateSum start < coordinateSum (addVec start step) := by
        rw [coordinateSum_addVec, hstep]
        omega
      cases steps with
      | nil =>
          simp only [List.scanl_nil, List.mem_singleton] at hp
          subst p
          exact hfirst
      | cons next steps =>
          rw [List.scanl_cons] at hp
          rcases List.mem_cons.mp hp with rfl | hp
          · exact hfirst
          · apply lt_trans hfirst
            apply ih (addVec start step) hrest p
            simpa only [List.scanl_cons, List.tail_cons] using hp

lemma scanl_pairwise_coordinateSum (steps : List Vec3) (start : Vec3)
    (hsteps : ∀ v ∈ steps, PositiveDirection v) :
    (steps.scanl addVec start).Pairwise
      (fun p q => coordinateSum p < coordinateSum q) := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      have hrest : ∀ v ∈ steps, PositiveDirection v :=
        fun v hv => hsteps v (by simp [hv])
      rw [List.scanl_cons]
      constructor
      · intro p hp
        apply coordinateSum_lt_scanl_tail (step :: steps) start hsteps p
        simpa only [List.scanl_cons, List.tail_cons] using hp
      · exact ih (addVec start step) hrest

@[simp] lemma directionTail_length (previous axis : Vec3) (rs : List Rotation) :
    (directionTail previous axis rs).length = rs.length := by
  induction rs generalizing previous axis with
  | nil => rfl
  | cons r rs ih => simp [directionTail, ih]

lemma wedge_centers (rs : List Rotation) :
    (wedges rs).map Wedge.center = centersFromDirections (directions rs) := by
  unfold wedges wedgesFromDirections
  rw [List.map_map]
  change List.map Prod.fst
      ((centersFromDirections (directions rs)).zip
        ((directions rs).zip (directions rs).tail)) =
    centersFromDirections (directions rs)
  apply List.map_fst_zip
  simp [centersFromDirections, directions, directionsFrom]

lemma increasingFormula_validList (choices : List Bool) :
    ValidList (increasingFormula ey ex choices) := by
  let rs := increasingFormula ey ex choices
  let ds := directions rs
  let steps := (ds.drop 1).dropLast
  have htail :
      ∀ v ∈ directionTail ey ex rs, PositiveDirection v := by
    exact directionTail_increasingFormula
      (by simp [PositiveDirection]) (by simp [PositiveDirection])
      (by native_decide) choices
  have hsteps : ∀ v ∈ steps, PositiveDirection v := by
    intro v hv
    have hv' : v ∈ ex :: directionTail ey ex rs := by
      apply List.mem_of_mem_dropLast
      simpa [steps, ds, directions, directionsFrom] using hv
    rcases List.mem_cons.mp hv' with rfl | hv'
    · simp [PositiveDirection]
    · exact htail v hv'
  have hcenters :
      (centersFromDirections ds).Pairwise
        (fun p q => coordinateSum p < coordinateSum q) := by
    exact scanl_pairwise_coordinateSum steps zeroVec hsteps
  have hcenterNodup : (centersFromDirections ds).Nodup :=
    hcenters.imp fun hpq hpqeq => by
      rw [hpqeq] at hpq
      exact lt_irrefl _ hpq
  have hmapped : ((wedges rs).map Wedge.center).Nodup := by
    rw [wedge_centers]
    exact hcenterNodup
  change ((wedges rs).map Wedge.center).Pairwise (· ≠ ·) at hmapped
  rw [List.pairwise_map] at hmapped
  unfold ValidList collisionFree
  exact hmapped.imp fun hne => Or.inl hne

def increasingWord {k : ℕ} (choices : Fin k → Bool) : Word k :=
  fun i =>
    (increasingFormula ey ex (List.ofFn choices)).get
      (Fin.cast (by simp) i)

lemma increasingWord_toList {k : ℕ} (choices : Fin k → Bool) :
    List.ofFn (increasingWord choices) =
      increasingFormula ey ex (List.ofFn choices) := by
  apply List.ext_get
  · simp
  · intro i hi₁ hi₂
    simp [increasingWord]

lemma increasingWord_valid {k : ℕ} (choices : Fin k → Bool) :
    Valid (increasingWord choices) := by
  unfold Valid
  rw [increasingWord_toList]
  exact increasingFormula_validList (List.ofFn choices)

lemma increasingWord_injective (k : ℕ) :
    Function.Injective (@increasingWord k) := by
  intro a b hab
  apply List.ofFn_injective
  apply increasingFormula_injective ey ex
  rw [← increasingWord_toList a, ← increasingWord_toList b, hab]

def increasingValidWord (k : ℕ) :
    (Fin k → Bool) → {w : Word k // Valid w} :=
  fun choices => ⟨increasingWord choices, increasingWord_valid choices⟩

lemma increasingValidWord_injective (k : ℕ) :
    Function.Injective (increasingValidWord k) := by
  intro a b hab
  apply increasingWord_injective k
  exact congrArg Subtype.val hab

lemma Sn_lower_bound_2n (n : ℕ+) : S n ≥ 2 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  change 2 ^ (n - 1) ≤ max 1 (countWords (n - 1) Valid)
  apply le_trans ?_ (le_max_right 1 (countWords (n - 1) Valid))
  calc
    2 ^ (n - 1) = Nat.card (Fin (n - 1) → Bool) := by simp
    _ ≤ Nat.card {w : Word (n - 1) // Valid w} :=
      Nat.card_le_card_of_injective
        (increasingValidWord (n - 1)) (increasingValidWord_injective (n - 1))
    _ = countWords (n - 1) Valid := rfl


end RubiksSnake

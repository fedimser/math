import RubiksSnake.ReflectionTransform
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-! Definitions and lemmas about rotation-restricted sequences. -/

namespace RubiksSnake

noncomputable section

/-
Rubik's Snake has 4 possible rotations, encoded in formula by numbers 0,1,2,3.

Define SR_r(n) - number of n-wedge snakes when only rotations from set r
are allowed.
-/

/-- Shapes using only rotation symbols from `allowed`. -/
def SR (allowed : Finset Rotation) (n : ℕ+) : ℕ :=
  countFormulas ((n : ℕ) - 1) fun w => Valid w ∧ ∀ i, w i ∈ allowed

/-- Unrestricted rotations: SR_0123 is S. -/
lemma SR_unrestricted_is_S: SR {0, 1, 2, 3} = S := by
  ext n
  unfold SR S countValidFormulas
  apply congrArg (countFormulas (n - 1))
  grind


/-- Reflect an allowed rotation alphabet by exchanging symbols `1` and `3`;
the corresponding restricted counts are unchanged, as shown by `mirrorSameCount`. -/
def mirrorAllowedRotations (X : Finset Rotation) := X.image mirrorRotation

/-- A symbol lies in the reflected alphabet exactly when its mirror lies in
the original alphabet, since mirroring a symbol is involutive. -/
lemma mem_mirrorAllowedRotations_iff (allowed : Finset Rotation)
    (r : Rotation) :
    r ∈ mirrorAllowedRotations allowed ↔ mirrorRotation r ∈ allowed := by
  constructor
  · intro h
    obtain ⟨x, hx, hxr⟩ := Finset.mem_image.mp h
    rw [← hxr, mirrorRotation_involutive]
    exact hx
  · intro h
    exact Finset.mem_image.mpr
      ⟨mirrorRotation r, h, mirrorRotation_involutive r⟩

/-- Reflect a valid `n`-rotation formula over `allowed`, retaining its validity
proof and replacing its alphabet by the reflected one. -/
def mirrorToRestricted (allowed : Finset Rotation) {n : ℕ}
    (w : {w : Formula n // Valid w ∧ ∀ i, w i ∈ allowed}) :
    {w : Formula n //
      Valid w ∧ ∀ i, w i ∈ mirrorAllowedRotations allowed} := by
  refine ⟨mirrorFormula w.val, ?_, ?_⟩
  · exact (valid_mirrorFormula w.val).mpr w.property.1
  · intro i
    exact Finset.mem_image.mpr ⟨w.val i, w.property.2 i, rfl⟩

/-- Reflect a valid formula over the mirrored alphabet back to a valid formula
over the original allowed symbols. -/
def mirrorFromRestricted (allowed : Finset Rotation) {n : ℕ}
    (w : {w : Formula n //
      Valid w ∧ ∀ i, w i ∈ mirrorAllowedRotations allowed}) :
    {w : Formula n // Valid w ∧ ∀ i, w i ∈ allowed} := by
  refine ⟨mirrorFormula w.val, ?_, ?_⟩
  · exact (valid_mirrorFormula w.val).mpr w.property.1
  · intro i
    exact (mem_mirrorAllowedRotations_iff allowed (w.val i)).mp
      (w.property.2 i)

/-- Reflection gives a bijection between valid `n`-rotation formulas over an
allowed alphabet and those over its mirror. -/
def mirrorRestrictedEquiv (allowed : Finset Rotation) (n : ℕ) :
    {w : Formula n // Valid w ∧ ∀ i, w i ∈ allowed} ≃
      {w : Formula n //
        Valid w ∧ ∀ i, w i ∈ mirrorAllowedRotations allowed} where
  toFun := mirrorToRestricted allowed
  invFun := mirrorFromRestricted allowed
  left_inv w := by
    apply Subtype.ext
    exact mirrorFormula_involutive w.val
  right_inv w := by
    apply Subtype.ext
    exact mirrorFormula_involutive w.val

example : mirrorAllowedRotations {1} = {3} := by native_decide
example : mirrorAllowedRotations {0, 1} = {0, 3} := by native_decide

/-- Reflecting the allowed alphabet preserves the count for every number of wedges. -/
lemma mirrorSameCount: SR allowed  = SR (mirrorAllowedRotations allowed) := by
  funext n
  unfold SR countFormulas
  exact Nat.card_congr
    (mirrorRestrictedEquiv allowed ((n : ℕ) - 1))




/-- Counts when only one rotation is allowed. -/
def SR_0 := SR {0}
/-- Counts of `n`-wedge snakes whose `n - 1` joint settings are all `1`. -/
def SR_1 := SR {1}
/-- Counts of `n`-wedge snakes whose `n - 1` joint settings are all `2`. -/
def SR_2 := SR {2}
/-- Counts of `n`-wedge snakes whose `n - 1` joint settings are all `3`. -/
def SR_3 := SR {3}

/-- Formula 000..0 is valid. -/
lemma allZeroFormula_valid (k : ℕ) :
    Valid (fun _ : Fin k => (0 : Rotation)) := by
  have hencode :
      ∀ m (previous axis : Vec3),
        increasingRotationList previous axis (List.replicate m false) =
          List.replicate m (0 : Rotation) := by
    intro m
    induction m with
    | zero => simp [increasingRotationList]
    | succ m ih =>
        intro previous axis
        simp only [List.replicate_succ, increasingRotationList, increasingRotation,
          Bool.false_eq_true, ↓reduceIte]
        have hzero : rotateQuarter axis (0 : Rotation) previous = previous := rfl
        rw [hzero]
        rw [ih axis previous]
  unfold Valid
  have hvalid := increasingRotationList_valid (List.replicate k false)
  rw [hencode k ey ex] at hvalid
  rw [List.ofFn_const]
  exact hvalid

/-- For every positive wedge count, the all-zero formula is the unique valid
formula using only rotation symbol `0`. -/
lemma SR_0_exact_formula (n : ℕ+) : SR {0} n = 1 := by
  unfold SR countFormulas
  apply Nat.card_eq_one_iff_exists.mpr
  let zeroFormula :
      {w : Formula ((n : ℕ) - 1) // Valid w ∧ ∀ i, w i ∈ ({0} : Finset Rotation)} :=
    ⟨fun _ => 0, ⟨allZeroFormula_valid _, by simp⟩⟩
  refine ⟨zeroFormula, ?_⟩
  intro formula
  apply Subtype.ext
  funext i
  simpa using formula.property.2 i

/-- Formula 111..1 is valid. -/
lemma all1Formula_valid (k : ℕ) :
    Valid (fun _ : Fin k => (1 : Rotation)) := by
  have hencode :
      ∀ m,
        increasingRotationList ey ex (List.replicate m true) =
            List.replicate m (1 : Rotation) ∧
          increasingRotationList ex ez (List.replicate m true) =
            List.replicate m (1 : Rotation) ∧
          increasingRotationList ez ey (List.replicate m true) =
            List.replicate m (1 : Rotation) := by
    intro m
    induction m with
    | zero => simp [increasingRotationList]
    | succ m ih =>
        simp only [List.replicate_succ, increasingRotationList]
        simp [increasingRotation, PositiveDirection, rotateQuarter,
          cross, ex, ey, ez]
        simpa [ex, ey, ez] using ⟨ih.2.1, ih.2.2, ih.1⟩
  unfold Valid
  have hvalid := increasingRotationList_valid (List.replicate k true)
  rw [(hencode k).1] at hvalid
  rw [List.ofFn_const]
  exact hvalid

/-- For every positive wedge count, the all-`1` formula is the unique valid
formula over the singleton alphabet `{1}`. -/
lemma SR_1_exact_formula (n : ℕ+) : SR_1 n = 1 := by
  unfold SR_1 SR countFormulas
  apply Nat.card_eq_one_iff_exists.mpr
  let oneFormula :
      {w : Formula ((n : ℕ) - 1) // Valid w ∧ ∀ i, w i ∈ ({1} : Finset Rotation)} :=
    ⟨fun _ => 1, ⟨all1Formula_valid _, by simp⟩⟩
  refine ⟨oneFormula, ?_⟩
  intro formula
  apply Subtype.ext
  funext i
  simpa using formula.property.2 i

/-- Any word beginning with four `2` rotations has the same first five wedges,
independently of its later rotations. -/
lemma fourTwos_wedges_prefix (rest : List Rotation) :
    (wedges (List.replicate 4 (2 : Rotation) ++ rest)).take 5 =
      wedges (List.replicate 4 (2 : Rotation)) := by
  cases rest with
  | nil => native_decide
  | cons r rest =>
      simp [wedges, wedgesFromDirections, centersFromDirections, directions,
        directionsFrom, directionTail, rotateQuarter]

/-- Four initial `2` rotations already force a collision: the fifth wedge
repeats the first, and no continuation can remove that overlap. -/
lemma fourTwos_not_collisionFree (rest : List Rotation) :
    ¬collisionFree (List.replicate 4 (2 : Rotation) ++ rest) := by
  intro h
  unfold collisionFree at h
  have hprefix :
      ((wedges (List.replicate 4 (2 : Rotation) ++ rest)).take 5).Pairwise
        interiorDisjoint :=
    h.take
  rw [fourTwos_wedges_prefix] at hprefix
  have hinvalid :
      ¬collisionFree (List.replicate 4 (2 : Rotation)) := by
    native_decide
  exact hinvalid hprefix

/-- An all-`2` formula is valid exactly through three rotations, corresponding
to at most four wedges rather than four rotations. -/
lemma allTwoFormula_valid_iff (k : ℕ) :
    Valid (fun _ : Fin k => (2 : Rotation)) ↔ k ≤ 3 := by
  constructor
  · intro hvalid
    by_contra hlength
    have hk : 4 ≤ k := by omega
    unfold Valid ValidList at hvalid
    rw [List.ofFn_const] at hvalid
    have hreplicate :
        List.replicate k (2 : Rotation) =
          List.replicate 4 (2 : Rotation) ++
            List.replicate (k - 4) (2 : Rotation) := by
      rw [← List.replicate_add]
      congr
      omega
    rw [hreplicate] at hvalid
    exact fourTwos_not_collisionFree _ hvalid
  · intro hk
    have hcases : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
    rcases hcases with rfl | rfl | rfl | rfl <;> native_decide

/-- The all-`2` alphabet admits one valid formula for up to four wedges and
none for longer snakes. -/
lemma SR_2_exact_formula (n : ℕ+) : SR_2 n = if n ≤ 4 then 1 else 0 := by
  unfold SR_2 SR countFormulas
  by_cases hn : n ≤ 4
  · rw [if_pos hn]
    apply Nat.card_eq_one_iff_exists.mpr
    let twoFormula :
        {w : Formula ((n : ℕ) - 1) //
          Valid w ∧ ∀ i, w i ∈ ({2} : Finset Rotation)} :=
      ⟨fun _ => 2, ⟨(allTwoFormula_valid_iff _).mpr (by
        exact Nat.sub_le_iff_le_add.mpr hn), by simp⟩⟩
    refine ⟨twoFormula, ?_⟩
    intro formula
    apply Subtype.ext
    funext i
    simpa using formula.property.2 i
  · rw [if_neg hn]
    apply Finite.card_eq_zero_iff.mpr
    constructor
    intro formula
    have hformula :
        formula.val = (fun _ : Fin ((n : ℕ) - 1) => (2 : Rotation)) := by
      funext i
      simpa using formula.property.2 i
    have hvalid : Valid (fun _ : Fin ((n : ℕ) - 1) => (2 : Rotation)) := by
      rw [← hformula]
      exact formula.property.1
    have hlength := (allTwoFormula_valid_iff _).mp hvalid
    change ¬(n : ℕ) ≤ 4 at hn
    omega

/-- Every positive wedge count has exactly one valid all-`3` formula, obtained
by reflecting the all-`1` formula. -/
lemma SR_3_exact_formula (n : ℕ+) : SR_3 n = 1 := by
  have hmirror := congrFun (mirrorSameCount (allowed := {1})) n
  rw [show mirrorAllowedRotations ({1} : Finset Rotation) = {3} by
    native_decide] at hmirror
  unfold SR_3
  rw [← hmirror]
  exact SR_1_exact_formula n

/-- Counts when 2 rotations are allowed. -/
def SR_01 := SR {0, 1}
/-- Counts of `n`-wedge snakes with each joint restricted to symbols `0` or `2`. -/
def SR_02 := SR {0, 2}
/-- Counts of `n`-wedge snakes with each joint restricted to symbols `1` or `2`. -/
def SR_12 := SR {1, 2}
/-- Counts of `n`-wedge snakes with each joint restricted to the quarter turns `1` or `3`. -/
def SR_13 := SR {1, 3}

/-- The alphabets `{0, 3}` and `{0, 1}` give equal counts at every wedge length
because reflection exchanges them. -/
lemma SR_03_equals_SR_01 : SR {0, 3} = SR_01 := by
  unfold SR_01
  simpa [show mirrorAllowedRotations ({0, 3} : Finset Rotation) = {0, 1} by
    native_decide] using (mirrorSameCount (allowed := {0, 3}))

/-- The alphabets `{2, 3}` and `{1, 2}` have identical wedge-count sequences
under reflection. -/
lemma SR_23_equals_SR_12 : SR {2, 3} = SR_12 := by
  unfold SR_12
  simpa [show mirrorAllowedRotations ({2, 3} : Finset Rotation) = {1, 2} by
    native_decide] using (mirrorSameCount (allowed := {2, 3}))


/-- Counts when 3 rotations are allowed. -/
def SR_013 := SR {0, 1, 3}
/-- Counts of `n`-wedge snakes allowing symbols `0`, `2`, and `3`, but excluding `1`. -/
def SR_023 := SR {0, 2, 3}
/-- Counts of `n`-wedge snakes allowing symbols `1`, `2`, and `3`, but excluding `0`. -/
def SR_123 := SR {1, 2, 3}

/-- Excluding `3` or excluding `1` gives the same count at every wedge length,
since the two three-symbol alphabets are reflections of one another. -/
lemma SR_012_equals_SR_023 : SR {0, 1, 2} = SR_023 := by
  unfold SR_023
  simpa [show mirrorAllowedRotations ({0, 1, 2} : Finset Rotation) = {0, 2, 3} by
    native_decide] using (mirrorSameCount (allowed := {0, 1, 2}))

end

/-- Simple upper bound: SR_r(n) ≤ |r|^(n-1). -/
lemma SR_upper_bound (r: Finset Rotation) (n : ℕ+) :
    (SR r n) ≤ r.card ^ ((n : ℕ) - 1) := by
  unfold SR countFormulas
  let restrictFormula :
      {w : Formula ((n : ℕ) - 1) // Valid w ∧ ∀ i, w i ∈ r} →
        Fin ((n : ℕ) - 1) → {x : Rotation // x ∈ r} :=
    fun w i => ⟨w.val i, w.property.2 i⟩
  have hinjective : Function.Injective restrictFormula := by
    intro a b hab
    apply Subtype.ext
    funext i
    exact congrArg (fun w => (w i).val) hab
  calc
    Nat.card
        {w : Formula ((n : ℕ) - 1) // Valid w ∧ ∀ i, w i ∈ r} ≤
        Nat.card (Fin ((n : ℕ) - 1) → {x : Rotation // x ∈ r}) :=
      Nat.card_le_card_of_injective restrictFormula hinjective
    _ = r.card ^ ((n : ℕ) - 1) := by simp


end RubiksSnake

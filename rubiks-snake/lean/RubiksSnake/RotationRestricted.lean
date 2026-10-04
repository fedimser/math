import RubiksSnake.Definitions
import RubiksSnake.SnAsymptotic
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


/-- Mirroring allowed rotations doesn't affect count. -/
def mirrorRotation (r: Rotation) := if r =1 ∨ r=3 then 4-r else r
def mirrorAllowedRotations (X : Finset Rotation) := X.image mirrorRotation

def mirrorVec (v : Vec3) : Vec3 :=
  fun i => if i = 2 then -v i else v i

@[simp] lemma mirrorVec_involutive (v : Vec3) : mirrorVec (mirrorVec v) = v := by
  funext i
  fin_cases i <;> simp [mirrorVec]

lemma mirrorVec_injective : Function.Injective mirrorVec :=
  Function.Involutive.injective mirrorVec_involutive

@[simp] lemma mirrorVec_zero : mirrorVec zeroVec = zeroVec := by
  funext i
  simp [mirrorVec, zeroVec]

@[simp] lemma mirrorVec_ex : mirrorVec ex = ex := by
  funext i
  fin_cases i <;> simp [mirrorVec, ex]

@[simp] lemma mirrorVec_ey : mirrorVec ey = ey := by
  funext i
  fin_cases i <;> simp [mirrorVec, ey]

@[simp] lemma mirrorVec_add (u v : Vec3) :
    mirrorVec (addVec u v) = addVec (mirrorVec u) (mirrorVec v) := by
  funext i
  fin_cases i <;> simp [mirrorVec, addVec]; ring

@[simp] lemma mirrorVec_neg (v : Vec3) :
    mirrorVec (negVec v) = negVec (mirrorVec v) := by
  funext i
  fin_cases i <;> simp [mirrorVec, negVec]

@[simp] lemma negVec_negVec (v : Vec3) : negVec (negVec v) = v := by
  funext i
  simp [negVec]

lemma mirrorVec_cross (u v : Vec3) :
    mirrorVec (cross u v) = negVec (cross (mirrorVec u) (mirrorVec v)) := by
  funext i
  fin_cases i <;> simp [mirrorVec, cross, negVec] <;> ring

@[simp] lemma mirrorRotation_involutive (r : Rotation) :
    mirrorRotation (mirrorRotation r) = r := by
  fin_cases r <;> native_decide

lemma mirrorRotation_injective : Function.Injective mirrorRotation :=
  Function.Involutive.injective mirrorRotation_involutive

lemma rotateQuarter_mirror (axis : Vec3) (r : Rotation) (v : Vec3) :
    mirrorVec (rotateQuarter axis r v) =
      rotateQuarter (mirrorVec axis) (mirrorRotation r) (mirrorVec v) := by
  fin_cases r
  · change mirrorVec (rotateQuarter axis 0 v) =
      rotateQuarter (mirrorVec axis) (mirrorRotation 0) (mirrorVec v)
    rw [show mirrorRotation 0 = 0 by native_decide]
    simp [rotateQuarter]
  · change mirrorVec (rotateQuarter axis 1 v) =
      rotateQuarter (mirrorVec axis) (mirrorRotation 1) (mirrorVec v)
    rw [show mirrorRotation 1 = 3 by native_decide]
    simp [rotateQuarter, mirrorVec_cross]
  · change mirrorVec (rotateQuarter axis 2 v) =
      rotateQuarter (mirrorVec axis) (mirrorRotation 2) (mirrorVec v)
    rw [show mirrorRotation 2 = 2 by native_decide]
    simp [rotateQuarter]
  · change mirrorVec (rotateQuarter axis 3 v) =
      rotateQuarter (mirrorVec axis) (mirrorRotation 3) (mirrorVec v)
    rw [show mirrorRotation 3 = 1 by native_decide]
    simp [rotateQuarter, mirrorVec_cross]

lemma directionTail_mirror (previous axis : Vec3) (rotations : List Rotation) :
    directionTail (mirrorVec previous) (mirrorVec axis)
        (rotations.map mirrorRotation) =
      (directionTail previous axis rotations).map mirrorVec := by
  induction rotations generalizing previous axis with
  | nil => simp [directionTail]
  | cons r rotations ih =>
      simp only [List.map_cons, directionTail]
      rw [rotateQuarter_mirror]
      have htail := ih axis (rotateQuarter axis r previous)
      rw [rotateQuarter_mirror] at htail
      exact congrArg
        (rotateQuarter (mirrorVec axis) (mirrorRotation r) (mirrorVec previous) :: ·)
        htail

lemma directions_mirror (rotations : List Rotation) :
    directions (rotations.map mirrorRotation) =
      (directions rotations).map mirrorVec := by
  unfold directions directionsFrom
  simp only [List.map_cons, mirrorVec_ey, mirrorVec_ex]
  exact congrArg (ey :: ex :: ·)
    (by simpa only [mirrorVec_ey, mirrorVec_ex] using
      directionTail_mirror ey ex rotations)

lemma scanl_addVec_mirror (start : Vec3) (steps : List Vec3) :
    List.scanl addVec (mirrorVec start) (steps.map mirrorVec) =
      (List.scanl addVec start steps).map mirrorVec := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      simp only [List.map_cons, List.scanl_cons]
      rw [← mirrorVec_add]
      exact congrArg (mirrorVec start :: ·) (ih (addVec start step))

lemma centersFromDirections_mirror (ds : List Vec3) :
    centersFromDirections (ds.map mirrorVec) =
      (centersFromDirections ds).map mirrorVec := by
  unfold centersFromDirections
  rw [show ((ds.map mirrorVec).drop 1).dropLast =
      ((ds.drop 1).dropLast).map mirrorVec by simp]
  simpa only [mirrorVec_zero] using
    scanl_addVec_mirror zeroVec ((ds.drop 1).dropLast)

def mirrorWedge (w : Wedge) : Wedge :=
  ⟨mirrorVec w.center, mirrorVec w.entrance, mirrorVec w.exit⟩

lemma zip_map_map {α β γ δ : Type} (f : α → γ) (g : β → δ)
    (xs : List α) (ys : List β) :
    (xs.map f).zip (ys.map g) =
      (xs.zip ys).map fun p => (f p.1, g p.2) := by
  induction xs generalizing ys with
  | nil => simp
  | cons x xs ih =>
      cases ys with
      | nil => simp
      | cons y ys => simp [ih]

lemma wedgesFromDirections_mirror (ds : List Vec3) :
    wedgesFromDirections (ds.map mirrorVec) =
      (wedgesFromDirections ds).map mirrorWedge := by
  unfold wedgesFromDirections
  rw [centersFromDirections_mirror]
  have htail : (ds.map mirrorVec).tail = ds.tail.map mirrorVec := by
    cases ds <;> rfl
  rw [htail, zip_map_map mirrorVec mirrorVec ds ds.tail]
  rw [zip_map_map mirrorVec
    (fun p : Vec3 × Vec3 => (mirrorVec p.1, mirrorVec p.2))]
  simp only [List.map_map]
  congr 1
  funext p
  cases p with
  | mk center directions =>
      cases directions
      simp [mirrorWedge]

lemma wedges_mirror (rotations : List Rotation) :
    wedges (rotations.map mirrorRotation) =
      (wedges rotations).map mirrorWedge := by
  unfold wedges
  rw [directions_mirror, wedgesFromDirections_mirror]

@[simp] lemma mirrorVec_eq_iff (u v : Vec3) :
    mirrorVec u = mirrorVec v ↔ u = v :=
  mirrorVec_injective.eq_iff

lemma interiorDisjoint_mirror (a b : Wedge) :
    interiorDisjoint (mirrorWedge a) (mirrorWedge b) ↔
      interiorDisjoint a b := by
  unfold interiorDisjoint sameUnorderedPair mirrorWedge
  dsimp only
  rw [show negVec (mirrorVec b.entrance) =
      mirrorVec (negVec b.entrance) by simp]
  rw [show negVec (mirrorVec b.exit) =
      mirrorVec (negVec b.exit) by simp]
  have hcenter :
      mirrorVec a.center ≠ mirrorVec b.center ↔ a.center ≠ b.center :=
    mirrorVec_injective.ne_iff
  simp only [hcenter, mirrorVec_eq_iff]

lemma collisionFree_mirror (rotations : List Rotation) :
    collisionFree (rotations.map mirrorRotation) ↔ collisionFree rotations := by
  unfold collisionFree
  rw [wedges_mirror]
  simp only [List.pairwise_map]
  generalize wedges rotations = ws
  induction ws with
  | nil => simp
  | cons a ws ih =>
      simp only [List.pairwise_cons]
      constructor
      · intro h
        refine ⟨?_, ih.mp h.2⟩
        intro b hb
        exact (interiorDisjoint_mirror a b).mp (h.1 b hb)
      · intro h
        refine ⟨?_, ih.mpr h.2⟩
        intro b hb
        exact (interiorDisjoint_mirror a b).mpr (h.1 b hb)

def mirrorFormula {n : ℕ} (w : Formula n) : Formula n :=
  fun i => mirrorRotation (w i)

@[simp] lemma mirrorFormula_involutive {n : ℕ} (w : Formula n) :
    mirrorFormula (mirrorFormula w) = w := by
  funext i
  simp [mirrorFormula]

lemma ofFn_mirrorFormula {n : ℕ} (w : Formula n) :
    List.ofFn (mirrorFormula w) = (List.ofFn w).map mirrorRotation := by
  unfold mirrorFormula
  exact List.ofFn_comp' w mirrorRotation

lemma valid_mirrorFormula {n : ℕ} (w : Formula n) :
    Valid (mirrorFormula w) ↔ Valid w := by
  unfold Valid ValidList
  rw [ofFn_mirrorFormula, collisionFree_mirror]

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

def mirrorToRestricted (allowed : Finset Rotation) {n : ℕ}
    (w : {w : Formula n // Valid w ∧ ∀ i, w i ∈ allowed}) :
    {w : Formula n //
      Valid w ∧ ∀ i, w i ∈ mirrorAllowedRotations allowed} := by
  refine ⟨mirrorFormula w.val, ?_, ?_⟩
  · exact (valid_mirrorFormula w.val).mpr w.property.1
  · intro i
    exact Finset.mem_image.mpr ⟨w.val i, w.property.2 i, rfl⟩

def mirrorFromRestricted (allowed : Finset Rotation) {n : ℕ}
    (w : {w : Formula n //
      Valid w ∧ ∀ i, w i ∈ mirrorAllowedRotations allowed}) :
    {w : Formula n // Valid w ∧ ∀ i, w i ∈ allowed} := by
  refine ⟨mirrorFormula w.val, ?_, ?_⟩
  · exact (valid_mirrorFormula w.val).mpr w.property.1
  · intro i
    exact (mem_mirrorAllowedRotations_iff allowed (w.val i)).mp
      (w.property.2 i)

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

lemma mirrorSameCount: SR allowed  = SR (mirrorAllowedRotations allowed) := by
  funext n
  unfold SR countFormulas
  exact Nat.card_congr
    (mirrorRestrictedEquiv allowed ((n : ℕ) - 1))




/-- Counts when only one rotation is allowed. -/
def SR_0 := SR {0}
def SR_1 := SR {1}
def SR_2 := SR {2}
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

lemma fourTwos_wedges_prefix (rest : List Rotation) :
    (wedges (List.replicate 4 (2 : Rotation) ++ rest)).take 5 =
      wedges (List.replicate 4 (2 : Rotation)) := by
  cases rest with
  | nil => native_decide
  | cons r rest =>
      simp [wedges, wedgesFromDirections, centersFromDirections, directions,
        directionsFrom, directionTail, rotateQuarter]

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

lemma SR_3_exact_formula (n : ℕ+) : SR_3 n = 1 := by
  have hmirror := congrFun (mirrorSameCount (allowed := {1})) n
  rw [show mirrorAllowedRotations ({1} : Finset Rotation) = {3} by
    native_decide] at hmirror
  unfold SR_3
  rw [← hmirror]
  exact SR_1_exact_formula n

/-- Counts when 2 rotations are allowed. -/
def SR_01 := SR {0, 1}
def SR_02 := SR {0, 2}
def SR_12 := SR {1, 2}
def SR_13 := SR {1, 3}

lemma SR_03_equals_SR_01 : SR {0, 3} = SR_01 := by
  unfold SR_01
  simpa [show mirrorAllowedRotations ({0, 3} : Finset Rotation) = {0, 1} by
    native_decide] using (mirrorSameCount (allowed := {0, 3}))

lemma SR_23_equals_SR_12 : SR {2, 3} = SR_12 := by
  unfold SR_12
  simpa [show mirrorAllowedRotations ({2, 3} : Finset Rotation) = {1, 2} by
    native_decide] using (mirrorSameCount (allowed := {2, 3}))


/-- Counts when 3 rotations are allowed. -/
def SR_013 := SR {0, 1, 3}
def SR_023 := SR {0, 2, 3}
def SR_123 := SR {1, 2, 3}

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

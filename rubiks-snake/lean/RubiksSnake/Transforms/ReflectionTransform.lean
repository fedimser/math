import RubiksSnake.Transforms.FormulaTransform
import RubiksSnake.Geometry

import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-! Reflection of Rubik's Snake formulas and its geometric invariance. -/

namespace RubiksSnake

/-- Reverse a joint's handedness: exchange quarter-turn symbols `1` and `3`,
leaving `0` and `2` fixed. -/
def mirrorRotation (r : Rotation) := if r = 1 ∨ r = 3 then 4 - r else r

/-- Reflect a lattice vector across the coordinate plane spanned by `ex` and `ey`. -/
def mirrorVec (v : Vec3) : Vec3 :=
  fun i => if i = 2 then -v i else v i

/-- Reflecting a vector twice restores all three coordinates. -/
@[simp] lemma mirrorVec_involutive (v : Vec3) : mirrorVec (mirrorVec v) = v := by
  funext i
  fin_cases i <;> simp [mirrorVec]

/-- Distinct lattice vectors remain distinct under reflection. -/
lemma mirrorVec_injective : Function.Injective mirrorVec :=
  Function.Involutive.injective mirrorVec_involutive

/-- The reflecting plane contains the origin, so reflection fixes it. -/
@[simp] lemma mirrorVec_zero : mirrorVec zeroVec = zeroVec := by
  funext i
  simp [mirrorVec, zeroVec]

/-- Reflection fixes the canonical initial exit direction along the first axis. -/
@[simp] lemma mirrorVec_ex : mirrorVec ex = ex := by
  funext i
  fin_cases i <;> simp [mirrorVec, ex]

/-- Reflection fixes the canonical initial incoming travel direction along the second axis. -/
@[simp] lemma mirrorVec_ey : mirrorVec ey = ey := by
  funext i
  fin_cases i <;> simp [mirrorVec, ey]

/-- Reflecting a sum of displacements agrees with adding their reflected vectors. -/
@[simp] lemma mirrorVec_add (u v : Vec3) :
    mirrorVec (addVec u v) = addVec (mirrorVec u) (mirrorVec v) := by
  funext i
  fin_cases i <;> simp [mirrorVec, addVec]
  ring

/-- Reflection commutes with reversing a displacement or face direction. -/
@[simp] lemma mirrorVec_neg (v : Vec3) :
    mirrorVec (negVec v) = negVec (mirrorVec v) := by
  funext i
  fin_cases i <;> simp [mirrorVec, negVec]

/-- Reversing a vector's direction twice restores the original vector. -/
@[simp] lemma negVec_negVec (v : Vec3) : negVec (negVec v) = v := by
  funext i
  simp [negVec]

/-- Reflection reverses orientation, so a reflected cross product has the opposite
sign from the cross product of the reflected inputs. -/
lemma mirrorVec_cross (u v : Vec3) :
    mirrorVec (cross u v) = negVec (cross (mirrorVec u) (mirrorVec v)) := by
  funext i
  fin_cases i <;> simp [mirrorVec, cross, negVec] <;> ring

/-- Reversing a joint's handedness twice restores its rotation symbol. -/
@[simp] lemma mirrorRotation_involutive (r : Rotation) :
    mirrorRotation (mirrorRotation r) = r := by
  fin_cases r <;> native_decide

/-- Reflection never identifies two different joint settings. -/
lemma mirrorRotation_injective : Function.Injective mirrorRotation :=
  Function.Involutive.injective mirrorRotation_involutive

/-- Reflecting a turn is equivalent to reflecting its input vectors and reversing
the rotation symbol's handedness. -/
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

/-- Reflecting the initial frame and every rotation symbol reflects every
subsequently generated travel direction. -/
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

/-- Mirroring a rotation word reflects its complete travel-direction list;
the canonical initial pair stays fixed in the reflecting plane. -/
lemma directions_mirror (rotations : List Rotation) :
    directions (rotations.map mirrorRotation) =
      (directions rotations).map mirrorVec := by
  unfold directions directionsFrom
  simp only [List.map_cons, mirrorVec_ey, mirrorVec_ex]
  exact congrArg (ey :: ex :: ·)
    (by simpa only [mirrorVec_ey, mirrorVec_ex] using
      directionTail_mirror ey ex rotations)

/-- Accumulating reflected steps from a reflected start reflects every position
of the original path. -/
lemma scanl_addVec_mirror (start : Vec3) (steps : List Vec3) :
    List.scanl addVec (mirrorVec start) (steps.map mirrorVec) =
      (List.scanl addVec start steps).map mirrorVec := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      simp only [List.map_cons, List.scanl_cons]
      rw [← mirrorVec_add]
      exact congrArg (mirrorVec start :: ·) (ih (addVec start step))

/-- Reflecting all travel directions reflects all accumulated cell centers. -/
lemma centersFromDirections_mirror (ds : List Vec3) :
    centersFromDirections (ds.map mirrorVec) =
      (centersFromDirections ds).map mirrorVec := by
  unfold centersFromDirections
  rw [show ((ds.map mirrorVec).drop 1).dropLast =
      ((ds.drop 1).dropLast).map mirrorVec by simp]
  simpa only [mirrorVec_zero] using
    scanl_addVec_mirror zeroVec ((ds.drop 1).dropLast)

/-- Reflect a wedge's cell center and both outward face directions in the same plane. -/
def mirrorWedge (w : Wedge) : Wedge :=
  ⟨mirrorVec w.center, mirrorVec w.entrance, mirrorVec w.exit⟩

/-- Mapping two lists before zipping is equivalent to mapping each component of
their zipped pairs, even when their lengths differ. -/
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

/-- Constructing wedges from reflected travel directions reflects their cell
centers and outward faces, without changing their order. -/
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

/-- Exchanging symbols `1` and `3` in a formula reflects every wedge in its geometry. -/
lemma wedges_mirror (rotations : List Rotation) :
    wedges (rotations.map mirrorRotation) =
      (wedges rotations).map mirrorWedge := by
  unfold wedges
  rw [directions_mirror, wedgesFromDirections_mirror]

/-- Two reflected vectors coincide exactly when their original vectors coincide. -/
@[simp] lemma mirrorVec_eq_iff (u v : Vec3) :
    mirrorVec u = mirrorVec v ↔ u = v :=
  mirrorVec_injective.eq_iff

/-- Reflecting both wedges preserves the collision test, including the
complementary-face exception at a shared cell center. -/
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

/-- A rotation word is collision-free exactly when its mirror word is collision-free. -/
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

/-- Reflect an `n`-rotation formula by reversing each joint's handedness,
without reversing the order of its rotations. -/
def mirrorFormula {n : ℕ} (w : Formula n) : Formula n :=
  fun i => mirrorRotation (w i)

/-- Reflecting every joint twice restores the complete indexed formula. -/
@[simp] lemma mirrorFormula_involutive {n : ℕ} (w : Formula n) :
    mirrorFormula (mirrorFormula w) = w := by
  funext i
  simp [mirrorFormula]

/-- Reading a reflected indexed formula as a list mirrors each entry of the original list. -/
lemma ofFn_mirrorFormula {n : ℕ} (w : Formula n) :
    List.ofFn (mirrorFormula w) = (List.ofFn w).map mirrorRotation := by
  unfold mirrorFormula
  exact List.ofFn_comp' w mirrorRotation

/-- Reflection preserves and reflects validity of all `n + 1` wedges of an
`n`-rotation formula. -/
lemma valid_mirrorFormula {n : ℕ} (w : Formula n) :
    Valid (mirrorFormula w) ↔ Valid w := by
  unfold Valid ValidList
  rw [ofFn_mirrorFormula, collisionFree_mirror]

/-- Reflection as an involutive validity-preserving formula transform. -/
def reflectionTransform (n : ℕ) : InvolutiveFormulaTransform n where
  toFun := mirrorFormula
  involutive := mirrorFormula_involutive
  valid_iff := valid_mirrorFormula

end RubiksSnake

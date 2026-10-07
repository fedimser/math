import RubiksSnake.ReflectionTransform
import RubiksSnake.FormulaTransform

import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-! Definition of formula reversal and proofs of its properties. -/

namespace RubiksSnake


/-- Head-tail reversal of an `n`-rotation formula: reverse the index order
without changing any rotation symbol. -/
def reverseFormula {n : ℕ} (w : Formula n) : Formula n :=
  fun i => w i.rev

/-- Exchanging the head and tail twice restores the original indexed formula. -/
@[simp] lemma reverseFormula_involutive {n : ℕ} (w : Formula n) :
    reverseFormula (reverseFormula w) = w := by
  funext i
  simp [reverseFormula]

/-- An invertible lattice frame change preserving the origin, addition, negation,
and cross products. Cross-product preservation distinguishes these frame changes
from orientation-reversing reflections. -/
structure RigidVecEquiv where
  toEquiv : Vec3 ≃ Vec3
  map_zero : toEquiv zeroVec = zeroVec
  map_add : ∀ u v, toEquiv (addVec u v) =
    addVec (toEquiv u) (toEquiv v)
  map_neg : ∀ v, toEquiv (negVec v) = negVec (toEquiv v)
  map_cross : ∀ u v, toEquiv (cross u v) =
    cross (toEquiv u) (toEquiv v)

/-- Apply a bundled lattice frame change directly to a vector. -/
instance : CoeFun RigidVecEquiv (fun _ => Vec3 → Vec3) :=
  ⟨fun e => e.toEquiv⟩

/-- The identity lattice frame, leaving every position and direction unchanged. -/
def RigidVecEquiv.refl : RigidVecEquiv where
  toEquiv := Equiv.refl _
  map_zero := rfl
  map_add _ _ := rfl
  map_neg _ := rfl
  map_cross _ _ := rfl

/-- Compose lattice frame changes, applying `e` first and then `f`. -/
def RigidVecEquiv.trans (e f : RigidVecEquiv) : RigidVecEquiv where
  toEquiv := e.toEquiv.trans f.toEquiv
  map_zero := by simp [e.map_zero, f.map_zero]
  map_add u v := by simp [e.map_add, f.map_add]
  map_neg v := by simp [e.map_neg, f.map_neg]
  map_cross u v := by simp [e.map_cross, f.map_cross]

/-- The inverse frame change, with the same origin and cross-product preservation laws. -/
def RigidVecEquiv.symm (e : RigidVecEquiv) : RigidVecEquiv where
  toEquiv := e.toEquiv.symm
  map_zero := e.toEquiv.injective <| by simp [e.map_zero]
  map_add u v := e.toEquiv.injective <| by
    simp [e.map_add]
  map_neg v := e.toEquiv.injective <| by
    simp [e.map_neg]
  map_cross u v := e.toEquiv.injective <| by
    simp [e.map_cross]

/-- The canonical frame update for joint setting `r`: send `ey` to the old axis
`ex`, and send `ex` to the newly rotated outgoing direction. -/
def frameStep (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => ![v 1, v 0, -v 2]
  | 1 => ![v 1, v 2, v 0]
  | 2 => ![v 1, -v 0, v 2]
  | _ => ![v 1, -v 2, -v 0]

/-- The explicit inverse coordinate permutation and sign changes for one frame update. -/
def frameStepInv (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => ![v 1, v 0, -v 2]
  | 1 => ![v 2, v 0, v 1]
  | 2 => ![-v 1, v 0, v 2]
  | _ => ![-v 2, v 0, -v 1]

/-- A single joint's frame update packaged as an invertible, cross-product-preserving map. -/
def frameStepEquiv (r : Rotation) : RigidVecEquiv where
  toEquiv :=
    { toFun := frameStep r
      invFun := frameStepInv r
      left_inv := by
        intro v
        funext i
        fin_cases r <;> fin_cases i <;> simp [frameStep, frameStepInv]
      right_inv := by
        intro v
        funext i
        fin_cases r <;> fin_cases i <;> simp [frameStep, frameStepInv] }
  map_zero := by
    funext i
    fin_cases r <;> fin_cases i <;> simp [frameStep, zeroVec]
  map_add := by
    intro u v
    funext i
    fin_cases r <;> fin_cases i <;> simp [frameStep, addVec] <;> ring
  map_neg := by
    intro v
    funext i
    fin_cases r <;> fin_cases i <;> simp [frameStep, negVec]
  map_cross := by
    intro u v
    funext i
    fin_cases r <;> fin_cases i <;> simp [frameStep, cross] <;> ring

/-- After one joint, the old outgoing axis becomes the new previous travel direction. -/
@[simp] lemma frameStep_ey (r : Rotation) :
    frameStep r ey = ex := by
  funext i
  fin_cases r <;> fin_cases i <;> simp [frameStep, ey, ex]

/-- The canonical frame update sends its outgoing axis to the direction selected
by rotating `ey` about `ex`. -/
@[simp] lemma frameStep_ex (r : Rotation) :
    frameStep r ex = rotateQuarter ex r ey := by
  funext i
  fin_cases r <;> fin_cases i <;>
    simp [frameStep, rotateQuarter, cross, negVec, ex, ey]

/-- An orientation-preserving lattice frame change commutes with the turn rule
without changing the rotation symbol. -/
lemma RigidVecEquiv.map_rotateQuarter (e : RigidVecEquiv)
    (axis : Vec3) (r : Rotation) (v : Vec3) :
    e (rotateQuarter axis r v) = rotateQuarter (e axis) r (e v) := by
  fin_cases r
  · rfl
  · exact e.map_cross axis v
  · exact e.map_neg v
  · change e (negVec (cross axis v)) =
      negVec (cross (e axis) (e v))
    rw [e.map_neg, e.map_cross]

/-- Update the current frame by a joint rotation, applying the canonical step
inside the existing frame `e`. -/
def advanceFrame (e : RigidVecEquiv) (r : Rotation) : RigidVecEquiv :=
  (frameStepEquiv r).trans e

/-- The frame reached after a rotation word; its images of `ey` and `ex` are
the final two travel directions, in their original order. -/
def terminalFrameFrom : RigidVecEquiv → List Rotation → RigidVecEquiv
  | e, [] => e
  | e, r :: rs => terminalFrameFrom (advanceFrame e r) rs

/-- Advancing a frame makes its old outgoing direction the new previous direction. -/
@[simp] lemma advanceFrame_ey (e : RigidVecEquiv) (r : Rotation) :
    advanceFrame e r ey = e ex := by
  change e (frameStep r ey) = e ex
  rw [frameStep_ey]

/-- The advanced frame's exit direction is obtained by turning the previous
direction around the current axis by the given joint setting. -/
@[simp] lemma advanceFrame_ex (e : RigidVecEquiv) (r : Rotation) :
    advanceFrame e r ex = rotateQuarter (e ex) r (e ey) := by
  change e (frameStep r ex) = rotateQuarter (e ex) r (e ey)
  rw [frameStep_ex, e.map_rotateQuarter]

/-- Separate the first travel direction and continue with the pair produced
by the first rotation. -/
lemma directionsFrom_cons (previous axis : Vec3) (r : Rotation)
    (rs : List Rotation) :
    directionsFrom previous axis (r :: rs) =
      previous :: directionsFrom axis (rotateQuarter axis r previous) rs := by
  rfl

/-- Removing the first direction of a framed path leaves the path generated
from the frame advanced through its first joint. -/
lemma directionsFrom_cons_frame (e : RigidVecEquiv) (r : Rotation)
    (rs : List Rotation) :
    directionsFrom (e ey) (e ex) (r :: rs) =
      e ey :: directionsFrom (advanceFrame e r ey)
        (advanceFrame e r ex) rs := by
  simp [directionsFrom, directionTail]

/-- Appending a joint preserves all existing directions and adds the exit
direction of the newly advanced terminal frame. -/
lemma directionsFrom_append_singleton_frame (e : RigidVecEquiv)
    (rs : List Rotation) (r : Rotation) :
    directionsFrom (e ey) (e ex) (rs ++ [r]) =
      directionsFrom (e ey) (e ex) rs ++
        [(advanceFrame (terminalFrameFrom e rs) r) ex] := by
  induction rs generalizing e with
  | nil =>
      simp [directionsFrom, directionTail, terminalFrameFrom]
  | cons s rs ih =>
      rw [List.cons_append, directionsFrom_cons_frame,
        directionsFrom_cons_frame, ih]
      rfl

/-- The terminal frame of a word extended by one joint is one update of the
original terminal frame. -/
lemma terminalFrameFrom_append_singleton (e : RigidVecEquiv)
    (rs : List Rotation) (r : Rotation) :
    terminalFrameFrom e (rs ++ [r]) =
      advanceFrame (terminalFrameFrom e rs) r := by
  induction rs generalizing e with
  | nil => rfl
  | cons s rs ih => exact ih (advanceFrame e s)

/-- Traversing a joint backwards uses the same rotation symbol: negating the
axis and the new direction recovers the negated old previous direction. -/
lemma rotateQuarter_reverse_step (e : RigidVecEquiv) (r : Rotation) :
    rotateQuarter (negVec (e ex)) r
        (negVec (advanceFrame e r ex)) = negVec (e ey) := by
  rw [advanceFrame_ex]
  calc
    rotateQuarter (negVec (e ex)) r
        (negVec (rotateQuarter (e ex) r (e ey))) =
        e (rotateQuarter (negVec ex) r
          (negVec (rotateQuarter ex r ey))) := by
            rw [e.map_rotateQuarter, e.map_neg, e.map_neg,
              e.map_rotateQuarter]
    _ = e (negVec ey) := by
      apply congrArg e
      funext i
      fin_cases r <;> fin_cases i <;>
        simp [rotateQuarter, cross, negVec, ex, ey]
    _ = negVec (e ey) := e.map_neg ey

/-- Starting from the negated, swapped terminal pair and reversing the rotation
word reverses and negates the complete travel-direction list. -/
lemma directionsFrom_reverse_frame (e : RigidVecEquiv)
    (rs : List Rotation) :
    directionsFrom
        (negVec (terminalFrameFrom e rs ex))
        (negVec (terminalFrameFrom e rs ey)) rs.reverse =
      (directionsFrom (e ey) (e ex) rs).reverse.map negVec := by
  induction rs using List.reverseRecOn generalizing e with
  | nil =>
      simp [directionsFrom, directionTail, terminalFrameFrom]
  | append_singleton rs r ih =>
      rw [terminalFrameFrom_append_singleton, List.reverse_append,
        List.reverse_singleton, List.singleton_append, directionsFrom_cons,
        directionsFrom_append_singleton_frame,
        List.reverse_append, List.map_append]
      simp only [advanceFrame_ey, advanceFrame_ex]
      have hback :=
        rotateQuarter_reverse_step (terminalFrameFrom e rs) r
      rw [advanceFrame_ex] at hback
      rw [hback]
      simp only [List.reverse_singleton, List.map_singleton,
        List.singleton_append]
      simpa [terminalFrameFrom_append_singleton] using
        congrArg (negVec (advanceFrame (terminalFrameFrom e rs) r ex) :: ·)
          (ih e)

/-- The total lattice displacement of a list of steps, starting from the origin. -/
def sumVec (steps : List Vec3) : Vec3 :=
  steps.foldl addVec zeroVec

/-- Translating the initial position translates every accumulated position by
the same offset, with all step vectors unchanged. -/
lemma scanl_addVec_translate (offset start : Vec3) (steps : List Vec3) :
    List.scanl addVec (addVec offset start) steps =
      (List.scanl addVec start steps).map (addVec offset) := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      simp only [List.scanl_cons, List.map_cons]
      rw [addVec_assoc, ih]

/-- Appending one step adds that step to the path's total displacement. -/
lemma sumVec_append_singleton (steps : List Vec3) (step : Vec3) :
    sumVec (steps ++ [step]) = addVec (sumVec steps) step := by
  simp [sumVec, List.foldl_append]

/-- Appending one step preserves the existing position list and appends its new endpoint. -/
lemma scanl_addVec_append_singleton (steps : List Vec3) (step : Vec3) :
    List.scanl addVec zeroVec (steps ++ [step]) =
      List.scanl addVec zeroVec steps ++ [addVec (sumVec steps) step] := by
  rw [List.scanl_append]
  simp [sumVec]

/-- Traversing steps in reverse with opposite signs reverses the position list
and translates the old endpoint to the new origin. -/
lemma scanl_reverse_negVec (steps : List Vec3) :
    List.scanl addVec zeroVec (steps.reverse.map negVec) =
      (List.scanl addVec zeroVec steps).reverse.map
        (fun p => addVec p (negVec (sumVec steps))) := by
  induction steps using List.reverseRecOn with
  | nil => simp [sumVec]
  | append_singleton steps step ih =>
      rw [List.reverse_append, List.reverse_singleton, List.map_append,
        List.map_singleton, List.singleton_append, List.scanl_cons,
        scanl_addVec_append_singleton, List.reverse_append,
        List.reverse_singleton, List.singleton_append,
        sumVec_append_singleton]
      simp only [List.map_cons]
      congr 1
      · funext i
        simp [addVec, negVec, zeroVec]
      · rw [addVec_zero_left,
          ← addVec_zero_right (negVec step),
          scanl_addVec_translate (negVec step) zeroVec, ih,
          List.map_map]
        congr 1
        funext p
        funext i
        simp [addVec, negVec]
        ring

/-- For a list with at least two entries, removing its first and last entries
commutes with reversal followed by elementwise mapping. -/
lemma middle_reverse_map {α β : Type} (f : α → β)
    (first second : α) (rest : List α) :
    ((((first :: second :: rest).reverse.map f).drop 1).dropLast) =
      (((first :: second :: rest).drop 1).dropLast).reverse.map f := by
  rw [List.map_reverse, List.drop_one, List.drop_one]
  rw [List.tail_reverse, List.dropLast_reverse, List.map_reverse,
    List.reverse_inj, List.map_dropLast]
  simp

/-- Reverse a consecutive pair of travel directions by swapping and negating
both entries; these are travel directions, not outward wedge faces. -/
def reverseDirectionPair (p : Vec3 × Vec3) : Vec3 × Vec3 :=
  (negVec p.2, negVec p.1)

/-- Appending `b` to a list ending in `a` adds exactly the consecutive pair `(a, b)`. -/
lemma zipTail_append_two {α : Type} (xs : List α) (a b : α) :
    (xs ++ [a, b]).zip (xs ++ [a, b]).tail =
      (xs ++ [a]).zip (xs ++ [a]).tail ++ [(a, b)] := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      cases xs with
      | nil => simp
      | cons y xs =>
          simp only [List.cons_append, List.tail_cons, List.zip_cons_cons,
            List.cons.injEq, true_and]
          exact ih

/-- Reversing and negating a direction list reverses its consecutive pairs and
swaps and negates the two directions within each pair. -/
lemma zipTail_reverse_negVec (ds : List Vec3) :
    (ds.reverse.map negVec).zip (ds.reverse.map negVec).tail =
      (ds.zip ds.tail).reverse.map reverseDirectionPair := by
  induction ds using List.reverseRecOn with
  | nil => simp
  | append_singleton ds d ih =>
      cases ds using List.reverseRecOn with
      | nil => simp
      | append_singleton pre e =>
          simp only [List.reverse_append, List.reverse_singleton,
            List.singleton_append, List.map_cons, List.tail_cons,
            List.zip_cons_cons]
          have ih' := ih
          simp only [List.reverse_append, List.reverse_singleton,
            List.singleton_append, List.map_cons, List.tail_cons] at ih'
          rw [ih']
          rw [show pre ++ [e] ++ [d] = pre ++ [e, d] by simp]
          rw [zipTail_append_two]
          simp [reverseDirectionPair]

/-- For equally long lists, zipping their reversals gives the reversal of their zip. -/
lemma zip_reverse_of_length_eq {α β : Type} (xs : List α) (ys : List β)
    (h : xs.length = ys.length) :
    xs.reverse.zip ys.reverse = (xs.zip ys).reverse := by
  induction xs using List.reverseRecOn generalizing ys with
  | nil =>
      simp at h
      simp
  | append_singleton xs x ih =>
      have hys : ys ≠ [] := by
        intro hnil
        subst ys
        simp at h
      rw [← List.dropLast_append_getLast hys]
      have hlength : xs.length = ys.dropLast.length := by
        simp only [List.length_append, List.length_singleton,
          List.length_dropLast] at h ⊢
        omega
      rw [List.reverse_append, List.reverse_singleton,
        List.reverse_append, List.reverse_singleton,
        List.singleton_append, List.singleton_append,
        List.zip_cons_cons, ih ys.dropLast hlength,
        List.zip_append hlength, List.reverse_append]
      rfl

/-- Re-express a wedge for traversal from the other end: move the origin to
`endpoint` and swap entrance and exit faces, without negating those outward faces. -/
def reverseWedge (endpoint : Vec3) (w : Wedge) : Wedge :=
  ⟨addVec w.center (negVec endpoint), w.exit, w.entrance⟩

/-- Reversing and negating travel directions reverses the wedge list, swaps
entrance and exit roles, and translates the original last center to the origin. -/
lemma wedgesFromDirections_reverse (first second : Vec3)
    (rest : List Vec3) :
    wedgesFromDirections ((first :: second :: rest).reverse.map negVec) =
      (wedgesFromDirections (first :: second :: rest)).reverse.map
        (reverseWedge
          (sumVec (((first :: second :: rest).drop 1).dropLast))) := by
  unfold wedgesFromDirections centersFromDirections
  rw [middle_reverse_map, scanl_reverse_negVec,
    zipTail_reverse_negVec]
  rw [zip_map_map
    (fun p => addVec p
      (negVec (sumVec (((first :: second :: rest).drop 1).dropLast))))
    reverseDirectionPair]
  rw [zip_reverse_of_length_eq]
  · simp only [List.map_reverse, List.map_map]
    rw [List.reverse_inj]
    apply List.map_congr_left
    intro p hp
    cases p with
    | mk center directions =>
        cases directions
        simp [reverseWedge, reverseDirectionPair, negVec_negVec]
  · simp [List.length_scanl]

/-- A common endpoint translation and exchange of face roles preserve wedge
disjointness, including complementary wedges sharing a center. -/
lemma reverseWedge_interiorDisjoint (endpoint : Vec3) (a b : Wedge) :
    interiorDisjoint (reverseWedge endpoint a) (reverseWedge endpoint b) ↔
      interiorDisjoint a b := by
  unfold interiorDisjoint sameUnorderedPair reverseWedge
  dsimp only
  have hcenter :
      addVec a.center (negVec endpoint) ≠
          addVec b.center (negVec endpoint) ↔
        a.center ≠ b.center := by
    constructor <;> contrapose!
    · exact congrArg (fun p => addVec p (negVec endpoint))
    · intro h
      funext i
      have hi := congrFun h i
      simp [addVec, negVec] at hi ⊢
      omega
  rw [hcenter]
  tauto

/-- Wedge disjointness is symmetric, both for different centers and for
complementary face pairs at one center. -/
lemma interiorDisjoint_comm (a b : Wedge) :
    interiorDisjoint a b ↔ interiorDisjoint b a := by
  have hforward : ∀ x y : Wedge,
      interiorDisjoint x y → interiorDisjoint y x := by
    intro x y
    unfold interiorDisjoint sameUnorderedPair
    rintro (hcenter | hpairs)
    · exact Or.inl hcenter.symm
    · right
      rcases hpairs with (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩)
      · left
        constructor
        · rw [← negVec_negVec y.entrance, ← h₁]
        · rw [← negVec_negVec y.exit, ← h₂]
      · right
        constructor
        · rw [← negVec_negVec y.entrance, ← h₂]
        · rw [← negVec_negVec y.exit, ← h₁]
  exact ⟨hforward a b, hforward b a⟩

/-- A direction list of length at least two describes pairwise-disjoint wedges
exactly when its reversed, negated list does. -/
lemma collisionFreeDirections_reverse (first second : Vec3)
    (rest : List Vec3) :
    (wedgesFromDirections ((first :: second :: rest).reverse.map negVec)).Pairwise
        interiorDisjoint ↔
      (wedgesFromDirections (first :: second :: rest)).Pairwise
        interiorDisjoint := by
  rw [wedgesFromDirections_reverse, List.pairwise_map,
    List.pairwise_reverse]
  simp only [reverseWedge_interiorDisjoint, interiorDisjoint_comm]

/-- Apply a lattice frame change to a wedge's cell center and both outward faces. -/
def rigidWedge (e : RigidVecEquiv) (w : Wedge) : Wedge :=
  ⟨e w.center, e w.entrance, e w.exit⟩

/-- Changing the initial frame rigidly changes every generated tail direction
by the same map, while leaving the rotation word unchanged. -/
lemma directionTail_rigid (e : RigidVecEquiv) (previous axis : Vec3)
    (rotations : List Rotation) :
    directionTail (e previous) (e axis) rotations =
      (directionTail previous axis rotations).map e := by
  induction rotations generalizing previous axis with
  | nil => simp [directionTail]
  | cons r rotations ih =>
      simp only [directionTail, List.map_cons]
      rw [← e.map_rotateQuarter]
      exact congrArg (e (rotateQuarter axis r previous) :: ·)
        (ih axis (rotateQuarter axis r previous))

/-- An orientation-preserving frame change carries the entire travel-direction
list to the list generated from the changed initial pair. -/
lemma directionsFrom_rigid (e : RigidVecEquiv) (previous axis : Vec3)
    (rotations : List Rotation) :
    directionsFrom (e previous) (e axis) rotations =
      (directionsFrom previous axis rotations).map e := by
  unfold directionsFrom
  simp only [List.map_cons]
  exact congrArg (e previous :: e axis :: ·)
    (directionTail_rigid e previous axis rotations)

/-- Applying a lattice frame change to the start and every step applies that
same change to every accumulated position. -/
lemma scanl_addVec_rigid (e : RigidVecEquiv) (start : Vec3)
    (steps : List Vec3) :
    List.scanl addVec (e start) (steps.map e) =
      (List.scanl addVec start steps).map e := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      simp only [List.map_cons, List.scanl_cons]
      rw [← e.map_add]
      exact congrArg (e start :: ·) (ih (addVec start step))

/-- Computing cell centers commutes with an origin-preserving lattice frame change. -/
lemma centersFromDirections_rigid (e : RigidVecEquiv) (ds : List Vec3) :
    centersFromDirections (ds.map e) =
      (centersFromDirections ds).map e := by
  unfold centersFromDirections
  rw [show (((ds.map e).drop 1).dropLast) =
      ((ds.drop 1).dropLast).map e by simp]
  simpa [e.map_zero] using
    scanl_addVec_rigid e zeroVec ((ds.drop 1).dropLast)

/-- Constructing wedges after a rigid change of travel directions gives exactly
the rigidly transformed original wedges. -/
lemma wedgesFromDirections_rigid (e : RigidVecEquiv) (ds : List Vec3) :
    wedgesFromDirections (ds.map e) =
      (wedgesFromDirections ds).map (rigidWedge e) := by
  unfold wedgesFromDirections
  rw [centersFromDirections_rigid]
  have htail : (ds.map e).tail = ds.tail.map e := by
    cases ds <;> rfl
  rw [htail, zip_map_map e e ds ds.tail]
  rw [zip_map_map e
    (fun p : Vec3 × Vec3 => (e p.1, e p.2))]
  simp only [List.map_map]
  apply congrArg (fun f =>
    ((centersFromDirections ds).zip (ds.zip ds.tail)).map f)
  funext p
  cases p with
  | mk center directions =>
      cases directions
      simp [rigidWedge, e.map_neg]

/-- Rigid frame changes preserve the equality and complementary-face conditions
that determine whether two wedges have disjoint interiors. -/
lemma rigidWedge_interiorDisjoint (e : RigidVecEquiv) (a b : Wedge) :
    interiorDisjoint (rigidWedge e a) (rigidWedge e b) ↔
      interiorDisjoint a b := by
  unfold interiorDisjoint sameUnorderedPair rigidWedge
  dsimp only
  rw [show negVec (e b.entrance) = e (negVec b.entrance) by
      rw [e.map_neg]]
  rw [show negVec (e b.exit) = e (negVec b.exit) by
      rw [e.map_neg]]
  simp only [e.toEquiv.injective.eq_iff, e.toEquiv.injective.ne_iff]

/-- Changing all travel directions by one rigid frame map preserves collision
freedom of the resulting wedge list in both directions. -/
lemma collisionFreeDirections_rigid (e : RigidVecEquiv) (ds : List Vec3) :
    (wedgesFromDirections (ds.map e)).Pairwise interiorDisjoint ↔
      (wedgesFromDirections ds).Pairwise interiorDisjoint := by
  rw [wedgesFromDirections_rigid, List.pairwise_map]
  simp only [rigidWedge_interiorDisjoint]

/-- The proper frame change `(x, y, z)` to `(-y, -x, -z)`, sending the reversed
canonical pair `(-ex, -ey)` back to `(ey, ex)`. -/
def reverseFrameVec (v : Vec3) : Vec3 :=
  ![-v 1, -v 0, -v 2]

/-- The reversal frame adjustment packaged with its inverse and cross-product laws. -/
def reverseFrameEquiv : RigidVecEquiv where
  toEquiv :=
    { toFun := reverseFrameVec
      invFun := reverseFrameVec
      left_inv := by
        intro v
        funext i
        fin_cases i <;> simp [reverseFrameVec]
      right_inv := by
        intro v
        funext i
        fin_cases i <;> simp [reverseFrameVec] }
  map_zero := by
    funext i
    fin_cases i <;> simp [reverseFrameVec, zeroVec]
  map_add := by
    intro u v
    funext i
    fin_cases i <;> simp [reverseFrameVec, addVec] <;> ring
  map_neg := by
    intro v
    funext i
    fin_cases i <;> simp [reverseFrameVec, negVec]
  map_cross := by
    intro u v
    funext i
    fin_cases i <;> simp [reverseFrameVec, cross]

/-- The reversal adjustment makes the old negated exit the canonical previous direction. -/
@[simp] lemma reverseFrameVec_neg_ex :
    reverseFrameVec (negVec ex) = ey := by
  funext i
  fin_cases i <;> simp [reverseFrameVec, negVec, ex, ey]

/-- The reversal adjustment makes the old negated previous direction the canonical exit. -/
@[simp] lemma reverseFrameVec_neg_ey :
    reverseFrameVec (negVec ey) = ex := by
  funext i
  fin_cases i <;> simp [reverseFrameVec, negVec, ex, ey]

/-- Undo the terminal frame `e`, then normalize its negated, swapped pair of
travel directions to the canonical initial pair `(ey, ex)`. -/
def reversalNormalizer (e : RigidVecEquiv) : RigidVecEquiv :=
  e.symm.trans reverseFrameEquiv

/-- Normalization sends the negated terminal exit direction to canonical `ey`. -/
@[simp] lemma reversalNormalizer_neg_ex (e : RigidVecEquiv) :
    reversalNormalizer e (negVec (e ex)) = ey := by
  change reverseFrameVec (e.toEquiv.symm (negVec (e ex))) = ey
  rw [show e.toEquiv.symm (negVec (e ex)) = negVec ex by
    rw [← e.map_neg]
    simp]
  exact reverseFrameVec_neg_ex

/-- Normalization sends the negated terminal previous direction to canonical `ex`. -/
@[simp] lemma reversalNormalizer_neg_ey (e : RigidVecEquiv) :
    reversalNormalizer e (negVec (e ey)) = ex := by
  change reverseFrameVec (e.toEquiv.symm (negVec (e ey))) = ex
  rw [show e.toEquiv.symm (negVec (e ey)) = negVec ey by
    rw [← e.map_neg]
    simp]
  exact reverseFrameVec_neg_ey

/-- Reversing a rotation word preserves collision freedom: the geometric path
is traversed backwards and then rigidly normalized to the canonical frame. -/
lemma collisionFree_reverse (rotations : List Rotation) :
    collisionFree rotations.reverse ↔ collisionFree rotations := by
  let terminal := terminalFrameFrom RigidVecEquiv.refl rotations
  let normalizer := reversalNormalizer terminal
  have hdirections := directionsFrom_reverse_frame
    RigidVecEquiv.refl rotations
  have hnormalize := directionsFrom_rigid normalizer
    (negVec (terminal ex)) (negVec (terminal ey)) rotations.reverse
  dsimp only [normalizer] at hnormalize
  rw [reversalNormalizer_neg_ex, reversalNormalizer_neg_ey] at hnormalize
  rw [hdirections] at hnormalize
  change directions rotations.reverse =
    ((directions rotations).reverse.map negVec).map normalizer at hnormalize
  unfold collisionFree wedges
  rw [hnormalize, collisionFreeDirections_rigid]
  exact collisionFreeDirections_reverse ey ex (directionTail ey ex rotations)

/-- Reversing finite formula indices becomes ordinary list reversal when the
rotations are read in order. -/
lemma ofFn_reverseFormula {n : ℕ} (w : Formula n) :
    List.ofFn (reverseFormula w) = (List.ofFn w).reverse := by
  apply List.ext_get
  · simp
  · intro i hleft hright
    simp only [List.get_eq_getElem, List.getElem_ofFn,
      List.getElem_reverse, reverseFormula]
    apply congrArg w
    apply Fin.ext
    simp
    omega

/-- An `n`-rotation formula and its head-tail reversal are valid simultaneously. -/
lemma valid_reverseFormula {n : ℕ} (w : Formula n) :
    Valid (reverseFormula w) ↔ Valid w := by
  unfold Valid ValidList
  rw [ofFn_reverseFormula, collisionFree_reverse]

/-- Head-tail reversal bundled as an involutive validity-preserving transform
of `n`-rotation formulas, hence of snakes with `n + 1` wedges. -/
def reversalTransform (n : ℕ) : InvolutiveFormulaTransform n where
  toFun := reverseFormula
  involutive := reverseFormula_involutive
  valid_iff := valid_reverseFormula

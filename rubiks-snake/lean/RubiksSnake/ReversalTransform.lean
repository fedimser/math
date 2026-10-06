import RubiksSnake.ReflectionTransform
import RubiksSnake.FormulaTransform

import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-! Definition of formula reversal and proofs of its properties. -/

namespace RubiksSnake


def reverseFormula {n : ℕ} (w : Formula n) : Formula n :=
  fun i => w i.rev

@[simp] lemma reverseFormula_involutive {n : ℕ} (w : Formula n) :
    reverseFormula (reverseFormula w) = w := by
  funext i
  simp [reverseFormula]

structure RigidVecEquiv where
  toEquiv : Vec3 ≃ Vec3
  map_zero : toEquiv zeroVec = zeroVec
  map_add : ∀ u v, toEquiv (addVec u v) =
    addVec (toEquiv u) (toEquiv v)
  map_neg : ∀ v, toEquiv (negVec v) = negVec (toEquiv v)
  map_cross : ∀ u v, toEquiv (cross u v) =
    cross (toEquiv u) (toEquiv v)

instance : CoeFun RigidVecEquiv (fun _ => Vec3 → Vec3) :=
  ⟨fun e => e.toEquiv⟩

def RigidVecEquiv.refl : RigidVecEquiv where
  toEquiv := Equiv.refl _
  map_zero := rfl
  map_add _ _ := rfl
  map_neg _ := rfl
  map_cross _ _ := rfl

def RigidVecEquiv.trans (e f : RigidVecEquiv) : RigidVecEquiv where
  toEquiv := e.toEquiv.trans f.toEquiv
  map_zero := by simp [e.map_zero, f.map_zero]
  map_add u v := by simp [e.map_add, f.map_add]
  map_neg v := by simp [e.map_neg, f.map_neg]
  map_cross u v := by simp [e.map_cross, f.map_cross]

def RigidVecEquiv.symm (e : RigidVecEquiv) : RigidVecEquiv where
  toEquiv := e.toEquiv.symm
  map_zero := e.toEquiv.injective <| by simp [e.map_zero]
  map_add u v := e.toEquiv.injective <| by
    simp [e.map_add]
  map_neg v := e.toEquiv.injective <| by
    simp [e.map_neg]
  map_cross u v := e.toEquiv.injective <| by
    simp [e.map_cross]

def frameStep (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => ![v 1, v 0, -v 2]
  | 1 => ![v 1, v 2, v 0]
  | 2 => ![v 1, -v 0, v 2]
  | _ => ![v 1, -v 2, -v 0]

def frameStepInv (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => ![v 1, v 0, -v 2]
  | 1 => ![v 2, v 0, v 1]
  | 2 => ![-v 1, v 0, v 2]
  | _ => ![-v 2, v 0, -v 1]

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

@[simp] lemma frameStep_ey (r : Rotation) :
    frameStep r ey = ex := by
  funext i
  fin_cases r <;> fin_cases i <;> simp [frameStep, ey, ex]

@[simp] lemma frameStep_ex (r : Rotation) :
    frameStep r ex = rotateQuarter ex r ey := by
  funext i
  fin_cases r <;> fin_cases i <;>
    simp [frameStep, rotateQuarter, cross, negVec, ex, ey]

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

def advanceFrame (e : RigidVecEquiv) (r : Rotation) : RigidVecEquiv :=
  (frameStepEquiv r).trans e

def terminalFrameFrom : RigidVecEquiv → List Rotation → RigidVecEquiv
  | e, [] => e
  | e, r :: rs => terminalFrameFrom (advanceFrame e r) rs

@[simp] lemma advanceFrame_ey (e : RigidVecEquiv) (r : Rotation) :
    advanceFrame e r ey = e ex := by
  change e (frameStep r ey) = e ex
  rw [frameStep_ey]

@[simp] lemma advanceFrame_ex (e : RigidVecEquiv) (r : Rotation) :
    advanceFrame e r ex = rotateQuarter (e ex) r (e ey) := by
  change e (frameStep r ex) = rotateQuarter (e ex) r (e ey)
  rw [frameStep_ex, e.map_rotateQuarter]

lemma directionsFrom_cons (previous axis : Vec3) (r : Rotation)
    (rs : List Rotation) :
    directionsFrom previous axis (r :: rs) =
      previous :: directionsFrom axis (rotateQuarter axis r previous) rs := by
  rfl

lemma directionsFrom_cons_frame (e : RigidVecEquiv) (r : Rotation)
    (rs : List Rotation) :
    directionsFrom (e ey) (e ex) (r :: rs) =
      e ey :: directionsFrom (advanceFrame e r ey)
        (advanceFrame e r ex) rs := by
  simp [directionsFrom, directionTail]

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

lemma terminalFrameFrom_append_singleton (e : RigidVecEquiv)
    (rs : List Rotation) (r : Rotation) :
    terminalFrameFrom e (rs ++ [r]) =
      advanceFrame (terminalFrameFrom e rs) r := by
  induction rs generalizing e with
  | nil => rfl
  | cons s rs ih => exact ih (advanceFrame e s)

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

def sumVec (steps : List Vec3) : Vec3 :=
  steps.foldl addVec zeroVec

lemma scanl_addVec_translate (offset start : Vec3) (steps : List Vec3) :
    List.scanl addVec (addVec offset start) steps =
      (List.scanl addVec start steps).map (addVec offset) := by
  induction steps generalizing start with
  | nil => simp
  | cons step steps ih =>
      simp only [List.scanl_cons, List.map_cons]
      rw [addVec_assoc, ih]

lemma sumVec_append_singleton (steps : List Vec3) (step : Vec3) :
    sumVec (steps ++ [step]) = addVec (sumVec steps) step := by
  simp [sumVec, List.foldl_append]

lemma scanl_addVec_append_singleton (steps : List Vec3) (step : Vec3) :
    List.scanl addVec zeroVec (steps ++ [step]) =
      List.scanl addVec zeroVec steps ++ [addVec (sumVec steps) step] := by
  rw [List.scanl_append]
  simp [sumVec]

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

lemma middle_reverse_map {α β : Type} (f : α → β)
    (first second : α) (rest : List α) :
    ((((first :: second :: rest).reverse.map f).drop 1).dropLast) =
      (((first :: second :: rest).drop 1).dropLast).reverse.map f := by
  rw [List.map_reverse, List.drop_one, List.drop_one]
  rw [List.tail_reverse, List.dropLast_reverse, List.map_reverse,
    List.reverse_inj, List.map_dropLast]
  simp

def reverseDirectionPair (p : Vec3 × Vec3) : Vec3 × Vec3 :=
  (negVec p.2, negVec p.1)

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

def reverseWedge (endpoint : Vec3) (w : Wedge) : Wedge :=
  ⟨addVec w.center (negVec endpoint), w.exit, w.entrance⟩

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

lemma collisionFreeDirections_reverse (first second : Vec3)
    (rest : List Vec3) :
    (wedgesFromDirections ((first :: second :: rest).reverse.map negVec)).Pairwise
        interiorDisjoint ↔
      (wedgesFromDirections (first :: second :: rest)).Pairwise
        interiorDisjoint := by
  rw [wedgesFromDirections_reverse, List.pairwise_map,
    List.pairwise_reverse]
  simp only [reverseWedge_interiorDisjoint, interiorDisjoint_comm]

def rigidWedge (e : RigidVecEquiv) (w : Wedge) : Wedge :=
  ⟨e w.center, e w.entrance, e w.exit⟩

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

lemma directionsFrom_rigid (e : RigidVecEquiv) (previous axis : Vec3)
    (rotations : List Rotation) :
    directionsFrom (e previous) (e axis) rotations =
      (directionsFrom previous axis rotations).map e := by
  unfold directionsFrom
  simp only [List.map_cons]
  exact congrArg (e previous :: e axis :: ·)
    (directionTail_rigid e previous axis rotations)

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

lemma centersFromDirections_rigid (e : RigidVecEquiv) (ds : List Vec3) :
    centersFromDirections (ds.map e) =
      (centersFromDirections ds).map e := by
  unfold centersFromDirections
  rw [show (((ds.map e).drop 1).dropLast) =
      ((ds.drop 1).dropLast).map e by simp]
  simpa [e.map_zero] using
    scanl_addVec_rigid e zeroVec ((ds.drop 1).dropLast)

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

lemma collisionFreeDirections_rigid (e : RigidVecEquiv) (ds : List Vec3) :
    (wedgesFromDirections (ds.map e)).Pairwise interiorDisjoint ↔
      (wedgesFromDirections ds).Pairwise interiorDisjoint := by
  rw [wedgesFromDirections_rigid, List.pairwise_map]
  simp only [rigidWedge_interiorDisjoint]

def reverseFrameVec (v : Vec3) : Vec3 :=
  ![-v 1, -v 0, -v 2]

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

@[simp] lemma reverseFrameVec_neg_ex :
    reverseFrameVec (negVec ex) = ey := by
  funext i
  fin_cases i <;> simp [reverseFrameVec, negVec, ex, ey]

@[simp] lemma reverseFrameVec_neg_ey :
    reverseFrameVec (negVec ey) = ex := by
  funext i
  fin_cases i <;> simp [reverseFrameVec, negVec, ex, ey]

def reversalNormalizer (e : RigidVecEquiv) : RigidVecEquiv :=
  e.symm.trans reverseFrameEquiv

@[simp] lemma reversalNormalizer_neg_ex (e : RigidVecEquiv) :
    reversalNormalizer e (negVec (e ex)) = ey := by
  change reverseFrameVec (e.toEquiv.symm (negVec (e ex))) = ey
  rw [show e.toEquiv.symm (negVec (e ex)) = negVec ex by
    rw [← e.map_neg]
    simp]
  exact reverseFrameVec_neg_ex

@[simp] lemma reversalNormalizer_neg_ey (e : RigidVecEquiv) :
    reversalNormalizer e (negVec (e ey)) = ex := by
  change reverseFrameVec (e.toEquiv.symm (negVec (e ey))) = ex
  rw [show e.toEquiv.symm (negVec (e ey)) = negVec ey by
    rw [← e.map_neg]
    simp]
  exact reverseFrameVec_neg_ey

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

lemma valid_reverseFormula {n : ℕ} (w : Formula n) :
    Valid (reverseFormula w) ↔ Valid w := by
  unfold Valid ValidList
  rw [ofFn_reverseFormula, collisionFree_reverse]

def reversalTransform (n : ℕ) : InvolutiveFormulaTransform n where
  toFun := reverseFormula
  involutive := reverseFormula_involutive
  valid_iff := valid_reverseFormula

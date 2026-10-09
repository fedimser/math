import Mathlib.Data.ZMod.Basic
import Mathlib.Data.List.Rotate

import RubiksSnake.Definitions
import RubiksSnake.OtherSequences.ReversalTransform

namespace RubiksSnake

/-- Loop formula is converted to regular formula by removing last rotation. -/
def loopFrmToFrm {n : ℕ+} (w : Formula n) : Formula (n - 1) :=
  fun i => w (Fin.castLE (Nat.sub_le n 1) i)

/-- Closure test for `n` joint rotations: the extra terminal wedge returns to
the origin with the initial directions. The loop has `n` wedges before this
repetition; collision freedom is not checked. -/
def isLoop {n : ℕ} (w : Formula n) : Prop :=
  let ds := directions (List.ofFn w)
  (centersFromDirections ds).getLastD zeroVec = zeroVec ∧
    ds.reverse.take 2 = [ex, ey]

/-- Checks whether Formula describes valid n-wedge loop. -/
def isValidLoop {n : ℕ+} (w : Formula n) : Prop :=
  Valid (loopFrmToFrm w) ∧ isLoop w

/-- Formula describing an n-wedge loop. -/
structure LoopFormula (n : ℕ+) where
  f : Formula n
  validLoop: isValidLoop f

/-- Open an `n`-wedge loop at its chosen cut by omitting the closing rotation,
obtaining an `(n - 1)`-rotation formula with exactly `n` wedges. -/
def toShapeFormula {n : ℕ+}: LoopFormula n → Formula (n-1) :=
  fun lf => loopFrmToFrm lf.f

/-- Example: smallest loop. -/
def loop2222 : LoopFormula 4 where
  f := ![2, 2, 2, 2]
  validLoop := by
    unfold isValidLoop Valid ValidList loopFrmToFrm isLoop
    native_decide

/-- Transform of a loop-formula that turns it into another loop formula. -/
structure LoopTransform (n : ℕ+) where
  t: Formula n → Formula n
  preservesValidLoop: ∀ w, isValidLoop w → isValidLoop (t w)

/-- Change a loop's chosen cut by reading entry `i + k` modulo `n`.
Positive shifts move leading rotations to the end; negative shifts move backwards. -/
def shiftLoopFormula {n : ℕ} (k : ℤ) (w : Formula n) : Formula n :=
  fun i =>
    w ⟨Int.natMod ((i.1 : ℤ) + k) n,
      Int.natMod_lt (Nat.ne_of_gt (Nat.zero_lt_of_lt i.2))⟩

/-- Opening a loop formula corresponds to deleting the last entry of its rotation list. -/
lemma ofFn_loopFrmToFrm {n : ℕ+} (w : Formula n) :
    List.ofFn (loopFrmToFrm w) = (List.ofFn w).dropLast := by
  apply List.ext_get
  · simp
  · intro i hi hj
    simp only [List.get_eq_getElem, List.getElem_ofFn, List.getElem_dropLast,
      loopFrmToFrm]
    rfl

/-- An integer shift of formula indices is a left rotation of the rotation list,
with the shift reduced modulo the number of joints. -/
lemma ofFn_shiftLoopFormula {n : ℕ} (k : ℤ) (w : Formula n) :
    List.ofFn (shiftLoopFormula k w) =
      (List.ofFn w).rotate (Int.natMod k n) := by
  apply List.ext_get
  · simp
  · intro i hi hj
    simp only [List.get_eq_getElem, List.getElem_ofFn, List.getElem_rotate,
      List.length_ofFn, shiftLoopFormula]
    apply congrArg w
    apply Fin.ext
    change Int.natMod ((i : ℤ) + k) n = (i + Int.natMod k n) % n
    have hn : (n : ℤ) ≠ 0 := by
      have hi' : i < n := by simpa using hi
      omega
    apply Int.ofNat_inj.mp
    simp only [Int.natMod, Int.natCast_mod, Int.natCast_add,
      Int.toNat_of_nonneg (Int.emod_nonneg _ hn)]
    exact (Int.add_emod_emod _ _ _).symm

/-- The last two travel directions, read backwards, are the terminal frame's
images of `ex` and `ey`. -/
private lemma directionsFrom_terminalFrame (e : RigidVecEquiv)
    (rs : List Rotation) :
    (directionsFrom (e ey) (e ex) rs).reverse.take 2 =
      [terminalFrameFrom e rs ex, terminalFrameFrom e rs ey] := by
  induction rs generalizing e with
  | nil => rfl
  | cons r rs ih =>
      rw [directionsFrom_cons_frame, List.reverse_cons,
        List.take_append_of_le_length (by simp [directionsFrom])]
      exact ih (advanceFrame e r)

/-- A list ending in the reverse-read pair `[b, a]` has a prefix followed by `[a, b]`. -/
private lemma eq_append_of_reverse_take_two {α : Type} (ds : List α) (a b : α)
    (h : ds.reverse.take 2 = [b, a]) :
    ∃ pre, ds = pre ++ [a, b] := by
  refine ⟨(ds.reverse.drop 2).reverse, ?_⟩
  have heq := congrArg List.reverse (List.take_append_drop 2 ds.reverse)
  rw [h] at heq
  simpa using heq.symm

/-- List-based loop validity: the complete word closes in position and frame,
while collision freedom is tested without the final rotation that repeats
the initial wedge. -/
private def ValidLoopList (rs : List Rotation) : Prop :=
  ValidList rs.dropLast ∧
    (centersFromDirections (directions rs)).getLastD zeroVec = zeroVec ∧
    (directions rs).reverse.take 2 = [ex, ey]

/-- Indexed validity for an `n`-wedge loop agrees with the list-based closure
and collision conditions on its `n` rotations. -/
private lemma isValidLoop_iff_list {n : ℕ+} (w : Formula n) :
    isValidLoop w ↔ ValidLoopList (List.ofFn w) := by
  unfold isValidLoop Valid ValidLoopList isLoop
  rw [ofFn_loopFrmToFrm]

/-- Moving the first rotation to the end changes only the cut of a valid loop,
preserving closure and disjointness of its nonrepeated wedges. -/
private lemma validLoopList_rotate_one (r : Rotation) (rs : List Rotation)
    (h : ValidLoopList (r :: rs)) : ValidLoopList (rs ++ [r]) := by
  let e := advanceFrame RigidVecEquiv.refl r
  let ds := directionsFrom (e ey) (e ex) rs
  let first : Wedge := ⟨zeroVec, negVec ey, ex⟩
  have hey : e ey = ex := by simp [e, RigidVecEquiv.refl]
  have hex : e ex = rotateQuarter ex r ey := by simp [e, RigidVecEquiv.refl]
  have hdirs : directions (r :: rs) = ey :: ds := by
    exact directionsFrom_cons_frame RigidVecEquiv.refl r rs
  have hstart : ds = ex :: directionTail ey ex (r :: rs) := by
    simp only [ds, directionsFrom, hey, hex, directionTail]
  have hterminal :
      terminalFrameFrom e rs ex = ex ∧ terminalFrameFrom e rs ey = ey := by
    have hend := h.2.2
    change (directionsFrom (RigidVecEquiv.refl ey) (RigidVecEquiv.refl ex)
      (r :: rs)).reverse.take 2 = [ex, ey] at hend
    rw [directionsFrom_terminalFrame] at hend
    simpa only [terminalFrameFrom, List.cons.injEq, and_true] using hend
  have hend : ds.reverse.take 2 = [ex, ey] := by
    rw [directionsFrom_terminalFrame, hterminal.1, hterminal.2]
  obtain ⟨pre, hpre⟩ := eq_append_of_reverse_take_two ds ey ex hend
  have hshift :
      (directions (rs ++ [r])).map e = ds ++ [e ex] := by
    change (directionsFrom ey ex (rs ++ [r])).map e = _
    rw [← directionsFrom_rigid, directionsFrom_append_singleton_frame]
    simp only [advanceFrame_ex, hterminal.1, hterminal.2, hex, ds]

  have hfull :
      wedges (r :: rs) = wedges ((r :: rs).dropLast) ++ [first] := by
    obtain ⟨pre, hpre⟩ :=
      eq_append_of_reverse_take_two (directions (r :: rs)) ey ex h.2.2
    have hcenter :
        (centersFromDirections (pre ++ [ey, ex])).getLastD zeroVec = zeroVec := by
      rw [← hpre]
      exact h.2.1
    have hdrop : directions ((r :: rs).dropLast) = pre ++ [ey] := by
      rw [directions_dropLast _ (by simp), hpre]
      simp
    unfold wedges
    rw [hpre, hdrop, wedgesFromDirections_append_two, hcenter]
  have hcons :
      wedges (r :: rs) =
        first :: ((wedges rs).map (rigidWedge e)).map (translateWedge ex) := by
    change wedgesFromDirections (ey :: ex :: directionTail ey ex (r :: rs)) = _
    rw [wedgesFromDirections_cons ey ex _ (by simp [directionTail])]
    have hrigid : ex :: directionTail ey ex (r :: rs) =
        (directions rs).map e := by
      rw [← hstart]
      exact directionsFrom_rigid e ey ex rs
    rw [hrigid, wedgesFromDirections_rigid]
    rfl
  have hperm :
      (((wedges rs).map (rigidWedge e)).map (translateWedge ex)).Perm
        (wedges ((r :: rs).dropLast)) := by
    apply List.Perm.cons_inv (a := first)
    rw [← hcons, hfull]
    exact List.perm_append_singleton first _
  have hvalid :
      (((wedges rs).map (rigidWedge e)).map (translateWedge ex)).Pairwise
        interiorDisjoint :=
    hperm.symm.pairwise h.1 (fun {_ _} => (interiorDisjoint_comm _ _).mp)
  have hvalid' : ValidList rs := by
    simpa only [ValidList, collisionFree, List.pairwise_map,
      translateWedge_interiorDisjoint, rigidWedge_interiorDisjoint] using hvalid

  have hsteps : ds.tail.Perm ds.dropLast := by
    apply List.Perm.cons_inv (a := ex)
    have hlast : ds = ds.dropLast ++ [ex] := by rw [hpre]; simp
    have hhead : ex :: ds.tail = ds := by rw [hstart]; rfl
    rw [hhead]
    conv_lhs => rw [hlast]
    exact List.perm_append_singleton ex _
  have hsum : ds.tail.foldl addVec zeroVec = ds.dropLast.foldl addVec zeroVec :=
    hsteps.foldl_eq' (fun u _ v _ p => by
      funext i
      simp [addVec, Int.add_right_comm]) zeroVec
  have hzero : ds.dropLast.foldl addVec zeroVec = zeroVec := by
    have hc := h.2.1
    rw [hdirs] at hc
    simpa [centersFromDirections, List.getLastD_eq_getLast?,
      List.getLast?_scanl] using hc
  have hmiddle : ((ds ++ [e ex]).drop 1).dropLast = ds.tail := by
    rw [hstart]
    simp
  have hcenter :
      (centersFromDirections (ds ++ [e ex])).getLastD zeroVec = zeroVec := by
    simp only [centersFromDirections, hmiddle, List.getLastD_eq_getLast?,
      List.getLast?_scanl, Option.getD_some, hsum, hzero]
  refine ⟨by simpa using hvalid', ?_, ?_⟩
  · apply e.toEquiv.injective
    calc
      _ = ((centersFromDirections (directions (rs ++ [r]))).map e).getLastD
          (e zeroVec) := List.getLastD_map.symm
      _ = (centersFromDirections (ds ++ [e ex])).getLastD zeroVec := by
        rw [e.map_zero, ← centersFromDirections_rigid, hshift]
      _ = e zeroVec := by rw [hcenter, e.map_zero]
  · apply List.map_injective_iff.mpr e.toEquiv.injective
    rw [List.map_take, List.map_reverse, hshift, hpre]
    simp [hey]

/-- Any natural-number cyclic shift preserves the list-based valid-loop conditions. -/
private lemma validLoopList_rotate (rs : List Rotation) (m : ℕ)
    (h : ValidLoopList rs) : ValidLoopList (rs.rotate m) := by
  induction m generalizing rs with
  | zero => simpa using h
  | succ m ih =>
      cases rs with
      | nil => simpa using h
      | cons r rs =>
          rw [List.rotate_cons_succ]
          exact ih _ (validLoopList_rotate_one r rs h)

/-- Changing the cut of a closed snake preserves closure and collision freedom. -/
lemma shiftPreservesValidLoop (n: ℕ+) (k: ℤ) (w: Formula n):
    isValidLoop w → isValidLoop ((shiftLoopFormula k) w) := by
  intro h
  rw [isValidLoop_iff_list] at h ⊢
  rw [ofFn_shiftLoopFormula]
  exact validLoopList_rotate _ _ h

/-- A change of cut by any integer number of joints, bundled as a validity-preserving
transform of `n`-wedge loops. -/
def shiftTransform (n : ℕ+) (k : ℤ) : LoopTransform n where
  t := shiftLoopFormula k
  preservesValidLoop := shiftPreservesValidLoop n k



/-- Membership among the six signed coordinate unit directions of the cubic lattice. -/
private def SignedAxis (v : Vec3) : Prop :=
  v = ex ∨ v = negVec ex ∨ v = ey ∨ v = negVec ey ∨
    v = ez ∨ v = negVec ez

/-- Two signed coordinate directions on different axes, hence perpendicular,
forming an admissible pair of successive travel directions. -/
private def AxisFrame (previous axis : Vec3) : Prop :=
  SignedAxis previous ∧ SignedAxis axis ∧
    previous ≠ axis ∧ previous ≠ negVec axis

/-- The canonical starting directions form a perpendicular cardinal frame. -/
private lemma initial_axisFrame : AxisFrame ey ex := by
  unfold AxisFrame SignedAxis
  native_decide

/-- Every joint setting advances a perpendicular cardinal frame to another
perpendicular cardinal frame. -/
private lemma axisFrame_step (previous axis : Vec3) (r : Rotation)
    (h : AxisFrame previous axis) :
    AxisFrame axis (rotateQuarter axis r previous) := by
  rcases h.1 with h | h | h | h | h | h <;> subst previous <;>
    rcases h.2.1 with h | h | h | h | h | h <;> subst axis <;>
    fin_cases r <;>
    simp_all [AxisFrame, SignedAxis, rotateQuarter, cross, negVec, ex, ey, ez]
  all_goals native_decide

/-- From an admissible cardinal frame, every generated travel direction remains
a signed coordinate unit vector. -/
private lemma directionTail_signedAxis (previous axis : Vec3)
    (rs : List Rotation) (h : AxisFrame previous axis) :
    ∀ v ∈ directionTail previous axis rs, SignedAxis v := by
  induction rs generalizing previous axis with
  | nil => simp [directionTail]
  | cons r rs ih =>
      have hstep := axisFrame_step previous axis r h
      intro v hv
      simp only [directionTail, List.mem_cons] at hv
      rcases hv with rfl | hv
      · exact hstep.2.1
      · exact ih axis (rotateQuarter axis r previous) hstep v hv

/-- All travel directions of a canonically embedded rotation word are cardinal
unit directions, regardless of whether the word is collision-free. -/
private lemma directions_signedAxis (rs : List Rotation) :
    ∀ v ∈ directions rs, SignedAxis v := by
  intro v hv
  simp only [directions, directionsFrom, List.mem_cons] at hv
  rcases hv with rfl | rfl | hv
  · simp [SignedAxis]
  · simp [SignedAxis]
  · exact directionTail_signedAxis ey ex rs initial_axisFrame v hv

/-- Parity of the sum of lattice coordinates, giving the two checkerboard colors. -/
private def checkerColor (v : Vec3) : ZMod 2 :=
  v 0 + v 1 + v 2

/-- Every signed coordinate unit step has odd coordinate sum and flips checkerboard color. -/
private lemma signedAxis_checkerColor {v : Vec3} (h : SignedAxis v) :
    checkerColor v = 1 := by
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

/-- The checkerboard color of a vector sum is the sum of the two colors modulo two. -/
private lemma checkerColor_addVec (u v : Vec3) :
    checkerColor (addVec u v) = checkerColor u + checkerColor v := by
  simp [checkerColor, addVec]
  ring

/-- Following cardinal unit steps changes checkerboard color by the parity of
the number of steps, independently of their signs and axes. -/
private lemma checkerColor_foldl (vs : List Vec3) (initial : Vec3)
    (h : ∀ v ∈ vs, SignedAxis v) :
    checkerColor (vs.foldl addVec initial) =
      checkerColor initial + (vs.length : ZMod 2) := by
  induction vs generalizing initial with
  | nil => simp
  | cons v vs ih =>
      have hvs : ∀ u ∈ vs, SignedAxis u := by
        intro u hu
        exact h u (by simp [hu])
      rw [List.foldl_cons, ih (addVec initial v) hvs]
      rw [checkerColor_addVec, signedAxis_checkerColor (h v (by simp))]
      simp only [List.length_cons, Nat.cast_add, Nat.cast_one]
      ring

/-- The last accumulated position is the endpoint of the step fold; the default
is irrelevant because the position list always includes the initial point. -/
private lemma scanl_getLastD (vs : List Vec3) (initial default : Vec3) :
    (vs.scanl addVec initial).getLastD default = vs.foldl addVec initial := by
  induction vs generalizing initial default with
  | nil => simp
  | cons v vs ih =>
      simp only [List.scanl_cons, List.foldl_cons]
      rw [List.getLastD_cons]
      exact ih (addVec initial v) initial

/-- Loop cannot have odd length. -/
lemma noOddLoops (n : ℕ+) :
    Odd (n : ℕ) → ∀ w: Formula n, ¬isLoop w := by
  intro hn w hloop
  let ds := directions (List.ofFn w)
  let steps := (ds.drop 1).dropLast
  have hcenter : steps.foldl addVec zeroVec = zeroVec := by
    have h := hloop.1
    unfold isLoop at hloop
    change (steps.scanl addVec zeroVec).getLastD zeroVec = zeroVec at h
    rw [scanl_getLastD] at h
    exact h
  have hsteps : ∀ v ∈ steps, SignedAxis v := by
    intro v hv
    exact directions_signedAxis (List.ofFn w) v (by
      have hv' : v ∈ ds.drop 1 := List.mem_of_mem_dropLast hv
      simp only [List.drop_one] at hv'
      exact List.mem_of_mem_tail hv')
  have hlength : steps.length = (n : ℕ) := by
    simp [steps, ds, directions, directionsFrom, directionTail_length]
  have hcast : ((n : ℕ) : ZMod 2) = 0 := by
    have hcolor := checkerColor_foldl steps zeroVec hsteps
    rw [hcenter] at hcolor
    simpa [checkerColor, zeroVec, hlength] using hcolor.symm
  have heven : Even (n : ℕ) :=
    even_iff_two_dvd.mpr ((ZMod.natCast_eq_zero_iff (n : ℕ) 2).mp hcast)
  exact (Nat.not_odd_iff_even.mpr heven) hn




noncomputable section

/-- Count valid `n`-wedge loop formulas with a chosen cut. All `n` rotations,
including the closing one, are recorded; collision freedom is checked on the
`n` wedges obtained after omitting that last rotation. -/
def L1 (n : ℕ+) : ℕ :=
  countFormulas n (fun f => isLoop f ∧ Valid (loopFrmToFrm f))

end

/-- The number of valid loop formulas is zero for every odd number of wedges. -/
lemma noOddLoopsNumeric (n : ℕ+): Odd (n : ℕ) → L1 n = 0 := by
  intro hn
  unfold L1 countFormulas
  rw [Finite.card_eq_zero_iff]
  exact ⟨fun ⟨w, hw⟩ => noOddLoops n hn w hw.1⟩


/-- No one-wedge loop formula exists, by the odd-length obstruction. -/
lemma L1_1_value : L1 1 = 0 := noOddLoopsNumeric 1 (by simp)
/-- No three-wedge loop formula exists, by the odd-length obstruction. -/
lemma L1_3_value : L1 3 = 0 := noOddLoopsNumeric 3 ⟨1, by norm_num⟩
/-- No five-wedge loop formula exists, by the odd-length obstruction. -/
lemma L1_5_value : L1 5 = 0 := noOddLoopsNumeric 5 ⟨2, by norm_num⟩
/-- No seven-wedge loop formula exists, by the odd-length obstruction. -/
lemma L1_7_value : L1 7 = 0 := noOddLoopsNumeric 7 ⟨3, by norm_num⟩



end RubiksSnake

import RubiksSnake.Definitions

/-! Geometric constructions and lemmas shared by counting arguments. -/

namespace RubiksSnake

@[simp] lemma addVec_zero_left (v : Vec3) : addVec zeroVec v = v := by
  funext i
  simp [addVec, zeroVec]

@[simp] lemma addVec_zero_right (v : Vec3) : addVec v zeroVec = v := by
  funext i
  simp [addVec, zeroVec]

lemma addVec_assoc (u v w : Vec3) :
    addVec (addVec u v) w = addVec u (addVec v w) := by
  funext i
  simp [addVec, Int.add_assoc]

@[simp] lemma addVec_neg_right (v : Vec3) : addVec v (negVec v) = zeroVec := by
  funext i
  simp [addVec, negVec, zeroVec]

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

def increasingRotationList : Vec3 → Vec3 → List Bool → List Rotation
  | _, _, [] => []
  | previous, axis, choice :: choices =>
      let rotation := increasingRotation previous axis choice
      let next := rotateQuarter axis rotation previous
      rotation :: increasingRotationList axis next choices

@[simp] lemma increasingRotationList_length (previous axis : Vec3) (choices : List Bool) :
    (increasingRotationList previous axis choices).length = choices.length := by
  induction choices generalizing previous axis with
  | nil => rfl
  | cons choice choices ih =>
      simp [increasingRotationList, ih]

lemma increasingRotationList_injective (previous axis : Vec3) :
    Function.Injective (increasingRotationList previous axis) := by
  intro xs
  induction xs generalizing previous axis with
  | nil =>
      intro ys h
      cases ys <;> simp_all [increasingRotationList]
  | cons choice choices ih =>
      intro ys h
      cases ys with
      | nil => simp [increasingRotationList] at h
      | cons choice' choices' =>
          simp only [increasingRotationList, List.cons.injEq] at h
          have hchoice : choice = choice' :=
            increasingRotation_injective previous axis h.1
          subst choice'
          have hchoices : choices = choices' := ih axis _ h.2
          subst choices'
          rfl

lemma directionTail_increasingRotationList {previous axis : Vec3}
    (hprevious : PositiveDirection previous) (haxis : PositiveDirection axis)
    (hne : previous ≠ axis) (choices : List Bool) :
    ∀ v ∈ directionTail previous axis (increasingRotationList previous axis choices),
      PositiveDirection v := by
  induction choices generalizing previous axis with
  | nil => simp [increasingRotationList, directionTail]
  | cons choice choices ih =>
      let rotation := increasingRotation previous axis choice
      let next := rotateQuarter axis rotation previous
      have hnext : PositiveDirection next ∧ next ≠ axis := by
        simpa [rotation, next] using
          increasingRotation_next hprevious haxis hne choice
      intro v hv
      simp only [increasingRotationList, directionTail, List.mem_cons] at hv
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

lemma directionTail_dropLast (previous axis : Vec3) (rs : List Rotation) :
    directionTail previous axis rs.dropLast =
      (directionTail previous axis rs).dropLast := by
  induction rs generalizing previous axis with
  | nil => rfl
  | cons r rs ih =>
      cases rs with
      | nil => rfl
      | cons s rs => simp only [List.dropLast_cons_cons, directionTail, ih]

lemma directions_dropLast (rs : List Rotation) (hne : rs ≠ []) :
    directions rs.dropLast = (directions rs).dropLast := by
  cases rs with
  | nil => contradiction
  | cons r rs =>
      simp [directions, directionsFrom, directionTail, directionTail_dropLast]

/-- Translation changes a wedge's center, but not its two face directions. -/
def translateWedge (offset : Vec3) (w : Wedge) : Wedge :=
  ⟨addVec offset w.center, w.entrance, w.exit⟩

@[simp] lemma translateWedge_zero (w : Wedge) :
    translateWedge zeroVec w = w := by
  cases w
  simp [translateWedge]

lemma translateWedge_add (p q : Vec3) (w : Wedge) :
    translateWedge p (translateWedge q w) = translateWedge (addVec p q) w := by
  cases w
  simp [translateWedge, addVec_assoc]

lemma translateWedge_interiorDisjoint (offset : Vec3) (a b : Wedge) :
    interiorDisjoint (translateWedge offset a) (translateWedge offset b) ↔
      interiorDisjoint a b := by
  have hcenter : addVec offset a.center = addVec offset b.center ↔
      a.center = b.center := by
    constructor
    · intro h
      funext i
      have hi := congrFun h i
      simpa [addVec] using hi
    · exact congrArg (addVec offset)
  simp only [interiorDisjoint, translateWedge, ne_eq, hcenter]

lemma centersFromDirections_cons (first second : Vec3) (rest : List Vec3)
    (hne : rest ≠ []) :
    centersFromDirections (first :: second :: rest) =
      zeroVec :: (centersFromDirections (second :: rest)).map (addVec second) := by
  cases rest with
  | nil => contradiction
  | cons third rest =>
      simp only [centersFromDirections, List.drop_succ_cons, List.drop_zero,
        List.dropLast_cons_cons, List.scanl_cons]
      congr 1
      apply List.ext_get
      · simp
      · intro i hi hj
        simp only [List.get_eq_getElem, List.getElem_scanl, List.getElem_map]
        have hzero : addVec second zeroVec = addVec zeroVec second := by
          funext j
          simp [addVec, zeroVec]
        rw [← hzero]
        apply List.foldl_hom (addVec second)
        intro u v
        funext j
        simp [addVec, Int.add_assoc]

lemma wedgesFromDirections_cons (first second : Vec3) (rest : List Vec3)
    (hne : rest ≠ []) :
    wedgesFromDirections (first :: second :: rest) =
      ⟨zeroVec, negVec first, second⟩ ::
        (wedgesFromDirections (second :: rest)).map (translateWedge second) := by
  unfold wedgesFromDirections
  rw [centersFromDirections_cons first second rest hne]
  simp only [List.tail_cons, List.zip_cons_cons, List.map_cons]
  congr 1
  rw [List.zip_map_left, List.map_map, List.map_map]
  rfl

lemma wedgesFromDirections_append_two (pre : List Vec3) (a b : Vec3) :
    wedgesFromDirections (pre ++ [a, b]) =
      wedgesFromDirections (pre ++ [a]) ++
        [⟨(centersFromDirections (pre ++ [a, b])).getLastD zeroVec,
          negVec a, b⟩] := by
  induction pre with
  | nil =>
      simp [wedgesFromDirections, centersFromDirections]
  | cons first pre ih =>
      cases pre with
      | nil =>
          simp [wedgesFromDirections, centersFromDirections]
      | cons second pre =>
          have hlong : pre ++ [a, b] ≠ [] := by simp
          have hshort : pre ++ [a] ≠ [] := by simp
          have hlast :
              (centersFromDirections (first :: second :: (pre ++ [a, b]))).getLastD
                  zeroVec =
                addVec second
                  ((centersFromDirections (second :: (pre ++ [a, b]))).getLastD
                    zeroVec) := by
            rw [centersFromDirections_cons first second _ hlong, List.getLastD_cons]
            simp [List.getLastD_eq_getLast?,
              centersFromDirections, List.getLast?_scanl]
          simp only [List.cons_append] at ih ⊢
          rw [wedgesFromDirections_cons first second _ hlong,
            wedgesFromDirections_cons first second _ hshort, ih,
            List.map_append, List.map_singleton, List.cons_append, hlast]
          rfl

def wedgePath (center incoming outgoing : Vec3) : List Vec3 → List Wedge
  | [] => [⟨center, negVec incoming, outgoing⟩]
  | next :: rest =>
      ⟨center, negVec incoming, outgoing⟩ ::
        wedgePath (addVec center outgoing) outgoing next rest

lemma wedgesFromDirections_eq_wedgePath
    (center incoming outgoing : Vec3) (rest : List Vec3) :
    (((outgoing :: rest).dropLast.scanl addVec center).zip
        ((incoming, outgoing) :: (outgoing :: rest).zip rest)).map
      (fun x => ⟨x.1, negVec x.2.1, x.2.2⟩) =
      wedgePath center incoming outgoing rest := by
  induction rest generalizing center incoming outgoing with
  | nil => simp [wedgePath]
  | cons next rest ih => simp [wedgePath, ih]

structure SlabState where
  center : Vec3
  previous : Vec3
  axis : Vec3
deriving DecidableEq

def SlabState.wedge (s : SlabState) : Wedge :=
  ⟨s.center, negVec s.previous, s.axis⟩

def slabStep (s : SlabState) (r : Rotation) : SlabState :=
  ⟨addVec s.center s.axis, s.axis, rotateQuarter s.axis r s.previous⟩

def slabRun : SlabState → List Rotation → SlabState
  | s, [] => s
  | s, r :: rs => slabRun (slabStep s r) rs

def slabNewWedges : SlabState → List Rotation → List Wedge
  | _, [] => []
  | s, r :: rs =>
      (slabStep s r).wedge :: slabNewWedges (slabStep s r) rs

lemma slabRun_append (s : SlabState) (a b : List Rotation) :
    slabRun s (a ++ b) = slabRun (slabRun s a) b := by
  induction a generalizing s with
  | nil => rfl
  | cons r rs ih => simpa [slabRun] using ih (slabStep s r)

lemma slabNewWedges_append (s : SlabState) (a b : List Rotation) :
    slabNewWedges s (a ++ b) =
      slabNewWedges s a ++ slabNewWedges (slabRun s a) b := by
  induction a generalizing s with
  | nil => rfl
  | cons r rs ih => simp [slabNewWedges, slabRun, ih]

lemma wedgePath_eq_slabNewWedges (s : SlabState) (rs : List Rotation) :
    wedgePath s.center s.previous s.axis (directionTail s.previous s.axis rs) =
      s.wedge :: slabNewWedges s rs := by
  induction rs generalizing s with
  | nil => rfl
  | cons r rs ih =>
      simpa [wedgePath, directionTail, slabNewWedges, slabStep, SlabState.wedge]
        using congrArg (s.wedge :: ·) (ih (slabStep s r))

lemma wedges_eq_slabNewWedges (rs : List Rotation) :
    wedges rs =
      (SlabState.mk zeroVec ey ex).wedge ::
        slabNewWedges ⟨zeroVec, ey, ex⟩ rs := by
  unfold wedges wedgesFromDirections centersFromDirections directions directionsFrom
  simp only [List.drop_succ_cons, List.drop_zero, List.tail_cons, List.zip_cons_cons]
  rw [wedgesFromDirections_eq_wedgePath]
  exact wedgePath_eq_slabNewWedges ⟨zeroVec, ey, ex⟩ rs

def slabTranslate (p : Vec3) (s : SlabState) : SlabState :=
  ⟨addVec p s.center, s.previous, s.axis⟩

lemma slabStep_translate (p : Vec3) (s : SlabState) (r : Rotation) :
    slabStep (slabTranslate p s) r = slabTranslate p (slabStep s r) := by
  simp [slabStep, slabTranslate, addVec_assoc]

lemma slabRun_translate (p : Vec3) (s : SlabState) (rs : List Rotation) :
    slabRun (slabTranslate p s) rs = slabTranslate p (slabRun s rs) := by
  induction rs generalizing s with
  | nil => rfl
  | cons r rs ih => simp [slabRun, slabStep_translate, ih]

lemma slabNewWedges_translate (p : Vec3) (s : SlabState) (rs : List Rotation) :
    slabNewWedges (slabTranslate p s) rs =
      (slabNewWedges s rs).map (translateWedge p) := by
  induction rs generalizing s with
  | nil => rfl
  | cons r rs ih =>
      simp only [slabNewWedges, slabStep_translate, ih, List.map_cons]
      rfl

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

lemma increasingRotationList_valid (choices : List Bool) :
    ValidList (increasingRotationList ey ex choices) := by
  let rs := increasingRotationList ey ex choices
  let ds := directions rs
  let steps := (ds.drop 1).dropLast
  have htail :
      ∀ v ∈ directionTail ey ex rs, PositiveDirection v := by
    exact directionTail_increasingRotationList
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

end RubiksSnake

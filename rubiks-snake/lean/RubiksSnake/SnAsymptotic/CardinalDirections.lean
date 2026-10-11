import RubiksSnake.Geometry
import RubiksSnake.Transforms.ReversalTransform

/-!
Cardinal-direction words and their rotation encodings. The initial directions
`d₀ = ey` and `d₁ = ex` have indices `2` and `0`, respectively.
-/

namespace RubiksSnake.CardinalDirections

/-- A signed coordinate direction, indexed in the order
`+x`, `-x`, `+y`, `-y`, `+z`, `-z`. -/
abbrev Direction := Fin 6

/-- Convert a cardinal-direction index into its integer unit vector. -/
def vector : Direction → Vec3 :=
  ![ex, negVec ex, ey, negVec ey, ez, negVec ez]

/-- The six cardinal-direction indices represent six distinct lattice vectors. -/
lemma vector_injective : Function.Injective vector := by
  decide

/-- Two cardinal directions are perpendicular when their axis indices differ,
regardless of their signs. -/
def Perpendicular (a b : Direction) : Prop :=
  a.val / 2 ≠ b.val / 2

/-- Comparing the axis indices decides perpendicularity of cardinal directions. -/
instance (a b : Direction) : Decidable (Perpendicular a b) := by
  unfold Perpendicular
  infer_instance

/-- Every ordered perpendicular cardinal pair is the terminal frame of some
three-rotation word starting in the canonical frame. -/
private lemma frame_reachable :
    ∀ previous incoming : Direction, Perpendicular previous incoming →
      ∃ a b c : Rotation,
        terminalFrameFrom RigidVecEquiv.refl [a, b, c] ey = vector previous ∧
        terminalFrameFrom RigidVecEquiv.refl [a, b, c] ex = vector incoming := by
  decide

/-- Any ordered perpendicular cardinal pair can be obtained as the images
of `(ey, ex)` under a rigid, orientation-preserving lattice frame change. -/
lemma exists_rigid_frame {previous incoming : Direction}
    (h : Perpendicular previous incoming) :
    ∃ e : RigidVecEquiv, e ey = vector previous ∧ e ex = vector incoming := by
  obtain ⟨a, b, c, hprevious, hincoming⟩ := frame_reachable previous incoming h
  exact ⟨terminalFrameFrom RigidVecEquiv.refl [a, b, c], hprevious, hincoming⟩

/-- Every consecutive pair of travel directions, starting with `incoming`, uses
different coordinate axes. This imposes no collision-freedom condition. -/
def Compatible : Direction → List Direction → Prop
  | _, [] => True
  | incoming, outgoing :: rest =>
      Perpendicular incoming outgoing ∧ Compatible outgoing rest

/-- Compatibility is decidable by checking each consecutive pair in the finite word. -/
instance (incoming : Direction) (ds : List Direction) :
    Decidable (Compatible incoming ds) := by
  induction ds generalizing incoming with
  | nil => exact isTrue trivial
  | cons outgoing ds ih =>
      letI := ih outgoing
      exact inferInstanceAs
        (Decidable (Perpendicular incoming outgoing ∧ Compatible outgoing ds))

/-- The selected turn is specified only when both directions are perpendicular
to `incoming`; `quarterTurn_eq_iff` gives its correctness and uniqueness. -/
def quarterTurn (previous incoming outgoing : Direction) : Rotation :=
  if outgoing = previous then 0
  else if vector outgoing = cross (vector incoming) (vector previous) then 1
  else if vector outgoing = negVec (vector previous) then 2
  else 3

/-- When both adjacent pairs are perpendicular, exactly one rotation symbol
turns `previous` about `incoming` into `outgoing`, namely the selected quarter turn. -/
lemma quarterTurn_eq_iff :
    ∀ {previous incoming outgoing : Direction},
      Perpendicular previous incoming →
      Perpendicular incoming outgoing →
      ∀ r : Rotation,
        rotateQuarter (vector incoming) r (vector previous) = vector outgoing ↔
          r = quarterTurn previous incoming outgoing := by
  decide

/-- The selected rotation produces the requested outgoing vector whenever the
previous and outgoing directions are perpendicular to the incoming axis. -/
lemma rotateQuarter_quarterTurn {previous incoming outgoing : Direction}
    (hprevious : Perpendicular previous incoming)
    (houtgoing : Perpendicular incoming outgoing) :
    rotateQuarter (vector incoming) (quarterTurn previous incoming outgoing)
        (vector previous) = vector outgoing :=
  (quarterTurn_eq_iff hprevious houtgoing _).2 rfl

/-- Encode each subsequent cardinal direction as a joint rotation relative to
the preceding pair. Correct decoding requires a perpendicular initial pair and
a compatible direction tail. -/
def encode (previous incoming : Direction) : List Direction → List Rotation
  | [] => []
  | outgoing :: rest =>
      quarterTurn previous incoming outgoing :: encode incoming outgoing rest

/-- Encoding emits one rotation per direction after the initial pair. -/
@[simp] lemma encode_length (previous incoming : Direction) (ds : List Direction) :
    (encode previous incoming ds).length = ds.length := by
  induction ds generalizing previous incoming with
  | nil => rfl
  | cons outgoing ds ih => simp [encode, ih]

/-- From a perpendicular initial pair, decoding the encoding of a compatible
tail recovers exactly its cardinal vectors, without adding the initial pair. -/
lemma directionTail_encode {previous incoming : Direction} {ds : List Direction}
    (hprevious : Perpendicular previous incoming) (hds : Compatible incoming ds) :
    directionTail (vector previous) (vector incoming) (encode previous incoming ds) =
      ds.map vector := by
  induction ds generalizing previous incoming with
  | nil => rfl
  | cons outgoing ds ih =>
      rcases hds with ⟨houtgoing, hrest⟩
      simp only [encode, directionTail, List.map_cons,
        rotateQuarter_quarterTurn hprevious houtgoing]
      rw [ih houtgoing hrest]

/-- For a fixed perpendicular initial pair, distinct compatible direction tails
have distinct rotation encodings. -/
lemma encode_injOn {previous incoming : Direction}
    (hprevious : Perpendicular previous incoming) :
    Set.InjOn (encode previous incoming) {ds | Compatible incoming ds} := by
  intro xs hxs ys hys h
  apply List.map_injective_iff.mpr vector_injective
  rw [← directionTail_encode hprevious hxs, ← directionTail_encode hprevious hys, h]

/-- Encoding is injective on the subtype of compatible tails for a fixed
perpendicular initial pair; collision freedom is not required. -/
lemma encode_injective {previous incoming : Direction}
    (hprevious : Perpendicular previous incoming) :
    Function.Injective (fun ds : {ds : List Direction // Compatible incoming ds} =>
      encode previous incoming ds.val) := by
  intro xs ys h
  exact Subtype.ext (encode_injOn hprevious xs.property ys.property h)

/-- Encoding from indices `(2, 0)` yields the canonical directions `ey`, `ex`,
followed by the compatible tail's vectors. -/
lemma directions_encode {ds : List Direction} (hds : Compatible 0 ds) :
    directions (encode 2 0 ds) = ey :: ex :: ds.map vector := by
  change ey :: ex :: directionTail (vector 2) (vector 0) (encode 2 0 ds) = _
  rw [directionTail_encode (show Perpendicular 2 0 by decide) hds]

/-- Build one wedge per outgoing direction, starting at `p`. Each entrance
opposes the preceding travel direction, and each next center lies along the exit.
No perpendicularity or collision check is performed. -/
def directionalPath (p : Vec3) (incoming : Direction) : List Direction → List Wedge
  | [] => []
  | outgoing :: ds =>
      Wedge.mk p (negVec (vector incoming)) (vector outgoing) ::
        directionalPath (addVec p (vector outgoing)) outgoing ds

/-- A directional path has one wedge per listed outgoing direction, rather
than one more wedge as in a rotation word. -/
@[simp] lemma directionalPath_length (p : Vec3) (incoming : Direction)
    (ds : List Direction) :
    (directionalPath p incoming ds).length = ds.length := by
  induction ds generalizing p incoming with
  | nil => rfl
  | cons outgoing ds ih => simp [directionalPath, ih]

/-- Translating the initial center translates every wedge in the directional
path without changing its entrance or exit faces. -/
lemma directionalPath_translate (offset p : Vec3) (incoming : Direction)
    (ds : List Direction) :
    directionalPath (addVec offset p) incoming ds =
      (directionalPath p incoming ds).map (translateWedge offset) := by
  induction ds generalizing p incoming with
  | nil => rfl
  | cons outgoing ds ih =>
      simp [directionalPath, translateWedge, addVec_assoc, ih]

/-- Concatenating direction words joins their wedge paths after the prefix's
displacements, retaining its last travel direction, or `incoming` when the
prefix is empty. -/
lemma directionalPath_append (p : Vec3) (incoming : Direction)
    (xs ys : List Direction) :
    directionalPath p incoming (xs ++ ys) =
      directionalPath p incoming xs ++
        directionalPath ((xs.map vector).foldl addVec p)
          (xs.getLastD incoming) ys := by
  induction xs generalizing p incoming with
  | nil => rfl
  | cons outgoing xs ih =>
      simpa only [List.cons_append, directionalPath, List.map_cons, List.foldl_cons,
        List.getLastD_cons] using
        congrArg (Wedge.mk p (negVec (vector incoming)) (vector outgoing) :: ·)
          (ih (addVec p (vector outgoing)) outgoing)

/-- A cardinal path with its first outgoing direction supplied separately is
the vector-based recursive wedge path with the same start and travel directions. -/
lemma directionalPath_eq_wedgePath (p : Vec3) (previous incoming : Direction)
    (ds : List Direction) :
    directionalPath p previous (incoming :: ds) =
      wedgePath p (vector previous) (vector incoming) (ds.map vector) := by
  induction ds generalizing p previous incoming with
  | nil => rfl
  | cons outgoing ds ih =>
      simpa only [directionalPath, List.map_cons, wedgePath] using
        congrArg (Wedge.mk p (negVec (vector previous)) (vector incoming) :: ·)
          (ih (addVec p (vector incoming)) incoming outgoing)

/-- Normalizing the first frame preserves validity of the complete wedge path. -/
lemma valid_encode_iff (p : Vec3) {previous incoming : Direction}
    {ds : List Direction} (hprevious : Perpendicular previous incoming)
    (hds : Compatible incoming ds) :
    ValidList (encode previous incoming ds) ↔
      (directionalPath p previous (incoming :: ds)).Pairwise interiorDisjoint := by
  obtain ⟨e, hprevious_frame, hincoming_frame⟩ := exists_rigid_frame hprevious
  have hframe :
      (directions (encode previous incoming ds)).map e =
        vector previous :: vector incoming :: ds.map vector := by
    rw [directions, ← directionsFrom_rigid, hprevious_frame, hincoming_frame]
    simp only [directionsFrom, directionTail_encode hprevious hds]
  have hpath :
      wedgesFromDirections (vector previous :: vector incoming :: ds.map vector) =
        directionalPath zeroVec previous (incoming :: ds) := by
    rw [directionalPath_eq_wedgePath]
    exact wedgesFromDirections_eq_wedgePath zeroVec (vector previous) (vector incoming)
      (ds.map vector)
  have htranslate := directionalPath_translate p zeroVec previous (incoming :: ds)
  rw [addVec_zero_right] at htranslate
  rw [htranslate, List.pairwise_map]
  simp only [translateWedge_interiorDisjoint]
  rw [← hpath, ← hframe, collisionFreeDirections_rigid]
  rfl

/-- With perpendicular adjacent directions, an encoded slab run adds exactly
the directional tail starting at `p + vector incoming`, excluding the initial wedge. -/
lemma slabNewWedges_encode (p : Vec3) {previous incoming : Direction}
    {ds : List Direction} (hprevious : Perpendicular previous incoming)
    (hds : Compatible incoming ds) :
    slabNewWedges (SlabState.mk p (vector previous) (vector incoming))
        (encode previous incoming ds) =
      directionalPath (addVec p (vector incoming)) incoming ds := by
  induction ds generalizing p previous incoming with
  | nil => rfl
  | cons outgoing ds ih =>
      rcases hds with ⟨houtgoing, hrest⟩
      simp only [encode, slabNewWedges, slabStep, SlabState.wedge,
        rotateQuarter_quarterTurn hprevious houtgoing, directionalPath]
      rw [ih (addVec p (vector incoming)) houtgoing hrest]

/-- Encoding a compatible tail from `(2, 0)` gives the fixed initial wedge at
the origin followed by its directional path from `ex`; the resulting formula
has `ds.length` rotations and `ds.length + 1` wedges. -/
lemma wedges_encode {ds : List Direction} (hds : Compatible 0 ds) :
    wedges (encode 2 0 ds) =
      Wedge.mk zeroVec (negVec ey) ex :: directionalPath ex 0 ds := by
  rw [wedges_eq_slabNewWedges]
  change Wedge.mk zeroVec (negVec ey) ex ::
    slabNewWedges (SlabState.mk zeroVec (vector 2) (vector 0)) (encode 2 0 ds) = _
  rw [slabNewWedges_encode zeroVec (show Perpendicular 2 0 by decide) hds,
    addVec_zero_left]
  rfl

end RubiksSnake.CardinalDirections

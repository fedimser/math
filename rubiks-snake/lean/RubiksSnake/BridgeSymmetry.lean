import RubiksSnake.BridgeCode

/-!
# Fourfold symmetry of bridge codes

Rotating about the bridge axis preserves its geometry and irreducibility.
When every seed starts in the same transverse direction, its four rotations
are disjoint and multiply every length coefficient by four.
-/

namespace RubiksSnake.BridgeSymmetry

open CardinalDirections SlabEnumeration BridgeWords

/-- A quarter turn around the x axis, with the x coordinate fixed. -/
def turn (p : Vec3) : Vec3 := ![p 0, -p 2, p 1]

/-- Apply the quarter turn simultaneously to a wedge's center and faces. -/
def turnWedge (w : Wedge) : Wedge :=
  ⟨turn w.center, turn w.entrance, turn w.exit⟩

/-- Quarter-turn permutation of natural direction indices; other indices are fixed. -/
def turnDirection (d : Nat) : Nat :=
  match d with
  | 2 => 4
  | 3 => 5
  | 4 => 3
  | 5 => 2
  | _ => d

/-- Rotate each letter of a direction word around the bridge axis. -/
def turnWord (w : List Nat) : List Nat := w.map turnDirection

/-- Four quarter turns return every direction index to itself. -/
lemma turnDirection_four (d : Nat) :
    turnDirection (turnDirection (turnDirection (turnDirection d))) = d := by
  rcases d with _ | _ | _ | _ | _ | _ | d <;> rfl

/-- The direction permutation is injective. -/
lemma turnDirection_injective : Function.Injective turnDirection := by
  intro a b h
  have := congrArg (fun d => turnDirection (turnDirection (turnDirection d))) h
  simpa only [turnDirection_four] using this

/-- Rotating a word is injective. -/
lemma turnWord_injective : Function.Injective turnWord :=
  List.map_injective_iff.mpr turnDirection_injective

/-- The vector rotation is injective. -/
lemma turn_injective : Function.Injective turn := by
  intro a b h
  funext i
  have h0 := congrFun h 0
  have h1 := congrFun h 1
  have h2 := congrFun h 2
  fin_cases i <;> simp_all [turn]

/-- Quarter turns commute with vector addition. -/
lemma turn_add (p q : Vec3) : turn (addVec p q) = addVec (turn p) (turn q) := by
  funext i
  fin_cases i <;> simp [turn, addVec, add_comm]

/-- Quarter turns commute with vector negation. -/
lemma turn_neg (p : Vec3) : turn (negVec p) = negVec (turn p) := by
  funext i
  fin_cases i <;> simp [turn, negVec]

/-- The origin is fixed by the rotation. -/
lemma turn_zero : turn zeroVec = zeroVec := by
  funext i
  fin_cases i <;> simp [turn, zeroVec]

/-- The direction permutation agrees with the coordinate rotation. -/
lemma vector_turnDirection (d : Nat) (hd : d < 6) :
    vectorOf (turnDirection d) = turn (vectorOf d) := by
  interval_cases d <;> decide

/-- Valid direction indices remain valid. -/
lemma turnDirection_lt (d : Nat) (hd : d < 6) : turnDirection d < 6 := by
  interval_cases d <;> decide

/-- Direction rotation preserves perpendicularity. -/
lemma turnDirection_perpendicular (a b : Nat) (ha : a < 6) (hb : b < 6) :
    Perpendicular (toDirection (turnDirection a)) (toDirection (turnDirection b)) ↔
      Perpendicular (toDirection a) (toDirection b) := by
  interval_cases a <;> interval_cases b <;> decide

/-- The x increment is unchanged by rotating a valid direction. -/
lemma xStep_turnDirection (d : Nat) (hd : d < 6) :
    xStep (turnDirection d) = xStep d := by
  interval_cases d <;> decide

/-- Interior disjointness is invariant under the quarter turn. -/
lemma turnWedge_disjoint (a b : Wedge) :
    interiorDisjoint (turnWedge a) (turnWedge b) ↔ interiorDisjoint a b := by
  simp only [interiorDisjoint, sameUnorderedPair, turnWedge, ← turn_neg,
    turn_injective.eq_iff, turn_injective.ne_iff]

/-- Rotating a directional path rotates each geometric wedge. -/
lemma path_turn (p : Vec3) (incoming : Nat) (w : List Nat)
    (hin : incoming < 6) (hw : ∀ d ∈ w, d < 6) :
    path (turn p) (turnDirection incoming) (turnWord w) =
      (path p incoming w).map turnWedge := by
  induction w generalizing p incoming with
  | nil => rfl
  | cons d w ih =>
    have hd := hw d (by simp)
    have htail : ∀ e ∈ w, e < 6 := fun e he => hw e (by simp [he])
    simp only [turnWord, List.map_cons, path_cons, List.map_cons]
    rw [vector_turnDirection d hd, ← turn_add]
    change _ :: path (turn (addVec p (vectorOf d))) (turnDirection d) (turnWord w) = _
    rw [ih _ d hd htail]
    congr 1
    simp only [placed, SlabBoard.Placed.wedge, turnWedge]
    change Wedge.mk (turn p) (negVec (vectorOf (turnDirection incoming)))
        (vectorOf (turnDirection d)) =
      Wedge.mk (turn p) (turn (negVec (vectorOf incoming))) (turn (vectorOf d))
    rw [vector_turnDirection incoming hin, vector_turnDirection d hd, turn_neg]

/-- Rotating a word preserves its total x height. -/
lemma height_turnWord (w : List Nat) (hw : ∀ d ∈ w, d < 6) :
    height xStep (turnWord w) = height xStep w := by
  simp only [height, turnWord, List.map_map]
  congr 1
  apply List.map_congr_left
  exact fun d hd => xStep_turnDirection d (hw d hd)

/-- A bridge remains a bridge after rotating its transverse coordinates. -/
lemma bridge_turnWord {w : List Nat} (hw : ∀ d ∈ w, d < 6)
    (h : IsBridge xStep w) : IsBridge xStep (turnWord w) := by
  refine ⟨by simpa [height_turnWord w hw] using h.1, ?_⟩
  intro u v huv hv
  obtain ⟨a, b, hab, rfl, rfl⟩ := List.map_eq_append_iff.mp huv
  have ha : ∀ d ∈ a, d < 6 := fun d hd => hw d (by simp [hab, hd])
  have hb : b ≠ [] := by intro hb; simp [hb] at hv
  change 0 ≤ height xStep (turnWord a) ∧ height xStep (turnWord a) < height xStep (turnWord w)
  rw [height_turnWord a ha, height_turnWord w hw]
  exact h.2 a b hab hb

/-- Four word rotations give back the original word. -/
lemma turnWord_four (w : List Nat) :
    turnWord (turnWord (turnWord (turnWord w))) = w := by
  simp [turnWord, List.map_map, Function.comp_def, turnDirection_four]

/-- Rotating an irreducible bridge preserves irreducibility. -/
lemma irreducible_turnWord {w : List Nat} (hw : ∀ d ∈ w, d < 6)
    (h : IsIrreducible xStep w) : IsIrreducible xStep (turnWord w) := by
  refine ⟨bridge_turnWord hw h.1, ?_⟩
  rintro ⟨a, b, ha, hb, hab⟩
  obtain ⟨u, v, huv, rfl, rfl⟩ := List.map_eq_append_iff.mp hab
  have hu : ∀ d ∈ u, d < 6 := fun d hd => hw d (by simp [huv, hd])
  have hv : ∀ d ∈ v, d < 6 := fun d hd => hw d (by simp [huv, hd])
  have back {s : List Nat} (hs : ∀ d ∈ s, d < 6)
      (hbridge : IsBridge xStep (turnWord s)) : IsBridge xStep s := by
    have small {t : List Nat} (ht : ∀ d ∈ t, d < 6) :
        ∀ d ∈ turnWord t, d < 6 := by
      intro d hd
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
      exact turnDirection_lt e (ht e he)
    simpa only [turnWord_four] using
      bridge_turnWord (small (small (small hs)))
        (bridge_turnWord (small (small hs)) (bridge_turnWord (small hs) hbridge))
  exact h.2 ⟨u, v, back hu ha, back hv hb, huv⟩

/-- A rotated compatible word is compatible with the rotated incoming direction. -/
lemma compatible_turnWord (incoming : Nat) (w : List Nat)
    (hin : incoming < 6) (hw : ∀ d ∈ w, d < 6)
    (h : Compatible (toDirection incoming) (w.map toDirection)) :
    Compatible (toDirection (turnDirection incoming)) ((turnWord w).map toDirection) := by
  induction w generalizing incoming with
  | nil => trivial
  | cons d w ih =>
    have hd := hw d (by simp)
    exact ⟨(turnDirection_perpendicular incoming d hin hd).mpr h.1,
      ih d hd (fun e he => hw e (by simp [he])) h.2⟩

/-- Rotate an entire bridge code without changing any word lengths. -/
def rotate (C : BridgeCode.Code) : BridgeCode.Code where
  words := C.words.map turnWord
  nodup := C.nodup.map turnWord_injective
  irreducible := by
    intro w hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    exact irreducible_turnWord (C.directions v hv).1 (C.irreducible v hv)
  directions := by
    intro w hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    refine ⟨?_, compatible_turnWord 0 v (by decide) (C.directions v hv).1
      (C.directions v hv).2⟩
    intro d hd
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
    exact turnDirection_lt e ((C.directions v hv).1 e he)
  last := by
    intro w hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    change (v.map turnDirection).getLastD (turnDirection 0) = turnDirection 0
    rw [List.getLastD_map, C.last v hv]
  valid := by
    intro w hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    rw [← turn_zero, ← show turnDirection 0 = 0 from rfl,
      path_turn zeroVec 0 v (by decide) (C.directions v hv).1, List.pairwise_map]
    exact (C.valid v hv).imp fun h => (turnWedge_disjoint _ _).mpr h

/-- Every word of a code begins in the prescribed direction. -/
def Heading (C : BridgeCode.Code) (d : Nat) : Prop :=
  ∀ w ∈ C.words, w.head? = some d

/-- Quarter turns rotate the first letter of every seed. -/
lemma heading_rotate {C : BridgeCode.Code} {d : Nat} (h : Heading C d) :
    Heading (rotate C) (turnDirection d) := by
  intro w hw
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
  simp [turnWord, h v hv]

/-- Different first directions separate whole code lists. -/
lemma disjoint_of_heading {C D : BridgeCode.Code} {a b : Nat}
    (ha : Heading C a) (hb : Heading D b) (hab : a ≠ b) :
    C.words.Disjoint D.words := by
  intro w hw hd
  have := (ha w hw).symm.trans (hb w hd)
  exact hab (Option.some.inj this)

/-- The four disjoint rotations of a seed code starting in `+y`. -/
def fourfold (C : BridgeCode.Code) (hhead : Heading C 2) : BridgeCode.Code where
  words := C.words ++ (rotate C).words ++ (rotate (rotate C)).words ++
    (rotate (rotate (rotate C))).words
  nodup := by
    have h1 : Heading (rotate C) 4 := heading_rotate hhead
    have h2 : Heading (rotate (rotate C)) 3 := heading_rotate h1
    have h3 : Heading (rotate (rotate (rotate C))) 5 := heading_rotate h2
    have h01 := disjoint_of_heading hhead h1 (by decide : 2 ≠ 4)
    have h02 := disjoint_of_heading hhead h2 (by decide : 2 ≠ 3)
    have h03 := disjoint_of_heading hhead h3 (by decide : 2 ≠ 5)
    have h12 := disjoint_of_heading h1 h2 (by decide : 4 ≠ 3)
    have h13 := disjoint_of_heading h1 h3 (by decide : 4 ≠ 5)
    have h23 := disjoint_of_heading h2 h3 (by decide : 3 ≠ 5)
    refine List.nodup_append.mpr ⟨?_, (rotate (rotate (rotate C))).nodup, ?_⟩
    · refine List.nodup_append.mpr ⟨?_, (rotate (rotate C)).nodup, ?_⟩
      · refine List.nodup_append.mpr ⟨C.nodup, (rotate C).nodup, ?_⟩
        intro a ha b hb hab
        subst b
        exact h01 ha hb
      · intro a ha b hb hab
        subst b
        rcases List.mem_append.mp ha with ha | ha
        · exact h02 ha hb
        · exact h12 ha hb
    · intro a ha b hb hab
      subst b
      rcases List.mem_append.mp ha with ha | ha
      · rcases List.mem_append.mp ha with ha | ha
        · exact h03 ha hb
        · exact h13 ha hb
      · exact h23 ha hb
  irreducible := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · rcases List.mem_append.mp hw with hw | hw
      · rcases List.mem_append.mp hw with hw | hw
        · exact C.irreducible w hw
        · exact (rotate C).irreducible w hw
      · exact (rotate (rotate C)).irreducible w hw
    · exact (rotate (rotate (rotate C))).irreducible w hw
  directions := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · rcases List.mem_append.mp hw with hw | hw
      · rcases List.mem_append.mp hw with hw | hw
        · exact C.directions w hw
        · exact (rotate C).directions w hw
      · exact (rotate (rotate C)).directions w hw
    · exact (rotate (rotate (rotate C))).directions w hw
  last := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · rcases List.mem_append.mp hw with hw | hw
      · rcases List.mem_append.mp hw with hw | hw
        · exact C.last w hw
        · exact (rotate C).last w hw
      · exact (rotate (rotate C)).last w hw
    · exact (rotate (rotate (rotate C))).last w hw
  valid := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · rcases List.mem_append.mp hw with hw | hw
      · rcases List.mem_append.mp hw with hw | hw
        · exact C.valid w hw
        · exact (rotate C).valid w hw
      · exact (rotate (rotate C)).valid w hw
    · exact (rotate (rotate (rotate C))).valid w hw

/-- Rotating a code preserves every length coefficient. -/
lemma rotate_blocks_length (C : BridgeCode.Code) (n : Nat) :
    (BridgeCode.blocks (rotate C) n).length = (BridgeCode.blocks C n).length := by
  simp [BridgeCode.blocks, rotate, turnWord, List.filter_map, Function.comp_def]

/-- The fourfold code has exactly four times each seed coefficient. -/
lemma fourfold_blocks_length (C : BridgeCode.Code) (hhead : Heading C 2) (n : Nat) :
    (BridgeCode.blocks (fourfold C hhead) n).length =
      4 * (BridgeCode.blocks C n).length := by
  change ((C.words ++ (rotate C).words ++ (rotate (rotate C)).words ++
    (rotate (rotate (rotate C))).words).filter (fun w => w.length == n)).length = _
  simp only [List.filter_append, List.length_append]
  change (BridgeCode.blocks C n).length + (BridgeCode.blocks (rotate C) n).length +
    (BridgeCode.blocks (rotate (rotate C)) n).length +
    (BridgeCode.blocks (rotate (rotate (rotate C))) n).length = _
  simp only [rotate_blocks_length]
  omega

end RubiksSnake.BridgeSymmetry

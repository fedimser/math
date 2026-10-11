import RubiksSnake.SnAsymptotic.CapCatalogue
import RubiksSnake.Transforms.ReversalTransform

/-! Head-to-tail reversal, reusing the existing geometric reversal theorem. -/

namespace RubiksSnake.CapReversal

open SlabEnumeration CardinalDirections BridgeWords

/-- Reverse the free transverse direction while preserving the two slab axes. -/
def flipZ (d : Nat) : Nat :=
  if d = 4 then 5 else if d = 5 then 4 else d

/-- The involution on direction indices used in reversed head interiors. -/
@[simp] lemma flipZ_flipZ (d : Nat) : flipZ (flipZ d) = d := by
  by_cases h4 : d = 4
  · subst d; rfl
  · by_cases h5 : d = 5
    · subst d; rfl
    · simp [flipZ, h4, h5]

/-- Tail words are reversed head interiors with the entry face changed to an exit face. -/
def tailWord (internal : List Nat) : List Nat :=
  internal.reverse.map flipZ ++ [2]

/-- Reversal and reflection cannot identify two different head interiors. -/
theorem tailWord_injective : Function.Injective tailWord := by
  intro a b h
  have hm := List.append_cancel_right h
  have heq := congrArg (List.map flipZ) hm
  apply List.reverse_injective
  simpa [List.map_map, Function.comp_def] using heq

/-- A half turn in the two constrained coordinates is a proper rigid frame map. -/
def halfTurn : RigidVecEquiv :=
  (frameStepEquiv 2).trans (frameStepEquiv 2)

/-- Reversing a travel direction and applying the half turn gives `flipZ`. -/
lemma flipZ_vector (d : Nat) (hd : d < 6) :
    vectorOf (flipZ d) = halfTurn (negVec (vectorOf d)) := by
  interval_cases d <;> decide

/-- Recursive cardinal paths and the existing direction-list geometry agree. -/
lemma path_eq_wedges (incoming : Nat) (w : List Nat) :
    path zeroVec incoming w =
      wedgesFromDirections ((incoming :: w).map vectorOf) := by
  cases w with
  | nil => rfl
  | cons d w =>
    change directionalPath zeroVec (toDirection incoming)
      (toDirection d :: w.map toDirection) = _
    rw [directionalPath_eq_wedgePath]
    rw [List.map_map]
    change wedgePath zeroVec (vectorOf incoming) (vectorOf d) (w.map vectorOf) = _
    simpa [wedgesFromDirections, centersFromDirections] using
      (wedgesFromDirections_eq_wedgePath zeroVec (vectorOf incoming)
        (vectorOf d) (w.map vectorOf)).symm

/-- The existing reversal theorem applies to any direction list with two entries. -/
lemma reverse_valid {ds : List Vec3} (hlen : 2 ≤ ds.length)
    (h : (wedgesFromDirections ds).Pairwise interiorDisjoint) :
    (wedgesFromDirections (ds.reverse.map negVec)).Pairwise interiorDisjoint := by
  cases ds with
  | nil => simp at hlen
  | cons first ds =>
    cases ds with
    | nil => simp at hlen
    | cons second rest => exact (collisionFreeDirections_reverse first second rest).mpr h

/-- A valid head becomes a valid tail, including every complementary-wedge collision rule. -/
theorem tailWord_valid (internal : List Nat)
    (hsmall : ∀ d ∈ internal, d < 6)
    (hvalid : (path zeroVec 2 (internal ++ [0])).Pairwise interiorDisjoint) :
    (path zeroVec 0 (tailWord internal)).Pairwise interiorDisjoint := by
  rw [path_eq_wedges] at hvalid ⊢
  have hrev := reverse_valid (by simp : 2 ≤ ((2 :: (internal ++ [0])).map vectorOf).length)
    hvalid
  have hrigid := (collisionFreeDirections_rigid halfTurn _).mpr hrev
  have hm : internal.reverse.map (fun d => halfTurn (negVec (vectorOf d))) =
      internal.reverse.map (fun d => vectorOf (flipZ d)) := by
    apply List.map_congr_left
    intro d hd
    exact (flipZ_vector d (hsmall d (List.mem_reverse.mp hd))).symm
  simpa only [List.map_cons, List.map_append, List.reverse_cons, List.reverse_append,
    List.reverse_singleton, List.map_reverse, List.map_map, Function.comp_def,
    List.map_singleton, List.singleton_append,
    show halfTurn (negVec (vectorOf 0)) = vectorOf 0 from rfl,
    show halfTurn (negVec (vectorOf 2)) = vectorOf 2 from rfl,
    ← List.map_reverse, hm, tailWord, List.append_assoc, List.reverse_nil,
    List.map_nil, List.nil_append, List.append_nil, List.cons_append] using hrigid

/-- The head-to-tail reflection leaves the primary increment unchanged. -/
@[simp] lemma xStep_flipZ (d : Nat) : xStep (flipZ d) = xStep d := by
  by_cases h4 : d = 4
  · subst d; rfl
  · by_cases h5 : d = 5
    · subst d; rfl
    · simp [flipZ, h4, h5]

/-- Total height is unchanged by reversing letter order. -/
@[simp] lemma height_reverse (step : Nat → ℤ) (w : List Nat) :
    height step w.reverse = height step w := by
  simp [height, List.map_reverse]

/-- The free-coordinate reflection preserves primary word heights. -/
@[simp] lemma height_flipZ (w : List Nat) :
    height xStep (w.map flipZ) = height xStep w := by
  simp [height, List.map_map, Function.comp_def]

/-- Every head interior prefix lies at or below its terminal occupied layer. -/
lemma upperCap_internal_bound {w : List Nat} (h : IsUpperCap xStep (w ++ [0])) :
    ∀ u v, w = u ++ v → height xStep u ≤ height xStep w := by
  intro u v huv
  have hb := h u (v ++ [0]) (by rw [huv, List.append_assoc]) (by simp)
  have hh : height xStep (w ++ [0]) = height xStep w + 1 := by
    rw [height_append, height_singleton]
    rfl
  omega

/-- Reversing a word bounded above by its endpoint gives nonnegative prefixes. -/
lemma reverse_nonnegative {w : List Nat}
    (h : ∀ u v, w = u ++ v → height xStep u ≤ height xStep w) :
    NonnegativePrefixes xStep w.reverse := by
  intro u v huv
  have hrev : w = v.reverse ++ u.reverse := by
    simpa only [List.reverse_reverse, List.reverse_append] using congrArg List.reverse huv
  have hb := h v.reverse u.reverse hrev
  have hh := congrArg (height xStep) hrev
  simp only [height_reverse, height_append] at hb hh
  omega

/-- Reflecting the free coordinate preserves nonnegative primary prefixes. -/
lemma flipZ_nonnegative {w : List Nat} (h : NonnegativePrefixes xStep w) :
    NonnegativePrefixes xStep (w.map flipZ) := by
  intro u v huv
  obtain ⟨a, b, hab, rfl, rfl⟩ := List.map_eq_append_iff.mp huv
  simpa only [height_flipZ] using h a b hab

/-- An overhanging head produces a tail that never dips below its entry. -/
theorem tailWord_nonnegative {w : List Nat} (h : IsUpperCap xStep (w ++ [0])) :
    NonnegativePrefixes xStep (tailWord w) := by
  apply nonnegative_prefixes_append (flipZ_nonnegative
    (reverse_nonnegative (upperCap_internal_bound h)))
  intro u v huv
  rcases List.singleton_eq_append_iff.mp huv with ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> decide

/-- Reversed head interiors retain their primary displacement. -/
@[simp] lemma tailWord_height (w : List Nat) :
    height xStep (tailWord w) = height xStep w := by
  simp only [tailWord, height_append, height_singleton, height_flipZ, height_reverse]
  exact add_zero _

/-- Positive head-cut witnesses become all the backward cuts required of a tail. -/
theorem tailWord_isTail {w : List Nat} (hu : IsUpperCap xStep (w ++ [0]))
    (hcross : ∀ c : ℤ, 0 < c → c ≤ height xStep w →
      ∃ u v, w = u ++ [1] ++ v ∧ height xStep u = c) :
    IsTail xStep (tailWord w) := by
  apply tail_of_backward_crossings (tailWord_nonnegative hu)
  intro c hc hcw
  rw [tailWord_height] at hcw
  obtain ⟨u, v, huv, huc⟩ := hcross (height xStep w - c + 1) (by omega) (by omega)
  have hh := congrArg (height xStep) huv
  have hone : xStep 1 = -1 := rfl
  simp only [height_append, height_singleton, hone] at hh
  have hvc : height xStep v = c := by omega
  refine ⟨v.reverse.map flipZ, 1, u.reverse.map flipZ ++ [2], ?_, ?_, hone⟩
  · simp only [tailWord, huv, List.reverse_append, List.reverse_singleton,
      List.map_append, List.map_singleton, show flipZ 1 = 1 from rfl,
      List.append_assoc]
  · simpa only [height_flipZ, height_reverse] using hvc

/-- Every actually enumerated head gives a canonical terminal cap. -/
theorem catalogue_tail (width transverse limit startY : Nat)
    (hy : startY ≤ transverse) {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ CapEnumeration.leavesAt width transverse limit 0 startY true) :
    IsTail xStep (tailWord w) := by
  apply tailWord_isTail (CapEnumeration.leavesAt_head
    width transverse limit 0 startY true hy hw).1
  intro c hc hcw
  exact CapGeometry.words_positive_crossings transverse (2 * limit + 1) (limit - 1)
    (CapEnumeration.pieceCursor limit startY true) (initialBoard transverse limit)
    hy rfl (CapEnumeration.leavesAt_mem_words width transverse limit 0 startY true hw)
    c hc hcw

/-- Perpendicularity is symmetric, so a direction chain can be read backwards. -/
lemma compatible_reverse (incoming outgoing : Direction) (w : List Direction)
    (h : Compatible incoming (w ++ [outgoing])) :
    Compatible outgoing (w.reverse ++ [incoming]) := by
  induction w generalizing incoming with
  | nil =>
    exact ⟨Ne.symm h.1, trivial⟩
  | cons d w ih =>
    have hr := ih d h.2
    have he : (w.reverse ++ [d]).getLastD outgoing = d := by simp
    have hf : Compatible d [incoming] := ⟨Ne.symm h.1, trivial⟩
    have hp := compatible_append outgoing (w.reverse ++ [d]) [incoming] hr
      (by rw [he]; exact hf)
    simpa only [List.reverse_cons, List.append_assoc] using hp

/-- Reflection preserves valid direction indices. -/
lemma flipZ_lt (d : Nat) (hd : d < 6) : flipZ d < 6 := by
  interval_cases d <;> decide

/-- Reflection preserves perpendicularity of cardinal directions. -/
lemma flipZ_perpendicular (a b : Nat) (ha : a < 6) (hb : b < 6) :
    Perpendicular (toDirection (flipZ a)) (toDirection (flipZ b)) ↔
      Perpendicular (toDirection a) (toDirection b) := by
  interval_cases a <;> interval_cases b <;> decide

/-- The reflection preserves every join of a compatible direction word. -/
lemma compatible_flipZ (incoming : Nat) (w : List Nat) (hin : incoming < 6)
    (hw : ∀ d ∈ w, d < 6) (hc : Compatible (toDirection incoming) (w.map toDirection)) :
    Compatible (toDirection (flipZ incoming)) ((w.map flipZ).map toDirection) := by
  induction w generalizing incoming with
  | nil => trivial
  | cons d w ih =>
    have hd := hw d (by simp)
    exact ⟨(flipZ_perpendicular incoming d hin hd).mpr hc.1,
      ih d hd (fun e he => hw e (by simp [he])) hc.2⟩

/-- Reversed tails enter along the separation axis and exit along the longitudinal axis. -/
theorem tailWord_directions (w : List Nat) (hw : ∀ d ∈ w, d < 6)
    (hc : Compatible 2 ((w ++ [0]).map toDirection)) :
    (∀ d ∈ tailWord w, d < 6) ∧ Compatible 0 ((tailWord w).map toDirection) := by
  have hs : ∀ d ∈ w.reverse ++ [2], d < 6 := by
    intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · exact hw d (List.mem_reverse.mp hd)
    · have : d = 2 := by simpa using hd
      omega
  have hr : Compatible (toDirection 0) ((w.reverse ++ [2]).map toDirection) := by
    simpa [List.map_append, List.map_reverse] using
      compatible_reverse (toDirection 2) (toDirection 0) (w.map toDirection)
        (by simpa only [List.map_append, List.map_singleton,
          show toDirection 2 = (2 : Direction) from rfl] using hc)
  have hm := compatible_flipZ 0 (w.reverse ++ [2]) (by decide) hs hr
  constructor
  · intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
      exact flipZ_lt e (hw e (List.mem_reverse.mp he))
    · have : d = 2 := by simpa using hd
      omega
  · change Compatible (toDirection 0) ((tailWord w).map toDirection)
    simpa [tailWord, List.map_append, List.map_map, flipZ] using hm

/-- The tail reflection preserves the longitudinal increment as well. -/
@[simp] lemma secondaryStep_flipZ (d : Nat) :
    CapGeometry.coordinateStep 1 (flipZ d) = CapGeometry.coordinateStep 1 d := by
  by_cases h4 : d = 4
  · subst d; rfl
  · by_cases h5 : d = 5
    · subst d; rfl
    · simp [flipZ, h4, h5]

/-- Secondary displacement of reflected words. -/
@[simp] lemma secondaryHeight_flipZ (w : List Nat) :
    height (CapGeometry.coordinateStep 1) (w.map flipZ) =
      height (CapGeometry.coordinateStep 1) w := by
  simp [height, List.map_map, Function.comp_def]

/-- The tail's last direction crosses out of the longitudinal slab. -/
lemma tailWord_secondary_height (w : List Nat) :
    height (CapGeometry.coordinateStep 1) (tailWord w) =
      height (CapGeometry.coordinateStep 1) w + 1 := by
  simp only [tailWord, height_append, height_singleton, secondaryHeight_flipZ,
    height_reverse]
  rfl

/-- Reversing a confined head puts every tail interior prefix in the same slab. -/
theorem tail_interior_bounds (width : Nat) (w : List Nat)
    (h : ∀ u v, w = u ++ v →
      0 ≤ height (CapGeometry.coordinateStep 1) u ∧
        height (CapGeometry.coordinateStep 1) u ≤ width) :
    ∀ u v, w.reverse.map flipZ = u ++ v →
      0 ≤ (width : ℤ) - height (CapGeometry.coordinateStep 1) w +
        height (CapGeometry.coordinateStep 1) u ∧
      (width : ℤ) - height (CapGeometry.coordinateStep 1) w +
        height (CapGeometry.coordinateStep 1) u ≤ width := by
  intro u v huv
  obtain ⟨a, b, hab, rfl, rfl⟩ := List.map_eq_append_iff.mp huv
  have hrev : w = b.reverse ++ a.reverse := by
    simpa using congrArg List.reverse hab
  have hb := h b.reverse a.reverse hrev
  have hh := congrArg (height (CapGeometry.coordinateStep 1)) hrev
  simp only [height_append, height_reverse] at hb hh
  simp only [secondaryHeight_flipZ]
  constructor <;> omega

end RubiksSnake.CapReversal

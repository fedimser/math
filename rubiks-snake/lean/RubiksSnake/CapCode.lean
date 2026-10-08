import RubiksSnake.CapAssembly
import RubiksSnake.BridgeSymmetry

/-! Put the cap construction into the standard longitudinal frame and add its opposite
transverse orientation. -/

namespace RubiksSnake.CapAssembly

open CardinalDirections SlabEnumeration BridgeWords CapGeometry CapPieces CapReversal
open BridgeSymmetry

/-- Exchange the longitudinal and separation axes by a proper rigid frame change. -/
def swapDirection (d : Nat) : Nat :=
  match d with
  | 0 => 2
  | 1 => 3
  | 2 => 0
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | _ => d

@[simp] lemma swapDirection_twice (d : Nat) : swapDirection (swapDirection d) = d := by
  rcases d with _ | _ | _ | _ | _ | _ | d <;> rfl

lemma swapDirection_injective : Function.Injective swapDirection := by
  intro a b h
  simpa using congrArg swapDirection h

lemma swapDirection_lt (d : Nat) (hd : d < 6) : swapDirection d < 6 := by
  interval_cases d <;> decide

lemma swapDirection_perpendicular (a b : Nat) (ha : a < 6) (hb : b < 6) :
    Perpendicular (toDirection (swapDirection a)) (toDirection (swapDirection b)) ↔
      Perpendicular (toDirection a) (toDirection b) := by
  interval_cases a <;> interval_cases b <;> decide

lemma swapDirection_vector (d : Nat) (hd : d < 6) :
    vectorOf (swapDirection d) = frameStepEquiv 0 (vectorOf d) := by
  interval_cases d <;> decide

lemma swapDirection_steps (d : Nat) (hd : d < 6) :
    xStep (swapDirection d) = longStep d ∧ longStep (swapDirection d) = xStep d := by
  interval_cases d <;> decide

private lemma height_map_steps {α β : Type*} (f : α → β) (s : α → ℤ) (t : β → ℤ)
    (w : List α) (h : ∀ d ∈ w, t (f d) = s d) :
    height t (w.map f) = height s w := by
  simp only [height, List.map_map]
  congr 1
  exact List.map_congr_left h

/-- Bridge geometry depends on letter increments, not on the choice of alphabet or frame. -/
private lemma bridge_map_steps {α β : Type*} (f : α → β) (s : α → ℤ) (t : β → ℤ)
    (w : List α) (h : ∀ d ∈ w, t (f d) = s d) :
    IsBridge t (w.map f) ↔ IsBridge s w := by
  have hh := height_map_steps f s t w h
  constructor
  · intro hb
    refine ⟨by simpa only [hh] using hb.1, ?_⟩
    intro u v huv hv
    have hu : ∀ d ∈ u, t (f d) = s d := fun d hd => h d (by simp [huv, hd])
    have hp := hb.2 (u.map f) (v.map f) (by rw [huv, List.map_append]) (by simpa using hv)
    simpa only [hh, height_map_steps f s t u hu] using hp
  · intro hb
    refine ⟨by simpa only [hh] using hb.1, ?_⟩
    intro u v huv hv
    obtain ⟨a, b, hab, rfl, rfl⟩ := List.map_eq_append_iff.mp huv
    have ha : ∀ d ∈ a, t (f d) = s d := fun d hd => h d (by simp [hab, hd])
    rw [height_map_steps f s t a ha, hh]
    exact hb.2 a b hab (by simpa using hv)

lemma swap_irreducible {w : List Nat} (hsmall : ∀ d ∈ w, d < 6)
    (hi : IsIrreducible longStep w) : IsIrreducible xStep (w.map swapDirection) := by
  have hsteps : ∀ d ∈ w, xStep (swapDirection d) = longStep d :=
    fun d hd => (swapDirection_steps d (hsmall d hd)).1
  refine ⟨(bridge_map_steps swapDirection longStep xStep w hsteps).mpr hi.1, ?_⟩
  rintro ⟨a, b, ha, hb, hab⟩
  obtain ⟨u, v, huv, rfl, rfl⟩ := List.map_eq_append_iff.mp hab
  apply hi.2
  exact ⟨u, v,
    (bridge_map_steps swapDirection longStep xStep u
      (fun d hd => hsteps d (by simp [huv, hd]))).mp ha,
    (bridge_map_steps swapDirection longStep xStep v
      (fun d hd => hsteps d (by simp [huv, hd]))).mp hb, huv⟩

lemma swap_compatible (incoming : Nat) (w : List Nat) (hin : incoming < 6)
    (hw : ∀ d ∈ w, d < 6)
    (hc : Compatible (toDirection incoming) (w.map toDirection)) :
    Compatible (toDirection (swapDirection incoming))
      ((w.map swapDirection).map toDirection) := by
  induction w generalizing incoming with
  | nil => trivial
  | cons d w ih =>
    have hd := hw d (by simp)
    exact ⟨(swapDirection_perpendicular incoming d hin hd).mpr hc.1,
      ih d hd (fun e he => hw e (by simp [he])) hc.2⟩

lemma swap_valid (w : List Nat) (hw : ∀ d ∈ w, d < 6)
    (hv : (path zeroVec 2 w).Pairwise interiorDisjoint) :
    (path zeroVec 0 (w.map swapDirection)).Pairwise interiorDisjoint := by
  rw [path_eq_wedges] at hv ⊢
  have hr := (collisionFreeDirections_rigid (frameStepEquiv 0) _).mpr hv
  have hm : (w.map vectorOf).map (frameStepEquiv 0) = (w.map swapDirection).map vectorOf := by
    simp only [List.map_map]
    apply List.map_congr_left
    intro d hd
    exact (swapDirection_vector d (hw d hd)).symm
  simpa only [List.map_cons, hm,
    show frameStepEquiv 0 (vectorOf 2) = vectorOf 0 from rfl] using hr

/-- Every complete cap assembly travels a strictly positive distance in its separation axis. -/
lemma traces_transverse_positive (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4)
    {t : List (List Nat)} (ht : t ∈ traces width limit n) :
    0 < height xStep t.flatten := by
  obtain ⟨p, hp, ht⟩ := List.mem_flatMap.mp ht
  split_ifs at ht with hlen
  · obtain ⟨rest, hrest, rfl⟩ := List.mem_map.mp ht
    have hp' := head_spec hl hp
    have hhead := upperCap_height_pos hp'.2.1.1 hp'.2.1.2.1
    have hn := suffixes_nonnegative width limit _ _ _ hl hw hp'.2.2.1 hrest
    have htail := hn rest.flatten [] (by simp)
    rw [List.flatten_cons, height_append]
    omega
  · simp at ht

/-- One fixed-width, fixed-length code in the standard incoming and outgoing `+x` frame. -/
def codeAt (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) : BridgeCode.Code where
  words := ((traces width limit n).map List.flatten).map (List.map swapDirection)
  nodup := (words_nodup width limit n hl hw).map
    (List.map_injective_iff.mpr swapDirection_injective)
  irreducible := by
    intro w hmem
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
    have hs := traces_spec width limit n hl hw ht
    exact swap_irreducible hs.1.small hs.2.1
  directions := by
    intro w hmem
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
    have hs := (traces_spec width limit n hl hw ht).1
    refine ⟨?_, swap_compatible 2 t.flatten (by decide) hs.small hs.compatible⟩
    intro d hd
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
    exact swapDirection_lt e (hs.small e he)
  last := by
    intro w hmem
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
    change (t.flatten.map swapDirection).getLastD (swapDirection 2) = swapDirection 2
    rw [List.getLastD_map, (traces_spec width limit n hl hw ht).1.last]
  valid := by
    intro w hmem
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
    have hs := (traces_spec width limit n hl hw ht).1
    exact swap_valid t.flatten hs.small hs.valid

lemma codeAt_metadata (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4)
    {w : List Nat} (hmem : w ∈ (codeAt width limit n hl hw).words) :
    w.length = n ∧ height xStep w = (width : ℤ) + 1 ∧ 0 < height longStep w := by
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
  obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
  have hs := traces_spec width limit n hl hw ht
  refine ⟨by simpa only [List.length_map] using hs.2.2, ?_, ?_⟩
  · rw [height_map_steps swapDirection longStep xStep t.flatten
      (fun d hd => (swapDirection_steps d (hs.1.small d hd)).1)]
    simpa using hs.1.span.1
  · rw [height_map_steps swapDirection xStep longStep t.flatten
      (fun d hd => (swapDirection_steps d (hs.1.small d hd)).2)]
    exact traces_transverse_positive width limit n hl hw ht

/-- A half turn reverses the transverse displacement but keeps the bridge axis fixed. -/
lemma height_halfTurn (w : List Nat) (hw : ∀ d ∈ w, d < 6) :
    height longStep (turnWord (turnWord w)) = -height longStep w := by
  induction w with
  | nil => rfl
  | cons d w ih =>
    have hd := hw d (by simp)
    have hs : longStep (turnDirection (turnDirection d)) = -longStep d := by
      interval_cases d <;> decide
    change longStep (turnDirection (turnDirection d)) +
      height longStep (turnWord (turnWord w)) = -(longStep d + height longStep w)
    rw [hs, ih (fun e he => hw e (by simp [he]))]
    omega

/-- Join two already valid codes whose word lists are disjoint. -/
def appendCode (C D : BridgeCode.Code) (hd : List.Disjoint C.words D.words) :
    BridgeCode.Code where
  words := C.words ++ D.words
  nodup := List.nodup_append'.mpr ⟨C.nodup, D.nodup, hd⟩
  irreducible := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · exact C.irreducible w hw
    · exact D.irreducible w hw
  directions := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · exact C.directions w hw
    · exact D.directions w hw
  last := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · exact C.last w hw
    · exact D.last w hw
  valid := by
    intro w hw
    rcases List.mem_append.mp hw with hw | hw
    · exact C.valid w hw
    · exact D.valid w hw

/-- Positive and negative transverse orientations cannot produce the same bridge. -/
lemma codeAt_halfTurn_disjoint (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) :
    List.Disjoint (codeAt width limit n hl hw).words
      (rotate (rotate (codeAt width limit n hl hw))).words := by
  intro w hpos hneg
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hneg
  obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hv
  have h1 := (codeAt_metadata width limit n hl hw hpos).2.2
  have h2 := (codeAt_metadata width limit n hl hw hu).2.2
  rw [height_halfTurn u ((codeAt width limit n hl hw).directions u hu).1] at h1
  omega

/-- Both opposite transverse orientations, still with no duplicate words. -/
def bothAt (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) : BridgeCode.Code :=
  appendCode (codeAt width limit n hl hw) (rotate (rotate (codeAt width limit n hl hw)))
    (codeAt_halfTurn_disjoint width limit n hl hw)

lemma bothAt_length (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) :
    (bothAt width limit n hl hw).words.length = 2 * (traces width limit n).length := by
  simp [bothAt, appendCode, rotate, codeAt, two_mul]

lemma bothAt_metadata (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4)
    {w : List Nat} (hmem : w ∈ (bothAt width limit n hl hw).words) :
    w.length = n ∧ height xStep w = (width : ℤ) + 1 := by
  rcases List.mem_append.mp hmem with hmem | hmem
  · exact ⟨(codeAt_metadata width limit n hl hw hmem).1,
      (codeAt_metadata width limit n hl hw hmem).2.1⟩
  · obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hv
    have hd := ((codeAt width limit n hl hw).directions u hu).1
    have hd' : ∀ d ∈ turnWord u, d < 6 := by
      intro d hm
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hm
      exact turnDirection_lt e (hd e he)
    rw [height_turnWord _ hd', height_turnWord _ hd]
    simpa [turnWord] using
      (show u.length = n ∧ height xStep u = (width : ℤ) + 1 from
        ⟨(codeAt_metadata width limit n hl hw hu).1,
          (codeAt_metadata width limit n hl hw hu).2.1⟩)

end RubiksSnake.CapAssembly

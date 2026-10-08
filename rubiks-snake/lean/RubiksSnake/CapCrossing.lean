import RubiksSnake.CapEnumeration
import RubiksSnake.CapGeometry

/-! Secondary-coordinate and backward-cut semantics of counted cap pieces. -/

namespace RubiksSnake.CapEnumeration

open SlabEnumeration BridgeWords CapGeometry

/-- Run the secondary coordinate and cut mask along an internal direction word. -/
def crossRun (s : CrossState) (w : List Nat) : CrossState :=
  w.foldl crossAdvance s

/-- Every internal step respects the secondary slab boundaries. -/
def CrossAllowedWord (width : Nat) : CrossState → List Nat → Prop
  | _, [] => True
  | s, d :: w => crossAllowed width s d = true ∧
      CrossAllowedWord width (crossAdvance s d) w

/-- The terminal tag records the actual endpoint, mask, and total wedge length. -/
theorem leaves_tag (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leaves width transverse side limit remaining s cross board) :
    tag = slot width limit (crossRun cross w) (s.length + w.length + 1) ∧
      CrossAllowedWord width cross w := by
  induction remaining generalizing s cross board w tag with
  | zero =>
    rw [leaves] at hw
    split at hw
    · have heq : (w, tag) = ([], slot width limit cross (s.length + 1)) := by
        simpa only [List.append_nil, List.mem_singleton] using hw
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj heq
      exact ⟨rfl, trivial⟩
    · simp at hw
  | succ remaining ih =>
    rw [leaves] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw
      · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp hw)
        exact ⟨rfl, trivial⟩
      · simp at hw
    · obtain ⟨outgoing, _, hw⟩ := List.mem_flatMap.mp hw
      split at hw
      · rename_i hmove
        simp only [Bool.and_eq_true] at hmove
        obtain ⟨⟨tail, label⟩, htail, heq⟩ := List.mem_map.mp hw
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj heq
        obtain ⟨hlabel, hallowed⟩ := ih _ _ _ htail
        refine ⟨?_, hmove.2, hallowed⟩
        simpa [crossRun, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hlabel
      · simp at hw

/-- Accepted secondary moves cannot leave their slab. -/
lemma crossAdvance_le (width : Nat) (s : CrossState) (d : Nat)
    (hx : s.x ≤ width) (h : crossAllowed width s d = true) :
    (crossAdvance s d).x ≤ width := by
  have hspec : (d = 2 → s.x < width) ∧ (d = 3 → 0 < s.x) := by
    simpa [crossAllowed, or_iff_not_imp_left] using h
  simp only [crossAdvance, beq_iff_eq]
  split_ifs <;> omega

/-- Secondary coordinate displacement agrees with the geometric direction. -/
lemma crossAdvance_coordinate (width : Nat) (s : CrossState) (d : Nat)
    (hd : d < 6) (h : crossAllowed width s d = true) :
    ((crossAdvance s d).x : ℤ) = s.x + coordinateStep 1 d := by
  have hspec : (d = 2 → s.x < width) ∧ (d = 3 → 0 < s.x) := by
    simpa [crossAllowed, or_iff_not_imp_left] using h
  have hleft := hspec.2
  interval_cases d <;>
    simp [crossAdvance, coordinateStep, vectorOf, toDirection,
      CardinalDirections.vector, ex, ey, ez, negVec]
  rw [Nat.cast_sub (by have := hleft rfl; omega)]
  ring

/-- An allowed word has the expected secondary endpoint. -/
theorem crossRun_coordinate (width : Nat) (s : CrossState) (w : List Nat)
    (hw : ∀ d ∈ w, d < 6) (ha : CrossAllowedWord width s w) :
    ((crossRun s w).x : ℤ) = s.x + height (coordinateStep 1) w := by
  induction w generalizing s with
  | nil => simp [crossRun]
  | cons d w ih =>
    obtain ⟨hd, ht⟩ := ha
    have hstep := crossAdvance_coordinate width s d (hw d (by simp)) hd
    have hrest := ih (crossAdvance s d) (fun e he => hw e (by simp [he])) ht
    simpa [crossRun, height, Nat.cast_add, add_assoc, hstep] using hrest

/-- An allowed word's endpoint stays within the secondary slab. -/
theorem crossRun_le (width : Nat) (s : CrossState) (w : List Nat)
    (hx : s.x ≤ width) (ha : CrossAllowedWord width s w) :
    (crossRun s w).x ≤ width := by
  induction w generalizing s with
  | nil => exact hx
  | cons d w ih =>
    exact ih _ (crossAdvance_le width s d hx ha.1) ha.2

/-- Any prefix of an allowed word is allowed. -/
lemma crossAllowed_prefix (width : Nat) (s : CrossState) (u v : List Nat)
    (h : CrossAllowedWord width s (u ++ v)) : CrossAllowedWord width s u := by
  induction u generalizing s with
  | nil => trivial
  | cons d u ih => exact ⟨h.1, ih _ h.2⟩

/-- Every accepted prefix stays between the same two longitudinal boundary planes. -/
theorem cross_prefix_bounds (width : Nat) (s : CrossState) (w : List Nat)
    (hx : s.x ≤ width) (hw : ∀ d ∈ w, d < 6)
    (ha : CrossAllowedWord width s w) :
    ∀ u v, w = u ++ v →
      0 ≤ (s.x : ℤ) + height (coordinateStep 1) u ∧
        (s.x : ℤ) + height (coordinateStep 1) u ≤ width := by
  intro u v huv
  have hp := crossAllowed_prefix width s u v (huv ▸ ha)
  have hsmall : ∀ d ∈ u, d < 6 := fun d hd => hw d (by simp [huv, hd])
  have hc := crossRun_coordinate width s u hsmall hp
  have hb := crossRun_le width s u hx hp
  constructor <;> omega

/-- A cut bit is inherited or comes from the current backward crossing. -/
lemma crossAdvance_mask (s : CrossState) (d j : Nat) :
    (crossAdvance s d).mask.testBit j = true ↔
      s.mask.testBit j = true ∨ d = 3 ∧ (crossAdvance s d).x = j := by
  by_cases h : d = 3
  · subst d
    simp [crossAdvance, Nat.shiftLeft_eq, Nat.testBit_two_pow]
  · simp [crossAdvance, h]

/-- No allowed step introduces a bit outside the secondary slab. -/
lemma crossAdvance_mask_lt (width : Nat) (s : CrossState) (d : Nat)
    (hx : s.x ≤ width) (hm : s.mask < 2 ^ width)
    (ha : crossAllowed width s d = true) :
    (crossAdvance s d).mask < 2 ^ width := by
  by_cases hd : d = 3
  · subst d
    have hpos : 0 < s.x := by simpa [crossAllowed] using ha
    simp only [crossAdvance, beq_iff_eq, if_true, Nat.shiftLeft_eq]
    apply Nat.or_lt_two_pow hm
    simpa using (Nat.pow_lt_pow_right (by decide : 1 < 2)
      (by omega : s.x - 1 < width))
  · simpa [crossAdvance, hd] using hm

/-- The final cut mask fits the coefficient array's mask dimension. -/
theorem crossRun_mask_lt (width : Nat) (s : CrossState) (w : List Nat)
    (hx : s.x ≤ width) (hm : s.mask < 2 ^ width)
    (ha : CrossAllowedWord width s w) :
    (crossRun s w).mask < 2 ^ width := by
  induction w generalizing s with
  | nil => exact hm
  | cons d w ih =>
    exact ih _ (crossAdvance_le width s d hx ha.1)
      (crossAdvance_mask_lt width s d hx hm ha.1) ha.2

/-- Every set bit is witnessed by an actual backward crossing, unless inherited. -/
theorem crossRun_backward_crossings (width : Nat) (s : CrossState) (w : List Nat)
    (hw : ∀ d ∈ w, d < 6) (ha : CrossAllowedWord width s w) (j : Nat)
    (hj : (crossRun s w).mask.testBit j = true) :
    s.mask.testBit j = true ∨
      ∃ u v, w = u ++ [3] ++ v ∧
        (s.x : ℤ) + height (coordinateStep 1) u = (j : ℤ) + 1 := by
  induction w generalizing s with
  | nil => exact Or.inl hj
  | cons d w ih =>
    have hd := hw d (by simp)
    have ht : ∀ e ∈ w, e < 6 := fun e he => hw e (by simp [he])
    have hstep := crossAdvance_coordinate width s d hd ha.1
    rcases ih (crossAdvance s d) ht ha.2 hj with hmask | ⟨u, v, huv, hu⟩
    · rcases (crossAdvance_mask s d j).mp hmask with hmask | ⟨rfl, hx⟩
      · exact Or.inl hmask
      · refine Or.inr ⟨[], w, by simp, ?_⟩
        have hthree : coordinateStep 1 3 = -1 := rfl
        rw [hx, hthree] at hstep
        simp only [height_nil]
        omega
    · refine Or.inr ⟨d :: u, v, by simp [huv], ?_⟩
      change (s.x : ℤ) + (coordinateStep 1 d + height (coordinateStep 1) u) = _
      omega

end RubiksSnake.CapEnumeration

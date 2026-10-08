import RubiksSnake.CapReversal

/-! Longitudinal geometry of individual cap pieces and their concatenations. -/

namespace RubiksSnake.CapPieces

open SlabEnumeration CardinalDirections BridgeWords CapGeometry CapEnumeration CapReversal

/-- Longitudinal increment in the computation's coordinate system. -/
abbrev longStep := coordinateStep 1

/-- Longitudinal endpoint and occupied-layer bounds for a piece. -/
def Span (width : Nat) (start finish : ℤ) (w : List Nat) : Prop :=
  start + height longStep w = finish ∧
    ∀ u v, w = u ++ v → v ≠ [] →
      0 ≤ start + height longStep u ∧ start + height longStep u ≤ width

/-- Every set bit has an actual longitudinal backward-crossing witness. -/
def Crossings (start : ℤ) (mask : Nat) (w : List Nat) : Prop :=
  ∀ j, mask.testBit j = true →
    ∃ u v, w = u ++ [3] ++ v ∧ start + height longStep u = (j : ℤ) + 1

/-- Geometric data carried by a short piece or an assembled path. -/
structure Spec (width : Nat) (start finish : ℤ) (mask incoming outgoing : Nat)
    (w : List Nat) : Prop where
  small : ∀ d ∈ w, d < 6
  compatible : Compatible (toDirection incoming) (w.map toDirection)
  valid : (path zeroVec incoming w).Pairwise interiorDisjoint
  last : w.getLastD incoming = outgoing
  span : Span width start finish w
  crossings : Crossings start mask w

/-- Longitudinal confinement composes at a common endpoint. -/
lemma span_append {width : Nat} {start middle finish : ℤ} {a b : List Nat}
    (ha : Span width start middle a) (hb : Span width middle finish b) :
    Span width start finish (a ++ b) := by
  refine ⟨?_, ?_⟩
  · rw [height_append]
    have h1 := ha.1
    have h2 := hb.1
    omega
  · intro u v huv hv
    rcases List.append_eq_append_iff.mp huv with
      ⟨c, huc, hbc⟩ | ⟨c, hac, hvc⟩
    · have h := hb.2 c v hbc hv
      rw [huc, height_append]
      have := ha.1
      constructor <;> omega
    · by_cases hc : c = []
      · have hau : a = u := by simpa [hc] using hac
        have hbv : b = v := by simpa only [hc, List.nil_append] using hvc.symm
        have hbne : b ≠ [] := by intro hnil; apply hv; rw [← hbv, hnil]
        have h := hb.2 [] b (by simp) hbne
        simp only [height_nil, add_zero] at h
        rw [← hau, ha.1]
        exact h
      · exact ha.2 u c hac hc

/-- Union of the masks records precisely the witnesses needed after concatenation. -/
lemma crossings_append {start middle : ℤ} {ma mb : Nat} {a b : List Nat}
    (hend : start + height longStep a = middle)
    (ha : Crossings start ma a) (hb : Crossings middle mb b) :
    Crossings start (ma ||| mb) (a ++ b) := by
  intro j hj
  simp only [Nat.testBit_or, Bool.or_eq_true] at hj
  rcases hj with hj | hj
  · obtain ⟨u, v, huv, hu⟩ := ha j hj
    exact ⟨u, v ++ b, by simp [huv, List.append_assoc], hu⟩
  · obtain ⟨u, v, huv, hu⟩ := hb j hj
    refine ⟨a ++ u, v, by simp [huv, List.append_assoc], ?_⟩
    rw [height_append]
    omega

/-- Separated pieces preserve all geometric and longitudinal metadata. -/
theorem spec_append {width : Nat} {start middle finish : ℤ}
    {ma mb incoming join outgoing : Nat} {a b : List Nat}
    (ha : Spec width start middle ma incoming join a)
    (hb : Spec width middle finish mb join outgoing b)
    (hupper : IsUpperCap xStep a) (hlower : NonnegativePrefixes xStep b) :
    Spec width start finish (ma ||| mb) incoming outgoing (a ++ b) := by
  refine ⟨?_, ?_, ?_, ?_, span_append ha.span hb.span,
    crossings_append ha.span.1 ha.crossings hb.crossings⟩
  · intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · exact ha.small d hd
    · exact hb.small d hd
  · rw [List.map_append]
    apply compatible_append (toDirection incoming) _ _ ha.compatible
    have hlast : (a.map toDirection).getLastD (toDirection incoming) = toDirection join := by
      rw [List.getLastD_map, ha.last]
    rw [hlast]
    exact hb.compatible
  · exact separated_concat_valid 0 incoming join hupper hlower ha.last ha.valid hb.valid
  · have hlast : (a ++ b).getLastD incoming = b.getLastD (a.getLastD incoming) := by simp
    rw [hlast, ha.last, hb.last]

/-- Confined interior prefixes supply the occupied-layer bounds before any final exit. -/
lemma span_exit (width : Nat) (start finish : ℤ) (w : List Nat) (d : Nat)
    (hend : start + height longStep w + longStep d = finish)
    (hbound : ∀ u v, w = u ++ v →
      0 ≤ start + height longStep u ∧ start + height longStep u ≤ width) :
    Span width start finish (w ++ [d]) := by
  refine ⟨by simpa [height_append, height_singleton, add_assoc] using hend, ?_⟩
  intro u v huv hv
  rcases List.append_eq_append_iff.mp huv with
    ⟨c, huc, hlast⟩ | ⟨c, hwc, _⟩
  · rcases List.singleton_eq_append_iff.mp hlast with ⟨hc, _⟩ | ⟨_, hvnil⟩
    · have hu : u = w := by simpa [hc] using huc
      rw [hu]
      exact hbound w [] (by simp)
    · exact (hv hvnil).elim
  · exact hbound u c hwc

/-- The geometric specification of every actually counted head or middle piece. -/
theorem leavesAt_spec (width transverse limit startX startY : Nat) (head : Bool)
    (hx : startX ≤ width) (hy : startY ≤ transverse) {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX startY head) :
    Spec width startX (crossRun ⟨startX, 0⟩ w).x (crossRun ⟨startX, 0⟩ w).mask
      (if head then 2 else 0) 0 (w ++ [0]) := by
  have hwords := leavesAt_mem_words width transverse limit startX startY head hw
  have hdirs := words_directions transverse (2 * limit + 1) true (limit - 1)
    (pieceCursor limit startY head) (initialBoard transverse limit)
    (by cases head <;> simp [pieceCursor]) w hwords
  have ha := (leaves_tag width transverse (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit startY head) ⟨startX, 0⟩ (initialBoard transverse limit) hw).2
  refine ⟨?_, hdirs.2, leavesAt_valid width transverse limit startX startY head hy hw,
    by simp, ?_, ?_⟩
  · intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · exact hdirs.1 d hd
    · have : d = 0 := by simpa using hd
      omega
  · exact span_exit width startX _ w 0
      (by simpa only [show longStep 0 = 0 from rfl, add_zero] using
        (crossRun_coordinate width ⟨startX, 0⟩ w hdirs.1 ha).symm)
      (cross_prefix_bounds width ⟨startX, 0⟩ w hx hdirs.1 ha)
  · intro j hj
    have h := crossRun_backward_crossings width ⟨startX, 0⟩ w hdirs.1 ha j hj
    simp only [Nat.zero_testBit, Bool.false_eq_true, false_or] at h
    obtain ⟨u, v, huv, hu⟩ := h
    exact ⟨u, v ++ [0], by simp [huv, List.append_assoc], hu⟩

/-- The small masks used by this certificate have an exact reflected-bit interpretation. -/
private theorem reflectMask_checked :
    ∀ width : Fin 5, ∀ mask : Fin 16,
      reflectMask width.val mask.val < 2 ^ width.val ∧
        ∀ j : Fin 4, j.val < width.val →
          (reflectMask width.val mask.val).testBit j.val =
            mask.val.testBit (width.val - 1 - j.val) := by
  decide

/-- A set bit in a reflected mask identifies the corresponding original cut. -/
lemma reflectMask_witness (width mask : Nat) (hw : width ≤ 4) (hm : mask < 16)
    (j : Nat) (hj : (reflectMask width mask).testBit j = true) :
    j < width ∧ mask.testBit (width - 1 - j) = true := by
  have hc := reflectMask_checked ⟨width, by omega⟩ ⟨mask, hm⟩
  dsimp only at hc
  have hjw : j < width := by
    by_contra h
    have hp : 2 ^ width ≤ 2 ^ j :=
      pow_le_pow_right₀ (by decide : (1 : Nat) ≤ 2) (by omega)
    have hb := Nat.ge_two_pow_of_testBit hj
    omega
  exact ⟨hjw, (hc.2 ⟨j, by omega⟩ hjw) ▸ hj⟩

/-- Reversing a head reflects its longitudinal backward-cut witnesses. -/
theorem tail_crossings (width mask : Nat) (hw : width ≤ 4) (hm : mask < 16)
    (w : List Nat) (hc : Crossings 0 mask (w ++ [0])) :
    Crossings ((width : ℤ) - height longStep w) (reflectMask width mask) (tailWord w) := by
  intro j hj
  obtain ⟨hjw, hjm⟩ := reflectMask_witness width mask hw hm j hj
  obtain ⟨u, v, huv, hu⟩ := hc (width - 1 - j) hjm
  have hexit : ∃ v', v = v' ++ [0] ∧ w = u ++ [3] ++ v' := by
    have hlast : v ≠ [] := by
      intro hv
      have h := congrArg (fun a : List Nat => a.getLastD 0) huv
      simp [hv] at h
    have hsplit : ∃ v' d, v = v' ++ [d] :=
      ⟨v.dropLast, v.getLast hlast, (List.dropLast_append_getLast hlast).symm⟩
    obtain ⟨v', d, rfl⟩ := hsplit
    have hd : d = 0 := by
      have heq : w ++ [0] = ((u ++ [3]) ++ v') ++ [d] := by
        simpa only [List.append_assoc] using huv
      have h := congrArg (fun a : List Nat => a.getLastD 0) heq
      simpa only [List.getLastD_concat] using h.symm
    subst d
    refine ⟨v', rfl, ?_⟩
    apply List.append_cancel_right
    simpa only [List.append_assoc] using huv
  obtain ⟨v', rfl, hwv⟩ := hexit
  have hh := congrArg (height longStep) hwv
  have hthree : longStep 3 = -1 := rfl
  simp only [height_append, height_singleton, hthree, zero_add] at hh hu
  have hcut : ((width - 1 - j : Nat) : ℤ) = (width : ℤ) - 1 - j := by omega
  refine ⟨v'.reverse.map flipZ, u.reverse.map flipZ ++ [2], ?_, ?_⟩
  · simp only [tailWord, hwv, List.reverse_append, List.reverse_singleton,
      List.map_append, List.map_singleton, show flipZ 3 = 3 from rfl, List.append_assoc]
  · rw [secondaryHeight_flipZ, height_reverse]
    change (width : ℤ) - height longStep w + height longStep v' = (j : ℤ) + 1
    omega

/-- Every counted head produces a confined terminal piece with the reflected mask. -/
theorem leavesAt_tail_spec (width transverse limit startY : Nat)
    (hwidth : width ≤ 4) (hy : startY ≤ transverse) {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit 0 startY true) :
    Spec width ((width : ℤ) - (crossRun ⟨0, 0⟩ w).x) ((width : ℤ) + 1)
      (reflectMask width (crossRun ⟨0, 0⟩ w).mask) 0 2 (tailWord w) := by
  have hp := leavesAt_spec width transverse limit 0 startY true (Nat.zero_le _) hy hw
  have hwords := leavesAt_mem_words width transverse limit 0 startY true hw
  have hd := (words_directions transverse (2 * limit + 1) true (limit - 1)
    (pieceCursor limit startY true) (initialBoard transverse limit)
    (by simp [pieceCursor]) w hwords).1
  have ha := (leaves_tag width transverse (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit startY true) ⟨0, 0⟩ (initialBoard transverse limit) hw).2
  have he : height longStep w = ((crossRun ⟨0, 0⟩ w).x : ℤ) := by
    simpa using (crossRun_coordinate width ⟨0, 0⟩ w hd ha).symm
  have hb : ∀ u v, w = u ++ v → 0 ≤ height longStep u ∧ height longStep u ≤ width := by
    simpa using cross_prefix_bounds width ⟨0, 0⟩ w (Nat.zero_le _) hd ha
  have hm : (crossRun ⟨0, 0⟩ w).mask < 16 := by
    have h := crossRun_mask_lt width ⟨0, 0⟩ w (Nat.zero_le _) (by simp) ha
    have hpow : 2 ^ width ≤ 2 ^ 4 :=
      pow_le_pow_right₀ (by decide : (1 : Nat) ≤ 2) hwidth
    omega
  obtain ⟨hsmall, hcompatible⟩ := tailWord_directions w hd hp.compatible
  refine ⟨hsmall, hcompatible, tailWord_valid w hd hp.valid, by simp [tailWord], ?_, ?_⟩
  · apply span_exit width _ _ (w.reverse.map flipZ) 2
    · rw [secondaryHeight_flipZ, height_reverse, he]
      change (width : ℤ) - (crossRun ⟨0, 0⟩ w).x + (crossRun ⟨0, 0⟩ w).x + 1 = _
      omega
    · simpa only [he] using tail_interior_bounds width w hb
  · simpa only [he] using tail_crossings width (crossRun ⟨0, 0⟩ w).mask hwidth hm w hp.crossings

/-- A completely crossed longitudinal slab is an irreducible bridge. -/
theorem spec_irreducible {width incoming : Nat} {w : List Nat}
    (h : Spec width 0 ((width : ℤ) + 1) (2 ^ width - 1) incoming 2 w) :
    IsIrreducible longStep w := by
  have hend : height longStep w = (width : ℤ) + 1 := by simpa using h.span.1
  apply backward_crossings_irreducible
  · refine ⟨by rw [hend]; omega, ?_⟩
    intro u v huv hv
    have hb := h.span.2 u v huv hv
    constructor <;> simp only [zero_add] at hb ⊢ <;> omega
  · intro c hc hcw
    let j := (c - 1).toNat
    have hj : (j : ℤ) = c - 1 := Int.toNat_of_nonneg (by omega)
    have hjw : j < width := by omega
    obtain ⟨u, v, huv, hu⟩ := h.crossings j (by simp [Nat.testBit_two_pow_sub_one, hjw])
    exact ⟨u, 3, v, huv, by simpa [hj] using hu, rfl⟩

end RubiksSnake.CapPieces

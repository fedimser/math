import RubiksSnake.SnAsymptotic.BridgeCaps
import RubiksSnake.SnAsymptotic.SlabBlockValidity
import RubiksSnake.SnAsymptotic.SlabIrreducible

/-!
# Geometric separation with overhanging caps

Only an upper bound on the first piece and a lower bound on the second
piece are needed. The first piece need not remain above its starting plane,
and the second need not remain below its final plane.
-/

namespace RubiksSnake.CapGeometry

open SlabEnumeration BridgeWords

/-- The increment of one cardinal direction in a chosen coordinate. -/
def coordinateStep (axis : Fin 3) (d : Nat) : ℤ := vectorOf d axis

/-- Coordinate displacement of a direction word. -/
lemma endPoint_coordinate (axis : Fin 3) (p : Vec3) (ds : List Nat) :
    endPoint p ds axis = p axis + height (coordinateStep axis) ds := by
  induction ds generalizing p with
  | nil => simp [endPoint, height]
  | cons outgoing ds ih =>
    simpa [endPoint, height, addVec, coordinateStep, add_assoc] using
      ih (addVec p (vectorOf outgoing))

/-- Every occupied center corresponds to a proper prefix, in any coordinate. -/
lemma path_center_coordinate (axis : Fin 3) (p : Vec3) (incoming : Nat)
    (ds : List Nat) :
    ∀ w ∈ path p incoming ds, ∃ u v,
      ds = u ++ v ∧ v ≠ [] ∧
        w.center axis = p axis + height (coordinateStep axis) u := by
  induction ds generalizing p incoming with
  | nil => simp
  | cons outgoing ds ih =>
    intro w hw
    rw [path_cons] at hw
    rcases List.mem_cons.mp hw with rfl | hw
    · exact ⟨[], outgoing :: ds, rfl, by simp,
        by simp [placed, SlabBoard.Placed.wedge]⟩
    · obtain ⟨u, v, hds, hv, hcenter⟩ := ih _ _ w hw
      refine ⟨outgoing :: u, v, by simp [hds], hv, ?_⟩
      simpa [height, addVec, coordinateStep, add_assoc] using hcenter

/-- An upper cap's occupied centers are strictly below its exit plane. -/
lemma upperCap_centers (axis : Fin 3) (p : Vec3) (incoming : Nat)
    {ds : List Nat} (h : IsUpperCap (coordinateStep axis) ds) :
    ∀ w ∈ path p incoming ds, w.center axis < endPoint p ds axis := by
  intro w hw
  obtain ⟨u, v, hds, hv, hcenter⟩ := path_center_coordinate axis p incoming ds w hw
  have := h u v hds hv
  rw [endPoint_coordinate]
  omega

/-- Nonnegative prefixes keep all occupied centers on or above the entry plane. -/
lemma lowerCap_centers (axis : Fin 3) (p : Vec3) (incoming : Nat)
    {ds : List Nat} (h : NonnegativePrefixes (coordinateStep axis) ds) :
    ∀ w ∈ path p incoming ds, p axis ≤ w.center axis := by
  intro w hw
  obtain ⟨u, v, hds, _, hcenter⟩ := path_center_coordinate axis p incoming ds w hw
  have := h u v hds
  omega

/-- Upper and lower caps join without cross-piece collisions, in any axis or frame. -/
theorem separated_concat_valid (axis : Fin 3) (incoming join : Nat)
    {a b : List Nat}
    (ha : IsUpperCap (coordinateStep axis) a)
    (hb : NonnegativePrefixes (coordinateStep axis) b)
    (hlast : a.getLastD incoming = join)
    (havalid : (path zeroVec incoming a).Pairwise interiorDisjoint)
    (hbvalid : (path zeroVec join b).Pairwise interiorDisjoint) :
    (path zeroVec incoming (a ++ b)).Pairwise interiorDisjoint := by
  rw [path_append, hlast]
  refine List.pairwise_append.mpr ⟨havalid, path_valid_at _ _ _ hbvalid, ?_⟩
  intro old hold new hnew
  have hlo := upperCap_centers axis zeroVec incoming ha old hold
  have hhi := lowerCap_centers axis (endPoint zeroVec a) join hb new hnew
  left
  intro heq
  have := congrFun heq axis
  omega

/-- The net transverse displacement distinguishes a positive cap construction
from any construction with nonpositive transverse displacement. -/
theorem head_tail_height_pos {step : Nat → ℤ} {head tail : List Nat}
    (hh : IsHead step head) (ht : NonnegativePrefixes step tail) :
    0 < height step (head ++ tail) := by
  have hhead := upperCap_height_pos hh.1 hh.2.1
  have htail := ht tail [] (by simp)
  rw [height_append]
  omega

/-- Starting anywhere in a slab gives an upper cap, not necessarily a bridge. -/
theorem words_upperCap (width side remaining : Nat) (s : Cursor) (board : ByteArray)
    (hx : s.x ≤ width) {w : List Nat}
    (hw : w ∈ words width side true remaining s board) :
    IsUpperCap xStep (w ++ [0]) := by
  have hspan := words_span width side true remaining s board hx hw
  have hheight : (s.x : ℤ) + height xStep (w ++ [0]) = (width : ℤ) + 1 := by
    rw [height_append, height_singleton]
    change (s.x : ℤ) + (height xStep w + 1) = _
    omega
  intro u v huv hv
  have hprefix : ∃ rest, w = u ++ rest := by
    rcases List.append_eq_append_iff.mp huv with
      ⟨c, hu, hlast⟩ | ⟨c, hword, _⟩
    · rcases List.singleton_eq_append_iff.mp hlast with ⟨hc, _⟩ | ⟨_, hvnil⟩
      · exact ⟨[], by simpa only [hc, List.append_nil] using hu.symm⟩
      · exact (hv hvnil).elim
    · exact ⟨c, hword⟩
  obtain ⟨rest, hrest⟩ := hprefix
  have hp := hspan.2 u rest hrest
  omega

/-- Every positive cut up to the internal endpoint has a backward-crossing witness. -/
theorem words_positive_crossings (width side remaining : Nat) (s : Cursor) (board : ByteArray)
    (hx : s.x ≤ width) (hmask : s.mask = 0) {w : List Nat}
    (hw : w ∈ words width side true remaining s board)
    (c : ℤ) (hc : 0 < c) (hcw : c ≤ height xStep w) :
    ∃ u v, w = u ++ [1] ++ v ∧ height xStep u = c := by
  have hspan := (words_span width side true remaining s board hx hw).1
  let j := (c + s.x - 1).toNat
  have hjcast : (j : ℤ) = c + s.x - 1 := Int.toNat_of_nonneg (by omega)
  have hj : j < width := by omega
  have hcross := words_backward_crossings width side remaining s board hw j hj
  simp only [hmask, Nat.zero_testBit, Bool.false_eq_true, false_or] at hcross
  obtain ⟨u, v, huv, hu⟩ := hcross
  exact ⟨u, v, huv, by omega⟩

/-- The existing full-cut-mask search already certifies the overhanging head condition. -/
theorem words_isHead (width side remaining : Nat) (s : Cursor) (board : ByteArray)
    (hx : s.x ≤ width) (hmask : s.mask = 0) {w : List Nat}
    (hw : w ∈ words width side true remaining s board) :
    IsHead xStep (w ++ [0]) := by
  apply head_of_backward_crossings (words_upperCap width side remaining s board hx hw)
    (by simp)
  intro c hc hcw
  have hheight : height xStep (w ++ [0]) = height xStep w + 1 := by
    rw [height_append, height_singleton]
    rfl
  obtain ⟨u, v, huv, hu⟩ := words_positive_crossings width side remaining s board hx hmask
    hw c hc (by omega)
  refine ⟨u, 1, v ++ [0], ?_, ?_, rfl⟩
  · rw [huv]
    simp only [List.append_assoc]
  · exact hu

end RubiksSnake.CapGeometry

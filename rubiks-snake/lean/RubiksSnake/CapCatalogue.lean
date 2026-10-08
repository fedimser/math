import RubiksSnake.CapCrossing

/-! Geometric finite catalogues underlying the short cap coefficient arrays. -/

namespace RubiksSnake.CapEnumeration

open SlabEnumeration BridgeWords CapGeometry CardinalDirections

/-- Initial primary-slab cursor used by the executable piece counter. -/
def pieceCursor (limit startY : Nat) (head : Bool) : Cursor :=
  let side := 2 * limit + 1
  ⟨0, startY, startY * side * side + limit * side + limit, if head then 2 else 0, 0⟩

/-- Tagged short-piece words from a fixed starting layer and incoming direction. -/
def leavesAt (width transverse limit startX startY : Nat) (head : Bool) :
    List (List Nat × Nat) :=
  leaves width transverse (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit startY head) ⟨startX, 0⟩ (initialBoard transverse limit)

/-- The initial cursor is located at its chosen primary layer. -/
lemma pieceCursor_located (transverse limit startY : Nat) (head : Bool)
    (hy : startY ≤ transverse) :
    Located transverse limit (limit - 1) (pieceCursor limit startY head)
      ![startY, 0, 0] := by
  constructor
  · simp [pieceCursor]
  · rfl
  · exact hy
  · simp [pieceCursor]
  · simp [pieceCursor]
  · have hp : packed limit ![startY, 0, 0] =
        ((startY * (2 * limit + 1) * (2 * limit + 1) +
          limit * (2 * limit + 1) + limit : Nat) : ℤ) := by
      simp [packed]
      ring
    change _ = (packed limit ![startY, 0, 0]).toNat
    rw [hp, Int.toNat_natCast]
    rfl
  · cases head <;> simp [pieceCursor]

/-- A tagged catalogue entry belongs to the old geometric slab enumeration. -/
lemma leavesAt_mem_words (width transverse limit startX startY : Nat) (head : Bool)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX startY head) :
    w ∈ words transverse (2 * limit + 1) true (limit - 1)
      (pieceCursor limit startY head) (initialBoard transverse limit) :=
  leaves_subset width transverse (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit startY head) ⟨startX, 0⟩ (initialBoard transverse limit) hw

/-- Every completed catalogue word has pairwise-disjoint wedges in its own entry frame. -/
theorem leavesAt_valid (width transverse limit startX startY : Nat) (head : Bool)
    (hy : startY ≤ transverse) {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX startY head) :
    (path zeroVec (if head then 2 else 0) (w ++ [0])).Pairwise interiorDisjoint := by
  have h := (words_path_sound transverse limit true (limit - 1)
    (pieceCursor limit startY head) (initialBoard transverse limit) ![startY, 0, 0] []
    (pieceCursor_located transverse limit startY head hy)
    (by simp [initialBoard, ByteArray.size]) (SlabBoard.empty_represents _ _)
    w (leavesAt_mem_words width transverse limit startX startY head hw)).1
  rw [path_at, List.pairwise_map] at h
  exact h.imp fun h => (translateWedge_interiorDisjoint _ _ _).mp h

/-- Every completed piece is a canonical upper cap, including nonminimal entry layers. -/
theorem leavesAt_head (width transverse limit startX startY : Nat) (head : Bool)
    (hy : startY ≤ transverse) {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX startY head) :
    IsHead xStep (w ++ [0]) :=
  words_isHead transverse (2 * limit + 1) (limit - 1)
    (pieceCursor limit startY head) (initialBoard transverse limit) hy rfl
    (leavesAt_mem_words width transverse limit startX startY head hw)

/-- A piece entering at its minimal primary level has nonnegative prefixes. -/
theorem leavesAt_nonnegative (width transverse limit startX : Nat) (head : Bool)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX 0 head) :
    NonnegativePrefixes xStep (w ++ [0]) := by
  have hspan := words_span transverse (2 * limit + 1) true (limit - 1)
    (pieceCursor limit 0 head) (initialBoard transverse limit) (by simp [pieceCursor])
    (leavesAt_mem_words width transverse limit startX 0 head hw)
  apply nonnegative_prefixes_append
  · intro u v huv
    simpa [pieceCursor] using (hspan.2 u v huv).1
  · intro u v huv
    rcases List.singleton_eq_append_iff.mp huv with ⟨rfl, _⟩ | ⟨rfl, _⟩
    · simp
    · change (0 : ℤ) ≤ 1
      decide

/-- Middle pieces are ordinary irreducible bridges. -/
theorem leavesAt_middle (width transverse limit startX : Nat)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX 0 false) :
    IsIrreducible xStep (w ++ [0]) := by
  have hh := leavesAt_head width transverse limit startX 0 false (Nat.zero_le _) hw
  have hn := leavesAt_nonnegative width transverse limit startX false hw
  refine ⟨⟨upperCap_height_pos hh.1 hh.2.1, fun u v huv hv =>
    ⟨hn u v huv, hh.1 u v huv hv⟩⟩, ?_⟩
  rintro ⟨a, b, ha, hb, hab⟩
  exact hh.2.2 ⟨a, b, bridge_upperCap ha, bridge_not_nil ha, hb, hab⟩

/-- The full primary cut mask guarantees an actual visit to the bottom layer. -/
lemma words_minimum (transverse side remaining : Nat) (s : Cursor) (board : ByteArray)
    (hy : s.x ≤ transverse) (hm : s.mask = 0) {w : List Nat}
    (hw : w ∈ words transverse side true remaining s board) :
    ∃ u v, w = u ++ v ∧ (s.x : ℤ) + height xStep u = 0 := by
  by_cases hs : s.x = 0
  · exact ⟨[], w, by simp, by simp [hs]⟩
  · have hcross := words_backward_crossings transverse side remaining s board hw 0 (by omega)
    simp only [hm, Nat.zero_testBit, Bool.false_eq_true, false_or] at hcross
    obtain ⟨u, v, huv, hu⟩ := hcross
    refine ⟨u ++ [1], v, by simpa only [List.append_assoc] using huv, ?_⟩
    rw [height_append, height_singleton]
    change (s.x : ℤ) + (height xStep u + -1) = 0
    omega

/-- A head word determines its transverse span and starting layer uniquely. -/
theorem leavesAt_parameters_unique (width limit startX r s y z : Nat) (head : Bool)
    (hy : y ≤ r) (hz : z ≤ s)
    {w : List Nat} {tag label : Nat}
    (hw : (w, tag) ∈ leavesAt width r limit startX y head)
    (hv : (w, label) ∈ leavesAt width s limit startX z head) :
    r = s ∧ y = z := by
  have hwa := leavesAt_mem_words width r limit startX y head hw
  have hwb := leavesAt_mem_words width s limit startX z head hv
  have hsa := words_span r (2 * limit + 1) true (limit - 1)
    (pieceCursor limit y head) (initialBoard r limit) hy hwa
  have hsb := words_span s (2 * limit + 1) true (limit - 1)
    (pieceCursor limit z head) (initialBoard s limit) hz hwb
  obtain ⟨u, v, huv, hmin⟩ := words_minimum r (2 * limit + 1) (limit - 1)
    (pieceCursor limit y head) (initialBoard r limit) hy rfl hwa
  obtain ⟨a, b, hab, hmin'⟩ := words_minimum s (2 * limit + 1) (limit - 1)
    (pieceCursor limit z head) (initialBoard s limit) hz rfl hwb
  have habound := (hsa.2 a b hab).1
  have hubound := (hsb.2 u v huv).1
  simp only [pieceCursor] at hmin hmin' habound hubound hsa hsb
  constructor <;> omega

/-- Concatenate the catalogues for finitely many span/start-layer pairs. -/
def catalogue (width limit startX : Nat) (head : Bool) (parameters : List (Nat × Nat)) :
    List (List Nat × Nat) :=
  parameters.flatMap fun p => leavesAt width p.1 limit startX p.2 head

/-- Different layer/span parameters cannot duplicate an internal word. -/
theorem catalogue_nodup (width limit startX : Nat) (head : Bool)
    (parameters : List (Nat × Nat)) (hparams : parameters.Nodup)
    (hvalid : ∀ p ∈ parameters, p.2 ≤ p.1) :
    ((catalogue width limit startX head parameters).map Prod.fst).Nodup := by
  rw [catalogue, List.map_flatMap]
  apply List.nodup_flatMap.mpr
  constructor
  · intro p _
    exact leaves_nodup width p.1 (2 * limit + 1) limit (limit - 1)
      (pieceCursor limit p.2 head) ⟨startX, 0⟩ (initialBoard p.1 limit)
  · apply hparams.imp_of_mem
    intro a b ha hb hne w hwa hwb
    obtain ⟨⟨u, tag⟩, hu, rfl⟩ := List.mem_map.mp hwa
    obtain ⟨⟨v, label⟩, hv, heq⟩ := List.mem_map.mp hwb
    dsimp only at heq
    subst v
    have h := leavesAt_parameters_unique width limit startX a.1 b.1 a.2 b.2 head
      (hvalid a ha) (hvalid b hb) hu hv
    exact hne (Prod.ext h.1 h.2)

/-- Restoring traversal arrays have exactly the expected coefficient dimension. -/
@[simp] theorem countsAt_size (width transverse limit startX startY : Nat) (head : Bool) :
    (countsAt width transverse limit startX startY head).size =
      (width + 1) * 2 ^ width * (limit + 1) := by
  unfold countsAt
  dsimp only
  rw [search_eq_countTree]
  simp

/-- Every native short-piece coefficient equals the tagged catalogue count. -/
theorem countsAt_get (width transverse limit startX startY : Nat) (head : Bool)
    (n : Nat) (hn : n < (width + 1) * 2 ^ width * (limit + 1)) :
    (countsAt width transverse limit startX startY head)[n]! =
      histogram n (leavesAt width transverse limit startX startY head) := by
  unfold countsAt
  dsimp only
  rw [search_eq_countTree]
  rw [countTree_get _ _ _ _ _ _ _ _ _ n (by simpa using hn)]
  simp only [leavesAt, pieceCursor, initialBoard]
  simp [hn]

end RubiksSnake.CapEnumeration

import RubiksSnake.SnAsymptotic.CapPieces

/-! Exact coefficient-slot semantics and aggregation of the finite piece catalogues. -/

namespace RubiksSnake.CapEnumeration

open SlabEnumeration BridgeWords CapPieces

/-- All allowed head span/start-layer pairs in the certificate. -/
def headParameters : List (Nat × Nat) := [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

/-- The middle pieces always enter at their lowest transverse layer. -/
def middleParameters : List (Nat × Nat) := [(0, 0), (1, 0), (2, 0)]

/-- Tagged head catalogue, before its final exit direction is appended. -/
def heads (width limit : Nat) : List (List Nat × Nat) :=
  catalogue width limit 0 true headParameters

/-- Tagged middle catalogue from one longitudinal coordinate. -/
def middles (width limit start : Nat) : List (List Nat × Nat) :=
  catalogue width limit start false middleParameters

/-- A bounded base/remainder representation has its expected quotient. -/
private lemma packed_div (a base rest : Nat) (hb : 0 < base) (hr : rest < base) :
    (a * base + rest) / base = a := by
  rw [Nat.mul_comm a base, Nat.mul_add_div hb, Nat.div_eq_of_lt hr, Nat.add_zero]

/-- Coefficient-slot decoding recovers all three stored fields. -/
theorem slot_decode (width limit : Nat) (s : CrossState) (n : Nat)
    (hm : s.mask < 2 ^ width) (hn : n ≤ limit) :
    tagFinish width limit (slot width limit s n) = s.x ∧
      tagMask width limit (slot width limit s n) = s.mask ∧
      tagLength limit (slot width limit s n) = n := by
  have hdiv : slot width limit s n / (limit + 1) = s.x * 2 ^ width + s.mask :=
    packed_div _ _ _ (by omega) (by omega)
  refine ⟨?_, ?_, ?_⟩
  · unfold tagFinish
    rw [hdiv, packed_div _ _ _ (by positivity) hm]
  · unfold tagMask
    rw [hdiv, Nat.mul_add_mod_of_lt hm]
  · exact Nat.mul_add_mod_of_lt (by omega)

/-- Every tag belongs to its native histogram array and has positive wedge length. -/
theorem leavesAt_tag_spec (width transverse limit startX startY : Nat) (head : Bool)
    (hlimit : 0 < limit) (hx : startX ≤ width)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leavesAt width transverse limit startX startY head) :
    tag < (width + 1) * 2 ^ width * (limit + 1) ∧
      tagFinish width limit tag = (crossRun ⟨startX, 0⟩ w).x ∧
      tagMask width limit tag = (crossRun ⟨startX, 0⟩ w).mask ∧
      tagLength limit tag = w.length + 1 := by
  have hwords := leavesAt_mem_words width transverse limit startX startY head hw
  have hlen := words_length_le transverse (2 * limit + 1) true (limit - 1)
    (pieceCursor limit startY head) (initialBoard transverse limit) w hwords
  obtain ⟨htag, ha⟩ := leaves_tag width transverse (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit startY head) ⟨startX, 0⟩ (initialBoard transverse limit) hw
  have he := crossRun_le width ⟨startX, 0⟩ w hx ha
  have hm := crossRun_mask_lt width ⟨startX, 0⟩ w hx (by simp) ha
  have hn : w.length + 1 ≤ limit := by omega
  simp only [pieceCursor, Nat.zero_add] at htag
  rw [htag]
  refine ⟨?_, slot_decode width limit _ _ hm hn⟩
  unfold slot
  have hbase : (crossRun ⟨startX, 0⟩ w).x * 2 ^ width +
      (crossRun ⟨startX, 0⟩ w).mask < (width + 1) * 2 ^ width := by
    have hxmul := Nat.mul_le_mul_right (2 ^ width) he
    rw [Nat.add_mul, Nat.one_mul]
    omega
  have hmul := Nat.mul_le_mul_right (limit + 1) (Nat.succ_le_of_lt hbase)
  rw [Nat.succ_mul] at hmul
  omega

/-- Every head word appears only once, even across its different spans and entry layers. -/
theorem heads_nodup (width limit : Nat) : ((heads width limit).map Prod.fst).Nodup :=
  catalogue_nodup width limit 0 true headParameters (by decide) (by decide)

/-- Every middle word appears only once at a fixed entry state. -/
theorem middles_nodup (width limit start : Nat) :
    ((middles width limit start).map Prod.fst).Nodup :=
  catalogue_nodup width limit start false middleParameters (by decide) (by decide)

/-- Pointwise addition preserves the coefficient-array shape. -/
@[simp] lemma addCounts_size (a b : Array Nat) : (addCounts a b).size = a.size := by
  simp [addCounts]

/-- Pointwise array addition has the expected in-range coefficient. -/
lemma addCounts_get (a b : Array Nat) (n : Nat) (hn : n < a.size) :
    (addCounts a b)[n]! = a[n]! + b[n]! := by
  simp [addCounts, getElem!_pos, hn]

attribute [local irreducible] countsAt leavesAt

/-- The aggregate head histogram has one entry for every endpoint/mask/length triple. -/
@[simp] theorem headCounts_size (width limit : Nat) :
    (headCounts width 2 limit).size = (width + 1) * 2 ^ width * (limit + 1) := by
  simp [headCounts, List.range_succ]

/-- Aggregate middle histogram shape. -/
@[simp] theorem middleCounts_size (width limit start : Nat) :
    (middleCounts width 2 limit start).size = (width + 1) * 2 ^ width * (limit + 1) := by
  simp [middleCounts, List.range_succ]

/-- The native head array counts the duplicate-free union of all allowed head catalogues. -/
theorem headCounts_get (width limit n : Nat)
    (hn : n < (width + 1) * 2 ^ width * (limit + 1)) :
    (headCounts width 2 limit)[n]! = histogram n (heads width limit) := by
  simp [-getElem!_pos, headCounts, List.range_succ, addCounts_get, hn, countsAt_get,
    heads, catalogue, headParameters, histogram, List.filter_append, Nat.add_assoc]
  simp [hn]

/-- The native middle array counts the duplicate-free middle catalogue. -/
theorem middleCounts_get (width limit start n : Nat)
    (hn : n < (width + 1) * 2 ^ width * (limit + 1)) :
    (middleCounts width 2 limit start)[n]! = histogram n (middles width limit start) := by
  simp [-getElem!_pos, middleCounts, List.range_succ, addCounts_get, hn, countsAt_get,
    middles, catalogue, middleParameters, histogram, List.filter_append, Nat.add_assoc]
  simp [hn]

end RubiksSnake.CapEnumeration

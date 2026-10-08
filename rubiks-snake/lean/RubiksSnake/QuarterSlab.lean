import RubiksSnake.BridgeSymmetry
import RubiksSnake.PrunedSlabEnumeration

/-!
# One representative per fourfold slab orbit

Only the first `+y` branch is counted. The geometric symmetry theorem, rather
than an unchecked symmetry assumption in the counter, supplies the other
three branches. Remaining-budget pruning is used only for undercounting.
-/

namespace RubiksSnake.QuarterSlab

open SlabEnumeration CardinalDirections

/-- The initial state after the mandatory first transverse step. -/
def cursor (limit : Nat) : Cursor :=
  advance (2 * limit + 1) (initialCursor limit) 2

/-- The board with just the first `+x` to `+y` wedge placed. -/
def board (width limit : Nat) : ByteArray :=
  (initialBoard width limit).set! (initialCursor limit).position (entry 0 0 2)

/-- Candidate suffixes after the first `+y` step, before the final `+x` exit. -/
def suffixes (width remaining : Nat) : List (List Nat) :=
  words width (2 * (remaining + 1) + 1) true remaining
    (cursor (remaining + 1)) (board width (remaining + 1))

/-- Finish each suffix with the mandatory entry branch and terminal exit. -/
def seedWords (width remaining : Nat) : List (List Nat) :=
  (suffixes width remaining).map fun w => 2 :: (w ++ [0])

/-- Looking up any entry of the empty initial board gives byte zero. -/
private lemma initialBoard_get (width limit position : Nat) :
    (initialBoard width limit)[position]! = 0 := by
  by_cases h : position < (initialBoard width limit).size
  · rw [getElem!_pos (initialBoard width limit) position h]
    unfold initialBoard at h ⊢
    change (Array.replicate _ (0 : UInt8))[position]'h = 0
    simp
  · rw [getElem!_neg (initialBoard width limit) position h]
    rfl

/-- Every seed is one of the original geometrically verified slab blocks. -/
lemma seedWords_subset (width remaining : Nat) :
    seedWords width remaining ⊆ blockWords width (remaining + 1) true := by
  intro w hw
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
  apply List.mem_map.mpr
  refine ⟨2 :: v, ?_, by simp⟩
  unfold internalWords
  rw [words]
  apply List.mem_append.mpr
  right
  apply List.mem_flatMap.mpr
  refine ⟨2, by decide, ?_⟩
  rw [initialBoard_get]
  have hmove : canMove width (initialCursor (remaining + 1)) 0 2 = true := by
    simp [canMove, initialCursor]
  rw [if_pos hmove]
  exact List.mem_map.mpr ⟨v, hv, rfl⟩

/-- The seed list has no repeated words. -/
lemma seedWords_nodup (width remaining : Nat) :
    (seedWords width remaining).Nodup := by
  apply (words_nodup width (2 * (remaining + 1) + 1) true remaining
    (cursor (remaining + 1)) (board width (remaining + 1))).map
  intro a b h
  exact List.append_cancel_right (List.cons.inj h).2

/-- A geometric bridge code consisting of one initial-direction branch. -/
def code (width remaining : Nat) : BridgeCode.Code where
  words := seedWords width remaining
  nodup := seedWords_nodup width remaining
  irreducible := fun _ hw => blockWords_irreducible width (remaining + 1)
    (seedWords_subset width remaining hw)
  directions := fun _ hw => blockWords_directions width (remaining + 1) true
    (seedWords_subset width remaining hw)
  last := fun _ hw => blockWords_last width (remaining + 1) true
    (seedWords_subset width remaining hw) 0
  valid := fun _ hw => blockWords_valid width (remaining + 1) true
    (seedWords_subset width remaining hw)

/-- Every seed begins with `+y`, so its four rotated copies are disjoint. -/
lemma heading (width remaining : Nat) : BridgeSymmetry.Heading (code width remaining) 2 := by
  intro w hw
  obtain ⟨v, _, rfl⟩ := List.mem_map.mp hw
  rfl

/-- Executable quarter-orbit count; index `n` still means `n` internal edges. -/
def counts (width remaining : Nat) : Array Nat :=
  let limit := remaining + 1
  let s := cursor limit
  (budgetSearch width (2 * limit + 1) true (2 ^ width) (slabBudgetTable width true)
    remaining s.length s.x s.position s.incoming s.mask
    (board width limit) (Array.replicate (limit + 1) 0)).2

/-- The seed coefficient is the suffix histogram, with one initial edge already used. -/
lemma blocks_length (width remaining n : Nat) :
    (BridgeCode.blocks (code width remaining) (n + 1)).length =
      histogram 1 n (suffixes width remaining) := by
  simp [BridgeCode.blocks, code, seedWords, histogram, List.filter_map, Function.comp_def,
    Nat.add_comm, Nat.add_left_comm]

/-- Budget pruning undercounts the seed code without any completeness assumption. -/
lemma counts_le (width remaining n : Nat) :
    (counts width remaining)[n]?.getD 0 ≤
      (BridgeCode.blocks (code width remaining) (n + 1)).length := by
  rw [blocks_length]
  unfold counts
  dsimp only
  rw [budgetSearch_eq_prunedCountSearch]
  by_cases hn : n < remaining + 2
  · have h := prunedCountSearch_le_countSearch width (2 * (remaining + 1) + 1) true
      (tableBudgetKeep width (2 ^ width) (slabBudgetTable width true)) remaining
      (cursor (remaining + 1)) (board width (remaining + 1))
      (Array.replicate (remaining + 2) 0) n (by simpa using hn)
    rw [countSearch_eq_countTree] at h
    dsimp only at h
    rw [countTree_get _ _ _ _ _ _ _ n (by simpa using hn)] at h
    have hsize := prunedCountSearch_size width (2 * (remaining + 1) + 1) true
      (tableBudgetKeep width (2 ^ width) (slabBudgetTable width true)) remaining
      (cursor (remaining + 1)) (board width (remaining + 1))
      (Array.replicate (remaining + 2) 0)
    simpa [getElem?_pos, getElem!_pos, hsize, hn, cursor, suffixes,
      initialCursor, advance, Nat.add_assoc] using h
  · rw [getElem?_neg]
    · exact Nat.zero_le _
    · simpa [Nat.add_assoc] using hn

/-- Join seed codes of distinct widths; height separates their word lists. -/
def ofSlabs (slabs : List (Nat × Nat)) (hwidths : (slabs.map Prod.fst).Nodup) :
    BridgeCode.Code where
  words := slabs.flatMap fun p => seedWords p.1 p.2
  nodup := by
    apply List.nodup_flatMap.mpr
    refine ⟨fun p _ => seedWords_nodup _ _, ?_⟩
    change (slabs.map Prod.fst).Pairwise (fun a b => a ≠ b) at hwidths
    rw [List.pairwise_map] at hwidths
    apply hwidths.imp
    intro a b hne w hwa hwb
    have ha := blockWords_height a.1 (a.2 + 1) true (seedWords_subset a.1 a.2 hwa)
    have hb := blockWords_height b.1 (b.2 + 1) true (seedWords_subset b.1 b.2 hwb)
    apply hne
    omega
  irreducible := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact (code p.1 p.2).irreducible w hw
  directions := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact (code p.1 p.2).directions w hw
  last := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact (code p.1 p.2).last w hw
  valid := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact (code p.1 p.2).valid w hw

/-- Combining different widths retains the common first direction. -/
lemma ofSlabs_heading (slabs : List (Nat × Nat)) (hwidths : (slabs.map Prod.fst).Nodup) :
    BridgeSymmetry.Heading (ofSlabs slabs hwidths) 2 := by
  intro w hw
  obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
  exact heading p.1 p.2 w hw

/-- A combined seed coefficient is the sum of its separate width coefficients. -/
lemma ofSlabs_blocks_length (slabs : List (Nat × Nat))
    (hwidths : (slabs.map Prod.fst).Nodup) (n : Nat) :
    (BridgeCode.blocks (ofSlabs slabs hwidths) n).length =
      (slabs.map fun p => (BridgeCode.blocks (code p.1 p.2) n).length).sum := by
  simp [BridgeCode.blocks, ofSlabs, code, List.filter_flatMap, List.length_flatMap]

/-- Fourfold symmetry turns any verified quarter-row undercounts into full-code undercounts. -/
theorem fourfold_counts_le (slabs : List (Nat × Nat))
    (hwidths : (slabs.map Prod.fst).Nodup) (n : Nat) :
    4 * (slabs.map fun p => (counts p.1 p.2)[n]?.getD 0).sum ≤
      (BridgeCode.blocks
        (BridgeSymmetry.fourfold (ofSlabs slabs hwidths) (ofSlabs_heading slabs hwidths))
        (n + 1)).length := by
  rw [BridgeSymmetry.fourfold_blocks_length, ofSlabs_blocks_length]
  apply Nat.mul_le_mul_left
  apply List.sum_le_sum
  intro p hp
  exact counts_le p.1 p.2 n

end RubiksSnake.QuarterSlab

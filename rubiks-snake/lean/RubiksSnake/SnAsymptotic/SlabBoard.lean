import RubiksSnake.SnAsymptotic.CardinalDirections
import RubiksSnake.SnAsymptotic.Computation
import Init.Data.ByteArray.Lemmas

/-!
Geometric soundness of slab-board occupancy bytes. The index need not be
injective: aliases can reject placements, but cannot admit an intersection.
-/

namespace RubiksSnake.SlabBoard

open CardinalDirections (Direction vector)

/-- A wedge placement specified by its center and incoming/outgoing cardinal
travel directions. -/
structure Placed where
  center : Vec3
  incoming : Direction
  outgoing : Direction

/-- Convert travel directions to a geometric wedge, reversing the incoming
direction to obtain its entrance face. -/
def Placed.wedge (placed : Placed) : Wedge :=
  Wedge.mk placed.center (negVec (vector placed.incoming)) (vector placed.outgoing)

/-- Safety invariant: each past wedge's indexed cell holds its corner byte or
the full marker `255`; distinct centers are allowed to share an index. -/
def Represents (index : Vec3 → Nat) (board : ByteArray) (past : List Placed) : Prop :=
  ∀ old ∈ past,
    board[index old.center]! = SlabEnumeration.corner old.incoming.val old.outgoing.val ∨
      board[index old.center]! = 255

/-- A proposed wedge fits when its cell is empty or contains exactly the
complementary corner byte. -/
def Fits (index : Vec3 → Nat) (board : ByteArray) (new : Placed) : Prop :=
  board[index new.center]! = 0 ∨
    board[index new.center]! = SlabEnumeration.complement new.incoming.val new.outgoing.val

/-- The precomputed corner table agrees with direct encoding on valid indices. -/
@[simp] lemma corner_lookup : ∀ incoming outgoing : Direction,
    SlabEnumeration.corners[incoming.val * 6 + outgoing.val]! =
      SlabEnumeration.corner incoming.val outgoing.val := by
  intro incoming outgoing
  have hbound : incoming.val * 6 + outgoing.val < 36 := by omega
  have hdiv : (incoming.val * 6 + outgoing.val) / 6 = incoming.val := by omega
  have hmod : (incoming.val * 6 + outgoing.val) % 6 = outgoing.val := by omega
  rw [getElem!_pos _ _ (by simpa [SlabEnumeration.corners] using hbound)]
  simp [SlabEnumeration.corners, hdiv, hmod]

/-- The precomputed complement table agrees with direct encoding on valid indices. -/
@[simp] lemma complement_lookup : ∀ incoming outgoing : Direction,
    SlabEnumeration.complements[incoming.val * 6 + outgoing.val]! =
      SlabEnumeration.complement incoming.val outgoing.val := by
  intro incoming outgoing
  have hbound : incoming.val * 6 + outgoing.val < 36 := by omega
  have hdiv : (incoming.val * 6 + outgoing.val) / 6 = incoming.val := by omega
  have hmod : (incoming.val * 6 + outgoing.val) % 6 = outgoing.val := by omega
  rw [getElem!_pos _ _ (by simpa [SlabEnumeration.complements] using hbound)]
  simp [SlabEnumeration.complements, hdiv, hmod]

/-- No cardinal-direction corner is encoded by the empty-cell byte. -/
lemma corner_ne_zero : ∀ incoming outgoing : Direction,
    SlabEnumeration.corner incoming.val outgoing.val ≠ 0 := by
  decide

/-- No complementary corner is encoded by the full-cell marker. -/
lemma complement_ne_full : ∀ incoming outgoing : Direction,
    SlabEnumeration.complement incoming.val outgoing.val ≠ 255 := by
  decide

/-- Equality of corner and complement bytes is precisely the antipodal
unordered face-pair condition for disjoint wedges at a common center. -/
lemma corner_eq_complement_iff : ∀ incoming outgoing incoming' outgoing' : Direction,
    SlabEnumeration.corner incoming.val outgoing.val =
        SlabEnumeration.complement incoming'.val outgoing'.val ↔
      sameUnorderedPair (negVec (vector incoming)) (vector outgoing)
        (negVec (negVec (vector incoming'))) (negVec (vector outgoing')) := by
  unfold sameUnorderedPair
  decide

/-- Placing into an empty cell records the new wedge's corner byte. -/
@[simp] lemma entry_zero (incoming outgoing : Direction) :
    SlabEnumeration.entry 0 incoming.val outgoing.val =
      SlabEnumeration.corner incoming.val outgoing.val := by
  simp [SlabEnumeration.entry]

/-- Updating any occupied cell marks it full, preventing further acceptance. -/
lemma entry_of_ne_zero {old : UInt8} (h : old ≠ 0) (incoming outgoing : Direction) :
    SlabEnumeration.entry old incoming.val outgoing.val = 255 := by
  simp [SlabEnumeration.entry, h]

/-- Every occupancy update stores either the new corner or the full marker. -/
lemma entry_eq_corner_or_full (old : UInt8) (incoming outgoing : Direction) :
    SlabEnumeration.entry old incoming.val outgoing.val =
        SlabEnumeration.corner incoming.val outgoing.val ∨
      SlabEnumeration.entry old incoming.val outgoing.val = 255 := by
  by_cases h : old = 0
  · subst old
    exact Or.inl (entry_zero incoming outgoing)
  · exact Or.inr (entry_of_ne_zero h incoming outgoing)

/-- With no past wedges, the representation invariant holds for any board. -/
@[simp] lemma empty_represents (index : Vec3 → Nat) (board : ByteArray) :
    Represents index board [] := by
  simp [Represents]

/-- The indexed cell of any represented past wedge is necessarily occupied. -/
lemma represents_ne_zero {index : Vec3 → Nat} {board : ByteArray} {past : List Placed}
    (hrep : Represents index board past) {old : Placed} (hold : old ∈ past) :
    board[index old.center]! ≠ 0 := by
  rcases hrep old hold with hcorner | hfull
  · rw [hcorner]
    exact corner_ne_zero old.incoming old.outgoing
  · rw [hfull]
    decide

/-- A successful byte-level fit test implies disjoint interiors from every
represented past wedge, without requiring an injective board index. -/
theorem fits_disjoint {index : Vec3 → Nat} {board : ByteArray} {past : List Placed}
    {new : Placed} (hrep : Represents index board past) (hfits : Fits index board new) :
    ∀ old ∈ past, interiorDisjoint old.wedge new.wedge := by
  intro old hold
  by_cases hcenter : old.center = new.center
  · apply Or.inr
    have holdrep := hrep old hold
    have hnonzero := represents_ne_zero hrep hold
    rw [hcenter] at holdrep hnonzero
    rcases hfits with hzero | hcomplement
    · exact (hnonzero hzero).elim
    · rcases holdrep with hcorner | hfull
      · exact (corner_eq_complement_iff old.incoming old.outgoing
          new.incoming new.outgoing).mp (hcorner.symm.trans hcomplement)
      · exact (complement_ne_full new.incoming new.outgoing
          (hcomplement.symm.trans hfull)).elim
  · exact Or.inl hcenter

/-- An in-bounds occupancy update preserves representation after adding the
new wedge, even without assuming that the placement fits. -/
theorem represents_place {index : Vec3 → Nat} {board : ByteArray} {past : List Placed}
    {new : Placed} (hrep : Represents index board past)
    (hbound : index new.center < board.size) :
    Represents index
      (board.set! (index new.center)
        (SlabEnumeration.entry board[index new.center]! new.incoming.val new.outgoing.val))
      (new :: past) := by
  intro old hold
  rcases List.mem_cons.mp hold with rfl | hold
  · rw [ByteArray.getElem!_set!_self board _ _ hbound]
    exact entry_eq_corner_or_full _ _ _
  · by_cases hslot : index new.center = index old.center
    · right
      rw [← hslot, ByteArray.getElem!_set!_self board _ _ hbound]
      apply entry_of_ne_zero
      rw [hslot]
      exact represents_ne_zero hrep hold
    · rw [ByteArray.getElem!_set!_ne board _ _ _ hslot]
      exact hrep old hold

end RubiksSnake.SlabBoard

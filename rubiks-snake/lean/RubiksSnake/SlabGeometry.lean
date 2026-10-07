import RubiksSnake.SlabEnumeration
import RubiksSnake.CardinalDirections
import RubiksSnake.BridgeWords
import RubiksSnake.SlabBoard

/-! Geometric invariants of the slab enumeration. -/

namespace RubiksSnake.SlabEnumeration

open CardinalDirections

/-- Interpret a natural direction index modulo six as a cardinal direction. -/
def toDirection (i : Nat) : Direction := ⟨i % 6, Nat.mod_lt _ (by decide)⟩

/-- Valid indices below six are unchanged by conversion to cardinal directions. -/
@[simp] lemma toDirection_val {i : Nat} (hi : i < 6) :
    (toDirection i).val = i := Nat.mod_eq_of_lt hi

/-- The unit lattice vector associated with a natural direction index modulo six. -/
def vectorOf (i : Nat) : Vec3 := vector (toDirection i)

/-- The `x`-coordinate increment of the indexed cardinal step. -/
def xStep (i : Nat) : ℤ := vectorOf i 0

/-- Expand move acceptance into perpendicular axes, the two slab boundary
conditions, and compatibility with the occupancy byte. -/
lemma canMove_spec (width : Nat) (s : Cursor) (old : UInt8) (outgoing : Nat) :
    canMove width s old outgoing = true ↔
      outgoing / 2 ≠ s.incoming / 2 ∧
      (outgoing = 0 → s.x < width) ∧ (outgoing = 1 → 0 < s.x) ∧
      (old = 0 ∨ old = complements[s.incoming * 6 + outgoing]!) := by
  simp [canMove, and_assoc, or_iff_not_imp_left]

/-- Expand exit acceptance into the right boundary, transverse incoming axis,
occupancy fit, and the full backward-crossing mask in irreducible mode. -/
lemma canExit_spec (width : Nat) (irreducible : Bool) (s : Cursor) (old : UInt8) :
    canExit width irreducible s old = true ↔
      s.x = width ∧ s.incoming / 2 ≠ 0 ∧
      (old = 0 ∨ old = complements[s.incoming * 6]!) ∧
      (irreducible = true → s.mask = 2 ^ width - 1) := by
  cases irreducible <;> simp [canExit, and_assoc]

/-- The search's direction list contains precisely the six valid indices. -/
lemma mem_directions {outgoing : Nat} :
    outgoing ∈ directions ↔ outgoing < 6 := by
  simp [directions]
  omega

/-- Advancing a cursor consumes one internal edge. -/
@[simp] lemma advance_length (side : Nat) (s : Cursor) (outgoing : Nat) :
    (advance side s outgoing).length = s.length + 1 := rfl

/-- The chosen outgoing direction becomes the next cursor's incoming direction. -/
@[simp] lemma advance_incoming (side : Nat) (s : Cursor) (outgoing : Nat) :
    (advance side s outgoing).incoming = outgoing := rfl

/-- An accepted cardinal move updates the natural cursor coordinate by the
corresponding integer `x` displacement, without subtraction underflow. -/
lemma move_x_eq (width side : Nat) (s : Cursor) (old : UInt8) (outgoing : Nat)
    (houtgoing : outgoing < 6) (hmove : canMove width s old outgoing = true) :
    ((advance side s outgoing).x : ℤ) = (s.x : ℤ) + xStep outgoing := by
  have hleft := (canMove_spec _ _ _ _).mp hmove |>.2.2.1
  interval_cases outgoing <;>
    simp [advance, xStep, vectorOf, toDirection, vector, ex, ey, ez, negVec]
  rw [Nat.cast_sub (by have := hleft rfl; omega)]
  ring

/-- An accepted move cannot cross the right slab boundary if the cursor
already satisfies `x <= width`. -/
lemma move_x_le (width side : Nat) (s : Cursor) (old : UInt8) (outgoing : Nat)
    (hx : s.x ≤ width) (hmove : canMove width s old outgoing = true) :
    (advance side s outgoing).x ≤ width := by
  have hright := (canMove_spec _ _ _ _).mp hmove |>.2.1
  dsimp [advance]
  split_ifs <;> simp_all
  omega

/-- Signed flattened coordinate with side length `2 * limit + 1`, shifting
the transverse `y` and `z` coordinates by `limit`. -/
def packed (limit : Nat) (p : Vec3) : ℤ :=
  ((p 0 * (2 * limit + 1 : ℕ) + p 1 + limit) * (2 * limit + 1 : ℕ)) + p 2 + limit

/-- Natural board index of a packed coordinate; negative packed values
truncate to zero, while the search invariant guarantees nonnegative values. -/
def index (limit : Nat) (p : Vec3) : Nat := (packed limit p).toNat

/-- Coordinates in the slab box `0 <= x <= width`, `|y|, |z| <= limit`
pack to a nonnegative integer strictly below the allocated board size. -/
lemma packed_bounds (width limit : Nat) (p : Vec3)
    (hx : 0 ≤ p 0 ∧ p 0 ≤ width)
    (hy : -(limit : ℤ) ≤ p 1 ∧ p 1 ≤ limit)
    (hz : -(limit : ℤ) ≤ p 2 ∧ p 2 ≤ limit) :
    0 ≤ packed limit p ∧
      packed limit p < ((width + 1) * (2 * limit + 1) * (2 * limit + 1) : ℕ) := by
  let side : ℤ := 2 * limit + 1
  have hside : 0 ≤ side := by dsimp [side]; omega
  have hy' : 0 ≤ p 1 + limit ∧ p 1 + limit ≤ side - 1 := by dsimp [side]; omega
  have hz' : 0 ≤ p 2 + limit ∧ p 2 + limit ≤ side - 1 := by dsimp [side]; omega
  have hnonneg : 0 ≤ (p 0 * side + (p 1 + limit)) * side + (p 2 + limit) :=
    add_nonneg (mul_nonneg (add_nonneg (mul_nonneg hx.1 hside) hy'.1) hside) hz'.1
  have hupper :
      (p 0 * side + (p 1 + limit)) * side + (p 2 + limit) ≤
        ((width : ℤ) * side + (side - 1)) * side + (side - 1) := by
    gcongr
    · exact hx.2
    · exact hy'.2
    · exact hz'.2
  have heq :
      ((width : ℤ) * side + (side - 1)) * side + (side - 1) =
        ((width : ℤ) + 1) * side * side - 1 := by ring
  rw [heq] at hupper
  constructor
  · simpa [packed, side, Nat.cast_add, Nat.cast_mul, Nat.cast_one, Nat.cast_ofNat,
      add_assoc] using hnonneg
  · simpa [packed, side, Nat.cast_add, Nat.cast_mul, Nat.cast_one, Nat.cast_ofNat,
      add_assoc] using (by omega :
        (p 0 * side + (p 1 + limit)) * side + (p 2 + limit) <
          ((width : ℤ) + 1) * side * side)

/-- Relate a search cursor to a geometric point: the length budget, slab and
transverse bounds, packed position, and valid incoming direction all agree. -/
structure Located (width limit remaining : Nat) (s : Cursor) (p : Vec3) : Prop where
  budget : s.length + remaining ≤ limit
  x_eq : p 0 = s.x
  x_le : s.x ≤ width
  y : -(s.length : ℤ) ≤ p 1 ∧ p 1 ≤ s.length
  z : -(s.length : ℤ) ≤ p 2 ∧ p 2 ≤ s.length
  position : s.position = index limit p
  incoming : s.incoming < 6

/-- A located cursor has a nonnegative packed coordinate and an in-bounds
natural index in the allocated slab board. -/
lemma Located.packed_bounds {width limit remaining : Nat} {s : Cursor} {p : Vec3}
    (h : Located width limit remaining s p) :
    0 ≤ packed limit p ∧
      index limit p < (width + 1) * (2 * limit + 1) * (2 * limit + 1) := by
  have hs : (s.length : ℤ) ≤ limit := by
    have := h.budget
    exact_mod_cast (show s.length ≤ limit by omega)
  have hx : 0 ≤ p 0 ∧ p 0 ≤ width := by
    rw [h.x_eq]
    exact ⟨Nat.cast_nonneg _, by exact_mod_cast h.x_le⟩
  have hy : -(limit : ℤ) ≤ p 1 ∧ p 1 ≤ limit := by have := h.y; omega
  have hz : -(limit : ℤ) ≤ p 2 ∧ p 2 ≤ limit := by have := h.z; omega
  have hp := SlabEnumeration.packed_bounds width limit p hx hy hz
  refine ⟨hp.1, ?_⟩
  unfold index
  omega

/-- Each coordinate of a cardinal unit vector lies between `-1` and `1`. -/
private lemma vector_coordinate_bounds :
    ∀ d : Direction, ∀ i : Fin 3,
      (-1 : ℤ) ≤ vector d i ∧ vector d i ≤ 1 := by decide

/-- The initial cursor represents the origin with the full internal-edge
budget and a valid board position. -/
lemma located_initial (width limit : Nat) :
    Located width limit limit (initialCursor limit) zeroVec := by
  refine ⟨by simp [initialCursor], rfl, by simp [initialCursor],
    by simp [initialCursor, zeroVec], by simp [initialCursor, zeroVec], ?_,
    by simp [initialCursor]⟩
  change limit * (2 * limit + 1) + limit = (packed limit zeroVec).toNat
  have hp : packed limit zeroVec = ((limit * (2 * limit + 1) + limit : Nat) : ℤ) := by
    simp [packed, zeroVec]
  rw [hp, Int.toNat_natCast]

/-- Signed flattened-index displacement for a cardinal step, using strides
`side^2`, `side`, and one for the three coordinate axes. -/
private def offset (side : Nat) : Nat → ℤ
  | 0 => (side * side : Nat)
  | 1 => -(side * side : Nat)
  | 2 => side
  | 3 => -(side : ℤ)
  | 4 => 1
  | _ => -1

/-- Packing a point after a cardinal step adds the corresponding signed stride. -/
private lemma packed_step (limit : Nat) (p : Vec3) (outgoing : Nat)
    (houtgoing : outgoing < 6) :
    packed limit (addVec p (vectorOf outgoing)) =
      packed limit p + offset (2 * limit + 1) outgoing := by
  interval_cases outgoing <;>
    norm_num [packed, offset, addVec, vectorOf, toDirection, vector, ex, ey, ez, negVec,
      Matrix.cons_val_two] <;>
    ring

/-- With a nonnegative current packing, the cursor's natural index update
agrees with packing the translated geometric point. -/
private lemma index_step (limit : Nat) (s : Cursor) (p : Vec3) (outgoing : Nat)
    (houtgoing : outgoing < 6) (hpacked : 0 ≤ packed limit p)
    (hposition : s.position = index limit p) :
    (advance (2 * limit + 1) s outgoing).position =
      index limit (addVec p (vectorOf outgoing)) := by
  change s.position = (packed limit p).toNat at hposition
  rw [index, packed_step limit p outgoing houtgoing]
  interval_cases outgoing <;>
    dsimp only [advance, offset] <;>
    simp only [← sub_eq_add_neg, Int.toNat_add_nat hpacked, Int.toNat_sub', ← hposition]
  all_goals omega

/-- One accepted cardinal move preserves the location invariant while
reducing the remaining internal-edge budget by one. -/
lemma Located.advance {width limit remaining : Nat} {s : Cursor} {p : Vec3}
    (h : Located width limit (remaining + 1) s p)
    (old : UInt8) (outgoing : Nat) (houtgoing : outgoing < 6)
    (hmove : canMove width s old outgoing = true) :
    Located width limit remaining (advance (2 * limit + 1) s outgoing)
      (addVec p (vectorOf outgoing)) := by
  refine ⟨?_, ?_, move_x_le width _ s old outgoing h.x_le hmove, ?_, ?_, ?_, ?_⟩
  · rw [advance_length]
    have := h.budget
    omega
  · rw [move_x_eq width _ s old outgoing houtgoing hmove]
    change p 0 + vectorOf outgoing 0 = (s.x : ℤ) + xStep outgoing
    rw [h.x_eq]
    rfl
  · have hy := h.y
    have hv := vector_coordinate_bounds (toDirection outgoing) 1
    change (-1 : ℤ) ≤ vectorOf outgoing 1 ∧ vectorOf outgoing 1 ≤ 1 at hv
    simp only [advance_length, Nat.cast_add, Nat.cast_one, addVec]
    constructor <;> omega
  · have hz := h.z
    have hv := vector_coordinate_bounds (toDirection outgoing) 2
    change (-1 : ℤ) ≤ vectorOf outgoing 2 ∧ vectorOf outgoing 2 ≤ 1 at hv
    simp only [advance_length, Nat.cast_add, Nat.cast_one, addVec]
    constructor <;> omega
  · exact index_step limit s p outgoing houtgoing h.packed_bounds.1 h.position
  · simpa using houtgoing

end RubiksSnake.SlabEnumeration

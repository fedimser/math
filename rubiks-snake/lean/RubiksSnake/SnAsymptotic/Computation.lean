import Init.Data.ByteArray

/-!
Executable slab enumeration. This small library is compiled to native code
before certificate checking; it does not import the mathematical proofs.
-/

namespace RubiksSnake.SlabEnumeration

/-- Encode the unordered entrance/exit face pair for direction indices `0..5`;
the entrance is opposite to `incoming`, and byte zero is reserved for emptiness. -/
def corner (incoming outgoing : Nat) : UInt8 :=
  UInt8.ofNat (6 * min (incoming ^^^ 1) outgoing + max (incoming ^^^ 1) outgoing + 1)

/-- Encode the antipodal face pair that may share a center with the proposed
wedge without an interior intersection. -/
def complement (incoming outgoing : Nat) : UInt8 :=
  UInt8.ofNat (6 * min incoming (outgoing ^^^ 1) + max incoming (outgoing ^^^ 1) + 1)

/-- Corner bytes indexed by `6 * incoming + outgoing` for all six directions. -/
def corners : Array UInt8 := (Array.range 36).map fun i => corner (i / 6) (i % 6)
/-- Complementary corner bytes in the same incoming/outgoing indexing. -/
def complements : Array UInt8 := (Array.range 36).map fun i => complement (i / 6) (i % 6)

/-- Search state: internal-edge length, current `x`, flattened board position,
incoming direction, and bits recording backward crossings of slab cuts. -/
structure Cursor where
  length : Nat
  x : Nat
  position : Nat
  incoming : Nat
  mask : Nat

/-- Direction indices in the order `+x, -x, +y, -y, +z, -z`. -/
def directions : List Nat := [0, 1, 2, 3, 4, 5]

/-- Test whether a final `+x` wedge can exit at `x = width`; irreducible mode
additionally requires a backward crossing of every internal cut. -/
@[inline] def canExit (width : Nat) (irreducible : Bool) (s : Cursor) (old : UInt8) : Bool :=
  s.x == width && s.incoming / 2 != 0 &&
    (old == 0 || old == complements[s.incoming * 6]!) &&
    (!irreducible || s.mask == 2 ^ width - 1)

/-- Permit perpendicular turns that stay within `0 <= x <= width` and whose
current-center wedge fits the occupancy byte. -/
@[inline] def canMove (width : Nat) (s : Cursor) (old : UInt8) (outgoing : Nat) : Bool :=
  outgoing / 2 != s.incoming / 2 &&
    (outgoing != 0 || s.x < width) && (outgoing != 1 || 0 < s.x) &&
    (old == 0 || old == complements[s.incoming * 6 + outgoing]!)

/-- Advance one internal edge on a board with transverse side length `side`,
incrementing the length and recording a cut bit when the move is in `-x`. -/
@[inline] def advance (side : Nat) (s : Cursor) (outgoing : Nat) : Cursor :=
  let x := if outgoing == 0 then s.x + 1 else if outgoing == 1 then s.x - 1 else s.x
  let position :=
    match outgoing with
    | 0 => s.position + side * side
    | 1 => s.position - side * side
    | 2 => s.position + side
    | 3 => s.position - side
    | 4 => s.position + 1
    | _ => s.position - 1
  ⟨s.length + 1, x, position, outgoing,
    if outgoing == 1 then s.mask ||| (1 <<< x) else s.mask⟩

/-- After an allowed placement, store its corner in an empty cell, or mark
a previously occupied cell full with byte `255`. -/
@[inline] def entry (old : UInt8) (incoming outgoing : Nat) : UInt8 :=
  if old == 0 then corners[incoming * 6 + outgoing]! else 255

/-- Add one accepted block to the count at its internal-edge length index. -/
@[inline] def increment (counts : Array Nat) (length : Nat) : Array Nat :=
  counts.set! length (counts[length]! + 1)

/-- Depth-first slab enumeration with at most `remaining` further internal
edges. Accepted exits increment the current length count; recursive branches
restore their occupancy changes before returning. -/
def countSearch (width side : Nat) (irreducible : Bool) :
    Nat → Cursor → ByteArray → Array Nat → ByteArray × Array Nat
  | remaining, s, board, counts =>
    let old := board[s.position]!
    let counts := if canExit width irreducible s old then increment counts s.length else counts
    match remaining with
    | 0 => (board, counts)
    | remaining + 1 =>
      directions.foldl (fun (board, counts) outgoing =>
        if canMove width s old outgoing then
          let (returned, updated) := countSearch width side irreducible remaining
            (advance side s outgoing)
            (board.set! s.position (entry old s.incoming outgoing)) counts
          (returned.set! s.position old, updated)
        else (board, counts)) (board, counts)

/-- Count accepted slab blocks through internal-edge length `limit`, with entry
`k` counting words of length `k + 1`. Width `d - 1` corresponds to total advance
`d`; irreducible mode requires backward crossings of all internal cuts. -/
def counts (width limit : Nat) (irreducible : Bool) : Array Nat :=
  let side := 2 * limit + 1
  let board := ByteArray.mk (Array.replicate ((width + 1) * side * side) 0)
  let position := limit * side + limit
  (countSearch width side irreducible limit ⟨0, 0, position, 0, 0⟩ board
    (Array.replicate (limit + 1) 0)).2

end RubiksSnake.SlabEnumeration

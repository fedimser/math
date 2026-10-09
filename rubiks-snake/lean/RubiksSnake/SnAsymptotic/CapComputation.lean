import RubiksSnake.SnAsymptotic.Computation

/-!
# Short transverse-cap computation

The original slab cursor tracks the transverse separation coordinate. A second
coordinate and backward-crossing mask restrict each piece to its eventual
longitudinal slab. Only short pieces are enumerated; long blocks are assembled
by a separate coefficient recurrence.
-/

namespace RubiksSnake.CapEnumeration

open SlabEnumeration

/-- Secondary-coordinate state, measured in the eventual longitudinal slab. -/
structure CrossState where
  x : Nat
  mask : Nat
deriving DecidableEq

/-- Keep the secondary coordinate inside its slab. -/
@[inline] def crossAllowed (width : Nat) (s : CrossState) (outgoing : Nat) : Bool :=
  (outgoing != 2 || s.x < width) && (outgoing != 3 || 0 < s.x)

/-- Update the secondary coordinate and its backward-crossing mask. -/
@[inline] def crossAdvance (s : CrossState) (outgoing : Nat) : CrossState :=
  let x := if outgoing == 2 then s.x + 1 else if outgoing == 3 then s.x - 1 else s.x
  ⟨x, if outgoing == 3 then s.mask ||| (1 <<< x) else s.mask⟩

/-- Dense coefficient index: endpoint, backward mask, then wedge length. -/
@[inline] def slot (width limit : Nat) (s : CrossState) (length : Nat) : Nat :=
  (s.x * 2 ^ width + s.mask) * (limit + 1) + length

/-- Count all accepted pieces from one starting state, restoring occupancy after each branch. -/
def search (width transverse side limit : Nat) :
    Nat → Cursor → CrossState → ByteArray → Array Nat → ByteArray × Array Nat
  | remaining, s, cross, board, counts =>
    let old := board[s.position]!
    let counts := if canExit transverse true s old then
      increment counts (slot width limit cross (s.length + 1)) else counts
    match remaining with
    | 0 => (board, counts)
    | remaining + 1 =>
      directions.foldl (fun (board, counts) outgoing =>
        if canMove transverse s old outgoing && crossAllowed width cross outgoing then
          let result := search width transverse side limit remaining
            (advance side s outgoing) (crossAdvance cross outgoing)
            (board.set! s.position (entry old s.incoming outgoing)) counts
          (result.1.set! s.position old, result.2)
        else (board, counts)) (board, counts)

/-- One short-piece histogram. `head` changes the incoming axis, not the exit rule. -/
def countsAt (width transverse limit startX startY : Nat) (head : Bool) : Array Nat :=
  let side := 2 * limit + 1
  let board := ByteArray.mk (Array.replicate ((transverse + 1) * side * side) 0)
  let position := startY * side * side + limit * side + limit
  (search width transverse side limit (limit - 1)
    ⟨0, startY, position, if head then 2 else 0, 0⟩ ⟨startX, 0⟩ board
    (Array.replicate ((width + 1) * 2 ^ width * (limit + 1)) 0)).2

/-- Add two coefficient arrays of the same prescribed index layout. -/
def addCounts (a b : Array Nat) : Array Nat :=
  a.mapIdx fun i n => n + b[i]!

/-- All head caps of transverse span at most `transverse`, from longitudinal level zero. -/
def headCounts (width transverse limit : Nat) : Array Nat :=
  (List.range (transverse + 1)).foldl (fun total r =>
    (List.range (r + 1)).foldl (fun total y =>
      addCounts total (countsAt width r limit 0 y true)) total)
    (Array.replicate ((width + 1) * 2 ^ width * (limit + 1)) 0)

/-- All middle bridges with a prescribed longitudinal starting coordinate. -/
def middleCounts (width transverse limit startX : Nat) : Array Nat :=
  (List.range (transverse + 1)).foldl (fun total r =>
    addCounts total (countsAt width r limit startX 0 false))
    (Array.replicate ((width + 1) * 2 ^ width * (limit + 1)) 0)

/-- One nonzero short-piece coefficient: endpoint, backward mask, length, count. -/
structure Entry where
  finish : Nat
  mask : Nat
  length : Nat
  count : Nat
deriving Repr

/-- Sparse view of a dense coefficient array. -/
def entries (width limit : Nat) (counts : Array Nat) : Array Entry :=
  (Array.range counts.size).foldl (fun rows i =>
    let count := counts[i]!
    if count == 0 then rows
    else rows.push ⟨i / ((limit + 1) * 2 ^ width),
      i / (limit + 1) % 2 ^ width, i % (limit + 1), count⟩) #[]

/-- Reflect a set of longitudinal cuts across the middle of its slab. -/
def reflectMask (width mask : Nat) : Nat :=
  (List.range width).foldl (fun result i =>
    if mask.testBit i then result ||| (1 <<< (width - 1 - i)) else result) 0

/-- Tail entries obtained by reversing and reflecting head entries. -/
def tailEntries (width startX : Nat) (heads : Array Entry) : Array Entry :=
  heads.foldl (fun rows e =>
    if e.finish + startX == width then
      rows.push ⟨width, reflectMask width e.mask, e.length, e.count⟩
    else rows) #[]

/-- Coefficients of head, zero or more middle pieces, and tail, through total degree `degree`.
The mask union enforces a backward crossing of every longitudinal cut. -/
def assembledCounts (width transverse limit degree : Nat) : Array Nat := Id.run do
  let heads := entries width limit (headCounts width transverse limit)
  let middles := (Array.range (width + 1)).map fun start =>
    entries width limit (middleCounts width transverse limit start)
  let tails := (Array.range (width + 1)).map fun start => tailEntries width start heads
  let masks := 2 ^ width
  let stateCount := (width + 1) * masks
  let mut states := Array.replicate ((degree + 1) * stateCount) 0
  let mut result := Array.replicate (degree + 1) 0
  for e in heads do
    if e.length ≤ degree then
      let i := e.length * stateCount + e.finish * masks + e.mask
      states := states.set! i (states[i]! + e.count)
  for length in [:degree + 1] do
    for start in [:width + 1] do
      for mask in [:masks] do
        let count := states[length * stateCount + start * masks + mask]!
        if count != 0 then
          for e in tails[start]! do
            if length + e.length ≤ degree && (mask ||| e.mask) == masks - 1 then
              let n := length + e.length
              result := result.set! n (result[n]! + count * e.count)
          for e in middles[start]! do
            if length + e.length ≤ degree then
              let i := (length + e.length) * stateCount +
                e.finish * masks + (mask ||| e.mask)
              states := states.set! i (states[i]! + count * e.count)
  return result

/-- Decode the final longitudinal coordinate of a coefficient slot. -/
def tagFinish (width limit tag : Nat) : Nat := tag / (limit + 1) / 2 ^ width

/-- Decode its cut mask. -/
def tagMask (width limit tag : Nat) : Nat := tag / (limit + 1) % 2 ^ width

/-- Decode its wedge length. -/
def tagLength (limit tag : Nat) : Nat := tag % (limit + 1)

/-- Terminal acceptance, expressed entirely in terms of a head's coefficient tag. -/
def terminalWeight (width limit n start mask tag : Nat) : Nat :=
  if tagFinish width limit tag + start == width &&
      tagLength limit tag == n &&
      (mask ||| reflectMask width (tagMask width limit tag)) == 2 ^ width - 1
  then 1 else 0

/-- The contribution after selecting one positive-length middle piece. -/
def middleWeight (width limit n mask tag : Nat) (value : Nat → Nat → Nat → Nat) : Nat :=
  if 0 < tagLength limit tag ∧ tagLength limit tag ≤ n then
    value (n - tagLength limit tag) (tagFinish width limit tag)
      (mask ||| tagMask width limit tag)
  else 0

/-- The contribution after selecting the initial head. -/
def headWeight (width limit n tag : Nat) (value : Nat → Nat → Nat → Nat) : Nat :=
  if 0 < tagLength limit tag ∧ tagLength limit tag ≤ n then
    value (n - tagLength limit tag) (tagFinish width limit tag) (tagMask width limit tag)
  else 0

/-- Dot product of a dense histogram and a tag-dependent weight. -/
def weightedCounts (counts : Array Nat) (weight : Nat → Nat) : Nat :=
  ((List.range counts.size).map fun tag => counts[tag]! * weight tag).sum

/-- Local recurrence checked by the certificate, independently of the table-building algorithm. -/
def suffixStep (width limit n start mask : Nat) (heads middle : Array Nat)
    (value : Nat → Nat → Nat → Nat) : Nat :=
  weightedCounts heads (terminalWeight width limit n start mask) +
    weightedCounts middle (fun tag => middleWeight width limit n mask tag value)

/-- Dense state-table lookup, with remaining length as its outermost index. -/
def cacheValue (width : Nat) (table : Array Nat) (n start mask : Nat) : Nat :=
  table[(n * (width + 1) + start) * 2 ^ width + mask]!

/-- Construct a small backwards recurrence table. Its correctness is checked separately. -/
def continuationTable (width limit degree : Nat) (heads : Array Nat)
    (middles : Array (Array Nat)) : Array Nat := Id.run do
  let mut table := Array.replicate ((degree + 1) * (width + 1) * 2 ^ width) 0
  for n in [:degree + 1] do
    for start in [:width + 1] do
      for mask in [:2 ^ width] do
        let value := suffixStep width limit n start mask heads middles[start]!
          (cacheValue width table)
        table := table.set! ((n * (width + 1) + start) * 2 ^ width + mask) value
  return table

end RubiksSnake.CapEnumeration

import RubiksSnakeComputation

/-!
Restoring slab enumeration with an arbitrary pruning predicate. Rejected nodes
leave both the board and the coefficient array unchanged. This computation-only
module is precompiled independently of the mathematical proofs.
-/

namespace RubiksSnake.SlabEnumeration

/-- Depth-first slab enumeration that may discard any node through `keep`.
Accepted branches update the length histogram and restore the board before returning. -/
@[specialize] def prunedCountSearch (width side : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) :
    Nat → Cursor → ByteArray → Array Nat → ByteArray × Array Nat
  | remaining, s, board, counts =>
    if keep remaining s then
      let old := board[s.position]!
      let counts := if canExit width irreducible s old then increment counts s.length else counts
      match remaining with
      | 0 => (board, counts)
      | remaining + 1 =>
        directions.foldl (fun (board, counts) outgoing =>
          if canMove width s old outgoing then
            let (returned, updated) := prunedCountSearch width side irreducible keep remaining
              (advance side s outgoing)
              (board.set! s.position (entry old s.incoming outgoing)) counts
            (returned.set! s.position old, updated)
          else (board, counts)) (board, counts)
    else (board, counts)

/-- Coefficients are indexed by internal edge count, as in `counts`. -/
def prunedCounts (width limit : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) : Array Nat :=
  let side := 2 * limit + 1
  let board := ByteArray.mk (Array.replicate ((width + 1) * side * side) 0)
  let position := limit * side + limit
  (prunedCountSearch width side irreducible keep limit ⟨0, 0, position, 0, 0⟩ board
    (Array.replicate (limit + 1) 0)).2

/--
Cheap remaining-edge budget for an x-slab. In irreducible mode, the scan finds
the lowest missing cut and counts missing cuts at or above the cursor. No
completeness claim is needed: any predicate yields certified lower coefficients.
-/
def slabBudget (width : Nat) (irreducible : Bool) (s : Cursor) : Nat := Id.run do
  let mut lo := width
  let mut upperMissing := 0
  if irreducible then
    for cut in [:width] do
      if !(s.mask.testBit cut) then
        lo := min lo cut
        if s.x ≤ cut then
          upperMissing := upperMissing + 1
  let h := if lo == width then width - s.x
    else (width - s.x) + 2 * ((s.x - lo) + upperMissing)
  return 2 * h + (if s.incoming / 2 == 0 then 1 else 0)

/-- Retain a cursor only when its remaining fuel meets the directly computed slab budget. -/
@[inline] def slabBudgetKeep (width : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) : Bool :=
  decide (slabBudget width irreducible s ≤ remaining)

/-- Cache budgets by x-coordinate, backward-cut mask, and whether the incoming axis is x.
The index is `((x * 2^width + mask) * 2 + xAxisFlag)`. -/
def slabBudgetTable (width : Nat) (irreducible : Bool) : Array Nat :=
  let masks := 2 ^ width
  (Array.range ((width + 1) * masks * 2)).map fun i =>
    slabBudget width irreducible
      ⟨0, i / (2 * masks), 0, if i % 2 == 0 then 2 else 0, (i / 2) % masks⟩

/-- Test the cached budget, rejecting out-of-range coordinates, masks, or table entries. -/
@[inline] def tableBudgetKeep (width masks : Nat) (table : Array Nat)
    (remaining : Nat) (s : Cursor) : Bool :=
  s.x ≤ width && s.mask < masks &&
    match table[((s.x * masks + s.mask) * 2 + if s.incoming / 2 == 0 then 1 else 0)]? with
    | none => false
    | some budget => decide (budget ≤ remaining)

/-- Budget-pruned traversal with cursor fields passed separately to avoid recursive allocations.
The natural arguments are remaining fuel, length, x-coordinate, board index, incoming face,
and backward-cut mask; `PrunedSlabEnumeration` proves agreement with the reference traversal. -/
def budgetSearch (width side : Nat) (irreducible : Bool) (masks : Nat) (table : Array Nat) :
    Nat → Nat → Nat → Nat → Nat → Nat → ByteArray → Array Nat → ByteArray × Array Nat
  | remaining, length, x, position, incoming, mask, board, counts =>
    if tableBudgetKeep width masks table remaining ⟨length, x, position, incoming, mask⟩ then
      let old := board[position]!
      let counts :=
        if canExit width irreducible ⟨length, x, position, incoming, mask⟩ old then
          increment counts length else counts
      match remaining with
      | 0 => (board, counts)
      | remaining + 1 =>
        directions.foldl (fun (board, counts) outgoing =>
          if canMove width ⟨length, x, position, incoming, mask⟩ old outgoing then
            let next := advance side ⟨length, x, position, incoming, mask⟩ outgoing
            let (returned, updated) := budgetSearch width side irreducible masks table remaining
              next.length next.x next.position next.incoming next.mask
              (board.set! position (entry old incoming outgoing)) counts
            (returned.set! position old, updated)
          else (board, counts)) (board, counts)
    else (board, counts)

/-- The budget-pruned coefficients, suitable for small native certificates. -/
def budgetCounts (width limit : Nat) (irreducible : Bool) : Array Nat :=
  let side := 2 * limit + 1
  let board := ByteArray.mk (Array.replicate ((width + 1) * side * side) 0)
  let position := limit * side + limit
  let table := slabBudgetTable width irreducible
  (budgetSearch width side irreducible (2 ^ width) table limit
    0 0 position 0 0 board (Array.replicate (limit + 1) 0)).2

end RubiksSnake.SlabEnumeration

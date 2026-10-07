import Std.Data.HashSet
import Std.Data.HashMap
import Batteries.Data.List.Basic

/-!
# Executable data for prefix certificates

This module contains only the finite computations. Their geometric meaning
and their counting consequences are proved in the upper-bound proof modules.
Keeping the finite check separate avoids loading the real-analysis proof
environment while constructing a large certificate.
-/

namespace RubiksSnake
namespace WindowComputation

/-- Integer triples representing lattice positions and directions in the compact checker. -/
abbrev Coord := Int × Int × Int

/-- Reverses a compact direction, in particular converting incoming to entrance direction. -/
def neg (p : Coord) : Coord := (-p.1, -p.2.1, -p.2.2)

/-- Componentwise addition used to advance a wedge center along its outgoing direction. -/
def add (p q : Coord) : Coord := (p.1 + q.1, p.2.1 + q.2.1, p.2.2 + q.2.2)

/-- Integer cross product supplying the perpendicular direction for a quarter turn. -/
def crossCoord (p q : Coord) : Coord :=
  (p.2.1 * q.2.2 - p.2.2 * q.2.1,
   p.2.2 * q.1 - p.1 * q.2.2,
   p.1 * q.2.1 - p.2.1 * q.1)

/-- Compact quarter-turn rule; for perpendicular unit coordinate directions, rotates
`previous` about `axis` by the number of quarter turns specified by `r`. -/
def turn (axis : Coord) (r : Fin 4) (previous : Coord) : Coord :=
  match r.val with
  | 0 => previous
  | 1 => crossCoord axis previous
  | 2 => neg previous
  | _ => neg (crossCoord axis previous)

/-- A wedge's center and outward entrance/exit directions stored as triples.
The data type itself does not impose geometric frame invariants. -/
structure CompactWedge where
  center : Coord
  entrance : Coord
  exit : Coord
deriving DecidableEq

/-- Compact interior-disjointness test: distinct centers, or opposite unordered
entrance/exit pairs when the centers coincide. -/
def disjoint (a b : CompactWedge) : Prop :=
  a.center ≠ b.center ∨
    (a.entrance = neg b.entrance ∧ a.exit = neg b.exit) ∨
    (a.entrance = neg b.exit ∧ a.exit = neg b.entrance)

/-- Decides compact disjointness using integer-coordinate equalities, enabling
the executable collision checker. -/
instance (a b : CompactWedge) : Decidable (disjoint a b) := by
  unfold disjoint
  infer_instance

/-- Builds compact wedges from a starting center and frame; `rs.length` rotations
produce `rs.length + 1` wedges, including the initial wedge. -/
def path (center incoming outgoing : Coord) : List (Fin 4) → List CompactWedge
  | [] => [⟨center, neg incoming, outgoing⟩]
  | r :: rs =>
      ⟨center, neg incoming, outgoing⟩ ::
        path (add center outgoing) outgoing (turn outgoing r incoming) rs

/-- Canonical compact realization rooted at the origin with incoming direction
`+y` and outgoing direction `+x`; a word of length `n` gives `n + 1` wedges. -/
def compactWedges (rs : List (Fin 4)) : List CompactWedge :=
  path (0, 0, 0) (0, 1, 0) (1, 0, 0) rs

/-- Boolean test that all wedges in the canonical realization are pairwise
interior-disjoint, not merely disjoint from their immediate neighbors. -/
def valid (rs : List (Fin 4)) : Bool :=
  decide ((compactWedges rs).Pairwise disjoint)

/-- All four rotation labels in a fixed order, each occurring once in transition loops. -/
def rotations : List (Fin 4) := [0, 1, 2, 3]

end WindowComputation
namespace PrefixAutomaton

open WindowComputation

/-- Reads a state's integer weight through `index`, using zero for an out-of-range
array index. -/
def arrayWeight (index : List (Fin 4) → Nat) (values : Array Nat)
    (rs : List (Fin 4)) : Nat :=
  values[index rs]?.getD 0

/-- Sums destination weights along a graph row, preserving edge multiplicities;
out-of-range destinations contribute zero. -/
def rowWeight (values : Array Nat) (row : List Nat) : Nat :=
  row.foldr (fun i total => values[i]?.getD 0 + total) 0

/-- Evaluates one weighted adjacency row, treating an absent source row as empty
and absent destination weights as zero. -/
def arrayOutgoing (graph : Array (List Nat)) (values : Array Nat) (i : Nat) : Nat :=
  rowWeight values (graph[i]?.getD [])

/-- Candidate integer potentials starting at the constant `10^18` vector and
iterating `floor(outgoing / 4)`; the final inequalities require separate certification. -/
def scaledWeights (graph : Array (List Nat)) : Nat → Array Nat
  | 0 => Array.replicate graph.size 1000000000000000000
  | k + 1 =>
      let previous := scaledWeights graph k
      graph.map fun row => rowWeight previous row / 4

/-- The all-ones vector and its exact adjacency iterates, retaining terminal
path weights for every horizon from zero through the requested length. -/
def terminalSequence (graph : Array (List Nat)) : Nat → Array (Array Nat)
  | 0 => #[Array.replicate graph.size 1]
  | k + 1 =>
      let previous := terminalSequence graph k
      previous.push (graph.map (rowWeight (previous[k]?.getD #[])))

/-- Manhattan distance to the origin, a lower bound on the number of unit center
steps needed to close a path. -/
private def centerDistance (p : Coord) : Nat :=
  p.1.natAbs + p.2.1.natAbs + p.2.2.natAbs

/-- The canonical first wedge, kept separate from the later wedges during the
search for collisions between the first and last wedges. -/
private def initialWedge : CompactWedge :=
  ⟨(0, 0, 0), (0, -1, 0), (1, 0, 0)⟩

/-- Depth-first search carrying a valid proper prefix and its past wedges.
Records the prefix and number of final turns colliding only with the first wedge;
Manhattan-distance pruning discards branches unable to return to the origin. -/
private def collectCollisionPrefixes {α : Type}
    (record : List (Fin 4) → Nat → α → α) :
    Coord → Coord → Coord → List CompactWedge → List (Fin 4) → Nat → α → α
  | _, _, _, _, _, 0, result => result
  | center, incoming, outgoing, past, reversed, k + 1, result =>
      let next := add center outgoing
      if centerDistance next ≤ k then
        if k = 0 then
          let count := rotations.countP fun r =>
            let wedge : CompactWedge := ⟨next, neg outgoing, turn outgoing r incoming⟩
            !decide (disjoint initialWedge wedge) &&
              past.all (fun previous => decide (disjoint previous wedge))
          if count = 0 then result else record reversed.reverse count result
        else
          rotations.foldl (fun result r =>
            let axis := turn outgoing r incoming
            let wedge : CompactWedge := ⟨next, neg outgoing, axis⟩
            if decide (disjoint initialWedge wedge) &&
                past.all (fun previous => decide (disjoint previous wedge)) then
              collectCollisionPrefixes record next outgoing axis (wedge :: past)
                (r :: reversed) k result
            else result) result
      else result

/-- For each requested rotation length, folds over proper prefixes admitting an
endpoint-only collision, passing each prefix and its number of closing rotations
to `record`. -/
def foldCollisionPrefixes {α : Type} (record : List (Fin 4) → Nat → α → α)
    (lengths : List Nat) (initial : α) : α :=
  lengths.foldl
    (fun result length =>
      collectCollisionPrefixes record (0, 0, 0) (0, 1, 0) (1, 0, 0) [] [] length result)
    initial

/-- Returns the number of recorded collision-ending words and the prefix-closed
dictionary of their proper prefixes, including the empty root. The collision
count and dictionary size count different objects. -/
def collisionPrefixes (lengths : List Nat) : Nat × Std.HashSet (List (Fin 4)) :=
  foldCollisionPrefixes (fun word count result =>
    (result.1 + count,
      word.inits.foldl (fun dictionary rs => dictionary.insert rs) result.2)) lengths
    (0, Std.HashSet.ofList [[]])

end PrefixAutomaton
namespace EncodedPrefixAutomaton

open WindowComputation

/-- Length-preserving base-four key with sentinel `1` for the empty word;
the first rotation is the least significant digit. -/
def encode : List (Fin 4) → Nat
  | [] => 1
  | r :: rs => 4 * encode rs + r.val

/-- Decodes least-significant base-four digits until the remaining code is below
four, which yields the empty word. Arbitrary keys need not round-trip through
`encode`; canonical keys do. -/
def decode (code : Nat) : List (Fin 4) :=
  if h : code < 4 then [] else
    (⟨code % 4, Nat.mod_lt _ (by decide)⟩ : Fin 4) :: decode (code / 4)
termination_by code
decreasing_by
  exact Nat.div_lt_self (Nat.lt_of_lt_of_le (by decide) (Nat.le_of_not_gt h)) (by decide)

/-- Key of the longest suffix whose encoding is in `codes`, with empty-word key
`1` as the fallback even when that key is absent. -/
def longest (codes : Std.HashSet Nat) : List (Fin 4) → Nat
  | [] => 1
  | r :: rs => if codes.contains (encode (r :: rs)) then encode (r :: rs) else longest codes rs

/-- Destination indices for geometrically valid one-rotation extensions of the
decoded source, retaining the longest encoded suffix and one edge per rotation. -/
def successors (codes : Std.HashSet Nat) (index : Nat → Nat) (code : Nat) : List Nat :=
  let rs := decode code
  rotations.filterMap fun r =>
    if valid (rs ++ [r]) then some (index (longest codes (rs ++ [r]))) else none

/-- Indexed transition rows in `states` order; repeated destinations remain
distinct rotation-labeled edges. -/
def graph (codes : Std.HashSet Nat) (states : Array Nat) (index : Nat → Nat) :
    Array (List Nat) :=
  states.map (successors codes index)

/-- Collision-word count and prefix-closed set of sentinel-encoded proper
prefixes, avoiding persistent list-valued states in large finite certificates. -/
def collisionPrefixes (lengths : List Nat) : Nat × Std.HashSet Nat :=
  PrefixAutomaton.foldCollisionPrefixes (fun word count result =>
    (result.1 + count,
      word.inits.foldl (fun codes rs => codes.insert (encode rs)) result.2)) lengths
    (0, Std.HashSet.ofList [1])

end EncodedPrefixAutomaton
end RubiksSnake

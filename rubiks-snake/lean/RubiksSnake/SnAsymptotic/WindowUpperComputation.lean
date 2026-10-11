import RubiksSnake.ComputeSnakes
import RubiksSnake.SnAsymptotic.PrefixAutomatonData

/-!
# Executable geometry for small upper-bound certificates

Coordinates are stored as triples of integers rather than functions. The
conversion to the original wedge geometry is proved exactly, so the faster
checker does not introduce a separate notion of snake validity.
-/

namespace RubiksSnake
namespace WindowComputation

/-- Converts an integer triple into the function-valued vector used by the
original geometric definitions. -/
def vec (p : Coord) : Vec3 := ![p.1, p.2.1, p.2.2]

/-- Conversion to function-valued vectors loses no coordinate information,
allowing equality tests to transfer between representations. -/
lemma vec_injective : Function.Injective vec := by
  rintro ⟨x, y, z⟩ ⟨x', y', z'⟩ h
  have hx := congrFun h 0
  have hy := congrFun h 1
  have hz := congrFun h 2
  simpa [vec, Prod.mk.injEq] using And.intro hx (And.intro hy hz)

/-- Reversing a compact direction agrees exactly with negation in the original geometry. -/
@[simp] lemma vec_neg (p : Coord) : vec (neg p) = negVec (vec p) := by
  funext i
  fin_cases i <;> rfl

/-- Advancing a compact center by a direction agrees with vector addition. -/
@[simp] lemma vec_add (p q : Coord) : vec (add p q) = addVec (vec p) (vec q) := by
  funext i
  fin_cases i <;> rfl

/-- The compact integer cross product agrees with the original vector cross product. -/
@[simp] lemma vec_cross (p q : Coord) :
    vec (crossCoord p q) = cross (vec p) (vec q) := rfl

/-- Every compact turn agrees with the original quarter-turn rule, without
requiring extra hypotheses on the input triples. -/
@[simp] lemma vec_turn (axis : Coord) (r : Rotation) (previous : Coord) :
    vec (turn axis r previous) = rotateQuarter (vec axis) r (vec previous) := by
  fin_cases r <;> simp [turn, rotateQuarter]

/-- Views a compact wedge as an original wedge by converting its center and
entrance/exit directions coordinatewise. -/
def CompactWedge.toWedge (w : CompactWedge) : Wedge :=
  ⟨vec w.center, vec w.entrance, vec w.exit⟩

/-- The compact disjointness predicate is exactly the original interior-disjointness
predicate after conversion, not an approximation to it. -/
lemma disjoint_iff (a b : CompactWedge) :
    disjoint a b ↔ interiorDisjoint a.toWedge b.toWedge := by
  simp only [disjoint, interiorDisjoint, sameUnorderedPair, CompactWedge.toWedge,
    ← vec_neg, ne_eq, vec_injective.eq_iff]

/-- Compact path construction yields the same wedge list as the original frame
recursion, for any starting center, frame, and rotation word. -/
lemma path_map (center incoming outgoing : Coord) (rs : List Rotation) :
    (path center incoming outgoing rs).map CompactWedge.toWedge =
      wedgePath (vec center) (vec incoming) (vec outgoing)
        (directionTail (vec incoming) (vec outgoing) rs) := by
  induction rs generalizing center incoming outgoing with
  | nil => simp [path, wedgePath, directionTail, CompactWedge.toWedge]
  | cons r rs ih =>
      simp [path, wedgePath, directionTail, CompactWedge.toWedge, ih]

/-- The canonical compact realization converts to exactly the original wedge
list, with one more wedge than rotations. -/
lemma compactWedges_map (rs : List Rotation) :
    (compactWedges rs).map CompactWedge.toWedge = wedges rs := by
  rw [compactWedges, path_map]
  have hzero : vec (0, 0, 0) = zeroVec := by
    funext i
    fin_cases i <;> rfl
  rw [hzero]
  change wedgePath zeroVec ey ex (directionTail ey ex rs) = wedges rs
  symm
  unfold wedges directions directionsFrom wedgesFromDirections centersFromDirections
  simp only [List.drop_succ_cons, List.drop_zero, List.tail_cons, List.zip_cons_cons]
  exact wedgesFromDirections_eq_wedgePath _ _ _ _

/-- The executable compact checker accepts exactly the geometrically valid
rotation words, justifying its use in finite certificates. -/
lemma valid_iff (rs : List Rotation) : valid rs = true ↔ ValidList rs := by
  rw [valid, decide_eq_true_eq]
  change _ ↔ (wedges rs).Pairwise interiorDisjoint
  rw [← compactWedges_map, List.pairwise_map]
  simp only [← disjoint_iff]

/-- Compact paths always contain their initial wedge. -/
private lemma path_ne_nil (center incoming outgoing : Coord) (rs : List Rotation) :
    path center incoming outgoing rs ≠ [] := by
  cases rs <;> simp [path]

/-- Dropping a nonterminal head does not change a nonempty list's final element. -/
private lemma getLastD_cons_of_ne_nil {α : Type*} (a fallback : α) (xs : List α)
    (h : xs ≠ []) : (a :: xs).getLastD fallback = xs.getLastD fallback := by
  simp only [List.getLastD_eq_getLast?, List.getLast?_cons_of_ne_nil h]

/-- Appending one rotation adds precisely the wedge constructed from the current terminal frame. -/
lemma path_append_singleton (center incoming outgoing : Coord) (rs : List Rotation)
    (r : Rotation) (fallback : CompactWedge) :
    path center incoming outgoing (rs ++ [r]) =
      path center incoming outgoing rs ++
        [nextWedge ((path center incoming outgoing rs).getLastD fallback) r] := by
  induction rs generalizing center incoming outgoing with
  | nil => simp [path, nextWedge, neg]
  | cons a rs ih =>
    simp only [List.cons_append, path, ih]
    rw [getLastD_cons_of_ne_nil _ _ _ (path_ne_nil _ _ _ _)]

/-- Shared-prefix extension checks are exactly the old whole-path validity test,
including when the source prefix itself is invalid. -/
theorem extensionAllowed_eq_valid (rs : List Rotation) (r : Rotation) :
    extensionAllowed (extensionContext rs) r = valid (rs ++ [r]) := by
  have hp : compactWedges (rs ++ [r]) = compactWedges rs ++
      [nextWedge ((compactWedges rs).getLastD ⟨(0, 0, 0), (0, -1, 0), (1, 0, 0)⟩) r] :=
    path_append_singleton _ _ _ rs r _
  apply Bool.eq_iff_iff.mpr
  simp only [extensionAllowed, extensionContext, Bool.and_eq_true,
    decide_eq_true_eq, List.all_eq_true]
  rw [valid, decide_eq_true_eq, hp, List.pairwise_append]
  simp

/-- Drops the oldest rotation and appends a new one; this preserves a nonempty
window's length and performs no validity check by itself. -/
def next (rs : List Rotation) (r : Rotation) : List Rotation :=
  rs.tail ++ [r]

/-- Sum of successor weights for extensions passing the compact check on
`rs ++ [r]`; the destination retains only the shifted window. -/
def outgoing (w : List Rotation → ℕ) (rs : List Rotation) : ℕ :=
  (rotations.map fun r => if valid (rs ++ [r]) then w (next rs r) else 0).sum

/-- Number of accepted one-rotation window extensions, counting rotation labels
rather than distinct destination states. -/
def degree (rs : List Rotation) : ℕ := outgoing (fun _ => 1) rs

/-- On an already valid word, checking the whole extended compact path is
equivalent to checking the newly appended wedge against the existing wedges. -/
lemma valid_append_eq_canAppend (rs : List Rotation) (r : Rotation)
    (hvalid : ValidList rs) : valid (rs ++ [r]) = canAppend rs r := by
  apply Bool.eq_iff_iff.mpr
  rw [valid_iff]
  change collisionFree (rs ++ [r]) ↔ canAppend rs r
  rw [collisionFree_append_iff]
  exact and_iff_right hvalid

/-- For a valid source word, compact outgoing weights coincide with the
original `canAppend`-based weighted transition sum. -/
lemma outgoing_eq_canAppend (w : List Rotation → ℕ) (rs : List Rotation)
    (hvalid : ValidList rs) :
    outgoing w rs =
      (rotations.map fun r => if canAppend rs r then w (next rs r) else 0).sum := by
  unfold outgoing
  simp_rw [valid_append_eq_canAppend rs _ hvalid]

/-- All valid one-rotation extensions of the entire word, without discarding
older rotations as a window transition would. -/
def children (rs : List Rotation) : List (List Rotation) :=
  rotations.filterMap fun r =>
    if valid (rs ++ [r]) then some (rs ++ [r]) else none

/-- Enumerates globally valid words with the requested number of rotations
by extending the full prefix tree; length `n` corresponds to `n + 1` wedges. -/
def validWords : ℕ → List (List Rotation)
  | 0 => [[]]
  | n + 1 => (validWords n).flatMap children

/-- The compact prefix-tree enumeration is exactly the existing list of valid
`n`-rotation formulas, including its ordering. -/
lemma validWords_eq (n : ℕ) : validWords n = validRotationLists n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [validWords, ih]
      change
        (validRotationLists n).flatMap children =
          (validRotationLists n).flatMap (fun rs => rotations.filterMap fun r =>
            if canAppend rs r then some (rs ++ [r]) else none)
      suffices ∀ xs : List (List Rotation), (∀ rs ∈ xs, ValidList rs) →
          xs.flatMap children =
            xs.flatMap (fun rs => rotations.filterMap fun r =>
              if canAppend rs r then some (rs ++ [r]) else none) from
        this _ (validRotationLists_valid n)
      intro xs
      induction xs with
      | nil => simp
      | cons rs xs ih =>
          intro hvalid
          rw [List.flatMap_cons, List.flatMap_cons]
          congr 1
          · unfold children
            congr 1
            funext r
            rw [valid_append_eq_canAppend rs r (hvalid rs (by simp))]
          · exact ih (fun s hs => hvalid s (by simp [hs]))

/-- Big-endian base-four index for a fixed-width rotation word; unlike sentinel
prefix keys, this encoding does not record the word's length. -/
def encode (rs : List Rotation) : ℕ :=
  rs.foldl (fun i r => 4 * i + r.val) 0

/-- The `width` base-four digits of an in-range index, including leading zeros,
listed in rotation order. -/
def decode (width : ℕ) (i : Fin (4 ^ width)) : List Rotation :=
  List.ofFn fun j : Fin width =>
    (⟨(i.val / 4 ^ (width - 1 - j.val)) % 4, Nat.mod_lt _ (by decide)⟩ : Rotation)

/-- Transition rows for all `4^width` encoded windows, accepting an edge only
when the full one-rotation extension is valid. For positive width, destinations
are indices of shifted windows of the same width. -/
def edges (width : ℕ) : Array (List ℕ) :=
  Array.ofFn fun i : Fin (4 ^ width) =>
    let rs := decode width i
    rotations.filterMap fun r =>
      if valid (rs ++ [r]) then some (encode (next rs r)) else none

/-- Exact adjacency iterates starting from all ones; on a well-indexed graph,
entries count paths of the requested length, not globally valid formulas. -/
def iterateWeights (graph : Array (List ℕ)) : ℕ → Array ℕ
  | 0 => Array.replicate graph.size 1
  | n + 1 =>
      let previous := iterateWeights graph n
      graph.map fun es => (es.map fun i => previous[i]?.getD 0).sum

end WindowComputation
end RubiksSnake

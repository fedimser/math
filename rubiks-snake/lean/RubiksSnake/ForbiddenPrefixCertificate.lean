import RubiksSnake.PrefixAutomaton

/-!
# A compact integer upper certificate

Closed center walks through twelve steps generate short collision factors.
Their proper prefixes give 46599 dictionary states, compared with 2333757
states in the corresponding full window graph. All transitions are checked
against the original geometry through the proved-equivalent compact checker.
-/

namespace RubiksSnake
namespace ForbiddenPrefixUpper

open WindowComputation

/-- Manhattan distance to the origin, used to prune center walks that cannot
close in the remaining unit steps. -/
private def distance (p : Coord) : ℕ :=
  p.1.natAbs + p.2.1.natAbs + p.2.2.natAbs

/-- From a unit coordinate frame, enumerates words of the requested remaining
rotation length whose center walk ends at the origin, pruning unreachable
branches but not testing collisions. -/
private def closedWords : Coord → Coord → Coord → ℕ → List (List Rotation)
  | center, _, _, 0 => if center = (0, 0, 0) then [[]] else []
  | center, previous, axis, k + 1 =>
      let next := add center axis
      if distance next ≤ k then
        rotations.flatMap fun r =>
          (closedWords next axis (turn axis r previous) k).map (r :: ·)
      else []

/-- Closed center-walk words of even rotation lengths four through twelve that
are invalid, but become valid after deleting either the first or last rotation;
these supply the short forbidden factors. -/
def forbiddenWords : List (List Rotation) :=
  (([4, 6, 8, 10, 12] : List ℕ).flatMap
    (closedWords (0, 0, 0) (0, 1, 0) (1, 0, 0))).filter fun rs =>
      !valid rs && valid rs.dropLast && valid rs.tail

/-- The empty root and all proper prefixes of the generated forbidden words,
used as variable-length suffix states rather than full formula histories. -/
def dictionary : PrefixAutomaton.Dictionary :=
  Std.HashSet.ofList ([] :: forbiddenWords.flatMap (fun rs => rs.dropLast.inits))

/-- Enumeration of dictionary words for graph indexing, not an enumeration of
valid formulas at a single fixed length. -/
def states : List (List Rotation) := dictionary.toList

/-- Assigns each dictionary word its position in the state enumeration for
array-backed transition and potential lookups. -/
private def indices : Std.HashMap (List Rotation) ℕ :=
  Std.HashMap.ofList states.zipIdx

/-- Locally checked transition rows in dictionary-state order, with one edge
per accepted rotation leading to the longest retained dictionary suffix. -/
private def graph : Array (List ℕ) :=
  states.toArray.map fun rs =>
    rotations.filterMap fun r =>
      if valid (rs ++ [r]) then
        some (indices.getD (PrefixAutomaton.longest dictionary (rs ++ [r])) 0)
      else none

/-- Candidate potential from twenty unscaled adjacency iterations starting at
all ones; its terminal and growth inequalities are certified separately. -/
private def potential : Array ℕ := iterateWeights graph 20

/-- Reads the potential at a word's dictionary index, defaulting to index zero
for absent words and to weight zero for an out-of-range array access. -/
def weight (rs : List Rotation) : ℕ :=
  potential[indices.getD rs 0]?.getD 0

/-- Exact local graph-path count vectors for terminal horizons zero through four. -/
private def terminalVectors : Array (Array ℕ) :=
  #[iterateWeights graph 0, iterateWeights graph 1, iterateWeights graph 2,
    iterateWeights graph 3, iterateWeights graph 4]

/-- Array-backed terminal path weight at a word's dictionary index; horizons
beyond the stored zero-to-four range have weight zero. -/
def terminalWeight (t : ℕ) (rs : List Rotation) : ℕ :=
  ((terminalVectors[t]?).getD #[])[indices.getD rs 0]?.getD 0

/-- The empty word belongs to the dictionary, supplying the automaton's root
and its fallback suffix state. -/
lemma dictionary_zero : [] ∈ dictionary := by
  simp [dictionary]

/-- Native finite verification of prefix closure, graph metadata, four-step
terminal domination, the exact integer row inequality with ratio `147 / 40`,
and the root potential. No spectral-radius estimate is used. -/
private lemma checked :
    (∀ rs ∈ states, dictionary.contains rs.dropLast = true) ∧
    (forbiddenWords.length = 44736 ∧ states.length = 46599 ∧
      (graph.toList.map List.length).sum = 136848) ∧
    (∀ rs ∈ states, valid rs →
      1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 4, PrefixAutomaton.outgoing dictionary (terminalWeight t.val) rs ≤
        terminalWeight (t.val + 1) rs) ∧
      361538007 * terminalWeight 4 rs ≤ weight rs ∧
      40 * PrefixAutomaton.outgoing dictionary weight rs ≤ 147 * weight rs) ∧
    weight [] = 284786048601 := by
  native_decide

/-- Deleting the final rotation preserves dictionary membership, establishing
the prefix closure required by the suffix-state update. -/
lemma dictionary_closed (rs : List Rotation) (h : rs ∈ dictionary) :
    rs.dropLast ∈ dictionary := by
  exact (Std.HashSet.contains_iff_mem).mp
    (checked.1 rs ((Std.HashSet.mem_toList).mpr h))

/-- Certified counts of 44736 generated forbidden words, 46599 variable-length
prefix states, and 136848 rotation-labeled edges; these are distinct objects,
not a count of valid twelve-rotation formulas. -/
lemma graph_counts :
    forbiddenWords.length = 44736 ∧ states.length = 46599 ∧
      (graph.toList.map List.length).sum = 136848 :=
  checked.2.1

/-- On each valid dictionary state, exact inequalities dominate four terminal
steps at scale `361538007` and bound outgoing potential with ratio `147 / 40`;
zero potentials at dead ends are allowed. -/
lemma certificate (rs : List Rotation) (h : rs ∈ dictionary) (hv : ValidList rs) :
    1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 4, PrefixAutomaton.outgoing dictionary (terminalWeight t.val) rs ≤
        terminalWeight (t.val + 1) rs) ∧
      361538007 * terminalWeight 4 rs ≤ weight rs ∧
      40 * PrefixAutomaton.outgoing dictionary weight rs ≤ 147 * weight rs :=
  checked.2.2.1 rs ((Std.HashSet.mem_toList).mpr h) ((valid_iff rs).mpr hv)

/-- Exact empty-root potential used to control the pointwise prefactor, not
a count of formulas or of dictionary states. -/
lemma initialWeight_value : weight [] = 284786048601 := checked.2.2.2

end ForbiddenPrefixUpper
end RubiksSnake

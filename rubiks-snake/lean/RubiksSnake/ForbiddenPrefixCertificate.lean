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

private def distance (p : Coord) : ℕ :=
  p.1.natAbs + p.2.1.natAbs + p.2.2.natAbs

private def closedWords : Coord → Coord → Coord → ℕ → List (List Rotation)
  | center, _, _, 0 => if center = (0, 0, 0) then [[]] else []
  | center, previous, axis, k + 1 =>
      let next := add center axis
      if distance next ≤ k then
        rotations.flatMap fun r =>
          (closedWords next axis (turn axis r previous) k).map (r :: ·)
      else []

def forbiddenWords : List (List Rotation) :=
  (([4, 6, 8, 10, 12] : List ℕ).flatMap
    (closedWords (0, 0, 0) (0, 1, 0) (1, 0, 0))).filter fun rs =>
      !valid rs && valid rs.dropLast && valid rs.tail

def dictionary : PrefixAutomaton.Dictionary :=
  Std.HashSet.ofList ([] :: forbiddenWords.flatMap (fun rs => rs.dropLast.inits))

def states : List (List Rotation) := dictionary.toList

private def indices : Std.HashMap (List Rotation) ℕ :=
  Std.HashMap.ofList states.zipIdx

private def graph : Array (List ℕ) :=
  states.toArray.map fun rs =>
    rotations.filterMap fun r =>
      if valid (rs ++ [r]) then
        some (indices.getD (PrefixAutomaton.longest dictionary (rs ++ [r])) 0)
      else none

private def potential : Array ℕ := iterateWeights graph 20

def weight (rs : List Rotation) : ℕ :=
  potential[indices.getD rs 0]?.getD 0

private def terminalVectors : Array (Array ℕ) :=
  #[iterateWeights graph 0, iterateWeights graph 1, iterateWeights graph 2,
    iterateWeights graph 3, iterateWeights graph 4]

def terminalWeight (t : ℕ) (rs : List Rotation) : ℕ :=
  ((terminalVectors[t]?).getD #[])[indices.getD rs 0]?.getD 0

lemma dictionary_zero : [] ∈ dictionary := by
  simp [dictionary]

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

lemma dictionary_closed (rs : List Rotation) (h : rs ∈ dictionary) :
    rs.dropLast ∈ dictionary := by
  exact (Std.HashSet.contains_iff_mem).mp
    (checked.1 rs ((Std.HashSet.mem_toList).mpr h))

lemma graph_counts :
    forbiddenWords.length = 44736 ∧ states.length = 46599 ∧
      (graph.toList.map List.length).sum = 136848 :=
  checked.2.1

lemma certificate (rs : List Rotation) (h : rs ∈ dictionary) (hv : ValidList rs) :
    1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 4, PrefixAutomaton.outgoing dictionary (terminalWeight t.val) rs ≤
        terminalWeight (t.val + 1) rs) ∧
      361538007 * terminalWeight 4 rs ≤ weight rs ∧
      40 * PrefixAutomaton.outgoing dictionary weight rs ≤ 147 * weight rs :=
  checked.2.2.1 rs ((Std.HashSet.mem_toList).mpr h) ((valid_iff rs).mpr hv)

lemma initialWeight_value : weight [] = 284786048601 := checked.2.2.2

end ForbiddenPrefixUpper
end RubiksSnake

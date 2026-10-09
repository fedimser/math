import RubiksSnake.SnAsymptotic.PrefixAutomaton

/-!
# Indexed computation for prefix certificates

The graph uses the existing geometric transition checker. Verifying that each
dictionary word is found at its assigned index then identifies every array
potential with the semantic outgoing operator, without repeating geometry for
each potential.
-/

namespace RubiksSnake
namespace PrefixAutomaton

open WindowComputation

/-- Indices of longest-suffix destinations for locally valid one-rotation
extensions, preserving one edge for each accepted rotation label. -/
def indexedSuccessors (dictionary : Dictionary) (index : List Rotation → ℕ)
    (rs : List Rotation) : List ℕ :=
  rotations.filterMap fun r =>
    if valid (rs ++ [r]) then some (index (longest dictionary (rs ++ [r]))) else none

/-- Materializes the locally checked transition rows in the supplied state
order; correctness of the supplied indices is verified separately. -/
def indexedGraph (dictionary : Dictionary) (states : Array (List Rotation))
    (index : List Rotation → ℕ) : Array (List ℕ) :=
  states.map (indexedSuccessors dictionary index)

/-- Summing an indexed successor row gives exactly the semantic outgoing sum
for the corresponding array-backed weight, including repeated destinations. -/
lemma rowWeight_indexedSuccessors (dictionary : Dictionary)
    (index : List Rotation → ℕ) (values : Array ℕ) (rs : List Rotation) :
    rowWeight values (indexedSuccessors dictionary index rs) =
      outgoing dictionary (arrayWeight index values) rs := by
  unfold rowWeight indexedSuccessors outgoing arrayWeight
  generalize rotations = xs
  induction xs with
  | nil => simp
  | cons r xs ih =>
      by_cases hr : valid (rs ++ [r]) <;> simp [hr, ih]

/-- When the source word occurs at its assigned index, its graph-row weight
equals the semantic prefix-automaton outgoing weight for any potential array. -/
theorem arrayOutgoing_eq (dictionary : Dictionary) (states : Array (List Rotation))
    (index : List Rotation → ℕ) (values : Array ℕ) (rs : List Rotation)
    (hindex : states[index rs]? = some rs) :
    arrayOutgoing (indexedGraph dictionary states index) values (index rs) =
      outgoing dictionary (arrayWeight index values) rs := by
  unfold arrayOutgoing indexedGraph
  rw [Array.getElem?_map, hindex]
  simpa using rowWeight_indexedSuccessors dictionary index values rs

end PrefixAutomaton
end RubiksSnake

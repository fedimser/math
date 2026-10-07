import RubiksSnake.PrefixAutomatonComputation

/-!
# A length-fourteen collision-prefix certificate

Closed paths with valid proper factors generate 452552 prefix states and
1350436 labeled edges. The graph is built with the existing geometric checker;
the indexed transition theorem transfers all array checks to that semantics.

The potential starts at `10^18` and applies `w := floor(T w / 4)` 48 times.
The final integer inequalities, not the rounding procedure, certify the bound.
Five terminal steps cover transient states whose potential is zero.
-/

namespace RubiksSnake
namespace ForbiddenPrefixFourteenUpper

open WindowComputation

/-- Generates the collision-word count and proper-prefix dictionary using
even rotation lengths four through fourteen. -/
private def generated : ℕ × PrefixAutomaton.Dictionary :=
  PrefixAutomaton.collisionPrefixes [4, 6, 8, 10, 12, 14]

/-- Variable-length suffix states built from generated collision prefixes,
not the set of all valid fourteen-rotation formulas. -/
def dictionary : PrefixAutomaton.Dictionary := generated.2

/-- Enumeration of dictionary words fixing the row order of the finite graph. -/
def states : List (List Rotation) := dictionary.toList

/-- Maps each dictionary word to its position in the state enumeration. -/
private def indices : Std.HashMap (List Rotation) ℕ :=
  Std.HashMap.ofList states.zipIdx

/-- Looks up a word's state position, using zero when absent; the finite
certificate verifies correct lookup for every dictionary state. -/
private def index (rs : List Rotation) : ℕ := indices.getD rs 0

/-- Locally valid, rotation-labeled transitions in dictionary-state order,
retaining only the longest dictionary suffix after each extension. -/
private def graph : Array (List ℕ) :=
  PrefixAutomaton.indexedGraph dictionary states.toArray index

/-- Candidate potential from 48 iterations of `floor(outgoing / 4)` starting
at `10^18`; its validity comes from the later exact integer inequalities. -/
private def potential : Array ℕ := PrefixAutomaton.scaledWeights graph 48

/-- Exact graph-path count vectors for terminal horizons zero through five,
including the initial all-ones vector. -/
private def terminals : Array (Array ℕ) := PrefixAutomaton.terminalSequence graph 5

/-- Word-indexed view of the candidate potential array, used by the semantic
prefix-automaton outgoing inequality. -/
def weight : List Rotation → ℕ := PrefixAutomaton.arrayWeight index potential

/-- Local terminal path weight at horizon `t`, read through the dictionary
index; horizons beyond five yield zero. -/
def terminalWeight (t : ℕ) : List Rotation → ℕ :=
  PrefixAutomaton.arrayWeight index (terminals[t]?.getD #[])

/-- Native verification of the empty root, graph counts, prefix closure, index
correctness, five-step terminal domination, and the exact integer outgoing
inequality with ratio `3667542939 / 1000000000`, together with the root weight. -/
private lemma checked :
    dictionary.contains [] = true ∧
    (generated.1 = 419104 ∧ states.length = 452552 ∧
      (graph.toList.map List.length).sum = 1350436) ∧
    (∀ rs ∈ states,
      dictionary.contains rs.dropLast = true ∧
      states.toArray[index rs]? = some rs ∧
      1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 5,
        PrefixAutomaton.arrayOutgoing graph (terminals[t.val]?.getD #[]) (index rs) ≤
          terminalWeight (t.val + 1) rs) ∧
      11500000000000 * terminalWeight 5 rs ≤ weight rs ∧
      1000000000 * PrefixAutomaton.arrayOutgoing graph potential (index rs) ≤
        3667542939 * weight rs) ∧
    weight [] = 22467483660955156 := by
  native_decide

/-- The generated dictionary contains the empty word needed as the initial
and fallback suffix state. -/
lemma dictionary_zero : [] ∈ dictionary :=
  (Std.HashSet.contains_iff_mem).mp checked.1

/-- The checked deletion property makes the dictionary prefix-closed, as
required to update its longest-suffix state from one rotation to the next. -/
lemma dictionary_closed (rs : List Rotation) (h : rs ∈ dictionary) :
    rs.dropLast ∈ dictionary :=
  (Std.HashSet.contains_iff_mem).mp
    (checked.2.2.1 rs ((Std.HashSet.mem_toList).mpr h)).1

/-- Exact counts of 419104 generated collision words, 452552 variable-length
prefix states, and 1350436 labeled edges, rather than fixed-length formula counts. -/
lemma graph_counts :
    generated.1 = 419104 ∧ states.length = 452552 ∧
      (graph.toList.map List.length).sum = 1350436 :=
  checked.2.1

/-- Transfers the array checks to a valid dictionary state: five terminal steps
are dominated at scale `11500000000000`, and the potential satisfies the exact
integer outgoing bound with ratio `3667542939 / 1000000000`. -/
lemma certificate (rs : List Rotation) (h : rs ∈ dictionary) (_hv : ValidList rs) :
    1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 5, PrefixAutomaton.outgoing dictionary (terminalWeight t.val) rs ≤
        terminalWeight (t.val + 1) rs) ∧
      11500000000000 * terminalWeight 5 rs ≤ weight rs ∧
      1000000000 * PrefixAutomaton.outgoing dictionary weight rs ≤ 3667542939 * weight rs := by
  obtain ⟨_, hindex, hbase, hstep, hterminal, hweight⟩ :=
    checked.2.2.1 rs ((Std.HashSet.mem_toList).mpr h)
  refine ⟨hbase, ?_, hterminal, ?_⟩
  · intro t
    change PrefixAutomaton.outgoing dictionary
      (PrefixAutomaton.arrayWeight index (terminals[t.val]?.getD #[])) rs ≤ _
    rw [← PrefixAutomaton.arrayOutgoing_eq dictionary states.toArray index _ rs hindex]
    exact hstep t
  · change 1000000000 * PrefixAutomaton.outgoing dictionary
      (PrefixAutomaton.arrayWeight index potential) rs ≤ _
    rw [← PrefixAutomaton.arrayOutgoing_eq dictionary states.toArray index _ rs hindex]
    exact hweight

/-- Certified empty-root potential supplying the initial factor in the weighted
counting bound; this is a potential value, not a formula count. -/
lemma initialWeight_value : weight [] = 22467483660955156 := checked.2.2.2

end ForbiddenPrefixFourteenUpper
end RubiksSnake

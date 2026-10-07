import RubiksSnake.EncodedPrefixAutomaton
import RubiksSnake.ForbiddenPrefixSixteenComputation

/-!
# A length-sixteen collision-prefix certificate

The generated prefix graph has 4748260 states and 14133721 labeled edges.
Integer prefix keys avoid allocating millions of persistent rotation lists.
The proved decoding bridge preserves the existing geometric transitions.
A rescaled integer potential uses six terminal steps for dead ends.
-/

namespace RubiksSnake
namespace ForbiddenPrefixSixteenUpper

/-- Semantic list-valued suffix dictionary decoded from the integer certificate;
the finite computation itself stores only encoded keys. -/
def dictionary : PrefixAutomaton.Dictionary := EncodedPrefixAutomaton.dictionary codes

/-- Every stored key has canonical sentinel encoding, so decoding and re-encoding
recovers it, permitting the exact dictionary-membership transfer. -/
private lemma codes_roundtrip (code : ℕ) (h : code ∈ codes) :
    EncodedPrefixAutomaton.encode (EncodedPrefixAutomaton.decode code) = code :=
  (checked.2.2.1 code ((Std.HashSet.mem_toList).mpr h)).1

/-- The decoded dictionary contains the empty rotation word, represented by
sentinel key `1` in the finite check. -/
lemma dictionary_zero : [] ∈ dictionary := by
  apply (EncodedPrefixAutomaton.mem_dictionary_iff codes codes_roundtrip []).mpr
  exact (Std.HashSet.contains_iff_mem).mp checked.1

/-- Deleting the last rotation preserves decoded dictionary membership,
transferring checked encoded prefix closure to the semantic automaton. -/
lemma dictionary_closed (rs : List Rotation) (h : rs ∈ dictionary) :
    rs.dropLast ∈ dictionary := by
  have hcode := (EncodedPrefixAutomaton.mem_dictionary_iff codes codes_roundtrip rs).mp h
  have hdrop :=
    (checked.2.2.1 (EncodedPrefixAutomaton.encode rs)
      ((Std.HashSet.mem_toList).mpr hcode)).2.1
  apply (EncodedPrefixAutomaton.mem_dictionary_iff codes codes_roundtrip rs.dropLast).mpr
  exact (Std.HashSet.contains_iff_mem).mp (by simpa using hdrop)

/-- Certified counts of 4441614 generated collision words, 4748260 prefix
states, and 14133721 rotation-labeled edges, not a count of all valid
sixteen-rotation formulas. -/
lemma graph_counts :
    generated.1 = 4441614 ∧ states.length = 4748260 ∧
      graph.foldl (fun total row => total + row.length) 0 = 14133721 :=
  checked.2.1

/-- On valid dictionary states, six-step terminal weights are dominated at scale
`2990000000000` and the potential satisfies the exact integer outgoing bound
with ratio `3661786723 / 1000000000`; no numerical spectral estimate is assumed. -/
lemma certificate (rs : List Rotation) (h : rs ∈ dictionary) (_hv : ValidList rs) :
    1 ≤ terminalWeight 0 rs ∧
      (∀ t : Fin 6, PrefixAutomaton.outgoing dictionary (terminalWeight t.val) rs ≤
        terminalWeight (t.val + 1) rs) ∧
      2990000000000 * terminalWeight 6 rs ≤ weight rs ∧
      1000000000 * PrefixAutomaton.outgoing dictionary weight rs ≤ 3661786723 * weight rs := by
  exact EncodedPrefixAutomaton.certificate_of_checks codes codes_roundtrip stateArray index
    potential terminalVector 3661786723 1000000000 2990000000000 6
    (fun code hcode =>
      (checked.2.2.1 code ((Std.HashSet.mem_toList).mpr hcode)).2.2) rs h

/-- Exact empty-root potential used with the terminal scale to establish the
pointwise prefactor, rather than a count of formulas or states. -/
lemma initialWeight_value : weight [] = 21309021059784288 := checked.2.2.2

end ForbiddenPrefixSixteenUpper
end RubiksSnake

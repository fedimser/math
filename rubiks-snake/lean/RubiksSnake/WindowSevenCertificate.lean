import RubiksSnake.WindowUpperComputation

/-!
# The seven-symbol integer certificate

The graph has 12585 valid states and 46471 edges. Twelve sparse
adjacency-vector products suffice for the exact base `463/125 = 3.704`.
Two terminal steps, rather than one, account for states which have a successor
but no two-step continuation. All checks use the proved-equivalent compact
geometry; no floating-point values or external tables are used.
-/

namespace RubiksSnake
namespace WindowSevenUpper

open WindowComputation

/-- Seven-rotation window transition graph; all encoded windows have rows,
with edges only for geometrically valid one-rotation extensions. -/
private def graph : Array (List ℕ) := edges 7

/-- Candidate potential given by twelve exact adjacency iterations from all
ones, counting local graph paths rather than full snake extensions. -/
private def potential : Array ℕ := iterateWeights graph 12

/-- Looks up the seven-window potential at the word's base-four index,
returning zero when the index is out of range. -/
def weight (rs : List Rotation) : ℕ :=
  potential[encode rs]?.getD 0

/-- Exact finite counts of 12585 valid seven-rotation window states and 46471
rotation-labeled edges; only the local window graph is enumerated. -/
theorem graph_counts :
    (validWords 7).length = 12585 ∧ (graph.toList.map List.length).sum = 46471 := by
  native_decide

/-- Native proof that every valid seven-rotation formula satisfies terminal
two-step domination at scale `242294` and the exact integer outgoing
inequality `125 * outgoing weight <= 463 * weight`. -/
private theorem finite_certificate :
    ∀ f : Formula 7, valid (List.ofFn f) →
      242294 * outgoing degree (List.ofFn f) ≤ weight (List.ofFn f) ∧
      125 * outgoing weight (List.ofFn f) ≤ 463 * weight (List.ofFn f) := by
  native_decide

/-- Transfers the finite check to any valid list of seven rotations: its
two-step path count is dominated by the potential, whose outgoing weights
satisfy the exact rational bound with ratio `463 / 125`. -/
lemma certificate (rs : List Rotation) (hlen : rs.length = 7)
    (hvalid : ValidList rs) :
    242294 * outgoing degree rs ≤ weight rs ∧
      125 * outgoing weight rs ≤ 463 * weight rs := by
  have hvalid' : valid (List.ofFn (formulaOfList rs hlen)) := by
    rw [ofFn_formulaOfList]
    exact (valid_iff rs).mpr hvalid
  simpa only [ofFn_formulaOfList] using
    finite_certificate (formulaOfList rs hlen) hvalid'

/-- Total initial potential over all valid seven-rotation window states,
providing the starting value for the weighted counting argument. -/
def initialWeight : ℕ := ((validWords 7).map weight).sum

/-- Certified exact sum of initial window potentials, not the number of
seven-rotation states or formulas. -/
lemma initialWeight_value : initialWeight = 83514793291 := by
  native_decide

end WindowSevenUpper
end RubiksSnake

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

private def graph : Array (List ℕ) := edges 7

private def potential : Array ℕ := iterateWeights graph 12

def weight (rs : List Rotation) : ℕ :=
  potential[encode rs]?.getD 0

theorem graph_counts :
    (validWords 7).length = 12585 ∧ (graph.toList.map List.length).sum = 46471 := by
  native_decide

private theorem finite_certificate :
    ∀ f : Formula 7, valid (List.ofFn f) →
      242294 * outgoing degree (List.ofFn f) ≤ weight (List.ofFn f) ∧
      125 * outgoing weight (List.ofFn f) ≤ 463 * weight (List.ofFn f) := by
  native_decide

lemma certificate (rs : List Rotation) (hlen : rs.length = 7)
    (hvalid : ValidList rs) :
    242294 * outgoing degree rs ≤ weight rs ∧
      125 * outgoing weight rs ≤ 463 * weight rs := by
  have hvalid' : valid (List.ofFn (formulaOfList rs hlen)) := by
    rw [ofFn_formulaOfList]
    exact (valid_iff rs).mpr hvalid
  simpa only [ofFn_formulaOfList] using
    finite_certificate (formulaOfList rs hlen) hvalid'

def initialWeight : ℕ := ((validWords 7).map weight).sum

lemma initialWeight_value : initialWeight = 83514793291 := by
  native_decide

end WindowSevenUpper
end RubiksSnake

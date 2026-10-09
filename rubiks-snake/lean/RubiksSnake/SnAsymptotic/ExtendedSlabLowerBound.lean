import RubiksSnake.SnAsymptotic.BridgeCode
import RubiksSnake.SnAsymptotic.PrunedSlabEnumeration
import RubiksSnake.SnAsymptotic.ExtendedSlabCertificate

/-! The geometric renewal bound accepts certified undercounts from the pruned search. -/

namespace RubiksSnake.ExtendedSlab

open SlabEnumeration

attribute [local irreducible] counts budgetCounts blockWords

/-- The bridge code using widths 0, 1, 2, and 3 with internal-edge cutoffs 28, 23, 22, and 23. -/
def code : BridgeCode.Code :=
  BridgeCode.ofSlabs [(0, 28), (1, 23), (2, 22), (3, 23)] (by decide)

/-- Verified pruned rows undercount the actual blocks at each positive length in this code. -/
lemma coefficient_le (hrows : RowsVerified) (n : Nat) :
    retainedCount (n + 1) ≤ (BridgeCode.blocks code (n + 1)).length := by
  rw [code, BridgeCode.ofSlabs_blocks_length]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  have htwo := budgetCounts_getD_le_counts 1 23 true n
  have hthree := budgetCounts_getD_le_counts 2 22 true n
  have hfour := budgetCounts_getD_le_counts 3 23 true n
  rw [hrows.1] at htwo
  rw [hrows.2.1] at hthree
  rw [hrows.2.2] at hfour
  rw [SlabEnumeration.rows_checked.1]
  simpa only [retainedCount, Nat.add_one_ne_zero, if_false, Nat.add_sub_cancel,
    Nat.add_assoc, Nat.add_zero] using
    Nat.add_le_add (Nat.add_le_add (Nat.add_le_add
      (le_refl (SlabEnumeration.rowOne[n]?.getD 0)) htwo) hthree) hfour

/-- Conditional lower bound 3.429771044: the proposed rows must first satisfy `RowsVerified`. -/
theorem growthConstant_lower_bound_of_rows (hrows : RowsVerified) :
    (3429771044 / 1000000000 : ℝ) ≤ snakeGrowthConstant := by
  apply BridgeCode.le_growthConstant code 29 (by decide)
    (fun j => retainedCount (j.val + 1)) (fun j => coefficient_le hrows j.val)
    ?_ ?_ (3429771044 / 1000000000) (by norm_num) (by norm_num) polynomial
  · exact (by decide : 1 ≤ retainedCount 2).trans (coefficient_le hrows 1)
  · exact (by decide : 1 ≤ retainedCount 3).trans (coefficient_le hrows 2)

end RubiksSnake.ExtendedSlab

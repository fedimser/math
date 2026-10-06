import RubiksSnake.RecordBlocks

/-! The exact occupied-interface potential, separated from its counting proof. -/

namespace RubiksSnake.RecordWeighting

def coefficient (j : ℕ) : ℕ := 3193 ^ (7 - j) * 1000 ^ j

def weight (s : RecordState) : ℕ := recordWeightedCount s coefficient

def tables : Fin 4 → List (RecordDatum × ℕ) :=
  ![(recordTables 0).map (fun d => (d, weight d.next)),
    (recordTables 1).map (fun d => (d, weight d.next)),
    (recordTables 2).map (fun d => (d, weight d.next)),
    (recordTables 3).map (fun d => (d, weight d.next))]

def image (s : RecordState) : ℕ :=
  (((tables s.previous).filter (fun d => recordFits s d.1)).map
    (fun d => coefficient d.1.word.length * d.2)).sum

/-- Raw block totals bound the potential without unfolding the geometric language. -/
def maximumWeight : ℕ :=
  4 * coefficient 2 + 8 * coefficient 3 + 16 * coefficient 4 +
    24 * coefficient 5 + 80 * coefficient 6 + 296 * coefficient 7

private theorem checked :
    0 < maximumWeight ∧
    ∀ s ∈ recordStates, let w := weight s
      0 < w ∧ w ≤ maximumWeight ∧ 3193 ^ 7 * w ≤ image s := by
  native_decide

lemma maximumWeight_pos : 0 < maximumWeight := checked.1

lemma weight_checked (s : RecordState) (hs : s ∈ recordStates) :
    0 < weight s ∧ weight s ≤ maximumWeight ∧ 3193 ^ 7 * weight s ≤ image s :=
  checked.2 s hs

end RubiksSnake.RecordWeighting

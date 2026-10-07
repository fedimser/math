import RubiksSnake.RecordBlocks

/-! The exact occupied-interface potential, separated from its counting proof. -/

namespace RubiksSnake.RecordWeighting

/-- Integer length factor at rate `3.193`: for the used lengths `2 <= j <= 7`,
this is `3193^7` times the reciprocal of `3.193^j`. -/
def coefficient (j : ℕ) : ℕ := 3193 ^ (7 - j) * 1000 ^ j

/-- State potential obtained by summing the length factors over all record
blocks that fit the state's occupied interface. -/
def weight (s : RecordState) : ℕ := recordWeightedCount s coefficient

/-- Phase tables pairing each block datum with its successor state's potential. -/
def tables : Fin 4 → List (RecordDatum × ℕ) :=
  ![(recordTables 0).map (fun d => (d, weight d.next)),
    (recordTables 1).map (fun d => (d, weight d.next)),
    (recordTables 2).map (fun d => (d, weight d.next)),
    (recordTables 3).map (fun d => (d, weight d.next))]

/-- Apply one more weighted transition to the potential, summing each fitting
block's length factor times its successor potential. -/
def image (s : RecordState) : ℕ :=
  (((tables s.previous).filter (fun d => recordFits s d.1)).map
    (fun d => coefficient d.1.word.length * d.2)).sum

/-- Raw block totals bound the potential without unfolding the geometric language. -/
def maximumWeight : ℕ :=
  4 * coefficient 2 + 8 * coefficient 3 + 16 * coefficient 4 +
    24 * coefficient 5 + 80 * coefficient 6 + 296 * coefficient 7

/-- Native certificate that all listed-state potentials are positive, bounded
by `maximumWeight`, and satisfy the subsolution inequality at rate `3.193`. -/
private theorem checked :
    0 < maximumWeight ∧
    ∀ s ∈ recordStates, let w := weight s
      0 < w ∧ w ≤ maximumWeight ∧ 3193 ^ 7 * w ≤ image s := by
  native_decide

/-- The uniform potential bound is strictly positive, so it can normalize
an exponential counting bound. -/
lemma maximumWeight_pos : 0 < maximumWeight := checked.1

/-- A listed state has a positive bounded potential, and its weighted image
is at least `3193^7` times that potential. -/
lemma weight_checked (s : RecordState) (hs : s ∈ recordStates) :
    0 < weight s ∧ weight s ≤ maximumWeight ∧ 3193 ^ 7 * weight s ≤ image s :=
  checked.2 s hs

end RubiksSnake.RecordWeighting

import RubiksSnake.CapRecurrence

/-! Small, independently checked recurrence tables connect the native assembled coefficients
to the semantic duplicate-free bridge family. -/

set_option Elab.async false

namespace RubiksSnake.CapAssembly

open CapEnumeration

/-- Continuation tables for longitudinal widths two, three and four. -/
def continuationTables : Array (Array Nat) :=
  (Array.range 3).map fun i =>
    let width := i + 2
    continuationTable width 14 64 (headCounts width 2 14)
      ((Array.range (width + 1)).map fun start => middleCounts width 2 14 start)

/-- Share each small-piece table across all local checks rather than recounting it per row. -/
def certificateCheck : Bool :=
  (List.finRange 3).all fun i =>
    let width := i.val + 2
    let heads := headCounts width 2 14
    let middles := (Array.range (width + 1)).map fun start => middleCounts width 2 14 start
    let value := cacheValue width continuationTables[i.val]!
    let assembled := assembledCounts width 2 14 64
    (List.range 65).all (fun n =>
      (List.range (width + 1)).all fun start =>
        (List.range (2 ^ width)).all fun mask =>
          decide (value n start mask ≤
            suffixStep width 14 n start mask heads middles[start]! value)) &&
    (List.range 65).all (fun n =>
      decide (assembled[n]! ≤ weightedCounts heads (fun tag => headWeight width 14 n tag value)))

/-- Exact native verification of the finite local inequalities. -/
theorem certificate_checked : certificateCheck = true := by
  native_decide

/-- Every table row is a recurrence subsolution, and the original forward assembler is
bounded by the initial-head step of this independently computed backwards recurrence. -/
theorem recurrence_checked (i : Fin 3) :
      let width := i.val + 2
      let heads := headCounts width 2 14
      let middles := (Array.range (width + 1)).map fun start => middleCounts width 2 14 start
      let value := cacheValue width continuationTables[i.val]!
      (∀ n : Fin 65, ∀ start : Fin (width + 1), ∀ mask : Fin (2 ^ width),
        value n.val start.val mask.val ≤
          suffixStep width 14 n.val start.val mask.val heads middles[start.val]! value) ∧
      (∀ n : Fin 65, (assembledCounts width 2 14 64)[n.val]! ≤
        weightedCounts heads (fun tag => headWeight width 14 n.val tag value)) := by
  have h := certificate_checked
  unfold certificateCheck at h
  have hi := List.all_eq_true.mp h i (List.mem_finRange i)
  simp only [Bool.and_eq_true] at hi
  constructor
  · intro n start mask
    have hn := List.all_eq_true.mp hi.1 n.val (List.mem_range.mpr n.isLt)
    have hs := List.all_eq_true.mp hn start.val (List.mem_range.mpr start.isLt)
    exact of_decide_eq_true (List.all_eq_true.mp hs mask.val (List.mem_range.mpr mask.isLt))
  · intro n
    exact of_decide_eq_true (List.all_eq_true.mp hi.2 n.val (List.mem_range.mpr n.isLt))

/-- The finite native check satisfies the generic logical recurrence contract. -/
theorem certified_subsolution (i : Fin 3) :
    Subsolution (i.val + 2) 14 64
      (cacheValue (i.val + 2) continuationTables[i.val]!) := by
  intro n hn start hs mask hm
  have h := (recurrence_checked i).1 ⟨n, by omega⟩ ⟨start, by omega⟩ ⟨mask, hm⟩
  simpa [getElem!_pos, show start < i.val + 2 + 1 from by omega] using h

/-- Each claimed coefficient counts no more than the number of actual distinct cap traces. -/
theorem assembledCounts_le_traces (i : Fin 3) (n : Nat) (hn : n ≤ 64) :
    (assembledCounts (i.val + 2) 2 14 64)[n]! ≤ (traces (i.val + 2) 14 n).length := by
  exact ((recurrence_checked i).2 ⟨n, by omega⟩).trans
    (traces_underestimate (i.val + 2) 14 64 (by decide)
      (cacheValue (i.val + 2) continuationTables[i.val]!)
      (certified_subsolution i) n hn)

/-- Consequently the coefficients undercount distinct valid irreducible bridge words,
not merely formal decompositions. -/
theorem assembledCounts_le_words (i : Fin 3) (n : Nat) (hn : n ≤ 64) :
    (assembledCounts (i.val + 2) 2 14 64)[n]! ≤
      ((traces (i.val + 2) 14 n).map List.flatten).length := by
  simpa only [List.length_map] using assembledCounts_le_traces i n hn

end RubiksSnake.CapAssembly

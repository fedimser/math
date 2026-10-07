import RubiksSnakePrunedComputation
import RubiksSnake.SlabEnumeration

/-!
# Certified undercounting by arbitrary slab pruning

The predicate may discard any node. Restoration and coefficient domination use
only the original traversal's additive histogram formula, not completeness of
the predicate or any geometric assumptions on the cursor and board.
-/

namespace RubiksSnake.SlabEnumeration

/-- Folding two update rules over the same list preserves a relation respected by each step. -/
private lemma foldl_rel {α β γ : Type*} (xs : List α)
    (f : β → α → β) (g : γ → α → γ) (r : β → γ → Prop)
    (step : ∀ b c a, r b c → r (f b a) (g c a))
    (b : β) (c : γ) (h : r b c) :
    r (xs.foldl f b) (xs.foldl g c) := by
  induction xs generalizing b c with
  | nil => exact h
  | cons a xs ih => exact ih (f b a) (g c a) (step b c a h)

/-- Arbitrary pruning restores the board, preserves histogram size, and undercounts every
in-range coefficient of the unpruned tree, even for an arbitrary initial cursor and board. -/
private theorem prunedCountSearch_spec (width side : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) (remaining : Nat) (s : Cursor)
    (board : ByteArray) (counts : Array Nat) :
    (prunedCountSearch width side irreducible keep remaining s board counts).1 = board ∧
    (prunedCountSearch width side irreducible keep remaining s board counts).2.size =
      counts.size ∧
    ∀ n, n < counts.size →
      (prunedCountSearch width side irreducible keep remaining s board counts).2[n]! ≤
        (countTree width side irreducible remaining s board counts)[n]! := by
  induction remaining generalizing s board counts with
  | zero =>
    by_cases hk : keep 0 s = true
    · rw [prunedCountSearch, if_pos hk, countTree]
      refine ⟨rfl, ?_, fun _ _ => le_rfl⟩
      dsimp only
      split_ifs <;> simp
    · rw [prunedCountSearch, if_neg hk]
      refine ⟨rfl, rfl, ?_⟩
      intro n hn
      rw [countTree_get _ _ _ _ _ _ _ n hn]
      exact Nat.le_add_right _ _
  | succ remaining ih =>
    by_cases hk : keep (remaining + 1) s = true
    · let old := board[s.position]!
      let start := if canExit width irreducible s old then increment counts s.length else counts
      let next := fun (state : ByteArray × Array Nat) outgoing =>
        if canMove width s old outgoing then
          let result := prunedCountSearch width side irreducible keep remaining
            (advance side s outgoing)
            (state.1.set! s.position (entry old s.incoming outgoing)) state.2
          (result.1.set! s.position old, result.2)
        else state
      let full := fun counts outgoing =>
        if canMove width s old outgoing then
          countTree width side irreducible remaining (advance side s outgoing)
            (board.set! s.position (entry old s.incoming outgoing)) counts
        else counts
      let r := fun (state : ByteArray × Array Nat) (upper : Array Nat) =>
        state.1 = board ∧ state.2.size = counts.size ∧ upper.size = counts.size ∧
          ∀ n, n < counts.size → state.2[n]! ≤ upper[n]!
      have hstart : start.size = counts.size := by
        dsimp [start]
        split_ifs <;> simp
      have hfold : r (directions.foldl next (board, start)) (directions.foldl full start) := by
        apply foldl_rel directions next full r
        · rintro ⟨current, lower⟩ upper outgoing ⟨hboard, hlo, hup, hle⟩
          dsimp only at hboard
          subst current
          dsimp only [next, full]
          split_ifs with hmove
          · obtain ⟨hb, hs, hbound⟩ := ih (advance side s outgoing)
              (board.set! s.position (entry old s.incoming outgoing)) lower
            refine ⟨?_, hs.trans hlo, (countTree_size _ _ _ _ _ _ _).trans hup, ?_⟩
            · rw [hb]
              exact restore board s.position (entry old s.incoming outgoing)
            · intro n hn
              have hnlo : n < lower.size := by simpa only [hlo] using hn
              have hnup : n < upper.size := by simpa only [hup] using hn
              refine (hbound n hnlo).trans ?_
              rw [countTree_get _ _ _ _ _ _ lower n hnlo,
                countTree_get _ _ _ _ _ _ upper n hnup]
              exact Nat.add_le_add_right (hle n hn) _
          · exact ⟨rfl, hlo, hup, hle⟩
        · exact ⟨rfl, hstart, hstart, fun _ _ => le_rfl⟩
      rw [prunedCountSearch, if_pos hk, countTree]
      exact ⟨hfold.1, hfold.2.1, hfold.2.2.2⟩
    · rw [prunedCountSearch, if_neg hk]
      refine ⟨rfl, rfl, ?_⟩
      intro n hn
      rw [countTree_get _ _ _ _ _ _ _ n hn]
      exact Nat.le_add_right _ _

/-- Backtracking leaves the input board unchanged, whether or not the current node is retained. -/
@[simp] theorem prunedCountSearch_restore (width side : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) (remaining : Nat) (s : Cursor)
    (board : ByteArray) (counts : Array Nat) :
    (prunedCountSearch width side irreducible keep remaining s board counts).1 = board :=
  (prunedCountSearch_spec width side irreducible keep remaining s board counts).1

/-- Pruning and histogram updates never change the number of coefficient slots. -/
@[simp] theorem prunedCountSearch_size (width side : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) (remaining : Nat) (s : Cursor)
    (board : ByteArray) (counts : Array Nat) :
    (prunedCountSearch width side irreducible keep remaining s board counts).2.size =
      counts.size :=
  (prunedCountSearch_spec width side irreducible keep remaining s board counts).2.1

/-- Every coefficient is bounded by the original restoring search's coefficient. -/
theorem prunedCountSearch_le_countSearch (width side : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) (remaining : Nat) (s : Cursor)
    (board : ByteArray) (counts : Array Nat) (n : Nat) (hn : n < counts.size) :
    (prunedCountSearch width side irreducible keep remaining s board counts).2[n]! ≤
      (countSearch width side irreducible remaining s board counts).2[n]! := by
  rw [countSearch_eq_countTree]
  exact (prunedCountSearch_spec width side irreducible keep remaining s board counts).2.2 n hn

/-- The initialized pruned histogram has one slot for each internal length from zero to `limit`. -/
@[simp] theorem prunedCounts_size (width limit : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) :
    (prunedCounts width limit irreducible keep).size = limit + 1 := by
  simp [prunedCounts]

/-- Arbitrary pruning gives lower coefficients at the standard initial board. -/
theorem prunedCounts_le_counts (width limit : Nat) (irreducible : Bool)
    (keep : Nat → Cursor → Bool) (n : Nat) (hn : n ≤ limit) :
    (prunedCounts width limit irreducible keep)[n]! ≤ (counts width limit irreducible)[n]! :=
  prunedCountSearch_le_countSearch width (2 * limit + 1) irreducible keep limit
    (initialCursor limit) (initialBoard width limit) (Array.replicate (limit + 1) 0)
    n (by simpa using Nat.lt_succ_of_le hn)

section Unpacked

attribute [local irreducible] budgetSearch prunedCountSearch

/-- Passing cursor fields separately has exactly the reference pruning semantics, for any
budget table and initial state; the optimization needs no geometric invariant. -/
theorem budgetSearch_eq_prunedCountSearch (width side : Nat) (irreducible : Bool)
    (masks : Nat) (table : Array Nat) (remaining length x position incoming mask : Nat)
    (board : ByteArray) (counts : Array Nat) :
    budgetSearch width side irreducible masks table remaining
        length x position incoming mask board counts =
      prunedCountSearch width side irreducible (tableBudgetKeep width masks table) remaining
        ⟨length, x, position, incoming, mask⟩ board counts := by
  induction remaining generalizing length x position incoming mask board counts with
  | zero =>
    rw [budgetSearch, prunedCountSearch]
  | succ remaining ih =>
    by_cases hk : tableBudgetKeep width masks table (remaining + 1)
        ⟨length, x, position, incoming, mask⟩ = true
    · rw [budgetSearch, prunedCountSearch, if_pos hk, if_pos hk]
      dsimp only
      congr 1
      funext state outgoing
      split_ifs
      · rw [ih]
      · rfl
    · rw [budgetSearch, prunedCountSearch, if_neg hk, if_neg hk]

/-- The optimized initialized counter equals the generic counter with the cached-budget predicate. -/
theorem budgetCounts_eq_prunedCounts (width limit : Nat) (irreducible : Bool) :
    budgetCounts width limit irreducible =
      prunedCounts width limit irreducible
        (tableBudgetKeep width (2 ^ width) (slabBudgetTable width irreducible)) := by
  unfold budgetCounts prunedCounts
  dsimp only
  rw [budgetSearch_eq_prunedCountSearch]

end Unpacked

/-- Budget pruning retains all `limit + 1` histogram slots, including zero coefficients. -/
@[simp] theorem budgetCounts_size (width limit : Nat) (irreducible : Bool) :
    (budgetCounts width limit irreducible).size = limit + 1 := by
  rw [budgetCounts_eq_prunedCounts]
  simp

/-- Each in-range budget-pruned coefficient is a certified lower bound on the full slab count. -/
theorem budgetCounts_le_counts (width limit : Nat) (irreducible : Bool)
    (n : Nat) (hn : n ≤ limit) :
    (budgetCounts width limit irreducible)[n]! ≤ (counts width limit irreducible)[n]! := by
  rw [budgetCounts_eq_prunedCounts]
  exact prunedCounts_le_counts width limit irreducible
    (tableBudgetKeep width (2 ^ width) (slabBudgetTable width irreducible)) n hn

/-- Coefficient domination holds at every natural index when out-of-range entries are read as zero. -/
theorem budgetCounts_getD_le_counts (width limit : Nat) (irreducible : Bool) (n : Nat) :
    (budgetCounts width limit irreducible)[n]?.getD 0 ≤
      (counts width limit irreducible)[n]?.getD 0 := by
  by_cases hn : n ≤ limit
  · have hp : n < (budgetCounts width limit irreducible).size := by simp; omega
    have hf : n < (counts width limit irreducible).size := by simp; omega
    simpa only [getElem?_pos (budgetCounts width limit irreducible) n hp,
      getElem?_pos (counts width limit irreducible) n hf, Option.getD_some,
      getElem!_pos (budgetCounts width limit irreducible) n hp,
      getElem!_pos (counts width limit irreducible) n hf] using
      budgetCounts_le_counts width limit irreducible n hn
  · have hp : ¬n < (budgetCounts width limit irreducible).size := by simp; omega
    have hf : ¬n < (counts width limit irreducible).size := by simp; omega
    simp only [getElem?_neg (budgetCounts width limit irreducible) n hp,
      getElem?_neg (counts width limit irreducible) n hf, Option.getD_none, le_refl]

end RubiksSnake.SlabEnumeration

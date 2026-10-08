import RubiksSnakeCapComputation
import RubiksSnake.SlabEnumeration

/-!
# Semantics of the short cap traversal

Leaves retain both their direction word and their coefficient slot. This
separates geometric word membership from the executable histogram.
-/

namespace RubiksSnake.CapEnumeration

open SlabEnumeration

/-- Accepted internal words, tagged with their terminal coefficient slot. -/
def leaves (width transverse side limit : Nat) :
    Nat → Cursor → CrossState → ByteArray → List (List Nat × Nat)
  | remaining, s, cross, board =>
    let old := board[s.position]!
    (if canExit transverse true s old then
      [([], slot width limit cross (s.length + 1))] else []) ++
    match remaining with
    | 0 => []
    | remaining + 1 =>
      directions.flatMap fun outgoing =>
        if canMove transverse s old outgoing && crossAllowed width cross outgoing then
          (leaves width transverse side limit remaining (advance side s outgoing)
            (crossAdvance cross outgoing)
            (board.set! s.position (entry old s.incoming outgoing))).map
              (fun p => (outgoing :: p.1, p.2))
        else []

/-- Number of accepted words assigned to one coefficient slot. -/
def histogram (n : Nat) (ws : List (List Nat × Nat)) : Nat :=
  (ws.filter fun p => p.2 == n).length

/-- An immutable-board specification of the executable histogram traversal. -/
def countTree (width transverse side limit : Nat) :
    Nat → Cursor → CrossState → ByteArray → Array Nat → Array Nat
  | remaining, s, cross, board, counts =>
    let old := board[s.position]!
    let counts := if canExit transverse true s old then
      increment counts (slot width limit cross (s.length + 1)) else counts
    match remaining with
    | 0 => counts
    | remaining + 1 =>
      directions.foldl (fun counts outgoing =>
        if canMove transverse s old outgoing && crossAllowed width cross outgoing then
          countTree width transverse side limit remaining (advance side s outgoing)
            (crossAdvance cross outgoing)
            (board.set! s.position (entry old s.incoming outgoing)) counts
        else counts) counts

/-- Carrying a restored board through a fold does not alter the coefficient fold. -/
private lemma foldl_pair {α β γ : Type*} (xs : List α) (b : β) (c : γ)
    (f : β × γ → α → β × γ) (g : γ → α → γ)
    (h : ∀ c a, f (b, c) a = (b, g c a)) :
    xs.foldl f (b, c) = (b, xs.foldl g c) := by
  induction xs generalizing c with
  | nil => rfl
  | cons a xs ih => simp only [List.foldl_cons, h, ih]

/-- The executable traversal restores its board and agrees with the pure tree. -/
theorem search_eq_countTree (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray) (counts : Array Nat) :
    search width transverse side limit remaining s cross board counts =
      (board, countTree width transverse side limit remaining s cross board counts) := by
  induction remaining generalizing s cross board counts with
  | zero => rfl
  | succ remaining ih =>
    rw [search, countTree]
    apply foldl_pair
    intro counts outgoing
    dsimp only
    split_ifs
    · rw [ih]
      simp only [restore]
    · rfl

/-- Every accepted word is also in the original geometrically verified traversal. -/
theorem leaves_subset (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray)
    {w : List Nat} {tag : Nat}
    (hw : (w, tag) ∈ leaves width transverse side limit remaining s cross board) :
    w ∈ SlabEnumeration.words transverse side true remaining s board := by
  induction remaining generalizing s cross board w tag with
  | zero =>
    simp only [leaves, SlabEnumeration.words, List.append_nil] at hw ⊢
    split at hw
    · have hw' : w = [] := by simpa using congrArg Prod.fst (List.mem_singleton.mp hw)
      simp_all
    · simp at hw
  | succ remaining ih =>
    rw [leaves] at hw
    rw [SlabEnumeration.words]
    rcases List.mem_append.mp hw with hw | hw
    · apply List.mem_append.mpr
      left
      split at hw
      · have hw' : w = [] := by simpa using congrArg Prod.fst (List.mem_singleton.mp hw)
        simp_all
      · simp at hw
    · obtain ⟨outgoing, houtgoing, hw⟩ := List.mem_flatMap.mp hw
      split at hw
      · rename_i hmove
        simp only [Bool.and_eq_true] at hmove
        have hm := hmove.1
        obtain ⟨⟨tail, label⟩, htail, heq⟩ := List.mem_map.mp hw
        have hw' : w = outgoing :: tail := (congrArg Prod.fst heq).symm
        subst w
        apply List.mem_append.mpr
        right
        apply List.mem_flatMap.mpr
        refine ⟨outgoing, houtgoing, ?_⟩
        rw [if_pos hm]
        exact List.mem_map.mpr ⟨tail, ih _ _ _ htail, rfl⟩
      · simp at hw

/-- Pruning removes words without changing their order or creating duplicates. -/
theorem leaves_sublist (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray) :
    List.Sublist ((leaves width transverse side limit remaining s cross board).map Prod.fst)
      (SlabEnumeration.words transverse side true remaining s board) := by
  induction remaining generalizing s cross board with
  | zero =>
    simp only [leaves, SlabEnumeration.words, List.append_nil]
    split_ifs <;> simp
  | succ remaining ih =>
    rw [leaves, SlabEnumeration.words, List.map_append]
    apply List.Sublist.append
    · split_ifs <;> simp
    · rw [List.map_flatMap]
      apply List.Sublist.flatMap_right
      intro outgoing _
      by_cases hm : canMove transverse s board[s.position]! outgoing = true
      · by_cases hc : crossAllowed width cross outgoing = true
        · simp only [hm, hc, Bool.and_self, if_true, List.map_map, Function.comp_def]
          simpa only [List.map_map, Function.comp_def] using
            (ih (advance side s outgoing) (crossAdvance cross outgoing)
              (board.set! s.position (entry board[s.position]! s.incoming outgoing))).map
                (List.cons outgoing)
        · simp [hm, hc]
      · simp [hm]

/-- Each counted word occurs only once, independently of its terminal tag. -/
theorem leaves_nodup (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray) :
    ((leaves width transverse side limit remaining s cross board).map Prod.fst).Nodup :=
  (SlabEnumeration.words_nodup transverse side true remaining s board).sublist
    (leaves_sublist width transverse side limit remaining s cross board)

/-- Prefixing direction words preserves the histogram of their terminal tags. -/
@[simp] lemma histogram_prepend (n outgoing : Nat) (ws : List (List Nat × Nat)) :
    histogram n (ws.map fun p => (outgoing :: p.1, p.2)) = histogram n ws := by
  simp [histogram, List.filter_map, Function.comp_def]

/-- Histograms add over disjoint traversal branches, without any geometric assumption. -/
@[simp] lemma histogram_append (n : Nat) (a b : List (List Nat × Nat)) :
    histogram n (a ++ b) = histogram n a + histogram n b := by
  simp [histogram, List.filter_append]

/-- A terminal leaf contributes one exactly at its own slot. -/
@[simp] lemma histogram_singleton (n tag : Nat) (w : List Nat) :
    histogram n [(w, tag)] = if tag = n then 1 else 0 := by
  by_cases h : tag = n <;> simp [histogram, h]

/-- Size preservation for a fold of array updates. -/
private lemma foldl_size {α : Type*} (xs : List α) (f : Array Nat → α → Array Nat)
    (h : ∀ counts a, (f counts a).size = counts.size) (counts : Array Nat) :
    (xs.foldl f counts).size = counts.size := by
  induction xs generalizing counts with
  | nil => rfl
  | cons a xs ih => simp only [List.foldl_cons, ih, h]

/-- Every terminal increment and recursive branch preserves histogram size. -/
@[simp] theorem countTree_size (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray) (counts : Array Nat) :
    (countTree width transverse side limit remaining s cross board counts).size =
      counts.size := by
  induction remaining generalizing s cross board counts with
  | zero =>
    rw [countTree]
    split_ifs <;> simp
  | succ remaining ih =>
    rw [countTree]
    rw [foldl_size]
    · split_ifs <;> simp
    · intro counts outgoing
      split_ifs
      · exact ih _ _ _ _
      · rfl

/-- Additive updates accumulate the sum of their per-branch contributions. -/
private lemma foldl_get_add {α : Type*} (xs : List α)
    (f : Array Nat → α → Array Nat) (delta : α → Nat) (n : Nat)
    (hsize : ∀ counts a, (f counts a).size = counts.size)
    (hget : ∀ counts a, n < counts.size → (f counts a)[n]! = counts[n]! + delta a)
    (counts : Array Nat) (hn : n < counts.size) :
    (xs.foldl f counts)[n]! = counts[n]! + (xs.map delta).sum := by
  induction xs generalizing counts with
  | nil => simp
  | cons a xs ih =>
    rw [List.foldl_cons, ih (f counts a) (by simpa [hsize] using hn), hget counts a hn]
    simp [Nat.add_assoc]

/-- Each executable coefficient counts precisely the leaves tagged with that slot. -/
theorem countTree_get (width transverse side limit remaining : Nat)
    (s : Cursor) (cross : CrossState) (board : ByteArray) (counts : Array Nat)
    (n : Nat) (hn : n < counts.size) :
    (countTree width transverse side limit remaining s cross board counts)[n]! =
      counts[n]! + histogram n (leaves width transverse side limit remaining s cross board) := by
  induction remaining generalizing s cross board counts with
  | zero =>
    rw [countTree, leaves]
    split_ifs
    · rw [increment_get counts _ n hn]
      simp only [List.append_nil, histogram_singleton]
    · simp [histogram]
  | succ remaining ih =>
    let old := board[s.position]!
    let start := if canExit transverse true s old then
      increment counts (slot width limit cross (s.length + 1)) else counts
    let next := fun counts outgoing =>
      if canMove transverse s old outgoing && crossAllowed width cross outgoing then
        countTree width transverse side limit remaining (advance side s outgoing)
          (crossAdvance cross outgoing)
          (board.set! s.position (entry old s.incoming outgoing)) counts
      else counts
    let children := fun outgoing =>
      if canMove transverse s old outgoing && crossAllowed width cross outgoing then
        (leaves width transverse side limit remaining (advance side s outgoing)
          (crossAdvance cross outgoing)
          (board.set! s.position (entry old s.incoming outgoing))).map
            (fun p => (outgoing :: p.1, p.2))
      else []
    have hsize : ∀ counts outgoing, (next counts outgoing).size = counts.size := by
      intro counts outgoing
      dsimp [next]
      split_ifs <;> simp
    have hget : ∀ counts outgoing, n < counts.size →
        (next counts outgoing)[n]! = counts[n]! + histogram n (children outgoing) := by
      intro counts outgoing hn
      dsimp [next, children]
      split_ifs
      · rw [ih _ _ _ _ hn, histogram_prepend]
      · simp [histogram]
    have hstart : start.size = counts.size := by
      dsimp [start]
      split_ifs <;> simp
    have hflat : histogram n (directions.flatMap children) =
        (directions.map fun outgoing => histogram n (children outgoing)).sum := by
      simp [histogram, List.filter_flatMap, List.length_flatMap]
    change (directions.foldl next start)[n]! = _
    rw [foldl_get_add directions next (fun outgoing => histogram n (children outgoing))
      n hsize hget start (by simpa [hstart] using hn)]
    rw [leaves, histogram_append]
    change _ = counts[n]! + (histogram n
      (if canExit transverse true s old then
        [([], slot width limit cross (s.length + 1))] else []) +
      histogram n (directions.flatMap children))
    rw [hflat]
    dsimp [start]
    split_ifs
    · rw [increment_get counts _ n hn, histogram_singleton]
      omega
    · simp [histogram]

end RubiksSnake.CapEnumeration

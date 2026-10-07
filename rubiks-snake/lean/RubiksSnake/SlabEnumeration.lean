import RubiksSnakeComputation
import Init.Data.ByteArray.Lemmas
import Mathlib.Tactic

/-! Correctness of the restoring traversal as a finite word enumeration. -/

namespace RubiksSnake.SlabEnumeration

/-- Writing a byte and then restoring its saved value leaves the board unchanged. -/
@[simp] lemma restore (board : ByteArray) (i : Nat) (value : UInt8) :
    (board.set! i value).set! i board[i]! = board := by
  apply ByteArray.ext
  apply Array.ext
  · simp [ByteArray.set!]
  · intro j _ hj
    have hj' : j < board.size := hj
    change ((board.set! i value).set! i board[i]!)[j] = board[j]
    by_cases hij : i = j
    · subst j
      simp [hj']
    · rw [ByteArray.getElem_set!_ne _ i j _ hij (by simpa using hj'),
        ByteArray.getElem_set!_ne _ i j _ hij hj']

/-- Immutable-board specification of slab counting: accepted exits increment
the current internal-edge length, and sibling branches share the same starting board. -/
def countTree (width side : Nat) (irreducible : Bool) :
    Nat → Cursor → ByteArray → Array Nat → Array Nat
  | remaining, s, board, counts =>
    let old := board[s.position]!
    let counts := if canExit width irreducible s old then increment counts s.length else counts
    match remaining with
    | 0 => counts
    | remaining + 1 =>
      directions.foldl (fun counts outgoing =>
        if canMove width s old outgoing then
          countTree width side irreducible remaining (advance side s outgoing)
            (board.set! s.position (entry old s.incoming outgoing)) counts
        else counts) counts

/-- A fold that preserves its first component is the corresponding fold of
the second component with the first carried unchanged. -/
private lemma foldl_pair_left {α β γ : Type*} (xs : List α) (b : β) (c : γ)
    (f : β × γ → α → β × γ) (g : γ → α → γ)
    (h : ∀ c a, f (b, c) a = (b, g c a)) :
    xs.foldl f (b, c) = (b, xs.foldl g c) := by
  induction xs generalizing c with
  | nil => rfl
  | cons a xs ih => simp only [List.foldl_cons, h, ih]

/-- The restoring traversal returns the original board and exactly the count
array produced by the immutable-board specification. -/
theorem countSearch_eq_countTree (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) (counts : Array Nat) :
    countSearch width side irreducible remaining s board counts =
      (board, countTree width side irreducible remaining s board counts) := by
  induction remaining generalizing s board counts with
  | zero => rfl
  | succ remaining ih =>
    rw [countSearch, countTree]
    apply foldl_pair_left
    intro counts outgoing
    dsimp only
    split_ifs
    · rw [ih]
      simp only [restore]
    · rfl

/-- Incrementing a length count never changes the count array's size. -/
@[simp] lemma increment_size (counts : Array Nat) (length : Nat) :
    (increment counts length).size = counts.size := by
  simp [increment]

/-- An in-bounds count entry increases by one precisely when its index equals
the length being incremented. -/
lemma increment_get (counts : Array Nat) (length n : Nat) (hn : n < counts.size) :
    (increment counts length)[n]! = counts[n]! + if length = n then 1 else 0 := by
  by_cases h : length = n
  · subst length
    simpa [increment] using Array.getElem!_set!_self counts n (counts[n]! + 1) hn
  · simpa [increment, h] using Array.getElem!_set!_ne counts length n
      (counts[length]! + 1) h

/-- Folding array updates that individually preserve size also preserves
the original array size. -/
private lemma foldl_size {α : Type*} (xs : List α) (f : Array Nat → α → Array Nat)
    (h : ∀ counts a, (f counts a).size = counts.size) (counts : Array Nat) :
    (xs.foldl f counts).size = counts.size := by
  induction xs generalizing counts with
  | nil => rfl
  | cons a xs ih => simp only [List.foldl_cons, ih, h]

/-- The entire counting traversal preserves the length of its count array. -/
@[simp] lemma countTree_size (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) (counts : Array Nat) :
    (countTree width side irreducible remaining s board counts).size = counts.size := by
  induction remaining generalizing s board counts with
  | zero =>
    rw [countTree]
    split_ifs <;> simp
  | succ remaining ih =>
    rw [countTree]
    rw [foldl_size]
    · split_ifs <;> simp
    · intro counts outgoing
      split_ifs
      · exact ih _ _ _
      · rfl

/-- Enumerate internal direction suffixes accepted by the slab traversal,
using at most `remaining` edges; the final `+x` exit is implicit. -/
def words (width side : Nat) (irreducible : Bool) :
    Nat → Cursor → ByteArray → List (List Nat)
  | remaining, s, board =>
    let old := board[s.position]!
    (if canExit width irreducible s old then [[]] else []) ++
    match remaining with
    | 0 => []
    | remaining + 1 =>
      directions.flatMap (fun outgoing =>
        if canMove width s old outgoing then
          (words width side irreducible remaining (advance side s outgoing)
            (board.set! s.position (entry old s.incoming outgoing))).map (outgoing :: ·)
        else [])

/-- Count list entries whose word length plus the already traversed `length`
equals the target internal-edge index `n`. -/
def histogram (length n : Nat) (ws : List (List Nat)) : Nat :=
  (ws.filter fun w => length + w.length == n).length

/-- An empty list of candidate words contributes zero to every length count. -/
@[simp] lemma histogram_nil (length n : Nat) : histogram length n [] = 0 := rfl

/-- Length counts add when two lists of candidate words are concatenated. -/
@[simp] lemma histogram_append (length n : Nat) (xs ys : List (List Nat)) :
    histogram length n (xs ++ ys) = histogram length n xs + histogram length n ys := by
  simp [histogram, List.filter_append]

/-- The length count of a flattened family is the sum of its members' length counts. -/
lemma histogram_flatMap {α : Type*} (length n : Nat) (xs : List α)
    (f : α → List (List Nat)) :
    histogram length n (xs.flatMap f) = (xs.map fun x => histogram length n (f x)).sum := by
  simp [histogram, List.filter_flatMap, List.length_flatMap]

/-- Prepending a direction to every candidate is equivalent to increasing
the already traversed length by one in the histogram. -/
lemma histogram_consMap (length n outgoing : Nat) (ws : List (List Nat)) :
    histogram length n (ws.map (outgoing :: ·)) = histogram (length + 1) n ws := by
  simp [histogram, List.filter_map, Function.comp_def, Nat.add_comm,
    Nat.add_left_comm]

/-- For size-preserving updates with fixed increments at index `n`, a fold
adds the sum of those increments to the original in-bounds entry. -/
private lemma foldl_get_add {α : Type*} (xs : List α) (f : Array Nat → α → Array Nat)
    (delta : α → Nat) (n : Nat)
    (hsize : ∀ counts a, (f counts a).size = counts.size)
    (hget : ∀ counts a, n < counts.size →
      (f counts a)[n]! = counts[n]! + delta a)
    (counts : Array Nat) (hn : n < counts.size) :
    (xs.foldl f counts)[n]! = counts[n]! + (xs.map delta).sum := by
  induction xs generalizing counts with
  | nil => simp
  | cons a xs ih =>
    rw [List.foldl_cons, ih (f counts a) (by simpa [hsize] using hn), hget counts a hn]
    simp [Nat.add_assoc]

/-- At each in-bounds index `n`, traversal adds exactly the number of accepted
suffixes whose length plus the cursor's existing internal-edge length is `n`. -/
lemma countTree_get (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) (counts : Array Nat)
    (n : Nat) (hn : n < counts.size) :
    (countTree width side irreducible remaining s board counts)[n]! =
      counts[n]! + histogram s.length n (words width side irreducible remaining s board) := by
  induction remaining generalizing s board counts with
  | zero =>
    rw [countTree, words]
    split_ifs <;> simp [histogram, List.filter_cons, apply_ite,
      increment_get counts s.length n hn]
  | succ remaining ih =>
    let old := board[s.position]!
    let start := if canExit width irreducible s old then increment counts s.length else counts
    let next := fun counts outgoing =>
      if canMove width s old outgoing then
        countTree width side irreducible remaining (advance side s outgoing)
          (board.set! s.position (entry old s.incoming outgoing)) counts
      else counts
    let delta := fun outgoing =>
      if canMove width s old outgoing then
        histogram (s.length + 1) n
          (words width side irreducible remaining (advance side s outgoing)
            (board.set! s.position (entry old s.incoming outgoing)))
      else 0
    have hsize : ∀ counts outgoing, (next counts outgoing).size = counts.size := by
      intro counts outgoing
      dsimp [next]
      split_ifs <;> simp
    have hget : ∀ counts outgoing, n < counts.size →
        (next counts outgoing)[n]! = counts[n]! + delta outgoing := by
      intro counts outgoing hk
      dsimp [next, delta]
      split_ifs
      · simpa [advance] using ih (advance side s outgoing)
          (board.set! s.position (entry old s.incoming outgoing)) counts hk
      · simp
    have hstartSize : start.size = counts.size := by
      dsimp [start]
      split_ifs <;> simp
    have hstart :
        start[n]! = counts[n]! +
          histogram s.length n (if canExit width irreducible s old then [[]] else []) := by
      dsimp [start]
      split_ifs <;> simp [histogram, List.filter_cons, apply_ite,
        increment_get counts s.length n hn]
    rw [countTree]
    change (directions.foldl next start)[n]! = _
    rw [foldl_get_add directions next delta n hsize hget start (by simpa [hstartSize] using hn),
      hstart, words, histogram_append]
    rw [Nat.add_assoc]
    congr 1
    congr 1
    rw [histogram_flatMap]
    congr 1
    apply List.map_congr_left
    intro outgoing _
    dsimp only [delta, old]
    split_ifs
    · exact (histogram_consMap s.length n outgoing _).symm
    · rfl

/-- Every accepted internal suffix respects the remaining edge budget. -/
lemma words_length_le (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) :
    ∀ w ∈ words width side irreducible remaining s board, w.length ≤ remaining := by
  induction remaining generalizing s board with
  | zero =>
    intro w hw
    rw [words] at hw
    split at hw <;> simp_all
  | succ remaining ih =>
    intro w hw
    rw [words] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw <;> simp_all
    · obtain ⟨outgoing, _, hw⟩ := List.mem_flatMap.mp hw
      split at hw
      · obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
        have := ih _ _ tail htail
        simp only [List.length_cons]
        omega
      · simp at hw

/-- Distinct search branches produce distinct internal direction words. -/
lemma words_nodup (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) :
    (words width side irreducible remaining s board).Nodup := by
  induction remaining generalizing s board with
  | zero =>
    rw [words]
    split_ifs <;> simp
  | succ remaining ih =>
    rw [words]
    apply List.nodup_append.mpr
    refine ⟨?_, ?_, ?_⟩
    · split_ifs <;> simp
    · apply List.nodup_flatMap.mpr
      constructor
      · intro outgoing _
        split_ifs
        · exact (ih _ _).map (fun _ _ h => (List.cons.inj h).2)
        · simp
      · apply (by decide : directions.Nodup).imp_of_mem
        intro a b _ _ hne w hwa hwb
        dsimp only at hwa hwb
        split at hwa <;> split at hwb <;> try simp_all only [List.not_mem_nil]
        obtain ⟨x, _, hx⟩ := List.mem_map.mp hwa
        obtain ⟨y, _, hy⟩ := List.mem_map.mp hwb
        exact hne (List.cons.inj (hx.trans hy.symm)).1
    · intro w hw v hmore heq
      subst v
      split at hw
      · have hw' : w = [] := by simpa using hw
        subst w
        obtain ⟨outgoing, _, hmore⟩ := List.mem_flatMap.mp hmore
        split at hmore <;> simp_all
      · simp at hw

/-- Origin cursor with incoming `+x`, zero internal edges, and no recorded cut
crossings; `limit` sets the transverse offset in the flattened board. -/
def initialCursor (limit : Nat) : Cursor :=
  ⟨0, 0, limit * (2 * limit + 1) + limit, 0, 0⟩

/-- Empty occupancy board for `width + 1` planes and transverse side length
`2 * limit + 1` in both directions. -/
def initialBoard (width limit : Nat) : ByteArray :=
  ByteArray.mk (Array.replicate ((width + 1) * (2 * limit + 1) * (2 * limit + 1)) 0)

/-- Accepted internal words starting at the origin on an empty slab board,
with at most `limit` edges and the final `+x` exit still implicit. -/
def internalWords (width limit : Nat) (irreducible : Bool) : List (List Nat) :=
  words width (2 * limit + 1) irreducible limit (initialCursor limit) (initialBoard width limit)

/-- Append the final `+x` exit to each accepted internal word: `k` internal
edges give block length `k + 1`, with total advance `d = width + 1`. -/
def blockWords (width limit : Nat) (irreducible : Bool) : List (List Nat) :=
  (internalWords width limit irreducible).map (· ++ [0])

/-- The count array has one entry for every internal-edge length from zero
through `limit`, inclusive. -/
@[simp] lemma counts_size (width limit : Nat) (irreducible : Bool) :
    (counts width limit irreducible).size = limit + 1 := by
  unfold counts
  dsimp only
  rw [countSearch_eq_countTree]
  simp

/-- For `n <= limit`, the executable count at index `n` equals the number of
enumerated internal words of length `n`. -/
theorem counts_get (width limit : Nat) (irreducible : Bool)
    (n : Nat) (hn : n ≤ limit) :
    (counts width limit irreducible)[n]! =
      histogram 0 n (internalWords width limit irreducible) := by
  change (countSearch width (2 * limit + 1) irreducible limit (initialCursor limit)
    (initialBoard width limit) (Array.replicate (limit + 1) 0)).2[n]! = _
  rw [countSearch_eq_countTree]
  dsimp only
  rw [countTree_get _ _ _ _ _ _ _ n (by simp; omega)]
  change (Array.replicate (limit + 1) 0)[n]! +
    histogram 0 n (internalWords width limit irreducible) = _
  rw [getElem!_pos (Array.replicate (limit + 1) (0 : Nat)) n
    (by simpa using Nat.lt_succ_of_le hn)]
  simp

/-- Internal-edge count index `n` is exactly the number of retained block
words of length `n + 1`, when `n` lies within the cutoff. -/
theorem blockWords_count (width limit : Nat) (irreducible : Bool)
    (n : Nat) (hn : n ≤ limit) :
    ((blockWords width limit irreducible).filter fun w => w.length == n + 1).length =
      (counts width limit irreducible)[n]! := by
  rw [counts_get width limit irreducible n hn]
  simp [blockWords, histogram, List.filter_map, Function.comp_def]

/-- Adding the common terminal exit preserves duplicate-free block enumeration. -/
lemma blockWords_nodup (width limit : Nat) (irreducible : Bool) :
    (blockWords width limit irreducible).Nodup :=
  (words_nodup width (2 * limit + 1) irreducible limit _ _).map
    (fun _ _ h => List.append_cancel_right h)

/-- Every complete block has between one and `limit + 1` direction letters,
including its terminal exit. -/
lemma blockWords_lengths (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) :
    1 ≤ w.length ∧ w.length ≤ limit + 1 := by
  obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
  have := words_length_le width (2 * limit + 1) irreducible limit _ _ tail htail
  simp only [List.length_append, List.length_singleton]
  omega

/-- For any internal-edge index `n`, the length-`n + 1` block count equals
the array lookup, extended by zero beyond the cutoff. -/
lemma blockWords_count_all (width limit : Nat) (irreducible : Bool) (n : Nat) :
    ((blockWords width limit irreducible).filter fun w => w.length == n + 1).length =
      (counts width limit irreducible)[n]?.getD 0 := by
  by_cases hn : n ≤ limit
  · rw [blockWords_count width limit irreducible n hn]
    have hindex : n < (counts width limit irreducible).size := by simp; omega
    rw [getElem!_pos (counts width limit irreducible) n hindex,
      getElem?_pos (counts width limit irreducible) n hindex]
    rfl
  · have hfilter :
        (blockWords width limit irreducible).filter (fun w => w.length == n + 1) = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro w hw
      obtain ⟨hw, hlength⟩ := List.mem_filter.mp hw
      have hbound := (blockWords_lengths width limit irreducible hw).2
      simp only [beq_iff_eq] at hlength
      omega
    have hindex : ¬n < (counts width limit irreducible).size := by simp; omega
    rw [hfilter, getElem?_neg (counts width limit irreducible) n hindex]
    rfl

end RubiksSnake.SlabEnumeration

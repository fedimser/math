import RubiksSnake.SlabGeometry

/-!
# Semantic slab bridges

The bounded enumeration keeps every internal prefix in the slab and ends at
its right boundary. The terminal exit step makes each block a bridge.
For irreducible blocks, each bit of the exit mask witnesses a backward crossing
of the corresponding internal cut.
-/

namespace RubiksSnake.SlabEnumeration

open BridgeWords

/-- Direction index zero, the `+x` step, increases height by one. -/
private lemma xStep_zero : xStep 0 = 1 := rfl

/-- Direction index one, the `-x` step, decreases height by one. -/
private lemma xStep_one : xStep 1 = -1 := rfl

/-- A nonempty direction word's height is its first increment plus its tail's height. -/
private lemma height_cons (outgoing : Nat) (w : List Nat) :
    height xStep (outgoing :: w) = xStep outgoing + height xStep w := by
  simp [height]

/-- From a cursor inside the slab, an accepted internal suffix finishes at
`x = width` and keeps every prefix within `0 <= x <= width`. -/
lemma words_span (width side : Nat) (irreducible : Bool)
    (remaining : Nat) (s : Cursor) (board : ByteArray) (hx : s.x ≤ width)
    {w : List Nat} (hw : w ∈ words width side irreducible remaining s board) :
    (s.x : ℤ) + height xStep w = (width : ℤ) ∧
      ∀ u v, w = u ++ v →
        0 ≤ (s.x : ℤ) + height xStep u ∧
          (s.x : ℤ) + height xStep u ≤ (width : ℤ) := by
  induction remaining generalizing s board w with
  | zero =>
    rw [words] at hw
    split at hw
    · rename_i hexit
      have hw' : w = [] := by simpa using hw
      subst w
      have hs := ((canExit_spec width irreducible s board[s.position]!).mp hexit).1
      refine ⟨by simp [hs], ?_⟩
      intro u v huv
      have hu : u = [] := (List.append_eq_nil_iff.mp huv.symm).1
      subst u
      simp [hs]
    · simp at hw
  | succ remaining ih =>
    rw [words] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw
      · rename_i hexit
        have hw' : w = [] := by simpa using hw
        subst w
        have hs := ((canExit_spec width irreducible s board[s.position]!).mp hexit).1
        refine ⟨by simp [hs], ?_⟩
        intro u v huv
        have hu : u = [] := (List.append_eq_nil_iff.mp huv.symm).1
        subst u
        simp [hs]
      · simp at hw
    · obtain ⟨outgoing, houtgoing, hw⟩ := List.mem_flatMap.mp hw
      split at hw
      · rename_i hmove
        obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
        have hspan := ih (s := advance side s outgoing)
          (board := board.set! s.position (entry board[s.position]! s.incoming outgoing))
          (w := tail) (move_x_le width side s board[s.position]! outgoing hx hmove) htail
        have hstep := move_x_eq width side s board[s.position]! outgoing
          (mem_directions.mp houtgoing) hmove
        refine ⟨?_, ?_⟩
        · rw [height_cons]
          have := hspan.1
          omega
        · intro u v huv
          cases u with
          | nil =>
            simp only [height_nil, add_zero]
            exact ⟨Nat.cast_nonneg _, by exact_mod_cast hx⟩
          | cons a u =>
            obtain ⟨rfl, htailEq⟩ := List.cons.inj huv
            have hp := hspan.2 u v htailEq
            simp only [height_cons]
            constructor <;> omega
      · simp at hw

/-- An origin-based internal word has total height `width`, with all prefix
heights between zero and `width`, inclusively. -/
lemma internalWords_span (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ internalWords width limit irreducible) :
    height xStep w = (width : ℤ) ∧
      ∀ u v, w = u ++ v →
        0 ≤ height xStep u ∧ height xStep u ≤ (width : ℤ) := by
  simpa only [initialCursor, Nat.cast_zero, zero_add] using
    words_span width (2 * limit + 1) irreducible limit (initialCursor limit)
      (initialBoard width limit) (by simp [initialCursor]) hw

/-- Mask bit `j` survives an update, or is newly set by a `-x` move whose
updated cursor coordinate is `j`. -/
private lemma advance_mask_testBit (side : Nat) (s : Cursor) (outgoing j : Nat) :
    (advance side s outgoing).mask.testBit j = true ↔
      s.mask.testBit j = true ∨ outgoing = 1 ∧ (advance side s outgoing).x = j := by
  by_cases h : outgoing = 1
  · subst outgoing
    simp [advance, Nat.shiftLeft_eq, Nat.testBit_two_pow]
  · simp [advance, h]

/-- An accepted irreducible suffix either inherits a cut's mask bit or crosses
that cut backwards within the suffix. Bit `j` records a crossing from `j + 1` to `j`. -/
lemma words_backward_crossings (width side remaining : Nat) (s : Cursor)
    (board : ByteArray) {w : List Nat}
    (hw : w ∈ words width side true remaining s board) (j : Nat) (hj : j < width) :
    s.mask.testBit j = true ∨
      ∃ u v, w = u ++ [1] ++ v ∧
        (s.x : ℤ) + height xStep u = (j : ℤ) + 1 := by
  induction remaining generalizing s board w with
  | zero =>
    rw [words] at hw
    split at hw
    · rename_i hexit
      have hmask :=
        ((canExit_spec width true s board[s.position]!).mp hexit).2.2.2 rfl
      exact Or.inl (by simp [hmask, Nat.testBit_two_pow_sub_one, hj])
    · simp at hw
  | succ remaining ih =>
    rw [words] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw
      · rename_i hexit
        have hmask :=
          ((canExit_spec width true s board[s.position]!).mp hexit).2.2.2 rfl
        exact Or.inl (by simp [hmask, Nat.testBit_two_pow_sub_one, hj])
      · simp at hw
    · obtain ⟨outgoing, houtgoing, hw⟩ := List.mem_flatMap.mp hw
      split at hw
      · rename_i hmove
        obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
        have hcross := ih (s := advance side s outgoing)
          (board := board.set! s.position (entry board[s.position]! s.incoming outgoing))
          (w := tail) htail
        have hstep := move_x_eq width side s board[s.position]! outgoing
          (mem_directions.mp houtgoing) hmove
        rcases hcross with hmask | ⟨u, v, huv, hheight⟩
        · rcases (advance_mask_testBit side s outgoing j).mp hmask with
            hmask | ⟨rfl, hx⟩
          · exact Or.inl hmask
          · refine Or.inr ⟨[], tail, by simp, ?_⟩
            rw [hx, xStep_one] at hstep
            simp only [height_nil, add_zero]
            omega
        · refine Or.inr ⟨outgoing :: u, v, ?_, ?_⟩
          · simp only [List.cons_append, huv]
          · rw [height_cons]
            omega
      · simp at hw

/-- With the initial mask empty, every internal word accepted in irreducible
mode contains a `-x` crossing from height `j + 1` to `j` for every `j < width`. -/
lemma internalWords_backward_crossings (width limit : Nat) {w : List Nat}
    (hw : w ∈ internalWords width limit true) (j : Nat) (hj : j < width) :
    ∃ u v, w = u ++ [1] ++ v ∧ height xStep u = (j : ℤ) + 1 := by
  simpa only [initialCursor, Nat.zero_testBit, Bool.false_eq_true, false_or,
    Nat.cast_zero, zero_add] using
    words_backward_crossings width (2 * limit + 1) limit (initialCursor limit)
      (initialBoard width limit) hw j hj

/-- The terminal exit raises total block height to `width + 1`, so a block
of displacement `d` is enumerated with width `d - 1`. -/
theorem blockWords_height (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) :
    height xStep w = (width : ℤ) + 1 := by
  obtain ⟨word, hword, rfl⟩ := List.mem_map.mp hw
  rw [height_append, height_singleton, xStep_zero,
    (internalWords_span width limit irreducible hword).1]

/-- Every accepted complete block is a bridge: its internal prefix heights
are at most `width`, strictly below the final height `width + 1`. -/
theorem blockWords_isBridge (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) :
    IsBridge xStep w := by
  have hheight := blockWords_height width limit irreducible hw
  obtain ⟨word, hword, rfl⟩ := List.mem_map.mp hw
  have hspan := internalWords_span width limit irreducible hword
  refine ⟨?_, ?_⟩
  · rw [hheight]
    omega
  · intro u v huv hv
    have hprefix : ∃ rest, word = u ++ rest := by
      rcases List.append_eq_append_iff.mp huv with
        ⟨c, hu, hlast⟩ | ⟨c, hwordEq, _⟩
      · rcases List.singleton_eq_append_iff.mp hlast with
          ⟨hc, _⟩ | ⟨_, hvnil⟩
        · refine ⟨[], ?_⟩
          simpa only [hc, List.append_nil] using hu.symm
        · exact (hv hvnil).elim
      · exact ⟨c, hwordEq⟩
    obtain ⟨rest, hrest⟩ := hprefix
    have hp := hspan.2 u rest hrest
    refine ⟨hp.1, ?_⟩
    rw [hheight]
    omega

/-- The full backward-crossing mask makes every block accepted in irreducible
mode primitive: it cannot be split into two positive-height bridges. -/
theorem blockWords_irreducible (width limit : Nat)
    {w : List Nat} (hw : w ∈ blockWords width limit true) :
    IsIrreducible xStep w := by
  apply backward_crossings_irreducible (blockWords_isBridge width limit true hw)
  intro c hc hcw
  rw [blockWords_height width limit true hw] at hcw
  let j := (c - 1).toNat
  have hjcast : (j : ℤ) = c - 1 := Int.toNat_of_nonneg (by omega)
  have hj : j < width := by omega
  obtain ⟨word, hword, rfl⟩ := List.mem_map.mp hw
  obtain ⟨u, v, huv, hheight⟩ :=
    internalWords_backward_crossings width limit hword j hj
  refine ⟨u, 1, v ++ [0], ?_, ?_, xStep_one⟩
  · rw [huv]
    simp only [List.append_assoc]
  · omega

end RubiksSnake.SlabEnumeration

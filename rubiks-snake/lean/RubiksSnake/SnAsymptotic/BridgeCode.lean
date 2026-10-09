import RubiksSnake.SnAsymptotic.SlabBlockValidity
import RubiksSnake.SnAsymptotic.SlabIrreducible
import RubiksSnake.SnAsymptotic.RenewalBounds
import RubiksSnake.SnAsymptotic.Submultiplicativity

/-!
# Finite codes of geometric bridges

The renewal argument depends on the bridge properties, not on how the blocks
were enumerated. Coefficient certificates may undercount the available blocks.
-/

namespace RubiksSnake.BridgeCode

open CardinalDirections SlabEnumeration

/-- A finite, duplicate-free collection of irreducible `x`-bridges with valid
cardinal-direction geometry, compatible with incoming and final direction `+x`. -/
structure Code where
  words : List (List Nat)
  nodup : words.Nodup
  irreducible : ∀ w ∈ words, BridgeWords.IsIrreducible xStep w
  directions : ∀ w ∈ words,
    (∀ d ∈ w, d < 6) ∧ Compatible 0 (w.map toDirection)
  last : ∀ w ∈ words, w.getLastD 0 = 0
  valid : ∀ w ∈ words, (path zeroVec 0 w).Pairwise interiorDisjoint

/-- Form a code from irreducible slabs with distinct widths and individual
internal-edge cutoffs. Width `d - 1` gives displacement `d`; `k` internal edges
give a block word of length `k + 1`. -/
def ofSlabs (slabs : List (Nat × Nat)) (hwidths : (slabs.map Prod.fst).Nodup) : Code where
  words := slabs.flatMap fun p => blockWords p.1 p.2 true
  nodup := by
    apply List.nodup_flatMap.mpr
    refine ⟨fun p _ => blockWords_nodup _ _ _, ?_⟩
    change (slabs.map Prod.fst).Pairwise (fun a b => a ≠ b) at hwidths
    rw [List.pairwise_map] at hwidths
    apply hwidths.imp
    intro a b hne w hwa hwb
    have ha := blockWords_height a.1 a.2 true hwa
    have hb := blockWords_height b.1 b.2 true hwb
    apply hne
    omega
  irreducible := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact blockWords_irreducible p.1 p.2 hw
  directions := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact blockWords_directions p.1 p.2 true hw
  last := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact blockWords_last p.1 p.2 true hw 0
  valid := by
    intro w hw
    obtain ⟨p, _, hw⟩ := List.mem_flatMap.mp hw
    exact blockWords_valid p.1 p.2 true hw

/-- The codewords with exactly `n` direction letters. -/
def blocks (C : Code) (n : Nat) : List (List Nat) :=
  C.words.filter fun w => w.length == n

/-- The number of length-`n + 1` slab codewords is the sum of the slabs' counts
at internal-edge index `n`, treating indices beyond a cutoff as zero. -/
lemma ofSlabs_blocks_length (slabs : List (Nat × Nat))
    (hwidths : (slabs.map Prod.fst).Nodup) (n : Nat) :
    (blocks (ofSlabs slabs hwidths) (n + 1)).length =
      (slabs.map fun p => (counts p.1 p.2 true)[n]?.getD 0).sum := by
  simp only [blocks, ofSlabs, List.filter_flatMap, List.length_flatMap, blockWords_count_all]

/-- Membership in a length class means being a codeword of that exact length. -/
lemma mem_blocks {C : Code} {n : Nat} {w : List Nat} :
    w ∈ blocks C n ↔ w ∈ C.words ∧ w.length = n := by
  simp [blocks]

/-- Filtering a duplicate-free code by length introduces no repetitions. -/
lemma blocks_nodup (C : Code) (n : Nat) : (blocks C n).Nodup :=
  C.nodup.filter _

/-- Enumerate concatenations of codewords of lengths `1` through `L` having
total direction-word length `n`; length zero contains only the empty word. -/
def language (C : Code) (L n : Nat) : List (List Nat) :=
  if n = 0 then [[]]
  else (List.finRange L).flatMap fun j =>
    if j.val + 1 ≤ n then
      (blocks C (j.val + 1)).flatMap fun b =>
        (language C L (n - (j.val + 1))).map (b ++ ·)
    else []
termination_by n
decreasing_by all_goals omega

/-- The unique zero-length concatenation is the empty word. -/
lemma language_zero (C : Code) (L : Nat) : language C L 0 = [[]] := by
  rw [language]
  simp

/-- A positive-length language word consists of a first block of length
`j + 1`, for `j < L`, and a language tail of the remaining length. -/
lemma mem_language {C : Code} {L n : Nat} (hn : n ≠ 0) {w : List Nat} :
    w ∈ language C L n ↔ ∃ j : Fin L, j.val + 1 ≤ n ∧
      ∃ b ∈ blocks C (j.val + 1), ∃ tail ∈ language C L (n - (j.val + 1)),
        b ++ tail = w := by
  rw [language, if_neg hn]
  constructor
  · intro hw
    obtain ⟨j, _, hw⟩ := List.mem_flatMap.mp hw
    split at hw
    · rename_i hj
      obtain ⟨b, hb, hw⟩ := List.mem_flatMap.mp hw
      obtain ⟨tail, htail, heq⟩ := List.mem_map.mp hw
      exact ⟨j, hj, b, hb, tail, htail, heq⟩
    · simp at hw
  · rintro ⟨j, hj, b, hb, tail, htail, rfl⟩
    apply List.mem_flatMap.mpr
    refine ⟨j, List.mem_finRange j, ?_⟩
    rw [if_pos hj]
    exact List.mem_flatMap.mpr ⟨b, hb, List.mem_map.mpr ⟨tail, htail, rfl⟩⟩

/-- Every enumerated concatenation has the prescribed total number of letters. -/
lemma language_word_length (C : Code) (L n : Nat) :
    ∀ w ∈ language C L n, w.length = n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro w hw
    by_cases hn : n = 0
    · subst n
      simpa [language_zero] using hw
    · obtain ⟨j, hj, b, hb, tail, htail, rfl⟩ := (mem_language hn).mp hw
      rw [List.length_append, (mem_blocks.mp hb).2, ih _ (by omega) tail htail]
      omega

/-- All prefixes of a concatenation stay on or to the right of its initial
`x`-plane. -/
lemma language_nonnegative (C : Code) (L n : Nat) :
    ∀ w ∈ language C L n, BridgeWords.NonnegativePrefixes xStep w := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro w hw
    by_cases hn : n = 0
    · subst n
      have hw' : w = [] := by simpa [language_zero] using hw
      subst w
      intro u v h
      simp only [List.nil_eq_append_iff] at h
      simp [h.1]
    · obtain ⟨j, hj, b, hb, tail, htail, rfl⟩ := (mem_language hn).mp hw
      exact BridgeWords.nonnegative_prefixes_append
        (BridgeWords.bridge_nonnegative_prefixes (C.irreducible b (mem_blocks.mp hb).1).1)
        (ih _ (by omega) tail htail)

/-- Primitive bridge irreducibility uniquely identifies the first block and
tail in equal language concatenations, even for mixed-width codes. -/
lemma language_append_injective {C : Code} {L n m : Nat} {a b x y : List Nat}
    (ha : a ∈ C.words) (hb : b ∈ C.words)
    (hx : x ∈ language C L n) (hy : y ∈ language C L m) (h : a ++ x = b ++ y) :
    a = b ∧ x = y :=
  BridgeWords.irreducible_append_injective (C.irreducible a ha) (C.irreducible b hb)
    (language_nonnegative C L n x hx) (language_nonnegative C L m y hy) h

/-- Unique bridge decoding ensures that enumeration counts each word once. -/
lemma language_nodup (C : Code) (L n : Nat) : (language C L n).Nodup := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    by_cases hn : n = 0
    · subst n
      simp [language_zero]
    · rw [language, if_neg hn]
      apply List.nodup_flatMap.mpr
      constructor
      · intro j _
        split_ifs with hj
        · apply List.nodup_flatMap.mpr
          constructor
          · intro b _
            exact (ih _ (by omega)).map (fun _ _ h => List.append_cancel_left h)
          · apply (blocks_nodup C (j.val + 1)).imp_of_mem
            intro a b ha hb hne w hwa hwb
            obtain ⟨x, hx, heqx⟩ := List.mem_map.mp hwa
            obtain ⟨y, hy, heqy⟩ := List.mem_map.mp hwb
            exact hne (language_append_injective (mem_blocks.mp ha).1 (mem_blocks.mp hb).1
              hx hy (heqx.trans heqy.symm)).1
        · simp
      · apply (List.nodup_finRange L).imp_of_mem
        intro i j _ _ hne w hwi hwj
        dsimp only at hwi hwj
        split at hwi <;> split at hwj <;> try simp_all only [List.not_mem_nil]
        obtain ⟨a, ha, hwi⟩ := List.mem_flatMap.mp hwi
        obtain ⟨b, hb, hwj⟩ := List.mem_flatMap.mp hwj
        obtain ⟨x, hx, heqx⟩ := List.mem_map.mp hwi
        obtain ⟨y, hy, heqy⟩ := List.mem_map.mp hwj
        have hab := (language_append_injective (mem_blocks.mp ha).1 (mem_blocks.mp hb).1
          hx hy (heqx.trans heqy.symm)).1
        have hla := (mem_blocks.mp ha).2
        have hlb := (mem_blocks.mp hb).2
        apply hne
        apply Fin.ext
        subst b
        omega

/-- Concatenated codewords use cardinal, perpendicular successive directions
and have pairwise interior-disjoint wedges from the standard incoming `+x` frame. -/
lemma language_geometry (C : Code) (L n : Nat) :
    ∀ w ∈ language C L n,
      (∀ d ∈ w, d < 6) ∧ Compatible 0 (w.map toDirection) ∧
        (path zeroVec 0 w).Pairwise interiorDisjoint := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro w hw
    by_cases hn : n = 0
    · subst n
      have hw' : w = [] := by simpa [language_zero] using hw
      subst w
      simp [Compatible]
    · obtain ⟨j, hj, b, hb, tail, htail, rfl⟩ := (mem_language hn).mp hw
      have hbcode := (mem_blocks.mp hb).1
      obtain ⟨hsmall, hcompatible, hvalid⟩ := ih _ (by omega) tail htail
      obtain ⟨hbsmall, hbcompatible⟩ := C.directions b hbcode
      refine ⟨?_, compatible_concat hbcompatible hcompatible (C.last b hbcode), ?_⟩
      · intro d hd
        rcases List.mem_append.mp hd with hd | hd
        · exact hbsmall d hd
        · exact hsmall d hd
      · exact bridge_concat_valid (C.irreducible b hbcode).1
          (language_nonnegative C L _ tail htail) (C.last b hbcode) (C.valid b hbcode) hvalid

/-- Encoding injects length-`n` bridge concatenations into valid formulas with
`n` rotations, hence `n + 1` wedges. -/
theorem language_length_le (C : Code) (L n : Nat) :
    (language C L n).length ≤ countValidFormulas n := by
  have hnodup : ((language C L n).map encodeWord).Nodup := by
    change ((language C L n).map encodeWord).Pairwise (fun a b => a ≠ b)
    rw [List.pairwise_map]
    apply (language_nodup C L n).imp_of_mem
    intro a b ha hb hne heq
    have hga := language_geometry C L n a ha
    have hgb := language_geometry C L n b hb
    exact hne (encodeWord_injective hga.1 hgb.1 hga.2.1 hgb.2.1 heq)
  have hsub :
      ((language C L n).map encodeWord).toFinset ⊆ (validRotationLists n).toFinset := by
    intro rs hrs
    obtain ⟨w, hw, rfl⟩ := List.mem_map.mp (List.mem_toFinset.mp hrs)
    apply List.mem_toFinset.mpr
    obtain ⟨_, hcompatible, hvalid⟩ := language_geometry C L n w hw
    have hv := (mem_validRotationLists (encodeWord w)).mpr
      (encodeWord_valid hcompatible (language_nonnegative C L n w hw) hvalid)
    simpa [language_word_length C L n w hw] using hv
  calc
    (language C L n).length = ((language C L n).map encodeWord).length :=
      (List.length_map ..).symm
    _ = ((language C L n).map encodeWord).toFinset.card :=
      (List.toFinset_card_of_nodup hnodup).symm
    _ ≤ (validRotationLists n).toFinset.card := Finset.card_le_card hsub
    _ = (validRotationLists n).length := List.toFinset_card_of_nodup (validRotationLists_nodup n)
    _ = countValidFormulas n := fastCountValidFormulas_eq n

/-- For positive `n`, count words by first-block length `j + 1` and multiply
the number of such blocks by the number of possible tails. -/
theorem language_length_rec (C : Code) (L n : Nat) (hn : n ≠ 0) :
    (language C L n).length = ∑ j : Fin L,
      if j.val + 1 ≤ n then (blocks C (j.val + 1)).length *
        (language C L (n - (j.val + 1))).length else 0 := by
  rw [language, if_neg hn, List.length_flatMap]
  have hinner (j : Fin L) :
      ((if j.val + 1 ≤ n then
        (blocks C (j.val + 1)).flatMap fun b =>
          (language C L (n - (j.val + 1))).map (b ++ ·)
      else []) : List (List Nat)).length =
        if j.val + 1 ≤ n then (blocks C (j.val + 1)).length *
          (language C L (n - (j.val + 1))).length else 0 := by
    split_ifs <;> simp [List.length_flatMap]
  simp_rw [hinner]
  rw [← List.ofFn_eq_map, List.sum_ofFn]

/-- Any one admissible first-block length contributes a lower bound on the
total number of language words. -/
lemma language_length_ge_term (C : Code) (L n : Nat)
    (j : Fin L) (hj : j.val + 1 ≤ n) :
    (blocks C (j.val + 1)).length * (language C L (n - (j.val + 1))).length ≤
      (language C L n).length := by
  rw [language_length_rec C L n (by omega)]
  have h := Finset.single_le_sum
    (f := fun i : Fin L =>
      if i.val + 1 ≤ n then (blocks C (i.val + 1)).length *
        (language C L (n - (i.val + 1))).length else 0)
    (fun i _ => Nat.zero_le _) (Finset.mem_univ j)
  simpa only [if_pos hj] using h

/-- Available blocks of lengths two and three produce at least one word of
every total length `n >= 2`, provided both lengths are allowed by `L`. -/
lemma language_length_pos (C : Code) (L : Nat) (hL : 2 < L)
    (htwo : 1 ≤ (blocks C 2).length) (hthree : 1 ≤ (blocks C 3).length)
    (n : Nat) (hn : 2 ≤ n) : 1 ≤ (language C L n).length := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
    by_cases htwo' : n = 2
    · subst n
      have h := language_length_ge_term C L 2 ⟨1, by omega⟩ (by simp)
      exact htwo.trans (by simpa [language_zero] using h)
    · by_cases hthree' : n = 3
      · subst n
        have h := language_length_ge_term C L 3 ⟨2, hL⟩ (by simp)
        exact hthree.trans (by simpa [language_zero] using h)
      · have h := language_length_ge_term C L n ⟨1, by omega⟩ (by simp; omega)
        have hi := ih (n - 2) (by omega) (by omega)
        have hprod : 1 ≤ (blocks C 2).length * (language C L (n - 2)).length := by
          simpa using Nat.mul_le_mul htwo hi
        exact hprod.trans h

/-- A renewal polynomial certificate at `0 < q <= 4` proves the pointwise
bound `q ^ k ≤ countValidFormulas k`. Coefficient `j` may undercount
length-`j + 1` blocks; lengths two and three supply the small base cases. -/
theorem pow_le_countValidFormulas (C : Code) (L : Nat) (hL : 2 < L)
    (coeff : Fin L → Nat)
    (hcoeff : ∀ j, coeff j ≤ (blocks C (j.val + 1)).length)
    (htwo : 1 ≤ (blocks C 2).length) (hthree : 1 ≤ (blocks C 3).length)
    (q : ℝ) (hq : 0 < q) (hqfour : q ≤ 4)
    (hpoly : q ^ L ≤ ∑ j : Fin L, (coeff j : ℝ) * q ^ (L - (j.val + 1)))
    (k : Nat) : q ^ k ≤ countValidFormulas k := by
  let a : ℝ := 1 / 4 ^ (L + 2)
  have habase (k : Nat) (hk : k < L + 2) : a * q ^ k ≤ 1 := by
    have hp : q ^ k ≤ (4 : ℝ) ^ (L + 2) :=
      (pow_le_pow_left₀ hq.le hqfour k).trans
        (pow_le_pow_right₀ (by norm_num) (by omega))
    calc
      a * q ^ k ≤ a * 4 ^ (L + 2) :=
        mul_le_mul_of_nonneg_left hp (by dsimp [a]; positivity)
      _ = 1 := by dsimp [a]; field_simp
  have hbound (k : Nat) (hk : 2 ≤ k) :
      a * q ^ k ≤ (language C L k).length := by
    apply renewal_ge_of_finite_certificate (fun n => (language C L n).length)
      L 2 coeff a q (by dsimp [a]; positivity) hq.le hpoly ?_ ?_ k hk
    · intro k hk hsmall
      exact (habase k (by omega)).trans
        (by exact_mod_cast language_length_pos C L hL htwo hthree k hk)
    · intro k hk
      have hn : (∑ j : Fin L, coeff j * (language C L (k - (j.val + 1))).length) ≤
          (language C L k).length := by
        rw [language_length_rec C L k (by omega)]
        apply Finset.sum_le_sum
        intro j _
        rw [if_pos (by omega : j.val + 1 ≤ k)]
        exact Nat.mul_le_mul_right _ (hcoeff j)
      exact_mod_cast hn
  apply pow_le_countValidFormulas_of_pointwise a q (by dsimp [a]; positivity) hq
  intro n
  by_cases hn : 2 ≤ n
  · exact (hbound n hn).trans (by exact_mod_cast language_length_le C L n)
  · exact (habase n (by omega)).trans (by exact_mod_cast countValidFormulas_pos n)

end RubiksSnake.BridgeCode

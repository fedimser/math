import RubiksSnake.CapCode
import RubiksSnake.CapAssemblyCertificate
import RubiksSnake.FastLowerBound

/-! Combine the cap families with the earlier short bridge code, excluding overlap by
longitudinal displacement and word length. -/

namespace RubiksSnake.CapAssembly

open SlabEnumeration BridgeWords BridgeSymmetry CapEnumeration

attribute [local irreducible] QuarterSlab.seedWords

/-- A disjoint finite family of codes is again a code. -/
def sumCode {α : Type*} (indices : List α) (codes : α → BridgeCode.Code)
    (hd : indices.Pairwise fun a b => List.Disjoint (codes a).words (codes b).words) :
    BridgeCode.Code where
  words := indices.flatMap fun i => (codes i).words
  nodup := List.nodup_flatMap.mpr ⟨fun i _ => (codes i).nodup, hd⟩
  irreducible := by
    intro w hw
    obtain ⟨i, _, hi⟩ := List.mem_flatMap.mp hw
    exact (codes i).irreducible w hi
  directions := by
    intro w hw
    obtain ⟨i, _, hi⟩ := List.mem_flatMap.mp hw
    exact (codes i).directions w hi
  last := by
    intro w hw
    obtain ⟨i, _, hi⟩ := List.mem_flatMap.mp hw
    exact (codes i).last w hi
  valid := by
    intro w hw
    obtain ⟨i, _, hi⟩ := List.mem_flatMap.mp hw
    exact (codes i).valid w hi

/-- All accepted lengths at a fixed longitudinal width, after the overlap cutoff. -/
def widthCode (width cutoff : Nat) (hw : width ≤ 4) : BridgeCode.Code :=
  sumCode ((List.range 65).filter fun n => cutoff < n)
    (fun n => bothAt width 14 n (by decide) hw) (by
      apply ((show (List.range 65).Nodup from List.nodup_range).filter _).imp
      intro a b hne w ha hb
      have h1 := (bothAt_metadata width 14 a (by decide) hw ha).1
      have h2 := (bothAt_metadata width 14 b (by decide) hw hb).1
      exact hne (h1.symm.trans h2))

lemma widthCode_metadata (width cutoff : Nat) (hw : width ≤ 4)
    {w : List Nat} (hmem : w ∈ (widthCode width cutoff hw).words) :
    height xStep w = (width : ℤ) + 1 ∧ cutoff < w.length ∧ w.length < 65 := by
  obtain ⟨n, hn, hw'⟩ := List.mem_flatMap.mp hmem
  have hmeta := bothAt_metadata width 14 n (by decide) hw hw'
  have hmem := List.mem_filter.mp hn
  simp only [decide_eq_true_eq, List.mem_range] at hmem
  exact ⟨hmeta.2, by omega, by omega⟩

lemma bothAt_blocks_length (width n m : Nat) (hw : width ≤ 4) :
    (BridgeCode.blocks (bothAt width 14 n (by decide) hw) m).length =
      if n = m then 2 * (traces width 14 n).length else 0 := by
  by_cases h : n = m
  · subst m
    rw [if_pos rfl, BridgeCode.blocks, List.filter_eq_self.mpr]
    · exact bothAt_length width 14 n (by decide) hw
    · intro w hw'
      simpa using (bothAt_metadata width 14 n (by decide) hw hw').1
  · rw [if_neg h, BridgeCode.blocks]
    have hnil : ((bothAt width 14 n (by decide) hw).words.filter
        fun w => w.length == m) = [] := by
      apply List.filter_eq_nil_iff.mpr
      intro w hw'
      simp [(bothAt_metadata width 14 n (by decide) hw hw').1, h]
    rw [hnil]
    rfl

/-- Filtering by length leaves exactly the corresponding doubled cap catalogue. -/
theorem widthCode_blocks_length (width cutoff n : Nat) (hw : width ≤ 4) (hn : n < 65) :
    (BridgeCode.blocks (widthCode width cutoff hw) n).length =
      if cutoff < n then 2 * (traces width 14 n).length else 0 := by
  change ((((List.range 65).filter fun k => cutoff < k).flatMap
    fun k => (bothAt width 14 k (by decide) hw).words).filter (fun w => w.length == n)).length = _
  rw [List.filter_flatMap, List.length_flatMap]
  change (((List.range 65).filter fun k => cutoff < k).map
    fun k => (BridgeCode.blocks (bothAt width 14 k (by decide) hw) n).length).sum = _
  simp_rw [bothAt_blocks_length]
  rw [List.sum_map_eq_nsmul_single n _ (by intro k hk _; simp [hk])]
  by_cases hc : cutoff < n
  · simp [hc, List.count_filter, List.count_range, hn]
  · have hz : List.count n ((List.range 65).filter fun k => cutoff < k) = 0 := by
      apply List.count_eq_zero.mpr
      simp [hc]
    simp [hc, hz]

lemma widthCode_disjoint (a b ca cb : Nat) (ha : a ≤ 4) (hb : b ≤ 4) (hne : a ≠ b) :
    List.Disjoint (widthCode a ca ha).words (widthCode b cb hb).words := by
  intro w hwa hwb
  have h1 := (widthCode_metadata a ca ha hwa).1
  have h2 := (widthCode_metadata b cb hb hwb).1
  omega

private def twoThreeCode : BridgeCode.Code :=
  appendCode (widthCode 2 19 (by decide)) (widthCode 3 22 (by decide))
    (widthCode_disjoint 2 3 19 22 (by decide) (by decide) (by decide))

private theorem twoThree_four_disjoint :
    List.Disjoint twoThreeCode.words (widthCode 4 0 (by decide)).words := by
  intro w hw hfour
  rcases List.mem_append.mp hw with htwo | hthree
  · exact widthCode_disjoint 2 4 19 0 (by decide) (by decide) (by decide) htwo hfour
  · exact widthCode_disjoint 3 4 22 0 (by decide) (by decide) (by decide) hthree hfour

/-- The three new widths have distinct longitudinal displacements. -/
def newCode : BridgeCode.Code :=
  appendCode twoThreeCode (widthCode 4 0 (by decide)) twoThree_four_disjoint

private lemma rotate_metadata (C : BridgeCode.Code) (P : Nat → ℤ → Prop)
    (hp : ∀ w ∈ C.words, P w.length (height xStep w)) :
    ∀ w ∈ (rotate C).words, P w.length (height xStep w) := by
  intro w hw
  obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
  rw [height_turnWord v (C.directions v hv).1]
  simpa only [turnWord, List.length_map] using hp v hv

private lemma fourfold_metadata (C : BridgeCode.Code) (hh : Heading C 2)
    (P : Nat → ℤ → Prop) (hp : ∀ w ∈ C.words, P w.length (height xStep w)) :
    ∀ w ∈ (fourfold C hh).words, P w.length (height xStep w) := by
  have h1 := rotate_metadata C P hp
  have h2 := rotate_metadata (rotate C) P h1
  have h3 := rotate_metadata (rotate (rotate C)) P h2
  intro w hw
  rcases List.mem_append.mp hw with hw | hw
  · rcases List.mem_append.mp hw with hw | hw
    · rcases List.mem_append.mp hw with hw | hw
      · exact hp w hw
      · exact h1 w hw
    · exact h2 w hw
  · exact h3 w hw

/-- Only the two original widths that overlap a new width need a length cutoff. -/
lemma fastCode_metadata {w : List Nat} (hw : w ∈ FastLower.code.words) :
    height xStep w = 1 ∨ height xStep w = 2 ∨
      (height xStep w = 3 ∧ w.length ≤ 19) ∨
      (height xStep w = 4 ∧ w.length ≤ 22) := by
  apply fourfold_metadata
    (QuarterSlab.ofSlabs [(0, 19), (1, 19), (2, 17), (3, 20)] (by decide))
    (QuarterSlab.ofSlabs_heading _ _) (fun n h => h = 1 ∨ h = 2 ∨
    (h = 3 ∧ n ≤ 19) ∨ (h = 4 ∧ n ≤ 22)) ?_ w hw
  intro v hv
  change v ∈ List.flatMap (fun p : Nat × Nat => QuarterSlab.seedWords p.1 p.2)
    [(0, 19), (1, 19), (2, 17), (3, 20)] at hv
  rw [List.mem_flatMap] at hv
  obtain ⟨p, hp, hv⟩ := hv
  have hs := QuarterSlab.seedWords_subset p.1 p.2 hv
  have hh := blockWords_height p.1 (p.2 + 1) true hs
  have hl := blockWords_lengths p.1 (p.2 + 1) true hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl <;> simp only at hh hl <;> omega

/-- The old and new code lists are disjoint, without comparing or enumerating their words. -/
theorem old_new_disjoint : List.Disjoint FastLower.code.words newCode.words := by
  intro w hold hnew
  have ho := fastCode_metadata hold
  rcases List.mem_append.mp hnew with hnew | hnew
  · rcases List.mem_append.mp hnew with hnew | hnew
    · have hn := widthCode_metadata 2 19 (by decide) hnew
      norm_num at hn
      omega
    · have hn := widthCode_metadata 3 22 (by decide) hnew
      norm_num at hn
      omega
  · have hn := widthCode_metadata 4 0 (by decide) hnew
    norm_num at hn
    omega

/-- The complete finite bridge code used in the improved lower bound. -/
def combinedCode : BridgeCode.Code := appendCode FastLower.code newCode old_new_disjoint

lemma appendCode_blocks_length (C D : BridgeCode.Code) (hd : List.Disjoint C.words D.words)
    (n : Nat) :
    (BridgeCode.blocks (appendCode C D hd) n).length =
      (BridgeCode.blocks C n).length + (BridgeCode.blocks D n).length := by
  simp only [BridgeCode.blocks, appendCode, List.filter_append, List.length_append]

/-- Exact length classes of the union, with the two overlap cutoffs made explicit. -/
theorem combinedCode_blocks_length (n : Nat) (hn : n < 65) (hpos : 0 < n) :
    (BridgeCode.blocks combinedCode n).length =
      (BridgeCode.blocks FastLower.code n).length +
        2 * ((if 19 < n then (traces 2 14 n).length else 0) +
          (if 22 < n then (traces 3 14 n).length else 0) +
          (traces 4 14 n).length) := by
  rw [combinedCode, appendCode_blocks_length, newCode, appendCode_blocks_length, twoThreeCode]
  simp only [appendCode_blocks_length,
    widthCode_blocks_length _ _ n _ hn, if_pos hpos]
  split_ifs <;> omega

end RubiksSnake.CapAssembly

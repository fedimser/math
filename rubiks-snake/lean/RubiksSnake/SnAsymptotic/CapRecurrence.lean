import RubiksSnake.SnAsymptotic.CapAssembly
import Mathlib.Algebra.BigOperators.Group.List.Basic

/-! Local recurrence certificates undercount the duplicate-free semantic assembly. -/

namespace RubiksSnake.CapAssembly

open CapEnumeration

/-- Group a tag-dependent sum by the exact histogram of the catalogue. -/
theorem histogram_sum (size : Nat) (ws : List (List Nat × Nat))
    (hb : ∀ p ∈ ws, p.2 < size) (weight : Nat → Nat) :
    (ws.map fun p => weight p.2).sum =
      ((List.range size).map fun tag => histogram tag ws * weight tag).sum := by
  induction ws with
  | nil => simp [histogram]
  | cons p ws ih =>
    have hp : p.2 < size := hb p (by simp)
    have ht : ∀ q ∈ ws, q.2 < size := fun q hq => hb q (by simp [hq])
    have hh (tag : Nat) : histogram tag (p :: ws) =
        (if p.2 = tag then 1 else 0) + histogram tag ws := by
      rw [show p :: ws = [p] ++ ws from rfl, histogram_append, histogram_singleton]
    simp only [List.map_cons, List.sum_cons, ih ht, hh, Nat.add_mul, List.sum_map_add]
    congr 1
    rw [List.sum_map_eq_nsmul_single p.2 _ (by
      intro tag hne _
      simp [Ne.symm hne])]
    simp [List.count_range, hp]

/-- A dense histogram gives the same weighted sum as its semantic catalogue. -/
theorem weightedCounts_eq (counts : Array Nat) (ws : List (List Nat × Nat))
    (hb : ∀ p ∈ ws, p.2 < counts.size)
    (hc : ∀ tag, tag < counts.size → counts[tag]! = histogram tag ws)
    (weight : Nat → Nat) :
    weightedCounts counts weight = (ws.map fun p => weight p.2).sum := by
  rw [histogram_sum counts.size ws hb weight]
  unfold weightedCounts
  apply congrArg List.sum
  apply List.map_congr_left
  intro tag ht
  rw [hc tag (List.mem_range.mp ht)]

/-- All head tags are inside the finite coefficient layout. -/
lemma heads_tag_lt {width limit : Nat} (hl : 0 < limit)
    {p : List Nat × Nat} (hp : p ∈ heads width limit) :
    p.2 < (width + 1) * 2 ^ width * (limit + 1) := by
  obtain ⟨r, y, _, hp⟩ := mem_heads hp
  exact (leavesAt_tag_spec width r limit 0 y true hl (Nat.zero_le _) hp).1

/-- All middle tags are inside the same layout. -/
lemma middles_tag_lt {width limit start : Nat} (hl : 0 < limit) (hx : start ≤ width)
    {p : List Nat × Nat} (hp : p ∈ middles width limit start) :
    p.2 < (width + 1) * 2 ^ width * (limit + 1) := by
  obtain ⟨r, hp⟩ := mem_middles hp
  exact (leavesAt_tag_spec width r limit start 0 false hl hx hp).1

/-- Head coefficients can weight any tag-dependent continuation. -/
lemma weighted_heads (width limit : Nat) (hl : 0 < limit) (weight : Nat → Nat) :
    weightedCounts (headCounts width 2 limit) weight =
      ((heads width limit).map fun p => weight p.2).sum := by
  apply weightedCounts_eq
  · intro p hp
    simpa using heads_tag_lt hl hp
  · intro tag ht
    exact headCounts_get width limit tag (by simpa using ht)

/-- Middle coefficients can weight any tag-dependent continuation. -/
lemma weighted_middles (width limit start : Nat) (hl : 0 < limit) (hx : start ≤ width)
    (weight : Nat → Nat) :
    weightedCounts (middleCounts width 2 limit start) weight =
      ((middles width limit start).map fun p => weight p.2).sum := by
  apply weightedCounts_eq
  · intro p hp
    simpa using middles_tag_lt hl hx hp
  · intro tag ht
    exact middleCounts_get width limit start tag (by simpa using ht)

private lemma filter_length {α : Type*} (ws : List α) (p : α → Bool) :
    (ws.filter p).length = (ws.map fun w => if p w then 1 else 0).sum := by
  induction ws with
  | nil => simp
  | cons w ws ih =>
    cases h : p w <;> simp [h, ih, Nat.add_comm]

/-- Terminal acceptance depends only on the decoded coefficient tag. -/
lemma terminal_length (width limit n start mask : Nat) (hl : 0 < limit) :
    (terminal width limit n start mask).length =
      weightedCounts (headCounts width 2 limit) (terminalWeight width limit n start mask) := by
  rw [weighted_heads width limit hl, terminal, List.length_map, filter_length]
  apply congrArg List.sum
  apply List.map_congr_left
  intro p hp
  simp only [terminalWeight, (head_spec hl hp).2.2.2]

/-- Summing a local dense recurrence is exactly summing the semantic choices. -/
lemma suffixStep_eq (width limit n start mask : Nat) (hl : 0 < limit) (hx : start ≤ width)
    (value : Nat → Nat → Nat → Nat) :
    suffixStep width limit n start mask (headCounts width 2 limit)
      (middleCounts width 2 limit start) value =
    (terminal width limit n start mask).length +
      ((middles width limit start).map fun p =>
        if p.1.length + 1 ≤ n then
          value (n - (p.1.length + 1)) (tagFinish width limit p.2)
            (mask ||| tagMask width limit p.2)
        else 0).sum := by
  rw [suffixStep, ← terminal_length width limit n start mask hl,
    weighted_middles width limit start hl hx]
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro p hp
  simp [middleWeight, (middle_spec hl hx hp).2.2.2]

/-- A local subsolution is enough; no trust in an imperative dynamic-programming implementation
is required. -/
def Subsolution (width limit degree : Nat) (value : Nat → Nat → Nat → Nat) : Prop :=
  ∀ n, n ≤ degree → ∀ start, start ≤ width → ∀ mask, mask < 2 ^ width →
    value n start mask ≤
      suffixStep width limit n start mask (headCounts width 2 limit)
        (middleCounts width 2 limit start) value

/-- Induction on remaining length converts local recurrence checks into true suffix counts. -/
theorem suffixes_underestimate (width limit degree : Nat) (hl : 0 < limit)
    (value : Nat → Nat → Nat → Nat) (hvalue : Subsolution width limit degree value)
    (n start mask : Nat) (hn : n ≤ degree) (hx : start ≤ width) (hm : mask < 2 ^ width) :
    value n start mask ≤ (suffixes width limit n start mask).length := by
  induction n using Nat.strong_induction_on generalizing start mask with
  | h n ih =>
    apply (hvalue n hn start hx mask hm).trans
    rw [suffixStep_eq width limit n start mask hl hx, suffixes, List.length_append,
      List.length_flatMap]
    apply Nat.add_le_add_left
    apply List.sum_le_sum
    intro p hp
    split_ifs with hlen
    · simp only [List.length_map]
      apply ih _ (by omega) _ _ (by omega) (middle_spec hl hx hp).2.2.1
      exact Nat.or_lt_two_pow hm (Nat.mod_lt _ (by positivity))
    · simp

/-- The initial head step undercounts the number of distinct accepted complete traces. -/
theorem traces_underestimate (width limit degree : Nat) (hl : 0 < limit)
    (value : Nat → Nat → Nat → Nat) (hvalue : Subsolution width limit degree value)
    (n : Nat) (hn : n ≤ degree) :
    weightedCounts (headCounts width 2 limit)
      (fun tag => headWeight width limit n tag value) ≤ (traces width limit n).length := by
  rw [weighted_heads width limit hl, traces, List.length_flatMap]
  apply List.sum_le_sum
  intro p hp
  simp only [headWeight, (head_spec hl hp).2.2.2]
  simp only [Nat.zero_lt_succ, true_and]
  split_ifs with hlen
  · simp only [List.length_map]
    apply suffixes_underestimate width limit degree hl value hvalue _ _ _ (by omega)
      (head_spec hl hp).2.2.1
    exact Nat.mod_lt _ (by positivity)
  · simp

end RubiksSnake.CapAssembly

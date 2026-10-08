import RubiksSnake.CapCounts

/-! Semantic head--middle--tail assembly. The lists are specifications, not computations. -/

namespace RubiksSnake.CapAssembly

open SlabEnumeration BridgeWords CapEnumeration CapPieces CapReversal

/-- Terminal choices preserve their original head tag; reflection changes the cut mask. -/
def terminal (width limit n start mask : Nat) : List (List (List Nat)) :=
  ((heads width limit).filter fun p =>
    tagFinish width limit p.2 + start == width &&
      p.1.length + 1 == n &&
      (mask ||| reflectMask width (tagMask width limit p.2)) == 2 ^ width - 1).map
    (fun p => [tailWord p.1])

/-- Suffix traces consist of zero or more middle bridges and exactly one tail. -/
def suffixes (width limit n start mask : Nat) : List (List (List Nat)) :=
  terminal width limit n start mask ++
    (middles width limit start).flatMap fun p =>
      if p.1.length + 1 ≤ n then
        (suffixes width limit (n - (p.1.length + 1))
          (tagFinish width limit p.2) (mask ||| tagMask width limit p.2)).map
          (fun t => (p.1 ++ [0]) :: t)
      else []
termination_by n
decreasing_by omega

/-- Complete traces start with a head and end with a suffix accepted with every cut crossed. -/
def traces (width limit n : Nat) : List (List (List Nat)) :=
  (heads width limit).flatMap fun p =>
    if p.1.length + 1 ≤ n then
      (suffixes width limit (n - (p.1.length + 1))
        (tagFinish width limit p.2) (tagMask width limit p.2)).map
        (fun t => (p.1 ++ [0]) :: t)
    else []

/-- Head catalogue membership supplies valid parameters for a unique short-piece search. -/
lemma mem_heads {width limit : Nat} {p : List Nat × Nat}
    (hp : p ∈ heads width limit) :
    ∃ r y, y ≤ r ∧ (p.1, p.2) ∈ leavesAt width r limit 0 y true := by
  obtain ⟨⟨r, y⟩, hparam, hp⟩ := List.mem_flatMap.mp hp
  exact ⟨r, y, (by simpa [headParameters] using
    (show ∀ q ∈ headParameters, q.2 ≤ q.1 from by decide) (r, y) hparam), hp⟩

/-- Middle catalogue membership has entry transverse layer zero. -/
lemma mem_middles {width limit start : Nat} {p : List Nat × Nat}
    (hp : p ∈ middles width limit start) :
    ∃ r, (p.1, p.2) ∈ leavesAt width r limit start 0 false := by
  obtain ⟨⟨r, y⟩, hparam, hp⟩ := List.mem_flatMap.mp hp
  have hy : y = 0 := (show ∀ q ∈ middleParameters, q.2 = 0 from by decide) (r, y) hparam
  subst y
  exact ⟨r, hp⟩

/-- Decoded head coefficients carry exactly the geometry of their counted words. -/
lemma head_spec {width limit : Nat} (hl : 0 < limit)
    {p : List Nat × Nat} (hp : p ∈ heads width limit) :
    Spec width 0 (tagFinish width limit p.2) (tagMask width limit p.2) 2 0 (p.1 ++ [0]) ∧
      IsHead xStep (p.1 ++ [0]) ∧
      tagFinish width limit p.2 ≤ width ∧ tagLength limit p.2 = p.1.length + 1 := by
  obtain ⟨r, y, hy, hp⟩ := mem_heads hp
  have ht := leavesAt_tag_spec width r limit 0 y true hl (Nat.zero_le _) hp
  have hs := leavesAt_spec width r limit 0 y true (Nat.zero_le _) hy hp
  have ha := (leaves_tag width r (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit y true) ⟨0, 0⟩ (initialBoard r limit) hp).2
  rw [ht.2.1, ht.2.2.1]
  exact ⟨hs, leavesAt_head width r limit 0 y true hy hp,
    crossRun_le width ⟨0, 0⟩ p.1 (Nat.zero_le _) ha, ht.2.2.2⟩

/-- The middle coefficient metadata agrees with the word's actual entry and exit. -/
lemma middle_spec {width limit start : Nat} (hl : 0 < limit) (hx : start ≤ width)
    {p : List Nat × Nat} (hp : p ∈ middles width limit start) :
    Spec width start (tagFinish width limit p.2) (tagMask width limit p.2) 0 0 (p.1 ++ [0]) ∧
      IsIrreducible xStep (p.1 ++ [0]) ∧
      tagFinish width limit p.2 ≤ width ∧ tagLength limit p.2 = p.1.length + 1 := by
  obtain ⟨r, hp⟩ := mem_middles hp
  have ht := leavesAt_tag_spec width r limit start 0 false hl hx hp
  have hs := leavesAt_spec width r limit start 0 false hx (Nat.zero_le _) hp
  have ha := (leaves_tag width r (2 * limit + 1) limit (limit - 1)
    (pieceCursor limit 0 false) ⟨start, 0⟩ (initialBoard r limit) hp).2
  rw [ht.2.1, ht.2.2.1]
  exact ⟨hs, leavesAt_middle width r limit start hp,
    crossRun_le width ⟨start, 0⟩ p.1 hx ha, ht.2.2.2⟩

/-- A reversed head is a terminal cap beginning at the complementary longitudinal layer. -/
lemma tail_spec {width limit : Nat} (hl : 0 < limit) (hw : width ≤ 4)
    {p : List Nat × Nat} (hp : p ∈ heads width limit) :
    Spec width ((width : ℤ) - tagFinish width limit p.2) ((width : ℤ) + 1)
      (reflectMask width (tagMask width limit p.2)) 0 2 (tailWord p.1) ∧
      IsTail xStep (tailWord p.1) := by
  obtain ⟨r, y, hy, hp⟩ := mem_heads hp
  have ht := leavesAt_tag_spec width r limit 0 y true hl (Nat.zero_le _) hp
  rw [ht.2.1, ht.2.2.1]
  exact ⟨leavesAt_tail_spec width r limit y hw hy hp,
    catalogue_tail width r limit y hy hp⟩

/-- Every suffix has the canonical middle/tail factorization and nonnegative prefixes. -/
theorem suffixes_shape (width limit n start mask : Nat) (hl : 0 < limit)
    (hw : width ≤ 4) (hx : start ≤ width)
    {t : List (List Nat)} (ht : t ∈ suffixes width limit n start mask) :
    ∃ pieces tail, t = pieces ++ [tail] ∧
      (∀ p ∈ pieces, IsIrreducible xStep p) ∧ IsTail xStep tail := by
  induction n using Nat.strong_induction_on generalizing start mask t with
  | h n ih =>
    rw [suffixes] at ht
    rcases List.mem_append.mp ht with ht | ht
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.mp ht
      have hp' := (List.mem_filter.mp hp).1
      exact ⟨[], tailWord p.1, rfl, by simp, (tail_spec hl hw hp').2⟩
    · obtain ⟨p, hp, ht⟩ := List.mem_flatMap.mp ht
      split_ifs at ht with hlen
      · obtain ⟨rest, hrest, rfl⟩ := List.mem_map.mp ht
        have hp' := middle_spec hl hx hp
        obtain ⟨pieces, tail, rfl, hpieces, htail⟩ :=
          ih (n - (p.1.length + 1)) (by omega) _ _ hp'.2.2.1 hrest
        refine ⟨(p.1 ++ [0]) :: pieces, tail, rfl, ?_, htail⟩
        intro q hq
        rcases List.mem_cons.mp hq with rfl | hq
        · exact hp'.2.1
        · exact hpieces q hq
      · simp at ht

/-- A suffix cannot be empty because it contains its terminal piece. -/
theorem suffixes_ne_nil (width limit n start mask : Nat) (hl : 0 < limit)
    (hw : width ≤ 4) (hx : start ≤ width)
    {t : List (List Nat)} (ht : t ∈ suffixes width limit n start mask) : t ≠ [] := by
  obtain ⟨pieces, tail, rfl, _, _⟩ := suffixes_shape width limit n start mask hl hw hx ht
  simp

/-- Canonical suffixes have nonnegative transverse prefixes. -/
theorem suffixes_nonnegative (width limit n start mask : Nat) (hl : 0 < limit)
    (hw : width ≤ 4) (hx : start ≤ width)
    {t : List (List Nat)} (ht : t ∈ suffixes width limit n start mask) :
    NonnegativePrefixes xStep t.flatten := by
  obtain ⟨pieces, tail, rfl, hp, ht⟩ := suffixes_shape width limit n start mask hl hw hx ht
  simpa using bridges_tail_nonnegative (fun p h => (hp p h).1) ht.1

/-- A suffix's own cut witnesses, together with its input mask, cover every cut. -/
theorem suffixes_spec (width limit n start mask : Nat) (hl : 0 < limit)
    (hw : width ≤ 4) (hx : start ≤ width)
    {t : List (List Nat)} (ht : t ∈ suffixes width limit n start mask) :
    ∃ ownMask, mask ||| ownMask = 2 ^ width - 1 ∧
      Spec width start ((width : ℤ) + 1) ownMask 0 2 t.flatten ∧ t.flatten.length = n := by
  induction n using Nat.strong_induction_on generalizing start mask t with
  | h n ih =>
    rw [suffixes] at ht
    rcases List.mem_append.mp ht with ht | ht
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.mp ht
      obtain ⟨hp, hguard⟩ := List.mem_filter.mp hp
      simp only [Bool.and_eq_true, beq_iff_eq] at hguard
      have he : (width : ℤ) - tagFinish width limit p.2 = start := by omega
      have hs := (tail_spec hl hw hp).1
      rw [he] at hs
      exact ⟨_, hguard.2, by simpa using hs, by simpa [tailWord] using hguard.1.2⟩
    · obtain ⟨p, hp, ht⟩ := List.mem_flatMap.mp ht
      split_ifs at ht with hlen
      · obtain ⟨rest, hrest, rfl⟩ := List.mem_map.mp ht
        have hp' := middle_spec hl hx hp
        obtain ⟨own, hmask, hs, hlength⟩ :=
          ih (n - (p.1.length + 1)) (by omega) _ _ hp'.2.2.1 hrest
        have hn := suffixes_nonnegative width limit _ _ _ hl hw hp'.2.2.1 hrest
        refine ⟨tagMask width limit p.2 ||| own, ?_, ?_, ?_⟩
        · simpa only [Nat.or_assoc] using hmask
        · simpa only [List.flatten_cons] using
            spec_append hp'.1 hs (bridge_upperCap hp'.2.1.1) hn
        · simp only [List.flatten_cons, List.length_append, List.length_singleton]
          omega
      · simp at ht

/-- Every accepted complete trace is a valid irreducible longitudinal bridge. -/
theorem traces_spec (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4)
    {t : List (List Nat)} (ht : t ∈ traces width limit n) :
    Spec width 0 ((width : ℤ) + 1) (2 ^ width - 1) 2 2 t.flatten ∧
      IsIrreducible longStep t.flatten ∧ t.flatten.length = n := by
  obtain ⟨p, hp, ht⟩ := List.mem_flatMap.mp ht
  split_ifs at ht with hlen
  · obtain ⟨rest, hrest, rfl⟩ := List.mem_map.mp ht
    have hp' := head_spec hl hp
    obtain ⟨own, hmask, hs, hlength⟩ :=
      suffixes_spec width limit _ _ _ hl hw hp'.2.2.1 hrest
    have hn := suffixes_nonnegative width limit _ _ _ hl hw hp'.2.2.1 hrest
    have hall := spec_append hp'.1 hs hp'.2.1.1 hn
    rw [hmask] at hall
    refine ⟨hall, spec_irreducible hall, ?_⟩
    simp only [List.flatten_cons, List.length_append, List.length_singleton]
    omega
  · simp at ht

/-- Equal words determine their complete canonical trace, even when the caps overhang. -/
theorem traces_flatten_injective {width limit n m : Nat} (hl : 0 < limit) (hw : width ≤ 4)
    {a b : List (List Nat)} (ha : a ∈ traces width limit n) (hb : b ∈ traces width limit m)
    (heq : a.flatten = b.flatten) : a = b := by
  obtain ⟨p, hp, ha⟩ := List.mem_flatMap.mp ha
  split_ifs at ha with hpn
  · obtain ⟨restA, hrestA, rfl⟩ := List.mem_map.mp ha
    obtain ⟨q, hq, hb⟩ := List.mem_flatMap.mp hb
    split_ifs at hb with hqm
    · obtain ⟨restB, hrestB, rfl⟩ := List.mem_map.mp hb
      have hp' := head_spec hl hp
      have hq' := head_spec hl hq
      obtain ⟨pieces, tail, rfl, hpieces, htail⟩ :=
        suffixes_shape width limit _ _ _ hl hw hp'.2.2.1 hrestA
      obtain ⟨other, final, rfl, hother, hfinal⟩ :=
        suffixes_shape width limit _ _ _ hl hw hq'.2.2.1 hrestB
      obtain ⟨hh, hm, ht⟩ := caps_factorization_injective hp'.2.1 hq'.2.1
        hpieces hother htail hfinal (by simpa using heq)
      rw [hh, hm, ht]
    · simp at hb
  · simp at ha

/-- Reversed heads give distinct singleton tail traces. -/
theorem terminal_nodup (width limit n start mask : Nat) :
    (terminal width limit n start mask).Nodup := by
  unfold terminal
  apply (((heads_nodup width limit).of_map Prod.fst).filter _).map_on
  intro p hp q hq heq
  apply List.inj_on_of_nodup_map (heads_nodup width limit)
    (List.mem_filter.mp hp).1 (List.mem_filter.mp hq).1
  apply tailWord_injective
  simpa using heq

/-- The semantic recurrence counts each suffix trace exactly once. -/
theorem suffixes_nodup (width limit n start mask : Nat) (hl : 0 < limit)
    (hw : width ≤ 4) (hx : start ≤ width) :
    (suffixes width limit n start mask).Nodup := by
  induction n using Nat.strong_induction_on generalizing start mask with
  | h n ih =>
    rw [suffixes]
    apply List.nodup_append.mpr
    refine ⟨terminal_nodup width limit n start mask, ?_, ?_⟩
    · apply List.nodup_flatMap.mpr
      constructor
      · intro p hp
        split_ifs with hlen
        · exact (ih _ (by omega) _ _ (middle_spec hl hx hp).2.2.1).map
            (fun _ _ h => (List.cons.inj h).2)
        · simp
      · apply ((middles_nodup width limit start).of_map Prod.fst).imp_of_mem
        intro p q hp hq hne t hpt hqt
        dsimp only at hpt hqt
        split at hpt <;> split at hqt <;> try simp_all only [List.not_mem_nil]
        obtain ⟨a, _, hea⟩ := List.mem_map.mp hpt
        obtain ⟨b, _, heb⟩ := List.mem_map.mp hqt
        apply hne
        apply List.inj_on_of_nodup_map (middles_nodup width limit start) hp hq
        exact List.append_cancel_right (List.cons.inj (hea.trans heb.symm)).1
    · intro t ht t' ht' hett
      obtain ⟨p, _, heq⟩ := List.mem_map.mp ht
      obtain ⟨q, hq, hqt⟩ := List.mem_flatMap.mp ht'
      split_ifs at hqt with hlen
      · obtain ⟨rest, hrest, herest⟩ := List.mem_map.mp hqt
        have hh := congrArg List.length (heq.trans (hett.trans herest.symm))
        have hr : rest = [] := by
          cases rest with
          | nil => rfl
          | cons a rest => simp at hh
        exact suffixes_ne_nil width limit _ _ _ hl hw (middle_spec hl hx hq).2.2.1 hrest hr
      · simp at hqt

/-- Every complete trace is enumerated once. -/
theorem traces_nodup (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) :
    (traces width limit n).Nodup := by
  unfold traces
  apply List.nodup_flatMap.mpr
  constructor
  · intro p hp
    split_ifs with hlen
    · exact (suffixes_nodup width limit _ _ _ hl hw (head_spec hl hp).2.2.1).map
        (fun _ _ h => (List.cons.inj h).2)
    · simp
  · apply ((heads_nodup width limit).of_map Prod.fst).imp_of_mem
    intro p q hp hq hne t hpt hqt
    dsimp only at hpt hqt
    split at hpt <;> split at hqt <;> try simp_all only [List.not_mem_nil]
    obtain ⟨a, _, hea⟩ := List.mem_map.mp hpt
    obtain ⟨b, _, heb⟩ := List.mem_map.mp hqt
    apply hne
    apply List.inj_on_of_nodup_map (heads_nodup width limit) hp hq
    exact List.append_cancel_right (List.cons.inj (hea.trans heb.symm)).1

/-- Flattening the traces neither identifies nor repeats any bridge word. -/
theorem words_nodup (width limit n : Nat) (hl : 0 < limit) (hw : width ≤ 4) :
    ((traces width limit n).map List.flatten).Nodup :=
  (traces_nodup width limit n hl hw).map_on
    (fun _ ha _ hb heq => traces_flatten_injective hl hw ha hb heq)

end RubiksSnake.CapAssembly

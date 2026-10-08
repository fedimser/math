import RubiksSnake.BridgeWords

/-!
# Canonical caps for separated bridge pieces

A head may dip below its initial level, provided its exit is above all its
occupied levels. A tail may rise above its final level, provided it never
dips below its entry. Irreducible caps make the intervening bridge cuts unique.
-/

namespace RubiksSnake.BridgeWords

variable {α : Type*} {step : α → ℤ}

/-- All proper prefix heights are strictly below the exit height. -/
def IsUpperCap (step : α → ℤ) (w : List α) : Prop :=
  ∀ u v, w = u ++ v → v ≠ [] → height step u < height step w

/-- A head cap with no earlier separating exit followed by a bridge. -/
def IsHead (step : α → ℤ) (w : List α) : Prop :=
  IsUpperCap step w ∧ w ≠ [] ∧
    ¬ ∃ a b, IsUpperCap step a ∧ a ≠ [] ∧ IsBridge step b ∧ w = a ++ b

/-- A tail cap with no initial bridge followed by a nonnegative suffix. -/
def IsTail (step : α → ℤ) (w : List α) : Prop :=
  NonnegativePrefixes step w ∧
    ¬ ∃ a b, IsBridge step a ∧ NonnegativePrefixes step b ∧ w = a ++ b

/-- A nonempty upper cap has strictly positive displacement. -/
lemma upperCap_height_pos {w : List α} (h : IsUpperCap step w) (hne : w ≠ []) :
    0 < height step w := by
  simpa using h [] w (by simp) hne

/-- A bridge is an upper cap. -/
lemma bridge_upperCap {w : List α} (h : IsBridge step w) : IsUpperCap step w :=
  fun u v huv hv => (h.2 u v huv hv).2

/-- An upper cap followed by a bridge is again an upper cap. -/
lemma upperCap_append {a b : List α} (ha : IsUpperCap step a)
    (hb : IsBridge step b) : IsUpperCap step (a ++ b) := by
  intro u v huv hv
  rcases List.append_eq_append_iff.mp huv with
    ⟨c, huc, hbc⟩ | ⟨c, hac, _⟩
  · have hc := (hb.2 c v hbc hv).2
    simp only [huc, height_append]
    omega
  · by_cases hc : c = []
    · have hau : a = u := by simpa [hc] using hac
      rw [height_append, hau]
      have := hb.1
      omega
    · have hu := ha u c hac hc
      rw [height_append]
      have := hb.1
      omega

/-- Backward crossings of every positive internal cut make an upper cap a head. -/
theorem head_of_backward_crossings {w : List α}
    (hw : IsUpperCap step w) (hne : w ≠ [])
    (hcross : ∀ c : ℤ, 0 < c → c < height step w →
      ∃ (u : List α) (t : α) (v : List α),
        w = u ++ [t] ++ v ∧ height step u = c ∧ step t = -1) :
    IsHead step w := by
  refine ⟨hw, hne, ?_⟩
  rintro ⟨a, b, ha, hane, hb, hab⟩
  have ha0 := upperCap_height_pos ha hane
  have hcut : height step a < height step w := by
    rw [hab, height_append]
    have := hb.1
    omega
  obtain ⟨u, t, v, hwt, huc, ht⟩ := hcross (height step a) ha0 hcut
  have heq : (u ++ [t]) ++ v = a ++ b := by
    simpa only [List.append_assoc] using hwt.symm.trans hab
  rcases List.append_eq_append_iff.mp heq with
    ⟨d, had, _⟩ | ⟨d, hud, hbd⟩
  · have hlt := ha u ([t] ++ d)
      (by simpa only [List.append_assoc] using had) (by simp)
    omega
  · have hd0 := bridge_nonnegative_prefixes hb d v hbd
    have hh := congrArg (height step) hud
    simp only [height_append, height_singleton, huc, ht] at hh
    omega

/-- A tail's final level is included among the cuts that must be crossed backwards. -/
theorem tail_of_backward_crossings {w : List α}
    (hw : NonnegativePrefixes step w)
    (hcross : ∀ c : ℤ, 0 < c → c ≤ height step w →
      ∃ (u : List α) (t : α) (v : List α),
        w = u ++ [t] ++ v ∧ height step u = c ∧ step t = -1) :
    IsTail step w := by
  refine ⟨hw, ?_⟩
  rintro ⟨a, b, ha, hb, hab⟩
  have hcut : height step a ≤ height step w := by
    have := hb b [] (by simp)
    rw [hab, height_append]
    omega
  obtain ⟨u, t, v, hwt, huc, ht⟩ := hcross (height step a) ha.1 hcut
  have heq : (u ++ [t]) ++ v = a ++ b := by
    simpa only [List.append_assoc] using hwt.symm.trans hab
  rcases List.append_eq_append_iff.mp heq with
    ⟨d, had, _⟩ | ⟨d, hud, hbd⟩
  · have hlt := (ha.2 u ([t] ++ d)
      (by simpa only [List.append_assoc] using had) (by simp)).2
    omega
  · have hd0 := hb d v hbd
    have hh := congrArg (height step) hud
    simp only [height_append, height_singleton, huc, ht] at hh
    omega

/-- A head prefix of another head exhausts it when the following tail stays nonnegative. -/
private lemma head_suffix_eq_nil {a b x : List α}
    (ha : IsUpperCap step a) (hane : a ≠ [])
    (hab : IsHead step (a ++ b)) (hbx : NonnegativePrefixes step (b ++ x)) :
    b = [] := by
  by_contra hb
  apply hab.2.2
  refine ⟨a, b, ha, hane, ⟨?_, ?_⟩, rfl⟩
  · have hlt := hab.1 a b rfl hb
    rw [height_append] at hlt
    omega
  · intro u v huv hv
    refine ⟨hbx u (v ++ x) (by rw [huv, List.append_assoc]), ?_⟩
    have hlt := hab.1 (a ++ u) v (by rw [huv, List.append_assoc]) hv
    simp only [height_append] at hlt
    omega

/-- Nonnegative continuations determine the first head uniquely, even with overhangs. -/
theorem head_append_injective {a b x y : List α}
    (ha : IsHead step a) (hb : IsHead step b)
    (hx : NonnegativePrefixes step x) (hy : NonnegativePrefixes step y)
    (h : a ++ x = b ++ y) : a = b ∧ x = y := by
  rcases List.append_eq_append_iff.mp h with
    ⟨c, hbc, hxc⟩ | ⟨c, hac, hyc⟩
  · have hc : c = [] := head_suffix_eq_nil ha.1 ha.2.1 (hbc ▸ hb) (hxc ▸ hx)
    exact ⟨by simpa only [hc, List.append_nil] using hbc.symm,
      by simpa only [hc, List.nil_append] using hxc⟩
  · have hc : c = [] := head_suffix_eq_nil hb.1 hb.2.1 (hac ▸ ha) (hyc ▸ hy)
    exact ⟨by simpa only [hc, List.append_nil] using hac,
      by simpa only [hc, List.nil_append] using hyc.symm⟩

/-- A tail cannot be confused with a bridge followed by a nonnegative continuation. -/
theorem tail_ne_bridge_append {tail b rest : List α}
    (ht : IsTail step tail) (hb : IsBridge step b)
    (hr : NonnegativePrefixes step rest) : tail ≠ b ++ rest :=
  fun h => ht.2 ⟨b, rest, hb, hr, h⟩

/-- Any list of bridges followed by a lower cap stays nonnegative. -/
lemma bridges_tail_nonnegative {pieces : List (List α)} {tail : List α}
    (hp : ∀ p ∈ pieces, IsBridge step p) (ht : NonnegativePrefixes step tail) :
    NonnegativePrefixes step (pieces.flatten ++ tail) := by
  induction pieces with
  | nil => simpa using ht
  | cons p pieces ih =>
    have hfirst := bridge_nonnegative_prefixes (hp p (by simp))
    have hrest := ih (fun q hq => hp q (by simp [hq]))
    simpa only [List.flatten_cons, List.append_assoc] using
      nonnegative_prefixes_append hfirst hrest

/-- Irreducible middle bridges and the terminal cap are uniquely decoded. -/
theorem middle_tail_injective {pieces other : List (List α)} {tail final : List α}
    (hp : ∀ p ∈ pieces, IsIrreducible step p)
    (ho : ∀ p ∈ other, IsIrreducible step p)
    (ht : IsTail step tail) (hf : IsTail step final)
    (h : pieces.flatten ++ tail = other.flatten ++ final) :
    pieces = other ∧ tail = final := by
  induction pieces generalizing other with
  | nil =>
    cases other with
    | nil => exact ⟨rfl, by simpa using h⟩
    | cons b other =>
      have hn := bridges_tail_nonnegative (pieces := other)
        (fun p hp => (ho p (by simp [hp])).1) hf.1
      exact (tail_ne_bridge_append ht (ho b (by simp)).1 hn
        (by simpa only [List.flatten_nil, List.nil_append, List.flatten_cons,
          List.append_assoc] using h)).elim
  | cons a pieces ih =>
    cases other with
    | nil =>
      have hn := bridges_tail_nonnegative (pieces := pieces)
        (fun p hp' => (hp p (by simp [hp'])).1) ht.1
      exact (tail_ne_bridge_append hf (hp a (by simp)).1 hn
        (by simpa only [List.flatten_nil, List.nil_append, List.flatten_cons,
          List.append_assoc] using h.symm)).elim
    | cons b other =>
      have hpa : ∀ p ∈ pieces, IsIrreducible step p :=
        fun p hmem => hp p (by simp [hmem])
      have hob : ∀ p ∈ other, IsIrreducible step p :=
        fun p hmem => ho p (by simp [hmem])
      have ha := hp a (by simp)
      have hb := ho b (by simp)
      have hleft := bridges_tail_nonnegative (fun p hmem => (hpa p hmem).1) ht.1
      have hright := bridges_tail_nonnegative (fun p hmem => (hob p hmem).1) hf.1
      have heq : a ++ (pieces.flatten ++ tail) = b ++ (other.flatten ++ final) := by
        simpa only [List.flatten_cons, List.append_assoc] using h
      obtain ⟨hab, hrest⟩ := irreducible_append_injective ha hb hleft hright heq
      obtain ⟨hpieces, htail⟩ := ih hpa hob hrest
      exact ⟨by rw [hab, hpieces], htail⟩

/-- The whole head-middle-tail factorization is injective on canonical pieces. -/
theorem caps_factorization_injective {head first tail final : List α}
    {pieces other : List (List α)}
    (hh : IsHead step head) (hh' : IsHead step first)
    (hp : ∀ p ∈ pieces, IsIrreducible step p)
    (ho : ∀ p ∈ other, IsIrreducible step p)
    (ht : IsTail step tail) (hf : IsTail step final)
    (h : head ++ (pieces.flatten ++ tail) = first ++ (other.flatten ++ final)) :
    head = first ∧ pieces = other ∧ tail = final := by
  have hleft := bridges_tail_nonnegative (fun p hmem => (hp p hmem).1) ht.1
  have hright := bridges_tail_nonnegative (fun p hmem => (ho p hmem).1) hf.1
  obtain ⟨hhead, hrest⟩ := head_append_injective hh hh' hleft hright h
  exact ⟨hhead, middle_tail_injective hp ho ht hf hrest⟩

end RubiksSnake.BridgeWords

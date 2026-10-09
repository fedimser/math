import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic

/-!
# Bridges in words

Bridge words have positive total height and all proper prefix heights lie
between zero and the final height, strictly below the latter. Irreducible
bridges are uniquely determined when followed by nonnegative-prefix tails.
A backward step across every internal integer cut certifies irreducibility.
-/

namespace RubiksSnake.BridgeWords

variable {α : Type*}

/-- Total integer displacement of a word under the given letter increments. -/
def height (step : α → ℤ) (w : List α) : ℤ :=
  (w.map step).sum

/-- A word of positive final height whose proper prefixes stay nonnegative and
strictly below that final height. -/
def IsBridge (step : α → ℤ) (w : List α) : Prop :=
  0 < height step w ∧
    ∀ u v, w = u ++ v → v ≠ [] →
      0 ≤ height step u ∧ height step u < height step w

/-- Every prefix, including the whole word, has nonnegative height. -/
def NonnegativePrefixes (step : α → ℤ) (w : List α) : Prop :=
  ∀ u v, w = u ++ v → 0 ≤ height step u

/-- A bridge that cannot be split into two bridges of positive height. -/
def IsIrreducible (step : α → ℤ) (w : List α) : Prop :=
  IsBridge step w ∧
    ¬ ∃ a b, IsBridge step a ∧ IsBridge step b ∧ w = a ++ b

/-- The empty word contributes no height displacement. -/
@[simp] lemma height_nil (step : α → ℤ) : height step [] = 0 := rfl

/-- A one-letter word has exactly that letter's height increment. -/
@[simp] lemma height_singleton (step : α → ℤ) (t : α) :
    height step [t] = step t := by
  simp [height]

/-- Height displacement is additive under concatenation. -/
@[simp] lemma height_append (step : α → ℤ) (a b : List α) :
    height step (a ++ b) = height step a + height step b := by
  simp [height, List.sum_append]

variable {step : α → ℤ} {w a b x y : List α}

/-- Positive final height rules out an empty bridge. -/
lemma bridge_not_nil (hw : IsBridge step w) : w ≠ [] := by
  rintro rfl
  exact Int.lt_irrefl 0 hw.1

/-- A bridge has nonnegative heights at all prefixes, including its endpoint. -/
lemma bridge_nonnegative_prefixes (hw : IsBridge step w) :
    NonnegativePrefixes step w := by
  intro u v huv
  by_cases hv : v = []
  · simpa only [huv, hv, List.append_nil] using Int.le_of_lt hw.1
  · exact (hw.2 u v huv hv).1

/-- Concatenating two words with nonnegative prefixes preserves that property. -/
lemma nonnegative_prefixes_append
    (ha : NonnegativePrefixes step a) (hb : NonnegativePrefixes step b) :
    NonnegativePrefixes step (a ++ b) := by
  intro u v huv
  rcases List.append_eq_append_iff.mp huv with
    ⟨c, huc, hbc⟩ | ⟨c, hac, _⟩
  · have ha0 := ha a [] (List.append_nil a).symm
    have hc0 := hb c v hbc
    rw [huc, height_append]
    omega
  · exact ha u c hac

/-- Concatenating bridges gives a bridge whose final height is the sum of theirs. -/
lemma bridge_append (ha : IsBridge step a) (hb : IsBridge step b) :
    IsBridge step (a ++ b) := by
  have ha0 := ha.1
  have hb0 := hb.1
  refine ⟨?_, ?_⟩
  · rw [height_append]
    omega
  · intro u v huv hv
    refine ⟨nonnegative_prefixes_append (bridge_nonnegative_prefixes ha)
      (bridge_nonnegative_prefixes hb) u v huv, ?_⟩
    rcases List.append_eq_append_iff.mp huv with
      ⟨c, huc, hbc⟩ | ⟨c, hac, _⟩
    · have hc := (hb.2 c v hbc hv).2
      simp only [huc, height_append]
      omega
    · by_cases hc : c = []
      · have hau : a = u := by simpa only [hc, List.append_nil] using hac
        rw [height_append, hau]
        omega
      · have hu := (ha.2 u c hac hc).2
        rw [height_append]
        omega

/-- A bridge prefix of an irreducible bridge exhausts it if the remaining word,
followed by `x`, has nonnegative prefixes. -/
private lemma irreducible_suffix_eq_nil
    (ha : IsBridge step a) (hab : IsIrreducible step (a ++ b))
    (hbx : NonnegativePrefixes step (b ++ x)) : b = [] := by
  by_contra hb
  apply hab.2
  refine ⟨a, b, ha, ⟨?_, ?_⟩, rfl⟩
  · have hlt := (hab.1.2 a b rfl hb).2
    rw [height_append] at hlt
    omega
  · intro u v huv hv
    refine ⟨hbx u (v ++ x) (by rw [huv, List.append_assoc]), ?_⟩
    have hlt := (hab.1.2 (a ++ u) v (by rw [huv, List.append_assoc]) hv).2
    simp only [height_append] at hlt
    omega

/-- Equal concatenations of irreducible bridges and nonnegative-prefix tails
have identical first blocks and tails; no prefix-free assumption is needed. -/
lemma irreducible_append_injective
    (ha : IsIrreducible step a) (hb : IsIrreducible step b)
    (hx : NonnegativePrefixes step x) (hy : NonnegativePrefixes step y)
    (h : a ++ x = b ++ y) : a = b ∧ x = y := by
  rcases List.append_eq_append_iff.mp h with
    ⟨c, hbc, hxc⟩ | ⟨c, hac, hyc⟩
  · have hc : c = [] :=
      irreducible_suffix_eq_nil ha.1 (hbc ▸ hb) (hxc ▸ hx)
    exact ⟨by simpa only [hc, List.append_nil] using hbc.symm,
      by simpa only [hc, List.nil_append] using hxc⟩
  · have hc : c = [] :=
      irreducible_suffix_eq_nil hb.1 (hac ▸ ha) (hyc ▸ hy)
    exact ⟨by simpa only [hc, List.append_nil] using hac,
      by simpa only [hc, List.nil_append] using hyc.symm⟩

/-- A bridge is irreducible if it takes a step of height `-1` from every integer
height strictly between zero and its final height. -/
lemma backward_crossings_irreducible
    (hw : IsBridge step w)
    (hcross : ∀ c : ℤ, 0 < c → c < height step w →
      ∃ (u : List α) (t : α) (v : List α),
        w = u ++ [t] ++ v ∧ height step u = c ∧ step t = -1) :
    IsIrreducible step w := by
  refine ⟨hw, ?_⟩
  rintro ⟨a, b, ha, hb, hab⟩
  have hcut : height step a < height step w := by
    rw [hab, height_append]
    have hb0 := hb.1
    omega
  obtain ⟨u, t, v, hwt, huc, ht⟩ :=
    hcross (height step a) ha.1 hcut
  have heq : (u ++ [t]) ++ v = a ++ b := by
    simpa only [List.append_assoc] using hwt.symm.trans hab
  rcases List.append_eq_append_iff.mp heq with
    ⟨d, had, _⟩ | ⟨d, hud, hbd⟩
  · have hlt := (ha.2 u ([t] ++ d)
      (by simpa only [List.append_assoc] using had) (by simp)).2
    omega
  · have hd0 := bridge_nonnegative_prefixes hb d v hbd
    have hh := congrArg (height step) hud
    simp only [height_append, height_singleton, ht] at hh
    omega

end RubiksSnake.BridgeWords

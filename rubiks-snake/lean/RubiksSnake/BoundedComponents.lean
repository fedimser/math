import RubiksSnake.SnAsymptotic_MuExistence
import RubiksSnake.CardinalDirections
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Topology.Algebra.InfiniteSum.Real

/-!
# Exponential-rate bounds for a bounded number of components

The convolution `componentCount m n` counts ordered tuples of `m` independently
valid rotation words with total rotation length `n`, allowing empty words.
The corresponding wedge count is `componentWedgeCount m n`; each component
has one more wedge than rotations. Initial cardinal frames can be normalized
without changing validity.
For fixed `m`, a polynomial factor does not prevent summability after division
by `q ^ n` when `q` exceeds the snake growth constant.

These counts encode no placements, mutual-disjointness constraints, or joining
construction. The results are rate comparisons, not a geometric multi-strand
construction or a new numerical lower bound.
-/

open Filter Set Topology

namespace RubiksSnake

noncomputable section

open CardinalDirections in
/-- A fixed initial frame and position do not increase the number of valid paths. -/
theorem card_directionalPaths_le_countValidFormulas (p : Vec3)
    {previous incoming : Direction} (hprevious : Perpendicular previous incoming) (n : ℕ) :
    Nat.card {ds : Fin n → Direction //
      Compatible incoming (List.ofFn ds) ∧
        (directionalPath p previous (incoming :: List.ofFn ds)).Pairwise interiorDisjoint} ≤
      countValidFormulas n := by
  let encodeWord :
      {ds : Fin n → Direction //
        Compatible incoming (List.ofFn ds) ∧
          (directionalPath p previous (incoming :: List.ofFn ds)).Pairwise interiorDisjoint} →
      {w : Formula n // Valid w} := fun ds =>
    ⟨formulaOfList (encode previous incoming (List.ofFn ds.val)) (by simp), by
      unfold Valid
      rw [ofFn_formulaOfList]
      exact (valid_encode_iff p hprevious ds.property.1).mpr ds.property.2⟩
  change Nat.card _ ≤ Nat.card {w : Formula n // Valid w}
  apply Nat.card_le_card_of_injective encodeWord
  intro a b h
  have hcodes := congrArg (fun w : {w : Formula n // Valid w} => List.ofFn w.val) h
  simp only [encodeWord, ofFn_formulaOfList] at hcodes
  exact Subtype.ext (List.ofFn_injective
    (encode_injOn hprevious a.property.1 b.property.1 hcodes))

/-- Above the growth constant, one uniform prefactor bounds every word length.
The prefactor can be chosen at least one. -/
theorem exists_countValidFormulas_le_mul_pow {r : ℝ}
    (hr : snakeGrowthConstant < r) :
    ∃ C : ℝ, 1 ≤ C ∧
      ∀ n : ℕ, (countValidFormulas n : ℝ) ≤ C * r ^ n := by
  have hrpos : 0 < r := snakeGrowthConstant_pos.trans hr
  have heventually :
      ∀ᶠ n : ℕ in atTop, (countValidFormulas n : ℝ) ≤ r ^ n := by
    filter_upwards [tendsto_logValidFormulaCount_div.eventually
      (gt_mem_nhds (Real.log_lt_log snakeGrowthConstant_pos hr)),
      eventually_ge_atTop 1] with n hlog hn
    have hnpos : (0 : ℝ) < n := by
      exact_mod_cast (show 0 < n by omega)
    have hle : logValidFormulaCount n ≤ (n : ℝ) * Real.log r := by
      simpa only [mul_comm] using (div_le_iff₀ hnpos).mp hlog.le
    calc
      (countValidFormulas n : ℝ) = Real.exp (logValidFormulaCount n) := by
        rw [logValidFormulaCount,
          Real.exp_log (by exact_mod_cast countValidFormulas_pos n)]
      _ ≤ Real.exp ((n : ℝ) * Real.log r) := Real.exp_le_exp.mpr hle
      _ = r ^ n := by rw [Real.exp_nat_mul, Real.exp_log hrpos]
  obtain ⟨N, hN⟩ := eventually_atTop.mp heventually
  have hnonneg (k : ℕ) :
      0 ≤ (countValidFormulas k : ℝ) / r ^ k :=
    div_nonneg (Nat.cast_nonneg _) (pow_nonneg hrpos.le _)
  let C : ℝ := 1 + ∑ k ∈ Finset.range N, (countValidFormulas k : ℝ) / r ^ k
  have hC : 1 ≤ C :=
    le_add_of_nonneg_right (Finset.sum_nonneg fun k _ => hnonneg k)
  refine ⟨C, hC, fun n => ?_⟩
  by_cases hn : N ≤ n
  · calc
      (countValidFormulas n : ℝ) ≤ r ^ n := hN n hn
      _ ≤ C * r ^ n := by
        simpa only [one_mul] using
          mul_le_mul_of_nonneg_right hC (pow_nonneg hrpos.le n)
  · apply (div_le_iff₀ (pow_pos hrpos n)).mp
    calc
      (countValidFormulas n : ℝ) / r ^ n ≤
          ∑ k ∈ Finset.range N, (countValidFormulas k : ℝ) / r ^ k :=
        Finset.single_le_sum (fun k _ => hnonneg k)
          (Finset.mem_range.mpr (Nat.lt_of_not_ge hn))
      _ ≤ C := le_add_of_nonneg_left zero_le_one

/-- The positive-prefactor form of the uniform exponential upper bound. -/
theorem exists_pos_countValidFormulas_le_mul_pow {r : ℝ}
    (hr : snakeGrowthConstant < r) :
    ∃ C : ℝ, 0 < C ∧
      ∀ n : ℕ, (countValidFormulas n : ℝ) ≤ C * r ^ n := by
  obtain ⟨C, hC, hbound⟩ := exists_countValidFormulas_le_mul_pow hr
  exact ⟨C, zero_lt_one.trans_le hC, hbound⟩

/-- Ordered convolution count for independent valid components.
Lengths are rotation-word lengths, and empty components are allowed. -/
def componentCount : ℕ → ℕ → ℕ
  | 0, n => if n = 0 then 1 else 0
  | m + 1, n =>
      ∑ k ∈ Finset.range (n + 1), countValidFormulas k * componentCount m (n - k)

/-- There is one tuple with no components and zero total rotations, and none at other lengths. -/
@[simp] theorem componentCount_zero (n : ℕ) :
    componentCount 0 n = if n = 0 then 1 else 0 := rfl

/-- Split off the first component and sum over all possible rotation lengths it can have. -/
theorem componentCount_succ (m n : ℕ) :
    componentCount (m + 1) n =
      ∑ k ∈ Finset.range (n + 1), countValidFormulas k * componentCount m (n - k) := rfl

/-- Enumerate ordered tuples of independently valid words with prescribed component count
and total rotation length; no mutual placement or collision constraint is imposed. -/
def componentLanguage : ℕ → ℕ → List (List (List Rotation))
  | 0, n => if n = 0 then [[]] else []
  | m + 1, n =>
      (List.finRange (n + 1)).flatMap fun k =>
        (validRotationLists k.val).flatMap fun w =>
          (componentLanguage m (n - k.val)).map (w :: ·)

/-- The explicit component-language list has exactly the recursively defined convolution count. -/
theorem componentLanguage_length (m n : ℕ) :
    (componentLanguage m n).length = componentCount m n := by
  induction m generalizing n with
  | zero =>
      by_cases hn : n = 0 <;> simp [componentLanguage, componentCount, hn]
  | succ m ih =>
      rw [componentLanguage, List.length_flatMap, componentCount_succ]
      have hinner (k : Fin (n + 1)) :
          ((validRotationLists k.val).flatMap fun w =>
            (componentLanguage m (n - k.val)).map (w :: ·)).length =
              countValidFormulas k.val * componentCount m (n - k.val) := by
        have hcount : (validRotationLists k.val).length = countValidFormulas k.val :=
          fastCountValidFormulas_eq k.val
        simp [List.length_flatMap, ih, hcount]
      simp_rw [hinner]
      rw [← List.ofFn_eq_map, List.sum_ofFn]
      exact (Finset.sum_range
        (fun k => countValidFormulas k * componentCount m (n - k))).symm

/-- Every tuple of valid words with the prescribed number of components and total rotations
occurs in the component-language enumeration. -/
lemma mem_componentLanguage_of_valid (words : List (List Rotation)) (m n : ℕ)
    (hlen : words.length = m) (hvalid : ∀ w ∈ words, ValidList w)
    (hsize : (words.map List.length).sum = n) :
    words ∈ componentLanguage m n := by
  induction words generalizing m n with
  | nil =>
      simp only [List.length_nil] at hlen
      simp only [List.map_nil, List.sum_nil] at hsize
      subst m
      subst n
      simp [componentLanguage]
  | cons word words ih =>
      cases m with
      | zero => simp at hlen
      | succ m =>
          have hlen' : words.length = m := by simpa using hlen
          have hsize' : word.length + (words.map List.length).sum = n := by
            simpa using hsize
          have hword : word ∈ validRotationLists word.length :=
            (mem_validRotationLists word).mpr (hvalid word (by simp))
          have htail : words ∈ componentLanguage m (n - word.length) :=
            ih m (n - word.length) hlen'
              (fun w hw => hvalid w (List.mem_cons_of_mem word hw)) (by omega)
          rw [componentLanguage]
          apply List.mem_flatMap.mpr
          refine ⟨⟨word.length, by omega⟩, List.mem_finRange _, ?_⟩
          exact List.mem_flatMap.mpr
            ⟨word, hword, List.mem_map.mpr ⟨words, htail, rfl⟩⟩

/-- Any finite family of distinct valid component tuples is bounded by the convolution count
when both component count and total rotation length are fixed. -/
theorem card_componentFamily_le (m n : ℕ) (family : Finset (List (List Rotation)))
    (hfamily : ∀ words ∈ family, words.length = m ∧
      (∀ w ∈ words, ValidList w) ∧ (words.map List.length).sum = n) :
    family.card ≤ componentCount m n := by
  have hsub : family ⊆ (componentLanguage m n).toFinset := by
    intro words hwords
    obtain ⟨hlen, hvalid, hsize⟩ := hfamily words hwords
    exact List.mem_toFinset.mpr
      (mem_componentLanguage_of_valid words m n hlen hvalid hsize)
  calc
    family.card ≤ (componentLanguage m n).toFinset.card := Finset.card_le_card hsub
    _ ≤ (componentLanguage m n).length := List.toFinset_card_le _
    _ = componentCount m n := componentLanguage_length m n

/-- The same ordered tuples, indexed by their total number of wedges. -/
def componentWedgeCount (m n : ℕ) : ℕ :=
  if m ≤ n then componentCount m (n - m) else 0

/-- Count ordered tuples with at most `M` independent components and exactly `n` wedges in total. -/
def boundedComponentWedgeCount (M n : ℕ) : ℕ :=
  ∑ m ∈ Finset.range (M + 1), componentWedgeCount m n

/-- Enumerate the same bounded-component tuples using total wedge count, including the empty
tuple only at total length zero. -/
def boundedComponentLanguage (M n : ℕ) : List (List (List Rotation)) :=
  (List.finRange (M + 1)).flatMap fun m =>
    if m.val ≤ n then componentLanguage m.val (n - m.val) else []

/-- The bounded-component list realizes the sum of the individual component-count convolutions. -/
theorem boundedComponentLanguage_length (M n : ℕ) :
    (boundedComponentLanguage M n).length = boundedComponentWedgeCount M n := by
  rw [boundedComponentLanguage, List.length_flatMap, boundedComponentWedgeCount]
  have hinner (m : Fin (M + 1)) :
      (if m.val ≤ n then componentLanguage m.val (n - m.val) else []).length =
        componentWedgeCount m.val n := by
    split_ifs <;> simp_all [componentWedgeCount, componentLanguage_length]
  simp_rw [hinner]
  rw [← List.ofFn_eq_map, List.sum_ofFn]
  exact (Finset.sum_range (fun m => componentWedgeCount m n)).symm

/-- Total wedge count is total rotation count plus one wedge for each component. -/
lemma sum_component_wedge_lengths (words : List (List Rotation)) :
    (words.map (fun w => w.length + 1)).sum =
      (words.map List.length).sum + words.length := by
  induction words with
  | nil => rfl
  | cons word words ih =>
      simp only [List.map_cons, List.sum_cons, List.length_cons, ih]
      omega

/-- A valid tuple belongs to the bounded language once its component count and total wedges
satisfy the specified bounds. -/
lemma mem_boundedComponentLanguage_of_valid (words : List (List Rotation)) (M n : ℕ)
    (hlen : words.length ≤ M) (hvalid : ∀ w ∈ words, ValidList w)
    (hsize : (words.map (fun w => w.length + 1)).sum = n) :
    words ∈ boundedComponentLanguage M n := by
  rw [sum_component_wedge_lengths] at hsize
  have hmn : words.length ≤ n := by omega
  rw [boundedComponentLanguage]
  apply List.mem_flatMap.mpr
  refine ⟨⟨words.length, by omega⟩, List.mem_finRange _, ?_⟩
  rw [if_pos hmn]
  exact mem_componentLanguage_of_valid words words.length (n - words.length)
    rfl hvalid (by omega)

/-- Bound a finite family of distinct component tuples by total wedges, allowing the number
of components to vary up to the fixed cap `M`. -/
theorem card_boundedComponentFamily_le (M n : ℕ) (family : Finset (List (List Rotation)))
    (hfamily : ∀ words ∈ family, words.length ≤ M ∧
      (∀ w ∈ words, ValidList w) ∧ (words.map (fun w => w.length + 1)).sum = n) :
    family.card ≤ boundedComponentWedgeCount M n := by
  have hsub : family ⊆ (boundedComponentLanguage M n).toFinset := by
    intro words hwords
    obtain ⟨hlen, hvalid, hsize⟩ := hfamily words hwords
    exact List.mem_toFinset.mpr
      (mem_boundedComponentLanguage_of_valid words M n hlen hvalid hsize)
  calc
    family.card ≤ (boundedComponentLanguage M n).toFinset.card := Finset.card_le_card hsub
    _ ≤ (boundedComponentLanguage M n).length := List.toFinset_card_le _
    _ = boundedComponentWedgeCount M n := boundedComponentLanguage_length M n

/-- A uniform bound on individual words gives a polynomial-times-geometric
bound on every fixed convolution. -/
theorem componentCount_le_mul_pow {C r : ℝ} (hC : 0 ≤ C) (hr : 0 < r)
    (hbound : ∀ n : ℕ, (countValidFormulas n : ℝ) ≤ C * r ^ n)
    (m n : ℕ) :
    (componentCount m n : ℝ) ≤ C ^ m * ((n : ℝ) + 1) ^ m * r ^ n := by
  induction m generalizing n with
  | zero =>
      by_cases hn : n = 0
      · subst n
        simp [componentCount]
      · simpa [componentCount, hn] using (pow_nonneg hr.le n)
  | succ m ih =>
      rw [componentCount_succ, Nat.cast_sum]
      simp only [Nat.cast_mul]
      calc
        (∑ k ∈ Finset.range (n + 1),
            (countValidFormulas k : ℝ) * (componentCount m (n - k) : ℝ)) ≤
            ∑ _k ∈ Finset.range (n + 1),
              C ^ (m + 1) * ((n : ℝ) + 1) ^ m * r ^ n := by
          apply Finset.sum_le_sum
          intro k hk
          have hkn : k + (n - k) = n := by
            have := Finset.mem_range.mp hk
            omega
          have hpoly : (((n - k : ℕ) : ℝ) + 1) ^ m ≤ ((n : ℝ) + 1) ^ m := by
            apply pow_le_pow_left₀ (by positivity)
            exact_mod_cast Nat.add_le_add_right (Nat.sub_le n k) 1
          have htail :
              (componentCount m (n - k) : ℝ) ≤
                C ^ m * ((n : ℝ) + 1) ^ m * r ^ (n - k) :=
            (ih (n - k)).trans
              (mul_le_mul_of_nonneg_right
                (mul_le_mul_of_nonneg_left hpoly (pow_nonneg hC m))
                (pow_nonneg hr.le (n - k)))
          calc
            (countValidFormulas k : ℝ) * (componentCount m (n - k) : ℝ) ≤
                (C * r ^ k) * (C ^ m * ((n : ℝ) + 1) ^ m * r ^ (n - k)) :=
              mul_le_mul (hbound k) htail (Nat.cast_nonneg _)
                (mul_nonneg hC (pow_nonneg hr.le k))
            _ = C ^ (m + 1) * ((n : ℝ) + 1) ^ m * (r ^ k * r ^ (n - k)) := by
              rw [pow_succ]
              ring
            _ = C ^ (m + 1) * ((n : ℝ) + 1) ^ m * r ^ n := by
              rw [← pow_add, hkn]
        _ = C ^ (m + 1) * ((n : ℝ) + 1) ^ (m + 1) * r ^ n := by
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul,
            Nat.cast_add, Nat.cast_one, pow_succ]
          ring

/-- A single prefactor works for all component counts at a given rate above
the growth constant. -/
theorem exists_componentCount_le_mul_pow {r : ℝ}
    (hr : snakeGrowthConstant < r) :
    ∃ C : ℝ, 1 ≤ C ∧ ∀ m n : ℕ,
      (componentCount m n : ℝ) ≤ C ^ m * ((n : ℝ) + 1) ^ m * r ^ n := by
  obtain ⟨C, hC, hbound⟩ := exists_countValidFormulas_le_mul_pow hr
  exact ⟨C, hC, componentCount_le_mul_pow (zero_le_one.trans hC)
    (snakeGrowthConstant_pos.trans hr) hbound⟩

/-- Fixed component multiplicity and a polynomial number of placements do not
increase the exponential rate. No geometric realization is assumed. -/
theorem summable_componentCount_div_pow {q : ℝ}
    (hq : snakeGrowthConstant < q) (m D : ℕ) :
    Summable (fun n : ℕ =>
      ((n : ℝ) + 1) ^ D * (componentCount m n : ℝ) / q ^ n) := by
  obtain ⟨r, hmur, hrq⟩ := exists_between hq
  obtain ⟨C, hC, hbound⟩ := exists_componentCount_le_mul_pow hmur
  have hrpos : 0 < r := snakeGrowthConstant_pos.trans hmur
  have hqpos : 0 < q := hrpos.trans hrq
  have hratio : 0 < r / q := div_pos hrpos hqpos
  have hnorm : ‖r / q‖ < 1 := by
    rw [Real.norm_eq_abs, abs_of_pos hratio]
    exact (div_lt_one hqpos).mpr hrq
  have hgeom := summable_pow_mul_geometric_of_norm_lt_one (D + m) hnorm
  have hshift :
      Summable (fun n : ℕ => ((n : ℝ) + 1) ^ (D + m) * (r / q) ^ n) := by
    have h := ((summable_nat_add_iff 1).mpr hgeom).mul_right (r / q)⁻¹
    simpa only [Nat.cast_add, Nat.cast_one, pow_succ, mul_assoc,
      mul_inv_cancel₀ hratio.ne', mul_one] using h
  have hcomparison :
      Summable (fun n : ℕ => C ^ m * (((n : ℝ) + 1) ^ (D + m) * (r / q) ^ n)) :=
    hshift.mul_left _
  refine Summable.of_nonneg_of_le (fun n => by positivity) (fun n => ?_) hcomparison
  calc
    ((n : ℝ) + 1) ^ D * (componentCount m n : ℝ) / q ^ n ≤
        ((n : ℝ) + 1) ^ D * (C ^ m * ((n : ℝ) + 1) ^ m * r ^ n) / q ^ n :=
      div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_left (hbound m n) (by positivity))
        (pow_nonneg hqpos.le n)
    _ = C ^ m * (((n : ℝ) + 1) ^ (D + m) * (r / q) ^ n) := by
      rw [pow_add, div_pow]
      ring

/-- Failure of the polynomially weighted summability forces the tested rate
to be at most the snake growth constant. -/
theorem le_snakeGrowthConstant_of_not_summable_componentCount
    {q : ℝ} (m D : ℕ)
    (h : ¬ Summable (fun n : ℕ =>
      ((n : ℝ) + 1) ^ D * (componentCount m n : ℝ) / q ^ n)) :
    q ≤ snakeGrowthConstant :=
  le_of_not_gt fun hq => h (summable_componentCount_div_pow hq m D)

/-- For a fixed number of components, polynomially weighted wedge counts are summable after
division by `q ^ n` whenever `q` exceeds the snake growth constant. -/
theorem summable_componentWedgeCount_div_pow {q : ℝ}
    (hq : snakeGrowthConstant < q) (m D : ℕ) :
    Summable (fun n : ℕ =>
      ((n : ℝ) + 1) ^ D * (componentWedgeCount m n : ℝ) / q ^ n) := by
  have hqpos : 0 < q := snakeGrowthConstant_pos.trans hq
  let C : ℝ := ((m : ℝ) + 1) ^ D / q ^ m
  have hcomparison := (summable_componentCount_div_pow hq m D).mul_left C
  apply (summable_nat_add_iff m).mp
  refine Summable.of_nonneg_of_le (fun n => by positivity) (fun n => ?_) hcomparison
  have hpoly :
      (((n + m : ℕ) : ℝ) + 1) ^ D ≤ (((m : ℝ) + 1) * ((n : ℝ) + 1)) ^ D := by
    apply pow_le_pow_left₀ (by positivity)
    push_cast
    nlinarith [mul_nonneg (Nat.cast_nonneg (α := ℝ) n) (Nat.cast_nonneg (α := ℝ) m)]
  simp only [componentWedgeCount, if_pos (by omega : m ≤ n + m), Nat.add_sub_cancel]
  calc
    (((n + m : ℕ) : ℝ) + 1) ^ D * (componentCount m n : ℝ) / q ^ (n + m) ≤
        (((m : ℝ) + 1) * ((n : ℝ) + 1)) ^ D *
          (componentCount m n : ℝ) / q ^ (n + m) :=
      div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_right hpoly (Nat.cast_nonneg _)) (pow_nonneg hqpos.le _)
    _ = C * (((n : ℝ) + 1) ^ D * (componentCount m n : ℝ) / q ^ n) := by
      dsimp [C]
      rw [mul_pow, pow_add]
      field_simp

/-- The number of components may vary within any fixed bound. -/
theorem summable_boundedComponentWedgeCount_div_pow {q : ℝ}
    (hq : snakeGrowthConstant < q) (M D : ℕ) :
    Summable (fun n : ℕ =>
      ((n : ℝ) + 1) ^ D * (boundedComponentWedgeCount M n : ℝ) / q ^ n) := by
  have hsum : Summable (fun n : ℕ =>
      ∑ m ∈ Finset.range (M + 1),
        ((n : ℝ) + 1) ^ D * (componentWedgeCount m n : ℝ) / q ^ n) :=
    summable_sum fun m _ => summable_componentWedgeCount_div_pow hq m D
  simpa only [boundedComponentWedgeCount, Nat.cast_sum, Finset.mul_sum,
    div_eq_mul_inv, Finset.sum_mul] using hsum

/-- Divergence of the polynomially weighted bounded-component series certifies `q <= mu`. -/
theorem le_snakeGrowthConstant_of_not_summable_boundedComponentWedgeCount
    {q : ℝ} (M D : ℕ)
    (h : ¬ Summable (fun n : ℕ =>
      ((n : ℝ) + 1) ^ D * (boundedComponentWedgeCount M n : ℝ) / q ^ n)) :
    q ≤ snakeGrowthConstant :=
  le_of_not_gt fun hq => h (summable_boundedComponentWedgeCount_div_pow hq M D)

/-- Counting by both construction steps and wedge length preserves summability
when every finite collection of step counts has the stated component bound. -/
theorem summable_of_component_coefficient_bound {q : ℝ}
    (hq : snakeGrowthConstant < q) (c : ℕ → ℕ → ℕ) (M D K : ℕ)
    (hcount : ∀ n (steps : Finset ℕ),
      ∑ N ∈ steps, c N n ≤ K * (n + 1) ^ D * boundedComponentWedgeCount M n) :
    Summable (fun N => ∑' n, (c N n : ℝ) / q ^ n) := by
  have hqpos : 0 < q := snakeGrowthConstant_pos.trans hq
  let b (n : ℕ) : ℝ :=
    (K : ℝ) * ((n : ℝ) + 1) ^ D * (boundedComponentWedgeCount M n : ℝ) / q ^ n
  have hb : Summable b := by
    have heq : b = fun n : ℕ =>
        (K : ℝ) * (((n : ℝ) + 1) ^ D * (boundedComponentWedgeCount M n : ℝ) / q ^ n) := by
      funext n
      dsimp [b]
      ring
    rw [heq]
    exact (summable_boundedComponentWedgeCount_div_pow hq M D).mul_left (K : ℝ)
  have hbound (n : ℕ) (steps : Finset ℕ) :
      ∑ N ∈ steps, (c N n : ℝ) / q ^ n ≤ b n := by
    calc
      (∑ N ∈ steps, (c N n : ℝ) / q ^ n) =
          ((∑ N ∈ steps, c N n : ℕ) : ℝ) / q ^ n := by
        simp only [Nat.cast_sum, div_eq_mul_inv, Finset.sum_mul]
      _ ≤ b n := div_le_div_of_nonneg_right
        (by exact_mod_cast hcount n steps) (pow_nonneg hqpos.le n)
  have hcolumn (n : ℕ) : Summable (fun N => (c N n : ℝ) / q ^ n) :=
    summable_of_sum_le (fun _ => by positivity) (hbound n)
  have hcolumn_bound (n : ℕ) :
      ∑' N, (c N n : ℝ) / q ^ n ≤ b n :=
    Real.tsum_le_of_sum_le (fun _ => by positivity) (hbound n)
  have hcolumns : Summable (fun n => ∑' N, (c N n : ℝ) / q ^ n) :=
    Summable.of_nonneg_of_le (fun _ => tsum_nonneg fun _ => by positivity)
      hcolumn_bound hb
  have hproduct : Summable (fun p : ℕ × ℕ => (c p.2 p.1 : ℝ) / q ^ p.1) :=
    (summable_prod_of_nonneg
      (f := fun p : ℕ × ℕ => (c p.2 p.1 : ℝ) / q ^ p.1)
      (fun _ => by positivity)).mpr ⟨hcolumn, hcolumns⟩
  exact hproduct.prod_symm.prod

end

end RubiksSnake

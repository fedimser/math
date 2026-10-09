import RubiksSnake.SnAsymptotic.WeightedRecordCertificate

/-!
# Weighting occupied interfaces

Uniform continuation counts discard the dependence on the preceding block.
One integer-weighted continuation step supplies a positive potential; a
second step certifies the stronger rate `3.193` on the same block language.
-/

namespace RubiksSnake
namespace RecordWeighting

/-- The precomputed image equals the direct sum of length factor times
successor potential over all fitting record transitions. -/
lemma image_eq (s : RecordState) :
    image s =
      (((recordTables s.previous).filter (recordFits s)).map
        (fun d => coefficient d.word.length * weight d.next)).sum := by
  have ht (i : Fin 4) :
      tables i = (recordTables i).map (fun d => (d, weight d.next)) := by
    fin_cases i <;> rfl
  rw [image, ht, List.filter_map, List.map_map]
  rfl

/-- Every phase-table datum has a rotation word of length two through seven. -/
private lemma datum_lengths {i : Fin 4} {d : RecordDatum} (hd : d ∈ recordTables i) :
    2 ≤ d.word.length ∧ d.word.length ≤ 7 := by
  obtain ⟨rs, hrs, rfl⟩ := mem_recordTables.mp hd
  exact recordCode_lengths rs hrs

/-- For `n >= 7`, every available block fits the length budget, so the
language size is the full sum of successor-language sizes over fitting blocks. -/
lemma language_length_eq (s : RecordState) (n : ℕ) (hn : 7 ≤ n) :
    (recordLanguage s n).length =
      (((recordTables s.previous).filter (recordFits s)).map
        (fun d => (recordLanguage d.next (n - d.word.length)).length)).sum := by
  rw [recordLanguage, if_neg (by omega : n ≠ 0), List.length_flatMap]
  have hm :
      ((recordTables s.previous).map fun d =>
        (if 0 < d.word.length ∧ d.word.length ≤ n ∧ recordFits s d = true then
          (recordLanguage d.next (n - d.word.length)).map (d.word ++ ·)
        else []).length) =
      (recordTables s.previous).map (fun d =>
        if recordFits s d then (recordLanguage d.next (n - d.word.length)).length else 0) := by
    apply List.map_congr_left
    intro d hd
    obtain ⟨hlo, hhi⟩ := datum_lengths hd
    have hpos : 0 < d.word.length := by omega
    have hle : d.word.length ≤ n := by omega
    by_cases hfit : recordFits s d = true <;> simp [hpos, hle, hfit]
  rw [hm]
  simp [List.sum_map_ite]

/-- Common integer normalization covering the finite base lengths two through
eight in the weighted renewal induction. -/
def baseScale : ℕ := maximumWeight * 3193 ^ 8

/-- The normalization is the uniform potential bound multiplied by `3193^8`. -/
lemma baseScale_eq : baseScale = maximumWeight * 3193 ^ 8 := rfl

attribute [local irreducible] baseScale maximumWeight weight

/-- The normalization is strictly positive, allowing division in the final
real-valued lower bound. -/
lemma baseScale_pos : 0 < baseScale := by
  rw [baseScale_eq]
  exact Nat.mul_pos maximumWeight_pos (by positivity)

/-- Length-two and length-three choices make the scalar record renewal count
positive for every rotation length at least two. -/
private lemma renewal_pos (n : ℕ) (hn : 2 ≤ n) : 0 < recordRenewal n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hsmall : n < 4
      · have h0 : recordRenewal 0 = 1 := by rw [recordRenewal]; norm_num
        have h1 : recordRenewal 1 = 0 := by rw [recordRenewal]; norm_num
        interval_cases n <;> rw [recordRenewal] <;> norm_num [h0, h1]
      · have hi := ih (n - 2) (by omega) (by omega)
        rw [recordRenewal, if_neg (by omega : n ≠ 0), if_pos hn]
        positivity

/-- From every listed state and for `n >= 2`, the record language has at
least `weight s / baseScale` times `3.193^n` words, expressed without division. -/
lemma weighted_language_lower (n : ℕ) (hn : 2 ≤ n) :
    ∀ s ∈ recordStates,
      weight s * 3193 ^ n ≤ baseScale * 1000 ^ n * (recordLanguage s n).length := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro s hs
      by_cases hsmall : n ≤ 8
      · have hcount : 1 ≤ (recordLanguage s n).length :=
          (renewal_pos n hn).trans_le (recordRenewal_le_language n s hs)
        calc
          weight s * 3193 ^ n ≤ maximumWeight * 3193 ^ 8 :=
            Nat.mul_le_mul (weight_checked s hs).2.1 (Nat.pow_le_pow_right (by decide) hsmall)
          _ = baseScale * 1 * 1 := by rw [mul_one, mul_one, baseScale_eq]
          _ ≤ baseScale * 1000 ^ n * (recordLanguage s n).length :=
            Nat.mul_le_mul (Nat.mul_le_mul_left _ (Nat.one_le_pow n 1000 (by decide))) hcount
      · have hlarge : 7 ≤ n := by omega
        let ds := (recordTables s.previous).filter (recordFits s)
        have lengths (d : RecordDatum) (hd : d ∈ ds) :
            2 ≤ d.word.length ∧ d.word.length ≤ 7 :=
          datum_lengths (List.mem_filter.mp hd).1
        calc
          weight s * 3193 ^ n = (3193 ^ 7 * weight s) * 3193 ^ (n - 7) := by
            rw [mul_right_comm, ← pow_add, Nat.add_sub_of_le hlarge, mul_comm]
          _ ≤ image s * 3193 ^ (n - 7) :=
            Nat.mul_le_mul_right _ (weight_checked s hs).2.2
          _ = (ds.map (fun d =>
              1000 ^ d.word.length * (weight d.next * 3193 ^ (n - d.word.length)))).sum := by
            rw [image_eq, ← List.sum_map_mul_right]
            congr 1
            apply List.map_congr_left
            intro d hd
            have hl := lengths d hd
            have hp : 3193 ^ (7 - d.word.length) * 3193 ^ (n - 7) =
                3193 ^ (n - d.word.length) := by
              rw [← pow_add]
              congr 1
              omega
            dsimp [coefficient]
            rw [← hp]
            ac_rfl
          _ ≤ (ds.map (fun d =>
              1000 ^ d.word.length *
                (baseScale * 1000 ^ (n - d.word.length) *
                  (recordLanguage d.next (n - d.word.length)).length))).sum := by
            apply List.sum_le_sum
            intro d hd
            have hl := lengths d hd
            apply Nat.mul_le_mul_left
            exact ih (n - d.word.length) (by omega) (by omega) d.next
              (recordNext_mem (List.mem_filter.mp hd).1)
          _ = baseScale * 1000 ^ n * (recordLanguage s n).length := by
            rw [language_length_eq s n hlarge, ← List.sum_map_mul_left]
            congr 1
            apply List.map_congr_left
            intro d hd
            have hl := lengths d hd
            have hp : 1000 ^ d.word.length * 1000 ^ (n - d.word.length) = 1000 ^ n := by
              rw [← pow_add, Nat.add_sub_of_le (by omega : d.word.length ≤ n)]
            rw [← hp]
            ac_rfl

/-- Integer form of the uniform `3.193^n / baseScale` lower bound for valid
`n`-rotation formulas, including lengths zero and one. -/
lemma count_lower (n : ℕ) :
    3193 ^ n ≤ baseScale * 1000 ^ n * countValidFormulas n := by
  by_cases hn : 2 ≤ n
  · calc
      3193 ^ n ≤ weight (recordInitial 0) * 3193 ^ n := by
        have hw := (weight_checked (recordInitial 0) (recordInitial_mem 0)).1
        exact Nat.le_mul_of_pos_left (3193 ^ n) hw
      _ ≤ baseScale * 1000 ^ n * (recordLanguage (recordInitial 0) n).length :=
        weighted_language_lower n hn (recordInitial 0) (recordInitial_mem 0)
      _ ≤ baseScale * 1000 ^ n * countValidFormulas n :=
        Nat.mul_le_mul_left _ (recordLanguage_length_le n)
  · have hscale : 3193 ≤ baseScale := by
      have hw := maximumWeight_pos
      unfold baseScale
      norm_num
      nlinarith
    generalize baseScale = c at *
    obtain rfl | rfl : n = 0 ∨ n = 1 := by omega
    · have h0 : countValidFormulas 0 = 1 := by simpa [S] using S1_value
      simp only [pow_zero, mul_one, h0]
      omega
    · have h1 : countValidFormulas 1 = 4 := by simpa [S] using S2_value
      simp only [pow_one, h1]
      omega

end RecordWeighting

/-- Weighted occupied-interface blocks certify at least
`3.193^n / RecordWeighting.baseScale` valid formulas with `n` rotations,
or `n + 1` wedges. -/
theorem countValidFormulas_lower_bound_3193_div_1000_with_prefactor (n : ℕ) :
    (1 / (RecordWeighting.baseScale : ℝ)) * (3193 / 1000 : ℝ) ^ n ≤
      countValidFormulas n := by
  have hc : 0 < (RecordWeighting.baseScale : ℝ) := by
    exact_mod_cast RecordWeighting.baseScale_pos
  have hnat := RecordWeighting.count_lower n
  have hreal : (3193 : ℝ) ^ n ≤
      (RecordWeighting.baseScale : ℝ) * (1000 : ℝ) ^ n * countValidFormulas n := by
    exact_mod_cast hnat
  calc
    (1 / (RecordWeighting.baseScale : ℝ)) * (3193 / 1000 : ℝ) ^ n =
        (3193 : ℝ) ^ n / ((RecordWeighting.baseScale : ℝ) * (1000 : ℝ) ^ n) := by
      rw [div_pow]
      ring
    _ ≤ countValidFormulas n := by
      apply (div_le_iff₀ (mul_pos hc (pow_pos (by norm_num) n))).mpr
      simpa only [mul_comm, mul_left_comm, mul_assoc] using hreal

end RubiksSnake

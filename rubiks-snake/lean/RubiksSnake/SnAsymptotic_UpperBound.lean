import RubiksSnake.SnAsymptotitc_MuUpperBound

/-!
# An explicit pointwise upper bound

Submultiplicativity splits a word length into six-letter blocks and a residue.
The exact block count supplies the exponential factor; the uniform constant
absorbs the five possible leftover letters.
-/

namespace RubiksSnake

noncomputable section

/-- Iterated submultiplicativity for a quotient-block-residue decomposition. -/
lemma countValidFormulas_mul_add_le (blocks block residue : ℕ) :
    countValidFormulas (blocks * block + residue) ≤
      countValidFormulas block ^ blocks * countValidFormulas residue := by
  induction blocks with
  | zero => simp
  | succ blocks ih =>
      calc
        countValidFormulas ((blocks + 1) * block + residue) =
            countValidFormulas (block + (blocks * block + residue)) := by
              congr 1
              simp only [Nat.add_mul, one_mul]
              ac_rfl
        _ ≤ countValidFormulas block *
              countValidFormulas (blocks * block + residue) :=
          countValidFormulas_submultiplicative _ _
        _ ≤ countValidFormulas block *
              (countValidFormulas block ^ blocks *
                countValidFormulas residue) :=
          Nat.mul_le_mul_left _ ih
        _ = countValidFormulas block ^ (blocks + 1) *
              countValidFormulas residue := by
          rw [pow_succ]
          ac_rfl

/-- A block count and finitely many residue estimates give a pointwise
exponential estimate.  To tighten the bound, change `block`, `count`, and
`q`, then discharge only the exact-count, power, and residue inequalities. -/
theorem countValidFormulas_le_of_block_certificate
    (block count : ℕ) (q C : ℝ) (hblock : 0 < block)
    (hexact : countValidFormulas block = count)
    (hpower : (count : ℝ) ≤ q ^ block) (hq : 1 ≤ q)
    (hresidue : ∀ r < block, (countValidFormulas r : ℝ) ≤ C * q ^ r)
    (k : ℕ) :
    (countValidFormulas k : ℝ) ≤ C * q ^ k := by
  let blocks := k / block
  let residue := k % block
  have hresidue_lt : residue < block := Nat.mod_lt k hblock
  have hdecomp : blocks * block + residue = k := by
    simpa [blocks, residue, Nat.mul_comm] using Nat.div_add_mod k block
  have hsplit :
      countValidFormulas k ≤
        countValidFormulas block ^ blocks * countValidFormulas residue := by
    rw [← hdecomp]
    exact countValidFormulas_mul_add_le blocks block residue
  calc
    (countValidFormulas k : ℝ) ≤
        (countValidFormulas block : ℝ) ^ blocks *
          countValidFormulas residue := by
      exact_mod_cast hsplit
    _ = (count : ℝ) ^ blocks * countValidFormulas residue := by
      rw [hexact]
    _ ≤ (q ^ block) ^ blocks * (C * q ^ residue) := by
      apply mul_le_mul
      · exact pow_le_pow_left₀ (by positivity) hpower blocks
      · exact hresidue residue hresidue_lt
      · positivity
      · positivity
    _ = C * (q ^ (block * blocks) * q ^ residue) := by
      rw [← pow_mul]
      ring
    _ = C * q ^ (block * blocks + residue) := by
      rw [pow_add]
    _ = C * q ^ k := by
      rw [show block * blocks + residue = k by
        simpa [Nat.mul_comm] using hdecomp]

/-- The block-six certificate from `S 7 = 3384`, with `1024 = 4^5`
absorbing every residue of length less than six. -/
theorem countValidFormulas_upper_bound_39_div_10 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      1024 * ((39 : ℝ) / 10) ^ k := by
  apply countValidFormulas_le_of_block_certificate
      6 3384 ((39 : ℝ) / 10) 1024 (by norm_num)
  · simpa [S] using S7_value
  · norm_num
  · norm_num
  · intro r hr
    have hcount : (countValidFormulas r : ℝ) ≤ (4 : ℝ) ^ r := by
      exact_mod_cast countFormulas_upper_bound r
    calc
      (countValidFormulas r : ℝ) ≤ (4 : ℝ) ^ r := hcount
      _ ≤ (4 : ℝ) ^ 5 :=
        pow_le_pow_right₀ (by norm_num) (by omega)
      _ = 1024 := by norm_num
      _ ≤ 1024 * ((39 : ℝ) / 10) ^ r := by
        have hp : (1 : ℝ) ≤ ((39 : ℝ) / 10) ^ r :=
          one_le_pow₀ (by norm_num)
        nlinarith

/-- Explicitly, every positive snake length satisfies
`S n ≤ 1024 * 3.9^(n-1)`. -/
theorem Sn_upper_bound_39_div_10 (n : ℕ+) :
    (S n : ℝ) ≤
      1024 * ((39 : ℝ) / 10) ^ ((n : ℕ) - 1) := by
  exact countValidFormulas_upper_bound_39_div_10 ((n : ℕ) - 1)

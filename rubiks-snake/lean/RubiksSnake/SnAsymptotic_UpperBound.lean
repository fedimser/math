import RubiksSnake.SnAsymptotitc_MuUpperBound

/-!
# An explicit pointwise upper bound

The length-sixteen collision-prefix certificate gives
`S n <= 3 * 3.661786723^(n-1)` for every positive snake length.
The earlier prefix and window certificates and the older
block-submultiplicativity API remain available.
-/

namespace RubiksSnake

noncomputable section

/-- The length-sixteen certificate has base `3.661786723` and prefactor `3`. -/
theorem countValidFormulas_upper_bound_3661786723_div_1000000000 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      3 * (3661786723 / 1000000000 : ℝ) ^ k :=
  countValidFormulas_le_forbidden_prefix_sixteen_upper k

/-- Reindex the strongest prefix-automaton bound for an `n`-wedge snake, whose word has `n - 1` rotations. -/
theorem Sn_upper_bound_3661786723_div_1000000000 (n : ℕ+) :
    (S n : ℝ) ≤
      3 * (3661786723 / 1000000000 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_3661786723_div_1000000000 ((n : ℕ) - 1)

/-- The length-fourteen certificate has base `3.667542939` and prefactor `3`. -/
theorem countValidFormulas_upper_bound_3667542939_div_1000000000 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      3 * (3667542939 / 1000000000 : ℝ) ^ k :=
  countValidFormulas_le_forbidden_prefix_fourteen_upper k

/-- The length-fourteen certificate gives the master-paper base 3.667542939 at every wedge length. -/
theorem Sn_upper_bound_3667542939_div_1000000000 (n : ℕ+) :
    (S n : ℝ) ≤
      3 * (3667542939 / 1000000000 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_3667542939_div_1000000000 ((n : ℕ) - 1)

/-- A uniform upper bound from the compressed collision-prefix automaton. -/
theorem countValidFormulas_upper_bound_147_div_40 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤ (9 / 2 : ℝ) * (147 / 40 : ℝ) ^ k :=
  countValidFormulas_le_forbidden_prefix_upper k

/-- The length-twelve prefix certificate gives `S n <= (9/2) * 3.675^(n - 1)`. -/
theorem Sn_upper_bound_147_div_40 (n : ℕ+) :
    (S n : ℝ) ≤ (9 / 2 : ℝ) * (147 / 40 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_147_div_40 ((n : ℕ) - 1)

/-- The seven-symbol window bound has base `3.704` and prefactor `8/3`. -/
theorem countValidFormulas_upper_bound_463_div_125 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤ (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ k :=
  countValidFormulas_le_seven_window_upper k

/-- For every positive snake length, `S n <= (8/3) * 3.704^(n-1)`. -/
theorem Sn_upper_bound_463_div_125 (n : ℕ+) :
    (S n : ℝ) ≤ (8 / 3 : ℝ) * (463 / 125 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_463_div_125 ((n : ℕ) - 1)

/-- The window certificate bounds every formula length, with base `3.7202`
and prefactor `9/4`. -/
theorem countValidFormulas_upper_bound_18601_div_5000 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ k :=
  countValidFormulas_le_window_upper k

/-- For every positive snake length, `S n <= (9/4) * 3.7202^(n-1)`. -/
theorem Sn_upper_bound_18601_div_5000 (n : ℕ+) :
    (S n : ℝ) ≤
      (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_18601_div_5000 ((n : ℕ) - 1)

/-- A rounded form of the window bound, still valid at every formula length. -/
theorem countValidFormulas_upper_bound_373_div_100 (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      (9 / 4 : ℝ) * (373 / 100 : ℝ) ^ k := by
  apply (countValidFormulas_upper_bound_18601_div_5000 k).trans
  apply mul_le_mul_of_nonneg_left
  · exact pow_le_pow_left₀ (by norm_num) (by norm_num) k
  · norm_num

/-- In particular, every positive snake length has an upper bound with
base `3.73` and prefactor `9/4`. -/
theorem Sn_upper_bound_373_div_100 (n : ℕ+) :
    (S n : ℝ) ≤
      (9 / 4 : ℝ) * (373 / 100 : ℝ) ^ ((n : ℕ) - 1) :=
  countValidFormulas_upper_bound_373_div_100 ((n : ℕ) - 1)

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

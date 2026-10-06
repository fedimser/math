import RubiksSnake.WindowUpperComputation
import RubiksSnake.FiniteWindowUpperBound
import Std.Data.HashSet
import Std.Data.HashMap
import Mathlib.Algebra.Order.BigOperators.Group.List

/-!
# Counting with a prefix-closed suffix dictionary

The state remembers the longest suffix in a finite, prefix-closed
dictionary. Every actual extension passes the geometric check on this
suffix. The dictionary need not contain every forbidden factor.
-/

namespace RubiksSnake
namespace PrefixAutomaton

open WindowComputation

abbrev Dictionary := Std.HashSet (List Rotation)

def longest (dictionary : Dictionary) : List Rotation → List Rotation
  | [] => []
  | r :: rs => if dictionary.contains (r :: rs) then r :: rs else longest dictionary rs

lemma longest_suffix (dictionary : Dictionary) (rs : List Rotation) :
    longest dictionary rs <:+ rs := by
  induction rs with
  | nil => simp [longest]
  | cons r rs ih =>
      simp only [longest]
      split
      · exact List.suffix_refl _
      · exact ih.trans (List.suffix_cons r rs)

lemma longest_mem (dictionary : Dictionary) (hzero : [] ∈ dictionary) (rs : List Rotation) :
    longest dictionary rs ∈ dictionary := by
  induction rs with
  | nil => exact hzero
  | cons r rs ih =>
      simp only [longest]
      split
      · rename_i h
        exact (Std.HashSet.contains_iff_mem).mp h
      · exact ih

lemma longest_covers (dictionary : Dictionary) {rs q : List Rotation}
    (hq : q ∈ dictionary) (hs : q <:+ rs) :
    q <:+ longest dictionary rs := by
  induction rs with
  | nil =>
      have : q = [] := List.suffix_nil.mp hs
      subst q
      simp [longest]
  | cons r rs ih =>
      simp only [longest]
      split
      · exact hs
      · rename_i hnot
        rcases List.suffix_cons_iff.mp hs with heq | hs
        · subst q
          exact False.elim (hnot ((Std.HashSet.contains_iff_mem).mpr hq))
        · exact ih hs

lemma suffix_append_singleton {a b : List Rotation} (h : a <:+ b) (r : Rotation) :
    a ++ [r] <:+ b ++ [r] :=
  (List.suffix_append_self_iff).mpr h

lemma longest_append (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (rs : List Rotation) (r : Rotation) :
    longest dictionary (rs ++ [r]) =
      longest dictionary (longest dictionary rs ++ [r]) := by
  let p := longest dictionary rs
  have hp : p <:+ rs := longest_suffix dictionary rs
  have hshort :
      longest dictionary (p ++ [r]) <:+ longest dictionary (rs ++ [r]) :=
    longest_covers dictionary (longest_mem dictionary hzero _)
      ((longest_suffix dictionary _).trans (suffix_append_singleton hp r))
  have hlong :
      longest dictionary (rs ++ [r]) <:+ longest dictionary (p ++ [r]) := by
    have hm := longest_mem dictionary hzero (rs ++ [r])
    have hs := longest_suffix dictionary (rs ++ [r])
    rcases List.suffix_concat_iff.mp hs with heq | ⟨pre, heq, hpre⟩
    · rw [heq]
      exact List.nil_suffix
    · have hpre_mem : pre ∈ dictionary := by
        have h := hclosed _ hm
        simpa [heq] using h
      rw [heq]
      apply longest_covers dictionary (by simpa [heq] using hm)
      exact suffix_append_singleton (longest_covers dictionary hpre_mem hpre) r
  exact hlong.sublist.antisymm hshort.sublist

def outgoing (dictionary : Dictionary) (f : List Rotation → ℕ) (rs : List Rotation) : ℕ :=
  (rotations.map fun r =>
    if valid (rs ++ [r]) then f (longest dictionary (rs ++ [r])) else 0).sum

def continuations (dictionary : Dictionary) : ℕ → List Rotation → ℕ
  | 0 => fun _ => 1
  | t + 1 => outgoing dictionary (continuations dictionary t)

def totalWeight (dictionary : Dictionary) (f : List Rotation → ℕ) (k : ℕ) : ℕ :=
  ((validWords k).map fun rs => f (longest dictionary rs)).sum

private lemma valid_suffix {rs q : List Rotation} (hs : q <:+ rs) (hv : ValidList rs) :
    ValidList q := by
  obtain ⟨pre, rfl⟩ := hs
  simpa using validList_drop (pre ++ q) pre.length hv

lemma totalWeight_next_le (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (f : List Rotation → ℕ) (k : ℕ) :
    totalWeight dictionary f (k + 1) ≤ totalWeight dictionary (outgoing dictionary f) k := by
  unfold totalWeight
  rw [validWords, FiniteWindowUpper.sum_map_flatMap]
  apply List.sum_le_sum
  intro rs _
  rw [FiniteWindowUpper.children_sum]
  apply List.sum_le_sum
  intro r _
  by_cases hvalid : valid (rs ++ [r])
  · have hshort : valid (longest dictionary rs ++ [r]) :=
      (valid_iff _).mpr (valid_suffix
        (suffix_append_singleton (longest_suffix dictionary rs) r) ((valid_iff _).mp hvalid))
    rw [if_pos hvalid, if_pos hshort, longest_append dictionary hzero hclosed]
  · simp [hvalid]

lemma totalWeight_one (dictionary : Dictionary) (k : ℕ) :
    totalWeight dictionary (fun _ => 1) k = countValidFormulas k := by
  simp [totalWeight, validWords_eq, ← fastCountValidFormulas_eq, fastCountValidFormulas]

lemma totalWeight_mono (dictionary : Dictionary) (hzero : [] ∈ dictionary)
    (f g : List Rotation → ℕ) (a b k : ℕ)
    (h : ∀ rs ∈ dictionary, ValidList rs → a * f rs ≤ b * g rs) :
    a * totalWeight dictionary f k ≤ b * totalWeight dictionary g k := by
  unfold totalWeight
  rw [← List.sum_map_mul_left, ← List.sum_map_mul_left]
  apply List.sum_le_sum
  intro rs hrs
  have hv : ValidList rs :=
    validRotationLists_valid k rs (by simpa only [validWords_eq] using hrs)
  exact h _ (longest_mem dictionary hzero rs) (valid_suffix (longest_suffix dictionary rs) hv)

lemma count_terminal_le (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (k terminal : ℕ) :
    countValidFormulas (k + terminal) ≤
      totalWeight dictionary (continuations dictionary terminal) k := by
  induction terminal generalizing k with
  | zero => simp [continuations, totalWeight_one]
  | succ terminal ih =>
      calc
        countValidFormulas (k + (terminal + 1)) =
            countValidFormulas ((k + 1) + terminal) := by congr 1; omega
        _ ≤ totalWeight dictionary (continuations dictionary terminal) (k + 1) := ih _
        _ ≤ totalWeight dictionary (continuations dictionary (terminal + 1)) k :=
          totalWeight_next_le dictionary hzero hclosed _ k

lemma count_terminal_family_le (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (v : ℕ → List Rotation → ℕ) (terminal : ℕ)
    (hbase : ∀ rs ∈ dictionary, ValidList rs → 1 ≤ v 0 rs)
    (hstep : ∀ t < terminal, ∀ rs ∈ dictionary, ValidList rs →
      outgoing dictionary (v t) rs ≤ v (t + 1) rs)
    (k : ℕ) :
    countValidFormulas (k + terminal) ≤ totalWeight dictionary (v terminal) k := by
  have hgeneral (t : ℕ) (ht : t ≤ terminal) :
      ∀ k, countValidFormulas (k + t) ≤ totalWeight dictionary (v t) k := by
    induction t with
    | zero =>
        intro k
        have h := totalWeight_mono dictionary hzero (fun _ => 1) (v 0) 1 1 k
          (by simpa using hbase)
        simpa only [one_mul, Nat.add_zero, totalWeight_one] using h
    | succ t ih =>
        intro k
        calc
          countValidFormulas (k + (t + 1)) =
              countValidFormulas ((k + 1) + t) := by congr 1; omega
          _ ≤ totalWeight dictionary (v t) (k + 1) := ih (by omega) (k + 1)
          _ ≤ totalWeight dictionary (outgoing dictionary (v t)) k :=
            totalWeight_next_le dictionary hzero hclosed _ k
          _ ≤ totalWeight dictionary (v (t + 1)) k := by
            simpa only [one_mul] using
              totalWeight_mono dictionary hzero (outgoing dictionary (v t)) (v (t + 1))
                1 1 k (by simpa using hstep t (by omega))
  exact hgeneral terminal le_rfl k

lemma totalWeight_bound (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (w : List Rotation → ℕ) (a b : ℕ) (hb : 0 < b)
    (h : ∀ rs ∈ dictionary, ValidList rs →
      b * outgoing dictionary w rs ≤ a * w rs)
    (k : ℕ) :
    (totalWeight dictionary w k : ℝ) ≤ w [] * ((a : ℝ) / b) ^ k := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  induction k with
  | zero => simp [totalWeight, validWords, longest]
  | succ k ih =>
      have hstep : b * totalWeight dictionary w (k + 1) ≤ a * totalWeight dictionary w k :=
        (Nat.mul_le_mul_left b (totalWeight_next_le dictionary hzero hclosed w k)).trans
          (totalWeight_mono dictionary hzero (outgoing dictionary w) w b a k h)
      have hstep' : (b : ℝ) * totalWeight dictionary w (k + 1) ≤
          (a : ℝ) * totalWeight dictionary w k := by exact_mod_cast hstep
      calc
        (totalWeight dictionary w (k + 1) : ℝ) ≤
            ((a : ℝ) / b) * totalWeight dictionary w k := by
          rw [div_mul_eq_mul_div]
          exact (le_div_iff₀ hb').mpr (by nlinarith)
        _ ≤ ((a : ℝ) / b) * (w [] * ((a : ℝ) / b) ^ k) :=
          mul_le_mul_of_nonneg_left ih (by positivity)
        _ = w [] * ((a : ℝ) / b) ^ (k + 1) := by rw [pow_succ]; ring

theorem count_bound_of_certificate (dictionary : Dictionary)
    (hzero : [] ∈ dictionary)
    (hclosed : ∀ rs ∈ dictionary, rs.dropLast ∈ dictionary)
    (w : List Rotation → ℕ) (v : ℕ → List Rotation → ℕ)
    (a b scale terminal : ℕ) (hb : 0 < b) (hscale : 0 < scale)
    (hbase : ∀ rs ∈ dictionary, ValidList rs → 1 ≤ v 0 rs)
    (hstep : ∀ t < terminal, ∀ rs ∈ dictionary, ValidList rs →
      outgoing dictionary (v t) rs ≤ v (t + 1) rs)
    (hterminal : ∀ rs ∈ dictionary, ValidList rs → scale * v terminal rs ≤ w rs)
    (hweight : ∀ rs ∈ dictionary, ValidList rs →
      b * outgoing dictionary w rs ≤ a * w rs)
    (C : ℝ) (hinitial : (w [] : ℝ) ≤ scale * C * ((a : ℝ) / b) ^ terminal)
    (k : ℕ) :
    (countValidFormulas (terminal + k) : ℝ) ≤ C * ((a : ℝ) / b) ^ (terminal + k) := by
  have hs : (0 : ℝ) < scale := by exact_mod_cast hscale
  have hcount : scale * countValidFormulas (terminal + k) ≤ totalWeight dictionary w k := by
    have h := count_terminal_family_le dictionary hzero hclosed v terminal hbase hstep k
    rw [Nat.add_comm k terminal] at h
    apply (Nat.mul_le_mul_left scale h).trans
    simpa only [one_mul] using
      totalWeight_mono dictionary hzero (v terminal) w scale 1 k
        (by simpa using hterminal)
  have hreal : (scale : ℝ) * countValidFormulas (terminal + k) ≤
      totalWeight dictionary w k := by exact_mod_cast hcount
  calc
    (countValidFormulas (terminal + k) : ℝ) ≤ (totalWeight dictionary w k : ℝ) / scale :=
      (le_div_iff₀ hs).mpr (by nlinarith)
    _ ≤ (w [] * ((a : ℝ) / b) ^ k) / scale :=
      div_le_div_of_nonneg_right
        (totalWeight_bound dictionary hzero hclosed w a b hb hweight k) hs.le
    _ ≤ (scale * C * ((a : ℝ) / b) ^ terminal * ((a : ℝ) / b) ^ k) / scale :=
      div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_right hinitial (by positivity)) hs.le
    _ = C * ((a : ℝ) / b) ^ (terminal + k) := by
      rw [pow_add]
      field_simp

end PrefixAutomaton
end RubiksSnake

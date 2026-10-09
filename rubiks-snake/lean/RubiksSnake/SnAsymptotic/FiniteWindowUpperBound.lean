import RubiksSnake.SnAsymptotic.WindowUpperComputation
import RubiksSnake.SnAsymptotic.Submultiplicativity
import Mathlib.Algebra.BigOperators.Ring.List
import Mathlib.Algebra.Order.BigOperators.Group.List

/-!
# Universal weighted finite-window counting

Validity of suffixes embeds every actual extension into the finite-window
extension relation. A nonnegative potential controls weighted growth, while
a separate finite number of terminal steps controls the unweighted count.
This accommodates transient dead ends without requiring positive weights.
-/

namespace RubiksSnake
namespace FiniteWindowUpper

open WindowComputation

/-- The last at most `width` rotations, retaining the whole word when it is
shorter than the window. -/
def suffix (width : ℕ) (rs : List Rotation) : List Rotation :=
  rs.drop (rs.length - width)

/-- Sum of suffix-state weights over actual valid words with `k` rotations
(`k + 1` wedges), retaining multiplicity when words share a suffix state. -/
def totalWeight (width : ℕ) (w : List Rotation → ℕ) (k : ℕ) : ℕ :=
  ((validWords k).map fun rs => w (suffix width rs)).sum

/-- Counts locally accepted window paths of a given length, with one empty path;
these counts may exceed the numbers of globally valid extensions. -/
def continuations : ℕ → List Rotation → ℕ
  | 0 => fun _ => 1
  | t + 1 => outgoing (continuations t)

/-- Once the word has filled the window, its retained suffix has exactly
`width` rotations. -/
private lemma suffix_length (width : ℕ) (rs : List Rotation) (hlen : width ≤ rs.length) :
    (suffix width rs).length = width := by
  simp only [suffix, List.length_drop]
  omega

/-- For a positive, already filled window, appending a rotation updates the
retained suffix by dropping its oldest rotation and appending the new one. -/
private lemma suffix_append (width : ℕ) (hw : 0 < width)
    (rs : List Rotation) (r : Rotation) (hlen : width ≤ rs.length) :
    suffix width (rs ++ [r]) = next (suffix width rs) r := by
  simp only [suffix, next, List.length_append, List.length_singleton]
  rw [List.drop_append_of_le_length (by omega), ← List.drop_one, List.drop_drop]
  congr 2
  omega

/-- Expresses a weighted sum over valid children as a sum over all four rotation
choices, with zero contribution from rejected extensions. -/
lemma children_sum (f : List Rotation → ℕ) (rs : List Rotation) :
    ((children rs).map f).sum =
      (rotations.map fun r => if valid (rs ++ [r]) then f (rs ++ [r]) else 0).sum := by
  unfold children
  generalize rotations = xs
  induction xs with
  | nil => simp
  | cons r xs ih =>
      by_cases hr : valid (rs ++ [r]) <;> simp [hr, ih]

/-- For a positive full window, every valid child passes the suffix-only check,
so its suffix weight is covered by the local outgoing sum. -/
private lemma children_sum_le (width : ℕ) (hw : 0 < width)
    (f : List Rotation → ℕ) (rs : List Rotation) (hlen : width ≤ rs.length) :
    ((children rs).map fun child => f (suffix width child)).sum ≤
      outgoing f (suffix width rs) := by
  rw [children_sum]
  apply List.sum_le_sum
  intro r hr
  by_cases hcan : valid (rs ++ [r])
  · have hshort : valid (suffix width rs ++ [r]) := by
      apply (valid_iff _).mpr
      have hdrop := validList_drop (rs ++ [r]) (rs.length - width)
        ((valid_iff _).mp hcan)
      rw [List.drop_append_of_le_length (Nat.sub_le _ _)] at hdrop
      exact hdrop
    simp [hcan, hshort, suffix_append width hw rs r hlen]
  · simp [hcan]

/-- Regroups natural-number weights over a flattened list into sums over its
component lists, as needed to sum extensions by parent word. -/
lemma sum_map_flatMap {α β : Type*} (xs : List α)
    (g : α → List β) (f : β → ℕ) :
    ((xs.flatMap g).map f).sum = (xs.map fun x => ((g x).map f).sum).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih]

/-- Constant-one state weights recover the count of valid `k`-rotation formulas,
independently of the window width. -/
lemma totalWeight_one (width k : ℕ) :
    totalWeight width (fun _ => 1) k = countValidFormulas k := by
  simp [totalWeight, validWords_eq, ← fastCountValidFormulas_eq, fastCountValidFormulas]

/-- At the initial full-window length, every word is its own retained suffix,
so the total is the sum of weights over valid `width`-rotation words. -/
lemma totalWeight_at_width (width : ℕ) (w : List Rotation → ℕ) :
    totalWeight width w width = ((validWords width).map w).sum := by
  unfold totalWeight
  congr 1
  apply List.map_congr_left
  intro rs hrs
  have hlen := validRotationLists_length width rs
    (by simpa only [validWords_eq] using hrs)
  simp [suffix, hlen]

/-- For positive width and `k >= width`, forgetting older rotations can only
increase the weighted one-step extension sum. -/
lemma totalWeight_next_le (width : ℕ) (hw : 0 < width)
    (f : List Rotation → ℕ) (k : ℕ) (hk : width ≤ k) :
    totalWeight width f (k + 1) ≤ totalWeight width (outgoing f) k := by
  unfold totalWeight
  rw [validWords, sum_map_flatMap]
  apply List.sum_le_sum
  intro rs hrs
  have hlen := validRotationLists_length k rs
    (by simpa only [validWords_eq] using hrs)
  exact children_sum_le width hw f rs (by omega)

/-- A scaled inequality on valid full-window states lifts to weighted totals
over actual `k`-rotation words once the window is filled. -/
lemma totalWeight_mono (width : ℕ) (f g : List Rotation → ℕ)
    (a b k : ℕ) (hk : width ≤ k)
    (h : ∀ rs, rs.length = width → ValidList rs → a * f rs ≤ b * g rs) :
    a * totalWeight width f k ≤ b * totalWeight width g k := by
  unfold totalWeight
  rw [← List.sum_map_mul_left, ← List.sum_map_mul_left]
  apply List.sum_le_sum
  intro rs hrs
  have hmem : rs ∈ validRotationLists k := by
    simpa only [validWords_eq] using hrs
  have hlen := validRotationLists_length k rs hmem
  exact h (suffix width rs) (suffix_length width rs (by omega))
    (validList_drop rs _ (validRotationLists_valid k rs hmem))

/-- With a positive full window, an integer row inequality
`b * outgoing w <= a * w` gives the same one-step inequality for actual totals. -/
lemma totalWeight_step (width : ℕ) (hw : 0 < width)
    (w : List Rotation → ℕ) (a b k : ℕ) (hk : width ≤ k)
    (h : ∀ rs, rs.length = width → ValidList rs →
      b * outgoing w rs ≤ a * w rs) :
    b * totalWeight width w (k + 1) ≤ a * totalWeight width w k :=
  (Nat.mul_le_mul_left b (totalWeight_next_le width hw w k hk)).trans
    (totalWeight_mono width (outgoing w) w b a k hk h)

/-- For a positive window filled by the first `k` rotations, local `t`-step
continuation weights dominate the count of valid `(k + t)`-rotation formulas. -/
lemma count_add_le_totalWeight (width : ℕ) (hw : 0 < width)
    (k t : ℕ) (hk : width ≤ k) :
    countValidFormulas (k + t) ≤ totalWeight width (continuations t) k := by
  induction t generalizing k with
  | zero => simp [continuations, totalWeight_one]
  | succ t ih =>
      calc
        countValidFormulas (k + (t + 1)) = countValidFormulas ((k + 1) + t) := by
          congr 1
          omega
        _ ≤ totalWeight width (continuations t) (k + 1) :=
          ih (k + 1) (by omega)
        _ ≤ totalWeight width (continuations (t + 1)) k :=
          totalWeight_next_le width hw (continuations t) k hk

/-- Domination of scaled terminal path counts by `w` on valid full windows
bounds actual formula counts after the terminal rotations. The positive window
must already be filled, but the potential may vanish at dead ends. -/
lemma count_terminal_le (width : ℕ) (hw : 0 < width)
    (w : List Rotation → ℕ) (scale terminal k : ℕ) (hk : width ≤ k)
    (h : ∀ rs, rs.length = width → ValidList rs →
      scale * continuations terminal rs ≤ w rs) :
    scale * countValidFormulas (k + terminal) ≤ totalWeight width w k := by
  apply (Nat.mul_le_mul_left scale (count_add_le_totalWeight width hw k terminal hk)).trans
  simpa only [one_mul] using
    totalWeight_mono width (continuations terminal) w scale 1 k hk
      (by simpa only [one_mul] using h)

/-- For positive width and denominator `b`, the integer row inequality yields
geometric growth with exact ratio `a / b` from the initial full-window total;
strict positivity of the potential is unnecessary. -/
lemma totalWeight_bound (width : ℕ) (hw : 0 < width)
    (w : List Rotation → ℕ) (a b : ℕ) (hb : 0 < b)
    (h : ∀ rs, rs.length = width → ValidList rs →
      b * outgoing w rs ≤ a * w rs) (t : ℕ) :
    (totalWeight width w (width + t) : ℝ) ≤
      totalWeight width w width * ((a : ℝ) / b) ^ t := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  induction t with
  | zero => simp
  | succ t ih =>
      have hstep :
          (b : ℝ) * totalWeight width w (width + t + 1) ≤
            (a : ℝ) * totalWeight width w (width + t) := by
        exact_mod_cast totalWeight_step width hw w a b (width + t) (by omega) h
      calc
        (totalWeight width w (width + (t + 1)) : ℝ) ≤
            ((a : ℝ) / b) * totalWeight width w (width + t) := by
          rw [div_mul_eq_mul_div]
          apply (le_div_iff₀ hb').mpr
          simpa only [Nat.add_assoc, mul_comm] using hstep
        _ ≤ ((a : ℝ) / b) *
              (totalWeight width w width * ((a : ℝ) / b) ^ t) :=
          mul_le_mul_of_nonneg_left ih (by positivity)
        _ = totalWeight width w width * ((a : ℝ) / b) ^ (t + 1) := by
          rw [pow_succ]
          ring

end FiniteWindowUpper
end RubiksSnake

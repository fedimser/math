import RubiksSnake.FiniteWindowUpperBound
import Mathlib.Tactic.IntervalCases

/-!
# A small finite-window upper certificate

Every valid word has valid six-symbol subwords. We keep its last five symbols
and allow an extension exactly when the resulting six-symbol word is valid.
This graph has 920 valid states and 3384 edges; longer-range collisions can
only remove extensions.

Twelve sparse adjacency-vector products generate a nonnegative integer
potential. The certificate checks its inequalities directly against
`ValidList` and `canAppend`, not against an externally supplied graph:

* `5000 * outgoingWeight <= 18601 * weight`;
* `1093187 * outgoingDegree <= weight`.

Some valid states have no successors, and their potential is zero. The second
inequality controls the final extension separately, so no positivity assumption
on the potential is needed. Summing over the actual prefix tree gives
`countValidFormulas k <= (9/4) * (18601/5000)^k` for every `k`.
-/

namespace RubiksSnake

namespace WindowUpper

/-- The four rotation labels, each listed once when enumerating local extensions. -/
def rotations : List Rotation := [0, 1, 2, 3]

/-- Retains the last five rotations, or the whole word when fewer than five
rotations have been chosen. -/
def suffix (rs : List Rotation) : List Rotation :=
  rs.drop (rs.length - 5)

/-- Shifts a nonempty window by discarding its oldest rotation and appending
the new one, without performing a collision check. -/
def next (rs : List Rotation) (r : Rotation) : List Rotation :=
  rs.tail ++ [r]

/-- Weights shifted windows for turns accepted by `canAppend`; on a valid
five-rotation source this checks the resulting six-rotation word. -/
def outgoing (w : List Rotation → ℕ) (rs : List Rotation) : ℕ :=
  (rotations.map fun r => if canAppend rs r then w (next rs r) else 0).sum

/-- Number of rotation labels accepted by `canAppend`, used to control the
final extension separately from the potential's growth. -/
def degree (rs : List Rotation) : ℕ :=
  outgoing (fun _ => 1) rs

/-- Big-endian base-four index used for five-rotation windows; the word's length
is not encoded, unlike the variable-length prefix keys. -/
private def encode (rs : List Rotation) : ℕ :=
  rs.foldl (fun i r => 4 * i + r.val) 0

/-- Converts an index below `4^5` into exactly five rotation digits, retaining
leading zero rotations. -/
private def decode (i : Fin 1024) : List Rotation :=
  List.ofFn fun j : Fin 5 =>
    (⟨(i.val / 4 ^ (4 - j.val)) % 4, Nat.mod_lt _ (by decide)⟩ : Rotation)

/-- Full 1024-row graph of five-rotation windows, with empty rows for invalid
windows and one edge for each locally allowed rotation at a valid window. -/
private def edges : Array (List ℕ) :=
  Array.ofFn fun i : Fin 1024 =>
    let rs := decode i
    if ValidList rs then
      rotations.filterMap fun r =>
        if canAppend rs r then some (encode (next rs r)) else none
    else []

/-- Exact adjacency iterates from the all-ones vector on the five-window graph,
counting local paths of the requested length. -/
private def iterateWeights (n : ℕ) : Array ℕ :=
  WindowComputation.iterateWeights edges n

/-- Candidate potential after twelve adjacency iterations, whose counting
usefulness is established by the finite integer certificate. -/
private def potential : Array ℕ := iterateWeights 12

/-- Array-backed five-window potential; out-of-range base-four indices have
weight zero. -/
def weight (rs : List Rotation) : ℕ :=
  potential[encode rs]?.getD 0

/-- Only the five-symbol window graph, not longer snake words, is enumerated. -/
theorem edge_count : (edges.toList.map List.length).sum = 3384 := by
  native_decide

/-- Native proof that every valid five-rotation formula satisfies one-step
terminal domination at scale `1093187` and the exact integer outgoing bound
`5000 * outgoing weight <= 18601 * weight`. -/
private theorem finite_certificate :
    ∀ f : Formula 5, Valid f →
      1093187 * degree (List.ofFn f) ≤ weight (List.ofFn f) ∧
      5000 * outgoing weight (List.ofFn f) ≤ 18601 * weight (List.ofFn f) := by
  native_decide

/-- List-form certificate for a valid five-rotation window, combining final-step
count domination with the exact rational outgoing bound of ratio `18601 / 5000`. -/
lemma certificate (rs : List Rotation) (hlen : rs.length = 5)
    (hvalid : ValidList rs) :
    1093187 * degree rs ≤ weight rs ∧
      5000 * outgoing weight rs ≤ 18601 * weight rs := by
  have hvalid' : Valid (formulaOfList rs hlen) := by
    change ValidList (List.ofFn (formulaOfList rs hlen))
    rw [ofFn_formulaOfList]
    exact hvalid
  simpa only [ofFn_formulaOfList] using
    finite_certificate (formulaOfList rs hlen) hvalid'

/-- Sum of retained five-window potentials over actual valid words with `k`
rotations (`k + 1` wedges), not merely over distinct suffix states. -/
def totalWeight (k : ℕ) : ℕ :=
  ((validRotationLists k).map fun rs => weight (suffix rs)).sum

/-- Identifies this five-window total with the generic finite-window total,
allowing the universal extension and terminal lemmas to be reused. -/
private lemma totalWeight_eq (k : ℕ) :
    totalWeight k = FiniteWindowUpper.totalWeight 5 weight k := by
  simp only [totalWeight, FiniteWindowUpper.totalWeight, WindowComputation.validWords_eq,
    suffix, FiniteWindowUpper.suffix]

/-- Once five rotations are present, the exact statewise integer inequality
lifts to the same one-step growth inequality for actual weighted totals. -/
lemma totalWeight_step (k : ℕ) (hk : 5 ≤ k) :
    5000 * totalWeight (k + 1) ≤ 18601 * totalWeight k := by
  rw [totalWeight_eq, totalWeight_eq]
  apply FiniteWindowUpper.totalWeight_step 5 (by decide) weight 18601 5000 k hk
  intro rs hlen hvalid
  rw [WindowComputation.outgoing_eq_canAppend weight rs hvalid]
  exact (certificate rs hlen hvalid).2

/-- For `k >= 5`, the potential total dominates `1093187` times the actual
`(k + 1)`-rotation formula count, reserving one terminal step for dead ends. -/
lemma count_next_le_totalWeight (k : ℕ) (hk : 5 ≤ k) :
    1093187 * countValidFormulas (k + 1) ≤ totalWeight k := by
  rw [totalWeight_eq]
  apply FiniteWindowUpper.count_terminal_le 5 (by decide) weight 1093187 1 k hk
  intro rs hlen hvalid
  change 1093187 * WindowComputation.outgoing (fun _ => 1) rs ≤ weight rs
  rw [WindowComputation.outgoing_eq_canAppend _ rs hvalid]
  exact (certificate rs hlen hvalid).1

/-- Exact initial sum of potentials at five rotations, distinct from the
number of valid five-rotation window states. -/
lemma totalWeight_five : totalWeight 5 = 6389332796 := by
  native_decide

/-- Bounds actual weighted totals after the starting five rotations by the
certified initial sum times the exact geometric factor `(18601 / 5000)^t`. -/
lemma totalWeight_bound (t : ℕ) :
    (totalWeight (5 + t) : ℝ) ≤ 6389332796 * (18601 / 5000 : ℝ) ^ t := by
  induction t with
  | zero => norm_num [totalWeight_five]
  | succ t ih =>
      have hstep :
          5000 * (totalWeight (5 + t + 1) : ℝ) ≤
            18601 * (totalWeight (5 + t) : ℝ) := by
        exact_mod_cast totalWeight_step (5 + t) (by omega)
      calc
        (totalWeight (5 + (t + 1)) : ℝ) ≤
            (18601 / 5000 : ℝ) * totalWeight (5 + t) := by
          simpa only [Nat.add_assoc] using (by linarith :
            (totalWeight (5 + t + 1) : ℝ) ≤
              (18601 / 5000 : ℝ) * totalWeight (5 + t))
        _ ≤ (18601 / 5000 : ℝ) *
              (6389332796 * (18601 / 5000 : ℝ) ^ t) :=
          mul_le_mul_of_nonneg_left ih (by norm_num)
        _ = 6389332796 * (18601 / 5000 : ℝ) ^ (t + 1) := by
          rw [pow_succ]
          ring

/-- Uses the five-rotation starting window and one terminal step to bound
valid `(6 + t)`-rotation formulas with prefactor `9 / 4` and exact base
`18601 / 5000`. -/
lemma count_bound_from_six (t : ℕ) :
    (countValidFormulas (6 + t) : ℝ) ≤
      (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ (6 + t) := by
  have hcount :
      1093187 * (countValidFormulas (6 + t) : ℝ) ≤
        (totalWeight (5 + t) : ℝ) := by
    have h := count_next_le_totalWeight (5 + t) (by omega)
    rw [show 5 + t + 1 = 6 + t by omega] at h
    exact_mod_cast h
  calc
    (countValidFormulas (6 + t) : ℝ) ≤ (totalWeight (5 + t) : ℝ) / 1093187 := by
      linarith
    _ ≤ (6389332796 * (18601 / 5000 : ℝ) ^ t) / 1093187 :=
      div_le_div_of_nonneg_right (totalWeight_bound t) (by norm_num)
    _ = (6389332796 / 1093187 : ℝ) * (18601 / 5000 : ℝ) ^ t := by ring
    _ ≤ ((9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ 6) *
          (18601 / 5000 : ℝ) ^ t := by
      apply mul_le_mul_of_nonneg_right
      · norm_num
      · positivity
    _ = (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ (6 + t) := by
      rw [pow_add]
      ring

end WindowUpper

/-- The five-symbol certificate bounds every valid formula, including lengths
below the window size; no asymptotic qualification is needed. -/
theorem countValidFormulas_le_window_upper (k : ℕ) :
    (countValidFormulas k : ℝ) ≤
      (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ k := by
  by_cases hk : 6 ≤ k
  · have h := WindowUpper.count_bound_from_six (k - 6)
    simpa [Nat.add_sub_of_le hk] using h
  · have hsmall : k < 6 := by omega
    calc
      (countValidFormulas k : ℝ) ≤ (4 : ℝ) ^ k := by
        exact_mod_cast countFormulas_upper_bound k
      _ ≤ (9 / 4 : ℝ) * (18601 / 5000 : ℝ) ^ k := by
        interval_cases k <;> norm_num

end RubiksSnake

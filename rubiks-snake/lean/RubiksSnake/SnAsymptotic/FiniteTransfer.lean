import RubiksSnake.SnAsymptotic.BoundedComponents
import Mathlib.Data.Finset.Max
import Mathlib.Topology.Algebra.InfiniteSum.NatInt

/-!
# Lower certificates for finite weighted transfers

A nonzero nonnegative subeigenvector forces divergence of a transfer generating
series. Coefficients may be zero, so forbidden transitions need no separate
state. No irreducibility or floating-point spectral computation is assumed.
An amortized rank inequality bounds zero-cost steps and excludes zero-cost cycles.

The final growth comparison is conditional: a geometric construction must still
prove summability above `snakeGrowthConstant`, for example by encoding its traces
as a bounded number of snake components.
-/

namespace RubiksSnake.FiniteTransfer

noncomputable section

variable {σ α : Type*} [Fintype α]

/-- Apply one weighted transfer step to a state potential, summing over all outgoing labels.
A zero coefficient represents a forbidden transition. -/
def step (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (v : σ → ℝ) (s : σ) : ℝ :=
  ∑ a, coefficient s a * v (next s a)

/-- Nonnegative transition coefficients preserve pointwise inequalities between potentials. -/
lemma step_mono (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) :
    Monotone (step next coefficient) := by
  intro u v huv s
  exact Finset.sum_le_sum fun a _ =>
    mul_le_mul_of_nonneg_left (huv (next s a)) (hc s a)

/-- A common scalar factor in the potential can be pulled outside a transfer step. -/
lemma step_const_mul (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (c : ℝ) (v : σ → ℝ) (s : σ) :
    step next coefficient (fun t => c * v t) s = c * step next coefficient v s := by
  simp only [step, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  ring

/-- Total weight of all label words of a given number of construction steps, starting at a state.
The empty word has weight one; step count need not equal wedge count. -/
def mass (next : σ → α → σ) (coefficient : σ → α → ℝ) : ℕ → σ → ℝ
  | 0, _ => 1
  | n + 1, s => step next coefficient (mass next coefficient n) s

/-- Every fixed-length transfer mass is nonnegative when all transition coefficients are. -/
lemma mass_nonneg (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (n : ℕ) (s : σ) :
    0 ≤ mass next coefficient n s := by
  induction n generalizing s with
  | zero => exact zero_le_one
  | succ n ih =>
      exact Finset.sum_nonneg fun a _ => mul_nonneg (hc s a) (ih (next s a))

section WordCosts

omit [Fintype α]

/-- Multiply the transition coefficients encountered along a label word from its initial state. -/
def wordWeight (next : σ → α → σ) (coefficient : σ → α → ℝ) :
    σ → List α → ℝ
  | _, [] => 1
  | s, a :: word => coefficient s a * wordWeight next coefficient (next s a) word

/-- Add the nonnegative integer costs along a label word, for example newly introduced wedges. -/
def totalCost (next : σ → α → σ) (cost : σ → α → ℕ) : σ → List α → ℕ
  | _, [] => 0
  | s, a :: word => cost s a + totalCost next cost (next s a) word

/-- A uniform per-transition cost bound gives the corresponding linear bound on total word cost. -/
theorem totalCost_le_mul_length (next : σ → α → σ) (cost : σ → α → ℕ)
    (L : ℕ) (hcost : ∀ s a, cost s a ≤ L) (s : σ) (word : List α) :
    totalCost next cost s word ≤ L * word.length := by
  induction word generalizing s with
  | nil => simp [totalCost]
  | cons a word ih =>
      have htail := ih (next s a)
      have hhead := hcost s a
      simp only [totalCost, List.length_cons, Nat.mul_add, Nat.mul_one]
      omega

/-- Products of nonnegative transition coefficients give nonnegative weights for every word. -/
lemma wordWeight_nonneg (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (s : σ) (word : List α) :
    0 ≤ wordWeight next coefficient s word := by
  induction word generalizing s with
  | nil => exact zero_le_one
  | cons a word ih => exact mul_nonneg (hc s a) (ih (next s a))

/-- A decreasing rank accounts for zero-cost steps without charging them as wedges. -/
theorem rank_add_length_le (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (cost : σ → α → ℕ)
    (rank : σ → ℕ) (A : ℕ)
    (hdrift : ∀ s a, 0 < coefficient s a →
      rank (next s a) + 1 ≤ rank s + A * cost s a)
    (s : σ) (word : List α) (hword : 0 < wordWeight next coefficient s word) :
    rank (word.foldl next s) + word.length ≤ rank s + A * totalCost next cost s word := by
  induction word generalizing s with
  | nil => simp [totalCost]
  | cons a word ih =>
      change 0 < coefficient s a * wordWeight next coefficient (next s a) word at hword
      obtain ⟨hfirst, hrest⟩ :
          0 < coefficient s a ∧ 0 < wordWeight next coefficient (next s a) word := by
        rcases mul_pos_iff.mp hword with h | h
        · exact h
        · linarith [hc s a]
      have htail := ih (next s a) hrest
      have hhead := hdrift s a hfirst
      simp only [List.foldl_cons, List.length_cons, totalCost, Nat.mul_add]
      omega

/-- A rank decrease pays for zero-cost steps: with initial rank at most `A`, a positive-weight
word has at most `A * (totalCost + 1)` construction steps. -/
theorem length_le_mul_totalCost_add_one
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (cost : σ → α → ℕ)
    (rank : σ → ℕ) (A : ℕ)
    (hdrift : ∀ s a, 0 < coefficient s a →
      rank (next s a) + 1 ≤ rank s + A * cost s a)
    (s : σ) (hstart : rank s ≤ A) (word : List α)
    (hword : 0 < wordWeight next coefficient s word) :
    word.length ≤ A * (totalCost next cost s word + 1) := by
  have h := rank_add_length_le next coefficient hc cost rank A hdrift s word hword
  rw [Nat.mul_add, Nat.mul_one]
  omega

/-- Under the rank drift inequality, a positive-weight cycle of total cost zero must be empty. -/
theorem zero_cost_cycle_eq_nil
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (cost : σ → α → ℕ)
    (rank : σ → ℕ) (A : ℕ)
    (hdrift : ∀ s a, 0 < coefficient s a →
      rank (next s a) + 1 ≤ rank s + A * cost s a)
    (s : σ) (word : List α) (hword : 0 < wordWeight next coefficient s word)
    (hcycle : word.foldl next s = s) (hcost : totalCost next cost s word = 0) :
    word = [] := by
  have h := rank_add_length_le next coefficient hc cost rank A hdrift s word hword
  rw [hcycle, hcost, Nat.mul_zero, Nat.add_zero] at h
  exact List.length_eq_zero_iff.mp (by omega)

end WordCosts

/-- Iterating the transfer operator equals summing the products of coefficients over all
fixed-length label words. -/
theorem mass_eq_sum_wordWeight (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (n : ℕ) (s : σ) :
    mass next coefficient n s =
      ∑ word : Fin n → α, wordWeight next coefficient s (List.ofFn word) := by
  induction n generalizing s with
  | zero => simp [mass, wordWeight]
  | succ n ih =>
      calc
        mass next coefficient (n + 1) s =
            ∑ a, coefficient s a *
              (∑ word : Fin n → α, wordWeight next coefficient (next s a) (List.ofFn word)) := by
          simp only [mass, step, ih]
        _ = ∑ p : α × (Fin n → α),
            wordWeight next coefficient s (List.ofFn (Fin.cons p.1 p.2)) := by
          rw [Fintype.sum_prod_type]
          simp only [List.ofFn_cons, wordWeight, Finset.mul_sum]
        _ = ∑ word : Fin (n + 1) → α,
            wordWeight next coefficient s (List.ofFn word) :=
          (Fin.consEquiv (fun _ : Fin (n + 1) => α)).sum_comp
            (fun word => wordWeight next coefficient s (List.ofFn word))

/-- A nonnegative potential satisfying `1 + T v <= v` bounds every finite sum of transfer masses. -/
theorem mass_partialSum_le_of_supersolution
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (v : σ → ℝ)
    (hv : ∀ s, 0 ≤ v s)
    (hsuper : ∀ s, 1 + step next coefficient v s ≤ v s)
    (n : ℕ) (s : σ) :
    (∑ k ∈ Finset.range n, mass next coefficient k s) ≤ v s := by
  induction n generalizing s with
  | zero => simpa using hv s
  | succ n ih =>
      calc
        (∑ k ∈ Finset.range (n + 1), mass next coefficient k s) =
            (∑ k ∈ Finset.range n, mass next coefficient (k + 1) s) + 1 :=
          Finset.sum_range_succ' _ _
        _ = step next coefficient
            (fun t => ∑ k ∈ Finset.range n, mass next coefficient k t) s + 1 := by
          congr 1
          simp only [mass, step, Finset.mul_sum]
          exact Finset.sum_comm
        _ ≤ step next coefficient v s + 1 :=
          add_le_add (step_mono next coefficient hc ih s) le_rfl
        _ ≤ v s := by simpa only [add_comm] using hsuper s

/-- A finite nonnegative renewal supersolution proves convergence of the transfer series
from each starting state. -/
theorem summable_mass_of_supersolution
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (v : σ → ℝ)
    (hv : ∀ s, 0 ≤ v s)
    (hsuper : ∀ s, 1 + step next coefficient v s ≤ v s)
    (s : σ) :
    Summable (fun n => mass next coefficient n s) :=
  summable_of_sum_range_le (fun n => mass_nonneg next coefficient hc n s)
    (fun n => mass_partialSum_le_of_supersolution next coefficient hc v hv hsuper n s)

variable [Fintype σ]

/-- A subeigenvector is incompatible with a finite nonnegative renewal supersolution. -/
theorem no_nonnegative_supersolution
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (w v : σ → ℝ)
    (hw : ∀ s, 0 ≤ w s) (hwpos : ∃ s, 0 < w s)
    (hsub : ∀ s, w s ≤ step next coefficient w s)
    (hv : ∀ s, 0 ≤ v s)
    (hsuper : ∀ s, 1 + step next coefficient v s ≤ v s) : False := by
  classical
  let support := Finset.univ.filter fun s => 0 < w s
  have hsupport : support.Nonempty := by
    obtain ⟨s, hs⟩ := hwpos
    exact ⟨s, Finset.mem_filter.mpr ⟨Finset.mem_univ s, hs⟩⟩
  obtain ⟨s, hs, hmin⟩ :=
    support.exists_min_image (fun t => v t / w t) hsupport
  have hspos : 0 < w s := (Finset.mem_filter.mp hs).2
  let c := v s / w s
  have hcpos : 0 ≤ c := div_nonneg (hv s) hspos.le
  have hdom (t : σ) : c * w t ≤ v t := by
    by_cases ht : 0 < w t
    · exact (le_div_iff₀ ht).mp
        (hmin t (Finset.mem_filter.mpr ⟨Finset.mem_univ t, ht⟩))
    · have htzero : w t = 0 := le_antisymm (le_of_not_gt ht) (hw t)
      simpa only [htzero, mul_zero] using hv t
  have hstep : c * w s ≤ step next coefficient v s := calc
    c * w s ≤ c * step next coefficient w s :=
      mul_le_mul_of_nonneg_left (hsub s) hcpos
    _ = step next coefficient (fun t => c * w t) s :=
      (step_const_mul next coefficient c w s).symm
    _ ≤ step next coefficient v s := step_mono next coefficient hc hdom s
  have hcancel : c * w s = v s := div_mul_cancel₀ _ hspos.ne'
  rw [hcancel] at hstep
  linarith [hsuper s]

/-- On a finite state space, a nonzero nonnegative vector satisfying `w <= T w` forces
divergence from at least one state; reachability from a particular root is not asserted. -/
theorem not_all_summable_of_subeigenvector
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (w : σ → ℝ)
    (hw : ∀ s, 0 ≤ w s) (hwpos : ∃ s, 0 < w s)
    (hsub : ∀ s, w s ≤ step next coefficient w s) :
    ¬ ∀ s, Summable (fun n => mass next coefficient n s) := by
  intro hsummable
  let v (s : σ) := ∑' n, mass next coefficient n s
  have hv (s : σ) : 0 ≤ v s :=
    tsum_nonneg fun n => mass_nonneg next coefficient hc n s
  have hrec (s : σ) : v s = 1 + step next coefficient v s := by
    calc
      v s = mass next coefficient 0 s + ∑' n, mass next coefficient (n + 1) s :=
        (hsummable s).tsum_eq_zero_add
      _ = 1 + ∑ a, ∑' n, coefficient s a * mass next coefficient n (next s a) := by
        simp only [mass, step]
        congr 1
        exact Summable.tsum_finsetSum fun a _ =>
          (hsummable (next s a)).mul_left (coefficient s a)
      _ = 1 + step next coefficient v s := by
        congr 1
        apply Finset.sum_congr rfl
        intro a _
        exact (hsummable (next s a)).tsum_mul_left (coefficient s a)
  exact no_nonnegative_supersolution next coefficient hc w v hw hwpos hsub hv
    (fun s => (hrec s).symm.le)

/-- A transfer subeigenvector gives `q <= mu` once a separate geometric comparison proves
that every starting-state series would converge if `mu < q`. -/
theorem le_snakeGrowthConstant_of_transfer {q : ℝ}
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (w : σ → ℝ)
    (hw : ∀ s, 0 ≤ w s) (hwpos : ∃ s, 0 < w s)
    (hsub : ∀ s, w s ≤ step next coefficient w s)
    (hcomparison : snakeGrowthConstant < q →
      ∀ s, Summable (fun n => mass next coefficient n s)) :
    q ≤ snakeGrowthConstant :=
  le_of_not_gt fun hq =>
    not_all_summable_of_subeigenvector next coefficient hc w hw hwpos hsub (hcomparison hq)

/-- Turn a transfer subeigenvector into a snake lower bound when weighted trace counts are
dominated by polynomially many bounded-component encodings at each total wedge count.
The coefficient-count comparison remains a hypothesis to prove for a concrete construction. -/
theorem le_snakeGrowthConstant_of_component_counts {q : ℝ}
    (next : σ → α → σ) (coefficient : σ → α → ℝ)
    (hc : ∀ s a, 0 ≤ coefficient s a) (w : σ → ℝ)
    (hw : ∀ s, 0 ≤ w s) (hwpos : ∃ s, 0 < w s)
    (hsub : ∀ s, w s ≤ step next coefficient w s)
    (c : σ → ℕ → ℕ → ℕ) (M D K : ℕ)
    (hmass : ∀ s N, mass next coefficient N s ≤ ∑' n, (c s N n : ℝ) / q ^ n)
    (hcount : ∀ s n (steps : Finset ℕ),
      ∑ N ∈ steps, c s N n ≤ K * (n + 1) ^ D * boundedComponentWedgeCount M n) :
    q ≤ snakeGrowthConstant := by
  apply le_snakeGrowthConstant_of_transfer next coefficient hc w hw hwpos hsub
  intro hq s
  exact Summable.of_nonneg_of_le (fun N => mass_nonneg next coefficient hc N s)
    (hmass s) (summable_of_component_coefficient_bound hq (c s) M D K (hcount s))

end

end RubiksSnake.FiniteTransfer

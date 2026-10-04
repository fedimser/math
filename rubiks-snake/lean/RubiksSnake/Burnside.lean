import RubiksSnake.ReversalTransform
import RubiksSnake.RotationRestricted

/-!
  Here we prove theorem that generalizes observation in
  https://github.com/fedimser/math/blob/master/rubiks-snake/count-shapes-with-reversal.ipynb

  Let D_n - number of n-wedge shapes "up to" some transformation.
  Let F_n - number of n-wedge shapes fixed by that transformation.
  Then 2*D_n = F_n + S_n
 -/

namespace RubiksSnake


noncomputable section

def validFormulaTransform {n : ℕ} (t : InvolutiveFormulaTransform n) :
    {w : Formula n // Valid w} → {w : Formula n // Valid w} :=
  fun w => ⟨t w, (t.valid_iff w).mpr w.property⟩

@[simp] lemma validFormulaTransform_involutive {n : ℕ}
    (t : InvolutiveFormulaTransform n) :
    Function.Involutive (validFormulaTransform t) := by
  intro w
  apply Subtype.ext
  exact t.involutive w

def transformOrbitSetoid {n : ℕ} (t : InvolutiveFormulaTransform n) :
    Setoid {w : Formula n // Valid w} where
  r a b := a = b ∨ validFormulaTransform t a = b
  iseqv := by
    refine ⟨fun a => Or.inl rfl, ?_, ?_⟩
    · intro a b hab
      rcases hab with rfl | hab
      · exact Or.inl rfl
      · exact Or.inr <| by
          rw [← hab]
          exact validFormulaTransform_involutive t a
    · intro a b c hab hbc
      rcases hab with rfl | hab <;> rcases hbc with rfl | hbc
      · exact Or.inl rfl
      · exact Or.inr hbc
      · exact Or.inr hab
      · exact Or.inl <| by
          rw [← hbc, ← hab]
          exact (validFormulaTransform_involutive t a).symm

/-- Number of formulas fixed by transform t. -/
def fixedShapeCount (n : ℕ+)
    (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) : ℕ :=
  countFormulas ((n : ℕ) - 1) fun w => Valid w ∧ t w = w

/--
Number of orbits of valid formulas under an involutive transform.
-/
def shapesUpToTransform
    (n : ℕ+) (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) : ℕ :=
  Nat.card (Quotient (transformOrbitSetoid t))

theorem BurnsideForRubiksSnake (n : ℕ+)
    (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) :
    S n + fixedShapeCount n t = 2 * shapesUpToTransform n t := by
  sorry



/-- Definition of reflection transform. -/
def reflectionTransform (n : ℕ) : InvolutiveFormulaTransform n where
  toFun := mirrorFormula
  involutive := mirrorFormula_involutive
  valid_iff := valid_mirrorFormula

lemma reverseFormula_mirrorFormula {n : ℕ} (w : Formula n) :
    reverseFormula (mirrorFormula w) = mirrorFormula (reverseFormula w) := by
  rfl

/-- Definition of reflection+reversal transform. -/
def reversalReflectionTransform (n : ℕ) : InvolutiveFormulaTransform n where
  toFun w := reverseFormula (mirrorFormula w)
  involutive w := by
    dsimp
    rw [reverseFormula_mirrorFormula]
    simp
  valid_iff w := by
    rw [valid_reverseFormula, valid_mirrorFormula]



/-- Shapes fixed by head-tail reversal. -/
def F (n : ℕ+) : ℕ := fixedShapeCount n (reversalTransform _)

/-- Shapes up to reversal. -/
def D (n : ℕ+) : ℕ := shapesUpToTransform n (reversalTransform _)

theorem BurnsideCorollary1 (n : ℕ+) :
    S n + F n = 2 * D n := by
  simpa [F, D] using BurnsideForRubiksSnake n (reversalTransform _)

theorem BurnsideCorollary2 (n : ℕ+) :
    S n + fixedShapeCount n (reflectionTransform _) =
      2 * shapesUpToTransform n (reflectionTransform _) := by
  exact BurnsideForRubiksSnake n (reflectionTransform _)

/-- Code below corresponds to
 https://github.com/fedimser/math/blob/master/rubiks-snake/count-loops.ipynb
and needs to be rewritten to use Burnside lemma.
 -/

def reflectionFixed (n : ℕ+) : ℕ :=
  fixedShapeCount n (reflectionTransform _)

def reversalReflectionFixed (n : ℕ+) : ℕ :=
  fixedShapeCount n (reversalReflectionTransform _)

def shapesUpToReflection (n : ℕ+) : ℕ :=
  (S n + reflectionFixed n) / 2

def shapesUpToReversalAndReflection (n : ℕ+) : ℕ :=
  (S n + F n + reflectionFixed n + reversalReflectionFixed n) / 4

def rotateFormula {n : ℕ} (k : ℕ) (w : Formula n) : Formula n :=
  fun i => w ⟨(i.1 + k) % n, Nat.mod_lt _ (Nat.zero_lt_of_lt i.2)⟩

def cyclicValid {n : ℕ} (w : Formula n) : Prop :=
  let ds := directions (List.ofFn w)
  collisionFree (List.ofFn w) ∧
    (centersFromDirections ds).getLastD zeroVec = zeroVec ∧
    ds.getLastD zeroVec = ex

def cyclicFixedCount (n : ℕ+) (k : ℕ) : ℕ :=
  countFormulas n fun w => cyclicValid w ∧ rotateFormula k w = w

def reflectionCyclicFixedCount (n : ℕ+) (k : ℕ) : ℕ :=
  countFormulas n fun w =>
    cyclicValid w ∧ rotateFormula k (reverseFormula w) = w

/-- Formulas describing loops, with the closing joint included in the encoding. -/
def L1 (n : ℕ+) : ℕ :=
  countFormulas n cyclicValid

/-- Loops up to reversal. -/
def L2 (n : ℕ+) : ℕ :=
  (L1 n + reflectionCyclicFixedCount n 0) / 2

/-- Loops up to cyclic shifts. -/
def L3 (n : ℕ+) : ℕ :=
  (∑ k ∈ Finset.range n, cyclicFixedCount n k) / n

/-- Loops up to cyclic shifts and reversal. -/
def L4 (n : ℕ+) : ℕ :=
  ((∑ k ∈ Finset.range n, cyclicFixedCount n k) +
    ∑ k ∈ Finset.range n, reflectionCyclicFixedCount n k) / (2 * n)

/-- The Burnside auxiliary `X(n,k)`: words whose `k`-fold repetition is a loop. -/
def X (n k : ℕ+) : ℕ :=
  countFormulas n fun w => cyclicValid (Fin.repeat k w)

/-- The reflection term in the dihedral Burnside sum. -/
def XR (n : ℕ+) (k : ℕ) : ℕ :=
  reflectionCyclicFixedCount n k



end

end RubiksSnake

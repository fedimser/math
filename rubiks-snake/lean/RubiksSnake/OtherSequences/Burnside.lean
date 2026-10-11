import RubiksSnake.ComputeSnakes
import RubiksSnake.Transforms.ReversalTransform
import RubiksSnake.OtherSequences.RotationRestricted
import RubiksSnake.Transforms.ReflectionTransform
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Fintype.Quotient
import Mathlib.GroupTheory.GroupAction.Quotient
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith

/-!
  Here we formalize results connecting number of formulas "fixed by" some
  transform and "up to" some transform.

  This formalizes:
    * Section 5.3.1 in https://github.com/fedimser/math/blob/master/rubiks-snake/count-shapes.ipynb
    * https://github.com/fedimser/math/blob/master/rubiks-snake/count-shapes-with-reversal.ipynb
 -/

namespace RubiksSnake


noncomputable section

/-- Restrict a validity-preserving involution to the subtype of valid
`n`-rotation formulas, carrying the validity proof with each image. -/
def validFormulaTransform {n : ℕ} (t : InvolutiveFormulaTransform n) :
    {w : Formula n // Valid w} → {w : Formula n // Valid w} :=
  fun w => ⟨t w, (t.valid_iff w).mpr w.property⟩

/-- Restricting the transform to valid formulas preserves its twofold cancellation law. -/
@[simp] lemma validFormulaTransform_involutive {n : ℕ}
    (t : InvolutiveFormulaTransform n) :
    Function.Involutive (validFormulaTransform t) := by
  intro w
  apply Subtype.ext
  exact t.involutive w

/-- Identify valid formulas that are equal or exchanged by the involution.
Each equivalence class consists of a fixed point or a two-element orbit. -/
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

/-- Number of valid `(n - 1)`-rotation formulas fixed by `t`, representing
`n`-wedge snakes; invalid fixed formulas are not counted. -/
def fixedShapeCount (n : ℕ+)
    (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) : ℕ :=
  countFormulas ((n : ℕ) - 1) fun w => Valid w ∧ t w = w

/--
Number of orbits of valid formulas under an involutive transform.
-/
def shapesUpToTransform
    (n : ℕ+) (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) : ℕ :=
  Nat.card (Quotient (transformOrbitSetoid t))

/-- For an involution on valid `n`-wedge formulas, the total count plus the
fixed-point count equals twice the number of orbits. -/
theorem BurnsideForRubiksSnake (n : ℕ+)
    (t : InvolutiveFormulaTransform ((n : ℕ) - 1)) :
    S n + fixedShapeCount n t = 2 * shapesUpToTransform n t := by
  let k := (n : ℕ) - 1
  let X := {w : Formula k // Valid w}
  let f : X → X := validFormulaTransform t
  let _ : AddAction (ZMod 2) X := {
    vadd g w := if g = 0 then w else f w
    zero_vadd _ := rfl
    add_vadd g h w := by
      fin_cases g <;> fin_cases h
      · rfl
      · rfl
      · rfl
      · change w = validFormulaTransform t (validFormulaTransform t w)
        exact (validFormulaTransform_involutive t w).symm
  }

  let fixedZeroEquiv : AddAction.fixedBy X (0 : ZMod 2) ≃ X := {
    toFun w := w.val
    invFun w := ⟨w, rfl⟩
    left_inv _ := rfl
    right_inv _ := rfl
  }

  let fixedOneEquiv :
      AddAction.fixedBy X (1 : ZMod 2) ≃
        {w : Formula k // Valid w ∧ t w = w} := {
    toFun w :=
      ⟨w.val.val, w.val.property, congrArg Subtype.val w.property⟩
    invFun w :=
      ⟨⟨w.val, w.property.1⟩, Subtype.ext w.property.2⟩
    left_inv _ := rfl
    right_inv _ := rfl
  }

  let orbitEquiv :
      Quotient (AddAction.orbitRel (ZMod 2) X) ≃
        Quotient (transformOrbitSetoid t) :=
    Quotient.congr (Equiv.refl X) fun a b => by
      change (∃ g : ZMod 2, g +ᵥ b = a) ↔
        a = b ∨ validFormulaTransform t a = b
      constructor
      · rintro ⟨g, hg⟩
        fin_cases g
        · exact Or.inl hg.symm
        · right
          rw [← hg]
          exact validFormulaTransform_involutive t b
      · rintro (rfl | hab)
        · exact ⟨0, rfl⟩
        · refine ⟨1, ?_⟩
          change validFormulaTransform t b = a
          rw [← hab]
          exact validFormulaTransform_involutive t a

  have hsum (counts : ZMod 2 → ℕ) :
      (∑ g, counts g) = counts 0 + counts 1 := by
    rw [← Equiv.sum_comp (ZMod.finEquiv 2).toEquiv counts]
    simp [Fin.sum_univ_succ]

  let _ : DecidableRel (AddAction.orbitRel (ZMod 2) X) :=
    fun _ _ => Classical.propDecidable _
  let _ : DecidableRel (transformOrbitSetoid t) :=
    fun _ _ => Classical.propDecidable _

  have hburnside :=
    AddAction.sum_card_fixedBy_eq_card_orbits_mul_card_addGroup
      (ZMod 2) X
  rw [hsum] at hburnside
  have hgroup : Fintype.card (ZMod 2) = 2 := ZMod.card 2
  rw [hgroup] at hburnside

  have hS : S n = Fintype.card X := by
    unfold S countValidFormulas countFormulas
    rw [Nat.card_eq_fintype_card]

  have hfixed :
      fixedShapeCount n t =
        Fintype.card (AddAction.fixedBy X (1 : ZMod 2)) := by
    unfold fixedShapeCount countFormulas
    rw [Nat.card_eq_fintype_card]
    exact (Fintype.card_congr fixedOneEquiv).symm

  have hzero :
      Fintype.card (AddAction.fixedBy X (0 : ZMod 2)) =
        Fintype.card X :=
    Fintype.card_congr fixedZeroEquiv

  have horbits :
      Fintype.card (Quotient (AddAction.orbitRel (ZMod 2) X)) =
        shapesUpToTransform n t := by
    unfold shapesUpToTransform
    rw [Nat.card_eq_fintype_card]
    exact Fintype.card_congr orbitEquiv

  rw [hzero, ← hS, ← hfixed, horbits] at hburnside
  simpa [Nat.mul_comm] using hburnside


/-- Head-tail reversal and reflection commute: reversing index order does not
affect the entrywise exchange of rotation symbols `1` and `3`. -/
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
def S_FIX_REV (n : ℕ+) : ℕ := fixedShapeCount n (reversalTransform _)

/-- Shapes up to reversal. -/
def S_UT_REV (n : ℕ+) : ℕ := shapesUpToTransform n (reversalTransform _)

/-- Twice the number of reversal classes is the unrestricted `n`-wedge count
plus the count fixed by head-tail reversal. -/
lemma BurnsideForReversal (n : ℕ+) :
    S n + S_FIX_REV n = 2 * S_UT_REV n := by
  simpa [S_FIX_REV, S_UT_REV] using BurnsideForRubiksSnake n (reversalTransform _)

/-- Shapes fixed by reflection. -/
def S_FIX_REFL (n : ℕ+) : ℕ := fixedShapeCount n (reflectionTransform _)

/-- Shapes up to reflection. -/
def S_UT_REFL (n : ℕ+) : ℕ := shapesUpToTransform n (reflectionTransform _)

/-- Twice the number of reflection classes is the unrestricted `n`-wedge count
plus the count fixed by reflection. -/
lemma BurnsideForReflection (n : ℕ+) :
    S n + S_FIX_REFL n = 2 * S_UT_REFL n := by
  simpa [S_FIX_REFL, S_UT_REFL] using BurnsideForRubiksSnake n (reflectionTransform _)

/-- Shapes fixed by reversal AND reflection. -/
def S_FIX_REV_REFL (n : ℕ+) : ℕ :=
  fixedShapeCount n (reversalReflectionTransform _)

end

/-- Explicilty computed values of sequences defined in this file. -/

lemma S_FIX_REV_value_at_1 : S_FIX_REV 1 = 1 := by compute_snakes_pred
lemma S_FIX_REV_value_at_2 : S_FIX_REV 2 = 4 := by compute_snakes_pred
lemma S_FIX_REV_value_at_3 : S_FIX_REV 3 = 4 := by compute_snakes_pred
lemma S_FIX_REV_value_at_4 : S_FIX_REV 4 = 16 := by compute_snakes_pred
lemma S_FIX_REV_value_at_5 : S_FIX_REV 5 = 13 := by compute_snakes_pred
lemma S_FIX_REV_value_at_6 : S_FIX_REV 6 = 60 := by compute_snakes_pred

lemma S_UT_REV_value_at_1 : S_UT_REV 1 = 1 := by
  linarith only [BurnsideForReversal 1, S_FIX_REV_value_at_1, S1_value]
lemma S_UT_REV_value_at_2 : S_UT_REV 2 = 4 := by
  linarith only [BurnsideForReversal 2, S_FIX_REV_value_at_2, S2_value]
lemma S_UT_REV_value_at_3 : S_UT_REV 3 = 10 := by
  linarith only [BurnsideForReversal 3, S_FIX_REV_value_at_3, S3_value]
lemma S_UT_REV_value_at_4 : S_UT_REV 4 = 40 := by
  linarith only [BurnsideForReversal 4, S_FIX_REV_value_at_4, S4_value]
lemma S_UT_REV_value_at_5 : S_UT_REV 5 = 127 := by
  linarith only [BurnsideForReversal 5, S_FIX_REV_value_at_5, S5_value]
lemma S_UT_REV_value_at_6 : S_UT_REV 6 = 490 := by
  linarith only [BurnsideForReversal 6, S_FIX_REV_value_at_6, S6_value]

lemma S_FIX_REFL_value_at_1 : S_FIX_REFL 1 = 1 := by compute_snakes_pred
lemma S_FIX_REFL_value_at_2 : S_FIX_REFL 2 = 2 := by compute_snakes_pred
lemma S_FIX_REFL_value_at_3 : S_FIX_REFL 3 = 4 := by compute_snakes_pred
lemma S_FIX_REFL_value_at_4 : S_FIX_REFL 4 = 8 := by compute_snakes_pred
lemma S_FIX_REFL_value_at_5 : S_FIX_REFL 5 = 13 := by compute_snakes_pred
lemma S_FIX_REFL_value_at_6 : S_FIX_REFL 6 = 24 := by compute_snakes_pred

lemma S_UT_REFL_value_at_1 : S_UT_REFL 1 = 1 := by
  linarith only [BurnsideForReflection 1, S_FIX_REFL_value_at_1, S1_value]
lemma S_UT_REFL_value_at_2 : S_UT_REFL 2 = 3 := by
  linarith only [BurnsideForReflection 2, S_FIX_REFL_value_at_2, S2_value]
lemma S_UT_REFL_value_at_3 : S_UT_REFL 3 = 10 := by
  linarith only [BurnsideForReflection 3, S_FIX_REFL_value_at_3, S3_value]
lemma S_UT_REFL_value_at_4 : S_UT_REFL 4 = 36 := by
  linarith only [BurnsideForReflection 4, S_FIX_REFL_value_at_4, S4_value]
lemma S_UT_REFL_value_at_5 : S_UT_REFL 5 = 127 := by
  linarith only [BurnsideForReflection 5, S_FIX_REFL_value_at_5, S5_value]
lemma S_UT_REFL_value_at_6 : S_UT_REFL 6 = 472 := by
  linarith only [BurnsideForReflection 6, S_FIX_REFL_value_at_6, S6_value]

lemma S_FIX_REV_REFL_value_at_1 : S_FIX_REV_REFL 1 = 1 := by compute_snakes_pred
lemma S_FIX_REV_REFL_value_at_2 : S_FIX_REV_REFL 2 = 2 := by compute_snakes_pred
lemma S_FIX_REV_REFL_value_at_3 : S_FIX_REV_REFL 3 = 4 := by compute_snakes_pred
lemma S_FIX_REV_REFL_value_at_4 : S_FIX_REV_REFL 4 = 8 := by compute_snakes_pred
lemma S_FIX_REV_REFL_value_at_5 : S_FIX_REV_REFL 5 = 13 := by compute_snakes_pred
lemma S_FIX_REV_REFL_value_at_6 : S_FIX_REV_REFL 6 = 28 := by compute_snakes_pred

end RubiksSnake

import RubiksSnake.Definitions

namespace RubiksSnake

/-- Involutive tranform that maps valid formulas to valid formulas. -/
structure InvolutiveFormulaTransform (n : ℕ) where
  toFun : Formula n → Formula n
  involutive : Function.Involutive toFun
  valid_iff : ∀ w, Valid (toFun w) ↔ Valid w

/-- Use a bundled involution directly as a function on `n`-rotation formulas. -/
instance {n : ℕ} : CoeFun (InvolutiveFormulaTransform n)
    (fun _ => Formula n → Formula n) :=
  ⟨InvolutiveFormulaTransform.toFun⟩

end RubiksSnake

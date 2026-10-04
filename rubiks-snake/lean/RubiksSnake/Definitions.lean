import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Fin.Tuple.Reflection
import Mathlib.Data.List.Infix
import Mathlib.Data.PNat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.SetTheory.Cardinal.NatCard
import Lean.Elab.Tactic.Omega

/-! Definitions for Rubik's Snake and related sequences. -/

open scoped BigOperators

namespace RubiksSnake

noncomputable section

abbrev Vec3 := Fin 3 → ℤ
abbrev Rotation := Fin 4
abbrev Formula (n : ℕ) := Fin n → Rotation

def zeroVec : Vec3 := fun _ => 0
def ex : Vec3 := ![1, 0, 0]
def ey : Vec3 := ![0, 1, 0]
def ez : Vec3 := ![0, 0, 1]

def negVec (v : Vec3) : Vec3 := fun i => -v i

def addVec (u v : Vec3) : Vec3 := fun i => u i + v i

def cross (u v : Vec3) : Vec3 :=
  ![u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0]

def rotateQuarter (axis : Vec3) (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => v
  | 1 => cross axis v
  | 2 => negVec v
  | _ => negVec (cross axis v)

def directionTail : Vec3 → Vec3 → List Rotation → List Vec3
  | _, _, [] => []
  | previous, axis, r :: rs =>
      let next := rotateQuarter axis r previous
      next :: directionTail axis next rs

def directionsFrom (previous axis : Vec3) (rs : List Rotation) : List Vec3 :=
  previous :: axis :: directionTail previous axis rs

def directions (rs : List Rotation) : List Vec3 :=
  directionsFrom ey ex rs

def centersFromDirections (ds : List Vec3) : List Vec3 :=
  (ds.drop 1).dropLast.scanl addVec zeroVec

structure Wedge where
  center : Vec3
  entrance : Vec3
  exit : Vec3
deriving DecidableEq

def wedgesFromDirections (ds : List Vec3) : List Wedge :=
  (centersFromDirections ds).zip (ds.zip ds.tail) |>.map fun
    | (p, incoming, outgoing) =>
        ⟨p, negVec incoming, outgoing⟩

def wedges (rs : List Rotation) : List Wedge :=
  wedgesFromDirections (directions rs)

def sameUnorderedPair (a b c d : Vec3) : Prop :=
  (a = c ∧ b = d) ∨ (a = d ∧ b = c)

def interiorDisjoint (a b : Wedge) : Prop :=
  a.center ≠ b.center ∨
    sameUnorderedPair a.entrance a.exit (negVec b.entrance) (negVec b.exit)

def collisionFree (rs : List Rotation) : Prop :=
  (wedges rs).Pairwise interiorDisjoint

instance (rs : List Rotation) : Decidable (collisionFree rs) := by
  unfold collisionFree interiorDisjoint sameUnorderedPair
  infer_instance

/-- A word is valid when the interiors of its wedges are pairwise disjoint. -/
def ValidList (rs : List Rotation) : Prop :=
  collisionFree rs

instance (rs : List Rotation) : Decidable (ValidList rs) := by
  unfold ValidList
  infer_instance

def Valid {n : ℕ} (w : Formula n) : Prop :=
  ValidList (List.ofFn w)

instance {n : ℕ} (w : Formula n) : Decidable (Valid w) := by
  unfold Valid ValidList
  infer_instance

def countFormulas (n : ℕ) (p : Formula n → Prop) : ℕ :=
  Nat.card {w : Formula n // p w}

/-- `countValidFormulas k` counts valid rotation formulas of length `k`. -/
def countValidFormulas (k : ℕ) : ℕ :=
  countFormulas k Valid

/-- `S n` is the number of valid formulas for an `n`-wedge snake. -/
def S (n : ℕ+) : ℕ :=
  countValidFormulas (n - 1)

end

end RubiksSnake

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

def formulaPrefix {p q : ℕ} (w : Formula (p + q)) : Formula p :=
  fun i => w (Fin.castAdd q i)

def suffix {p q : ℕ} (w : Formula (p + q)) : Formula q :=
  fun i => w (Fin.natAdd p i)

def positiveAxis (v : Vec3) : Prop :=
  ∃ i : Fin 3, v = fun j => if j = i then 1 else 0

def coordinatewiseNonnegative (v : Vec3) : Prop :=
  ∀ i, 0 ≤ v i

def coordinatewiseLe (u v : Vec3) : Prop :=
  ∀ i, u i ≤ v i

def GoodStep (rs : List Rotation) : Prop :=
  let ds := directions rs
  let cs := centersFromDirections ds
  let lastCenter := cs.getLastD zeroVec
  collisionFree rs ∧ positiveAxis (ds.getLastD zeroVec) ∧
    coordinatewiseNonnegative lastCenter ∧ lastCenter ≠ zeroVec ∧
    ∀ p ∈ cs.drop 1,
      coordinatewiseLe p lastCenter ∧ ∃ i, 0 < p i

/-- The concatenable-step count `a_l` from the original asymptotic notebook. -/
def a (l : ℕ+) : ℕ :=
  countFormulas l fun w => GoodStep (List.ofFn w)

/-- The empirical consecutive-ratio sequence `r_n = S_(n+1) / S_n`. -/
noncomputable def r (n : ℕ+) : ℝ :=
  (S (n + 1) : ℝ) / (S n : ℝ)

def inSlab (width : ℕ) (p : Vec3) : Prop :=
  0 ≤ p 0 ∧ p 0 ≤ width

def SlabBlock (d j : ℕ+) (w : Formula j) : Prop :=
  let ds := directionsFrom ex ey (List.ofFn w)
  collisionFree (List.ofFn w) ∧
    (centersFromDirections ds).getLastD zeroVec 0 = (d : ℕ) - 1 ∧
    ds.getLastD zeroVec = ex ∧
    ∀ p ∈ centersFromDirections ds, inSlab ((d : ℕ) - 1) p

/-- `B d j` counts slab blocks of progress `d` and total encoded length `j`. -/
def B (d j : ℕ+) : ℕ :=
  countFormulas j (SlabBlock d j)

def IsSeparator (w : List Rotation) (h : ℕ) : Prop :=
  let cs := centersFromDirections (directionsFrom ex ey w)
  h + 1 < cs.length ∧
    (∀ i, i ≤ h → ∀ hi : i < cs.length, (cs.get ⟨i, hi⟩) 0 ≤ h) ∧
    (∀ i, h < i → ∀ hi : i < cs.length, h + 1 ≤ (cs.get ⟨i, hi⟩) 0)

def IrreducibleSlabBlock (d j : ℕ+) (w : Formula j) : Prop :=
  SlabBlock d j w ∧ ∀ h < (d : ℕ) - 1, ¬IsSeparator (List.ofFn w) h

/-- `I d j` counts irreducible slab blocks. -/
def I (d j : ℕ+) : ℕ :=
  countFormulas j (IrreducibleSlabBlock d j)

/-- `i_j` is the irreducible count summed over all possible progresses. -/
def irreducibleLengthCount (j : ℕ+) : ℕ :=
  ∑ d ∈ Finset.range j, I ⟨d + 1, by omega⟩ j

/-- Renewal counts for uniquely decomposable concatenations of irreducible blocks. -/
def c : ℕ → ℕ
  | 0 => 1
  | k + 1 =>
      ∑ j ∈ Finset.range (k + 1),
        irreducibleLengthCount ⟨j + 1, by omega⟩ * c (k - j)
termination_by k => k
decreasing_by omega

def validWindow (m : ℕ+) := {w : Formula m // Valid w}

/-- The finite-window adjacency matrix from the paper. -/
def A (m : ℕ+) (v u : validWindow m) : ℕ :=
  Nat.card {r : Rotation //
    ∃ w : Formula ((m : ℕ) + 1), Valid w ∧
      (fun i : Fin (m : ℕ) => w ⟨i.1, by omega⟩) = u.1 ∧
      (fun i : Fin (m : ℕ) => w ⟨i.1 + 1, by omega⟩) = v.1}

/-- Words accepted by every length-`m` sliding window. -/
def windowFormula {m : ℕ+} {n : ℕ} (w : Formula n) (start : ℕ)
    (h : start + (m : ℕ) ≤ n) : Formula m :=
  fun i => w ⟨start + i.1, by omega⟩

def windowPathCount (m : ℕ+) (n : ℕ) : ℕ :=
  countFormulas n fun w =>
    ∀ start, ∀ h : start + (m : ℕ) ≤ n, Valid (windowFormula w start h)



end

end RubiksSnake

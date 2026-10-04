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
abbrev Word (n : ℕ) := Fin n → Rotation

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

def Valid {n : ℕ} (w : Word n) : Prop :=
  ValidList (List.ofFn w)

instance {n : ℕ} (w : Word n) : Decidable (Valid w) := by
  unfold Valid ValidList
  infer_instance

def countWords (n : ℕ) (p : Word n → Prop) : ℕ :=
  Nat.card {w : Word n // p w}

/-- `wordCount k` counts valid rotation words of length `k`. -/
def wordCount (k : ℕ) : ℕ :=
  max 1 (countWords k Valid)

/-- `S n` is the number of valid formulas for an `n`-wedge snake. -/
def S (n : ℕ+) : ℕ :=
  wordCount ((n : ℕ) - 1)

def wordPrefix {p q : ℕ} (w : Word (p + q)) : Word p :=
  fun i => w (Fin.castAdd q i)

def suffix {p q : ℕ} (w : Word (p + q)) : Word q :=
  fun i => w (Fin.natAdd p i)

def reverseWord {n : ℕ} (w : Word n) : Word n :=
  fun i => w ⟨n - 1 - i.1, by omega⟩

def reflectRotation (r : Rotation) : Rotation :=
  ⟨(4 - r.1) % 4, Nat.mod_lt _ (by omega)⟩

def reflectWord {n : ℕ} (w : Word n) : Word n :=
  fun i => reflectRotation (w i)

def fixedShapeCount (t : ∀ {n}, Word n → Word n) (n : ℕ+) : ℕ :=
  countWords ((n : ℕ) - 1) fun w => Valid w ∧ t w = w

/-- Shapes fixed by head-tail reversal. -/
def F (n : ℕ+) : ℕ :=
  fixedShapeCount (@reverseWord) n

/-- Shapes up to reversal, as given by Burnside's lemma. -/
def D (n : ℕ+) : ℕ :=
  (S n + F n) / 2

def reflectionFixed (n : ℕ+) : ℕ :=
  fixedShapeCount (@reflectWord) n

def reversalReflectionFixed (n : ℕ+) : ℕ :=
  fixedShapeCount (fun w => reverseWord (reflectWord w)) n

def shapesUpToReflection (n : ℕ+) : ℕ :=
  (S n + reflectionFixed n) / 2

def shapesUpToReversalAndReflection (n : ℕ+) : ℕ :=
  (S n + F n + reflectionFixed n + reversalReflectionFixed n) / 4

def rotateWord {n : ℕ} (k : ℕ) (w : Word n) : Word n :=
  fun i => w ⟨(i.1 + k) % n, Nat.mod_lt _ (Nat.zero_lt_of_lt i.2)⟩

def cyclicValid {n : ℕ} (w : Word n) : Prop :=
  let ds := directions (List.ofFn w)
  collisionFree (List.ofFn w) ∧
    (centersFromDirections ds).getLastD zeroVec = zeroVec ∧
    ds.getLastD zeroVec = ex

def cyclicFixedCount (n : ℕ+) (k : ℕ) : ℕ :=
  countWords n fun w => cyclicValid w ∧ rotateWord k w = w

def reflectionCyclicFixedCount (n : ℕ+) (k : ℕ) : ℕ :=
  countWords n fun w =>
    cyclicValid w ∧ rotateWord k (reverseWord w) = w

/-- Formulas describing loops, with the closing joint included in the encoding. -/
def L1 (n : ℕ+) : ℕ :=
  countWords n cyclicValid

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
  countWords n fun w => cyclicValid (Fin.repeat k w)

/-- The reflection term in the dihedral Burnside sum. -/
def XR (n : ℕ+) (k : ℕ) : ℕ :=
  reflectionCyclicFixedCount n k

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
  countWords l fun w => GoodStep (List.ofFn w)

/-- The empirical consecutive-ratio sequence `r_n = S_(n+1) / S_n`. -/
noncomputable def r (n : ℕ+) : ℝ :=
  (S (n + 1) : ℝ) / (S n : ℝ)

def inSlab (width : ℕ) (p : Vec3) : Prop :=
  0 ≤ p 0 ∧ p 0 ≤ width

def SlabBlock (d j : ℕ+) (w : Word j) : Prop :=
  let ds := directionsFrom ex ey (List.ofFn w)
  collisionFree (List.ofFn w) ∧
    (centersFromDirections ds).getLastD zeroVec 0 = (d : ℕ) - 1 ∧
    ds.getLastD zeroVec = ex ∧
    ∀ p ∈ centersFromDirections ds, inSlab ((d : ℕ) - 1) p

/-- `B d j` counts slab blocks of progress `d` and total encoded length `j`. -/
def B (d j : ℕ+) : ℕ :=
  countWords j (SlabBlock d j)

def IsSeparator (w : List Rotation) (h : ℕ) : Prop :=
  let cs := centersFromDirections (directionsFrom ex ey w)
  h + 1 < cs.length ∧
    (∀ i, i ≤ h → ∀ hi : i < cs.length, (cs.get ⟨i, hi⟩) 0 ≤ h) ∧
    (∀ i, h < i → ∀ hi : i < cs.length, h + 1 ≤ (cs.get ⟨i, hi⟩) 0)

def IrreducibleSlabBlock (d j : ℕ+) (w : Word j) : Prop :=
  SlabBlock d j w ∧ ∀ h < (d : ℕ) - 1, ¬IsSeparator (List.ofFn w) h

/-- `I d j` counts irreducible slab blocks. -/
def I (d j : ℕ+) : ℕ :=
  countWords j (IrreducibleSlabBlock d j)

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

def validWindow (m : ℕ+) := {w : Word m // Valid w}

/-- The finite-window adjacency matrix from the paper. -/
def A (m : ℕ+) (v u : validWindow m) : ℕ :=
  Nat.card {r : Rotation //
    ∃ w : Word ((m : ℕ) + 1), Valid w ∧
      (fun i : Fin (m : ℕ) => w ⟨i.1, by omega⟩) = u.1 ∧
      (fun i : Fin (m : ℕ) => w ⟨i.1 + 1, by omega⟩) = v.1}

/-- Words accepted by every length-`m` sliding window. -/
def windowWord {m : ℕ+} {n : ℕ} (w : Word n) (start : ℕ)
    (h : start + (m : ℕ) ≤ n) : Word m :=
  fun i => w ⟨start + i.1, by omega⟩

def windowPathCount (m : ℕ+) (n : ℕ) : ℕ :=
  countWords n fun w =>
    ∀ start, ∀ h : start + (m : ℕ) ≤ n, Valid (windowWord w start h)



end

end RubiksSnake

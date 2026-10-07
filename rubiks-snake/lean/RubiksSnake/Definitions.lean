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

/-- An integer lattice vector with coordinates indexed by `0`, `1`, and `2`. -/
abbrev Vec3 := Fin 3 → ℤ
/-- A joint setting, encoded by the number of quarter turns modulo four. -/
abbrev Rotation := Fin 4
/-- A word of `n` joint rotations, describing `n + 1` wedges before any validity check. -/
abbrev Formula (n : ℕ) := Fin n → Rotation

/-- The origin of the integer lattice, also used as the zero displacement. -/
def zeroVec : Vec3 := fun _ => 0
/-- The positive unit direction along the first coordinate axis. -/
def ex : Vec3 := ![1, 0, 0]
/-- The positive unit direction along the second coordinate axis. -/
def ey : Vec3 := ![0, 1, 0]
/-- The positive unit direction along the third coordinate axis. -/
def ez : Vec3 := ![0, 0, 1]

/-- Reverse a lattice displacement or face direction by negating every coordinate. -/
def negVec (v : Vec3) : Vec3 := fun i => -v i

/-- Coordinatewise addition of lattice positions and displacements. -/
def addVec (u v : Vec3) : Vec3 := fun i => u i + v i

/-- The right-handed cross product of two integer lattice vectors. -/
def cross (u v : Vec3) : Vec3 :=
  ![u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0]

/-- The four-way turn rule: `v`, `axis × v`, `-v`, or `-(axis × v)`.
It represents quarter turns when `axis` is a unit coordinate direction perpendicular
to `v`; it is not a general rotation formula for arbitrary axes and vectors. -/
def rotateQuarter (axis : Vec3) (r : Rotation) (v : Vec3) : Vec3 :=
  match r.1 with
  | 0 => v
  | 1 => cross axis v
  | 2 => negVec v
  | _ => negVec (cross axis v)

/-- Generate one new travel direction per rotation, updating the preceding pair
of directions after each joint. The two initial directions are not included. -/
def directionTail : Vec3 → Vec3 → List Rotation → List Vec3
  | _, _, [] => []
  | previous, axis, r :: rs =>
      let next := rotateQuarter axis r previous
      next :: directionTail axis next rs

/-- The complete travel-direction list from a chosen initial pair, containing
`rs.length + 2` directions for `rs.length + 1` wedges. -/
def directionsFrom (previous axis : Vec3) (rs : List Rotation) : List Vec3 :=
  previous :: axis :: directionTail previous axis rs

/-- Travel directions in the canonical initial frame, with `ey` preceding `ex`. -/
def directions (rs : List Rotation) : List Vec3 :=
  directionsFrom ey ex rs

/-- Lattice-cell centers obtained by accumulating the interior travel directions
from the origin. The first and last directions specify end faces, not center steps. -/
def centersFromDirections (ds : List Vec3) : List Vec3 :=
  (ds.drop 1).dropLast.scanl addVec zeroVec

/-- A wedge encoded by its lattice-cell center and outward entrance and exit faces.
Geometric wedges use perpendicular cardinal face directions; this record itself
does not impose that restriction or identify the cell center with a centroid. -/
structure Wedge where
  center : Vec3
  entrance : Vec3
  exit : Vec3
deriving DecidableEq

/-- Pair successive travel directions with their cell centers to form wedges.
The entrance face points opposite the incoming travel direction; the exit agrees
with the outgoing direction. -/
def wedgesFromDirections (ds : List Vec3) : List Wedge :=
  (centersFromDirections ds).zip (ds.zip ds.tail) |>.map fun
    | (p, incoming, outgoing) =>
        ⟨p, negVec incoming, outgoing⟩

/-- The `rs.length + 1` wedges described by a rotation word in the canonical frame. -/
def wedges (rs : List Rotation) : List Wedge :=
  wedgesFromDirections (directions rs)

/-- Equality of two pairs of face directions, allowing their order to be exchanged. -/
def sameUnorderedPair (a b c d : Vec3) : Prop :=
  (a = c ∧ b = d) ∨ (a = d ∧ b = c)

/-- The collision test for cardinal wedges: different cell centers are disjoint;
at the same center, wedges coexist only when their unordered face pairs are
complementary, obtained from one another by negating both directions. -/
def interiorDisjoint (a b : Wedge) : Prop :=
  a.center ≠ b.center ∨
    sameUnorderedPair a.entrance a.exit (negVec b.entrance) (negVec b.exit)

/-- Wedges at distinct list indices have disjoint interiors,
including the complementary-face exception for wedges sharing a cell center. -/
def collisionFree (rs : List Rotation) : Prop :=
  (wedges rs).Pairwise interiorDisjoint

/-- Collision freedom is decidable by comparing the finitely many wedge pairs. -/
instance (rs : List Rotation) : Decidable (collisionFree rs) := by
  unfold collisionFree interiorDisjoint sameUnorderedPair
  infer_instance

/-- A word is valid when the interiors of its wedges are pairwise disjoint. -/
def ValidList (rs : List Rotation) : Prop :=
  collisionFree rs

/-- List validity is decidable using the finite geometric collision test. -/
instance (rs : List Rotation) : Decidable (ValidList rs) := by
  unfold ValidList
  infer_instance

/-- An `n`-rotation formula is valid when its `n + 1` wedges have pairwise
disjoint interiors in the canonical embedding. -/
def Valid {n : ℕ} (w : Formula n) : Prop :=
  ValidList (List.ofFn w)

/-- Formula validity is decidable after reading its rotations in index order. -/
instance {n : ℕ} (w : Formula n) : Decidable (Valid w) := by
  unfold Valid ValidList
  infer_instance

/-- Count `n`-rotation formulas satisfying `p`; validity is required only if
included in `p`, and the corresponding open snakes have `n + 1` wedges. -/
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

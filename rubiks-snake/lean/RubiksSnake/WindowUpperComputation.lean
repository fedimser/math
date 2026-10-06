import RubiksSnake.SmallCounts

/-!
# Executable geometry for small upper-bound certificates

Coordinates are stored as triples of integers rather than functions. The
conversion to the original wedge geometry is proved exactly, so the faster
checker does not introduce a separate notion of snake validity.
-/

namespace RubiksSnake
namespace WindowComputation

abbrev Coord := ℤ × ℤ × ℤ

def vec (p : Coord) : Vec3 := ![p.1, p.2.1, p.2.2]

def neg (p : Coord) : Coord := (-p.1, -p.2.1, -p.2.2)

def add (p q : Coord) : Coord := (p.1 + q.1, p.2.1 + q.2.1, p.2.2 + q.2.2)

def crossCoord (p q : Coord) : Coord :=
  (p.2.1 * q.2.2 - p.2.2 * q.2.1,
   p.2.2 * q.1 - p.1 * q.2.2,
   p.1 * q.2.1 - p.2.1 * q.1)

def turn (axis : Coord) (r : Rotation) (previous : Coord) : Coord :=
  match r.val with
  | 0 => previous
  | 1 => crossCoord axis previous
  | 2 => neg previous
  | _ => neg (crossCoord axis previous)

lemma vec_injective : Function.Injective vec := by
  rintro ⟨x, y, z⟩ ⟨x', y', z'⟩ h
  have hx := congrFun h 0
  have hy := congrFun h 1
  have hz := congrFun h 2
  simpa [vec, Prod.mk.injEq] using And.intro hx (And.intro hy hz)

@[simp] lemma vec_neg (p : Coord) : vec (neg p) = negVec (vec p) := by
  funext i
  fin_cases i <;> rfl

@[simp] lemma vec_add (p q : Coord) : vec (add p q) = addVec (vec p) (vec q) := by
  funext i
  fin_cases i <;> rfl

@[simp] lemma vec_cross (p q : Coord) :
    vec (crossCoord p q) = cross (vec p) (vec q) := rfl

@[simp] lemma vec_turn (axis : Coord) (r : Rotation) (previous : Coord) :
    vec (turn axis r previous) = rotateQuarter (vec axis) r (vec previous) := by
  fin_cases r <;> simp [turn, rotateQuarter]

structure CompactWedge where
  center : Coord
  entrance : Coord
  exit : Coord
deriving DecidableEq

def CompactWedge.toWedge (w : CompactWedge) : Wedge :=
  ⟨vec w.center, vec w.entrance, vec w.exit⟩

def disjoint (a b : CompactWedge) : Prop :=
  a.center ≠ b.center ∨
    (a.entrance = neg b.entrance ∧ a.exit = neg b.exit) ∨
    (a.entrance = neg b.exit ∧ a.exit = neg b.entrance)

instance (a b : CompactWedge) : Decidable (disjoint a b) := by
  unfold disjoint
  infer_instance

lemma disjoint_iff (a b : CompactWedge) :
    disjoint a b ↔ interiorDisjoint a.toWedge b.toWedge := by
  simp only [disjoint, interiorDisjoint, sameUnorderedPair, CompactWedge.toWedge,
    ← vec_neg, ne_eq, vec_injective.eq_iff]

def path (center incoming outgoing : Coord) : List Rotation → List CompactWedge
  | [] => [⟨center, neg incoming, outgoing⟩]
  | r :: rs =>
      ⟨center, neg incoming, outgoing⟩ ::
        path (add center outgoing) outgoing (turn outgoing r incoming) rs

lemma path_map (center incoming outgoing : Coord) (rs : List Rotation) :
    (path center incoming outgoing rs).map CompactWedge.toWedge =
      wedgePath (vec center) (vec incoming) (vec outgoing)
        (directionTail (vec incoming) (vec outgoing) rs) := by
  induction rs generalizing center incoming outgoing with
  | nil => simp [path, wedgePath, directionTail, CompactWedge.toWedge]
  | cons r rs ih =>
      simp [path, wedgePath, directionTail, CompactWedge.toWedge, ih]

def compactWedges (rs : List Rotation) : List CompactWedge :=
  path (0, 0, 0) (0, 1, 0) (1, 0, 0) rs

lemma compactWedges_map (rs : List Rotation) :
    (compactWedges rs).map CompactWedge.toWedge = wedges rs := by
  rw [compactWedges, path_map]
  have hzero : vec (0, 0, 0) = zeroVec := by
    funext i
    fin_cases i <;> rfl
  rw [hzero]
  change wedgePath zeroVec ey ex (directionTail ey ex rs) = wedges rs
  symm
  unfold wedges directions directionsFrom wedgesFromDirections centersFromDirections
  simp only [List.drop_succ_cons, List.drop_zero, List.tail_cons, List.zip_cons_cons]
  exact wedgesFromDirections_eq_wedgePath _ _ _ _

def valid (rs : List Rotation) : Bool :=
  decide ((compactWedges rs).Pairwise disjoint)

lemma valid_iff (rs : List Rotation) : valid rs = true ↔ ValidList rs := by
  rw [valid, decide_eq_true_eq]
  change _ ↔ (wedges rs).Pairwise interiorDisjoint
  rw [← compactWedges_map, List.pairwise_map]
  simp only [← disjoint_iff]

def rotations : List Rotation := [0, 1, 2, 3]

def next (rs : List Rotation) (r : Rotation) : List Rotation :=
  rs.tail ++ [r]

def outgoing (w : List Rotation → ℕ) (rs : List Rotation) : ℕ :=
  (rotations.map fun r => if valid (rs ++ [r]) then w (next rs r) else 0).sum

def degree (rs : List Rotation) : ℕ := outgoing (fun _ => 1) rs

lemma valid_append_eq_canAppend (rs : List Rotation) (r : Rotation)
    (hvalid : ValidList rs) : valid (rs ++ [r]) = canAppend rs r := by
  apply Bool.eq_iff_iff.mpr
  rw [valid_iff]
  change collisionFree (rs ++ [r]) ↔ canAppend rs r
  rw [collisionFree_append_iff]
  exact and_iff_right hvalid

lemma outgoing_eq_canAppend (w : List Rotation → ℕ) (rs : List Rotation)
    (hvalid : ValidList rs) :
    outgoing w rs =
      (rotations.map fun r => if canAppend rs r then w (next rs r) else 0).sum := by
  unfold outgoing
  simp_rw [valid_append_eq_canAppend rs _ hvalid]

def children (rs : List Rotation) : List (List Rotation) :=
  rotations.filterMap fun r =>
    if valid (rs ++ [r]) then some (rs ++ [r]) else none

def validWords : ℕ → List (List Rotation)
  | 0 => [[]]
  | n + 1 => (validWords n).flatMap children

lemma validWords_eq (n : ℕ) : validWords n = validRotationLists n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [validWords, ih]
      change
        (validRotationLists n).flatMap children =
          (validRotationLists n).flatMap (fun rs => rotations.filterMap fun r =>
            if canAppend rs r then some (rs ++ [r]) else none)
      suffices ∀ xs : List (List Rotation), (∀ rs ∈ xs, ValidList rs) →
          xs.flatMap children =
            xs.flatMap (fun rs => rotations.filterMap fun r =>
              if canAppend rs r then some (rs ++ [r]) else none) from
        this _ (validRotationLists_valid n)
      intro xs
      induction xs with
      | nil => simp
      | cons rs xs ih =>
          intro hvalid
          rw [List.flatMap_cons, List.flatMap_cons]
          congr 1
          · unfold children
            congr 1
            funext r
            rw [valid_append_eq_canAppend rs r (hvalid rs (by simp))]
          · exact ih (fun s hs => hvalid s (by simp [hs]))

def encode (rs : List Rotation) : ℕ :=
  rs.foldl (fun i r => 4 * i + r.val) 0

def decode (width : ℕ) (i : Fin (4 ^ width)) : List Rotation :=
  List.ofFn fun j : Fin width =>
    (⟨(i.val / 4 ^ (width - 1 - j.val)) % 4, Nat.mod_lt _ (by decide)⟩ : Rotation)

def edges (width : ℕ) : Array (List ℕ) :=
  Array.ofFn fun i : Fin (4 ^ width) =>
    let rs := decode width i
    rotations.filterMap fun r =>
      if valid (rs ++ [r]) then some (encode (next rs r)) else none

def iterateWeights (graph : Array (List ℕ)) : ℕ → Array ℕ
  | 0 => Array.replicate graph.size 1
  | n + 1 =>
      let previous := iterateWeights graph n
      graph.map fun es => (es.map fun i => previous[i]?.getD 0).sum

end WindowComputation
end RubiksSnake

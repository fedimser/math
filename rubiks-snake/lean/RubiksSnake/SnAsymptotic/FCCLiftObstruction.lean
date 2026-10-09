import RubiksSnake.SnAsymptotic.CardinalDirections

/-!
# An obstruction to unrestricted FCC lifting

Pairing perpendicular cubic steps gives an FCC edge with two possible
midpoints. The simple eight-edge FCC path below has no perpendicular,
collision-free cubic lift, even allowing complementary wedges in one cube.
Only internal wedges are tested, so changing the two end faces cannot help.
This is an obstruction to a proposed comparison, not a bound on the growth
constant. All finite checks use kernel reduction, not `native_decide`, and
have at most 256 midpoint assignments.
-/

namespace RubiksSnake.FCCLiftObstruction

open CardinalDirections

/-- The nine distinct even-parity centers of the counterexample. -/
def backbone : Fin 9 → Vec3 :=
  ![![0, 0, 0], ![-1, 0, -1], ![-2, 0, 0], ![-1, -1, 0],
    ![-2, -1, -1], ![-2, -2, 0], ![-3, -2, 1], ![-3, -1, 0],
    ![-2, -1, 1]]

/-- The two possible cubic midpoints for each FCC edge, ordered by first axis. -/
def midpoint (i : Fin 8) (choice : Bool) : Vec3 :=
  if choice then
    (![![0, 0, -1], ![-1, 0, 0], ![-2, -1, 0], ![-1, -1, -1],
       ![-2, -1, 0], ![-2, -2, 1], ![-3, -2, 0], ![-3, -1, 1]] :
      Fin 8 → Vec3) i
  else
    (![![-1, 0, 0], ![-2, 0, -1], ![-1, 0, 0], ![-2, -1, 0],
       ![-2, -2, -1], ![-3, -2, 0], ![-3, -1, 1], ![-2, -1, 0]] :
      Fin 8 → Vec3) i

/-- The backbone never repeats an FCC vertex. -/
theorem backbone_injective : Function.Injective backbone := by
  decide

/-- Each backbone edge is the sum of two perpendicular cardinal directions. -/
theorem backbone_edges :
    ∀ i : Fin 8, ∃ a b : Direction, Perpendicular a b ∧
      addVec (backbone i.castSucc) (addVec (vector a) (vector b)) =
        backbone i.succ := by
  decide

/-- Any two-cardinal-step realization of an edge uses one of the listed midpoints. -/
theorem midpoints_exhaustive :
    ∀ (i : Fin 8) (a b : Direction),
      addVec (backbone i.castSucc) (addVec (vector a) (vector b)) = backbone i.succ →
        addVec (backbone i.castSucc) (vector a) = midpoint i false ∨
        addVec (backbone i.castSucc) (vector a) = midpoint i true := by
  decide

/-- Each listed midpoint really gives a perpendicular two-step realization. -/
theorem midpoint_realized :
    ∀ (i : Fin 8) (choice : Bool), ∃ a b : Direction, Perpendicular a b ∧
      addVec (backbone i.castSucc) (vector a) = midpoint i choice ∧
      addVec (midpoint i choice) (vector b) = backbone i.succ := by
  decide

/-- Alternating backbone vertices and chosen midpoints gives seventeen centers. -/
def center (choices : Fin 8 → Bool) (i : Fin 17) : Vec3 :=
  if h : i.val % 2 = 0 then
    backbone ⟨i.val / 2, by omega⟩
  else
    midpoint ⟨i.val / 2, by omega⟩ (choices ⟨i.val / 2, by omega⟩)

/-- The wedge at an internal center, with faces pointing to its two neighbors. -/
def internalWedge (choices : Fin 8 → Bool) (i : Fin 15) : Wedge :=
  let p := center choices ⟨i.val, by omega⟩
  let q := center choices ⟨i.val + 1, by omega⟩
  let r := center choices ⟨i.val + 2, by omega⟩
  ⟨q, addVec p (negVec q), addVec r (negVec q)⟩

/-- Necessary local turning and collision conditions, without either end wedge. -/
def InternallyValid (choices : Fin 8 → Bool) : Prop :=
  (∀ i : Fin 15,
    ∑ k : Fin 3, (internalWedge choices i).entrance k *
      (internalWedge choices i).exit k = 0) ∧
    (List.ofFn (internalWedge choices)).Pairwise interiorDisjoint

/-- All conditions for a fixed midpoint assignment are finite geometric tests. -/
instance (choices : Fin 8 → Bool) : Decidable (InternallyValid choices) := by
  unfold InternallyValid interiorDisjoint sameUnorderedPair
  infer_instance

set_option maxRecDepth 10000 in
set_option maxHeartbeats 2000000 in
/-- None of the 256 possible lifts satisfies even the internal snake conditions. -/
theorem no_valid_lift : ∀ choices : Fin 8 → Bool, ¬ InternallyValid choices := by
  decide

end RubiksSnake.FCCLiftObstruction

import RubiksSnake.Geometry
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-!
  Small counts explcilty calculated.

  This file exists as additional check that definitions are correct, so the
  counts according to Lean definitions match those computed in Python.
 -/

namespace RubiksSnake

private def rotations : List Rotation := [0, 1, 2, 3]

private instance (a b : Wedge) : Decidable (interiorDisjoint a b) := by
  unfold interiorDisjoint sameUnorderedPair
  infer_instance

/-- The wedge added by appending `r` to a rotation formula. -/
def appendedWedge (rs : List Rotation) (r : Rotation) : Wedge :=
  (wedges (rs ++ [r])).getLast (by simp [wedges, wedgesFromDirections,
    centersFromDirections, directions, directionsFrom])

/-- Check precisely the new pairwise-disjointness obligations after appending a rotation. -/
def canAppend (rs : List Rotation) (r : Rotation) : Bool :=
  (wedges rs).all fun old => decide (interiorDisjoint old (appendedWedge rs r))

/-- Valid formulas of a fixed length, generated as a prefix tree. -/
def validRotationLists : ℕ → List (List Rotation)
  | 0 => [[]]
  | n + 1 =>
      (validRotationLists n).flatMap fun rs =>
        rotations.filterMap fun r =>
          if canAppend rs r then some (rs ++ [r]) else none

/-- Executable count of valid rotation formulas of length `k`. -/
def fastCountValidFormulas (k : ℕ) : ℕ :=
  (validRotationLists k).length

/-- Executable version of `S`. -/
def fastS (n : ℕ+) : ℕ :=
  fastCountValidFormulas ((n : ℕ) - 1)

private lemma directionsFrom_append_dropLast (previous axis : Vec3)
    (rs : List Rotation) (r : Rotation) :
    (directionsFrom previous axis (rs ++ [r])).dropLast =
      directionsFrom previous axis rs := by
  induction rs generalizing previous axis with
  | nil => simp [directionsFrom, directionTail]
  | cons s rs ih =>
      simpa [directionsFrom, directionTail] using
        congrArg List.tail (ih axis (rotateQuarter axis s previous))

private def wedgePath (center incoming outgoing : Vec3) : List Vec3 → List Wedge
  | [] => [⟨center, negVec incoming, outgoing⟩]
  | next :: rest =>
      ⟨center, negVec incoming, outgoing⟩ ::
        wedgePath (addVec center outgoing) outgoing next rest

private lemma wedgePath_ne_nil (center incoming outgoing : Vec3) (rest : List Vec3) :
    wedgePath center incoming outgoing rest ≠ [] := by
  cases rest <;> simp [wedgePath]

private lemma wedgesFromDirections_eq_wedgePath
    (center incoming outgoing : Vec3) (rest : List Vec3) :
    (((outgoing :: rest).dropLast.scanl addVec center).zip
        ((incoming, outgoing) :: (outgoing :: rest).zip rest)).map
      (fun x => ⟨x.1, negVec x.2.1, x.2.2⟩) =
      wedgePath center incoming outgoing rest := by
  induction rest generalizing center incoming outgoing with
  | nil => simp [wedgePath]
  | cons next rest ih =>
      simp [wedgePath, ih]

private lemma wedgePath_append_dropLast (center incoming outgoing : Vec3)
    (rest : List Vec3) (d : Vec3) :
    (wedgePath center incoming outgoing (rest ++ [d])).dropLast =
      wedgePath center incoming outgoing rest := by
  induction rest generalizing center incoming outgoing with
  | nil => simp [wedgePath]
  | cons next rest ih =>
      change
        (Wedge.mk center (negVec incoming) outgoing ::
          wedgePath (addVec center outgoing) outgoing next
            (rest ++ [d])).dropLast =
        Wedge.mk center (negVec incoming) outgoing ::
          wedgePath (addVec center outgoing) outgoing next rest
      rw [List.dropLast_cons_of_ne_nil (wedgePath_ne_nil _ _ _ _), ih]

private lemma wedgesFromDirections_append_dropLast (a b : Vec3)
    (rest : List Vec3) (d : Vec3) :
    (wedgesFromDirections (a :: b :: rest ++ [d])).dropLast =
      wedgesFromDirections (a :: b :: rest) := by
  rw [show (a :: b :: rest) ++ [d] = a :: b :: (rest ++ [d]) by simp]
  simp only [wedgesFromDirections, centersFromDirections, List.drop_succ_cons,
    List.drop_zero, List.tail_cons, List.zip_cons_cons]
  rw [wedgesFromDirections_eq_wedgePath, wedgesFromDirections_eq_wedgePath,
    wedgePath_append_dropLast]

lemma wedges_append_singleton (rs : List Rotation) (r : Rotation) :
    wedges (rs ++ [r]) = wedges rs ++ [appendedWedge rs r] := by
  have hne : wedges (rs ++ [r]) ≠ [] := by
    simp [wedges, wedgesFromDirections, centersFromDirections, directions,
      directionsFrom]
  have hdirections :
      ∃ d, directions (rs ++ [r]) = directions rs ++ [d] := by
    have hdrop := directionsFrom_append_dropLast ey ex rs r
    have hne : directions (rs ++ [r]) ≠ [] := by simp [directions, directionsFrom]
    refine ⟨(directions (rs ++ [r])).getLast hne, ?_⟩
    calc
      directions (rs ++ [r]) =
          (directions (rs ++ [r])).dropLast ++
            [(directions (rs ++ [r])).getLast hne] :=
        (List.dropLast_append_getLast hne).symm
      _ = directions rs ++ [(directions (rs ++ [r])).getLast hne] := by
        exact congrArg
          (fun ds => ds ++ [(directions (rs ++ [r])).getLast hne]) hdrop
  obtain ⟨d, hd⟩ := hdirections
  have hdrop : (wedges (rs ++ [r])).dropLast = wedges rs := by
    unfold wedges
    rw [hd]
    unfold directions directionsFrom
    exact wedgesFromDirections_append_dropLast ey ex _ d
  calc
    wedges (rs ++ [r]) =
        (wedges (rs ++ [r])).dropLast ++
          [(wedges (rs ++ [r])).getLast hne] :=
      (List.dropLast_append_getLast hne).symm
    _ = wedges rs ++ [appendedWedge rs r] := by
      rw [hdrop]
      rfl

lemma collisionFree_append_iff (rs : List Rotation) (r : Rotation) :
    collisionFree (rs ++ [r]) ↔ collisionFree rs ∧ canAppend rs r := by
  rw [collisionFree, wedges_append_singleton, List.pairwise_append]
  simp [canAppend, collisionFree]

lemma validRotationLists_length (k : ℕ) (rs : List Rotation)
    (hrs : rs ∈ validRotationLists k) : rs.length = k := by
  induction k generalizing rs with
  | zero =>
      simpa [validRotationLists] using hrs
  | succ k ih =>
      simp only [validRotationLists, List.mem_flatMap, List.mem_filterMap] at hrs
      obtain ⟨pre, hpre, r, hr, hsome⟩ := hrs
      split at hsome
      · cases hsome
        simp [ih pre hpre]
      · simp at hsome

lemma validRotationLists_valid (k : ℕ) (rs : List Rotation)
    (hrs : rs ∈ validRotationLists k) : ValidList rs := by
  induction k generalizing rs with
  | zero =>
      have : rs = [] := by simpa [validRotationLists] using hrs
      subst rs
      native_decide
  | succ k ih =>
      simp only [validRotationLists, List.mem_flatMap, List.mem_filterMap] at hrs
      obtain ⟨pre, hpre, r, hr, hsome⟩ := hrs
      split at hsome
      · rename_i hcan
        cases hsome
        exact (collisionFree_append_iff pre r).mpr
          ⟨ih pre hpre, by simpa using hcan⟩
      · simp at hsome

lemma mem_validRotationLists (rs : List Rotation) :
    rs ∈ validRotationLists rs.length ↔ ValidList rs := by
  constructor
  · exact validRotationLists_valid _ rs
  · intro hvalid
    induction rs using List.reverseRecOn with
    | nil => simp [validRotationLists]
    | append_singleton pre r ih =>
        have hparts := (collisionFree_append_iff pre r).mp hvalid
        simp only [List.length_append, List.length_singleton, validRotationLists,
          List.mem_flatMap, List.mem_filterMap]
        refine ⟨pre, ih hparts.1, r, ?_, ?_⟩
        · fin_cases r <;> simp [rotations]
        · simp [hparts.2]

lemma validRotationLists_nodup (k : ℕ) : (validRotationLists k).Nodup := by
  classical
  induction k with
  | zero => simp [validRotationLists]
  | succ k ih =>
      simp only [validRotationLists]
      apply List.nodup_flatMap.mpr
      constructor
      · intro pre hpre
        apply List.Nodup.filterMap
        · intro a a' child hchild hchild'
          split at hchild <;> split at hchild' <;> simp_all
          have hsuffix := congrArg List.getLast? (hchild.trans hchild'.symm)
          simpa using hsuffix
        · simp [rotations]
      · apply ih.imp
        intro a b hab child hchilda hchildb
        simp only [List.mem_filterMap] at hchilda hchildb
        obtain ⟨ra, hra, hca⟩ := hchilda
        obtain ⟨rb, hrb, hcb⟩ := hchildb
        split at hca <;> simp_all
        have heq : a ++ [ra] = b ++ [rb] := hca.trans hcb.2.symm
        have hpre := congrArg List.dropLast heq
        simp at hpre
        exact hab hpre

def formulaOfList {k : ℕ} (rs : List Rotation) (h : rs.length = k) : Formula k :=
  fun i => rs.get (Fin.cast h.symm i)

@[simp] lemma ofFn_formulaOfList {k : ℕ} (rs : List Rotation) (h : rs.length = k) :
    List.ofFn (formulaOfList rs h) = rs := by
  cases h
  exact List.ofFn_get rs

def validFormulaEquiv (k : ℕ) :
    {w : Formula k // Valid w} ≃ {rs : List Rotation // rs ∈ validRotationLists k} where
  toFun w := ⟨List.ofFn w, by
    have hm := (mem_validRotationLists (List.ofFn w.val)).mpr w.property
    simpa using hm⟩
  invFun rs := ⟨formulaOfList rs.val (validRotationLists_length k rs.val rs.property), by
    unfold Valid
    rw [ofFn_formulaOfList]
    exact validRotationLists_valid k rs.val rs.property⟩
  left_inv w := by
    apply Subtype.ext
    funext i
    simp [formulaOfList]
  right_inv rs := by
    apply Subtype.ext
    exact ofFn_formulaOfList rs.val (validRotationLists_length k rs.val rs.property)

private def listMembershipEquiv [DecidableEq α] (xs : List α) :
    {x : α // x ∈ xs} ≃ ↥xs.toFinset where
  toFun x := ⟨x, List.mem_toFinset.mpr x.property⟩
  invFun x := ⟨x, List.mem_toFinset.mp x.property⟩
  left_inv _ := rfl
  right_inv _ := rfl

theorem fastCountValidFormulas_eq (k : ℕ) :
    fastCountValidFormulas k = countValidFormulas k := by
  unfold fastCountValidFormulas countValidFormulas countFormulas
  rw [Nat.card_congr (validFormulaEquiv k)]
  rw [Nat.card_congr (listMembershipEquiv (validRotationLists k))]
  rw [Nat.card_eq_fintype_card, Fintype.card_coe]
  simpa using (List.toFinset_card_of_nodup (validRotationLists_nodup k)).symm

theorem fastS_eq_S (n : ℕ+) : fastS n = S n := by
  exact fastCountValidFormulas_eq _

/-- Prove a concrete equality for `S n` using the prefix-tree evaluator. -/
macro "snake_decide" : tactic =>
  `(tactic| (rw [← fastS_eq_S]; native_decide))


/-- Show that for `n ≤ 4`, `S_n` is a power of four because all formulas are valid. -/
lemma Sn_is_power_of_4 (n : ℕ+) (hfour : n ≤ 4) :
    S n = 4 ^ ((n : ℕ) - 1) := by
  rcases n with ⟨n, hn⟩
  change n ≤ 4 at hfour
  have allValid : ∀ w : Formula (n - 1), Valid w := by
    obtain rfl | rfl | rfl | rfl : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
    all_goals native_decide
  let validEquiv : {w : Formula (n - 1) // Valid w} ≃ Formula (n - 1) :=
    { toFun := Subtype.val
      invFun := fun w => ⟨w, allValid w⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hcount : countFormulas (n - 1) Valid = 4 ^ (n - 1) := by
    rw [countFormulas, Nat.card_congr validEquiv]
    simp [Formula, Rotation]
  change countValidFormulas (n - 1) = 4 ^ (n - 1)
  exact hcount

/-- https://oeis.org/A375865 -/
lemma S1_value: S 1 = 1 := by simpa using Sn_is_power_of_4 1
lemma S2_value: S 2 = 4 := by simpa using Sn_is_power_of_4 2
lemma S3_value: S 3 = 16 := by simpa using Sn_is_power_of_4 3
lemma S4_value: S 4 = 64 := by simpa using Sn_is_power_of_4 4
lemma S5_value: S 5 = 241 := by snake_decide
lemma S6_value: S 6 = 920 := by snake_decide
lemma S7_value : S 7 = 3384 := by snake_decide

end RubiksSnake

import RubiksSnake.Geometry
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-!
This file provides:
 - a computable function fastS to evaluate S(n),
 - proof that fastS(n)=S(n),
 - tactic `compute_snakes` to prove statements about specific value of S(n).
 - explicitly evaluated S(n) up to n=7.
 - a computable predicate-filtered count fastCountValidShapesPred with a correctness proof.
 - tactic `compute_snakes_pred` to prove statements about number of valid snakes
     satisfying additional predicates.
-/

namespace RubiksSnake

/-- All four joint settings, listed once each for branching in the prefix-tree enumerator. -/
private def rotations : List Rotation := [0, 1, 2, 3]

/-- Decide whether two wedge records have distinct centers or complementary face pairs. -/
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

/-- Extending a rotation word adds only its final travel direction; deleting
that direction recovers the list generated from the original word. -/
private lemma directionsFrom_append_dropLast (previous axis : Vec3)
    (rs : List Rotation) (r : Rotation) :
    (directionsFrom previous axis (rs ++ [r])).dropLast =
      directionsFrom previous axis rs := by
  induction rs generalizing previous axis with
  | nil => simp [directionsFrom, directionTail]
  | cons s rs ih =>
      simpa [directionsFrom, directionTail] using
        congrArg List.tail (ih axis (rotateQuarter axis s previous))

/-- A recursive wedge path always contains its starting wedge, even with no
further travel directions. -/
private lemma wedgePath_ne_nil (center incoming outgoing : Vec3) (rest : List Vec3) :
    wedgePath center incoming outgoing rest ≠ [] := by
  cases rest <;> simp [wedgePath]

/-- Appending one travel direction adds only a terminal wedge to a recursive path. -/
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

/-- For a direction list with an initial pair, appending one direction and
then deleting the last wedge recovers all original wedges. -/
private lemma wedgesFromDirections_append_dropLast (a b : Vec3)
    (rest : List Vec3) (d : Vec3) :
    (wedgesFromDirections (a :: b :: rest ++ [d])).dropLast =
      wedgesFromDirections (a :: b :: rest) := by
  rw [show (a :: b :: rest) ++ [d] = a :: b :: (rest ++ [d]) by simp]
  simp only [wedgesFromDirections, centersFromDirections, List.drop_succ_cons,
    List.drop_zero, List.tail_cons, List.zip_cons_cons]
  rw [wedgesFromDirections_eq_wedgePath, wedgesFromDirections_eq_wedgePath,
    wedgePath_append_dropLast]

/-- Appending one rotation adds exactly the terminal wedge; all earlier cell
centers and face directions are unchanged. -/
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

/-- An extended word is collision-free exactly when its prefix is collision-free
and the new wedge passes every comparison performed by `canAppend`. -/
lemma collisionFree_append_iff (rs : List Rotation) (r : Rotation) :
    collisionFree (rs ++ [r]) ↔ collisionFree rs ∧ canAppend rs r := by
  rw [collisionFree, wedges_append_singleton, List.pairwise_append]
  simp [canAppend, collisionFree]

/-- Every word at enumeration depth `k` has exactly `k` rotations, hence `k + 1` wedges. -/
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

/-- The prefix-tree enumerator is sound: every generated word has pairwise-disjoint wedges. -/
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

/-- A rotation word occurs at its own depth in the enumerator exactly when it
is valid, establishing both completeness and soundness. -/
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

/-- No rotation word is generated twice at a fixed enumeration depth. -/
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

/-- Read a rotation list of proven length `k` as an indexed `k`-rotation formula. -/
def formulaOfList {k : ℕ} (rs : List Rotation) (h : rs.length = k) : Formula k :=
  fun i => rs.get (Fin.cast h.symm i)

/-- Converting a list to an indexed formula and reading it back preserves every entry. -/
@[simp] lemma ofFn_formulaOfList {k : ℕ} (rs : List Rotation) (h : rs.length = k) :
    List.ofFn (formulaOfList rs h) = rs := by
  cases h
  exact List.ofFn_get rs

/-- Valid indexed `k`-rotation formulas correspond bijectively to the rotation
lists generated at depth `k`. -/
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

/-- The subtype of entries belonging to a list is equivalent to the subtype of
its deduplicated finite set; no duplicate-freeness assumption is needed here. -/
private def listMembershipEquiv [DecidableEq α] (xs : List α) :
    {x : α // x ∈ xs} ≃ ↥xs.toFinset where
  toFun x := ⟨x, List.mem_toFinset.mpr x.property⟩
  invFun x := ⟨x, List.mem_toFinset.mp x.property⟩
  left_inv _ := rfl
  right_inv _ := rfl

/-- The executable prefix-tree count equals the abstract count of all valid
`k`-rotation formulas, using completeness and absence of duplicates. -/
theorem fastCountValidFormulas_eq (k : ℕ) :
    fastCountValidFormulas k = countValidFormulas k := by
  unfold fastCountValidFormulas countValidFormulas countFormulas
  rw [Nat.card_congr (validFormulaEquiv k)]
  rw [Nat.card_congr (listMembershipEquiv (validRotationLists k))]
  rw [Nat.card_eq_fintype_card, Fintype.card_coe]
  simpa using (List.toFinset_card_of_nodup (validRotationLists_nodup k)).symm

/-- The executable count with `n - 1` rotations computes exactly the `n`-wedge count `S n`. -/
theorem fastS_eq_S (n : ℕ+) : fastS n = S n := by
  exact fastCountValidFormulas_eq _

/-- Prove a concrete equality for `S n` using the prefix-tree evaluator. -/
macro "compute_snakes" : tactic =>
  `(tactic| (rw [← fastS_eq_S]; native_decide))

/-- Count valid `n`-wedge formulas satisfying a decidable predicate, testing only
completed words from the prefix-tree enumerator without allocating a filtered list. -/
def fastCountValidShapesPred (n : ℕ+) (p : Formula ((n : ℕ) - 1) → Prop)
    [DecidablePred p] : ℕ :=
  (validRotationLists ((n : ℕ) - 1)).attach.countP fun rs =>
    decide (p (formulaOfList rs.val (validRotationLists_length _ _ rs.property)))

/-- The predicate-filtered prefix-tree count agrees with the abstract count
of valid formulas satisfying `p`. -/
theorem countingWithPredicate (n : ℕ+) (p : Formula ((n : ℕ) - 1) → Prop)
    [DecidablePred p] :
    countFormulas ((n : ℕ) - 1) (fun f => Valid f ∧ p f) =
      fastCountValidShapesPred n p := by
  let k := (n : ℕ) - 1
  let q := fun rs : {rs : List Rotation // rs ∈ validRotationLists k} =>
    p ((validFormulaEquiv k).symm rs).val
  let selected := (validRotationLists k).attach.toFinset.filter q
  have e : {f : Formula k // Valid f ∧ p f} ≃ ↥selected :=
    ((Equiv.subtypeSubtypeEquivSubtypeInter Valid p).symm.trans
      (validFormulaEquiv k).subtypeEquivOfSubtype').trans
      (Equiv.subtypeEquivRight (by
        intro rs
        simp [selected, q]))
  unfold countFormulas
  rw [Nat.card_congr e, Nat.card_eq_fintype_card, Fintype.card_coe]
  exact (validRotationLists_nodup k).attach.card_eq_countP

/-- Prove a concrete predicate-restricted count, reducing transparent wrapper
definitions as needed and using a computable predicate. Already executable
goals are checked directly without rewriting. -/
macro "compute_snakes_pred" : tactic =>
  `(tactic| first
    | native_decide
    | (
      change countFormulas _ (fun f => Valid f ∧ _) = _
      rw [countingWithPredicate]
      native_decide))


/-- Counts, see https://oeis.org/A375865. -/
lemma S1_value : S 1 = 1 := by compute_snakes
lemma S2_value : S 2 = 4 := by compute_snakes
lemma S3_value : S 3 = 16 := by compute_snakes
lemma S4_value : S 4 = 64 := by compute_snakes
lemma S5_value : S 5 = 241 := by compute_snakes
lemma S6_value : S 6 = 920 := by compute_snakes
lemma S7_value : S 7 = 3384 := by compute_snakes

end RubiksSnake

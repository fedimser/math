import Mathlib.Data.ZMod.Basic

import RubiksSnake.Definitions
import RubiksSnake.ReversalTransform

namespace RubiksSnake

/-- Loop formula is converted to regular formula by removing last rotation. -/
def loopFrmToFrm {n : ℕ+} (w : Formula n) : Formula (n - 1) :=
  fun i => w (Fin.castLE (Nat.sub_le n 1) i)

/--
 Checks whether Formula of length n describes an n-wedge loop.
 Does not check that formula is valid (e.g. collision free).
-/
def isLoop {n : ℕ} (w : Formula n) : Prop :=
  let ds := directions (List.ofFn w)
  (centersFromDirections ds).getLastD zeroVec = zeroVec ∧
    ds.reverse.take 2 = [ex, ey]

/-- Checks whether Formula describes valid n-wedge loop. -/
def isValidLoop {n : ℕ+} (w : Formula n) : Prop :=
  Valid (loopFrmToFrm w) ∧ isLoop w

/-- Formula describing an n-wedge loop. -/
structure LoopFormula (n : ℕ+) where
  f : Formula n
  validLoop: isValidLoop f

def toShapeFormula {n : ℕ+}: LoopFormula n → Formula (n-1) :=
  fun lf => loopFrmToFrm lf.f

/-- Example: smallest loop. -/
def loop2222 : LoopFormula 4 where
  f := ![2, 2, 2, 2]
  validLoop := by
    unfold isValidLoop Valid ValidList loopFrmToFrm isLoop
    native_decide

#eval toShapeFormula loop2222

/-- Transform of a loop-formula that turns it into another loop formula. -/
structure LoopTransform (n : ℕ+) where
  t: Formula n → Formula n
  preservesValidLoop: ∀ w, isValidLoop w → isValidLoop (t w)

def shiftLoopFormula {n : ℕ} (k : ℤ) (w : Formula n) : Formula n :=
  fun i =>
    w ⟨Int.natMod ((i.1 : ℤ) + k) n,
      Int.natMod_lt (Nat.ne_of_gt (Nat.zero_lt_of_lt i.2))⟩

lemma shiftPreservesValidLoop (n: ℕ+) (k: ℤ) (w: Formula n):
    isValidLoop w → isValidLoop ((shiftLoopFormula k) w) := by
  sorry

def shiftTransform (n : ℕ+) (k : ℤ) : LoopTransform n where
  t := shiftLoopFormula k
  preservesValidLoop := shiftPreservesValidLoop n k



private def SignedAxis (v : Vec3) : Prop :=
  v = ex ∨ v = negVec ex ∨ v = ey ∨ v = negVec ey ∨
    v = ez ∨ v = negVec ez

private def AxisFrame (previous axis : Vec3) : Prop :=
  SignedAxis previous ∧ SignedAxis axis ∧
    previous ≠ axis ∧ previous ≠ negVec axis

private lemma initial_axisFrame : AxisFrame ey ex := by
  unfold AxisFrame SignedAxis
  native_decide

private lemma axisFrame_step (previous axis : Vec3) (r : Rotation)
    (h : AxisFrame previous axis) :
    AxisFrame axis (rotateQuarter axis r previous) := by
  rcases h.1 with h | h | h | h | h | h <;> subst previous <;>
    rcases h.2.1 with h | h | h | h | h | h <;> subst axis <;>
    fin_cases r <;>
    simp_all [AxisFrame, SignedAxis, rotateQuarter, cross, negVec, ex, ey, ez]
  all_goals native_decide

private lemma directionTail_signedAxis (previous axis : Vec3)
    (rs : List Rotation) (h : AxisFrame previous axis) :
    ∀ v ∈ directionTail previous axis rs, SignedAxis v := by
  induction rs generalizing previous axis with
  | nil => simp [directionTail]
  | cons r rs ih =>
      have hstep := axisFrame_step previous axis r h
      intro v hv
      simp only [directionTail, List.mem_cons] at hv
      rcases hv with rfl | hv
      · exact hstep.2.1
      · exact ih axis (rotateQuarter axis r previous) hstep v hv

private lemma directions_signedAxis (rs : List Rotation) :
    ∀ v ∈ directions rs, SignedAxis v := by
  intro v hv
  simp only [directions, directionsFrom, List.mem_cons] at hv
  rcases hv with rfl | rfl | hv
  · simp [SignedAxis]
  · simp [SignedAxis]
  · exact directionTail_signedAxis ey ex rs initial_axisFrame v hv

private def checkerColor (v : Vec3) : ZMod 2 :=
  v 0 + v 1 + v 2

private lemma signedAxis_checkerColor {v : Vec3} (h : SignedAxis v) :
    checkerColor v = 1 := by
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

private lemma checkerColor_addVec (u v : Vec3) :
    checkerColor (addVec u v) = checkerColor u + checkerColor v := by
  simp [checkerColor, addVec]
  ring

private lemma checkerColor_foldl (vs : List Vec3) (initial : Vec3)
    (h : ∀ v ∈ vs, SignedAxis v) :
    checkerColor (vs.foldl addVec initial) =
      checkerColor initial + (vs.length : ZMod 2) := by
  induction vs generalizing initial with
  | nil => simp
  | cons v vs ih =>
      have hvs : ∀ u ∈ vs, SignedAxis u := by
        intro u hu
        exact h u (by simp [hu])
      rw [List.foldl_cons, ih (addVec initial v) hvs]
      rw [checkerColor_addVec, signedAxis_checkerColor (h v (by simp))]
      simp only [List.length_cons, Nat.cast_add, Nat.cast_one]
      ring

private lemma scanl_getLastD (vs : List Vec3) (initial default : Vec3) :
    (vs.scanl addVec initial).getLastD default = vs.foldl addVec initial := by
  induction vs generalizing initial default with
  | nil => simp
  | cons v vs ih =>
      simp only [List.scanl_cons, List.foldl_cons]
      rw [List.getLastD_cons]
      exact ih (addVec initial v) initial

/-- Loop cannot have odd length. -/
lemma noOddLoops (n : ℕ+) :
    Odd (n : ℕ) → ∀ w: Formula n, ¬isLoop w := by
  intro hn w hloop
  let ds := directions (List.ofFn w)
  let steps := (ds.drop 1).dropLast
  have hcenter : steps.foldl addVec zeroVec = zeroVec := by
    have h := hloop.1
    unfold isLoop at hloop
    change (steps.scanl addVec zeroVec).getLastD zeroVec = zeroVec at h
    rw [scanl_getLastD] at h
    exact h
  have hsteps : ∀ v ∈ steps, SignedAxis v := by
    intro v hv
    exact directions_signedAxis (List.ofFn w) v (by
      have hv' : v ∈ ds.drop 1 := List.mem_of_mem_dropLast hv
      simp only [List.drop_one] at hv'
      exact List.mem_of_mem_tail hv')
  have hlength : steps.length = (n : ℕ) := by
    simp [steps, ds, directions, directionsFrom, directionTail_length]
  have hcast : ((n : ℕ) : ZMod 2) = 0 := by
    have hcolor := checkerColor_foldl steps zeroVec hsteps
    rw [hcenter] at hcolor
    simpa [checkerColor, zeroVec, hlength] using hcolor.symm
  have heven : Even (n : ℕ) :=
    even_iff_two_dvd.mpr ((ZMod.natCast_eq_zero_iff (n : ℕ) 2).mp hcast)
  exact (Nat.not_odd_iff_even.mpr heven) hn




noncomputable section

/--
  L1(n) - number of formulas of length n-1) that describe shapes that are loops.
-/
def L1 (n : ℕ+) : ℕ :=
  countFormulas n (fun f => isLoop f ∧ Valid (loopFrmToFrm f))

end

lemma noOddLoopsNumeric (n : ℕ+): Odd (n : ℕ) → L1 n = 0 := by
  intro hn
  unfold L1 countFormulas
  rw [Finite.card_eq_zero_iff]
  exact ⟨fun ⟨w, hw⟩ => noOddLoops n hn w hw.1⟩


lemma L1_1_value : L1 1 = 0 := noOddLoopsNumeric 1 (by simp)
lemma L1_3_value : L1 3 = 0 := noOddLoopsNumeric 3 ⟨1, by norm_num⟩
lemma L1_5_value : L1 5 = 0 := noOddLoopsNumeric 5 ⟨2, by norm_num⟩
lemma L1_7_value : L1 7 = 0 := noOddLoopsNumeric 7 ⟨3, by norm_num⟩



end RubiksSnake

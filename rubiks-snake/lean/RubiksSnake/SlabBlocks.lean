import RubiksSnake.SmallCounts
import RubiksSnake.ReversalTransform
import Mathlib.Tactic

/-!
# Concatenating plane blocks

Each block enters a new plane perpendicular to `ex`, stays in that plane,
and finishes with its outgoing face in direction `ex`. The four possible
incoming transverse directions are checked explicitly. Concatenations
occupy successive disjoint planes.
-/

namespace RubiksSnake

def planePrevious (i : Fin 4) : Vec3 :=
  ![ey, ez, negVec ey, negVec ez] i

def planeStart (i : Fin 4) (p : Vec3) : SlabState :=
  ⟨p, planePrevious i, ex⟩

lemma planeStart_translate (i : Fin 4) (p : Vec3) :
    planeStart i p = slabTranslate p (planeStart i zeroVec) := by
  simp [planeStart, slabTranslate]

def IsPlaneBlock (rs : List Rotation) : Prop :=
  ∀ i : Fin 4,
    let s := planeStart i zeroVec
    (slabNewWedges s rs).Pairwise interiorDisjoint ∧
    (∀ w ∈ slabNewWedges s rs, w.center 0 = 1) ∧
    (slabRun s rs).center 0 = 1 ∧
    (slabRun s rs).axis = ex ∧
    ∃ j : Fin 4, (slabRun s rs).previous = planePrevious j

instance (rs : List Rotation) : Decidable (IsPlaneBlock rs) := by
  unfold IsPlaneBlock interiorDisjoint sameUnorderedPair
  infer_instance

lemma planeBlock_at {rs : List Rotation} (h : IsPlaneBlock rs)
    (i : Fin 4) (p : Vec3) :
    (slabNewWedges (planeStart i p) rs).Pairwise interiorDisjoint ∧
    (∀ w ∈ slabNewWedges (planeStart i p) rs, w.center 0 = p 0 + 1) ∧
    ∃ j : Fin 4, ∃ q : Vec3,
      slabRun (planeStart i p) rs = planeStart j q ∧ q 0 = p 0 + 1 := by
  obtain ⟨hvalid, hplane, hlast, haxis, j, hprevious⟩ := h i
  rw [planeStart_translate i p, slabNewWedges_translate, slabRun_translate]
  refine ⟨?_, ?_, j, addVec p (slabRun (planeStart i zeroVec) rs).center, ?_, ?_⟩
  · rw [List.pairwise_map]
    exact hvalid.imp fun hab => (translateWedge_interiorDisjoint p _ _).mpr hab
  · intro w hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    simp [translateWedge, addVec, hplane v hv]
  · change SlabState.mk _ (slabRun (planeStart i zeroVec) rs).previous
        (slabRun (planeStart i zeroVec) rs).axis = SlabState.mk _ (planePrevious j) ex
    rw [hprevious, haxis]
  · simp [addVec, hlast]

def planeCandidates : SlabState → ℕ → List (List Rotation)
  | s, 0 => if s.axis = ex then [[]] else []
  | s, k + 1 =>
      if s.center 0 + s.axis 0 = 1 then
        ([0, 1, 2, 3] : List Rotation).flatMap fun r =>
          (planeCandidates (slabStep s r) k).map (r :: ·)
      else []

def planeCode : List (List Rotation) :=
  ((List.range 6).flatMap fun k =>
    planeCandidates (planeStart 0 zeroVec) (k + 2)).filter
    (fun rs => decide (IsPlaneBlock rs))

def planeBlocks (k : ℕ) : List (List Rotation) :=
  planeCode.filter (fun rs => rs.length == k)

lemma planeCode_valid {rs : List Rotation} (h : rs ∈ planeCode) :
    IsPlaneBlock rs := by
  exact of_decide_eq_true (List.mem_filter.mp h).2

lemma planeCode_nodup : planeCode.Nodup := by native_decide

lemma planeCode_lengths : ∀ rs ∈ planeCode, 2 ≤ rs.length ∧ rs.length ≤ 7 := by
  native_decide

lemma planeCode_prefix_free :
    ∀ a ∈ planeCode, ∀ b ∈ planeCode, a <+: b → a = b := by
  native_decide

lemma mem_planeBlocks {k : ℕ} {rs : List Rotation} :
    rs ∈ planeBlocks k ↔ rs ∈ planeCode ∧ rs.length = k := by
  simp [planeBlocks]

lemma planeCode_by_length :
    planeCode = planeBlocks 2 ++ planeBlocks 3 ++ planeBlocks 4 ++
      planeBlocks 5 ++ planeBlocks 6 ++ planeBlocks 7 := by
  native_decide

/-- Numbers of selected plane blocks, indexed by rotation-word length. -/
def planeBlockCount : ℕ → ℕ
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 24
  | 6 => 40
  | 7 => 72
  | _ => 0

lemma planeBlocks_counts :
    ∀ j : Fin 6, (planeBlocks (j.val + 2)).length = planeBlockCount (j.val + 2) := by
  native_decide

def planeLanguage (n : ℕ) : List (List Rotation) :=
  if n = 0 then [[]]
  else planeCode.flatMap fun b =>
    if 0 < b.length ∧ b.length ≤ n then
      (planeLanguage (n - b.length)).map (b ++ ·)
    else []
termination_by n
decreasing_by all_goals omega

lemma planeLanguage_zero : planeLanguage 0 = [[]] := by
  rw [planeLanguage]
  simp

lemma planeLanguage_word_length (n : ℕ) :
    ∀ rs ∈ planeLanguage n, rs.length = n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro rs hrs
      by_cases hn : n = 0
      · subst n
        simpa [planeLanguage_zero] using hrs
      · rw [planeLanguage, if_neg hn] at hrs
        obtain ⟨b, _, hb⟩ := List.mem_flatMap.mp hrs
        split at hb
        · rename_i hlen
          obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hb
          rw [List.length_append, ih (n - b.length) (by omega) tail htail]
          omega
        · simp at hb

lemma prefixCode_append_injective {α : Type*} (code : List (List α))
    (hprefix : ∀ a ∈ code, ∀ b ∈ code, a <+: b → a = b)
    {a b x y : List α}
    (ha : a ∈ code) (hb : b ∈ code) (h : a ++ x = b ++ y) :
    a = b ∧ x = y := by
  have hp : a <+: b ++ y := h ▸ List.prefix_append a x
  have hab : a = b := by
    rcases List.prefix_or_prefix_of_prefix hp (List.prefix_append b y) with hpre | hpre
    · exact hprefix a ha b hb hpre
    · exact (hprefix b hb a ha hpre).symm
  subst b
  exact ⟨rfl, List.append_cancel_left h⟩

lemma planeCode_append_injective {a b x y : List Rotation}
    (ha : a ∈ planeCode) (hb : b ∈ planeCode) (h : a ++ x = b ++ y) :
    a = b ∧ x = y :=
  prefixCode_append_injective planeCode planeCode_prefix_free ha hb h

lemma planeLanguage_nodup (n : ℕ) : (planeLanguage n).Nodup := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      by_cases hn : n = 0
      · subst n
        simp [planeLanguage_zero]
      · rw [planeLanguage, if_neg hn]
        apply List.nodup_flatMap.mpr
        constructor
        · intro b _
          split
          · rename_i hlen
            exact (ih (n - b.length) (by omega)).map
              (fun _ _ h => List.append_cancel_left h)
          · simp
        · apply planeCode_nodup.imp_of_mem
          intro a b ha hb hne word hwa hwb
          dsimp only at hwa hwb
          split at hwa <;> split at hwb <;> try simp_all only [List.not_mem_nil]
          obtain ⟨x, _, hx⟩ := List.mem_map.mp hwa
          obtain ⟨y, _, hy⟩ := List.mem_map.mp hwb
          exact hne (planeCode_append_injective ha hb (hx.trans hy.symm)).1

lemma planeLanguage_geometry (n : ℕ) :
    ∀ rs ∈ planeLanguage n, ∀ (i : Fin 4) (p : Vec3),
      (slabNewWedges (planeStart i p) rs).Pairwise interiorDisjoint ∧
      ∀ w ∈ slabNewWedges (planeStart i p) rs, p 0 < w.center 0 := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro rs hrs i p
      by_cases hn : n = 0
      · subst n
        have : rs = [] := by simpa [planeLanguage_zero] using hrs
        subst rs
        simp [slabNewWedges]
      · rw [planeLanguage, if_neg hn] at hrs
        obtain ⟨b, hb, hmem⟩ := List.mem_flatMap.mp hrs
        split at hmem
        · rename_i hlen
          obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hmem
          obtain ⟨hblock, hplane, j, q, hrun, hq⟩ :=
            planeBlock_at (planeCode_valid hb) i p
          obtain ⟨htvalid, htpos⟩ := ih (n - b.length) (by omega) tail htail j q
          rw [slabNewWedges_append, hrun]
          refine ⟨List.pairwise_append.mpr ⟨hblock, htvalid, ?_⟩, ?_⟩
          · intro a ha c hc
            left
            intro heq
            have hax := hplane a ha
            have hcx := htpos c hc
            have heqx := congrFun heq 0
            omega
          · intro w hw
            rcases List.mem_append.mp hw with hw | hw
            · have := hplane w hw
              omega
            · have := htpos w hw
              omega
        · simp at hmem

lemma planeLanguage_valid (n : ℕ) (rs : List Rotation) (hrs : rs ∈ planeLanguage n) :
    ValidList rs := by
  obtain ⟨hvalid, hpos⟩ := planeLanguage_geometry n rs hrs 0 zeroVec
  unfold ValidList collisionFree
  rw [wedges_eq_slabNewWedges]
  refine List.pairwise_cons.mpr ⟨?_, hvalid⟩
  intro w hw
  left
  intro heq
  have hx := hpos w hw
  have heqx := congrFun heq 0
  change 0 = w.center 0 at heqx
  change 0 < w.center 0 at hx
  omega

theorem planeLanguage_length_le (n : ℕ) :
    (planeLanguage n).length ≤ countValidFormulas n := by
  have hsub : (planeLanguage n).toFinset ⊆ (validRotationLists n).toFinset := by
    intro rs hrs
    have hmem := List.mem_toFinset.mp hrs
    apply List.mem_toFinset.mpr
    have hv := (mem_validRotationLists rs).mpr (planeLanguage_valid n rs hmem)
    simpa [planeLanguage_word_length n rs hmem] using hv
  calc
    (planeLanguage n).length = (planeLanguage n).toFinset.card :=
      (List.toFinset_card_of_nodup (planeLanguage_nodup n)).symm
    _ ≤ (validRotationLists n).toFinset.card := Finset.card_le_card hsub
    _ = (validRotationLists n).length :=
      List.toFinset_card_of_nodup (validRotationLists_nodup n)
    _ = countValidFormulas n := fastCountValidFormulas_eq n

private lemma planeLanguage_choices_length (n j : ℕ) (hj : 0 < j) :
    ((planeBlocks j).flatMap fun b =>
      if 0 < b.length ∧ b.length ≤ n then
        (planeLanguage (n - b.length)).map (b ++ ·)
      else []).length =
      if j ≤ n then (planeBlocks j).length * (planeLanguage (n - j)).length else 0 := by
  have heq :
      ((planeBlocks j).flatMap fun b =>
        if 0 < b.length ∧ b.length ≤ n then
          (planeLanguage (n - b.length)).map (b ++ ·)
        else []) =
      (planeBlocks j).flatMap (fun b =>
        if j ≤ n then (planeLanguage (n - j)).map (b ++ ·) else []) := by
    apply List.flatMap_congr
    intro b hb
    simp only [(mem_planeBlocks.mp hb).2, hj, true_and]
  rw [heq]
  split_ifs <;> simp [List.length_flatMap]

lemma planeLanguage_length_rec (n : ℕ) (hn : n ≠ 0) :
    (planeLanguage n).length =
      (if 2 ≤ n then 4 * (planeLanguage (n - 2)).length else 0) +
      (if 3 ≤ n then 8 * (planeLanguage (n - 3)).length else 0) +
      (if 4 ≤ n then 16 * (planeLanguage (n - 4)).length else 0) +
      (if 5 ≤ n then 24 * (planeLanguage (n - 5)).length else 0) +
      (if 6 ≤ n then 40 * (planeLanguage (n - 6)).length else 0) +
      (if 7 ≤ n then 72 * (planeLanguage (n - 7)).length else 0) := by
  rw [planeLanguage, if_neg hn, planeCode_by_length]
  simp only [List.flatMap_append, List.length_append]
  rw [planeLanguage_choices_length n 2 (by norm_num),
    planeLanguage_choices_length n 3 (by norm_num),
    planeLanguage_choices_length n 4 (by norm_num),
    planeLanguage_choices_length n 5 (by norm_num),
    planeLanguage_choices_length n 6 (by norm_num),
    planeLanguage_choices_length n 7 (by norm_num)]
  have h2 := planeBlocks_counts 0
  have h3 := planeBlocks_counts 1
  have h4 := planeBlocks_counts 2
  have h5 := planeBlocks_counts 3
  have h6 := planeBlocks_counts 4
  have h7 := planeBlocks_counts 5
  norm_num [planeBlockCount] at h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6, h7]

end RubiksSnake

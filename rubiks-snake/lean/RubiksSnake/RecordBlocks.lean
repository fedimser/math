import RubiksSnake.SlabBlocks
import RubiksSnake.WindowUpperComputation

/-!
# Blocks with an occupied interface

A block advances from plane zero to plane one, but may revisit plane zero.
The preceding block is retained as a finite boundary state. Compatibility
is checked only on the shared plane; all older wedges lie strictly behind it.
-/

namespace RubiksSnake

open WindowComputation (Coord CompactWedge)

def recordCoords (v : Vec3) : Coord := (v 0, v 1, v 2)

@[simp] lemma vec_recordCoords (v : Vec3) : WindowComputation.vec (recordCoords v) = v := by
  funext i
  fin_cases i <;> rfl

def recordHead (i : Fin 4) : CompactWedge :=
  ⟨(0, 0, 0), WindowComputation.neg (recordCoords (planePrevious i)), (1, 0, 0)⟩

def recordPath (i : Fin 4) (rs : List Rotation) : List CompactWedge :=
  WindowComputation.path (0, 0, 0) (recordCoords (planePrevious i)) (1, 0, 0) rs

lemma recordPath_map (i : Fin 4) (rs : List Rotation) :
    (recordPath i rs).map CompactWedge.toWedge =
      (planeStart i zeroVec).wedge :: slabNewWedges (planeStart i zeroVec) rs := by
  rw [recordPath, WindowComputation.path_map, vec_recordCoords]
  have hz : WindowComputation.vec (0, 0, 0) = zeroVec := by
    funext j
    fin_cases j <;> rfl
  rw [hz]
  exact wedgePath_eq_slabNewWedges (planeStart i zeroVec) rs

def recordTranslate (p : Coord) (w : CompactWedge) : CompactWedge :=
  ⟨WindowComputation.add p w.center, w.entrance, w.exit⟩

@[simp] lemma recordTranslate_toWedge (p : Coord) (w : CompactWedge) :
    (recordTranslate p w).toWedge =
      translateWedge (WindowComputation.vec p) w.toWedge := by
  simp [recordTranslate, CompactWedge.toWedge, translateWedge]

def recordPhase (v : Vec3) : Fin 4 :=
  if v = ey then 0 else if v = ez then 1 else if v = negVec ey then 2 else 3

def IsRecordBlock (rs : List Rotation) : Prop :=
  ∀ i : Fin 4,
    let s := planeStart i zeroVec
    let ws := recordPath i rs
    let last := slabRun s rs
    ws.Pairwise WindowComputation.disjoint ∧
    (∀ w ∈ ws.tail, 0 ≤ w.center.1 ∧ w.center.1 ≤ 1) ∧
    last.center 0 = 1 ∧ last.axis = ex ∧
    last.previous = planePrevious (recordPhase last.previous)

instance (rs : List Rotation) : Decidable (IsRecordBlock rs) := by
  unfold IsRecordBlock WindowComputation.disjoint
  infer_instance

def recordCandidates : Coord → Coord → Coord → ℕ → List (List Rotation)
  | p, _, axis, 0 => if p.1 = 1 ∧ axis = (1, 0, 0) then [[]] else []
  | p, previous, axis, k + 1 =>
      if 0 ≤ p.1 + axis.1 ∧ p.1 + axis.1 ≤ 1 then
        ([0, 1, 2, 3] : List Rotation).flatMap fun r =>
          (recordCandidates (WindowComputation.add p axis) axis
            (WindowComputation.turn axis r previous) k).map (r :: ·)
      else []

def recordCode : List (List Rotation) :=
  ((List.range 6).flatMap fun k =>
    recordCandidates (0, 0, 0) (0, 1, 0) (1, 0, 0) (k + 2)).filter
      (fun rs => decide (IsRecordBlock rs))

lemma recordCode_valid {rs : List Rotation} (h : rs ∈ recordCode) :
    IsRecordBlock rs :=
  of_decide_eq_true (List.mem_filter.mp h).2

structure RecordState where
  previous : Fin 4
  past : List CompactWedge
deriving DecidableEq

structure RecordDatum where
  word : List Rotation
  fresh : List CompactWedge
  left : List CompactWedge
  endpoint : SlabState
  next : RecordState

def recordDatum (i : Fin 4) (rs : List Rotation) : RecordDatum :=
  let s := planeStart i zeroVec
  let ws := recordPath i rs
  let last := slabRun s rs
  ⟨rs, ws.tail, ws.tail.filter (fun w => w.center.1 == 0), last,
    ⟨recordPhase last.previous,
      ws.map (recordTranslate (WindowComputation.neg (recordCoords last.center)))⟩⟩

@[simp] lemma recordDatum_word (i : Fin 4) (rs : List Rotation) :
    (recordDatum i rs).word = rs := rfl

def recordTables : Fin 4 → List RecordDatum :=
  ![recordCode.map (recordDatum 0), recordCode.map (recordDatum 1),
    recordCode.map (recordDatum 2), recordCode.map (recordDatum 3)]

lemma recordTables_eq (i : Fin 4) :
    recordTables i = recordCode.map (recordDatum i) := by
  fin_cases i <;> rfl

lemma mem_recordTables {i : Fin 4} {d : RecordDatum} :
    d ∈ recordTables i ↔ ∃ rs ∈ recordCode, recordDatum i rs = d := by
  rw [recordTables_eq]
  exact List.mem_map

def recordInitial (i : Fin 4) : RecordState :=
  ⟨i, [recordHead i]⟩

def recordStates : List RecordState :=
  (List.finRange 4).map recordInitial ++
    (List.finRange 4).flatMap (fun i => (recordTables i).map RecordDatum.next)

lemma recordInitial_mem (i : Fin 4) : recordInitial i ∈ recordStates := by
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩

lemma recordNext_mem {i : Fin 4} {d : RecordDatum} (h : d ∈ recordTables i) :
    d.next ∈ recordStates := by
  apply List.mem_append_right
  exact List.mem_flatMap.mpr ⟨i, List.mem_finRange i, List.mem_map.mpr ⟨d, h, rfl⟩⟩

def recordFits (s : RecordState) (d : RecordDatum) : Bool :=
  (s.past.filter (fun w => w.center.1 == 0)).all fun old =>
    d.left.all fun new => decide (WindowComputation.disjoint old new)

def recordOptions (s : RecordState) (j : ℕ) : List RecordDatum :=
  (recordTables s.previous).filter fun d => d.word.length == j && recordFits s d

def recordLowerCount : ℕ → ℕ
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 24
  | 6 => 56
  | 7 => 168
  | _ => 0

private lemma checked :
    recordCode.Nodup ∧
    (∀ rs ∈ recordCode, 2 ≤ rs.length ∧ rs.length ≤ 7) ∧
    (∀ a ∈ recordCode, ∀ b ∈ recordCode, a <+: b → a = b) ∧
    (recordCode =
      recordCode.filter (fun rs => rs.length == 2) ++
      recordCode.filter (fun rs => rs.length == 3) ++
      recordCode.filter (fun rs => rs.length == 4) ++
      recordCode.filter (fun rs => rs.length == 5) ++
      recordCode.filter (fun rs => rs.length == 6) ++
      recordCode.filter (fun rs => rs.length == 7)) ∧
    (∀ s ∈ recordStates, ∀ w ∈ s.past, w.center.1 ≤ 0) ∧
    (∀ s ∈ recordStates, ∀ j : Fin 6,
      recordLowerCount (j.val + 2) ≤ (recordOptions s (j.val + 2)).length) := by
  native_decide

lemma recordCode_nodup : recordCode.Nodup := checked.1

lemma recordCode_lengths :
    ∀ rs ∈ recordCode, 2 ≤ rs.length ∧ rs.length ≤ 7 := checked.2.1

lemma recordCode_prefix_free :
    ∀ a ∈ recordCode, ∀ b ∈ recordCode, a <+: b → a = b := checked.2.2.1

lemma recordStates_bound :
    ∀ s ∈ recordStates, ∀ w ∈ s.past, w.center.1 ≤ 0 := checked.2.2.2.2.1

lemma recordOptions_counts :
    ∀ s ∈ recordStates, ∀ j : Fin 6,
      recordLowerCount (j.val + 2) ≤ (recordOptions s (j.val + 2)).length :=
  checked.2.2.2.2.2

lemma recordDatum_endpoint {i : Fin 4} {rs : List Rotation} (h : rs ∈ recordCode) :
    (recordDatum i rs).endpoint =
      slabTranslate (recordDatum i rs).endpoint.center
        (planeStart (recordDatum i rs).next.previous zeroVec) := by
  obtain ⟨_, _, _, haxis, hprevious⟩ := recordCode_valid h i
  change SlabState.mk _ (slabRun (planeStart i zeroVec) rs).previous
      (slabRun (planeStart i zeroVec) rs).axis =
    SlabState.mk _ (planePrevious (recordPhase (slabRun (planeStart i zeroVec) rs).previous)) ex
  simp only [SlabState.mk.injEq]
  exact ⟨by funext j; simp [recordDatum, planeStart, addVec, zeroVec], hprevious, haxis⟩

lemma recordFits_cross {s : RecordState} (hs : s ∈ recordStates)
    {d : RecordDatum} (hd : d ∈ recordTables s.previous) (hfit : recordFits s d = true) :
    ∀ old ∈ s.past, ∀ new ∈ d.fresh, WindowComputation.disjoint old new := by
  obtain ⟨rs, hrs, rfl⟩ := mem_recordTables.mp hd
  obtain ⟨_, hspan, _, _, _⟩ := recordCode_valid hrs s.previous
  intro old hold new hnew
  by_cases hcenters : old.center = new.center
  · have holdx := recordStates_bound s hs old hold
    have hnewx := (hspan new hnew).1
    have heqx := congrArg Prod.fst hcenters
    have hold0 : old.center.1 = 0 := by omega
    have hnew0 : new.center.1 = 0 := by omega
    have hboundary : old ∈ s.past.filter (fun w => w.center.1 == 0) := by
      simp [hold, hold0]
    have hleft : new ∈ (recordDatum s.previous rs).left := by
      change new ∈ (recordPath s.previous rs).tail.filter (fun w => w.center.1 == 0)
      exact List.mem_filter.mpr ⟨hnew, by simpa using hnew0⟩
    exact of_decide_eq_true
      (List.all_eq_true.mp (List.all_eq_true.mp hfit old hboundary) new hleft)
  · exact Or.inl hcenters

lemma recordDatum_fresh {i : Fin 4} {d : RecordDatum} (hd : d ∈ recordTables i) :
    d.fresh.map CompactWedge.toWedge = slabNewWedges (planeStart i zeroVec) d.word := by
  obtain ⟨rs, _, rfl⟩ := mem_recordTables.mp hd
  simpa only [recordDatum, List.map_tail, List.tail_cons] using
    congrArg List.tail (recordPath_map i rs)

lemma recordDatum_next_past {i : Fin 4} {d : RecordDatum} (hd : d ∈ recordTables i) :
    d.next.past.map CompactWedge.toWedge =
      ((planeStart i zeroVec).wedge :: slabNewWedges (planeStart i zeroVec) d.word).map
        (translateWedge (negVec d.endpoint.center)) := by
  obtain ⟨rs, _, rfl⟩ := mem_recordTables.mp hd
  change ((recordPath i rs).map (recordTranslate _)).map CompactWedge.toWedge = _
  simp only [List.map_map, Function.comp_def, recordTranslate_toWedge,
    WindowComputation.vec_neg, vec_recordCoords]
  change (recordPath i rs).map
      (fun w => translateWedge (negVec (slabRun (planeStart i zeroVec) rs).center) w.toWedge) =
    ((planeStart i zeroVec).wedge :: slabNewWedges (planeStart i zeroVec) rs).map _
  rw [← recordPath_map]
  exact (List.map_map
    (g := translateWedge (negVec (slabRun (planeStart i zeroVec) rs).center))
    (f := CompactWedge.toWedge) (l := recordPath i rs)).symm

lemma recordDatum_geometry {i : Fin 4} {d : RecordDatum} (hd : d ∈ recordTables i) :
    (slabNewWedges (planeStart i zeroVec) d.word).Pairwise interiorDisjoint ∧
    (∀ w ∈ slabNewWedges (planeStart i zeroVec) d.word, 0 ≤ w.center 0) ∧
    d.endpoint.center 0 = 1 ∧
    slabRun (planeStart i zeroVec) d.word = d.endpoint ∧
    d.endpoint = slabTranslate d.endpoint.center (planeStart d.next.previous zeroVec) := by
  obtain ⟨rs, hrs, rfl⟩ := mem_recordTables.mp hd
  obtain ⟨hvalid, hspan, hx, _, _⟩ := recordCode_valid hrs i
  refine ⟨?_, ?_, hx, rfl, recordDatum_endpoint hrs⟩
  · have hmapped : ((recordPath i rs).map CompactWedge.toWedge).Pairwise interiorDisjoint := by
      rw [List.pairwise_map]
      exact hvalid.imp fun h => (WindowComputation.disjoint_iff _ _).mp h
    rw [recordPath_map] at hmapped
    exact hmapped.tail
  · intro w hw
    rw [← recordDatum_fresh hd] at hw
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
    exact (hspan v hv).1

def recordLanguage (s : RecordState) (n : ℕ) : List (List Rotation) :=
  if n = 0 then [[]]
  else (recordTables s.previous).flatMap fun d =>
    if 0 < d.word.length ∧ d.word.length ≤ n ∧ recordFits s d = true then
      (recordLanguage d.next (n - d.word.length)).map (d.word ++ ·)
    else []
termination_by n
decreasing_by all_goals omega

lemma recordLanguage_zero (s : RecordState) : recordLanguage s 0 = [[]] := by
  rw [recordLanguage]
  simp

lemma recordLanguage_geometry (n : ℕ) :
    ∀ s ∈ recordStates, ∀ rs ∈ recordLanguage s n,
      (slabNewWedges (planeStart s.previous zeroVec) rs).Pairwise interiorDisjoint ∧
      (∀ w ∈ slabNewWedges (planeStart s.previous zeroVec) rs, 0 ≤ w.center 0) ∧
      ∀ old ∈ s.past, ∀ w ∈ slabNewWedges (planeStart s.previous zeroVec) rs,
        interiorDisjoint old.toWedge w := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro s hs rs hrs
      by_cases hn : n = 0
      · subst n
        have : rs = [] := by simpa [recordLanguage_zero] using hrs
        subst rs
        simp [slabNewWedges]
      · rw [recordLanguage, if_neg hn] at hrs
        obtain ⟨d, hd, hmem⟩ := List.mem_flatMap.mp hrs
        split at hmem
        · rename_i hlen
          obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hmem
          obtain ⟨hblock, hnonneg, hcx, hrun, hframe⟩ := recordDatum_geometry hd
          obtain ⟨htvalid, htpos, htcross⟩ :=
            ih (n - d.word.length) (by omega) d.next (recordNext_mem hd) tail htail
          rw [slabNewWedges_append, hrun, hframe, slabNewWedges_translate]
          let c := d.endpoint.center
          have hparts :
              ∀ a ∈ slabNewWedges (planeStart s.previous zeroVec) d.word,
                ∀ b ∈ slabNewWedges (planeStart d.next.previous zeroVec) tail,
                  interiorDisjoint a (translateWedge c b) := by
            intro a ha b hb
            have hpa : translateWedge (negVec c) a ∈
                d.next.past.map CompactWedge.toWedge := by
              rw [recordDatum_next_past hd]
              exact List.mem_map.mpr ⟨a, List.mem_cons_of_mem _ ha, rfl⟩
            obtain ⟨old, hold, heq⟩ := List.mem_map.mp hpa
            have hdis := htcross old hold b hb
            rw [heq] at hdis
            have translated := (translateWedge_interiorDisjoint c _ _).mpr hdis
            simpa [translateWedge_add] using translated
          refine ⟨List.pairwise_append.mpr ⟨hblock, ?_, ?_⟩, ?_, ?_⟩
          · rw [List.pairwise_map]
            exact htvalid.imp fun h => (translateWedge_interiorDisjoint c _ _).mpr h
          · intro a ha b hb
            obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hb
            exact hparts a ha v hv
          · intro w hw
            rcases List.mem_append.mp hw with hw | hw
            · exact hnonneg w hw
            · obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
              have hpos := htpos v hv
              change 0 ≤ d.endpoint.center 0 + v.center 0
              omega
          · intro old hold w hw
            rcases List.mem_append.mp hw with hw | hw
            · rw [← recordDatum_fresh hd] at hw
              obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
              exact (WindowComputation.disjoint_iff _ _).mp
                (recordFits_cross hs hd hlen.2.2 old hold v hv)
            · obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hw
              left
              intro heq
              have holdx := recordStates_bound s hs old hold
              have hpos := htpos v hv
              have heqx := congrFun heq 0
              change old.center.1 = d.endpoint.center 0 + v.center 0 at heqx
              omega
        · simp at hmem

lemma recordLanguage_initial_valid (n : ℕ) (rs : List Rotation)
    (hrs : rs ∈ recordLanguage (recordInitial 0) n) : ValidList rs := by
  obtain ⟨hvalid, _, hcross⟩ :=
    recordLanguage_geometry n (recordInitial 0) (recordInitial_mem 0) rs hrs
  unfold ValidList collisionFree
  rw [wedges_eq_slabNewWedges]
  refine List.pairwise_cons.mpr ⟨?_, hvalid⟩
  intro w hw
  have hhead : (recordHead 0).toWedge = (SlabState.mk zeroVec ey ex).wedge := by
    simp only [recordHead, CompactWedge.toWedge, WindowComputation.vec_neg, vec_recordCoords]
    change Wedge.mk (WindowComputation.vec (0, 0, 0)) (negVec ey) ex =
      Wedge.mk zeroVec (negVec ey) ex
    congr 1
    funext i
    fin_cases i <;> rfl
  rw [← hhead]
  exact hcross (recordHead 0) (by simp [recordInitial]) w hw

lemma recordLanguage_word_length (n : ℕ) :
    ∀ s, ∀ rs ∈ recordLanguage s n, rs.length = n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro s rs hrs
      by_cases hn : n = 0
      · subst n
        simpa [recordLanguage_zero] using hrs
      · rw [recordLanguage, if_neg hn] at hrs
        obtain ⟨d, _, hd⟩ := List.mem_flatMap.mp hrs
        split at hd
        · rename_i hlen
          obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hd
          rw [List.length_append, ih (n - d.word.length) (by omega) d.next tail htail]
          omega
        · simp at hd

lemma recordLanguage_nodup (n : ℕ) : ∀ s, (recordLanguage s n).Nodup := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro s
      by_cases hn : n = 0
      · subst n
        simp [recordLanguage_zero]
      · rw [recordLanguage, if_neg hn]
        apply List.nodup_flatMap.mpr
        constructor
        · intro d _
          split
          · rename_i hlen
            exact (ih (n - d.word.length) (by omega) d.next).map
              (fun _ _ h => List.append_cancel_left h)
          · simp
        · rw [recordTables_eq]
          have hnodup : (recordCode.map (recordDatum s.previous)).Nodup :=
            recordCode_nodup.map fun a b h => congrArg RecordDatum.word h
          apply hnodup.imp_of_mem
          intro a b ha hb hne word hwa hwb
          dsimp only at hwa hwb
          split at hwa <;> split at hwb <;> try simp_all only [List.not_mem_nil]
          obtain ⟨x, _, hx⟩ := List.mem_map.mp hwa
          obtain ⟨y, _, hy⟩ := List.mem_map.mp hwb
          obtain ⟨ra, hra, rfl⟩ := List.mem_map.mp ha
          obtain ⟨rb, hrb, rfl⟩ := List.mem_map.mp hb
          have hab := (prefixCode_append_injective recordCode recordCode_prefix_free
            hra hrb (hx.trans hy.symm)).1
          subst rb
          exact hne rfl

theorem recordLanguage_length_le (n : ℕ) :
    (recordLanguage (recordInitial 0) n).length ≤ countValidFormulas n := by
  have hsub :
      (recordLanguage (recordInitial 0) n).toFinset ⊆ (validRotationLists n).toFinset := by
    intro rs hrs
    have hmem := List.mem_toFinset.mp hrs
    apply List.mem_toFinset.mpr
    have hv := (mem_validRotationLists rs).mpr (recordLanguage_initial_valid n rs hmem)
    simpa [recordLanguage_word_length n (recordInitial 0) rs hmem] using hv
  calc
    (recordLanguage (recordInitial 0) n).length =
        (recordLanguage (recordInitial 0) n).toFinset.card :=
      (List.toFinset_card_of_nodup (recordLanguage_nodup n _)).symm
    _ ≤ (validRotationLists n).toFinset.card := Finset.card_le_card hsub
    _ = (validRotationLists n).length :=
      List.toFinset_card_of_nodup (validRotationLists_nodup n)
    _ = countValidFormulas n := fastCountValidFormulas_eq n

private lemma recordTables_by_length (i : Fin 4) :
    recordTables i =
      (recordTables i).filter (fun d => d.word.length == 2) ++
      (recordTables i).filter (fun d => d.word.length == 3) ++
      (recordTables i).filter (fun d => d.word.length == 4) ++
      (recordTables i).filter (fun d => d.word.length == 5) ++
      (recordTables i).filter (fun d => d.word.length == 6) ++
      (recordTables i).filter (fun d => d.word.length == 7) := by
  simpa only [recordTables_eq, List.map_append, List.filter_map, Function.comp_def,
    recordDatum_word]
    using congrArg (List.map (recordDatum i)) checked.2.2.2.1

def recordWeightedCount (s : RecordState) (f : ℕ → ℕ) : ℕ :=
  (((recordTables s.previous).filter (recordFits s)).map (fun d => f d.word.length)).sum

private lemma recordWeightedCount_group (s : RecordState) (j : ℕ) (f : ℕ → ℕ) :
    ((((recordTables s.previous).filter (fun d => d.word.length == j)).filter
      (recordFits s)).map (fun d => f d.word.length)).sum =
      (recordOptions s j).length * f j := by
  have heq :
      ((recordTables s.previous).filter (fun d => d.word.length == j)).filter
        (recordFits s) = recordOptions s j := by
    simp only [recordOptions, List.filter_filter, Bool.and_comm]
  rw [heq]
  have hmap : (recordOptions s j).map (fun d => f d.word.length) =
      (recordOptions s j).map (fun _ => f j) := by
    apply List.map_congr_left
    intro d hd
    have hlen : d.word.length = j := by
      have hpair : (d.word.length == j) = true ∧ recordFits s d = true := by
        simpa only [Bool.and_eq_true] using (List.mem_filter.mp hd).2
      simpa only [beq_iff_eq] using hpair.1
    rw [hlen]
  rw [hmap]
  simp

lemma recordWeightedCount_eq (s : RecordState) (f : ℕ → ℕ) :
    recordWeightedCount s f =
      (recordOptions s 2).length * f 2 + (recordOptions s 3).length * f 3 +
      (recordOptions s 4).length * f 4 + (recordOptions s 5).length * f 5 +
      (recordOptions s 6).length * f 6 + (recordOptions s 7).length * f 7 := by
  unfold recordWeightedCount
  rw [recordTables_by_length s.previous]
  simp only [List.filter_append, List.map_append, List.sum_append,
    recordWeightedCount_group]

lemma recordLanguage_step_lower (n : ℕ) (hn : n ≠ 0)
    (s : RecordState) (f : ℕ → ℕ)
    (hnext : ∀ d ∈ recordTables s.previous,
      0 < d.word.length → d.word.length ≤ n → recordFits s d = true →
      f (n - d.word.length) ≤ (recordLanguage d.next (n - d.word.length)).length) :
    recordWeightedCount s (fun j => if j ≤ n then f (n - j) else 0) ≤
      (recordLanguage s n).length := by
  rw [recordLanguage, if_neg hn, List.length_flatMap]
  have hfilter :
      recordWeightedCount s (fun j => if j ≤ n then f (n - j) else 0) =
        ((recordTables s.previous).map fun d =>
          if recordFits s d then
            (if d.word.length ≤ n then f (n - d.word.length) else 0)
          else 0).sum := by
    unfold recordWeightedCount
    induction recordTables s.previous with
    | nil => rfl
    | cons d ds ih =>
        cases hf : recordFits s d <;> simp [hf, ih]
  rw [hfilter]
  apply List.sum_le_sum
  intro d hd
  obtain ⟨rs, hrs, hdata⟩ := mem_recordTables.mp hd
  have hpos : 0 < d.word.length := by
    have hlen := (recordCode_lengths rs hrs).1
    subst d
    exact lt_of_lt_of_le (by norm_num) hlen
  by_cases hfit : recordFits s d = true
  · by_cases hlen : d.word.length ≤ n
    · simpa [hpos, hlen, hfit] using hnext d hd hpos hlen hfit
    · simp [hlen]
  · simp [hfit]

def recordRenewal (n : ℕ) : ℕ :=
  if n = 0 then 1
  else
    (if 2 ≤ n then 4 * recordRenewal (n - 2) else 0) +
    (if 3 ≤ n then 8 * recordRenewal (n - 3) else 0) +
    (if 4 ≤ n then 16 * recordRenewal (n - 4) else 0) +
    (if 5 ≤ n then 24 * recordRenewal (n - 5) else 0) +
    (if 6 ≤ n then 56 * recordRenewal (n - 6) else 0) +
    (if 7 ≤ n then 168 * recordRenewal (n - 7) else 0)
termination_by n
decreasing_by all_goals omega

lemma recordRenewal_le_language (n : ℕ) :
    ∀ s ∈ recordStates, recordRenewal n ≤ (recordLanguage s n).length := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      intro s hs
      by_cases hn : n = 0
      · subst n
        simp [recordRenewal, recordLanguage_zero]
      · apply le_trans _ (recordLanguage_step_lower n hn s recordRenewal ?_)
        · rw [recordWeightedCount_eq, recordRenewal, if_neg hn]
          have h2 := recordOptions_counts s hs 0
          have h3 := recordOptions_counts s hs 1
          have h4 := recordOptions_counts s hs 2
          have h5 := recordOptions_counts s hs 3
          have h6 := recordOptions_counts s hs 4
          have h7 := recordOptions_counts s hs 5
          norm_num [recordLowerCount] at h2 h3 h4 h5 h6 h7
          simp only [mul_ite, mul_zero]
          gcongr <;> split_ifs <;> gcongr
        · intro d hd hpos hlen _
          exact ih (n - d.word.length) (by omega) d.next (recordNext_mem hd)

theorem recordRenewal_le_countValidFormulas (n : ℕ) :
    recordRenewal n ≤ countValidFormulas n :=
  (recordRenewal_le_language n (recordInitial 0) (recordInitial_mem 0)).trans
    (recordLanguage_length_le n)

end RubiksSnake

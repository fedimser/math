import RubiksSnake.SlabGeometry

/-! The words counted by the slab traversal describe valid wedge paths. -/

namespace RubiksSnake.SlabEnumeration

open CardinalDirections

/-- Wedges described by outgoing direction indices from center `p` and the
given incoming direction, with one wedge per letter. -/
def path (p : Vec3) (incoming : Nat) (ds : List Nat) : List Wedge :=
  directionalPath p (toDirection incoming) (ds.map toDirection)

/-- A board placement at `p`, interpreting its incoming and outgoing indices
as cardinal directions modulo six. -/
def placed (p : Vec3) (incoming outgoing : Nat) : SlabBoard.Placed :=
  ⟨p, toDirection incoming, toDirection outgoing⟩

/-- An empty direction word places no wedges. -/
@[simp] lemma path_nil (p : Vec3) (incoming : Nat) : path p incoming [] = [] := rfl

/-- A nonempty word places its first wedge at `p`, then continues one unit
along the first outgoing direction. -/
@[simp] lemma path_cons (p : Vec3) (incoming outgoing : Nat) (ds : List Nat) :
    path p incoming (outgoing :: ds) =
      (placed p incoming outgoing).wedge ::
        path (addVec p (vectorOf outgoing)) outgoing ds := rfl

/-- Enumerated internal words use valid direction indices, and appending the
final `+x` exit gives perpendicular consecutive directions from the cursor frame. -/
lemma words_directions (width side : Nat) (irreducible : Bool) (remaining : Nat)
    (s : Cursor) (board : ByteArray) (hincoming : s.incoming < 6) :
    ∀ w ∈ words width side irreducible remaining s board,
      (∀ d ∈ w, d < 6) ∧
        Compatible (toDirection s.incoming) ((w ++ [0]).map toDirection) := by
  induction remaining generalizing s board with
  | zero =>
    intro w hw
    rw [words] at hw
    split at hw
    · rename_i hexit
      have hw' : w = [] := by simpa using hw
      subst w
      have hin := (canExit_spec _ _ _ _).mp hexit |>.2.1
      simpa [Compatible, Perpendicular, toDirection_val hincoming] using hin
    · simp at hw
  | succ remaining ih =>
    intro w hw
    rw [words] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw
      · rename_i hexit
        have hw' : w = [] := by simpa using hw
        subst w
        have hin := (canExit_spec _ _ _ _).mp hexit |>.2.1
        simpa [Compatible, Perpendicular, toDirection_val hincoming] using hin
      · simp at hw
    · obtain ⟨outgoing, houtgoing, hw⟩ := List.mem_flatMap.mp hw
      have houtgoing' := mem_directions.mp houtgoing
      split at hw
      · rename_i hmove
        obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
        obtain ⟨hsmall, hcompatible⟩ :=
          ih (advance side s outgoing) _ (by simpa using houtgoing') tail htail
        refine ⟨?_, ?_⟩
        · intro d hd
          rcases List.mem_cons.mp hd with rfl | hd
          · exact houtgoing'
          · exact hsmall d hd
        · change Perpendicular (toDirection s.incoming) (toDirection outgoing) ∧
            Compatible (toDirection outgoing) ((tail ++ [0]).map toDirection)
          refine ⟨?_, by simpa using hcompatible⟩
          have hperp := (canMove_spec _ _ _ _).mp hmove |>.1
          simpa [Perpendicular, toDirection_val hincoming, toDirection_val houtgoing']
            using Ne.symm hperp
      · simp at hw

/-- Every complete slab block uses indices below six and perpendicular turns
from incoming direction `+x`, including its final exit letter. -/
lemma blockWords_directions (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) :
    (∀ d ∈ w, d < 6) ∧ Compatible 0 (w.map toDirection) := by
  obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
  obtain ⟨hsmall, hcompatible⟩ :=
    words_directions width (2 * limit + 1) irreducible limit (initialCursor limit)
      (initialBoard width limit) (by simp [initialCursor]) tail htail
  refine ⟨?_, hcompatible⟩
  intro d hd
  rcases List.mem_append.mp hd with hd | hd
  · exact hsmall d hd
  · have hd' : d = 0 := by simpa using hd
    subst d
    decide

/-- Converting valid natural indices to directions and taking their values
recovers the original word. -/
lemma directions_map_val {w : List Nat} (hw : ∀ d ∈ w, d < 6) :
    (w.map toDirection).map Fin.val = w := by
  rw [List.map_map]
  conv_rhs => rw [← List.map_id w]
  apply List.map_congr_left
  intro d hd
  exact toDirection_val (hw d hd)

/-- Cardinal-direction conversion is injective on words whose indices are below six. -/
lemma directions_map_injective {a b : List Nat}
    (ha : ∀ d ∈ a, d < 6) (hb : ∀ d ∈ b, d < 6)
    (hab : a.map toDirection = b.map toDirection) : a = b := by
  simpa [directions_map_val ha, directions_map_val hb] using
    congrArg (List.map Fin.val) hab

/-- With matching cursor and geometric positions and valid direction indices,
an accepted move passes the geometric board-fit predicate. -/
private lemma move_fits (width limit : Nat) (s : Cursor) (board : ByteArray)
    (p : Vec3) (outgoing : Nat) (hposition : s.position = index limit p)
    (hincoming : s.incoming < 6) (houtgoing : outgoing < 6)
    (hmove : canMove width s board[s.position]! outgoing = true) :
    SlabBoard.Fits (index limit) board (placed p s.incoming outgoing) := by
  have hfit := (canMove_spec _ _ _ _).mp hmove |>.2.2.2
  have hlookup := SlabBoard.complement_lookup (toDirection s.incoming) (toDirection outgoing)
  simp only [toDirection_val hincoming, toDirection_val houtgoing] at hlookup
  rw [hlookup] at hfit
  simpa [SlabBoard.Fits, placed, toDirection_val hincoming, toDirection_val houtgoing,
    ← hposition] using hfit

/-- An accepted terminal exit passes the geometric board-fit predicate for
the final `+x` wedge at the cursor's geometric center. -/
private lemma exit_fits (width limit : Nat) (irreducible : Bool) (s : Cursor)
    (board : ByteArray) (p : Vec3) (hposition : s.position = index limit p)
    (hincoming : s.incoming < 6)
    (hexit : canExit width irreducible s board[s.position]! = true) :
    SlabBoard.Fits (index limit) board (placed p s.incoming 0) := by
  have hfit := (canExit_spec _ _ _ _).mp hexit |>.2.2.1
  have hlookup := SlabBoard.complement_lookup (toDirection s.incoming) 0
  simp only [toDirection_val hincoming, Fin.val_zero, Nat.add_zero] at hlookup
  rw [hlookup] at hfit
  simpa [SlabBoard.Fits, placed, toDirection_val hincoming, ← hposition] using hfit

/-- A fitting terminal `+x` wedge is internally collision-free and disjoint
from every wedge represented by the current board. -/
private lemma terminal_sound (limit : Nat) (s : Cursor) (board : ByteArray)
    (p : Vec3) (past : List SlabBoard.Placed)
    (hrep : SlabBoard.Represents (index limit) board past)
    (hfit : SlabBoard.Fits (index limit) board (placed p s.incoming 0)) :
    (path p s.incoming [0]).Pairwise interiorDisjoint ∧
      ∀ old ∈ past, ∀ new ∈ path p s.incoming [0], interiorDisjoint old.wedge new := by
  constructor
  · simp
  · intro old hold new hnew
    have hnew' : new = (placed p s.incoming 0).wedge := by simpa using hnew
    subst new
    exact SlabBoard.fits_disjoint hrep hfit old hold

/-- From a located cursor and a correctly sized board representing the past,
every enumerated continuation plus its final exit has disjoint wedges and
avoids all represented past wedges. -/
theorem words_path_sound (width limit : Nat) (irreducible : Bool) (remaining : Nat)
    (s : Cursor) (board : ByteArray) (p : Vec3) (past : List SlabBoard.Placed)
    (hloc : Located width limit remaining s p)
    (hsize : board.size = (width + 1) * (2 * limit + 1) * (2 * limit + 1))
    (hrep : SlabBoard.Represents (index limit) board past) :
    ∀ w ∈ words width (2 * limit + 1) irreducible remaining s board,
      (path p s.incoming (w ++ [0])).Pairwise interiorDisjoint ∧
        ∀ old ∈ past, ∀ new ∈ path p s.incoming (w ++ [0]),
          interiorDisjoint old.wedge new := by
  induction remaining generalizing s board p past with
  | zero =>
    intro w hw
    rw [words] at hw
    split at hw
    · rename_i hexit
      have hw' : w = [] := by simpa using hw
      subst w
      exact terminal_sound limit s board p past hrep
        (exit_fits width limit irreducible s board p hloc.position hloc.incoming hexit)
    · simp at hw
  | succ remaining ih =>
    intro w hw
    rw [words] at hw
    rcases List.mem_append.mp hw with hw | hw
    · split at hw
      · rename_i hexit
        have hw' : w = [] := by simpa using hw
        subst w
        exact terminal_sound limit s board p past hrep
          (exit_fits width limit irreducible s board p hloc.position hloc.incoming hexit)
      · simp at hw
    · obtain ⟨outgoing, houtgoing, hw⟩ := List.mem_flatMap.mp hw
      have houtgoing' := mem_directions.mp houtgoing
      split at hw
      · rename_i hmove
        obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
        have hfit := move_fits width limit s board p outgoing hloc.position
          hloc.incoming houtgoing' hmove
        have hbound : index limit p < board.size := by
          rw [hsize]
          exact hloc.packed_bounds.2
        have hstore :
            SlabBoard.Represents (index limit)
              (board.set! s.position (entry board[s.position]! s.incoming outgoing))
              (placed p s.incoming outgoing :: past) := by
          have h := SlabBoard.represents_place (new := placed p s.incoming outgoing) hrep hbound
          simpa [placed, toDirection_val hloc.incoming, toDirection_val houtgoing',
            ← hloc.position] using h
        obtain ⟨hvalid, hcross⟩ :=
          ih (advance (2 * limit + 1) s outgoing) _
            (addVec p (vectorOf outgoing)) (placed p s.incoming outgoing :: past)
            (hloc.advance _ outgoing houtgoing' hmove)
            (by simpa using hsize) hstore tail htail
        simp only [advance_incoming] at hvalid hcross
        change ((placed p s.incoming outgoing).wedge ::
          path (addVec p (vectorOf outgoing)) outgoing (tail ++ [0])).Pairwise
            interiorDisjoint ∧ _
        refine ⟨List.pairwise_cons.mpr ⟨?_, hvalid⟩, ?_⟩
        · intro new hnew
          exact hcross _ (by simp) new hnew
        · intro old hold new hnew
          simp only [List.cons_append, path_cons, List.mem_cons] at hnew
          rcases hnew with rfl | hnew
          · exact SlabBoard.fits_disjoint hrep hfit old hold
          · exact hcross old (by simp [hold]) new hnew
      · simp at hw

/-- Every block generated from the empty board has pairwise disjoint wedge
interiors, in either unrestricted or irreducible mode. -/
theorem blockWords_valid (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) :
    (path zeroVec 0 w).Pairwise interiorDisjoint := by
  obtain ⟨tail, htail, rfl⟩ := List.mem_map.mp hw
  exact (words_path_sound width limit irreducible limit (initialCursor limit)
    (initialBoard width limit) zeroVec [] (located_initial width limit)
    (by simp [initialBoard, ByteArray.size]) (SlabBoard.empty_represents _ _) tail htail).1

/-- Every complete slab block ends with direction index zero, the `+x` exit,
regardless of the fallback supplied to `getLastD`. -/
lemma blockWords_last (width limit : Nat) (irreducible : Bool)
    {w : List Nat} (hw : w ∈ blockWords width limit irreducible) (incoming : Nat) :
    w.getLastD incoming = 0 := by
  obtain ⟨tail, _, rfl⟩ := List.mem_map.mp hw
  simp

/-- Position reached after taking all indexed cardinal steps from `p`. -/
def endPoint (p : Vec3) (ds : List Nat) : Vec3 :=
  (ds.map vectorOf).foldl addVec p

/-- The endpoint's `x` coordinate is the starting coordinate plus the word's
total `x` height. -/
lemma endPoint_x (p : Vec3) (ds : List Nat) :
    endPoint p ds 0 = p 0 + BridgeWords.height xStep ds := by
  induction ds generalizing p with
  | nil => simp [endPoint, BridgeWords.height]
  | cons outgoing ds ih =>
    simpa [endPoint, BridgeWords.height, addVec, xStep, add_assoc] using
      ih (addVec p (vectorOf outgoing))

/-- A concatenated path splits at the first word's endpoint, carrying its
last direction forward as the second path's incoming direction. -/
lemma path_append (p : Vec3) (incoming : Nat) (xs ys : List Nat) :
    path p incoming (xs ++ ys) =
      path p incoming xs ++ path (endPoint p xs) (xs.getLastD incoming) ys := by
  simp only [path, List.map_append, directionalPath_append, List.map_map,
    List.getLastD_map]
  rfl

/-- Starting a direction path at `p` translates its origin-based wedge path. -/
lemma path_at (p : Vec3) (incoming : Nat) (ds : List Nat) :
    path p incoming ds = (path zeroVec incoming ds).map (translateWedge p) := by
  simpa [path] using directionalPath_translate p zeroVec (toDirection incoming)
    (ds.map toDirection)

/-- Translating the starting center preserves pairwise interior disjointness. -/
lemma path_valid_at (p : Vec3) (incoming : Nat) (ds : List Nat)
    (hvalid : (path zeroVec incoming ds).Pairwise interiorDisjoint) :
    (path p incoming ds).Pairwise interiorDisjoint := by
  rw [path_at, List.pairwise_map]
  exact hvalid.imp fun h => (translateWedge_interiorDisjoint p _ _).mpr h

/-- Each wedge center's `x` coordinate is the starting coordinate plus the
height of a proper prefix of the direction word. -/
lemma path_center_prefix (p : Vec3) (incoming : Nat) (ds : List Nat) :
    ∀ w ∈ path p incoming ds, ∃ u v,
      ds = u ++ v ∧ v ≠ [] ∧ w.center 0 = p 0 + BridgeWords.height xStep u := by
  induction ds generalizing p incoming with
  | nil => simp
  | cons outgoing ds ih =>
    intro w hw
    rw [path_cons] at hw
    rcases List.mem_cons.mp hw with rfl | hw
    · exact ⟨[], outgoing :: ds, rfl, by simp, by simp [placed, SlabBoard.Placed.wedge]⟩
    · obtain ⟨u, v, hds, hv, hcenter⟩ := ih _ _ w hw
      refine ⟨outgoing :: u, v, by simp [hds], hv, ?_⟩
      simpa [BridgeWords.height, addVec, xStep, add_assoc] using hcenter

/-- Nonnegative prefix heights keep all wedge centers on or beyond the
starting `x`-plane. -/
lemma path_nonnegative (p : Vec3) (incoming : Nat) {ds : List Nat}
    (h : BridgeWords.NonnegativePrefixes xStep ds) :
    ∀ w ∈ path p incoming ds, p 0 ≤ w.center 0 := by
  intro w hw
  obtain ⟨u, v, hds, _, hcenter⟩ := path_center_prefix p incoming ds w hw
  have := h u v hds
  omega

/-- A bridge's wedge centers lie between its initial plane, inclusively,
and its final-height plane, strictly. -/
lemma path_height_bounds (p : Vec3) (incoming : Nat) {ds : List Nat}
    (h : BridgeWords.IsBridge xStep ds) :
    ∀ w ∈ path p incoming ds,
      p 0 ≤ w.center 0 ∧ w.center 0 < p 0 + BridgeWords.height xStep ds := by
  intro w hw
  obtain ⟨u, v, hds, hv, hcenter⟩ := path_center_prefix p incoming ds w hw
  have := h.2 u v hds hv
  omega

/-- A valid bridge ending in `+x` can precede a valid nonnegative-prefix path
without collisions: their center ranges are separated by the join plane. -/
lemma bridge_concat_valid {a b : List Nat}
    (ha : BridgeWords.IsBridge xStep a)
    (hb : BridgeWords.NonnegativePrefixes xStep b)
    (hlast : a.getLastD 0 = 0)
    (havalid : (path zeroVec 0 a).Pairwise interiorDisjoint)
    (hbvalid : (path zeroVec 0 b).Pairwise interiorDisjoint) :
    (path zeroVec 0 (a ++ b)).Pairwise interiorDisjoint := by
  rw [path_append, hlast]
  refine List.pairwise_append.mpr ⟨havalid, path_valid_at _ _ _ hbvalid, ?_⟩
  intro old hold new hnew
  have hlo := (path_height_bounds zeroVec 0 ha old hold).2
  have hhi := path_nonnegative (endPoint zeroVec a) 0 hb new hnew
  rw [endPoint_x] at hhi
  left
  intro heq
  have := congrFun heq 0
  omega

/-- Perpendicular-turn compatibility survives concatenation when the second
word starts with the incoming direction left by the first. -/
lemma compatible_append (incoming : Direction) (xs ys : List Direction)
    (hxs : Compatible incoming xs) (hys : Compatible (xs.getLastD incoming) ys) :
    Compatible incoming (xs ++ ys) := by
  induction xs generalizing incoming with
  | nil => exact hys
  | cons outgoing xs ih =>
    exact ⟨hxs.1, ih outgoing hxs.2 (by simpa only [List.getLastD_cons] using hys)⟩

/-- Words compatible with incoming `+x` remain compatible when concatenated
if the first word also ends in `+x`. -/
lemma compatible_concat {a b : List Nat}
    (ha : Compatible 0 (a.map toDirection))
    (hb : Compatible 0 (b.map toDirection)) (hlast : a.getLastD 0 = 0) :
    Compatible 0 ((a ++ b).map toDirection) := by
  rw [List.map_append]
  apply compatible_append 0 _ _ ha
  have hend : (a.map toDirection).getLastD (0 : Direction) = 0 := by
    change (a.map toDirection).getLastD (toDirection 0) = toDirection 0
    rw [List.getLastD_map, hlast]
  rw [hend]
  exact hb

/-- Encode direction letters as rotations in the standard frame with previous
direction `+y` and axis `+x`, one rotation per letter after the initial wedge. -/
def encodeWord (ds : List Nat) : List Rotation := encode 2 0 (ds.map toDirection)

/-- Direction-word length equals the encoded number of rotations. -/
@[simp] lemma encodeWord_length (ds : List Nat) : (encodeWord ds).length = ds.length := by
  simp [encodeWord]

/-- Rotation encoding is injective on valid-index words with perpendicular
turns from the standard incoming direction. -/
lemma encodeWord_injective {a b : List Nat}
    (ha : ∀ d ∈ a, d < 6) (hb : ∀ d ∈ b, d < 6)
    (hca : Compatible 0 (a.map toDirection)) (hcb : Compatible 0 (b.map toDirection))
    (hab : encodeWord a = encodeWord b) : a = b := by
  apply directions_map_injective ha hb
  exact encode_injOn (by decide : Perpendicular 2 0) hca hcb hab

/-- A compatible, collision-free direction path with nonnegative prefixes
encodes a valid formula: translating it by `+x` separates it from the initial
wedge, giving `n + 1` wedges for `n` encoded rotations. -/
lemma encodeWord_valid {ds : List Nat} (hcompatible : Compatible 0 (ds.map toDirection))
    (hnonnegative : BridgeWords.NonnegativePrefixes xStep ds)
    (hvalid : (path zeroVec 0 ds).Pairwise interiorDisjoint) :
    ValidList (encodeWord ds) := by
  unfold ValidList collisionFree encodeWord
  rw [wedges_encode hcompatible]
  change (Wedge.mk zeroVec (negVec ey) ex :: path ex 0 ds).Pairwise interiorDisjoint
  refine List.pairwise_cons.mpr ⟨?_, path_valid_at ex 0 ds hvalid⟩
  intro w hw
  have hpos := path_nonnegative ex 0 hnonnegative w hw
  left
  intro heq
  have hx := congrFun heq 0
  change 1 ≤ w.center 0 at hpos
  change 0 = w.center 0 at hx
  omega

end RubiksSnake.SlabEnumeration

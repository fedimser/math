import RubiksSnake.SnAsymptotic.BridgeCode
import RubiksSnake.SnAsymptotic.SlabIrreducible
import RubiksSnake.SnAsymptotic.SlabCountCertificate
import RubiksSnake.SnAsymptotic.SmallCounts

/-! Unique decoding and exact renewal counts for the retained slab blocks. -/

namespace RubiksSnake.SlabEnumeration

open CardinalDirections

attribute [local irreducible] blockWords

/-- Retained irreducible blocks of widths `0, 1, 2` through internal-edge
cutoffs `28, 20, 18`. This mixed-width code is decoded by bridge irreducibility,
not by prefix freedom. -/
def code : List (List Nat) :=
  blockWords 0 28 true ++ blockWords 1 20 true ++ blockWords 2 18 true

/-- Retained block words of direction length `n`, one more than their
internal-edge length. -/
def blocks (n : Nat) : List (List Nat) := code.filter fun w => w.length == n

/-- A retained word belongs to one of the three selected width/cutoff enumerations. -/
lemma mem_code {w : List Nat} :
    w ∈ code ↔ w ∈ blockWords 0 28 true ∨ w ∈ blockWords 1 20 true ∨
      w ∈ blockWords 2 18 true := by
  simp [code]

/-- Membership in a length class is retained-code membership with that exact length. -/
lemma mem_blocks {n : Nat} {w : List Nat} :
    w ∈ blocks n ↔ w ∈ code ∧ w.length = n := by
  simp [blocks]

/-- Every retained word is a primitive positive-height `x`-bridge. -/
lemma code_irreducible {w : List Nat} (hw : w ∈ code) :
    BridgeWords.IsIrreducible xStep w := by
  rcases mem_code.mp hw with hw | hw | hw
  · exact blockWords_irreducible 0 28 hw
  · exact blockWords_irreducible 1 20 hw
  · exact blockWords_irreducible 2 18 hw

/-- Retained words use valid cardinal indices and perpendicular turns from
incoming direction `+x`. -/
lemma code_directions {w : List Nat} (hw : w ∈ code) :
    (∀ d ∈ w, d < 6) ∧ Compatible 0 (w.map toDirection) := by
  rcases mem_code.mp hw with hw | hw | hw
  · exact blockWords_directions 0 28 true hw
  · exact blockWords_directions 1 20 true hw
  · exact blockWords_directions 2 18 true hw

/-- Each retained word describes a collision-free wedge path based at the origin. -/
lemma code_valid {w : List Nat} (hw : w ∈ code) :
    (path zeroVec 0 w).Pairwise interiorDisjoint := by
  rcases mem_code.mp hw with hw | hw | hw
  · exact blockWords_valid 0 28 true hw
  · exact blockWords_valid 1 20 true hw
  · exact blockWords_valid 2 18 true hw

/-- All retained blocks finish with direction index zero, the `+x` exit. -/
lemma code_last {w : List Nat} (hw : w ∈ code) : w.getLastD 0 = 0 := by
  rcases mem_code.mp hw with hw | hw | hw
  · exact blockWords_last 0 28 true hw 0
  · exact blockWords_last 1 20 true hw 0
  · exact blockWords_last 2 18 true hw 0

/-- Blocks from distinct widths cannot be equal, since their total heights
are the respective widths plus one. -/
private lemma blockWords_ne {width₁ width₂ limit₁ limit₂ : Nat} {a b : List Nat}
    (ha : a ∈ blockWords width₁ limit₁ true) (hb : b ∈ blockWords width₂ limit₂ true)
    (hne : width₁ ≠ width₂) : a ≠ b := by
  intro hab
  have ha' := blockWords_height width₁ limit₁ true ha
  have hb' := blockWords_height width₂ limit₂ true hb
  rw [hab] at ha'
  omega

/-- Each width is enumerated without duplicates, and distinct widths have
distinct heights, so their combined retained code is duplicate-free. -/
lemma code_nodup : code.Nodup := by
  unfold code
  apply List.nodup_append.mpr
  refine ⟨?_, blockWords_nodup _ _ _, ?_⟩
  · apply List.nodup_append.mpr
    refine ⟨blockWords_nodup _ _ _, blockWords_nodup _ _ _, ?_⟩
    intro a ha b hb
    exact blockWords_ne ha hb (by decide)
  · intro a ha b hb
    rcases List.mem_append.mp ha with ha | ha
    · exact blockWords_ne ha hb (by decide)
    · exact blockWords_ne ha hb (by decide)

/-- The number of retained blocks of length `n + 1` equals the certified
coefficient at that block length, corresponding to internal-edge index `n`. -/
theorem blocks_count (n : Nat) :
    (blocks (n + 1)).length = retainedCount (n + 1) := by
  simp only [blocks, code, List.filter_append, List.length_append, blockWords_count_all]
  rw [rows_checked.1, rows_checked.2.1, rows_checked.2.2]
  simp [retainedCount]

/-- Each retained length class inherits the code's absence of repeated words. -/
lemma blocks_nodup (n : Nat) : (blocks n).Nodup :=
  code_nodup.filter _

/-- Package the retained slab words with their geometry and irreducibility
proofs for the generic bridge-code counting results. -/
def bridgeCode : BridgeCode.Code where
  words := code
  nodup := code_nodup
  irreducible := fun _ hw => code_irreducible hw
  directions := fun _ hw => code_directions hw
  last := fun _ hw => code_last hw
  valid := fun _ hw => code_valid hw

/-- The generic code's length classes agree with the retained slab length classes. -/
private lemma bridgeCode_blocks (n : Nat) : BridgeCode.blocks bridgeCode n = blocks n := rfl

/-- Concatenations of retained bridge blocks with total direction length `n`;
individual block lengths are at most 29. -/
def language (n : Nat) : List (List Nat) := BridgeCode.language bridgeCode 29 n

/-- The empty word is the unique zero-length bridge concatenation. -/
lemma language_zero : language 0 = [[]] := BridgeCode.language_zero bridgeCode 29

/-- A positive-length word splits into a retained block of length `j + 1`,
for `j < 29`, followed by a language word of the remaining length. -/
lemma mem_language {n : Nat} (hn : n ≠ 0) {w : List Nat} :
    w ∈ language n ↔ ∃ j : Fin 29, j.val + 1 ≤ n ∧
      ∃ b ∈ blocks (j.val + 1), ∃ tail ∈ language (n - (j.val + 1)), b ++ tail = w := by
  exact BridgeCode.mem_language hn

/-- A language word at index `n` has exactly `n` direction letters, hence
encodes `n` rotations. -/
lemma language_word_length (n : Nat) :
    ∀ w ∈ language n, w.length = n :=
  BridgeCode.language_word_length bridgeCode 29 n

/-- All prefixes of retained-block concatenations have nonnegative `x` height. -/
lemma language_nonnegative (n : Nat) :
    ∀ w ∈ language n, BridgeWords.NonnegativePrefixes xStep w :=
  BridgeCode.language_nonnegative bridgeCode 29 n

/-- Primitive bridge irreducibility uniquely recovers the first retained
block and the language tail from their concatenation, without prefix freedom. -/
lemma language_append_injective {n m : Nat} {a b x y : List Nat}
    (ha : a ∈ code) (hb : b ∈ code)
    (hx : x ∈ language n) (hy : y ∈ language m) (h : a ++ x = b ++ y) :
    a = b ∧ x = y :=
  BridgeCode.language_append_injective ha hb hx hy h

/-- Unique bridge decoding ensures the concatenation list has no repeated words. -/
lemma language_nodup (n : Nat) : (language n).Nodup :=
  BridgeCode.language_nodup bridgeCode 29 n

/-- Every retained-block concatenation has valid cardinal indices,
perpendicular turns from incoming `+x`, and pairwise disjoint wedge interiors. -/
lemma language_geometry (n : Nat) :
    ∀ w ∈ language n,
      (∀ d ∈ w, d < 6) ∧ Compatible 0 (w.map toDirection) ∧
        (path zeroVec 0 w).Pairwise interiorDisjoint :=
  BridgeCode.language_geometry bridgeCode 29 n

/-- Encoding embeds the language into valid formulas with `n` rotations and
`n + 1` wedges, giving a lower bound on the formula count. -/
theorem language_length_le (n : Nat) : (language n).length ≤ countValidFormulas n :=
  BridgeCode.language_length_le bridgeCode 29 n

/-- Exact renewal recurrence over block lengths `j + 1` for `j < 29`, using
the retained coefficients and omitting blocks longer than the target word. -/
theorem language_length_rec (n : Nat) (hn : n ≠ 0) :
    (language n).length = ∑ j : Fin 29,
      if j.val + 1 ≤ n then retainedCount (j.val + 1) *
        (language (n - (j.val + 1))).length else 0 := by
  simpa only [language, bridgeCode_blocks, blocks_count] using
    BridgeCode.language_length_rec bridgeCode 29 n hn

end RubiksSnake.SlabEnumeration

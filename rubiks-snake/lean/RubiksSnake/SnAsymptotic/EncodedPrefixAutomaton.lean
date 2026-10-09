import RubiksSnake.SnAsymptotic.PrefixAutomatonComputation

/-!
# Integer keys for the existing prefix automaton

Words use little-endian base-four digits with a leading sentinel. Only integer
keys, graph rows, and potentials are materialized by the certificate. The
decoded dictionary is used in the proof, through exact membership and outgoing
identities, rather than allocated during the finite checks.
-/

namespace RubiksSnake
namespace EncodedPrefixAutomaton

open WindowComputation

/-- The sentinel makes every encoded word strictly positive, including the
empty word, which permits the recursive decoding argument. -/
lemma encode_pos (rs : List Rotation) : 0 < encode rs := by
  induction rs with
  | nil => decide
  | cons r rs ih => simp only [encode]; omega

/-- Decoding an encoded rotation word recovers it exactly; no dictionary
membership assumption is needed. -/
@[simp] lemma decode_encode (rs : List Rotation) : decode (encode rs) = rs := by
  induction rs with
  | nil => rw [encode, decode]; decide
  | cons r rs ih =>
      have hpos := encode_pos rs
      have hr := r.isLt
      rw [encode, decode, dif_neg (by omega)]
      have hdiv : (4 * encode rs + r.val) / 4 = encode rs := by omega
      have hmod : (4 * encode rs + r.val) % 4 = r.val := by omega
      simp only [hdiv, hmod, ih, Fin.eta]

/-- List-valued dictionary obtained by decoding the integer keys, used as the
semantic view of an integer-only certificate. -/
def dictionary (codes : Std.HashSet ℕ) : PrefixAutomaton.Dictionary :=
  Std.HashSet.ofList (codes.toList.map decode)

/-- If every stored key round-trips through decoding, membership in the decoded
dictionary is equivalent to membership of the word's encoded key. -/
lemma mem_dictionary_iff (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code) (rs : List Rotation) :
    rs ∈ dictionary codes ↔ encode rs ∈ codes := by
  simp only [dictionary, Std.HashSet.mem_ofList, List.contains_iff_mem, List.mem_map,
    Std.HashSet.mem_toList]
  constructor
  · rintro ⟨code, hcode, rfl⟩
    simpa [hcodes code hcode] using hcode
  · intro h
    exact ⟨encode rs, h, decode_encode rs⟩

/-- Under the key round-trip assumption, the Boolean dictionary lookup agrees
exactly with lookup of the encoded word. -/
lemma contains_dictionary (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code) (rs : List Rotation) :
    (dictionary codes).contains rs = codes.contains (encode rs) := by
  apply Bool.eq_iff_iff.mpr
  simpa only [Std.HashSet.contains_iff_mem] using mem_dictionary_iff codes hcodes rs

/-- For round-tripping keys, integer suffix lookup encodes exactly the semantic
longest-suffix state, including the empty fallback. -/
lemma longest_eq (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code) (rs : List Rotation) :
    longest codes rs = encode (PrefixAutomaton.longest (dictionary codes) rs) := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
      simp only [longest, PrefixAutomaton.longest, contains_dictionary codes hcodes]
      split <;> simp_all

/-- For round-tripping keys, integer and list-based successor computations
produce the same rotation-labeled destination indices. -/
lemma successors_eq (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code) (index : ℕ → ℕ)
    (rs : List Rotation) :
    successors codes index (encode rs) =
      PrefixAutomaton.indexedSuccessors (dictionary codes) (index ∘ encode) rs := by
  simp only [successors, decode_encode, PrefixAutomaton.indexedSuccessors, Function.comp_apply,
    WindowComputation.extensionAllowed_eq_valid, longest_eq codes hcodes]

/-- For round-tripping keys and a correctly indexed source, the integer graph's
weighted row is exactly the decoded automaton's semantic outgoing sum. -/
theorem arrayOutgoing_eq (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code)
    (states : Array ℕ) (index : ℕ → ℕ) (values : Array ℕ) (rs : List Rotation)
    (hindex : states[index (encode rs)]? = some (encode rs)) :
    PrefixAutomaton.arrayOutgoing (graph codes states index) values (index (encode rs)) =
      PrefixAutomaton.outgoing (dictionary codes)
        (PrefixAutomaton.arrayWeight (index ∘ encode) values) rs := by
  unfold PrefixAutomaton.arrayOutgoing graph
  rw [Array.getElem?_map, hindex]
  simp only [Option.map_some, Option.getD_some, successors_eq codes hcodes]
  exact PrefixAutomaton.rowWeight_indexedSuccessors (dictionary codes) (index ∘ encode) values rs

/-- Transfers per-key array checks to a decoded dictionary state, using key
round trips and source-index checks. Terminal initialization, terminal-step
domination, scaling, and the integer growth inequality are preserved exactly. -/
theorem certificate_of_checks (codes : Std.HashSet ℕ)
    (hcodes : ∀ code ∈ codes, encode (decode code) = code)
    (states : Array ℕ) (index : ℕ → ℕ) (values : Array ℕ) (terminalValues : ℕ → Array ℕ)
    (a b scale terminal : ℕ)
    (hcheck : ∀ code ∈ codes,
      states[index code]? = some code ∧
      1 ≤ (terminalValues 0)[index code]?.getD 0 ∧
      (∀ t : Fin terminal,
        PrefixAutomaton.arrayOutgoing (graph codes states index) (terminalValues t.val)
            (index code) ≤
          (terminalValues (t.val + 1))[index code]?.getD 0) ∧
      scale * (terminalValues terminal)[index code]?.getD 0 ≤ values[index code]?.getD 0 ∧
      b * PrefixAutomaton.arrayOutgoing (graph codes states index) values (index code) ≤
        a * values[index code]?.getD 0)
    (rs : List Rotation) (h : rs ∈ dictionary codes) :
    1 ≤ PrefixAutomaton.arrayWeight (index ∘ encode) (terminalValues 0) rs ∧
      (∀ t : Fin terminal,
        PrefixAutomaton.outgoing (dictionary codes)
            (PrefixAutomaton.arrayWeight (index ∘ encode) (terminalValues t.val)) rs ≤
          PrefixAutomaton.arrayWeight (index ∘ encode) (terminalValues (t.val + 1)) rs) ∧
      scale * PrefixAutomaton.arrayWeight (index ∘ encode) (terminalValues terminal) rs ≤
        PrefixAutomaton.arrayWeight (index ∘ encode) values rs ∧
      b * PrefixAutomaton.outgoing (dictionary codes)
          (PrefixAutomaton.arrayWeight (index ∘ encode) values) rs ≤
        a * PrefixAutomaton.arrayWeight (index ∘ encode) values rs := by
  have hcode := (mem_dictionary_iff codes hcodes rs).mp h
  obtain ⟨hindex, hbase, hstep, hterminal, hweight⟩ := hcheck (encode rs) hcode
  refine ⟨hbase, ?_, hterminal, ?_⟩
  · intro t
    rw [← arrayOutgoing_eq codes hcodes states index _ rs hindex]
    exact hstep t
  · rw [← arrayOutgoing_eq codes hcodes states index _ rs hindex]
    exact hweight

end EncodedPrefixAutomaton
end RubiksSnake

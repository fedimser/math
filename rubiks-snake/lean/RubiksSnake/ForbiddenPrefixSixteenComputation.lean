import RubiksSnake.PrefixAutomatonData
import Lean.Elab.Tactic.Decide

/-!
# The finite length-sixteen check

All states, transitions, and integer potentials are generated here. This
module deliberately avoids the geometric and real-analysis proof imports;
the next module proves that its checked arrays certify those semantics.
-/

namespace RubiksSnake
namespace ForbiddenPrefixSixteenUpper

/-- Generates the collision-word count and encoded proper-prefix dictionary
from even rotation lengths four through sixteen. -/
def generated : Nat × Std.HashSet Nat :=
  EncodedPrefixAutomaton.collisionPrefixes [4, 6, 8, 10, 12, 14, 16]

/-- Sentinel-encoded variable-length suffix states for the certificate, not
encodings of all valid sixteen-rotation formulas. -/
def codes : Std.HashSet Nat := generated.2

/-- Enumeration of stored prefix keys for graph indexing and statewise verification. -/
def states : List Nat := codes.toList

/-- The prefix-key enumeration as an array in the same order, enabling efficient
native traversal of all states. -/
def stateArray : Array Nat := states.toArray

/-- Associates each prefix key with its position in the state enumeration. -/
private def indices : Std.HashMap Nat Nat :=
  Std.HashMap.ofList states.zipIdx

/-- Returns a key's state index, defaulting to zero when absent; the certificate
checks the lookup at every stored key. -/
def index (code : Nat) : Nat := indices.getD code 0

/-- Encoded, locally valid rotation-labeled transition rows in `stateArray`
order; the automaton retains suffix states rather than complete histories. -/
def graph : Array (List Nat) :=
  EncodedPrefixAutomaton.graph codes stateArray index

/-- Candidate integer potential from 48 iterations of `floor(outgoing / 4)`
starting at `10^18`, to be certified by explicit inequalities rather than rounding. -/
def potential : Array Nat := PrefixAutomaton.scaledWeights graph 48

/-- Exact graph-path count vectors for zero through six terminal rotations,
starting with the all-ones vector. -/
private def terminals : Array (Array Nat) := PrefixAutomaton.terminalSequence graph 6

/-- Retrieves the terminal vector for horizon `t`, returning an empty array
outside the stored zero-to-six range. -/
def terminalVector (t : Nat) : Array Nat := terminals[t]?.getD #[]

/-- List-valued view of the potential, obtained by encoding a word and looking
up its state index; entries are weights, not formula counts. -/
def weight : List (Fin 4) → Nat :=
  PrefixAutomaton.arrayWeight (index ∘ EncodedPrefixAutomaton.encode) potential

/-- Local terminal path weight read through the encoded-word index, used to
absorb the final six rotations into a uniform counting prefactor. -/
def terminalWeight (t : Nat) : List (Fin 4) → Nat :=
  PrefixAutomaton.arrayWeight (index ∘ EncodedPrefixAutomaton.encode) (terminalVector t)

/-- Exact per-key obligations: canonical encoding, prefix closure, correct
indexing, six-step terminal domination at scale `2990000000000`, and the
integer outgoing bound with ratio `3661786723 / 1000000000`. -/
def stateCertificate (code : Nat) : Prop :=
  EncodedPrefixAutomaton.encode (EncodedPrefixAutomaton.decode code) = code ∧
  codes.contains
    (EncodedPrefixAutomaton.encode (EncodedPrefixAutomaton.decode code).dropLast) = true ∧
  stateArray[index code]? = some code ∧
  1 ≤ (terminalVector 0)[index code]?.getD 0 ∧
  (∀ t : Fin 6,
    PrefixAutomaton.arrayOutgoing graph (terminalVector t.val) (index code) ≤
      (terminalVector (t.val + 1))[index code]?.getD 0) ∧
  2990000000000 * (terminalVector 6)[index code]?.getD 0 ≤ potential[index code]?.getD 0 ∧
  1000000000 * PrefixAutomaton.arrayOutgoing graph potential (index code) ≤
    3661786723 * potential[index code]?.getD 0

/-- Decides one prefix-state certificate using finite array lookups and integer
comparisons, enabling the native exhaustive check. -/
instance (code : Nat) : Decidable (stateCertificate code) := by
  unfold stateCertificate
  infer_instance

/-- Native verification of all encoded state certificates, graph metadata,
empty-root membership, and the root potential. This certifies integer
inequalities rather than estimating a spectral radius. -/
private theorem array_checked :
    codes.contains 1 = true ∧
    (generated.1 = 4441614 ∧ states.length = 4748260 ∧
      graph.foldl (fun total row => total + row.length) 0 = 14133721) ∧
    stateArray.all (fun code => decide (stateCertificate code)) = true ∧
    weight [] = 21309021059784288 := by
  native_decide

/-- Exposes the native array verification as a certificate for every stored key,
with the root and graph metadata. The millions of checked states are prefix
automaton states, not a fixed-length enumeration of valid formulas. -/
theorem checked :
    codes.contains 1 = true ∧
    (generated.1 = 4441614 ∧ states.length = 4748260 ∧
      graph.foldl (fun total row => total + row.length) 0 = 14133721) ∧
    (∀ code ∈ states, stateCertificate code) ∧
    weight [] = 21309021059784288 := by
  obtain ⟨hzero, hcounts, hstates, hroot⟩ := array_checked
  refine ⟨hzero, hcounts, ?_, hroot⟩
  simpa only [stateArray, List.all_toArray, List.all_eq_true, decide_eq_true_eq] using hstates

end ForbiddenPrefixSixteenUpper
end RubiksSnake

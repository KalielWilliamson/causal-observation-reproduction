/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

abbrev Trace (Event : Type) := List Event

abbrev ObservationRegime (World Obs : Type) := World -> Obs

structure TransferInstance (World Policy Protocol : Type) where
  source : World
  target : World
  policy : Policy
  protocol : Protocol

def TraceEquivalent {World Obs : Type}
    (O : ObservationRegime World Obs)
    (x y : World) : Prop :=
  O x = O y

def ObservationClass {World Obs : Type}
    (O : ObservationRegime World Obs)
    (obs : Obs) : World -> Prop :=
  fun x => O x = obs

def SameObservationClass {World Obs : Type}
    (O : ObservationRegime World Obs)
    (x y : World) : Prop :=
  ∃ obs : Obs, ObservationClass O obs x ∧ ObservationClass O obs y

/-- A declared query can be recovered exactly from the policy-visible observation.

This is observation-relative query sufficiency, not Pearl/Shpitser causal
identification from an observational distribution. -/
def QuerySufficient {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) : Prop :=
  ∀ {x y : World}, TraceEquivalent O x y -> q x = q y

/-- Backward-compatible name for `QuerySufficient`.

New formal and manuscript-facing statements should use `QuerySufficient` to
avoid conflating recoverability under an observation map with causal
identification. -/
def Identifiable {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) : Prop :=
  QuerySufficient O q

theorem query_sufficient_iff_constant_on_trace_equivalence
    {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) :
    QuerySufficient O q ↔
      ∀ {x y : World}, TraceEquivalent O x y -> q x = q y := by
  rfl

def ConstantOnObservationClasses {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) : Prop :=
  ∀ obs : Obs, ∀ {x y : World},
    ObservationClass O obs x ->
    ObservationClass O obs y ->
    q x = q y

def Refines {World FineObs CoarseObs : Type}
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs) : Prop :=
  ∃ projection : FineObs -> CoarseObs,
    ∀ x : World, coarse x = projection (fine x)

def SimulatesCoarseDecision {World FineObs CoarseObs Decision : Type}
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (coarseDecision : CoarseObs -> Decision)
    (fineDecision : FineObs -> Decision) : Prop :=
  ∀ x : World, fineDecision (fine x) = coarseDecision (coarse x)

theorem trace_equivalence_impossibility
    {World Obs Decision : Type}
    (O : ObservationRegime World Obs)
    (decision : Obs -> Decision)
    {x y : World}
    (h : TraceEquivalent O x y) :
    decision (O x) = decision (O y) := by
  unfold TraceEquivalent at h
  rw [h]

theorem identifiable_iff_constant_on_trace_equivalence
    {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) :
    Identifiable O q ↔
      ∀ {x y : World}, TraceEquivalent O x y -> q x = q y := by
  rfl

theorem trace_equivalent_refl
    {World Obs : Type}
    (O : ObservationRegime World Obs)
    (x : World) :
    TraceEquivalent O x x := by
  rfl

theorem trace_equivalent_symm
    {World Obs : Type}
    {O : ObservationRegime World Obs}
    {x y : World}
    (h : TraceEquivalent O x y) :
    TraceEquivalent O y x := by
  exact h.symm

theorem trace_equivalent_trans
    {World Obs : Type}
    {O : ObservationRegime World Obs}
    {x y z : World}
    (hxy : TraceEquivalent O x y)
    (hyz : TraceEquivalent O y z) :
    TraceEquivalent O x z := by
  exact hxy.trans hyz

theorem trace_equivalent_iff_same_observation_class
    {World Obs : Type}
    (O : ObservationRegime World Obs)
    (x y : World) :
    TraceEquivalent O x y ↔ SameObservationClass O x y := by
  constructor
  · intro h
    exact ⟨O x, rfl, h.symm⟩
  · intro h
    rcases h with ⟨obs, hx, hy⟩
    unfold TraceEquivalent
    rw [hx, hy]

theorem identifiable_iff_constant_on_observation_classes
    {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target) :
    Identifiable O q ↔ ConstantOnObservationClasses O q := by
  constructor
  · intro hIdentifiable obs x y hx hy
    apply hIdentifiable
    unfold TraceEquivalent
    rw [hx, hy]
  · intro hConstant x y hTrace
    apply hConstant (O x)
    · rfl
    · exact hTrace.symm

theorem trace_equivalent_different_target_not_identifiable
    {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (q : World -> Target)
    {x y : World}
    (hTrace : TraceEquivalent O x y)
    (hTarget : q x ≠ q y) :
    ¬ Identifiable O q := by
  intro hIdentifiable
  exact hTarget (hIdentifiable hTrace)

theorem refinement_preserves_identifiability
    {World FineObs CoarseObs Target : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {q : World -> Target}
    (hRefines : Refines fine coarse)
    (hIdentifiable : Identifiable coarse q) :
    Identifiable fine q := by
  intro x y hFine
  rcases hRefines with ⟨projection, hProjection⟩
  apply hIdentifiable
  unfold TraceEquivalent at *
  rw [hProjection x, hProjection y, hFine]

theorem refinement_refl
    {World Obs : Type}
    (O : ObservationRegime World Obs) :
    Refines O O := by
  exact ⟨fun obs => obs, by intro x; rfl⟩

theorem refinement_trans
    {World FineObs MidObs CoarseObs : Type}
    {fine : ObservationRegime World FineObs}
    {mid : ObservationRegime World MidObs}
    {coarse : ObservationRegime World CoarseObs}
    (hFineMid : Refines fine mid)
    (hMidCoarse : Refines mid coarse) :
    Refines fine coarse := by
  rcases hFineMid with ⟨fineToMid, hFineProjection⟩
  rcases hMidCoarse with ⟨midToCoarse, hMidProjection⟩
  refine ⟨fun fineObs => midToCoarse (fineToMid fineObs), ?_⟩
  intro x
  rw [hMidProjection x, hFineProjection x]

theorem refinement_maps_fine_equivalence_to_coarse_equivalence
    {World FineObs CoarseObs : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {x y : World}
    (hRefines : Refines fine coarse)
    (hFine : TraceEquivalent fine x y) :
    TraceEquivalent coarse x y := by
  rcases hRefines with ⟨projection, hProjection⟩
  unfold TraceEquivalent at *
  rw [hProjection x, hProjection y, hFine]

theorem refinement_simulates_coarse_decision_rules
    {World FineObs CoarseObs Decision : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    (hRefines : Refines fine coarse)
    (coarseDecision : CoarseObs -> Decision) :
    ∃ fineDecision : FineObs -> Decision,
      SimulatesCoarseDecision fine coarse coarseDecision fineDecision := by
  rcases hRefines with ⟨projection, hProjection⟩
  refine ⟨fun obs => coarseDecision (projection obs), ?_⟩
  unfold SimulatesCoarseDecision
  intro x
  change coarseDecision (projection (fine x)) = coarseDecision (coarse x)
  rw [← hProjection x]

end Manuscript
end CausalObservationReproduction

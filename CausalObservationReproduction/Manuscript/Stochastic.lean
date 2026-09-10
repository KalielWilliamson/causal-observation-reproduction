import CausalObservationReproduction.Manuscript.Core

/-
Finite stochastic observation regimes for CausalObservationReproduction.

This file gives the deterministic observation-regime proof a small
Blackwell-style extension. It deliberately stays finite and rational-valued:
channels are explicit probability-weight tables over listed observations, and
garbling is represented by kernel composition. The file does not import mathlib
measure theory or attempt to prove the full Blackwell theorem.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Stochastic

structure FiniteDistribution (α : Type) where
  support : List α
  weight : α -> Rat
  nonnegative : ∀ value : α, 0 <= weight value
  normalized : (support.map weight).sum = 1
  zeroOutsideSupport : ∀ value : α, value ∉ support -> weight value = 0

theorem finite_distribution_weight_zero_outside_support
    {α : Type}
    (distribution : FiniteDistribution α)
    {value : α}
    (hOutside : value ∉ distribution.support) :
    distribution.weight value = 0 := by
  exact distribution.zeroOutsideSupport value hOutside

abbrev Channel (World Obs : Type) := World -> FiniteDistribution Obs

abbrev Kernel (α β : Type) := α -> FiniteDistribution β

structure RandomizedDecisionRule (Obs Decision : Type) where
  decisionSupport : List Decision
  weight : Obs -> Decision -> Rat
  nonnegative : ∀ obs decision, 0 <= weight obs decision
  normalized : ∀ obs, (decisionSupport.map (weight obs)).sum = 1
  zeroOutsideSupport :
    ∀ obs decision, decision ∉ decisionSupport -> weight obs decision = 0

theorem randomized_decision_weight_zero_outside_support
    {Obs Decision : Type}
    (rule : RandomizedDecisionRule Obs Decision)
    (obs : Obs)
    {decision : Decision}
    (hOutside : decision ∉ rule.decisionSupport) :
    rule.weight obs decision = 0 := by
  exact rule.zeroOutsideSupport obs decision hOutside

def pureDistribution {α : Type} [DecidableEq α] (value : α) :
    FiniteDistribution α where
  support := [value]
  weight := fun candidate => if candidate = value then 1 else 0
  nonnegative := by
    intro candidate
    by_cases h : candidate = value
    · simp [h]
      decide
    · simp [h]
  normalized := by
    simp
    rw [Rat.add_zero]
  zeroOutsideSupport := by
    intro candidate hOutside
    simp at hOutside
    simp [hOutside]

def deterministicChannel {World Obs : Type} [DecidableEq Obs]
    (O : World -> Obs) : Channel World Obs :=
  fun x => pureDistribution (O x)

def projectionKernel {FineObs CoarseObs : Type} [DecidableEq CoarseObs]
    (projection : FineObs -> CoarseObs) : Kernel FineObs CoarseObs :=
  fun fineObs => pureDistribution (projection fineObs)

def bindPointwiseWeight {World FineObs CoarseObs : Type}
    (fine : Channel World FineObs)
    (garbling : Kernel FineObs CoarseObs) : World -> CoarseObs -> Rat :=
  fun x coarseObs =>
    (fine x).support.map
      (fun fineObs => (fine x).weight fineObs * (garbling fineObs).weight coarseObs)
      |>.sum

def BlackwellRefinesBy {World FineObs CoarseObs : Type}
    (fine : Channel World FineObs)
    (coarse : Channel World CoarseObs)
    (garbling : Kernel FineObs CoarseObs) : Prop :=
  ∀ x coarseObs,
    (coarse x).weight coarseObs = bindPointwiseWeight fine garbling x coarseObs

def BlackwellRefines {World FineObs CoarseObs : Type}
    (fine : Channel World FineObs)
    (coarse : Channel World CoarseObs) : Prop :=
  ∃ garbling : Kernel FineObs CoarseObs,
    BlackwellRefinesBy fine coarse garbling

def pullbackDecisionWeight {FineObs CoarseObs Decision : Type}
    (coarseSupport : List CoarseObs)
    (garbling : Kernel FineObs CoarseObs)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision) :
    FineObs -> Decision -> Rat :=
  fun fineObs decision =>
    coarseSupport.map
      (fun coarseObs =>
        (garbling fineObs).weight coarseObs * coarseDecision.weight coarseObs decision)
      |>.sum

def inducedDecisionDistribution {World Obs Decision : Type}
    (observationSupport : World -> List Obs)
    (observationWeight : World -> Obs -> Rat)
    (decision : RandomizedDecisionRule Obs Decision) : World -> Decision -> Rat :=
  fun x decisionValue =>
    (observationSupport x).map
      (fun obs => observationWeight x obs * decision.weight obs decisionValue)
      |>.sum

def inducedCoarseDecisionDistribution {World Obs Decision : Type}
    (channel : Channel World Obs)
    (decision : RandomizedDecisionRule Obs Decision) : World -> Decision -> Rat :=
  inducedDecisionDistribution
    (fun x => (channel x).support)
    (fun x obs => (channel x).weight obs)
    decision

def inducedPulledBackDecisionDistribution
    {World FineObs CoarseObs Decision : Type}
    (coarseSupport : List CoarseObs)
    (fine : Channel World FineObs)
    (garbling : Kernel FineObs CoarseObs)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision) :
    World -> Decision -> Rat :=
  inducedDecisionDistribution
    (fun _ => coarseSupport)
    (bindPointwiseWeight fine garbling)
    coarseDecision

theorem deterministic_refinement_is_stochastic_garbling
    {World FineObs CoarseObs : Type}
    [DecidableEq FineObs] [DecidableEq CoarseObs]
    (fine : World -> FineObs)
    (coarse : World -> CoarseObs)
    (projection : FineObs -> CoarseObs)
    (hProjection : ∀ x : World, coarse x = projection (fine x)) :
    ∀ x coarseObs,
      bindPointwiseWeight
        (deterministicChannel fine)
        (projectionKernel projection) x coarseObs =
      (deterministicChannel coarse x).weight coarseObs := by
  intro x coarseObs
  unfold bindPointwiseWeight deterministicChannel projectionKernel pureDistribution
  simp [hProjection x]
  rw [Rat.add_zero]

theorem deterministic_refinement_blackwell_refines
    {World FineObs CoarseObs : Type}
    [DecidableEq FineObs] [DecidableEq CoarseObs]
    (fine : World -> FineObs)
    (coarse : World -> CoarseObs)
    (projection : FineObs -> CoarseObs)
    (hProjection : ∀ x : World, coarse x = projection (fine x)) :
    BlackwellRefines
      (deterministicChannel fine)
      (deterministicChannel coarse) := by
  refine ⟨projectionKernel projection, ?_⟩
  intro x coarseObs
  symm
  exact deterministic_refinement_is_stochastic_garbling fine coarse projection hProjection x coarseObs

theorem deterministic_refines_blackwell_refines
    {World FineObs CoarseObs : Type}
    [DecidableEq FineObs] [DecidableEq CoarseObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    (hRefines : Refines fine coarse) :
    BlackwellRefines
      (deterministicChannel fine)
      (deterministicChannel coarse) := by
  rcases hRefines with ⟨projection, hProjection⟩
  exact deterministic_refinement_blackwell_refines fine coarse projection hProjection

theorem blackwell_refinement_has_decision_pullback
    {World FineObs CoarseObs Decision : Type}
    {fine : Channel World FineObs}
    {coarse : Channel World CoarseObs}
    (coarseSupport : List CoarseObs)
    (hBlackwell : BlackwellRefines fine coarse)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision) :
    ∃ garbling : Kernel FineObs CoarseObs,
      BlackwellRefinesBy fine coarse garbling ∧
      ∃ fineDecisionWeight : FineObs -> Decision -> Rat,
        fineDecisionWeight = pullbackDecisionWeight coarseSupport garbling coarseDecision := by
  rcases hBlackwell with ⟨garbling, hRefinesBy⟩
  exact ⟨garbling, hRefinesBy,
    pullbackDecisionWeight coarseSupport garbling coarseDecision, rfl⟩

theorem blackwell_refinement_preserves_induced_decision_distribution
    {World FineObs CoarseObs Decision : Type}
    {fine : Channel World FineObs}
    {coarse : Channel World CoarseObs}
    (coarseSupport : List CoarseObs)
    (garbling : Kernel FineObs CoarseObs)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision)
    (hBlackwell : BlackwellRefinesBy fine coarse garbling) :
    inducedPulledBackDecisionDistribution
      coarseSupport fine garbling coarseDecision =
    inducedDecisionDistribution
      (fun _ => coarseSupport)
      (fun x coarseObs => (coarse x).weight coarseObs)
      coarseDecision := by
  funext x decision
  unfold inducedPulledBackDecisionDistribution inducedDecisionDistribution
  simp [hBlackwell x]

theorem blackwell_refinement_preserves_induced_decision_distribution_pointwise
    {World FineObs CoarseObs Decision : Type}
    {fine : Channel World FineObs}
    {coarse : Channel World CoarseObs}
    (coarseSupport : List CoarseObs)
    (garbling : Kernel FineObs CoarseObs)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision)
    (hBlackwell : BlackwellRefinesBy fine coarse garbling)
    (x : World)
    (decision : Decision) :
    inducedPulledBackDecisionDistribution
      coarseSupport fine garbling coarseDecision x decision =
    inducedDecisionDistribution
      (fun _ => coarseSupport)
      (fun x coarseObs => (coarse x).weight coarseObs)
      coarseDecision x decision := by
  rw [blackwell_refinement_preserves_induced_decision_distribution
    coarseSupport garbling coarseDecision hBlackwell]

theorem blackwell_refinement_preserves_risk_after_pullback
    {World FineObs CoarseObs Decision : Type}
    {fine : Channel World FineObs}
    {coarse : Channel World CoarseObs}
    (coarseSupport : List CoarseObs)
    (garbling : Kernel FineObs CoarseObs)
    (coarseDecision : RandomizedDecisionRule CoarseObs Decision)
    (risk : (World -> Decision -> Rat) -> Rat)
    (hBlackwell : BlackwellRefinesBy fine coarse garbling) :
    risk (inducedPulledBackDecisionDistribution
      coarseSupport fine garbling coarseDecision) =
    risk (inducedDecisionDistribution
      (fun _ => coarseSupport)
      (fun x coarseObs => (coarse x).weight coarseObs)
      coarseDecision) := by
  rw [blackwell_refinement_preserves_induced_decision_distribution
    coarseSupport garbling coarseDecision hBlackwell]

end Stochastic
end Manuscript
end CausalObservationReproduction

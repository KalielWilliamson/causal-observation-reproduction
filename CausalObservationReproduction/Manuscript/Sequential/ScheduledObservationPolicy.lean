import CausalObservationReproduction.Manuscript.Sequential.ObservabilityControl

/- A joint controller owns both environment and refinement actions at declared epochs. -/
namespace CausalObservationReproduction.Manuscript.Sequential

inductive RefinementOwnership where | policy | externalGate | oracle | forced deriving DecidableEq

structure ScheduledCostedObservationPolicy (History Refinement EnvironmentAction : Type) where
  eligible : Nat → Prop
  chooseRefinement : History → Nat → Option Refinement
  chooseEnvironment : History → Nat → EnvironmentAction
  ownership : RefinementOwnership

def PolicyOwnsEligibleRefinement {H R A} (p : ScheduledCostedObservationPolicy H R A) : Prop :=
  p.ownership = RefinementOwnership.policy

theorem policy_owned_refinement_is_not_external_gate {H R A}
    (p : ScheduledCostedObservationPolicy H R A)
    (h : PolicyOwnsEligibleRefinement p) : p.ownership ≠ RefinementOwnership.externalGate := by
  intro hExternal
  rw [h] at hExternal
  exact RefinementOwnership.noConfusion hExternal
end CausalObservationReproduction.Manuscript.Sequential

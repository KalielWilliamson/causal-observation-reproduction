import CausalObservationReproduction.Manuscript.Core

/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure FiniteDeterministicCausalModel (World Intervention Variable Value : Type) where
  structuralValue : World -> Variable -> Value
  intervene : World -> Intervention -> World

structure CausalAbstraction (World AbstractWorld : Type) where
  map : World -> AbstractWorld

def causalAbstractionObservationRegime {World AbstractWorld : Type}
    (abstraction : CausalAbstraction World AbstractWorld) :
    ObservationRegime World AbstractWorld :=
  abstraction.map

def StructuralAssignmentRespectingAbstraction
    {World Intervention Variable Value AbstractWorld : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld) : Prop :=
  forall x y : World, forall node : Variable,
    abstraction.map x = abstraction.map y ->
    model.structuralValue x node = model.structuralValue y node

def InterventionRespectingAbstraction
    {World Intervention Variable Value AbstractWorld : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld) : Prop :=
  forall x y : World, forall intervention : Intervention,
    abstraction.map x = abstraction.map y ->
    abstraction.map (model.intervene x intervention) =
      abstraction.map (model.intervene y intervention)

def InterventionalTarget
    {World Intervention Variable Value : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (intervention : Intervention)
    (node : Variable) : World -> Value :=
  fun world => model.structuralValue (model.intervene world intervention) node

theorem structural_assignment_respecting_abstraction_identifies_variable_query
    {World Intervention Variable Value AbstractWorld : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (node : Variable) :
    Identifiable
      (causalAbstractionObservationRegime abstraction)
      (fun world => model.structuralValue world node) := by
  intro x y hTrace
  exact hStructural x y node hTrace

theorem intervention_respecting_abstraction_identifies_intervened_observation_query
    {World Intervention Variable Value AbstractWorld : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (intervention : Intervention) :
    Identifiable
      (causalAbstractionObservationRegime abstraction)
      (fun world => abstraction.map (model.intervene world intervention)) := by
  intro x y hTrace
  exact hIntervention x y intervention hTrace

theorem causal_abstraction_identifies_interventional_target
    {World Intervention Variable Value AbstractWorld : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (intervention : Intervention)
    (node : Variable) :
    Identifiable
      (causalAbstractionObservationRegime abstraction)
      (InterventionalTarget model intervention node) := by
  intro x y hTrace
  unfold InterventionalTarget
  apply hStructural
  exact hIntervention x y intervention hTrace

end Manuscript
end CausalObservationReproduction

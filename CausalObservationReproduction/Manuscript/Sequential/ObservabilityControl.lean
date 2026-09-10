import CausalObservationReproduction.Manuscript.ObservabilityControl
import CausalObservationReproduction.Manuscript.Sequential.RefinementValue

/-
Finite active observability as a meta-decision problem.

Observation actions are represented separately from environment actions.  The
controller value includes acquisition cost plus optimal continuation value.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

structure SequentialMetaController (Info ObservationChoice EnvironmentAction : Type) where
  observationChoices : List ObservationChoice
  environmentActions : List EnvironmentAction
  observationCost : ObservationChoice -> Rat
  continuationValue : Info -> ObservationChoice -> Rat

def observationActionValue
    {Info ObservationChoice EnvironmentAction : Type}
    (controller :
      SequentialMetaController Info ObservationChoice EnvironmentAction)
    (info : Info)
    (choice : ObservationChoice) : Rat :=
  controller.continuationValue info choice - controller.observationCost choice

def ObservationActionOptimal
    {Info ObservationChoice EnvironmentAction : Type}
    (controller :
      SequentialMetaController Info ObservationChoice EnvironmentAction)
    (info : Info)
    (choice : ObservationChoice) : Prop :=
  forall other : ObservationChoice,
    observationActionValue controller info other <=
      observationActionValue controller info choice

def SequentialValueOfInformation
    (fineContinuationValue coarseContinuationValue : Rat) : Rat :=
  fineContinuationValue - coarseContinuationValue

structure ObservationActionDominanceCertificate
    {Info ObservationChoice EnvironmentAction : Type}
    (controller :
      SequentialMetaController Info ObservationChoice EnvironmentAction)
    (info : Info)
    (fineChoice coarseChoice : ObservationChoice) where
  voiExceedsCost :
    controller.observationCost fineChoice <
      SequentialValueOfInformation
        (controller.continuationValue info fineChoice)
        (controller.continuationValue info coarseChoice)
  dominatesOtherChoices :
    forall other : ObservationChoice,
      observationActionValue controller info other <=
        observationActionValue controller info fineChoice

theorem observation_action_is_optimal_when_voi_exceeds_cost
    {Info ObservationChoice EnvironmentAction : Type}
    {controller :
      SequentialMetaController Info ObservationChoice EnvironmentAction}
    {info : Info}
    {fineChoice coarseChoice : ObservationChoice}
    (certificate :
      ObservationActionDominanceCertificate
        controller info fineChoice coarseChoice) :
    ObservationActionOptimal controller info fineChoice := by
  intro other
  exact certificate.dominatesOtherChoices other

theorem sequential_value_of_information_is_fine_minus_coarse
    (fineContinuationValue coarseContinuationValue : Rat) :
    SequentialValueOfInformation fineContinuationValue coarseContinuationValue =
      fineContinuationValue - coarseContinuationValue := by
  rfl

end Sequential
end Manuscript
end CausalObservationReproduction

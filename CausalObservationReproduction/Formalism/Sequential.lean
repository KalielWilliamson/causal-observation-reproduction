import CausalObservationReproduction.Formalism.Core

/-!
Finite sequential skeleton for separating information and control actions.

The structures intentionally expose the objects that must be mapped to prior
work before a stronger theorem or empirical claim is pursued.
-/

namespace CausalObservationReproduction
namespace Formalism

structure ObservationHistory (Observation InformationAction ControlAction : Type) where
  initial : Observation
  steps : List (InformationAction × ControlAction × Observation)
  deriving Repr, DecidableEq

abbrev InformationPolicy (Observation InformationAction ControlAction : Type) :=
  Nat -> ObservationHistory Observation InformationAction ControlAction -> InformationAction

abbrev ControlPolicy (Observation InformationAction ControlAction : Type) :=
  Nat -> ObservationHistory Observation InformationAction ControlAction -> ControlAction

structure FiniteSequentialObservationControlProblem
    (World Observation InformationAction ControlAction : Type) where
  horizon : Nat
  informationBudget : Nat
  acquisitionCost : InformationAction -> Rat
  observe : Nat -> World -> InformationAction -> Observation
  informationAvailable : Nat -> Nat -> InformationAction -> Prop
  controlAvailable : Nat -> ControlAction -> Prop
  cumulativeReturn :
    InformationPolicy Observation InformationAction ControlAction ->
      ControlPolicy Observation InformationAction ControlAction -> Rat

def NetSequentialValue {World Observation InformationAction ControlAction : Type}
    (problem : FiniteSequentialObservationControlProblem
      World Observation InformationAction ControlAction)
    (informationPolicy : InformationPolicy Observation InformationAction ControlAction)
    (controlPolicy : ControlPolicy Observation InformationAction ControlAction)
    (totalAcquisitionCost : Rat) : Rat :=
  problem.cumulativeReturn informationPolicy controlPolicy - totalAcquisitionCost

end Formalism
end CausalObservationReproduction

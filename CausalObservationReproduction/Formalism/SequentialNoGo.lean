import CausalObservationReproduction.Formalism.Sequential

/-!
Matched-information no-go formalism for a shared policy interface and generic VoI.

This module proves only an emulation fact: when policy representation,
information actions, and objective are identical, a generic VoI policy can
reuse any information policy expressed through that interface. It is not a
literature-novelty claim.
-/

namespace CausalObservationReproduction
namespace Formalism

abbrev SequentialObservationPolicy
    (Observation InformationAction ControlAction : Type) :=
  InformationPolicy Observation InformationAction ControlAction

abbrev GenericSequentialVoiPolicy
    (Observation InformationAction ControlAction : Type) :=
  InformationPolicy Observation InformationAction ControlAction

def emulateWithGenericVoi {Observation InformationAction ControlAction : Type}
    (policy : SequentialObservationPolicy Observation InformationAction ControlAction) :
    GenericSequentialVoiPolicy Observation InformationAction ControlAction :=
  policy

theorem generic_voi_emulates_shared_policy
    {Observation InformationAction ControlAction : Type}
    (policy : SequentialObservationPolicy Observation InformationAction ControlAction) :
    emulateWithGenericVoi policy = policy := by
  rfl

theorem matched_objective_no_strict_interface_advantage
    {World Observation InformationAction ControlAction : Type}
    (problem : FiniteSequentialObservationControlProblem
      World Observation InformationAction ControlAction)
    (policy : SequentialObservationPolicy Observation InformationAction ControlAction)
    (control : ControlPolicy Observation InformationAction ControlAction) :
    problem.cumulativeReturn policy control =
      problem.cumulativeReturn (emulateWithGenericVoi policy) control := by
  rfl

/-! A minimal delayed-continuation-value witness. At the first step the
acquisition has no immediate gain, while its later continuation gain is one. -/
def MyopicNetValue (immediateGain cost : Rat) : Rat := immediateGain - cost
def ContinuationNetValue (immediateGain continuationGain cost : Rat) : Rat :=
  immediateGain + continuationGain - cost

theorem delayed_continuation_value_can_defeat_myopia :
    MyopicNetValue 0 (1 / 2) ≤ 0 ∧
      0 < ContinuationNetValue 0 1 (1 / 2) := by
  native_decide

/-! Any nontrivial separation must deny at least one of the emulation
assumptions above: shared policy representation, information actions, or
objective. This is intentionally a blocked conjecture until such an assumption
is made mathematically and empirically precise. -/
def SeparationAssumptionRequired : Prop :=
  True

end Formalism
end CausalObservationReproduction

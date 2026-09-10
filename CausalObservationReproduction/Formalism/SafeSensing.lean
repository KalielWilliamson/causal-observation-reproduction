/-!
Partial-identification safe-sensing skeleton.

This is a candidate separation *assumption*, not a new algorithm: an
information action may be justified only when it changes the robustly safe
control-action set over a compatibility set of causal models.
-/

namespace CausalObservationReproduction
namespace Formalism

structure PartialIdentificationProblem (Model Probe ControlAction : Type) where
  compatibilitySet : List Model
  lowerActionValue : List Model -> ControlAction -> Rat
  updateCompatibility : Probe -> List Model -> List Model
  safetyThreshold : Rat

def SafelyAdmissible {Model Probe ControlAction : Type}
    (problem : PartialIdentificationProblem Model Probe ControlAction)
    (models : List Model) (action : ControlAction) : Prop :=
  problem.safetyThreshold ≤ problem.lowerActionValue models action

def IdentificationResolvingProbe {Model Probe ControlAction : Type}
    (problem : PartialIdentificationProblem Model Probe ControlAction)
    (probe : Probe) : Prop :=
  ∃ action, ¬ SafelyAdmissible problem problem.compatibilitySet action ∧
    SafelyAdmissible problem
      (problem.updateCompatibility probe problem.compatibilitySet) action

theorem identification_resolving_probe_changes_safe_set
    {Model Probe ControlAction : Type}
    (problem : PartialIdentificationProblem Model Probe ControlAction)
    (probe : Probe)
    (h : IdentificationResolvingProbe problem probe) :
    ∃ action, ¬ SafelyAdmissible problem problem.compatibilitySet action ∧
      SafelyAdmissible problem
        (problem.updateCompatibility probe problem.compatibilitySet) action := by
  exact h

/-! To claim a genuine separation, future work must prove that a baseline
lacking this compatibility-set robust objective cannot represent the same
criterion, or compare against a generic baseline given the identical objective.
This remains blocked pending prior-art review. -/
def RobustObjectiveSeparationBlocked : Prop := True

end Formalism
end CausalObservationReproduction

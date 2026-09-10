import CausalObservationReproduction.Manuscript.ProtocolGates

/-
Formal gate for a preregistered embodied COO program.

The theorem is deliberately about the contract, rather than asserting empirical
robot outcomes.  Python validates the declared fields and exports them through
MLflow/NSHE; Isaac Sim remains the external evidence-producing system.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure OutcomeBlindSelectionGate where
  outcomeBlind : Prop
  semanticMappingDeclared : Prop
  interventionDeclared : Prop
  holdoutMechanismDeclared : Prop
  comparatorParityDeclared : Prop
  registeredHypothesesDeclared : Prop
  mlflowBackedEvidenceDeclared : Prop

def OutcomeBlindEmbodiedSelectionReady (gate : OutcomeBlindSelectionGate) : Prop :=
  gate.outcomeBlind ∧
    gate.semanticMappingDeclared ∧
    gate.interventionDeclared ∧
    gate.holdoutMechanismDeclared ∧
    gate.comparatorParityDeclared ∧
    gate.registeredHypothesesDeclared ∧
    gate.mlflowBackedEvidenceDeclared

theorem outcome_blind_selection_requires_semantic_and_holdout_gates
    {gate : OutcomeBlindSelectionGate}
    (hReady : OutcomeBlindEmbodiedSelectionReady gate) :
    gate.outcomeBlind ∧ gate.semanticMappingDeclared ∧ gate.holdoutMechanismDeclared := by
  exact ⟨hReady.left, hReady.right.left, hReady.right.right.right.left⟩

theorem outcome_blind_selection_requires_registered_hypotheses
    {gate : OutcomeBlindSelectionGate}
    (hReady : OutcomeBlindEmbodiedSelectionReady gate) :
    gate.registeredHypothesesDeclared ∧ gate.mlflowBackedEvidenceDeclared := by
  exact ⟨hReady.right.right.right.right.right.left, hReady.right.right.right.right.right.right⟩

/- A learned abstraction may inform the COO decision only through a calibrated,
non-privileged interface.  The fallback branch is intentionally explicit: a
failed calibration certificate is not permission to deploy a learned gate. -/
structure LearnedCausalAbstractionGate where
  taskRelevantVocabularyDeclared : Prop
  structuralAdequacyMeasured : Prop
  decisionAdequacyMeasured : Prop
  frozenMechanismTransferDeclared : Prop
  misspecificationBoundaryDeclared : Prop
  deploymentInputsNonPrivileged : Prop
  privilegedLabelsEvaluationOnly : Prop
  calibrationVerified : Prop
  declaredModelControlDeclared : Prop
  safeFallbackSelected : Prop

def LearnedCausalAbstractionReady (gate : LearnedCausalAbstractionGate) : Prop :=
  gate.taskRelevantVocabularyDeclared ∧
    gate.structuralAdequacyMeasured ∧
    gate.decisionAdequacyMeasured ∧
    gate.frozenMechanismTransferDeclared ∧
    gate.misspecificationBoundaryDeclared ∧
    gate.deploymentInputsNonPrivileged ∧
    gate.privilegedLabelsEvaluationOnly ∧
    gate.calibrationVerified ∧
    gate.declaredModelControlDeclared

def LearnedCausalAbstractionDeploymentSafe (gate : LearnedCausalAbstractionGate) : Prop :=
  LearnedCausalAbstractionReady gate ∨
    (¬ gate.calibrationVerified ∧ gate.safeFallbackSelected)

theorem learned_causal_abstraction_requires_calibrated_nonprivileged_interface
    {gate : LearnedCausalAbstractionGate}
    (hReady : LearnedCausalAbstractionReady gate) :
    gate.calibrationVerified ∧ gate.deploymentInputsNonPrivileged ∧
      gate.privilegedLabelsEvaluationOnly ∧ gate.declaredModelControlDeclared := by
  exact ⟨hReady.right.right.right.right.right.right.right.left,
    hReady.right.right.right.right.right.left,
    hReady.right.right.right.right.right.right.left,
    hReady.right.right.right.right.right.right.right.right⟩

theorem uncalibrated_learned_causal_abstraction_requires_safe_fallback
    {gate : LearnedCausalAbstractionGate}
    (hSafe : LearnedCausalAbstractionDeploymentSafe gate)
    (hUncalibrated : ¬ gate.calibrationVerified) :
    gate.safeFallbackSelected := by
  rcases hSafe with hReady | hFallback
  · exact False.elim (hUncalibrated hReady.right.right.right.right.right.right.right.left)
  · exact hFallback.right

end Manuscript
end CausalObservationReproduction

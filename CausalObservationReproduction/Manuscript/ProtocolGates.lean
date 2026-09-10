import CausalObservationReproduction.Manuscript.ClaimSupport

/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure ExperimentProtocolGate where
  observationRegimeDeclared : Prop
  claimTargetDeclared : Prop
  metricDeclared : Prop
  comparisonDeclared : Prop
  evidenceRoleDeclared : Prop
  analysisContractDeclared : Prop

structure ExternalEvidenceFacts where
  artifactExists : Prop
  mlflowRunComplete : Prop
  durableArtifactRegistered : Prop

def ProtocolFieldsDeclared (gate : ExperimentProtocolGate) : Prop :=
  gate.observationRegimeDeclared ∧
    gate.claimTargetDeclared ∧
    gate.metricDeclared ∧
    gate.comparisonDeclared ∧
    gate.evidenceRoleDeclared ∧
    gate.analysisContractDeclared

def ExternalEvidenceReady (facts : ExternalEvidenceFacts) : Prop :=
  facts.artifactExists ∧ facts.mlflowRunComplete ∧ facts.durableArtifactRegistered

def CentralEmpiricalSupport {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (spec : ClaimSupportSpec World Target)
    (gate : ExperimentProtocolGate)
    (facts : ExternalEvidenceFacts) : Prop :=
  FormalClaimSupported O spec ∧
    ProtocolFieldsDeclared gate ∧
    ExternalEvidenceReady facts

theorem central_empirical_support_requires_formal_claim_support
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    {gate : ExperimentProtocolGate}
    {facts : ExternalEvidenceFacts}
    (hSupport : CentralEmpiricalSupport O spec gate facts) :
    FormalClaimSupported O spec := by
  exact hSupport.left

theorem central_empirical_support_requires_protocol_fields
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    {gate : ExperimentProtocolGate}
    {facts : ExternalEvidenceFacts}
    (hSupport : CentralEmpiricalSupport O spec gate facts) :
    ProtocolFieldsDeclared gate := by
  exact hSupport.right.left

theorem central_empirical_support_requires_external_evidence_facts
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    {gate : ExperimentProtocolGate}
    {facts : ExternalEvidenceFacts}
    (hSupport : CentralEmpiricalSupport O spec gate facts) :
    ExternalEvidenceReady facts := by
  exact hSupport.right.right

end Manuscript
end CausalObservationReproduction

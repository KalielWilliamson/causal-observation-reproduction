import CausalObservationReproduction.Manuscript.Core

/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure ClaimTarget (World Target : Type) where
  query : World -> Target

structure EvidenceRole where
  name : String

structure NamedComparison (World : Type) where
  name : String
  baseline : World
  challenger : World

structure ClaimSupportSpec (World Target : Type) where
  target : ClaimTarget World Target
  roleAccepted : EvidenceRole -> Prop
  comparisonAccepted : NamedComparison World -> Prop

def FormalClaimSupported {World Obs Target : Type}
    (O : ObservationRegime World Obs)
    (spec : ClaimSupportSpec World Target) : Prop :=
  Identifiable O spec.target.query ∧
    (∀ role : EvidenceRole, spec.roleAccepted role) ∧
    (∀ comparison : NamedComparison World, spec.comparisonAccepted comparison)

theorem formal_claim_support_requires_identifiable_target
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    (hSupport : FormalClaimSupported O spec) :
    Identifiable O spec.target.query := by
  exact hSupport.left

theorem formal_claim_support_keeps_evidence_roles
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    (hSupport : FormalClaimSupported O spec) :
    ∀ role : EvidenceRole, spec.roleAccepted role := by
  exact hSupport.right.left

theorem formal_claim_support_keeps_named_comparisons
    {World Obs Target : Type}
    {O : ObservationRegime World Obs}
    {spec : ClaimSupportSpec World Target}
    (hSupport : FormalClaimSupported O spec) :
    ∀ comparison : NamedComparison World, spec.comparisonAccepted comparison := by
  exact hSupport.right.right

theorem refinement_preserves_formal_claim_support_when_requirements_unchanged
    {World FineObs CoarseObs Target : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {fineSpec coarseSpec : ClaimSupportSpec World Target}
    (hRefines : Refines fine coarse)
    (hSameTarget : fineSpec.target.query = coarseSpec.target.query)
    (hSameEvidence :
      ∀ role : EvidenceRole,
        fineSpec.roleAccepted role ↔ coarseSpec.roleAccepted role)
    (hSameComparisons :
      ∀ comparison : NamedComparison World,
        fineSpec.comparisonAccepted comparison ↔
          coarseSpec.comparisonAccepted comparison)
    (hSupport : FormalClaimSupported coarse coarseSpec) :
    FormalClaimSupported fine fineSpec := by
  constructor
  · rw [hSameTarget]
    exact refinement_preserves_identifiability hRefines hSupport.left
  · constructor
    · intro role
      exact (hSameEvidence role).mpr (hSupport.right.left role)
    · intro comparison
      exact (hSameComparisons comparison).mpr (hSupport.right.right comparison)

end Manuscript
end CausalObservationReproduction

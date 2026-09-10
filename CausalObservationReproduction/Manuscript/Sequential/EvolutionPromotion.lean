import CausalObservationReproduction.Manuscript.Sequential.VariationalBridge

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

/-! Runtime promotion contract for constrained evolutionary controllers. -/
structure EvolutionPromotionAudit where
  frozenImprovement : Prop
  boundaryActionsAudited : Prop
  frozenLabelsSeparated : Prop

def EvolutionPromotionEligible (audit : EvolutionPromotionAudit) : Prop :=
  audit.frozenImprovement ∧ audit.boundaryActionsAudited ∧ audit.frozenLabelsSeparated

theorem promoted_evolutionary_controller_has_audited_boundary_actions
    (audit : EvolutionPromotionAudit) (eligible : EvolutionPromotionEligible audit) :
    audit.boundaryActionsAudited := by exact eligible.2.1

theorem promoted_evolutionary_controller_uses_separated_frozen_labels
    (audit : EvolutionPromotionAudit) (eligible : EvolutionPromotionEligible audit) :
    audit.frozenLabelsSeparated := by exact eligible.2.2

theorem promoted_evolutionary_controller_improves_on_frozen_evaluation
    (audit : EvolutionPromotionAudit) (eligible : EvolutionPromotionEligible audit) :
    audit.frozenImprovement := by exact eligible.1

end Sequential
end Manuscript
end CausalObservationReproduction

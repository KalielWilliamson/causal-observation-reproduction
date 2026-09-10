import CausalObservationReproduction.Manuscript.ObservationKernel

/-! A hierarchy is a chain of public observation maps.  Each refinement
factors through its parent projection and carries a componentwise resource
certificate.  Router implementation is intentionally outside this theorem:
only validated declared requests cross the agent boundary. -/

namespace CausalObservationReproduction
namespace Manuscript

structure HierarchicalObservationMap (Raw Coarse Fine : Type) where
  coarse : Raw -> Coarse
  refine : Raw -> Fine
  forget : Fine -> Coarse
  refinementFactors : forall raw, forget (refine raw) = coarse raw

theorem hierarchical_refinement_preserves_parent_projection
    {Raw Coarse Fine : Type}
    (hierarchy : HierarchicalObservationMap Raw Coarse Fine)
    (raw : Raw) :
    hierarchy.forget (hierarchy.refine raw) = hierarchy.coarse raw := by
  exact hierarchy.refinementFactors raw

structure HierarchicalCostMap where
  contextTokens : Nat
  contextBudgetTokens : Nat
  sensingCost : ResourceVector
  budget : ResourceVector
  contextAdmissible : contextTokens ≤ contextBudgetTokens
  componentwiseAdmissible : sensingCost.withinBudget budget

structure AcceptedHierarchicalObservationRequest (costMap : HierarchicalCostMap) where
  contextWithinBudget : costMap.contextTokens ≤ costMap.contextBudgetTokens
  sensingWithinBudget : costMap.sensingCost.withinBudget costMap.budget

theorem hierarchical_accepted_request_is_componentwise_budgeted
    {costMap : HierarchicalCostMap}
    (request : AcceptedHierarchicalObservationRequest costMap) :
    costMap.sensingCost.withinBudget costMap.budget := by
  exact request.sensingWithinBudget

theorem hierarchical_accepted_request_is_context_budgeted
    {costMap : HierarchicalCostMap}
    (request : AcceptedHierarchicalObservationRequest costMap) :
    costMap.contextTokens ≤ costMap.contextBudgetTokens := by
  exact request.contextWithinBudget

theorem hierarchical_cost_map_declaration_is_componentwise_budgeted
    (costMap : HierarchicalCostMap) :
    costMap.sensingCost.withinBudget costMap.budget := by
  exact costMap.componentwiseAdmissible

theorem hierarchical_cost_map_declaration_is_context_budgeted
    (costMap : HierarchicalCostMap) :
    costMap.contextTokens ≤ costMap.contextBudgetTokens := by
  exact costMap.contextAdmissible

end Manuscript
end CausalObservationReproduction

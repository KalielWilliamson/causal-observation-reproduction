/-!
Cost-sensitive refinement-value definitions.

These definitions make the inherited one-step threshold explicit as a formal
foundation.  They do not establish a novel sequential decision rule.
-/

namespace CausalObservationReproduction
namespace Formalism

def RefinementGain (fineValue coarseValue : Rat) : Rat := fineValue - coarseValue

def NetRefinementValue (fineValue coarseValue acquisitionCost : Rat) : Rat :=
  RefinementGain fineValue coarseValue - acquisitionCost

def CostJustifiedRefinement
    (fineValue coarseValue acquisitionCost : Rat) : Prop :=
  acquisitionCost < RefinementGain fineValue coarseValue

theorem cost_justified_refinement_iff_gain_exceeds_cost
    (fineValue coarseValue acquisitionCost : Rat) :
    CostJustifiedRefinement fineValue coarseValue acquisitionCost <->
      acquisitionCost < RefinementGain fineValue coarseValue := by
  rfl

end Formalism
end CausalObservationReproduction

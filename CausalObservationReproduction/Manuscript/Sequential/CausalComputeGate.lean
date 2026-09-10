import Init.Data.Order.Lemmas
import CausalObservationReproduction.Manuscript.Sequential.RefinementValue

/-
Cost-sensitive causal-computation deployment.

The theorem is deliberately conditional: calibration and transfer are empirical
properties of a learned predictor, while Lean verifies the decision rule once a
sound lower bound has been supplied.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

inductive CausalComputeOption where
  | direct
  | can
  | coo
  | canAndCoo
  | refinement (tier : Nat)
  deriving Repr, DecidableEq

def NetOptionAdvantage (optionValue directValue totalCost : Rat) : Rat :=
  optionValue - directValue - totalCost

def LowerBoundSound (lowerBound trueAdvantage : Rat) : Prop :=
  forall threshold : Rat, threshold < lowerBound -> threshold < trueAdvantage

def ConservativeDeployment (lowerBound : Rat) : Prop :=
  0 < lowerBound

theorem conservative_deployment_has_positive_true_advantage
    {lowerBound trueAdvantage : Rat}
    (hSound : LowerBoundSound lowerBound trueAdvantage)
    (hDeploy : ConservativeDeployment lowerBound) :
    0 < trueAdvantage := by
  change 0 < lowerBound at hDeploy
  exact hSound 0 hDeploy

theorem conservative_deployment_has_nonnegative_true_advantage
    {lowerBound trueAdvantage : Rat}
    (hSound : LowerBoundSound lowerBound trueAdvantage)
    (hDeploy : ConservativeDeployment lowerBound) :
    0 <= trueAdvantage := by
  exact Rat.le_of_lt (conservative_deployment_has_positive_true_advantage hSound hDeploy)

theorem net_option_advantage_is_value_difference_minus_cost
    (optionValue directValue totalCost : Rat) :
    NetOptionAdvantage optionValue directValue totalCost =
      optionValue - directValue - totalCost := by
  rfl

end Sequential
end Manuscript
end CausalObservationReproduction

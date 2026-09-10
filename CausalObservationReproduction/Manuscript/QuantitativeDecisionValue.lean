import CausalObservationReproduction.Manuscript.DecisionValue
import CausalObservationReproduction.Manuscript.Stochastic

/-
Quantitative finite value layer for causal observability.

This file keeps the same finite/rational scope as the rest of the Lean
formalism. It turns the local quotient obstruction into a bounded expected
value-of-observation statement: a refinement is cost-justified when its
finite-support expected utility gain exceeds its observation cost.
-/

namespace CausalObservationReproduction
namespace Manuscript

abbrev FiniteWorldDistribution (World : Type) :=
  Stochastic.FiniteDistribution World

def ExpectedDecisionValue {World Obs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action) : Rat :=
  distribution.support.map
    (fun world => distribution.weight world * problem.utility world (rule (O world)))
    |>.sum

def ExpectedRefinementGain {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action) : Rat :=
  ExpectedDecisionValue distribution fine problem fineRule -
    ExpectedDecisionValue distribution coarse problem coarseRule

def NetRefinementValue {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost : Rat) : Rat :=
  ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule -
    observationCost

def RefinementFreeEnergy {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost ambiguityPenalty : Rat) : Rat :=
  ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule -
    observationCost - ambiguityPenalty

def PositiveRefinementFreeEnergy {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost ambiguityPenalty : Rat) : Prop :=
  observationCost + ambiguityPenalty <
    ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule

def CostJustifiedRefinement {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost : Rat) : Prop :=
  observationCost <
    ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule

def ExpectedValueDominates {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action) : Prop :=
  ExpectedDecisionValue distribution coarse problem coarseRule <
    ExpectedDecisionValue distribution fine problem fineRule

structure QuantitativeRefinementCertificate
    (World FineObs CoarseObs Action : Type)
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  refines : Refines fine coarse
  fineRule : DecisionRule FineObs Action
  coarseRule : DecisionRule CoarseObs Action
  observationCost : Rat
  expectedValueDominates :
    ExpectedValueDominates distribution fine coarse problem fineRule coarseRule
  costJustified :
    CostJustifiedRefinement distribution fine coarse problem fineRule coarseRule observationCost

structure QuantitativeDecisionRelevantRegionCertificate
    (World FineObs CoarseObs Action : Type)
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  regionWitness :
    DecisionRelevantRegionWitness World FineObs CoarseObs Action fine coarse problem
  quantitative :
    QuantitativeRefinementCertificate
      World FineObs CoarseObs Action distribution fine coarse problem

structure RefinementFreeEnergyCertificate
    (World FineObs CoarseObs Action : Type)
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  regionCertificate :
    QuantitativeDecisionRelevantRegionCertificate
      World FineObs CoarseObs Action distribution fine coarse problem
  ambiguityPenalty : Rat
  positiveFreeEnergy :
    PositiveRefinementFreeEnergy
      distribution
      fine
      coarse
      problem
      regionCertificate.quantitative.fineRule
      regionCertificate.quantitative.coarseRule
      regionCertificate.quantitative.observationCost
      ambiguityPenalty

theorem expected_decision_value_is_finite_weighted_sum
    {World Obs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action) :
    ExpectedDecisionValue distribution O problem rule =
      (distribution.support.map
        (fun world => distribution.weight world * problem.utility world (rule (O world)))
        |>.sum) := by
  rfl

theorem expected_refinement_gain_is_fine_minus_coarse_value
    {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action) :
    ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule =
      ExpectedDecisionValue distribution fine problem fineRule -
        ExpectedDecisionValue distribution coarse problem coarseRule := by
  rfl

theorem net_refinement_value_is_expected_gain_minus_cost
    {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost : Rat) :
    NetRefinementValue distribution fine coarse problem fineRule coarseRule observationCost =
      ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule -
        observationCost := by
  rfl

theorem refinement_free_energy_is_expected_gain_minus_cost_and_penalty
    {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost ambiguityPenalty : Rat) :
    RefinementFreeEnergy
        distribution fine coarse problem fineRule coarseRule observationCost ambiguityPenalty =
      ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule -
        observationCost - ambiguityPenalty := by
  rfl

theorem positive_refinement_free_energy_is_gain_above_cost_and_penalty
    {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost ambiguityPenalty : Rat) :
    PositiveRefinementFreeEnergy
        distribution fine coarse problem fineRule coarseRule observationCost ambiguityPenalty <->
      observationCost + ambiguityPenalty <
        ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule := by
  rfl

theorem cost_justified_refinement_is_expected_gain_threshold
    {World FineObs CoarseObs Action : Type}
    (distribution : FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (observationCost : Rat) :
    CostJustifiedRefinement distribution fine coarse problem fineRule coarseRule observationCost ↔
      observationCost <
        ExpectedRefinementGain distribution fine coarse problem fineRule coarseRule := by
  rfl

theorem quantitative_certificate_proves_expected_value_dominance
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeRefinementCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    ExpectedValueDominates
      distribution fine coarse problem certificate.fineRule certificate.coarseRule := by
  exact certificate.expectedValueDominates

theorem quantitative_certificate_proves_cost_justified_refinement
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeRefinementCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    CostJustifiedRefinement
      distribution
      fine
      coarse
      problem
      certificate.fineRule
      certificate.coarseRule
      certificate.observationCost := by
  exact certificate.costJustified

theorem quantitative_certificate_keeps_refinement_witness
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeRefinementCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    Refines fine coarse := by
  exact certificate.refines

def quantitative_region_certificate_has_region_witness
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeDecisionRelevantRegionCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    DecisionRelevantRegionWitness World FineObs CoarseObs Action fine coarse problem :=
  certificate.regionWitness

theorem quantitative_region_certificate_proves_cost_justified_refinement
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeDecisionRelevantRegionCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    CostJustifiedRefinement
      distribution
      fine
      coarse
      problem
      certificate.quantitative.fineRule
      certificate.quantitative.coarseRule
      certificate.quantitative.observationCost := by
  exact certificate.quantitative.costJustified

theorem quantitative_region_certificate_proves_expected_value_dominance
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : QuantitativeDecisionRelevantRegionCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    ExpectedValueDominates
      distribution
      fine
      coarse
      problem
      certificate.quantitative.fineRule
      certificate.quantitative.coarseRule := by
  exact certificate.quantitative.expectedValueDominates

theorem refinement_free_energy_certificate_proves_positive_free_energy
    {World FineObs CoarseObs Action : Type}
    {distribution : FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (certificate : RefinementFreeEnergyCertificate
      World FineObs CoarseObs Action distribution fine coarse problem) :
    PositiveRefinementFreeEnergy
      distribution
      fine
      coarse
      problem
      certificate.regionCertificate.quantitative.fineRule
      certificate.regionCertificate.quantitative.coarseRule
      certificate.regionCertificate.quantitative.observationCost
      certificate.ambiguityPenalty := by
  exact certificate.positiveFreeEnergy

end Manuscript
end CausalObservationReproduction

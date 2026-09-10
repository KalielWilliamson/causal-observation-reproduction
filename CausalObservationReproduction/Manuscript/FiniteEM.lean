import CausalObservationReproduction.Manuscript.QuantitativeDecisionValue

/-
Finite EM-style latent structure layer for causal observability.

This module gives the paper a bounded expectation-maximization bridge without
formalizing logs, convexity, or asymptotic consistency. Latent causal/POSCM
classes, traces, and responsibilities are finite objects; EM improvement is
represented as a checkable certificate over a finite expected complete-data
score.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure FiniteLatentTraceDataset (Trace Latent : Type) where
  traces : List Trace
  latentClasses : List Latent

structure ResponsibilityTable (Trace Latent : Type) where
  weight : Trace -> Latent -> Rat
  nonnegative : forall trace latent, 0 <= weight trace latent
  normalized :
    forall trace : Trace, forall latentClasses : List Latent,
      (latentClasses.map (weight trace)).sum = 1

structure FiniteEMModel (Trace Latent Parameter : Type) where
  completeDataScore : Parameter -> Trace -> Latent -> Rat

def ExpectedCompleteDataScore {Trace Latent Parameter : Type}
    (dataset : FiniteLatentTraceDataset Trace Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (model : FiniteEMModel Trace Latent Parameter)
    (parameter : Parameter) : Rat :=
  dataset.traces.map
    (fun trace =>
      dataset.latentClasses.map
        (fun latent =>
          responsibilities.weight trace latent *
            model.completeDataScore parameter trace latent)
        |>.sum)
    |>.sum

def ResponsibilityPosteriorValue {Trace Latent World Obs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action)
    (trace : Trace) : Rat :=
  latentClasses.map
    (fun latent =>
      responsibilities.weight trace latent *
        ExpectedDecisionValue (latentWorldDistribution latent) O problem rule)
    |>.sum

def PosteriorExpectedRefinementGain {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace) : Rat :=
  ResponsibilityPosteriorValue
      latentClasses responsibilities latentWorldDistribution fine problem fineRule trace -
    ResponsibilityPosteriorValue
      latentClasses responsibilities latentWorldDistribution coarse problem coarseRule trace

def PosteriorCostJustifiedRefinement
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (observationCost : Rat) : Prop :=
  observationCost <
    PosteriorExpectedRefinementGain
      latentClasses
      responsibilities
      latentWorldDistribution
      fine
      coarse
      problem
      fineRule
      coarseRule
      trace

structure FiniteEMStepCertificate (Trace Latent Parameter : Type)
    (dataset : FiniteLatentTraceDataset Trace Latent)
    (model : FiniteEMModel Trace Latent Parameter) where
  oldResponsibilities : ResponsibilityTable Trace Latent
  newResponsibilities : ResponsibilityTable Trace Latent
  oldParameter : Parameter
  newParameter : Parameter
  expectedCompleteDataScoreImproves :
    ExpectedCompleteDataScore dataset oldResponsibilities model oldParameter <=
      ExpectedCompleteDataScore dataset newResponsibilities model newParameter

theorem expected_complete_data_score_is_finite_double_sum
    {Trace Latent Parameter : Type}
    (dataset : FiniteLatentTraceDataset Trace Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (model : FiniteEMModel Trace Latent Parameter)
    (parameter : Parameter) :
    ExpectedCompleteDataScore dataset responsibilities model parameter =
      (dataset.traces.map
        (fun trace =>
          dataset.latentClasses.map
            (fun latent =>
              responsibilities.weight trace latent *
                model.completeDataScore parameter trace latent)
            |>.sum)
        |>.sum) := by
  rfl

theorem responsibility_posterior_value_is_weighted_value_sum
    {Trace Latent World Obs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action)
    (trace : Trace) :
    ResponsibilityPosteriorValue
      latentClasses responsibilities latentWorldDistribution O problem rule trace =
      (latentClasses.map
        (fun latent =>
          responsibilities.weight trace latent *
            ExpectedDecisionValue (latentWorldDistribution latent) O problem rule)
        |>.sum) := by
  rfl

theorem posterior_expected_refinement_gain_is_fine_minus_coarse_posterior_value
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace) :
    PosteriorExpectedRefinementGain
      latentClasses responsibilities latentWorldDistribution fine coarse problem fineRule coarseRule trace =
      ResponsibilityPosteriorValue
        latentClasses responsibilities latentWorldDistribution fine problem fineRule trace -
        ResponsibilityPosteriorValue
          latentClasses responsibilities latentWorldDistribution coarse problem coarseRule trace := by
  rfl

theorem posterior_cost_justified_refinement_is_gain_threshold
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (observationCost : Rat) :
    PosteriorCostJustifiedRefinement
      latentClasses responsibilities latentWorldDistribution fine coarse problem fineRule coarseRule
      trace observationCost <->
      observationCost <
        PosteriorExpectedRefinementGain
          latentClasses responsibilities latentWorldDistribution fine coarse problem fineRule coarseRule trace := by
  rfl

theorem finite_em_certificate_proves_score_improvement
    {Trace Latent Parameter : Type}
    {dataset : FiniteLatentTraceDataset Trace Latent}
    {model : FiniteEMModel Trace Latent Parameter}
    (certificate : FiniteEMStepCertificate Trace Latent Parameter dataset model) :
    ExpectedCompleteDataScore dataset certificate.oldResponsibilities model certificate.oldParameter <=
      ExpectedCompleteDataScore dataset certificate.newResponsibilities model certificate.newParameter := by
  exact certificate.expectedCompleteDataScoreImproves

end Manuscript
end CausalObservationReproduction

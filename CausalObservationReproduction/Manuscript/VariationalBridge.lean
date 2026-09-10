import CausalObservationReproduction.Manuscript.FiniteEM

/-
Variational/amortized bridge for causal observability.

This module connects the finite latent responsibility semantics to the practical
ML setting where a learned encoder produces an approximate posterior over latent
POSCM/quotient classes from traces. The result is deliberately certificate
based: if the learned posterior is calibrated enough to bound posterior
refinement-gain error, then the learned meta-controller has bounded regret
against the theorem-guided posterior controller.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure AmortizedPosteriorEncoder (Trace Feature Latent : Type) where
  encode : Trace -> Feature
  responsibility : Feature -> Latent -> Rat
  nonnegative : forall trace latent, 0 <= responsibility (encode trace) latent
  normalized :
    forall trace : Trace, forall latentClasses : List Latent,
      (latentClasses.map (responsibility (encode trace))).sum = 1

def AmortizedResponsibilityTable {Trace Feature Latent : Type}
    (encoder : AmortizedPosteriorEncoder Trace Feature Latent) :
    ResponsibilityTable Trace Latent where
  weight := fun trace latent => encoder.responsibility (encoder.encode trace) latent
  nonnegative := encoder.nonnegative
  normalized := encoder.normalized

def PosteriorRefinementGainError
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace) : Rat :=
  PosteriorExpectedRefinementGain
      latentClasses truePosterior latentWorldDistribution fine coarse problem fineRule
      coarseRule trace -
    PosteriorExpectedRefinementGain
      latentClasses approx latentWorldDistribution fine coarse problem fineRule
      coarseRule trace

def PosteriorRegretBounded
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (regretBound : Rat) : Prop :=
  PosteriorRefinementGainError
      latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
      coarseRule trace <= regretBound

structure PosteriorCalibrationCertificate
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (epsilon sensitivity : Rat) where
  posteriorDistanceCertified : Prop
  nonnegativeEpsilon : 0 <= epsilon
  nonnegativeSensitivity : 0 <= sensitivity
  gainErrorControlled :
    PosteriorRegretBounded
      latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
      coarseRule trace (sensitivity * epsilon)

structure VariationalObservabilityControllerCertificate
    {Trace Feature Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (encoder : AmortizedPosteriorEncoder Trace Feature Latent)
    (truePosterior : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (observationCost epsilon sensitivity : Rat) where
  learnedPosteriorCostJustified :
    PosteriorCostJustifiedRefinement
      latentClasses
      (AmortizedResponsibilityTable encoder)
      latentWorldDistribution
      fine
      coarse
      problem
      fineRule
      coarseRule
      trace
      observationCost
  calibration :
    PosteriorCalibrationCertificate
      latentClasses
      (AmortizedResponsibilityTable encoder)
      truePosterior
      latentWorldDistribution
      fine
      coarse
      problem
      fineRule
      coarseRule
      trace
      epsilon
      sensitivity

theorem amortized_encoder_induces_responsibility_table
    {Trace Feature Latent : Type}
    (encoder : AmortizedPosteriorEncoder Trace Feature Latent) :
    (AmortizedResponsibilityTable encoder).weight =
      fun trace latent => encoder.responsibility (encoder.encode trace) latent := by
  rfl

theorem posterior_regret_bound_is_gain_error_threshold
    {Trace Latent World FineObs CoarseObs Action : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (latentWorldDistribution : Latent -> FiniteWorldDistribution World)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (fineRule : DecisionRule FineObs Action)
    (coarseRule : DecisionRule CoarseObs Action)
    (trace : Trace)
    (regretBound : Rat) :
    PosteriorRegretBounded
      latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
      coarseRule trace regretBound <->
      PosteriorRefinementGainError
        latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
        coarseRule trace <= regretBound := by
  rfl

theorem calibrated_posterior_has_bounded_refinement_regret
    {Trace Latent World FineObs CoarseObs Action : Type}
    {latentClasses : List Latent}
    {approx truePosterior : ResponsibilityTable Trace Latent}
    {latentWorldDistribution : Latent -> FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    {fineRule : DecisionRule FineObs Action}
    {coarseRule : DecisionRule CoarseObs Action}
    {trace : Trace}
    {epsilon sensitivity : Rat}
    (certificate :
      PosteriorCalibrationCertificate
        latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
        coarseRule trace epsilon sensitivity) :
    PosteriorRegretBounded
      latentClasses approx truePosterior latentWorldDistribution fine coarse problem fineRule
      coarseRule trace (sensitivity * epsilon) := by
  exact certificate.gainErrorControlled

theorem variational_controller_has_bounded_regret
    {Trace Feature Latent World FineObs CoarseObs Action : Type}
    {latentClasses : List Latent}
    {encoder : AmortizedPosteriorEncoder Trace Feature Latent}
    {truePosterior : ResponsibilityTable Trace Latent}
    {latentWorldDistribution : Latent -> FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    {fineRule : DecisionRule FineObs Action}
    {coarseRule : DecisionRule CoarseObs Action}
    {trace : Trace}
    {observationCost epsilon sensitivity : Rat}
    (certificate :
      VariationalObservabilityControllerCertificate
        latentClasses encoder truePosterior latentWorldDistribution fine coarse problem fineRule
        coarseRule trace observationCost epsilon sensitivity) :
    PosteriorRegretBounded
      latentClasses
      (AmortizedResponsibilityTable encoder)
      truePosterior
      latentWorldDistribution
      fine
      coarse
      problem
      fineRule
      coarseRule
      trace
      (sensitivity * epsilon) := by
  exact certificate.calibration.gainErrorControlled

theorem variational_controller_keeps_learned_cost_justification
    {Trace Feature Latent World FineObs CoarseObs Action : Type}
    {latentClasses : List Latent}
    {encoder : AmortizedPosteriorEncoder Trace Feature Latent}
    {truePosterior : ResponsibilityTable Trace Latent}
    {latentWorldDistribution : Latent -> FiniteWorldDistribution World}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    {fineRule : DecisionRule FineObs Action}
    {coarseRule : DecisionRule CoarseObs Action}
    {trace : Trace}
    {observationCost epsilon sensitivity : Rat}
    (certificate :
      VariationalObservabilityControllerCertificate
        latentClasses encoder truePosterior latentWorldDistribution fine coarse problem fineRule
        coarseRule trace observationCost epsilon sensitivity) :
    PosteriorCostJustifiedRefinement
      latentClasses
      (AmortizedResponsibilityTable encoder)
      latentWorldDistribution
      fine
      coarse
      problem
      fineRule
      coarseRule
      trace
      observationCost := by
  exact certificate.learnedPosteriorCostJustified

end Manuscript
end CausalObservationReproduction

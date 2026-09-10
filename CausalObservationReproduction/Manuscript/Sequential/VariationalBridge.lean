import CausalObservationReproduction.Manuscript.VariationalBridge
import CausalObservationReproduction.Manuscript.Sequential.ObservabilityControl

/-
Posterior and variational bridge for sequential continuation value.

This extends the existing exact-vs-amortized responsibility architecture from
one-step refinement gain to finite-horizon continuation gain.  It proves only a
bounded-consequence theorem under an explicit calibration certificate.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

def PosteriorWeightedSequentialValue {Trace Latent ObsPolicy : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (latentPolicyValue : Latent -> ObsPolicy -> Rat)
    (policy : ObsPolicy)
    (trace : Trace) : Rat :=
  latentClasses.map
    (fun latent =>
      responsibilities.weight trace latent *
        latentPolicyValue latent policy)
    |>.sum

def PosteriorSequentialRefinementGain
    {Trace Latent FinePolicy CoarsePolicy : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (fineLatentPolicyValue : Latent -> FinePolicy -> Rat)
    (coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat)
    (finePolicy : FinePolicy)
    (coarsePolicy : CoarsePolicy)
    (trace : Trace) : Rat :=
  PosteriorWeightedSequentialValue
      latentClasses responsibilities fineLatentPolicyValue finePolicy trace -
    PosteriorWeightedSequentialValue
      latentClasses responsibilities coarseLatentPolicyValue coarsePolicy trace

def PosteriorSequentialGainError
    {Trace Latent FinePolicy CoarsePolicy : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (fineLatentPolicyValue : Latent -> FinePolicy -> Rat)
    (coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat)
    (finePolicy : FinePolicy)
    (coarsePolicy : CoarsePolicy)
    (trace : Trace) : Rat :=
  PosteriorSequentialRefinementGain
      latentClasses truePosterior fineLatentPolicyValue coarseLatentPolicyValue
      finePolicy coarsePolicy trace -
    PosteriorSequentialRefinementGain
      latentClasses approx fineLatentPolicyValue coarseLatentPolicyValue
      finePolicy coarsePolicy trace

def PosteriorSequentialRegretBounded
    {Trace Latent FinePolicy CoarsePolicy : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (fineLatentPolicyValue : Latent -> FinePolicy -> Rat)
    (coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat)
    (finePolicy : FinePolicy)
    (coarsePolicy : CoarsePolicy)
    (trace : Trace)
    (regretBound : Rat) : Prop :=
  PosteriorSequentialGainError
      latentClasses approx truePosterior fineLatentPolicyValue coarseLatentPolicyValue
      finePolicy coarsePolicy trace <= regretBound

structure SequentialPosteriorCalibrationCertificate
    {Trace Latent FinePolicy CoarsePolicy : Type}
    (latentClasses : List Latent)
    (approx truePosterior : ResponsibilityTable Trace Latent)
    (fineLatentPolicyValue : Latent -> FinePolicy -> Rat)
    (coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat)
    (finePolicy : FinePolicy)
    (coarsePolicy : CoarsePolicy)
    (trace : Trace)
    (epsilon sensitivity : Rat) where
  posteriorDistanceCertified : Prop
  nonnegativeEpsilon : 0 <= epsilon
  nonnegativeSensitivity : 0 <= sensitivity
  gainErrorControlled :
    PosteriorSequentialRegretBounded
      latentClasses approx truePosterior fineLatentPolicyValue coarseLatentPolicyValue
      finePolicy coarsePolicy trace (sensitivity * epsilon)

theorem posterior_sequential_refinement_gain_is_fine_minus_coarse_value
    {Trace Latent FinePolicy CoarsePolicy : Type}
    (latentClasses : List Latent)
    (responsibilities : ResponsibilityTable Trace Latent)
    (fineLatentPolicyValue : Latent -> FinePolicy -> Rat)
    (coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat)
    (finePolicy : FinePolicy)
    (coarsePolicy : CoarsePolicy)
    (trace : Trace) :
    PosteriorSequentialRefinementGain
        latentClasses responsibilities fineLatentPolicyValue coarseLatentPolicyValue
        finePolicy coarsePolicy trace =
      PosteriorWeightedSequentialValue
          latentClasses responsibilities fineLatentPolicyValue finePolicy trace -
        PosteriorWeightedSequentialValue
          latentClasses responsibilities coarseLatentPolicyValue coarsePolicy trace := by
  rfl

theorem calibrated_variational_sequential_controller_has_bounded_regret
    {Trace Latent FinePolicy CoarsePolicy : Type}
    {latentClasses : List Latent}
    {approx truePosterior : ResponsibilityTable Trace Latent}
    {fineLatentPolicyValue : Latent -> FinePolicy -> Rat}
    {coarseLatentPolicyValue : Latent -> CoarsePolicy -> Rat}
    {finePolicy : FinePolicy}
    {coarsePolicy : CoarsePolicy}
    {trace : Trace}
    {epsilon sensitivity : Rat}
    (certificate :
      SequentialPosteriorCalibrationCertificate
        latentClasses approx truePosterior fineLatentPolicyValue
        coarseLatentPolicyValue finePolicy coarsePolicy trace epsilon sensitivity) :
    PosteriorSequentialRegretBounded
      latentClasses approx truePosterior fineLatentPolicyValue coarseLatentPolicyValue
      finePolicy coarsePolicy trace (sensitivity * epsilon) := by
  exact certificate.gainErrorControlled

/- A calibrator implementation is intentionally abstract.  Python may use an
isotonic map, a conformal bound, or a neural estimator; deployment needs only
this certificate interface. -/
structure CalibratedContinuationCertificate
    {Info ObservationChoice : Type}
    (trueNetValue lowerBound : Info -> ObservationChoice -> Rat) where
  lowerBoundSound : forall info choice, lowerBound info choice <= trueNetValue info choice
  positiveDeploymentSound : forall info choice, 0 < lowerBound info choice -> 0 < trueNetValue info choice

def ConservativeRefinementDeploy
    {Info ObservationChoice : Type}
    (lowerBound : Info -> ObservationChoice -> Rat)
    (info : Info) (choice : ObservationChoice) : Prop :=
  0 < lowerBound info choice

theorem conservative_calibrated_deployment_has_positive_net_value
    {Info ObservationChoice : Type}
    {trueNetValue lowerBound : Info -> ObservationChoice -> Rat}
    (certificate : CalibratedContinuationCertificate trueNetValue lowerBound)
    {info : Info} {choice : ObservationChoice}
    (deploy : ConservativeRefinementDeploy lowerBound info choice) :
    0 < trueNetValue info choice := by
  exact certificate.positiveDeploymentSound info choice deploy

end Sequential
end Manuscript
end CausalObservationReproduction

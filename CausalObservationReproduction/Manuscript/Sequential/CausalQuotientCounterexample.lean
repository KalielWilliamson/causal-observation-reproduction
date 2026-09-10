import Mathlib.Tactic.NormNum
import CausalObservationReproduction.Manuscript.Sequential.DecisionRelativeQuotient

/-
A finite counterexample for a quotient that discards delayed continuation
value.  `Bool` is the latent history-relevant state.  The one-class quotient
cannot retain which of two later control actions is optimal.  A probe is
therefore worthwhile in cumulative value despite being rejected by an
immediate-only gate.

This is a counterexample to an *unqualified* quotient or myopic-gating claim;
it does not assert that every causal quotient has this failure.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

def delayedContinuationQuotient : DecisionRelativeQuotient Bool Unit where
  classOf := fun _ => ()

def delayedOptimalContinuationAction : Bool -> Bool :=
  fun latentHistory => latentHistory

def delayedContinuationReward (latentHistory controlAction : Bool) : Rat :=
  if latentHistory = controlAction then 1 else 0

def delayedActionValue (controlAction : Bool) : Bool -> Rat :=
  fun latentHistory => delayedContinuationReward latentHistory controlAction

def delayedFineTotalReturn : Rat :=
  delayedContinuationReward true (delayedOptimalContinuationAction true) +
    delayedContinuationReward false (delayedOptimalContinuationAction false)

def delayedCoarseTotalReturn (sharedAction : Bool) : Rat :=
  delayedContinuationReward true sharedAction +
    delayedContinuationReward false sharedAction

def delayedProbeCost : Rat := 1 / 2

def immediateAcquireValue : Bool -> Rat := fun _ => -delayedProbeCost

def immediateAbstainValue : Bool -> Rat := fun _ => 0

theorem delayed_quotient_collapses_conflicting_continuation_actions :
    delayedContinuationQuotient.classOf true = delayedContinuationQuotient.classOf false /\
      delayedOptimalContinuationAction true ≠ delayedOptimalContinuationAction false := by
  norm_num [delayedContinuationQuotient, delayedOptimalContinuationAction]

theorem delayed_one_class_cannot_factor_conflicting_action_values :
    IsEmpty (ActionValueFactorsThroughQuotient delayedContinuationQuotient
      (delayedActionValue true) (delayedActionValue false)) := by
  constructor
  intro certificate
  have hSame : delayedActionValue true true = delayedActionValue true false := by
    rw [certificate.acquireFactors true, certificate.acquireFactors false]
  norm_num [delayedActionValue, delayedContinuationReward] at hSame

theorem delayed_fine_policy_strictly_beats_every_shared_coarse_action
    (sharedAction : Bool) :
    delayedCoarseTotalReturn sharedAction < delayedFineTotalReturn := by
  cases sharedAction <;>
    norm_num [delayedCoarseTotalReturn, delayedFineTotalReturn,
      delayedContinuationReward, delayedOptimalContinuationAction]

theorem delayed_probe_has_positive_net_continuation_value :
    0 < delayedFineTotalReturn - delayedCoarseTotalReturn true - delayedProbeCost := by
  norm_num [delayedFineTotalReturn, delayedCoarseTotalReturn,
    delayedContinuationReward, delayedOptimalContinuationAction, delayedProbeCost]

theorem immediate_only_gate_rejects_delayed_probe :
    Not (AcquireIsJustified immediateAcquireValue immediateAbstainValue true) := by
  norm_num [AcquireIsJustified, NetAcquireValue, immediateAcquireValue,
    immediateAbstainValue, delayedProbeCost]

/-
Interpretation: any theorem claiming that a one-class history quotient preserves
sequential acquisition decisions must exclude this family, for example by
proving that the quotient retains all action-relevant continuation targets.
The failed factorization theorem makes that obstruction explicit rather than
only showing that the continuation-optimal actions differ.
-/

end Sequential
end Manuscript
end CausalObservationReproduction

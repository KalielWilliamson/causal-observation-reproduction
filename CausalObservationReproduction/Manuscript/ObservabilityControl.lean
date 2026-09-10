import CausalObservationReproduction.Manuscript.Core
import CausalObservationReproduction.Manuscript.Stochastic

/-
Closed-loop observability control for POSCM-style causal observation regimes.

This module deliberately keeps the mathematics finite and order-theoretic.  It
does not formalize measure-theoretic Bayesian inference or statistical
mechanics directly.  Instead it exposes the proof-bearing skeleton that the
Python simulator and controller can instantiate: traces induce posterior
sufficient statistics, statistics induce a free-energy-style value functional,
and the meta-controller chooses boundary, coarse, probe, or abstain actions by
threshold inequalities.
-/

namespace CausalObservationReproduction
namespace Manuscript

inductive ObservabilityAction where
  | useBoundary
  | useCoarse
  | probe
  | abstain
  deriving DecidableEq, Repr

structure POSCMPosteriorStats where
  boundaryInformationGain : Rat
  coarseInformationGain : Rat
  expectedCausalUtility : Rat
  boundaryObservationCost : Rat
  coarseObservationCost : Rat
  probeInformationGain : Rat
  probeCost : Rat
  uncertaintyRisk : Rat
  hiddenAmbiguityRisk : Rat

def boundaryFreeEnergy (stats : POSCMPosteriorStats) : Rat :=
  stats.boundaryInformationGain +
    stats.expectedCausalUtility -
    stats.boundaryObservationCost -
    stats.uncertaintyRisk -
    stats.hiddenAmbiguityRisk

def coarseFreeEnergy (stats : POSCMPosteriorStats) : Rat :=
  stats.coarseInformationGain +
    stats.expectedCausalUtility -
    stats.coarseObservationCost -
    stats.hiddenAmbiguityRisk

def probeFreeEnergy (stats : POSCMPosteriorStats) : Rat :=
  stats.probeInformationGain - stats.probeCost

def abstainFreeEnergy (_stats : POSCMPosteriorStats) : Rat := 0

def actionFreeEnergy (stats : POSCMPosteriorStats) :
    ObservabilityAction -> Rat
  | ObservabilityAction.useBoundary => boundaryFreeEnergy stats
  | ObservabilityAction.useCoarse => coarseFreeEnergy stats
  | ObservabilityAction.probe => probeFreeEnergy stats
  | ObservabilityAction.abstain => abstainFreeEnergy stats

def ActionOptimal (stats : POSCMPosteriorStats)
    (action : ObservabilityAction) : Prop :=
  ∀ other : ObservabilityAction,
    actionFreeEnergy stats other <= actionFreeEnergy stats action

structure BoundaryDominatesAssumptions (stats : POSCMPosteriorStats) where
  coarse_le_boundary :
    coarseFreeEnergy stats <= boundaryFreeEnergy stats
  probe_le_boundary :
    probeFreeEnergy stats <= boundaryFreeEnergy stats
  abstain_le_boundary :
    abstainFreeEnergy stats <= boundaryFreeEnergy stats

structure CoarseDominatesAssumptions (stats : POSCMPosteriorStats) where
  boundary_le_coarse :
    boundaryFreeEnergy stats <= coarseFreeEnergy stats
  probe_le_coarse :
    probeFreeEnergy stats <= coarseFreeEnergy stats
  abstain_le_coarse :
    abstainFreeEnergy stats <= coarseFreeEnergy stats

structure ProbeDominatesAssumptions (stats : POSCMPosteriorStats) where
  boundary_le_probe :
    boundaryFreeEnergy stats <= probeFreeEnergy stats
  coarse_le_probe :
    coarseFreeEnergy stats <= probeFreeEnergy stats
  abstain_le_probe :
    abstainFreeEnergy stats <= probeFreeEnergy stats

structure AbstainDominatesAssumptions (stats : POSCMPosteriorStats) where
  boundary_le_abstain :
    boundaryFreeEnergy stats <= abstainFreeEnergy stats
  coarse_le_abstain :
    coarseFreeEnergy stats <= abstainFreeEnergy stats
  probe_le_abstain :
    probeFreeEnergy stats <= abstainFreeEnergy stats

def theoremGuidedObservabilityAction (stats : POSCMPosteriorStats) : ObservabilityAction :=
  if boundaryFreeEnergy stats >= coarseFreeEnergy stats ∧
      boundaryFreeEnergy stats >= probeFreeEnergy stats ∧
      boundaryFreeEnergy stats >= abstainFreeEnergy stats then
    ObservabilityAction.useBoundary
  else if coarseFreeEnergy stats >= probeFreeEnergy stats ∧
      coarseFreeEnergy stats >= abstainFreeEnergy stats then
    ObservabilityAction.useCoarse
  else if probeFreeEnergy stats >= abstainFreeEnergy stats then
    ObservabilityAction.probe
  else
    ObservabilityAction.abstain

theorem boundary_dominates_is_action_optimal
    {stats : POSCMPosteriorStats}
    (h : BoundaryDominatesAssumptions stats) :
    ActionOptimal stats ObservabilityAction.useBoundary := by
  intro other
  cases other with
  | useBoundary => simp [actionFreeEnergy]
  | useCoarse => exact h.coarse_le_boundary
  | probe => exact h.probe_le_boundary
  | abstain => exact h.abstain_le_boundary

theorem coarse_dominates_is_action_optimal
    {stats : POSCMPosteriorStats}
    (h : CoarseDominatesAssumptions stats) :
    ActionOptimal stats ObservabilityAction.useCoarse := by
  intro other
  cases other with
  | useBoundary => exact h.boundary_le_coarse
  | useCoarse => simp [actionFreeEnergy]
  | probe => exact h.probe_le_coarse
  | abstain => exact h.abstain_le_coarse

theorem probe_dominates_is_action_optimal
    {stats : POSCMPosteriorStats}
    (h : ProbeDominatesAssumptions stats) :
    ActionOptimal stats ObservabilityAction.probe := by
  intro other
  cases other with
  | useBoundary => exact h.boundary_le_probe
  | useCoarse => exact h.coarse_le_probe
  | probe => simp [actionFreeEnergy]
  | abstain => exact h.abstain_le_probe

theorem abstain_dominates_is_action_optimal
    {stats : POSCMPosteriorStats}
    (h : AbstainDominatesAssumptions stats) :
    ActionOptimal stats ObservabilityAction.abstain := by
  intro other
  cases other with
  | useBoundary => exact h.boundary_le_abstain
  | useCoarse => exact h.coarse_le_abstain
  | probe => exact h.probe_le_abstain
  | abstain => simp [actionFreeEnergy]

structure TracePosteriorEstimator (World Obs : Type) where
  channel : Stochastic.Channel World Obs
  posteriorStats : Obs -> POSCMPosteriorStats

def posteriorStatsFromObservation {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs) : POSCMPosteriorStats :=
  estimator.posteriorStats obs

def closedLoopObservabilityPolicy {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs) : ObservabilityAction :=
  theoremGuidedObservabilityAction
    (posteriorStatsFromObservation estimator obs)

structure ObservabilityOption (World Obs : Type) where
  action : ObservabilityAction
  nextRegime : ObservationRegime World Obs
  termination : Obs -> Prop

structure HierarchicalObservabilityController (World HighObs LowObs : Type) where
  highEstimator : TracePosteriorEstimator World HighObs
  optionFor : HighObs -> ObservabilityOption World LowObs
  lowLevelPolicy : LowObs -> World

theorem closed_loop_boundary_choice_is_optimal_under_thresholds
    {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs)
    (h : BoundaryDominatesAssumptions
      (posteriorStatsFromObservation estimator obs)) :
    ActionOptimal
      (posteriorStatsFromObservation estimator obs)
      ObservabilityAction.useBoundary := by
  exact boundary_dominates_is_action_optimal h

theorem closed_loop_coarse_choice_is_optimal_under_thresholds
    {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs)
    (h : CoarseDominatesAssumptions
      (posteriorStatsFromObservation estimator obs)) :
    ActionOptimal
      (posteriorStatsFromObservation estimator obs)
      ObservabilityAction.useCoarse := by
  exact coarse_dominates_is_action_optimal h

theorem closed_loop_probe_choice_is_optimal_under_thresholds
    {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs)
    (h : ProbeDominatesAssumptions
      (posteriorStatsFromObservation estimator obs)) :
    ActionOptimal
      (posteriorStatsFromObservation estimator obs)
      ObservabilityAction.probe := by
  exact probe_dominates_is_action_optimal h

theorem closed_loop_abstain_choice_is_optimal_under_thresholds
    {World Obs : Type}
    (estimator : TracePosteriorEstimator World Obs)
    (obs : Obs)
    (h : AbstainDominatesAssumptions
      (posteriorStatsFromObservation estimator obs)) :
    ActionOptimal
      (posteriorStatsFromObservation estimator obs)
      ObservabilityAction.abstain := by
  exact abstain_dominates_is_action_optimal h

end Manuscript
end CausalObservationReproduction

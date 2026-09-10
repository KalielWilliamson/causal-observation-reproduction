import CausalObservationReproduction.Manuscript.Sequential.Bellman

/-
Sequential refinement value.

This module lifts deterministic observation refinement from single observations
to histories and finite-horizon policy spaces.  The weak monotonicity theorem is
proved from policy pullback plus value preservation.  Strict improvement is
kept separate and certificate-based: local separability is not enough without
positive reachability and a positive continuation gap.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

structure SequentialValueOptimization (Policy : Type) where
  support : List Policy
  value : Policy -> Rat
  optimum : Rat
  upperBound : forall policy : Policy, policy ∈ support -> value policy <= optimum
  witness : exists policy : Policy, policy ∈ support /\ value policy = optimum

def SequentialRefinementGain (fineValue coarseValue : Rat) : Rat :=
  fineValue - coarseValue

def NetSequentialRefinementValue
    (fineValue coarseValue observationCost : Rat) : Rat :=
  SequentialRefinementGain fineValue coarseValue - observationCost

def SequentialCostJustifiedRefinement
    (fineValue coarseValue observationCost : Rat) : Prop :=
  observationCost < SequentialRefinementGain fineValue coarseValue

theorem pulled_back_policy_preserves_expected_return
    {State FineObs CoarseObs Action : Type}
    [DecidableEq Action]
    (process : FiniteSequentialDecisionProcess State Action)
    (trajectorySupport : List (LatentTrajectory State Action))
    {fine : ObservationRegime State FineObs}
    {coarse : ObservationRegime State CoarseObs}
    (projection : FineObs -> CoarseObs)
    (hProjection : forall state : State, coarse state = projection (fine state))
    (coarsePolicy : SequentialPolicy CoarseObs Action)
    (simulation :
      SequentialPolicySimulation projection hProjection coarsePolicy) :
    expectedCumulativeReturn
        process trajectorySupport fine (pullbackPolicy projection coarsePolicy) =
      expectedCumulativeReturn
        process trajectorySupport coarse coarsePolicy := by
  unfold expectedCumulativeReturn
  apply congrArg List.sum
  apply List.map_congr_left
  intro trajectory _hTrajectory
  rw [pulled_back_policy_preserves_trajectory_distribution
    projection hProjection coarsePolicy simulation trajectory]

theorem refinement_weakly_increases_optimal_sequential_value
    {CoarsePolicy FinePolicy : Type}
    (coarse : SequentialValueOptimization CoarsePolicy)
    (pullback : CoarsePolicy -> FinePolicy)
    (fineProblem : SequentialValueOptimization FinePolicy)
    (fineContainsPullbacks :
      forall policy : CoarsePolicy,
        policy ∈ coarse.support -> pullback policy ∈ fineProblem.support)
    (preservesValue :
      forall policy : CoarsePolicy,
        policy ∈ coarse.support ->
          fineProblem.value (pullback policy) = coarse.value policy) :
    coarse.optimum <= fineProblem.optimum := by
  rcases coarse.witness with ⟨policy, hPolicyMem, hPolicyValue⟩
  calc
    coarse.optimum = coarse.value policy := hPolicyValue.symm
    _ = fineProblem.value (pullback policy) :=
      (preservesValue policy hPolicyMem).symm
    _ <= fineProblem.optimum :=
      fineProblem.upperBound (pullback policy)
        (fineContainsPullbacks policy hPolicyMem)

def CoarseContinuationPolicyBlocks
    {CoarseObs Action : Type}
    (t : Nat)
    (sharedHistory : ObservationHistory CoarseObs Action)
    (qLeft qRight : Action -> Rat)
    (policy : SequentialPolicy CoarseObs Action) : Prop :=
  (forall action : Action, qLeft action <= qLeft (policy t sharedHistory)) /\
    (forall action : Action, qRight action <= qRight (policy t sharedHistory))

def DisjointContinuationOptima {Action : Type}
    (qLeft qRight : Action -> Rat) : Prop :=
  forall action : Action,
    (forall candidate : Action, qLeft candidate <= qLeft action) ->
      Not (forall candidate : Action, qRight candidate <= qRight action)

structure ContinuationDecisionRelevantPair
    (FineObs CoarseObs Action : Type) where
  t : Nat
  coarseHistory : ObservationHistory CoarseObs Action
  fineLeftHistory : ObservationHistory FineObs Action
  fineRightHistory : ObservationHistory FineObs Action
  leftAction : Action
  rightAction : Action
  fallbackAction : Action
  qLeft : Action -> Rat
  qRight : Action -> Rat
  equalCoarseHistories : coarseHistory = coarseHistory
  fineSeparated : fineLeftHistory ≠ fineRightHistory
  leftOptimal : forall action : Action, qLeft action <= qLeft leftAction
  rightOptimal : forall action : Action, qRight action <= qRight rightAction
  disjointOptimalSets : DisjointContinuationOptima qLeft qRight
  leftReachabilityPositive : Prop
  rightReachabilityPositive : Prop
  actionValueGapPositive : Prop

structure DecisionRelevantHistoryRegion
    (FineObs CoarseObs Action : Type) where
  inRegion : ObservationHistory FineObs Action -> Prop
  coarseHistory : ObservationHistory CoarseObs Action
  witness :
    forall history : ObservationHistory FineObs Action, inRegion history ->
      exists other : ObservationHistory FineObs Action,
        Not (history = other) /\
          exists qLeft qRight : Action -> Rat,
            DisjointContinuationOptima qLeft qRight

theorem coarse_history_quotient_blocks_continuation_optimal_policy
    {CoarseObs Action : Type}
    (t : Nat)
    (sharedHistory : ObservationHistory CoarseObs Action)
    (qLeft qRight : Action -> Rat)
    (hDisjoint : DisjointContinuationOptima qLeft qRight) :
    Not (exists policy : SequentialPolicy CoarseObs Action,
      CoarseContinuationPolicyBlocks t sharedHistory qLeft qRight policy) := by
  intro hPolicy
  rcases hPolicy with ⟨policy, hBlocks⟩
  exact hDisjoint (policy t sharedHistory) hBlocks.1 hBlocks.2

def pairwiseFineContinuationPolicy
    {FineObs Action : Type}
    [DecidableEq FineObs] [DecidableEq Action]
    (pair : ContinuationDecisionRelevantPair FineObs CoarseObs Action) :
    SequentialPolicy FineObs Action :=
  fun t history =>
    if t = pair.t then
      if history = pair.fineLeftHistory then
        pair.leftAction
      else if history = pair.fineRightHistory then
        pair.rightAction
      else
        pair.fallbackAction
    else
      pair.fallbackAction

theorem fine_history_refinement_enables_pairwise_continuation_control
    {FineObs CoarseObs Action : Type}
    [DecidableEq FineObs] [DecidableEq Action]
    (pair : ContinuationDecisionRelevantPair FineObs CoarseObs Action) :
    exists policy : SequentialPolicy FineObs Action,
      policy pair.t pair.fineLeftHistory = pair.leftAction /\
        policy pair.t pair.fineRightHistory = pair.rightAction := by
  refine ⟨pairwiseFineContinuationPolicy pair, ?_⟩
  constructor
  · unfold pairwiseFineContinuationPolicy
    simp
  · unfold pairwiseFineContinuationPolicy
    have hNotRightEqLeft : pair.fineRightHistory ≠ pair.fineLeftHistory := by
      intro h
      exact pair.fineSeparated (Eq.symm h)
    simp [hNotRightEqLeft]

theorem decision_relevant_history_region_membership_supplies_separation
    {FineObs CoarseObs Action : Type}
    (region : DecisionRelevantHistoryRegion FineObs CoarseObs Action)
    {history : ObservationHistory FineObs Action}
    (hHistory : region.inRegion history) :
    exists other : ObservationHistory FineObs Action,
      Not (history = other) /\
        exists qLeft qRight : Action -> Rat,
          DisjointContinuationOptima qLeft qRight := by
  exact region.witness history hHistory

structure StrictSequentialRefinementCertificate
    (FineObs CoarseObs Action : Type)
    (fineValue coarseValue : Rat) where
  pair : ContinuationDecisionRelevantPair FineObs CoarseObs Action
  positiveReachabilityAssumptions :
    pair.leftReachabilityPositive /\ pair.rightReachabilityPositive
  positiveGapAssumption : pair.actionValueGapPositive
  expectedStrictGain : coarseValue < fineValue

structure QuantitativeStrictSequentialRefinementCertificate
    (FineObs CoarseObs Action : Type)
    (fineValue coarseValue ambiguousProbability qGap : Rat) where
  strictCertificate :
    StrictSequentialRefinementCertificate
      FineObs CoarseObs Action fineValue coarseValue
  positiveAmbiguousProbability : 0 < ambiguousProbability
  positiveQGap : 0 < qGap
  positiveWeightedGap : 0 < ambiguousProbability * qGap
  lowerBound :
    ambiguousProbability * qGap <=
      SequentialRefinementGain fineValue coarseValue

theorem decision_relevant_history_refinement_has_strict_sequential_value
    {FineObs CoarseObs Action : Type}
    {fineValue coarseValue : Rat}
    (certificate :
      StrictSequentialRefinementCertificate
        FineObs CoarseObs Action fineValue coarseValue) :
    coarseValue < fineValue := by
  exact certificate.expectedStrictGain

theorem quantitative_history_refinement_gain_lower_bound
    {FineObs CoarseObs Action : Type}
    {fineValue coarseValue ambiguousProbability qGap : Rat}
    (certificate :
      QuantitativeStrictSequentialRefinementCertificate
        FineObs CoarseObs Action fineValue coarseValue ambiguousProbability qGap) :
    ambiguousProbability * qGap <=
      SequentialRefinementGain fineValue coarseValue := by
  exact certificate.lowerBound

theorem sequential_refinement_gain_is_fine_minus_coarse_value
    (fineValue coarseValue : Rat) :
    SequentialRefinementGain fineValue coarseValue = fineValue - coarseValue := by
  rfl

theorem net_sequential_refinement_value_is_gain_minus_cost
    (fineValue coarseValue observationCost : Rat) :
    NetSequentialRefinementValue fineValue coarseValue observationCost =
      SequentialRefinementGain fineValue coarseValue - observationCost := by
  rfl

theorem sequential_cost_justified_refinement_is_gain_threshold
    (fineValue coarseValue observationCost : Rat) :
    SequentialCostJustifiedRefinement fineValue coarseValue observationCost <->
      observationCost < SequentialRefinementGain fineValue coarseValue := by
  rfl

/-
The paper-facing compression of the sequential refinement chain.  A causal
inductive bias supplies the relevant history refinement and continuation-gap
certificate; active RL treats observing as an action and selects it precisely
when the resulting continuation-value gain clears its acquisition cost.

This theorem adds no new mathematical assumption or conclusion: it packages
the existing strict-value and cost-threshold stages into one reader-facing
projection.
-/
theorem active_causal_meta_rl_refinement_is_cost_justified
    {FineObs CoarseObs Action : Type}
    {fineValue coarseValue : Rat}
    (certificate :
      StrictSequentialRefinementCertificate
        FineObs CoarseObs Action fineValue coarseValue)
    (observationCost : Rat)
    (hCost : observationCost < SequentialRefinementGain fineValue coarseValue) :
    coarseValue < fineValue /\
      SequentialCostJustifiedRefinement fineValue coarseValue observationCost := by
  constructor
  · exact decision_relevant_history_refinement_has_strict_sequential_value certificate
  · exact hCost

end Sequential
end Manuscript
end CausalObservationReproduction

import Std
import CausalObservationReproduction.Manuscript.ObservabilityControl

/-
Decision-value layer for causal observability.

The core observability files prove when a query is identifiable from an
observation quotient. This module lifts the same quotient idea to control:
an observation regime is decision-relevant when it separates latent worlds that
require incompatible optimal actions.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure DecisionProblem (World Action : Type) where
  utility : World -> Action -> Rat

abbrev DecisionRule (Obs Action : Type) := Obs -> Action

def OptimalAction {World Action : Type}
    (problem : DecisionProblem World Action)
    (world : World)
    (action : Action) : Prop :=
  forall candidate : Action,
    problem.utility world candidate <= problem.utility world action

def StrictlyPrefers {World Action : Type}
    (problem : DecisionProblem World Action)
    (world : World)
    (better worse : Action) : Prop :=
  problem.utility world worse < problem.utility world better

def PointwiseOptimal {World Obs Action : Type}
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action) : Prop :=
  forall world : World, OptimalAction problem world (rule (O world))

def PairPointwiseOptimalAt {World Obs Action : Type}
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action)
    (rule : DecisionRule Obs Action)
    (left right : World) : Prop :=
  OptimalAction problem left (rule (O left)) /\
    OptimalAction problem right (rule (O right))

def StrictLocalObservabilityValue {World FineObs CoarseObs Action : Type}
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action)
    (left right : World) : Prop :=
  (Not (exists rule : DecisionRule CoarseObs Action,
    PointwiseOptimal coarse problem rule)) /\
    exists rule : DecisionRule FineObs Action,
      PairPointwiseOptimalAt fine problem rule left right

def OptimalSetsDisjoint {World Action : Type}
    (problem : DecisionProblem World Action)
    (left right : World) : Prop :=
  forall action : Action,
    OptimalAction problem left action -> Not (OptimalAction problem right action)

def UtilityRespectsObservation {World Obs Action : Type}
    (O : ObservationRegime World Obs)
    (problem : DecisionProblem World Action) : Prop :=
  forall left right : World, forall action : Action,
    TraceEquivalent O left right ->
      problem.utility left action = problem.utility right action

def ActionValueTarget {World Action : Type}
    (problem : DecisionProblem World Action)
    (action : Action) : World -> Rat :=
  fun world => problem.utility world action

structure DecisionRelevantRefinementPair
    (World FineObs CoarseObs Action : Type)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  refines : Refines fine coarse
  left : World
  right : World
  coarseEquivalent : TraceEquivalent coarse left right
  fineSeparated : Not (TraceEquivalent fine left right)
  leftAction : Action
  rightAction : Action
  fallbackAction : Action
  leftOptimal : OptimalAction problem left leftAction
  rightOptimal : OptimalAction problem right rightAction
  disjointOptimalSets : OptimalSetsDisjoint problem left right

structure DecisionRelevantObservabilityRegion
    (World FineObs CoarseObs Action : Type)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  refines : Refines fine coarse
  inRegion : World -> Prop
  witness :
    forall world : World, inRegion world ->
      exists other : World,
        TraceEquivalent coarse world other /\
          Not (TraceEquivalent fine world other) /\
          OptimalSetsDisjoint problem world other

structure DecisionRelevantRegionWitness
    (World FineObs CoarseObs Action : Type)
    (fine : ObservationRegime World FineObs)
    (coarse : ObservationRegime World CoarseObs)
    (problem : DecisionProblem World Action) where
  refines : Refines fine coarse
  world : World
  other : World
  inRegion : World -> Prop
  worldInRegion : inRegion world
  coarseEquivalent : TraceEquivalent coarse world other
  fineSeparated : Not (TraceEquivalent fine world other)
  leftAction : Action
  rightAction : Action
  fallbackAction : Action
  leftOptimal : OptimalAction problem world leftAction
  rightOptimal : OptimalAction problem other rightAction
  disjointOptimalSets : OptimalSetsDisjoint problem world other

def pairDecisionRule {World FineObs Action : Type}
    [DecidableEq FineObs]
    (fine : ObservationRegime World FineObs)
    (left right : World)
    (leftAction rightAction fallbackAction : Action) :
    DecisionRule FineObs Action :=
  fun obs =>
    if obs = fine left then
      leftAction
    else if obs = fine right then
      rightAction
    else
      fallbackAction

theorem strictly_preferred_action_excludes_worse_optimality
    {World Action : Type}
    {problem : DecisionProblem World Action}
    {world : World}
    {better worse : Action}
    (hPrefers : StrictlyPrefers problem world better worse) :
    Not (OptimalAction problem world worse) := by
  intro hOptimal
  have hLoop :
      problem.utility world worse < problem.utility world worse :=
    Std.lt_of_lt_of_le hPrefers (hOptimal better)
  exact Std.lt_irrefl hLoop

theorem utility_respecting_observation_identifies_action_value
    {World Obs Action : Type}
    {O : ObservationRegime World Obs}
    {problem : DecisionProblem World Action}
    (hRespects : UtilityRespectsObservation O problem)
    (action : Action) :
    Identifiable O (ActionValueTarget problem action) := by
  intro left right hEquivalent
  exact hRespects left right action hEquivalent

theorem refinement_preserves_action_value_identifiability
    {World FineObs CoarseObs Action : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (hRefines : Refines fine coarse)
    (action : Action)
    (hIdentifiable : Identifiable coarse (ActionValueTarget problem action)) :
    Identifiable fine (ActionValueTarget problem action) := by
  exact refinement_preserves_identifiability hRefines hIdentifiable

theorem coarse_quotient_blocks_pointwise_optimal_decision_rule
    {World CoarseObs Action : Type}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    {left right : World}
    (hEquivalent : TraceEquivalent coarse left right)
    (hDisjoint : OptimalSetsDisjoint problem left right) :
    Not (exists rule : DecisionRule CoarseObs Action,
      PointwiseOptimal coarse problem rule) := by
  intro hExists
  rcases hExists with ⟨rule, hRuleOptimal⟩
  have hLeft : OptimalAction problem left (rule (coarse left)) :=
    hRuleOptimal left
  have hRight : OptimalAction problem right (rule (coarse right)) :=
    hRuleOptimal right
  have hLeftAtRightObservation :
      OptimalAction problem left (rule (coarse right)) := by
    rw [← hEquivalent]
    exact hLeft
  exact hDisjoint (rule (coarse right)) hLeftAtRightObservation hRight

theorem pair_decision_rule_selects_left
    {World FineObs Action : Type}
    [DecidableEq FineObs]
    (fine : ObservationRegime World FineObs)
    (left right : World)
    (leftAction rightAction fallbackAction : Action) :
    pairDecisionRule fine left right leftAction rightAction fallbackAction
      (fine left) = leftAction := by
  unfold pairDecisionRule
  simp

theorem pair_decision_rule_selects_right
    {World FineObs Action : Type}
    [DecidableEq FineObs]
    (fine : ObservationRegime World FineObs)
    (left right : World)
    (leftAction rightAction fallbackAction : Action)
    (hSeparated : Not (TraceEquivalent fine left right)) :
    pairDecisionRule fine left right leftAction rightAction fallbackAction
      (fine right) = rightAction := by
  unfold pairDecisionRule
  have hNotRightEqLeft : Not (fine right = fine left) := by
    intro hRightEqLeft
    exact hSeparated hRightEqLeft.symm
  simp [hNotRightEqLeft]

theorem fine_refinement_enables_pair_optimal_decision_rule
    {World FineObs CoarseObs Action : Type}
    [DecidableEq FineObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (pair : DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem) :
    exists rule : DecisionRule FineObs Action,
      PairPointwiseOptimalAt fine problem rule pair.left pair.right := by
  let rule :=
    pairDecisionRule
      fine
      pair.left
      pair.right
      pair.leftAction
      pair.rightAction
      pair.fallbackAction
  refine ⟨rule, ?_⟩
  constructor
  · unfold rule
    rw [pair_decision_rule_selects_left]
    exact pair.leftOptimal
  · unfold rule
    rw [pair_decision_rule_selects_right fine pair.left pair.right
      pair.leftAction pair.rightAction pair.fallbackAction pair.fineSeparated]
    exact pair.rightOptimal

def regionWitnessToPair
    {World FineObs CoarseObs Action : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (witness : DecisionRelevantRegionWitness
      World FineObs CoarseObs Action fine coarse problem) :
    DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem where
  refines := witness.refines
  left := witness.world
  right := witness.other
  coarseEquivalent := witness.coarseEquivalent
  fineSeparated := witness.fineSeparated
  leftAction := witness.leftAction
  rightAction := witness.rightAction
  fallbackAction := witness.fallbackAction
  leftOptimal := witness.leftOptimal
  rightOptimal := witness.rightOptimal
  disjointOptimalSets := witness.disjointOptimalSets

theorem decision_relevant_region_membership_supplies_quotient_obstruction
    {World FineObs CoarseObs Action : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (region : DecisionRelevantObservabilityRegion
      World FineObs CoarseObs Action fine coarse problem)
    {world : World}
    (hWorld : region.inRegion world) :
    exists other : World,
      TraceEquivalent coarse world other /\
        Not (TraceEquivalent fine world other) /\
        OptimalSetsDisjoint problem world other := by
  exact region.witness world hWorld

def decision_relevant_region_witness_to_pair
    {World FineObs CoarseObs Action : Type}
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (witness : DecisionRelevantRegionWitness
      World FineObs CoarseObs Action fine coarse problem) :
    DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem :=
  regionWitnessToPair witness

theorem decision_relevant_causal_quotient_value_theorem
    {World FineObs CoarseObs Action : Type}
    [DecidableEq FineObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (pair : DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem) :
    (Not (exists rule : DecisionRule CoarseObs Action,
      PointwiseOptimal coarse problem rule)) /\
      exists rule : DecisionRule FineObs Action,
        PairPointwiseOptimalAt fine problem rule pair.left pair.right := by
  constructor
  · exact coarse_quotient_blocks_pointwise_optimal_decision_rule
      pair.coarseEquivalent pair.disjointOptimalSets
  · exact fine_refinement_enables_pair_optimal_decision_rule pair

theorem decision_relevant_refinement_has_strict_local_observability_value
    {World FineObs CoarseObs Action : Type}
    [DecidableEq FineObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (pair : DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem) :
    StrictLocalObservabilityValue fine coarse problem pair.left pair.right := by
  exact decision_relevant_causal_quotient_value_theorem pair

theorem decision_relevant_region_witness_has_strict_local_observability_value
    {World FineObs CoarseObs Action : Type}
    [DecidableEq FineObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (witness : DecisionRelevantRegionWitness
      World FineObs CoarseObs Action fine coarse problem) :
    StrictLocalObservabilityValue
      fine coarse problem witness.world witness.other := by
  exact decision_relevant_refinement_has_strict_local_observability_value
    (regionWitnessToPair witness)

theorem boundary_threshold_selects_decision_relevant_refinement
    {World FineObs CoarseObs Action : Type}
    [DecidableEq FineObs]
    {fine : ObservationRegime World FineObs}
    {coarse : ObservationRegime World CoarseObs}
    {problem : DecisionProblem World Action}
    (pair : DecisionRelevantRefinementPair
      World FineObs CoarseObs Action fine coarse problem)
    (stats : POSCMPosteriorStats)
    (hBoundary : BoundaryDominatesAssumptions stats) :
    ActionOptimal stats ObservabilityAction.useBoundary /\
      (Not (exists rule : DecisionRule CoarseObs Action,
        PointwiseOptimal coarse problem rule)) /\
      exists rule : DecisionRule FineObs Action,
        PairPointwiseOptimalAt fine problem rule pair.left pair.right := by
  constructor
  · exact boundary_dominates_is_action_optimal hBoundary
  · exact decision_relevant_causal_quotient_value_theorem pair

end Manuscript
end CausalObservationReproduction

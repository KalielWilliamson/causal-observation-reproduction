import CausalObservationReproduction.Manuscript.Core

/-
Centralized observation-kernel semantics.

An agent can be partially observable only with respect to its complete visible
boundary.  Telemetry alone is not an observation regime if reward, termination,
action-result metadata, messages, visuals, or refinement receipts reveal a
distinction that telemetry hides.  This module makes that boundary explicit.
-/

namespace CausalObservationReproduction
namespace Manuscript

inductive FeedbackChannel where
  | telemetry
  | reward
  | termination
  | actionResult
  | message
  | visual
  | refinement
  deriving DecidableEq, Repr

structure RawTransition
    (Telemetry Reward Termination ActionResult Message Visual Refinement : Type) where
  telemetry : Telemetry
  reward : Reward
  termination : Termination
  actionResult : ActionResult
  message : Message
  visual : Visual
  refinement : Refinement

structure AgentTransition
    (Telemetry Reward Termination ActionResult Message Visual Refinement : Type) where
  telemetry : Telemetry
  reward : Reward
  termination : Termination
  actionResult : ActionResult
  message : Message
  visual : Visual
  refinement : Refinement

structure ObservationKernel
    (RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type) where
  telemetryProjection : RawTelemetry -> AgentTelemetry
  rewardProjection : RawReward -> AgentReward
  terminationProjection : RawTermination -> AgentTermination
  actionResultProjection : RawActionResult -> AgentActionResult
  messageProjection : RawMessage -> AgentMessage
  visualProjection : RawVisual -> AgentVisual
  refinementProjection : RawRefinement -> AgentRefinement

def ObservationKernel.project
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (raw : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement) :
    AgentTransition
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement :=
  {
    telemetry := kernel.telemetryProjection raw.telemetry
    reward := kernel.rewardProjection raw.reward
    termination := kernel.terminationProjection raw.termination
    actionResult := kernel.actionResultProjection raw.actionResult
    message := kernel.messageProjection raw.message
    visual := kernel.visualProjection raw.visual
    refinement := kernel.refinementProjection raw.refinement
  }

def ObservationKernel.regime
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement) :
    ObservationRegime
      (RawTransition
        RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
      (AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement) :=
  kernel.project

def ObservationKernel.compose
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement LearnedObservation : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (representation :
      AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement -> LearnedObservation) :
    ObservationRegime
      (RawTransition
        RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
      LearnedObservation :=
  fun raw => representation (kernel.project raw)

theorem kernel_refines_its_composed_representation
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement LearnedObservation : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (representation :
      AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement -> LearnedObservation) :
    Refines kernel.regime (kernel.compose representation) := by
  exact ⟨representation, by intro raw; rfl⟩

theorem kernel_composition_preserves_identifiability
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement LearnedObservation Target : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (representation :
      AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement -> LearnedObservation)
    (query : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement -> Target)
    (hIdentifiable : Identifiable (kernel.compose representation) query) :
    Identifiable kernel.regime query := by
  exact refinement_preserves_identifiability
    (kernel_refines_its_composed_representation kernel representation) hIdentifiable

theorem causal_query_identifiable_iff_constant_on_kernel_classes
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Target : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (query : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement -> Target) :
    Identifiable kernel.regime query ↔ ConstantOnObservationClasses kernel.regime query := by
  exact identifiable_iff_constant_on_observation_classes kernel.regime query

theorem observation_kernel_refines_telemetry
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement) :
    Refines kernel.regime (fun raw => kernel.telemetryProjection raw.telemetry) := by
  refine ⟨fun agent => agent.telemetry, ?_⟩
  intro raw
  rfl

theorem observation_kernel_project_eq_of_channels_eq
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hTelemetry : kernel.telemetryProjection left.telemetry = kernel.telemetryProjection right.telemetry)
    (hReward : kernel.rewardProjection left.reward = kernel.rewardProjection right.reward)
    (hTermination : kernel.terminationProjection left.termination = kernel.terminationProjection right.termination)
    (hActionResult : kernel.actionResultProjection left.actionResult = kernel.actionResultProjection right.actionResult)
    (hMessage : kernel.messageProjection left.message = kernel.messageProjection right.message)
    (hVisual : kernel.visualProjection left.visual = kernel.visualProjection right.visual)
    (hRefinement : kernel.refinementProjection left.refinement = kernel.refinementProjection right.refinement) :
    kernel.project left = kernel.project right := by
  cases left
  cases right
  simp_all [ObservationKernel.project]

theorem telemetry_alias_is_not_a_kernel_alias_when_feedback_leaks
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hTelemetry : kernel.telemetryProjection left.telemetry = kernel.telemetryProjection right.telemetry)
    (hKernel : kernel.project left ≠ kernel.project right) :
    TraceEquivalent (fun raw => kernel.telemetryProjection raw.telemetry) left right ∧
      ¬ TraceEquivalent kernel.regime left right := by
  constructor
  · exact hTelemetry
  · exact hKernel

theorem reward_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hReward : kernel.rewardProjection left.reward ≠ kernel.rewardProjection right.reward) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hReward
  have hField := congrArg (fun observed => observed.reward) hProject
  simpa [ObservationKernel.project] using hField

theorem termination_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hTermination : kernel.terminationProjection left.termination ≠ kernel.terminationProjection right.termination) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hTermination
  have hField := congrArg (fun observed => observed.termination) hProject
  simpa [ObservationKernel.project] using hField

theorem action_result_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hActionResult : kernel.actionResultProjection left.actionResult ≠ kernel.actionResultProjection right.actionResult) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hActionResult
  have hField := congrArg (fun observed => observed.actionResult) hProject
  simpa [ObservationKernel.project] using hField

theorem message_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hMessage : kernel.messageProjection left.message ≠ kernel.messageProjection right.message) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hMessage
  have hField := congrArg (fun observed => observed.message) hProject
  simpa [ObservationKernel.project] using hField

theorem visual_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hVisual : kernel.visualProjection left.visual ≠ kernel.visualProjection right.visual) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hVisual
  have hField := congrArg (fun observed => observed.visual) hProject
  simpa [ObservationKernel.project] using hField

theorem refinement_feedback_separates_kernel
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hRefinement : kernel.refinementProjection left.refinement ≠ kernel.refinementProjection right.refinement) :
    kernel.project left ≠ kernel.project right := by
  intro hProject
  apply hRefinement
  have hField := congrArg (fun observed => observed.refinement) hProject
  simpa [ObservationKernel.project] using hField

structure ResourceVector where
  time : Rat
  energy : Rat
  risk : Rat
  money : Rat

def ResourceVector.withinBudget (cost budget : ResourceVector) : Prop :=
  cost.time ≤ budget.time ∧
    cost.energy ≤ budget.energy ∧
    cost.risk ≤ budget.risk ∧
    cost.money ≤ budget.money

structure AcceptedRefinementAction (cost budget : ResourceVector) where
  componentwiseAdmissible : cost.withinBudget budget

theorem accepted_refinement_action_is_within_budget
    {cost budget : ResourceVector}
    (accepted : AcceptedRefinementAction cost budget) :
    cost.withinBudget budget := by
  exact accepted.componentwiseAdmissible

def ResourceVector.scalarize (weights cost : ResourceVector) : Rat :=
  weights.time * cost.time + weights.energy * cost.energy +
    weights.risk * cost.risk + weights.money * cost.money

inductive RefinementCostAccounting where
  | environmentInclusive
  | transportDeducts
  deriving DecidableEq, Repr

def adjustedReward
    (accounting : RefinementCostAccounting)
    (weights cost : ResourceVector)
    (rawReward : Rat) : Rat :=
  match accounting with
  | .environmentInclusive => rawReward
  | .transportDeducts => rawReward - ResourceVector.scalarize weights cost

theorem environment_inclusive_preserves_raw_reward
    (weights cost : ResourceVector)
    (rawReward : Rat) :
    adjustedReward .environmentInclusive weights cost rawReward = rawReward := by
  rfl

theorem transport_deducts_scalarized_refinement_cost_once
    (weights cost : ResourceVector)
    (rawReward : Rat) :
    adjustedReward .transportDeducts weights cost rawReward =
      rawReward - ResourceVector.scalarize weights cost := by
  rfl

end Manuscript
end CausalObservationReproduction

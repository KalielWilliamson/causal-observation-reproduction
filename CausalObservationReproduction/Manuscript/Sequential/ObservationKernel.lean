import CausalObservationReproduction.Manuscript.ObservationKernel
import CausalObservationReproduction.Manuscript.Sequential.Core
import CausalObservationReproduction.Manuscript.Sequential.RefinementValue

/-
Sequential semantics for the centralized observation kernel.

The policy boundary is not a single projected telemetry value.  It is the
entire history of projected transitions, including precisely the feedback
channels admitted by the kernel at each step.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

def observeKernelTrajectory
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Action : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (trajectory : LatentTrajectory
      (RawTransition
        RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
      Action) :
    ObservationHistory
      (AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement)
      Action :=
  observeTrajectory kernel.regime trajectory

abbrev KernelSequentialPolicy
    (AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Action : Type) :=
  SequentialPolicy
    (AgentTransition
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    Action

theorem kernel_policy_information_is_projected_transition_history
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Action : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (trajectory : LatentTrajectory
      (RawTransition
        RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
      Action) :
    observeKernelTrajectory kernel trajectory = observeTrajectory kernel.regime trajectory := by
  rfl

theorem kernel_history_refinement_preserves_coarse_policy_pullback
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement LearnedObservation Action : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (representation :
      AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement -> LearnedObservation)
    (coarsePolicy : SequentialPolicy LearnedObservation Action) :
    exists finePolicy : KernelSequentialPolicy
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Action,
      forall trajectory : LatentTrajectory
        (RawTransition
          RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
        Action,
        finePolicy (trajectoryLength trajectory)
          (observeKernelTrajectory kernel trajectory) =
        coarsePolicy (trajectoryLength trajectory)
          (projectObservationHistory representation
            (observeKernelTrajectory kernel trajectory)) := by
  refine ⟨pullbackPolicy representation coarsePolicy, ?_⟩
  intro trajectory
  rfl

theorem decision_relevant_kernel_alias_blocks_common_continuation_optimum
    {RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement Action : Type}
    (kernel : ObservationKernel
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement
      AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
      AgentRefinement)
    (left right : RawTransition
      RawTelemetry RawReward RawTermination RawActionResult RawMessage RawVisual RawRefinement)
    (hAlias : TraceEquivalent kernel.regime left right)
    (qLeft qRight : Action -> Rat)
    (hDisjoint : DisjointContinuationOptima qLeft qRight) :
    Not (exists decision :
      AgentTransition
        AgentTelemetry AgentReward AgentTermination AgentActionResult AgentMessage AgentVisual
        AgentRefinement -> Action,
      (forall action : Action, qLeft action <= qLeft (decision (kernel.project left))) /\
      (forall action : Action, qRight action <= qRight (decision (kernel.project right)))) := by
  intro hDecision
  rcases hDecision with ⟨decision, hLeft, hRight⟩
  apply hDisjoint (decision (kernel.project left)) hLeft
  have hProjection : kernel.project left = kernel.project right := hAlias
  rw [hProjection]
  exact hRight

end Sequential
end Manuscript
end CausalObservationReproduction

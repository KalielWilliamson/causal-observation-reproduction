import CausalObservationReproduction.Manuscript.Sequential.RefinementValue

/-
Finite rational belief-state bridge.

Beliefs are finite rational tables over the declared state support.  Bayesian
update is represented by an explicit update operator plus a preservation
certificate; this avoids continuous-simplex machinery and keeps the history
theory independent of belief-state reasoning.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

structure FiniteBelief (State : Type) where
  support : List State
  weight : State -> Rat
  nonnegative : forall state : State, 0 <= weight state
  normalized : (support.map weight).sum = 1
  zeroOutsideSupport : forall state : State, state ∉ support -> weight state = 0

structure BeliefUpdateKernel (State Action Obs : Type) where
  update : FiniteBelief State -> Action -> Obs -> FiniteBelief State

def BeliefPolicy (State _Obs Action : Type) :=
  Nat -> FiniteBelief State -> Action

def beliefPolicyInducesHistoryPolicy
    {State Obs Action : Type}
    (beliefFromHistory : ObservationHistory Obs Action -> FiniteBelief State)
    (policy : BeliefPolicy State Obs Action) :
    SequentialPolicy Obs Action :=
  fun t history => policy t (beliefFromHistory history)

structure BeliefSufficiencyCertificate (State Obs Action : Type) where
  beliefFromHistory : ObservationHistory Obs Action -> FiniteBelief State
  historyValue : ObservationHistory Obs Action -> Rat
  beliefValue : FiniteBelief State -> Rat
  valueDependsOnlyOnBelief :
    forall history : ObservationHistory Obs Action,
      historyValue history = beliefValue (beliefFromHistory history)

theorem belief_update_preserves_normalization
    {State Action Obs : Type}
    (kernel : BeliefUpdateKernel State Action Obs)
    (belief : FiniteBelief State)
    (action : Action)
    (obs : Obs) :
    ((kernel.update belief action obs).support.map
      (kernel.update belief action obs).weight).sum = 1 := by
  exact (kernel.update belief action obs).normalized

theorem belief_policy_induces_history_policy
    {State Obs Action : Type}
    (beliefFromHistory : ObservationHistory Obs Action -> FiniteBelief State)
    (policy : BeliefPolicy State Obs Action) :
    exists historyPolicy : SequentialPolicy Obs Action,
      historyPolicy = beliefPolicyInducesHistoryPolicy beliefFromHistory policy := by
  exact ⟨beliefPolicyInducesHistoryPolicy beliefFromHistory policy, rfl⟩

theorem bellman_value_depends_only_on_belief
    {State Obs Action : Type}
    (certificate : BeliefSufficiencyCertificate State Obs Action)
    (history : ObservationHistory Obs Action) :
    certificate.historyValue history =
      certificate.beliefValue (certificate.beliefFromHistory history) := by
  exact certificate.valueDependsOnlyOnBelief history

structure ReachableBeliefRefinement
    (State FineObs CoarseObs Action : Type) where
  fineBelief : ObservationHistory FineObs Action -> FiniteBelief State
  coarseBelief : ObservationHistory CoarseObs Action -> FiniteBelief State
  projectHistory : ObservationHistory FineObs Action ->
    ObservationHistory CoarseObs Action
  projectedBeliefAgrees :
    forall history : ObservationHistory FineObs Action,
      coarseBelief (projectHistory history) = fineBelief history

theorem observation_refinement_refines_reachable_beliefs
    {State FineObs CoarseObs Action : Type}
    (certificate :
      ReachableBeliefRefinement State FineObs CoarseObs Action)
    (history : ObservationHistory FineObs Action) :
    certificate.coarseBelief (certificate.projectHistory history) =
      certificate.fineBelief history := by
  exact certificate.projectedBeliefAgrees history

end Sequential
end Manuscript
end CausalObservationReproduction

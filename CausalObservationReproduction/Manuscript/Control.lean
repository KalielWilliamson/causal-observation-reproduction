import CausalObservationReproduction.Manuscript.Core

/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

structure FiniteDeterministicMDP (State Action : Type) where
  step : State -> Action -> State
  reward : State -> Action -> Rat

structure StateAbstraction (State AbstractState : Type) where
  map : State -> AbstractState

def abstractionObservationRegime {State AbstractState : Type}
    (abstraction : StateAbstraction State AbstractState) :
    ObservationRegime State AbstractState :=
  abstraction.map

def RewardRespectingAbstraction {State Action AbstractState : Type}
    (mdp : FiniteDeterministicMDP State Action)
    (abstraction : StateAbstraction State AbstractState) : Prop :=
  ∀ s t : State, ∀ action : Action,
    abstraction.map s = abstraction.map t ->
    mdp.reward s action = mdp.reward t action

def StepRespectingAbstraction {State Action AbstractState : Type}
    (mdp : FiniteDeterministicMDP State Action)
    (abstraction : StateAbstraction State AbstractState) : Prop :=
  ∀ s t : State, ∀ action : Action,
    abstraction.map s = abstraction.map t ->
    abstraction.map (mdp.step s action) =
      abstraction.map (mdp.step t action)

theorem reward_respecting_abstraction_identifies_one_step_reward_query
    {State Action AbstractState : Type}
    (mdp : FiniteDeterministicMDP State Action)
    (abstraction : StateAbstraction State AbstractState)
    (hReward : RewardRespectingAbstraction mdp abstraction)
    (action : Action) :
    Identifiable
      (abstractionObservationRegime abstraction)
      (fun state => mdp.reward state action) := by
  intro s t hTrace
  exact hReward s t action hTrace

theorem step_respecting_abstraction_identifies_next_observation_query
    {State Action AbstractState : Type}
    (mdp : FiniteDeterministicMDP State Action)
    (abstraction : StateAbstraction State AbstractState)
    (hStep : StepRespectingAbstraction mdp abstraction)
    (action : Action) :
    Identifiable
      (abstractionObservationRegime abstraction)
      (fun state => abstraction.map (mdp.step state action)) := by
  intro s t hTrace
  exact hStep s t action hTrace

end Manuscript
end CausalObservationReproduction

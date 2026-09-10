import CausalObservationReproduction.Formalism.Core

/-!
Finite sequential observation control.  This deliberately uses explicit finite
supports and rational kernels, so the statements are checkable certificates
rather than assumptions about an unbounded POMDP implementation.
-/

namespace CausalObservationReproduction.Formalism

structure FiniteSequentialDecisionProcess (State Action : Type) where
  stateSupport : List State
  actionSupport : List Action
  horizon : Nat
  transition : Nat -> State -> Action -> State -> Rat
  reward : Nat -> State -> Action -> Rat
  initial : State -> Rat

structure PolicyObservationHistory (Observation Action : Type) where
  observations : List Observation
  actions : List Action

abbrev SequentialPolicy (Observation Action : Type) :=
  PolicyObservationHistory Observation Action -> Action

structure LatentTrajectory (State Action : Type) where
  states : List State
  actions : List Action

def observeTrajectory {State Action Observation : Type}
    (observe : ObservationRegime State Observation)
    (trajectory : LatentTrajectory State Action) : PolicyObservationHistory Observation Action :=
  ⟨trajectory.states.map observe, trajectory.actions⟩

def projectObservationHistory {FineObservation CoarseObservation Action : Type}
    (project : FineObservation -> CoarseObservation)
    (history : PolicyObservationHistory FineObservation Action) : PolicyObservationHistory CoarseObservation Action :=
  ⟨history.observations.map project, history.actions⟩

def pullbackPolicy {FineObservation CoarseObservation Action : Type}
    (project : FineObservation -> CoarseObservation)
    (coarsePolicy : SequentialPolicy CoarseObservation Action) :
    SequentialPolicy FineObservation Action :=
  fun history => coarsePolicy (projectObservationHistory project history)

theorem observation_refinement_induces_history_refinement
    {State FineObservation CoarseObservation Action : Type}
    {fine : ObservationRegime State FineObservation}
    {coarse : ObservationRegime State CoarseObservation}
    (refinement : Refines fine coarse) :
    exists project : FineObservation -> CoarseObservation,
      forall trajectory : LatentTrajectory State Action,
        observeTrajectory coarse trajectory =
          projectObservationHistory project (observeTrajectory fine trajectory) := by
  rcases refinement with ⟨project, projects⟩
  refine ⟨project, ?_⟩
  rintro ⟨states, actions⟩
  unfold observeTrajectory projectObservationHistory
  change PolicyObservationHistory.mk (List.map coarse states) actions =
    PolicyObservationHistory.mk (List.map project (List.map fine states)) actions
  congr 1
  rw [List.map_map]
  apply List.map_congr_left
  intro state _
  exact projects state

theorem sequential_refinement_has_policy_pullback
    {State FineObservation CoarseObservation Action : Type}
    {fine : ObservationRegime State FineObservation}
    {coarse : ObservationRegime State CoarseObservation}
    (refinement : Refines fine coarse)
    (coarsePolicy : SequentialPolicy CoarseObservation Action) :
    exists finePolicy : SequentialPolicy FineObservation Action,
      forall trajectory : LatentTrajectory State Action,
        finePolicy (observeTrajectory fine trajectory) =
          coarsePolicy (observeTrajectory coarse trajectory) := by
  rcases observation_refinement_induces_history_refinement refinement with ⟨project, hproject⟩
  refine ⟨pullbackPolicy project coarsePolicy, ?_⟩
  intro trajectory
  unfold pullbackPolicy
  rw [← hproject trajectory]

end CausalObservationReproduction.Formalism

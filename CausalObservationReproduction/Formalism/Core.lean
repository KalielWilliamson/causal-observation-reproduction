/-!
Minimal finite observation formalism used by the novelty review.

This is a foundation/special-case layer.  It defines observation equivalence
and deterministic refinement; it does not claim a new value-of-information or
Blackwell theorem.
-/

namespace CausalObservationReproduction
namespace Formalism

abbrev ObservationMap (World Observation : Type) := World -> Observation

/-- A named observation regime, used by the sequential theory. -/
abbrev ObservationRegime (World Observation : Type) := ObservationMap World Observation

/-- Finite traces are the common representation of policy-visible histories. -/
abbrev Trace (Event : Type) := List Event

def ObservationEquivalent {World Observation : Type}
    (observe : ObservationMap World Observation) (x y : World) : Prop :=
  observe x = observe y

/-- The observation class containing a world. -/
def ObservationClass {World Observation : Type}
    (observe : ObservationMap World Observation) (x : World) : World -> Prop :=
  fun y => ObservationEquivalent observe x y

def SameObservationClass {World Observation : Type}
    (observe : ObservationMap World Observation) (x y : World) : Prop :=
  ObservationClass observe x = ObservationClass observe y

abbrev CausalQuery (World Target : Type) := World -> Target

def Identifiable {World Observation Target : Type}
    (observe : ObservationMap World Observation)
    (query : CausalQuery World Target) : Prop :=
  forall {x y : World}, ObservationEquivalent observe x y -> query x = query y

def Refines {World FineObservation CoarseObservation : Type}
    (fine : ObservationMap World FineObservation)
    (coarse : ObservationMap World CoarseObservation) : Prop :=
  exists project : FineObservation -> CoarseObservation,
    forall world : World, coarse world = project (fine world)

/-- A fine observation can simulate any decision rule written for a coarse one. -/
def SimulatesCoarseDecision {World FineObservation CoarseObservation Decision : Type}
    (fine : ObservationMap World FineObservation)
    (coarse : ObservationMap World CoarseObservation)
    (fineDecision : FineObservation -> Decision)
    (coarseDecision : CoarseObservation -> Decision) : Prop :=
  forall world, fineDecision (fine world) = coarseDecision (coarse world)

theorem observation_equivalence_blocks_distinct_decisions
    {World Observation Decision : Type}
    (observe : ObservationMap World Observation)
    (decide : Observation -> Decision)
    {x y : World}
    (equivalent : ObservationEquivalent observe x y) :
    decide (observe x) = decide (observe y) := by
  unfold ObservationEquivalent at equivalent
  rw [equivalent]

theorem refinement_preserves_identifiability
    {World FineObservation CoarseObservation Target : Type}
    {fine : ObservationMap World FineObservation}
    {coarse : ObservationMap World CoarseObservation}
    {query : CausalQuery World Target}
    (refinement : Refines fine coarse)
    (identifiable : Identifiable coarse query) :
    Identifiable fine query := by
  intro x y fineEquivalent
  rcases refinement with ⟨project, projects⟩
  apply identifiable
  unfold ObservationEquivalent at *
  rw [projects x, projects y, fineEquivalent]

theorem observation_equivalent_refl {World Observation : Type}
    (observe : ObservationMap World Observation) (x : World) :
    ObservationEquivalent observe x x := rfl

theorem observation_equivalent_symm {World Observation : Type}
    (observe : ObservationMap World Observation) {x y : World} :
    ObservationEquivalent observe x y -> ObservationEquivalent observe y x := by
  intro h
  exact h.symm

theorem observation_equivalent_trans {World Observation : Type}
    (observe : ObservationMap World Observation) {x y z : World} :
    ObservationEquivalent observe x y -> ObservationEquivalent observe y z ->
      ObservationEquivalent observe x z := by
  intro hxy hyz
  exact hxy.trans hyz

theorem refinement_simulates_coarse_decision
    {World FineObservation CoarseObservation Decision : Type}
    {fine : ObservationMap World FineObservation}
    {coarse : ObservationMap World CoarseObservation}
    (refinement : Refines fine coarse)
    (coarseDecision : CoarseObservation -> Decision) :
    exists fineDecision : FineObservation -> Decision,
      SimulatesCoarseDecision fine coarse fineDecision coarseDecision := by
  rcases refinement with ⟨project, projects⟩
  refine ⟨coarseDecision ∘ project, ?_⟩
  intro world
  simp only [Function.comp_apply]
  rw [← projects world]

end Formalism
end CausalObservationReproduction

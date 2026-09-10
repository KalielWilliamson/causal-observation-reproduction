import CausalObservationReproduction.Formalism.Core

/-! The complete policy-visible observation boundary.  A feedback channel is
policy-visible exactly through its declared projection; no hidden channel is
silently used by the decision rule. -/

namespace CausalObservationReproduction.Formalism

structure PolicyVisibleFeedback (Latent Feedback Visible : Type) where
  emit : Latent -> Feedback
  project : Feedback -> Visible

def visibleObservation {Latent Feedback Visible : Type}
    (kernel : PolicyVisibleFeedback Latent Feedback Visible) : Latent -> Visible :=
  kernel.project ∘ kernel.emit

def PolicyDependsOnlyOnVisibleFeedback {Latent Feedback Visible Action : Type}
    (kernel : PolicyVisibleFeedback Latent Feedback Visible)
    (policy : Feedback -> Action) : Prop :=
  exists visiblePolicy : Visible -> Action, policy = visiblePolicy ∘ kernel.project

theorem visible_equivalence_blocks_policy_difference
    {Latent Feedback Visible Action : Type}
    (kernel : PolicyVisibleFeedback Latent Feedback Visible)
    (policy : Feedback -> Action)
    (visible : PolicyDependsOnlyOnVisibleFeedback kernel policy)
    {x y : Latent}
    (sameVisible : visibleObservation kernel x = visibleObservation kernel y) :
    policy (kernel.emit x) = policy (kernel.emit y) := by
  rcases visible with ⟨visiblePolicy, rfl⟩
  exact congrArg visiblePolicy sameVisible

/-- Completeness is the explicit claim required before quotienting latent states
by visible observations: the policy factors through the declared boundary. -/
def CompletePolicyVisibleBoundary {Latent Feedback Visible Action : Type}
    (kernel : PolicyVisibleFeedback Latent Feedback Visible)
    (policy : Feedback -> Action) : Prop :=
  PolicyDependsOnlyOnVisibleFeedback kernel policy

end CausalObservationReproduction.Formalism

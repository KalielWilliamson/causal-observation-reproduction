import CausalObservationReproduction.Formalism.DecisionRelativeQuotient

namespace CausalObservationReproduction.Formalism

inductive DelayedContinuationWorld | left | right deriving DecidableEq
inductive DelayedContinuationAction | wait | commit deriving DecidableEq
inductive DelayedContinuationObservation | merged deriving DecidableEq

def delayedContinuationObservation : DelayedContinuationWorld -> DelayedContinuationObservation := fun _ => .merged

def delayedContinuationActionValue : DelayedContinuationWorld -> DelayedContinuationAction -> Rat
  | .left, .wait => 1
  | .right, .wait => -1
  | _, .commit => 0

theorem delayed_continuation_observation_aliases_opposite_values :
    delayedContinuationObservation .left = delayedContinuationObservation .right ∧
      delayedContinuationActionValue .left .wait = 1 ∧
      delayedContinuationActionValue .right .wait = -1 := by
  exact ⟨rfl, rfl, rfl⟩

/-- The delayed-continuation instance cannot be quotient-preserving for the
`wait` action: its sole observation class contains opposite action values. -/
theorem delayed_continuation_is_not_decision_relative :
    ¬ ∃ quotient : DecisionRelativeQuotient DelayedContinuationWorld DelayedContinuationAction DelayedContinuationObservation,
      quotient.observe = delayedContinuationObservation ∧ quotient.actionValue = delayedContinuationActionValue := by
  rintro ⟨quotient, hObserve, hValue⟩
  have same : quotient.observe .left = quotient.observe .right := by
    rw [hObserve]
  have factor := quotient.factors same .wait
  rw [hValue, delayedContinuationActionValue, delayedContinuationActionValue] at factor
  cases factor

end CausalObservationReproduction.Formalism

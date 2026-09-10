import CausalObservationReproduction.Formalism.CausalAbstraction

namespace CausalObservationReproduction.Formalism

structure DecisionRelativeQuotient (World Action Observation : Type) where
  observe : World -> Observation
  actionValue : World -> Action -> Rat
  factors : forall {x y}, observe x = observe y -> forall action, actionValue x action = actionValue y action

def NetAcquireValue (purchaseCost : Rat) (valueWithout valueWith : Rat) : Rat :=
  valueWith - valueWithout - purchaseCost

def AcquireIsJustified (purchaseCost valueWithout valueWith : Rat) : Prop :=
  0 < NetAcquireValue purchaseCost valueWithout valueWith

theorem quotient_preserves_action_value_sign
    {World Action Observation : Type}
    (quotient : DecisionRelativeQuotient World Action Observation)
    {x y : World} (sameObservation : quotient.observe x = quotient.observe y) (action : Action) :
    (0 < quotient.actionValue x action) ↔ (0 < quotient.actionValue y action) := by
  rw [quotient.factors sameObservation action]

/-- Number of full action-value evaluations before quotienting by policy-visible
observations.  The supports make the accounting assumption explicit. -/
def FullActionValueEvaluationCount (worldCount actionCount : Nat) : Nat :=
  worldCount * actionCount

def QuotientActionValueEvaluationCount (observationClassCount actionCount : Nat) : Nat :=
  observationClassCount * actionCount

theorem quotient_evaluation_count_le (observationClassCount worldCount actionCount : Nat)
    (classesBound : observationClassCount <= worldCount) :
    QuotientActionValueEvaluationCount observationClassCount actionCount <=
      FullActionValueEvaluationCount worldCount actionCount := by
  simp only [QuotientActionValueEvaluationCount, FullActionValueEvaluationCount]
  exact Nat.mul_le_mul_right actionCount classesBound

theorem quotient_evaluation_count_strict (observationClassCount worldCount actionCount : Nat)
    (actionPositive : 0 < actionCount) (classesStrict : observationClassCount < worldCount) :
    QuotientActionValueEvaluationCount observationClassCount actionCount <
      FullActionValueEvaluationCount worldCount actionCount := by
  simp only [QuotientActionValueEvaluationCount, FullActionValueEvaluationCount]
  exact Nat.mul_lt_mul_of_pos_right classesStrict actionPositive

end CausalObservationReproduction.Formalism

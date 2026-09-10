import CausalObservationReproduction.Manuscript.CausalAbstraction
import CausalObservationReproduction.Manuscript.Sequential.ObservabilityControl

/-
Decision-relative quotient gate.

This is deliberately a bounded result. It says what a quotient must preserve
for an acquire-versus-abstain choice to be preserved, derives that preservation
for declared interventional action values under explicit causal-abstraction
conditions, and gives only a symbolic count reduction for action-value
evaluations. It does not establish arbitrary sequential value factorization,
wall-clock, training, or sample-complexity claims.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

structure DecisionRelativeQuotient (History Quotient : Type) where
  classOf : History -> Quotient

def NetAcquireValue {History : Type} (acquireValue abstainValue : History -> Rat) :
    History -> Rat :=
  fun history => acquireValue history - abstainValue history

def AcquireIsJustified {History : Type}
    (acquireValue abstainValue : History -> Rat) (history : History) : Prop :=
  0 < NetAcquireValue acquireValue abstainValue history

structure ActionValueFactorsThroughQuotient {History Quotient : Type}
    (quotient : DecisionRelativeQuotient History Quotient)
    (acquireValue abstainValue : History -> Rat) where
  quotientAcquireValue : Quotient -> Rat
  quotientAbstainValue : Quotient -> Rat
  acquireFactors : forall history : History,
    acquireValue history = quotientAcquireValue (quotient.classOf history)
  abstainFactors : forall history : History,
    abstainValue history = quotientAbstainValue (quotient.classOf history)

theorem quotient_preserves_acquire_abstain_sign
    {History Quotient : Type}
    (quotient : DecisionRelativeQuotient History Quotient)
    (acquireValue abstainValue : History -> Rat)
    (certificate :
      ActionValueFactorsThroughQuotient quotient acquireValue abstainValue)
    (left right : History)
    (hClass : quotient.classOf left = quotient.classOf right) :
    AcquireIsJustified acquireValue abstainValue left <->
      AcquireIsJustified acquireValue abstainValue right := by
  unfold AcquireIsJustified NetAcquireValue
  rw [certificate.acquireFactors, certificate.abstainFactors,
    certificate.acquireFactors, certificate.abstainFactors, hClass]

/-
SCM-derived sufficient condition.

This is deliberately narrower than a theorem about arbitrary sequential value
functions.  It covers action values obtained by applying an action-specific
reward map to a declared interventional target.  The existing causal-abstraction
obligations then imply that every declared action value is constant on an
abstraction class, rather than taking that equality as an unexplained
certificate.
-/

def InterventionalActionValue
    {World Intervention Variable Value Action : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (actionIntervention : Action -> Intervention)
    (actionNode : Action -> Variable)
    (actionReward : Action -> Value -> Rat)
    (action : Action) : World -> Rat :=
  fun world => actionReward action
    (InterventionalTarget model (actionIntervention action) (actionNode action) world)

theorem causal_abstraction_preserves_interventional_action_value
    {World Intervention Variable Value AbstractWorld Action : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (actionIntervention : Action -> Intervention)
    (actionNode : Action -> Variable)
    (actionReward : Action -> Value -> Rat)
    (action : Action)
    (left right : World)
    (hClass : abstraction.map left = abstraction.map right) :
    InterventionalActionValue model actionIntervention actionNode actionReward action left =
      InterventionalActionValue model actionIntervention actionNode actionReward action right := by
  unfold InterventionalActionValue
  apply congrArg (actionReward action)
  apply causal_abstraction_identifies_interventional_target
    model abstraction hIntervention hStructural
    (actionIntervention action) (actionNode action)
  exact hClass

/-- Explicit causal-to-quotient bridge for two declared interventional action
values.  Values outside the abstraction image are assigned zero only to make
the quotient functions total; they are never consulted by a concrete world. -/
theorem causal_abstraction_action_values_factor_through_quotient
    {World Intervention Variable Value AbstractWorld Action : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (actionIntervention : Action -> Intervention)
    (actionNode : Action -> Variable)
    (actionReward : Action -> Value -> Rat)
    (acquire abstain : Action) :
    Nonempty (ActionValueFactorsThroughQuotient
      { classOf := abstraction.map }
      (InterventionalActionValue model actionIntervention actionNode actionReward acquire)
      (InterventionalActionValue model actionIntervention actionNode actionReward abstain)) := by
  classical
  refine ⟨?_⟩
  refine
    { quotientAcquireValue := fun abstractWorld =>
        dite (∃ world : World, abstraction.map world = abstractWorld)
          (fun h => InterventionalActionValue model actionIntervention actionNode actionReward
            acquire (Classical.choose h))
          (fun _ => 0)
      quotientAbstainValue := fun abstractWorld =>
        dite (∃ world : World, abstraction.map world = abstractWorld)
          (fun h => InterventionalActionValue model actionIntervention actionNode actionReward
            abstain (Classical.choose h))
          (fun _ => 0)
      acquireFactors := ?_
      abstainFactors := ?_ }
  · intro world
    have hExists : ∃ representative : World, abstraction.map representative = abstraction.map world :=
      ⟨world, rfl⟩
    rw [dif_pos hExists]
    exact causal_abstraction_preserves_interventional_action_value
      model abstraction hIntervention hStructural actionIntervention actionNode actionReward
      acquire world (Classical.choose hExists)
      (by rw [Classical.choose_spec hExists])
  · intro world
    have hExists : ∃ representative : World, abstraction.map representative = abstraction.map world :=
      ⟨world, rfl⟩
    rw [dif_pos hExists]
    exact causal_abstraction_preserves_interventional_action_value
      model abstraction hIntervention hStructural actionIntervention actionNode actionReward
      abstain world (Classical.choose hExists)
      (by rw [Classical.choose_spec hExists])

theorem causal_abstraction_preserves_acquire_abstain_sign
    {World Intervention Variable Value AbstractWorld Action : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (actionIntervention : Action -> Intervention)
    (actionNode : Action -> Variable)
    (actionReward : Action -> Value -> Rat)
    (acquire abstain : Action)
    (left right : World)
    (hClass : abstraction.map left = abstraction.map right) :
    AcquireIsJustified
        (InterventionalActionValue model actionIntervention actionNode actionReward acquire)
        (InterventionalActionValue model actionIntervention actionNode actionReward abstain)
        left <->
      AcquireIsJustified
        (InterventionalActionValue model actionIntervention actionNode actionReward acquire)
        (InterventionalActionValue model actionIntervention actionNode actionReward abstain)
        right := by
  unfold AcquireIsJustified NetAcquireValue
  rw [causal_abstraction_preserves_interventional_action_value
        model abstraction hIntervention hStructural actionIntervention actionNode actionReward
        acquire left right hClass,
      causal_abstraction_preserves_interventional_action_value
        model abstraction hIntervention hStructural actionIntervention actionNode actionReward
        abstain left right hClass]

def FullActionValueEvaluationCount (historyCount probeCount : Nat) : Nat :=
  historyCount * probeCount

def QuotientActionValueEvaluationCount (quotientCount probeCount : Nat) : Nat :=
  quotientCount * probeCount

theorem quotient_action_value_evaluation_count_not_greater
    (historyCount quotientCount probeCount : Nat)
    (hQuotientNoLarger : quotientCount <= historyCount) :
    QuotientActionValueEvaluationCount quotientCount probeCount <=
      FullActionValueEvaluationCount historyCount probeCount := by
  exact Nat.mul_le_mul_right probeCount hQuotientNoLarger

theorem quotient_action_value_evaluation_count_strictly_smaller
    (historyCount quotientCount probeCount : Nat)
    (hStrictCompression : quotientCount < historyCount)
    (hProbe : 0 < probeCount) :
    QuotientActionValueEvaluationCount quotientCount probeCount <
      FullActionValueEvaluationCount historyCount probeCount := by
  exact Nat.mul_lt_mul_of_pos_right hStrictCompression hProbe

theorem quotient_action_value_evaluation_count_equal_without_compression
    (historyCount probeCount : Nat) :
    QuotientActionValueEvaluationCount historyCount probeCount =
      FullActionValueEvaluationCount historyCount probeCount := by
  rfl

/-
The causal-to-efficiency bridge used by the paper.

The causal premises establish that abstraction classes preserve the declared
acquire-versus-abstain decision.  Strict compression is a separate, explicit
finite accounting premise: the theorem does not infer cardinalities, runtime,
or sample complexity from causal structure alone.
-/
theorem causal_abstraction_yields_decision_preserving_strict_evaluation_reduction
    {World Intervention Variable Value AbstractWorld Action : Type}
    (model : FiniteDeterministicCausalModel World Intervention Variable Value)
    (abstraction : CausalAbstraction World AbstractWorld)
    (hIntervention : InterventionRespectingAbstraction model abstraction)
    (hStructural : StructuralAssignmentRespectingAbstraction model abstraction)
    (actionIntervention : Action -> Intervention)
    (actionNode : Action -> Variable)
    (actionReward : Action -> Value -> Rat)
    (acquire abstain : Action)
    (left right : World)
    (hClass : abstraction.map left = abstraction.map right)
    (historyCount quotientCount probeCount : Nat)
    (hStrictCompression : quotientCount < historyCount)
    (hProbe : 0 < probeCount) :
    (AcquireIsJustified
        (InterventionalActionValue model actionIntervention actionNode actionReward acquire)
        (InterventionalActionValue model actionIntervention actionNode actionReward abstain)
        left <->
      AcquireIsJustified
        (InterventionalActionValue model actionIntervention actionNode actionReward acquire)
        (InterventionalActionValue model actionIntervention actionNode actionReward abstain)
        right) /\
      QuotientActionValueEvaluationCount quotientCount probeCount <
        FullActionValueEvaluationCount historyCount probeCount := by
  constructor
  · exact causal_abstraction_preserves_acquire_abstain_sign
      model abstraction hIntervention hStructural actionIntervention actionNode actionReward
      acquire abstain left right hClass
  · exact quotient_action_value_evaluation_count_strictly_smaller
      historyCount quotientCount probeCount hStrictCompression hProbe

/-
Novelty-gate interpretation: `ActionValueFactorsThroughQuotient` is the
nontrivial assumption. Supplying it as an unexplained equality of net values
would make the sign-preservation theorem a restatement, not a separation.

`causal_abstraction_preserves_acquire_abstain_sign` supplies one concrete,
checkable route to the factorization for declared interventional action values:
both intervention- and structural-assignment-respecting abstraction are
required. It does not yet derive arbitrary learned sequential continuation
values, so it is a sufficient-condition slice rather than a general theorem.
`causal_abstraction_yields_decision_preserving_strict_evaluation_reduction`
packages that decision-preservation result with a separately declared strict
compression premise. It is the formal causal-to-efficiency bridge, not a claim
that causal structure alone creates compression.
The strict evaluation-count result requires both strict compression and at
least one probe.  It is an accounting theorem for a fixed probe set, not an
asymptotic runtime, construction-cost, or sample-complexity claim.
-/

end Sequential
end Manuscript
end CausalObservationReproduction

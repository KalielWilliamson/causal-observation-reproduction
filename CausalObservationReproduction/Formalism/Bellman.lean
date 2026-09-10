import CausalObservationReproduction.Formalism.SequentialCore

/-!
Bellman-style finite-horizon certificates.

The recursive quantities are rational tables.  A certificate records local
Bellman inequalities and the correspondence between the initial value and a
finite policy value, avoiding a nonconstructive supremum.
-/

namespace CausalObservationReproduction.Formalism

abbrev SequentialValueTable (Information : Type) := Nat -> Information -> Rat

abbrev SequentialActionValueTable (Information Action : Type) :=
  Nat -> Information -> Action -> Rat

def ContinuationOptimalAction {Information Action : Type}
    (qValue : SequentialActionValueTable Information Action)
    (time : Nat)
    (information : Information)
    (action : Action) : Prop :=
  forall candidate : Action, qValue time information candidate <= qValue time information action

/-- A finite certificate connects local action-value bounds to an attained,
globally optimal policy value at the initial information state. -/
structure BellmanOptimalityCertificate (Information Action Policy : Type) where
  value : SequentialValueTable Information
  qValue : SequentialActionValueTable Information Action
  policyValue : Policy -> Rat
  initialInformation : Information
  optimalPolicy : Policy
  actionUpperBound :
    forall time information action, qValue time information action <= value time information
  actionWitness :
    forall time information, exists action : Action, value time information = qValue time information action
  optimalPolicyValue : policyValue optimalPolicy = value 0 initialInformation
  policyUpperBound : forall policy : Policy, policyValue policy <= value 0 initialInformation

theorem bellman_value_agrees_with_optimal_policy_value
    {Information Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Information Action Policy) :
    certificate.policyValue certificate.optimalPolicy =
      certificate.value 0 certificate.initialInformation :=
  certificate.optimalPolicyValue

theorem bellman_value_bounds_policy_value
    {Information Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Information Action Policy)
    (policy : Policy) :
    certificate.policyValue policy <= certificate.value 0 certificate.initialInformation :=
  certificate.policyUpperBound policy

theorem bellman_action_value_is_locally_bounded_by_value
    {Information Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Information Action Policy)
    (time : Nat)
    (information : Information)
    (action : Action) :
    certificate.qValue time information action <= certificate.value time information :=
  certificate.actionUpperBound time information action

end CausalObservationReproduction.Formalism

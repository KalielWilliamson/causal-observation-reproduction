import CausalObservationReproduction.Manuscript.Sequential.Core

/-
Bellman-style finite-horizon certificates.

The recursive quantities are ordinary rational tables.  A certificate records
the local Bellman inequalities and a value-policy correspondence, avoiding
nonconstructive suprema while still connecting the Bellman table to finite
policy value.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

abbrev SequentialValueTable (Info : Type) := Nat -> Info -> Rat

abbrev SequentialActionValueTable (Info Action : Type) :=
  Nat -> Info -> Action -> Rat

def ContinuationOptimalAction {Info Action : Type}
    (q : SequentialActionValueTable Info Action)
    (t : Nat)
    (info : Info)
    (action : Action) : Prop :=
  forall candidate : Action, q t info candidate <= q t info action

structure BellmanOptimalityCertificate (Info Action Policy : Type) where
  value : SequentialValueTable Info
  qValue : SequentialActionValueTable Info Action
  policyValue : Policy -> Rat
  initialInfo : Info
  optimalPolicy : Policy
  actionUpperBound :
    forall t info action, qValue t info action <= value t info
  actionWitness :
    forall t info, exists action : Action, value t info = qValue t info action
  optimalPolicyValue :
    policyValue optimalPolicy = value 0 initialInfo
  policyUpperBound :
    forall policy : Policy, policyValue policy <= value 0 initialInfo

theorem bellman_value_agrees_with_optimal_policy_value
    {Info Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Info Action Policy) :
    certificate.policyValue certificate.optimalPolicy =
      certificate.value 0 certificate.initialInfo := by
  exact certificate.optimalPolicyValue

theorem bellman_value_bounds_policy_value
    {Info Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Info Action Policy)
    (policy : Policy) :
    certificate.policyValue policy <=
      certificate.value 0 certificate.initialInfo := by
  exact certificate.policyUpperBound policy

theorem bellman_action_value_is_locally_bounded_by_value
    {Info Action Policy : Type}
    (certificate : BellmanOptimalityCertificate Info Action Policy)
    (t : Nat)
    (info : Info)
    (action : Action) :
    certificate.qValue t info action <= certificate.value t info := by
  exact certificate.actionUpperBound t info action

end Sequential
end Manuscript
end CausalObservationReproduction

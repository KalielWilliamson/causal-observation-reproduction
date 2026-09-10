import CausalObservationReproduction.Manuscript.Sequential.ObservabilityControl
import Mathlib.Algebra.Order.Ring.Rat

/-
Finite contracts for inductive biases over information actions.

The development deliberately describes the *semantics* of a bias rather than
the convergence of a neural optimizer.  A learned GNN, a hand-written value
rule, and a group-relative policy optimizer can all implement the same public
state map and action-ranking contract.  Scorer-only causal worlds are used to
certify invariance and sensitivity; they are not inputs to the policy state.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential

def InformationActionRanksAtLeast
    {PublicState InformationAction : Type}
    (score : PublicState -> InformationAction -> Rat)
    (state : PublicState)
    (preferred alternative : InformationAction) : Prop :=
  score state alternative <= score state preferred

structure PublicObservationBias (History PublicState InformationAction : Type) where
  publicState : History -> PublicState
  score : PublicState -> InformationAction -> Rat

def BiasIsObservationLocal
    {History PublicState InformationAction : Type}
    (bias : PublicObservationBias History PublicState InformationAction) : Prop :=
  forall left right : History,
    bias.publicState left = bias.publicState right ->
      forall action : InformationAction,
        bias.score (bias.publicState left) action =
          bias.score (bias.publicState right) action

theorem public_observation_bias_is_observation_local
    {History PublicState InformationAction : Type}
    (bias : PublicObservationBias History PublicState InformationAction) :
    BiasIsObservationLocal bias := by
  intro left right hState action
  rw [hState]

structure CausalObservationBias (History PublicState InformationAction : Type)
    extends PublicObservationBias History PublicState InformationAction where
  nuisanceEquivalent : History -> History -> Prop
  queryRelevantContrast : History -> History -> Prop
  nuisanceInvariant : forall left right : History,
    nuisanceEquivalent left right -> publicState left = publicState right

theorem nuisance_equivalent_histories_have_same_information_action_ranking
    {History PublicState InformationAction : Type}
    (bias : CausalObservationBias History PublicState InformationAction)
    {left right : History}
    (hNuisance : bias.nuisanceEquivalent left right)
    (preferred alternative : InformationAction) :
    InformationActionRanksAtLeast bias.score (bias.publicState left) preferred alternative <->
      InformationActionRanksAtLeast bias.score (bias.publicState right) preferred alternative := by
  have hState : bias.publicState left = bias.publicState right :=
    bias.nuisanceInvariant left right hNuisance
  rw [hState]

def NetInformationActionValue
    (continuationValue acquisitionCost : Rat) : Rat :=
  continuationValue - acquisitionCost

/-
The group baseline models GRPO-style relative credit assignment.  A baseline
shared by every candidate in a matched causal group affects variance/credit,
not the induced ranking of information actions.
-/
def CausalQueryRelativeScore
    (continuationValue acquisitionCost groupBaseline : Rat) : Rat :=
  NetInformationActionValue continuationValue acquisitionCost - groupBaseline

theorem shared_group_baseline_preserves_information_action_order
    (leftContinuation leftCost rightContinuation rightCost groupBaseline : Rat) :
    CausalQueryRelativeScore leftContinuation leftCost groupBaseline <=
        CausalQueryRelativeScore rightContinuation rightCost groupBaseline <->
      NetInformationActionValue leftContinuation leftCost <=
        NetInformationActionValue rightContinuation rightCost := by
  simp only [CausalQueryRelativeScore, NetInformationActionValue]
  exact sub_le_sub_iff_right groupBaseline

structure CausalBiasAdmissibilityCertificate
    {History PublicState InformationAction : Type}
    (bias : CausalObservationBias History PublicState InformationAction) where
  observationLocal : BiasIsObservationLocal bias.toPublicObservationBias
  relevantSensitivityWitness :
    exists left right : History,
      bias.queryRelevantContrast left right /\
        exists preferred alternative : InformationAction,
          Not (InformationActionRanksAtLeast bias.score (bias.publicState left) preferred alternative) /\
          Not (InformationActionRanksAtLeast bias.score (bias.publicState right) alternative preferred)

theorem admissible_causal_bias_preserves_nuisance_action_order
    {History PublicState InformationAction : Type}
    {bias : CausalObservationBias History PublicState InformationAction}
    (_certificate : CausalBiasAdmissibilityCertificate bias)
    {left right : History}
    (hNuisance : bias.nuisanceEquivalent left right)
    (preferred alternative : InformationAction) :
    InformationActionRanksAtLeast bias.score (bias.publicState left) preferred alternative <->
      InformationActionRanksAtLeast bias.score (bias.publicState right) preferred alternative := by
  exact nuisance_equivalent_histories_have_same_information_action_ranking
    bias hNuisance preferred alternative

end Sequential
end Manuscript
end CausalObservationReproduction

import CausalObservationReproduction.Manuscript.Core

/-
Causal observation regime formalism for CausalObservationReproduction.

This module is part of the research runtime validation surface. It intentionally
uses finite, abstract Lean objects rather than formalizing the Python runtime,
SCMs, MDPs, POMDPs, or the full Blackwell literature.
-/

namespace CausalObservationReproduction
namespace Manuscript

abbrev TransferObservationRegime (World Policy Protocol Obs : Type) :=
  ObservationRegime (TransferInstance World Policy Protocol) Obs

abbrev TransferTarget (World Policy Protocol Target : Type) :=
  TransferInstance World Policy Protocol -> Target

def TransferIdentifiable {World Policy Protocol Obs Target : Type}
    (O : TransferObservationRegime World Policy Protocol Obs)
    (q : TransferTarget World Policy Protocol Target) : Prop :=
  Identifiable O q

theorem transfer_identifiable_iff_constant_on_observation_classes
    {World Policy Protocol Obs Target : Type}
    (O : TransferObservationRegime World Policy Protocol Obs)
    (q : TransferTarget World Policy Protocol Target) :
    TransferIdentifiable O q ↔
      ∀ {x y : TransferInstance World Policy Protocol},
        TraceEquivalent O x y -> q x = q y := by
  rfl

theorem transfer_refinement_preserves_identifiability
    {World Policy Protocol FineObs CoarseObs Target : Type}
    {fine : TransferObservationRegime World Policy Protocol FineObs}
    {coarse : TransferObservationRegime World Policy Protocol CoarseObs}
    {q : TransferTarget World Policy Protocol Target}
    (hRefines : Refines fine coarse)
    (hIdentifiable : TransferIdentifiable coarse q) :
    TransferIdentifiable fine q := by
  exact refinement_preserves_identifiability hRefines hIdentifiable

theorem transfer_trace_equivalent_different_outcome_not_identifiable
    {World Policy Protocol Obs Target : Type}
    (O : TransferObservationRegime World Policy Protocol Obs)
    (q : TransferTarget World Policy Protocol Target)
    {x y : TransferInstance World Policy Protocol}
    (hTrace : TraceEquivalent O x y)
    (hTarget : q x ≠ q y) :
    ¬ TransferIdentifiable O q := by
  exact trace_equivalent_different_target_not_identifiable O q hTrace hTarget

end Manuscript
end CausalObservationReproduction

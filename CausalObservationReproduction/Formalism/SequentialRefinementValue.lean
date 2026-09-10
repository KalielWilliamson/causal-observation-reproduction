import CausalObservationReproduction.Formalism.Bellman

namespace CausalObservationReproduction.Formalism

/-- An optimization problem exposes its best attainable sequential value. -/
structure SequentialValueOptimization (Policy : Type) where
  value : Policy -> Rat
  optimalValue : Rat
  upperBound : forall policy, value policy <= optimalValue
  attained : exists policy, value policy = optimalValue

def SequentialRefinementGain {CoarsePolicy FinePolicy : Type}
    (coarse : SequentialValueOptimization CoarsePolicy)
    (fine : SequentialValueOptimization FinePolicy) : Rat :=
  fine.optimalValue - coarse.optimalValue

def NetSequentialRefinementValue {CoarsePolicy FinePolicy : Type}
    (coarse : SequentialValueOptimization CoarsePolicy)
    (fine : SequentialValueOptimization FinePolicy) (cost : Rat) : Rat :=
  SequentialRefinementGain coarse fine - cost

def SequentialCostJustifiedRefinement {CoarsePolicy FinePolicy : Type}
    (coarse : SequentialValueOptimization CoarsePolicy)
    (fine : SequentialValueOptimization FinePolicy) (cost : Rat) : Prop :=
  0 < NetSequentialRefinementValue coarse fine cost

/-- A policy pullback that preserves each coarse policy's value proves weak
monotonicity of the refined optimum. -/
theorem refinement_weakly_increases_optimal_sequential_value
    {CoarsePolicy FinePolicy : Type}
    (coarse : SequentialValueOptimization CoarsePolicy)
    (fine : SequentialValueOptimization FinePolicy)
    (lift : CoarsePolicy -> FinePolicy)
    (preserves : forall policy, fine.value (lift policy) = coarse.value policy) :
    coarse.optimalValue <= fine.optimalValue := by
  rcases coarse.attained with ⟨policy, hpolicy⟩
  rw [← hpolicy, ← preserves]
  exact fine.upperBound (lift policy)

/-- A concrete strict-improvement certificate, avoiding an unjustified claim
that every refinement is strictly valuable. -/
structure StrictSequentialRefinementCertificate (CoarsePolicy FinePolicy : Type) where
  coarse : SequentialValueOptimization CoarsePolicy
  fine : SequentialValueOptimization FinePolicy
  witness : FinePolicy
  strictlyBetter : coarse.optimalValue < fine.value witness
  optimalStrictlyImproves : coarse.optimalValue < fine.optimalValue

theorem strict_certificate_implies_positive_gain {CoarsePolicy FinePolicy : Type}
    (certificate : StrictSequentialRefinementCertificate CoarsePolicy FinePolicy) :
    0 < SequentialRefinementGain certificate.coarse certificate.fine := by
  unfold SequentialRefinementGain
  apply (Rat.lt_iff_sub_pos certificate.coarse.optimalValue certificate.fine.optimalValue).mp
  exact certificate.optimalStrictlyImproves

theorem strict_certificate_justifies_cost {CoarsePolicy FinePolicy : Type}
    (certificate : StrictSequentialRefinementCertificate CoarsePolicy FinePolicy)
    {cost : Rat} (costBelowGain : cost < SequentialRefinementGain certificate.coarse certificate.fine) :
    SequentialCostJustifiedRefinement certificate.coarse certificate.fine cost := by
  unfold SequentialCostJustifiedRefinement NetSequentialRefinementValue
  exact (Rat.lt_iff_sub_pos cost (SequentialRefinementGain certificate.coarse certificate.fine)).mp costBelowGain

end CausalObservationReproduction.Formalism

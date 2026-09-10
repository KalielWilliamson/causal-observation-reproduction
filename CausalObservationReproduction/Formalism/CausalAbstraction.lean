import CausalObservationReproduction.Formalism.Core

namespace CausalObservationReproduction.Formalism

/-- A bounded deterministic SCM used only for the finite abstraction theorem. -/
structure FiniteDeterministicCausalModel (Variable Value Intervention : Type) where
  variables : List Variable
  structuralValue : Variable -> Intervention -> Value

structure CausalAbstraction (FineVariable FineValue CoarseVariable CoarseValue Intervention : Type) where
  mapVariable : FineVariable -> CoarseVariable
  mapValue : FineValue -> CoarseValue
  mapIntervention : Intervention -> Intervention

end CausalObservationReproduction.Formalism

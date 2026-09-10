import CausalObservationReproduction.Manuscript.Sequential.Belief
import CausalObservationReproduction.Manuscript.Sequential.VariationalBridge

/-
Executable finite examples for sequential observability value.

The examples use rational optimal-value certificates.  They are intentionally
tiny so Lean can verify the value equalities/inequalities directly.
-/

namespace CausalObservationReproduction
namespace Manuscript
namespace Sequential
namespace Examples

inductive HiddenState where
  | left
  | right
  deriving Repr, DecidableEq

inductive EnvAction where
  | continue
  | chooseLeft
  | chooseRight
  deriving Repr, DecidableEq

inductive CoarseHiddenObs where
  | hidden
  deriving Repr, DecidableEq

inductive FineHiddenObs where
  | seesLeft
  | seesRight
  deriving Repr, DecidableEq

def coarseHiddenObservation (_state : HiddenState) : CoarseHiddenObs :=
  CoarseHiddenObs.hidden

def fineHiddenObservation : HiddenState -> FineHiddenObs
  | HiddenState.left => FineHiddenObs.seesLeft
  | HiddenState.right => FineHiddenObs.seesRight

def fineToCoarse (_obs : FineHiddenObs) : CoarseHiddenObs :=
  CoarseHiddenObs.hidden

theorem example_refinement :
    Refines fineHiddenObservation coarseHiddenObservation := by
  exact ⟨fineToCoarse, by intro state; cases state <;> rfl⟩

def noValueCoarseOptimal : Rat := 2

def noValueFineOptimal : Rat := 2

example : noValueFineOptimal = noValueCoarseOptimal := by
  native_decide

#eval noValueFineOptimal == noValueCoarseOptimal

def strictCoarseOptimal : Rat := 1

def strictFineOptimal : Rat := 2

example : strictCoarseOptimal < strictFineOptimal := by
  native_decide

#eval strictCoarseOptimal < strictFineOptimal

def noValueSequentialGain : Rat :=
  SequentialRefinementGain noValueFineOptimal noValueCoarseOptimal

def strictSequentialGain : Rat :=
  SequentialRefinementGain strictFineOptimal strictCoarseOptimal

example : noValueSequentialGain = 0 := by
  native_decide

example : strictSequentialGain = 1 := by
  native_decide

def strictLeftHistory : ObservationHistory FineHiddenObs EnvAction where
  initial := FineHiddenObs.seesLeft
  steps := [(EnvAction.continue, FineHiddenObs.seesLeft)]

def strictRightHistory : ObservationHistory FineHiddenObs EnvAction where
  initial := FineHiddenObs.seesRight
  steps := [(EnvAction.continue, FineHiddenObs.seesRight)]

def strictCoarseHistory : ObservationHistory CoarseHiddenObs EnvAction where
  initial := CoarseHiddenObs.hidden
  steps := [(EnvAction.continue, CoarseHiddenObs.hidden)]

def leftContinuationValue : EnvAction -> Rat
  | EnvAction.chooseLeft => 2
  | EnvAction.chooseRight => 0
  | EnvAction.continue => 0

def rightContinuationValue : EnvAction -> Rat
  | EnvAction.chooseLeft => 0
  | EnvAction.chooseRight => 2
  | EnvAction.continue => 0

def strictPair :
    ContinuationDecisionRelevantPair
      FineHiddenObs CoarseHiddenObs EnvAction where
  t := 1
  coarseHistory := strictCoarseHistory
  fineLeftHistory := strictLeftHistory
  fineRightHistory := strictRightHistory
  leftAction := EnvAction.chooseLeft
  rightAction := EnvAction.chooseRight
  fallbackAction := EnvAction.continue
  qLeft := leftContinuationValue
  qRight := rightContinuationValue
  equalCoarseHistories := rfl
  fineSeparated := by
    intro h
    cases h
  leftOptimal := by
    intro action
    cases action <;> native_decide
  rightOptimal := by
    intro action
    cases action <;> native_decide
  disjointOptimalSets := by
    intro action hLeft hRight
    cases action
    · have hBad : Not (leftContinuationValue EnvAction.chooseLeft <=
          leftContinuationValue EnvAction.continue) := by
        native_decide
      exact hBad (hLeft EnvAction.chooseLeft)
    · have hBad : Not (rightContinuationValue EnvAction.chooseRight <=
          rightContinuationValue EnvAction.chooseLeft) := by
        native_decide
      exact hBad (hRight EnvAction.chooseRight)
    · have hBad : Not (leftContinuationValue EnvAction.chooseLeft <=
          leftContinuationValue EnvAction.chooseRight) := by
        native_decide
      exact hBad (hLeft EnvAction.chooseLeft)
  leftReachabilityPositive := True
  rightReachabilityPositive := True
  actionValueGapPositive := True

def strictSequentialCertificate :
    StrictSequentialRefinementCertificate
      FineHiddenObs CoarseHiddenObs EnvAction
      strictFineOptimal strictCoarseOptimal where
  pair := strictPair
  positiveReachabilityAssumptions := by
    constructor <;> trivial
  positiveGapAssumption := by
    trivial
  expectedStrictGain := by native_decide

example :
    strictCoarseOptimal < strictFineOptimal := by
  exact decision_relevant_history_refinement_has_strict_sequential_value
    strictSequentialCertificate

def probeCostBelowThreshold : Rat := 1 / 2

def probeCostAboveThreshold : Rat := 3 / 2

def probeVoi : Rat := strictSequentialGain

example : probeCostBelowThreshold < probeVoi := by
  native_decide

example : Not (probeCostAboveThreshold < probeVoi) := by
  native_decide

end Examples
end Sequential
end Manuscript
end CausalObservationReproduction

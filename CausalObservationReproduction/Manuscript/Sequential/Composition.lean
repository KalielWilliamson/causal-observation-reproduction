import Mathlib.Algebra.Order.Field.Rat
import CausalObservationReproduction.Manuscript.Sequential.CausalComputeGate

namespace CausalObservationReproduction.Manuscript.Sequential

theorem composing_nonnegative_costs_has_nonnegative_total_cost (a b : Rat)
    (ha : 0 <= a) (hb : 0 <= b) : 0 <= a + b :=
  add_nonneg ha hb

theorem increasing_compute_cost_cannot_increase_net_advantage (v d c extra : Rat)
    (h : 0 <= extra) :
    NetOptionAdvantage v d (c + extra) <= NetOptionAdvantage v d c := by
  dsimp [NetOptionAdvantage]
  exact sub_le_sub_left (le_add_of_nonneg_right h) _

end CausalObservationReproduction.Manuscript.Sequential

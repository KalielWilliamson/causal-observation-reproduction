# Lean sequential separation extension audit

This page describes the independently strengthened `Formalism` modules, not the
namespace-renamed source corpus used by the manuscript. The exact manuscript
theory lives under `CausalObservationReproduction.Manuscript` and
`ManuscriptChecks`.

`SequentialNoGo.lean` defines a generic sequential observation policy and
generic VoI on the same finite-horizon interface. It proves generic VoI can
emulate a policy when public history, policy representation, information and
control actions, budget/cost, and cumulative-return objective are shared.

This is an emulation result, not a claim that all causal-control or VoI methods
are equivalent. The module also checks a delayed-continuation-value witness: an
acquisition with nonpositive immediate net value can have positive total
sequential value. The one-step threshold is therefore a base case, not a general
sequential optimum.

The companion modules make the scope of that statement explicit:
`SequentialCore.lean` checks history and policy pullback under observation
refinement; `Bellman.lean` checks finite optimal-action certificates;
`SequentialRefinementValue.lean` checks weak monotonicity and explicitly
witnessed strict net value; `DecisionRelativeQuotient.lean` checks
decision-relative action-value preservation and finite evaluation-count bounds;
`CausalQuotientCounterexample.lean` checks a case where the quotient condition
fails; and `ObservationKernel.lean` defines the complete policy-visible feedback
boundary required before applying that quotient.

## Blocked separation hypotheses

Any nontrivial separation must formally relax at least one shared condition:

- a restricted generic policy class or model access;
- an explicitly causal shift/robustness objective;
- partial identification with a safe continuation criterion; or
- an information-action constraint unavailable to generic VoI.

None is presently a theorem or empirical claim. Each needs its own prior-art
review, Lean definition, and matched comparator before implementation.

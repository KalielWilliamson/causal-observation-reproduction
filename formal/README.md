# Formal artifacts

The exact source-derived manuscript theory lives in
`CausalObservationReproduction/Manuscript` and the cited compatibility checks
live in `CausalObservationReproduction/ManuscriptChecks`. Imports and namespaces
are the only scientific transformations; Mathlib and Lean are pinned at 4.29.1.

The separate `Formalism` namespace contains independently strengthened public
extensions. Together the checked material covers:

- observation classes, deterministic refinement, and decision-rule pullback;
- finite sequential histories, policy pullback, and Bellman certificates;
- weak and certificate-based strict sequential refinement value;
- decision-relative quotients, including finite evaluation-count bounds;
- a delayed-continuation counterexample showing when a quotient is not valid;
- the complete policy-visible feedback boundary.

The main module imports these artifacts. Verify them with:

```powershell
lake build CausalObservationReproduction
```

The proofs do not themselves establish an empirical result. In the extension
namespace, `SafeSensing.lean` remains a blocked future direction rather than a
completed separation theorem. The finite structures are certificate languages;
they do not formalize unbounded SCMs or every possible POMDP.

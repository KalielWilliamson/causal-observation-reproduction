# Sequential benchmark contract

The exact contract is the typed `ConditionalEfficiencyConfirmationConfig`
snapshot in `configs/reference/sequential_confirmation.json`. It is copied from
the pinned scientific source and fixes:

- two disjoint training and two disjoint held-out topology families;
- 12 generated instances per held-out topology family;
- positive, null, and invalid regimes;
- distractors 0, 6, and 12;
- horizons 3 and 5;
- probe costs 0.1 and 0.4;
- 16 training and 16 evaluation seeds;
- bounded rollout budgets 1, 4, and 16; and
- nine policy/control arms.

The output has 152,064 policy-episode rows. Each gate-versus-comparator
budget-regime analysis uses 4,608 paired executions. The primary budget-16
comparison uses a 0.05 design tolerance and paired bootstrap resampling by the
24 held-out topology instances.

## Information boundary

The learned causal-quotient gate receives only the supplied query-disagreement
Boolean, probe cost, probe reliability, observation noise, and whether a probe
remains available. It cannot read the declared regime, latent context,
semantic-world object, or scorer sidecar. Its fitted rule is a six-cell
threshold table conditioned by query disagreement and cost.

The generic comparator receives a public rollout callback and evaluates paired
acquisition/no-acquisition trajectories up to its budget. Both arms use the same
generated family, random seed, sensing budget, horizon, rewards, and base
controller. Oracle rows are marked selection-ineligible.

## Evidence and parity

The exact implementation writes:

- `confirmation_rows.json`: raw source-compatible rows;
- `confirmation_metrics.json`: policy/regime/distractor aggregates;
- `conditional_efficiency_confirmation_summary.json`: paper-facing six-cell
  projection;
- `clustered_noninferiority.json`: topology-cluster comparisons;
- policy, feature, split, evidence, and locked configuration manifests;
- `online_compute_characterization.json`; and
- `parity_report.json` after all paper-value checks pass.

Golden equality includes every positive/null budget cell, paired outcome counts,
the primary interval/decision, declared rollout and branch counts, and primary
information-action agreement. Fixed-host nanosecond timings are not compared
because they are hardware dependent; deterministic work counts are.

## CUE boundary

The retained CUE files provide optional static validation for earlier public
fixture contracts. They do not define or alter the exact source-derived
reference configuration and are not imported at runtime. This keeps CUE useful
for schema review without making it another experiment orchestration layer.

## Full-history extension

`extension-sequential-robustness` implements the richer full history/belief
planner: observation choice at each time, multiple information actions, budget
updates, quotient preservation checks, exact ambient and quotient Bellman
values, and a bounded generic comparator. Its results are explicitly
post-manuscript and never replace the locked contract above.

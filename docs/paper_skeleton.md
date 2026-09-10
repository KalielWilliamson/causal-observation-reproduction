# Sequential causal-observation benchmark: paper skeleton

## Methods

We study budgeted, non-interventional observation allocation in synthetic,
finite-horizon SCM-generated control environments. The frozen confirmation
protocol compares a learned causal-quotient gate, a capacity-matched ablation,
and generic bounded rollout-VoI under identical held-out generated families,
costs, and seed blocks.

The central contribution is a benchmark and phase-characterization protocol, not
a claim that acquiring information when its value exceeds cost is novel.

## Results

The protocol emits 152,064 policy-episode rows across held-out
`qualifying_or_negative_boundary_scope` and `observability_scope_qualifier`
topology families. It includes positive, null, and invalid controls and generic
rollout budgets 1, 4, and 16. Each gate-versus-comparator budget-regime cell has
4,608 paired executions. At the primary positive budget, all pairs tie in return
while the gate avoids the comparator's online rollouts and branches. The result
remains bounded to the declared finite synthetic grid.

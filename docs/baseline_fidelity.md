# Baseline fidelity and privilege boundary

The benchmark compares policies only within paired blocks. A block fixes the
generated POSCM candidate, topology family, phase cell, scorer-world identity,
rollout seed, horizon, action availability, sensing budget, acquisition cost,
and reward function.

| Arm                               | Public state/actions    | Extra privilege          | Release interpretation                                                                                  |
| --------------------------------- | ----------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------- |
| Never / always / random / entropy | Identical               | None                     | Fixed generic controls                                                                                  |
| Generic query-targeted VoI        | Identical               | None                     | Primary matched reference                                                                               |
| Structured-sensing ablation       | Identical               | No additional access     | Cannot claim an advantage when it ties or loses to generic VoI                                          |
| Learned variants                  | Identical at evaluation | Development-only fitting | Capacity-matched ablations, not external replications                                                   |
| External reference adapter        | Identical               | None                     | Contract adapter; not evidence of an independently reproduced method until a pinned adapter is released |
| Oracle                            | Not identical           | Scorer-level ceiling     | Upper bound only; never deployable                                                                      |

The executor and tests reject incomplete paired blocks, mixed splits, missing
topology-matched candidates, and a raw-evidence hash mismatch. A future
third-party baseline adapter must pin its revision and dependencies, document
its state/action mapping, and pass these same parity checks before its result is
treated as an independent comparison.

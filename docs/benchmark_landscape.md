# Benchmark landscape and differentiation matrix

## Status

**Preliminary benchmark gap: plausible, not yet a novelty claim.** This matrix
records a focused audit of nearby resources as of August 31, 2026. It is not an
exhaustive systematic review. “Not established” means the reviewed paper or
repository did not demonstrate the feature; it is not a claim that the feature
is absent from every version or extension of that resource.

## Target benchmark contract

The proposed resource is distinct only if it combines all four axes below:

1. **Sequential downstream control:** information acquired within an episode
   changes later control actions and cumulative return.
2. **Costed selective observation:** an agent chooses among non-interventional
   observation/probe actions (and no acquisition) under an explicit budget and
   cost.
3. **Causal semantics:** the environment has an SCM/POSCM ground truth and a
   declared causal-query or causal-role interface.
4. **Diagnostic protocol:** positive, null, and harmful regimes; matched
   budget/control/latent-world baselines; held-out graph families; and an
   oracle/reference ceiling where appropriate.

An environment with only a causal graph is not enough. A benchmark with only
intervention selection is not enough. A sensor-scheduling benchmark without a
causal ground truth is not enough.

## Search protocol

Searches used combinations of: `causal reinforcement learning benchmark`,
`causal POMDP`, `active sensing`, `sensor scheduling`, `observation cost`,
`causal bandit`, `budgeted intervention`, `causal discovery benchmark`, and
`partial observability benchmark`. Inclusion required a primary paper,
conference page, or maintained public repository with a concrete environment or
benchmark claim. Results are grouped by the closest scientific objective rather
than by venue.

## Comparison matrix

| Resource                                                                                            | Sequential downstream control                            | Costed selective **observation** actions | SCM / causal-query semantics                              | Diagnostic phase protocol                                                                | Consequence                                                                                                              |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Causal-Gymnasium](https://github.com/CausalAILab/Causal-Gymnasium)                                 | Yes                                                      | Not established                          | Yes: SCM plus `see`/`do`/`ctf_do` interfaces              | Not established                                                                          | Closest causal-RL environment interface; reuse/interoperate rather than duplicate it.                                    |
| [MTG-Causal-RL](https://arxiv.org/abs/2605.06066)                                                   | Yes                                                      | Not established                          | Yes: explicit SCM and causal audit traces                 | Partial: baselines and transfer/audit metrics, but target phase protocol not established | Close partially observable causal-RL benchmark; distinct focus would have to be paid sensing and phase characterization. |
| [Causal Bayesian Optimization Benchmark](https://github.com/chenfeng-huang/CBO-Benchmark-TMLR-2026) | Sequential optimization, not within-episode control      | No: interventions are the design actions | Yes                                                       | Partial: budgets, metrics, graph-misspecification stress tests                           | Strong precedent for causal benchmark methodology and method-rank variation; different decision objective.               |
| [Budgeted causal bandits](https://proceedings.mlr.press/v130/nair21a.html)                          | Across-round bandit learning, not finite-horizon control | No: budgeted actions are interventions   | Yes                                                       | Not established as a reusable phase benchmark                                            | Required baseline family, but not the proposed environment class.                                                        |
| [Active Causal Discovery Bench](https://github.com/qpiai/Active-Causal-Discovery-Bench)             | No: terminal graph-recovery objective                    | No: actions are interventions            | Yes                                                       | Yes for discovery: truth-owned DAG, CPDAG ceiling, intervention-efficiency score         | Strong diagnostic-design precedent; objective is causal discovery, not control.                                          |
| [CSuite](https://github.com/microsoft/csuite)                                                       | No                                                       | No                                       | Yes: synthetic SCM datasets and intervention ground truth | Curated causal inference cases, not sequential diagnostic phases                         | Reuse its documentation/provenance practices where applicable.                                                           |
| [Active Measure RL](https://arxiv.org/abs/2005.12697)                                               | Yes                                                      | Yes: observations at a cost              | Not established                                           | Not established                                                                          | Closest non-causal observation-cost RL problem; must be a required baseline/reference.                                   |
| [POBAX](https://arxiv.org/abs/2508.00046)                                                           | Yes                                                      | Not established                          | No                                                        | Partial: benchmark-design guidance and partial-observability coverage                    | Strong benchmark-design precedent; causal and costed-sensing axes remain open.                                           |

## Preliminary differentiation statement

Do not use “first causal RL benchmark.” Subject to a complete review, the
defensible working statement is:

> A diagnostic suite for SCM-generated POMDPs in which an agent allocates a
> finite budget among non-interventional observation actions and control
> actions, with causal-query semantics and preregistered positive, null, and
> harmful regimes for matched sensing baselines.

The statement becomes a publishable claim only after the following checks pass.

## Reopening checks

1. Read the primary paper and execute a minimal example for every row marked
   “not established” or “partial.”
2. Add a row for every newly found resource with two or more target axes.
3. Demonstrate at least one non-obvious phase result that cannot be read from a
   generic POMDP or causal-bandit result alone.
4. Ship adapters or protocol-level interoperability for Causal-Gymnasium where
   practical, rather than presenting a parallel causal environment ecosystem.
5. Include independent control, active-sensing, causal-bandit, and causal-RL
   baseline families under the same latent worlds and sensing budgets.
6. Keep the current fail-closed novelty gate: if matched generic VoI explains
   all observed behavior, the benchmark may still be useful, but no causal
   policy advantage may be claimed.

## Current repository implications

The manuscript-facing commands implement the benchmark boundary only:
grammar-curated POSCM candidates, split isolation, cost/budget tracking, matched
baselines, a locked phase protocol, and a negative-result rule. Exploratory
transfer and policy-learning implementations are retained behind explicit
`extension-` command names and are not manuscript evidence.

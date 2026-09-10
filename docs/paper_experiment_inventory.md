# Paper experiment inventory

## Selection rule

The reference layer is the transitive scientific dependency closure of the
manuscript: code that defines an algorithm, constructs an evaluated fixture,
computes a reported quantity, states a cited formal result, or tests one of
those components. Experiment scheduling, cloud storage, tracking, knowledge
graph projection, dashboards, and publication automation are excluded.

The extraction is pinned to source revision
`b4c468ef4dfac5eeab8b11a9e7f990ba36824748`. Each copied Python source blob and
formal source tree is listed in `provenance/source_manifest.json` along with the
public destination and permitted refactor. Scientific behavior is checked by
ported source assertions and golden paper-result parity.

## Manuscript inventory

| ID  | Component                                                                                       | Exact public implementation                                                                                                                                                                | Evidence status                                                                       |
| --- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| F1  | Finite observation, refinement, query sufficiency, belief, Bellman, and sequential value theory | `CausalObservationReproduction/Manuscript` plus `ManuscriptChecks`; Mathlib and Lean versions are locked                                                                                   | Complete and proof-checked locally                                                    |
| E1  | One-step learned COO gate calibration                                                           | `reference.learned_coo_gate`, exact deterministic evaluator, grouped splits, observable-only features, linear/MLP/graph variants, bootstrap metrics                                        | Source-derived pilot; executable corpus contains 214 non-test and 60 frozen-test rows |
| E2  | Locked sequential POSCM comparison                                                              | `reference.sequential_experiment`, generated SCM evaluator, query-conditioned six-cell gate, bounded rollout-VoI comparator, matched controls, cluster bootstrap, compute characterization | Promoted finite-grid result; exact paper values are fail-closed golden checks         |
| E3  | MiniGrid external control audit                                                                 | Packaged 16 evaluated source-target rows, analysis-ready Parquet, JSONL, and recomputed relation summaries                                                                                 | Descriptive appendix result; not a general transfer predictor                         |

The broader POSCM transfer, DQN, PPO, cache-control, larger one-step, and full
history/Bellman experiments created during public-repository development are
useful follow-up work, but they are not inputs to the manuscript results. Their
CLI names therefore begin with `extension-`.

## E1: one-step calibration

The exact source configuration materializes 274 rows in total:

| Panel                   |    Rows |
| ----------------------- | ------: |
| Development             |     180 |
| Historical primary      |      20 |
| Historical confirmation |      14 |
| Frozen held-out         |      60 |
| **Total**               | **274** |

Only the 180 development rows are split into 144 training and 36 validation
rows. The 60 frozen rows contain 20 positive, 20 null, and 20 pathology cases.

This exposes an important manuscript/source mismatch: the manuscript prose
describes a “274-example learning corpus” plus 60 held-out rows, while the
executable source produces 214 non-test rows plus the 60 held-out rows. The
reference command preserves the executable source exactly. The previously
constructed 274-learning-row plus 300-independent-row repair remains available
only as `extension-one-step-strengthened`; it is not silently substituted for
the published protocol.

```sh
bash scripts/reproduce.sh one-step
```

## E2: sequential comparison

The exact locked grid contains:

- two held-out topology families with 12 instances each;
- 16 evaluation seeds;
- positive, null, and invalid regimes;
- distractor counts 0, 6, and 12;
- horizons 3 and 5;
- probe costs 0.1 and 0.4;
- generic rollout budgets 1, 4, and 16;
- nine policy/control arms, with the generic arm repeated for all three budgets.

That is 152,064 emitted policy-episode rows. Each gate-versus-comparator
budget-regime cell contains 4,608 paired executions and the primary bootstrap
resamples 24 topology instances.

The parity check covers all reported gate-versus-comparator cells. In the
positive regime, gate-minus-comparator return is -0.0078125, +0.03125, and 0 at
budgets 1, 4, and 16. At budget 16 all 4,608 positive pairs tie and all 13,824
primary pairs across positive/null/invalid regimes select the same information
action. The gate uses zero online rollouts and candidate branches; the positive
comparator averages 32 rollouts and 22.35546875 branches.

```sh
bash scripts/reproduce.sh sequential
```

The gate is a sequential acquisition policy, but the manuscript implementation
is not a learned quotient-map constructor or an exact history-state planner. It
receives the supplied query-disagreement Boolean and probe cost and consults a
six-cell fitted threshold table. The formal corpus proves richer sequential
properties; the independently implemented full belief/history Bellman planner is
isolated as `extension-sequential-robustness`.

## E3: MiniGrid external control

The paper artifact contains exactly 16 evaluated rows: two acquisition modes,
two seeds, and four declared source-target relations. The public extraction
retains 42 scientific design/outcome columns and omits storage/registry fields.

| Relation                      | Evaluated | Transferred | Mean readiness |
| ----------------------------- | --------: | ----------: | -------------: |
| Same environment, same seed   |         4 |           4 |           1.00 |
| Same environment, seed shift  |         4 |           4 |           1.00 |
| Same family, scale change     |         4 |           4 |   0.7811236821 |
| Cross-family negative control |         4 |           0 |   0.2653192179 |

```sh
bash scripts/reproduce.sh minigrid-audit
```

This command recomputes the appendix summary from the exact evaluated rows. It
does not retrain PPO and does not claim population-level MiniGrid transfer.

## Output contract

The preferred dataset contract is:

1. Parquet for analysis-ready tabular data with stable column types;
2. JSONL for transparent, streaming, language-neutral row interchange;
3. JSON for configurations, manifests, summaries, and parity decisions;
4. model checkpoints only where the experiment actually trains a model.

Every output directory also contains `run_manifest.json`. Raw rows are the
evidence record; summaries can always be regenerated from them.

## Extensions

The single API makes the boundary visible in names:

```sh
uv run causal-observation-run extension-sequential-robustness
uv run causal-observation-run extension-one-step-strengthened
uv run causal-observation-run extension-poscm-transfer
uv run causal-observation-run extension-poscm-dqn
uv run causal-observation-run extension-minigrid-design
uv run causal-observation-run extension-minigrid-policy
uv run causal-observation-run extension-minigrid-robustness
uv run causal-observation-run extension-certified-cache-control
```

Extension results must not be used as replacements for E1–E3 without a new
analysis and manuscript revision.

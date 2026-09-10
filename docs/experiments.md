# Running experiments

## One interface

```sh
uv run causal-observation-run EXPERIMENT --output-dir PATH
```

The shell and PowerShell wrappers use the same experiment names. `smoke` is the
default and `paper` runs the full manuscript workflow.

## Exact manuscript experiments

### Sequential POSCM comparison

```sh
bash scripts/reproduce.sh sequential
```

This runs the source-derived locked grid. Each emitted row records cost-adjusted
return, acquisition cost, probe count, generic rollout count, observed
candidate-branch count, topology split, policy metadata, and exact seed. The
runner then:

1. regenerates the six published budget-regime comparisons;
2. checks the primary topology-cluster analysis;
3. checks all 13,824 primary information-action pairs;
4. characterizes deterministic online work and descriptive host timing; and
5. writes `parity_report.json` only if all locked values agree.

Rollouts and branches are algorithmic work counts. Timing is descriptive and is
not used for cross-machine parity.

### One-step calibration

```sh
bash scripts/reproduce.sh one-step
```

This installs the `learned` extra and executes the exact observable-feature
dataset, grouped split, linear baseline, MLP, graph encoder, oracle controls,
cluster bootstrap, figures, tables, checkpoints, and local evidence writer. The
source-defined data has 274 total rows: 214 non-test and 60 frozen test.
`parity_report.json` verifies that the published held-out value, regret,
activation/tie, and relative-error conclusions agree. It also records
`corpus_cardinality_matches: false`: the manuscript's claimed 274-example
learning corpus plus 60 held-out rows is not the corpus emitted by the pinned
source.

### MiniGrid audit

```sh
bash scripts/reproduce.sh minigrid-audit
```

This exports the exact 16 evaluated rows used by the appendix and recomputes the
four relation-level summaries. It does not retrain PPO: the manuscript labels
this data as a descriptive external control, not a general benchmark.

## Post-manuscript extensions

Extensions use the same runner but cannot be confused with exact evidence:

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

The full sequential robustness extension makes repeated information choices,
updates beliefs and history, validates quotient reward/transition/observation
preservation, and solves exact Bellman continuation values. It is a stronger
algorithmic implementation than the paper's six-cell gate, but it is not the
implementation that generated the paper result.

The strengthened one-step extension uses the repaired 274-learning-row and
separate 300-row panel developed after the historical count mismatch was
identified. The DQN/PPO/MiniGrid and cache-control extensions likewise provide
useful follow-up evidence without modifying the manuscript bundles.

## Artifacts

All generated paths are below `artifacts/runs/` by default and are ignored by
Git. Raw records are retained alongside derived summaries. The recommended
archival package should include raw Parquet/JSONL data, JSON configuration and
manifest files, summary tables, code revision, environment lockfiles, and the
parity report.

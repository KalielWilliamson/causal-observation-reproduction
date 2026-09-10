# Causal Observation Quotients: Reproduction Repository

This repository contains the compact, public reproduction of the algorithms,
experiments, and Lean theory used by the causal observation quotients (COO)
manuscript. The scientific implementation was extracted from source revision
`b4c468ef4dfac5eeab8b11a9e7f990ba36824748`; cloud tracking, registries,
schedulers, and lab deployment code were deliberately removed.

The repository has one experiment interface:

```sh
uv run causal-observation-run <experiment>
```

Use `smoke` for a fast installation check or `paper` for every executable
manuscript experiment. Individual manuscript experiments are `sequential`,
`one-step`, and `minigrid-audit`. Additional work is available only under names
beginning with `extension-` and is not presented as published evidence.

## Quick start

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required.

```sh
git clone https://github.com/KalielWilliamson/causal-observation-reproduction.git
cd causal-observation-reproduction
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh smoke
```

Run the complete paper workflow (CPU-only, but substantially longer):

```sh
bash scripts/reproduce.sh paper
```

Treat `paper` as a long batch job. The first full reference execution on the
Windows host used for this release took about 4 hours 45 minutes for the
single-core sequential phase; timing is hardware-dependent. `smoke` remains the
seconds-scale setup check.

On Windows PowerShell, the matching commands are:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 setup
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 smoke
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 paper
```

The full quality workflow is:

```sh
bash scripts/reproduce.sh lint
bash scripts/reproduce.sh test
bash scripts/reproduce.sh formal
bash scripts/reproduce.sh docs
```

Docker is optional. `bash scripts/reproduce.sh container` builds a pinned local
runtime and executes the exact sequential smoke path. CUE is used only for
optional static contract checks through a pinned container; it is not a runtime
dependency. Kubernetes is intentionally absent because none of the experiments
requires a distributed service.

## What is reproduced

| Command          | Manuscript role                           | Main output                                                                                                                                  |
| ---------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `one-step`       | Learned probe-or-abstain calibration      | 274 total source-derived examples: 214 non-test and 60 frozen held-out rows, plus policy metrics and model artifacts                         |
| `sequential`     | Locked topology-held-out POSCM comparison | 152,064 policy episodes, the six gate-versus-comparator cells, clustered analysis, compute characterization, and a fail-closed parity report |
| `minigrid-audit` | Descriptive external control              | The exact 16 evaluated rows and four recomputed relation summaries                                                                           |
| `paper`          | Aggregate workflow                        | All three bundles above                                                                                                                      |

At the primary sequential budget, the source-derived run checks for the
manuscript values: 4,608 positive pairs tie in return, the information action
agrees in all 13,824 primary comparisons across regimes, the COO gate uses zero
online rollouts/branches, and the bounded comparator averages 32 rollouts and
22.35546875 branches in the positive regime. The parity runner fails if those or
the budget-1/budget-4 controls diverge.

The regenerated one-step outcome conclusions also match the manuscript: the
linear, MLP, and graph variants match oracle value and regret, refine the 20
positive and 20 zero-value tie cases, and the linear model has the lowest value
error. One protocol statement is disproved by the executable source: the paper
says 274 learning examples plus 60 held-out examples, while the pinned program
contains 274 rows total (180 development, 34 historical, and 60 held-out).
`one-step/parity_report.json` records both the matching outcomes and this known
cardinality mismatch instead of silently repairing it.

The implemented manuscript gate is sequential—it chooses whether to acquire
information during an episode—but it is intentionally the source's six-cell,
query-conditioned threshold table. The repository's richer belief/history
Bellman implementation is retained as `extension-sequential-robustness`; it is
not substituted for the published method.

## Artifacts and provenance

Large tabular observations are stored as Parquet when the experiment already
requires the learned-data extra, with JSONL as the transparent row-oriented
interchange form. The sequential bundle retains the source's exact JSON row
array as its canonical parity record. JSON also holds configurations, summaries,
manifests, and parity reports. Derived summaries never replace raw rows.

The source-to-public map, source blob/tree identifiers, allowed refactors, and
external MiniGrid object digest are recorded in
[`provenance/source_manifest.json`](provenance/source_manifest.json). See the
[`paper experiment inventory`](docs/paper_experiment_inventory.md),
[`reproduction guide`](docs/reproduction.md), and [`public API`](docs/api.md)
for details.

## License

MIT. See [LICENSE](LICENSE).

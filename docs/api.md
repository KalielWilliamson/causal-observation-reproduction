# Public API

## Command line

All experiments use one interface:

```text
causal-observation-run EXPERIMENT [--output-dir PATH] [--name NAME]
```

`EXPERIMENT` is one of:

- `smoke`: reduced exact sequential path;
- `paper`: all executable manuscript experiments;
- `sequential`: full locked sequential comparison and parity check;
- `one-step`: exact learned-gate calibration;
- `minigrid-audit`: exact 16-row external-control audit; or
- an explicitly named `extension-*` experiment.

The default is `smoke`. Output defaults to `artifacts/runs/<experiment>`. Every
call writes `run_manifest.json` and prints a JSON object containing the
experiment name, manifest, and ordered artifact paths.

## Python

The stable programmatic entry point is:

```python
from causal_observation_reproduction.benchmark import run

result = run("minigrid-audit", "artifacts/runs/minigrid-audit")
print(result.primary_artifact)
```

`run` returns a `BenchmarkRun` with a strongly typed `experiment`, an ordered
`tuple[Path, ...]` of artifacts, and `primary_artifact`. There is no separate
legacy/versioned interface and no mode-dependent return type.

## Reference layer

`causal_observation_reproduction.reference` contains the source-derived
scientific implementation. Important modules are:

- `sequential_experiment`: the locked generated-POSCM grid, nine policy/control
  arms, six-cell COO gate, bounded rollout-VoI comparator, and clustered
  analysis;
- `sequential_sensing`: the costed, budgeted information-before-control episode
  environment;
- `learned_coo_gate`: the exact one-step dataset and learned calibration;
- `minigrid_audit`: typed access to the exact 16 evaluated appendix rows;
- `compute_characterization`: deterministic online-work counts and descriptive
  fixed-host timing; and
- `parity`: fail-closed checks against manuscript result projections.

The reference layer is intentionally low-level and inspectable. The stable
`benchmark.run` interface should be preferred by downstream automation.

## COO theory and extensions

The `coo` package contains the independently strengthened finite one-step and
full belief/history Bellman abstractions. It is valuable executable theory but
is not substituted for the six-cell gate used by the manuscript experiment.

Post-manuscript runners are collected behind
`causal_observation_reproduction.extensions`; their public experiment names all
begin with `extension-`. Existing implementation modules under `experiments` are
internal to that boundary and are not manuscript evidence APIs.

## Dataset formats

The stable artifact convention is:

- Parquet for typed, analysis-ready tables;
- JSONL for inspectable row streams;
- JSON for configurations, summaries, manifests, and parity decisions; and
- framework-native checkpoints only for trained models.

The one-step bundle writes Parquet because PyArrow is part of its `learned`
extra. MiniGrid ships both Parquet and JSONL. The sequential implementation
retains its exact source JSON array as the parity record; derived export formats
must not replace that file.

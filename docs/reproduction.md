# Reproduction guide

## Fast path

Install Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/), then run:

```sh
git clone https://github.com/KalielWilliamson/causal-observation-reproduction.git
cd causal-observation-reproduction
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh smoke
```

`smoke` uses the exact sequential implementation with a reduced grid. It is an
installation/structural check, not manuscript evidence.

## Full paper reproduction

The aggregate command executes the exact one-step and sequential protocols and
recomputes the MiniGrid appendix audit:

```sh
bash scripts/reproduce.sh paper
```

Outputs are written to `artifacts/runs/paper/`. The sequential run is
CPU-intensive because it executes 152,064 policy episodes, including bounded
rollout comparators. It writes `parity_report.json` only after every published
comparison agrees with the checked reference values. A mismatch terminates the
command with an error.

Plan for an hours-scale, single-core batch run. The first full execution on the
Windows release host took about 4 hours 45 minutes for the sequential phase;
that observation is not a cross-machine performance claim. `smoke` is the
seconds-scale installation check.

Run components independently when iterating:

```sh
bash scripts/reproduce.sh sequential
bash scripts/reproduce.sh one-step
bash scripts/reproduce.sh minigrid-audit
```

The one-step run installs the optional `learned` dependency set (CPU Torch,
Pandas, PyArrow, and Matplotlib). A GPU is not required. Its parity report
checks the published held-out outcome conclusions and explicitly reports the
known corpus discrepancy: 274 executable rows total versus the manuscript's
274-learning-plus-60-held-out description.

## Windows PowerShell

Install Git, uv, and optionally Docker Desktop. Keep Docker Desktop in Linux
container mode. Install [elan](https://github.com/leanprover/elan) if checking
the formal corpus locally.

```powershell
git clone https://github.com/KalielWilliamson/causal-observation-reproduction.git
Set-Location causal-observation-reproduction
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 setup
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 smoke
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 paper
```

If Docker bind mounts fail, keep the checkout under the local user profile and
grant Docker Desktop access to that drive. A mapped network drive is not
recommended.

## macOS and Linux

On macOS, `brew install git uv` installs the native prerequisites; install
Docker Desktop and elan only when using their corresponding checks. On Linux,
install Git and uv through the distribution or uv installer. The shell wrapper
is identical on both platforms:

```sh
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh smoke
bash scripts/reproduce.sh paper
```

Apple Silicon is supported. The numerical protocols are CPU-only and use the
same locked Python dependency graph as other platforms.

## Direct API

The wrappers are syntax sugar for one command:

```sh
uv run causal-observation-run sequential \
  --output-dir artifacts/runs/sequential
```

With no experiment argument, the CLI runs `smoke`. `--help` prints every exact
manuscript command and every explicitly prefixed extension.

## Quality and formal checks

```sh
bash scripts/reproduce.sh lint
bash scripts/reproduce.sh test
bash scripts/reproduce.sh docs
bash scripts/reproduce.sh formal
bash scripts/reproduce.sh contract-check
```

The formal command checks the complete manuscript-relevant Lean closure with
Lean 4.29.1 and Mathlib pinned in `lake-manifest.json`. The CUE check runs in a
pinned container and validates retained static contracts; CUE is not imported by
the experiment runtime.

## Docker and Kubernetes

Docker is optional:

```sh
bash scripts/reproduce.sh container
```

The image executes `smoke` and writes through a single mounted output directory.
The full learned workflow stays native by default so model artifacts remain
directly accessible and image size stays reasonable.

Kubernetes is intentionally not included. These experiments are deterministic,
single-process scientific programs with no service-to-service or autoscaling
requirement. A Kubernetes layer would add operational state without improving
reproducibility.

## Output and evidence semantics

- Raw tabular rows are Parquet where a typed analysis table is appropriate and
  JSONL where a transparent streaming representation is useful. The exact
  sequential source array remains JSON because changing its canonical bytes
  would weaken source parity.
- JSON holds configurations, manifests, summaries, and parity reports.
- Generated artifacts live below the ignored `artifacts/runs/` directory.
- The checked MiniGrid rows are source evidence packaged in the wheel; exporting
  them does not rerun PPO training.
- Post-manuscript work is available only through `extension-*` commands and is
  never merged into paper summaries.

The complete inclusion/exclusion rationale and exact cardinalities are in the
[paper experiment inventory](paper_experiment_inventory.md).

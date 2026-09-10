#!/usr/bin/env bash
# Convenience wrapper for macOS, Linux, and Git Bash on Windows.
set -euo pipefail

readonly IMAGE="causal-observation-reproduction:local"
readonly CUE_IMAGE="cuelang/cue@sha256:520f883dcc92642389b3b75e8befe4e5a233a1aa903a8c846c4a3b338b202635"

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce.sh <command>

Core commands:
  setup, test, lint, docs, formal, contract-check
  smoke, paper, sequential, one-step, minigrid-audit, container

Optional post-manuscript extensions:
  extension-certified-cache-control, extension-sequential-robustness,
  extension-one-step-strengthened, extension-poscm-transfer, extension-poscm-dqn,
  extension-minigrid-design, extension-minigrid-policy,
  extension-minigrid-robustness
EOF
}

run_experiment() {
  local experiment="$1"
  uv run causal-observation-run "$experiment" --output-dir "artifacts/runs/$experiment"
}

case "${1:-}" in
  setup) uv sync --all-groups --all-extras ;;
  test) uv run --group test pytest ;;
  lint)
    uv run --with ruff ruff check src tests tools
    uv run --with ruff ruff format --check src tests tools
    uv run --all-extras --with mypy mypy
    uv run --no-project --with 'validate-pyproject==0.26' --with 'validate-pyproject-schema-store[all]' validate-pyproject pyproject.toml
    ;;
  docs) uv run --script noxfile.py -s docs --non-interactive ;;
  formal) lake build CausalObservationReproduction ;;
  contract-check)
    docker run --rm -v "$(pwd):/workspace" -w /workspace "$CUE_IMAGE" vet -d '#SequentialConfirmation' configs/reference/sequential_confirmation.json cue/reference/sequential.cue
    docker run --rm -v "$(pwd):/workspace" -w /workspace "$CUE_IMAGE" vet -d '#OneStepGate' configs/reference/one_step_gate.json cue/reference/one_step.cue
    docker run --rm -v "$(pwd):/workspace" -w /workspace "$CUE_IMAGE" vet -c=false ./cue/cache_control
    ;;
  paper | one-step | extension-one-step-strengthened)
    uv sync --extra learned
    run_experiment "$1"
    ;;
  extension-poscm-dqn)
    uv sync --extra rl
    run_experiment "$1"
    ;;
  extension-minigrid-policy | extension-minigrid-robustness)
    uv sync --extra minigrid
    run_experiment "$1"
    ;;
  smoke | sequential | minigrid-audit | extension-certified-cache-control | extension-sequential-robustness | extension-poscm-transfer | extension-minigrid-design)
    run_experiment "$1"
    ;;
  container)
    mkdir -p artifacts/runs
    docker build --tag "$IMAGE" .
    docker run --rm -v "$(pwd)/artifacts/runs:/outputs" "$IMAGE" smoke --output-dir /outputs/container-smoke
    ;;
  *) usage; exit 2 ;;
esac

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "setup", "test", "lint", "docs", "formal", "contract-check",
        "smoke", "paper", "sequential", "one-step", "minigrid-audit", "container",
        "extension-certified-cache-control", "extension-sequential-robustness",
        "extension-one-step-strengthened", "extension-poscm-transfer", "extension-poscm-dqn",
        "extension-minigrid-design", "extension-minigrid-policy", "extension-minigrid-robustness",
        "help"
    )]
    [string]$Command = "help"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$cueImage = "cuelang/cue@sha256:520f883dcc92642389b3b75e8befe4e5a233a1aa903a8c846c4a3b338b202635"
$containerImage = "causal-observation-reproduction:local"

Set-Location $repositoryRoot

function Show-Usage {
    Write-Output "Usage: powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 <command>"
    Write-Output "Core: setup, test, lint, docs, formal, contract-check, smoke, paper, sequential, one-step, minigrid-audit, container"
    Write-Output "Extensions: extension-certified-cache-control, extension-sequential-robustness, extension-one-step-strengthened, extension-poscm-transfer, extension-poscm-dqn, extension-minigrid-design, extension-minigrid-policy, extension-minigrid-robustness"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath"
    }
}

function Invoke-Experiment {
    param([Parameter(Mandatory = $true)][string]$Experiment)
    Invoke-Checked uv run causal-observation-run $Experiment --output-dir "artifacts/runs/$Experiment"
}

switch ($Command) {
    "setup" { Invoke-Checked uv sync --all-groups --all-extras }
    "test" { Invoke-Checked uv run --group test pytest }
    "lint" {
        Invoke-Checked uv run --with ruff ruff check src tests tools
        Invoke-Checked uv run --with ruff ruff format --check src tests tools
        Invoke-Checked uv run --all-extras --with mypy mypy
        Invoke-Checked uv run --no-project --with "validate-pyproject==0.26" --with "validate-pyproject-schema-store[all]" validate-pyproject pyproject.toml
    }
    "docs" { Invoke-Checked uv run --script noxfile.py -s docs --non-interactive }
    "formal" { Invoke-Checked lake build CausalObservationReproduction }
    "contract-check" {
        Invoke-Checked docker @("run", "--rm", "-v", "${repositoryRoot}:/workspace", "-w", "/workspace", $cueImage, "vet", "-d", "#SequentialConfirmation", "configs/reference/sequential_confirmation.json", "cue/reference/sequential.cue")
        Invoke-Checked docker @("run", "--rm", "-v", "${repositoryRoot}:/workspace", "-w", "/workspace", $cueImage, "vet", "-d", "#OneStepGate", "configs/reference/one_step_gate.json", "cue/reference/one_step.cue")
        Invoke-Checked docker @("run", "--rm", "-v", "${repositoryRoot}:/workspace", "-w", "/workspace", $cueImage, "vet", "-c=false", "./cue/cache_control")
    }
    { $_ -in @("paper", "one-step", "extension-one-step-strengthened") } {
        Invoke-Checked uv sync --extra learned
        Invoke-Experiment $Command
    }
    "extension-poscm-dqn" {
        Invoke-Checked uv sync --extra rl
        Invoke-Experiment $Command
    }
    { $_ -in @("extension-minigrid-policy", "extension-minigrid-robustness") } {
        Invoke-Checked uv sync --extra minigrid
        Invoke-Experiment $Command
    }
    { $_ -in @(
        "smoke", "sequential", "minigrid-audit", "extension-certified-cache-control",
        "extension-sequential-robustness", "extension-poscm-transfer", "extension-minigrid-design"
    ) } { Invoke-Experiment $Command }
    "container" {
        $outputRoot = Join-Path $repositoryRoot "artifacts/runs"
        New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
        Invoke-Checked docker build --tag $containerImage .
        Invoke-Checked docker run --rm --mount "type=bind,source=$outputRoot,target=/outputs" $containerImage smoke --output-dir /outputs/container-smoke
    }
    default { Show-Usage }
}

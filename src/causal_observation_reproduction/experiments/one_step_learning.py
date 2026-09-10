"""Local held-out calibration for the source-derived one-step COO corpus.

This module keeps the paper-visible learning experiment small: inputs are
restricted to the released observable features, fitted preprocessing sees only
development rows, and frozen rows are evaluated exactly once.  Torch is an
optional accelerator for the MLP and message-passing variants; a clearly
labelled ridge fallback keeps the artifact contract inspectable on a base
installation.
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np

from causal_observation_reproduction.experiments.one_step import (
    materialize_one_step_dataset,
    write_one_step_dataset,
)

ModelBackend = Literal["torch", "ridge_fallback"]
EvaluationPanel = Literal["historical_frozen_test", "independent_confirmation"]

FEATURE_NAMES: tuple[str, ...] = (
    "observation_reliability",
    "observation_cost",
    "probe_cost",
    "observation_regime_coarse",
    "query_continuation_action",
    "observable_node_count",
    "observable_edge_count",
    "observable_action_count",
    "observable_mediator_count",
    "observable_proxy_count",
    "observable_outcome_count",
)
LABEL_FIELDS: frozenset[str] = frozenset(
    {
        "V_coarse",
        "V_refined",
        "gross_refinement_value",
        "net_refinement_value",
        "oracle_action",
        "registered_region",
        "continuation_action_gap",
        "incompatible_optimal_actions",
        "positive_reachability",
    }
)


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(frozen=True)
class OneStepLearningConfig:
    """Frozen local protocol for the held-out calibration pilot."""

    seed: int = 17
    validation_fraction: float = 0.2
    hidden_dim: int = 24
    message_passing_layers: int = 2
    learning_rate: float = 0.01
    max_epochs: int = 160
    early_stopping_patience: int = 24
    auxiliary_sign_loss_weight: float = 0.1
    bootstrap_samples: int = 400
    robustness_seeds: tuple[int, ...] = (17, 29, 43, 59, 71)

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            _raise_invalid("validation_fraction must lie strictly between zero and one")
        if (
            self.hidden_dim < 1
            or self.message_passing_layers < 0
            or self.max_epochs < 1
            or self.early_stopping_patience < 1
            or self.bootstrap_samples < 1
        ):
            _raise_invalid(
                "model dimensions, epochs, patience, and bootstrap samples must be positive"
            )
        if self.learning_rate <= 0.0 or self.auxiliary_sign_loss_weight < 0.0:
            _raise_invalid(
                "learning rate must be positive and auxiliary loss weight nonnegative"
            )
        if not self.robustness_seeds or len(set(self.robustness_seeds)) != len(
            self.robustness_seeds
        ):
            _raise_invalid("robustness seeds must be nonempty and unique")


@dataclass(frozen=True)
class PolicyMetric:
    """Frozen-panel value and decision metrics for one declared policy."""

    policy: str
    evaluation_panel: EvaluationPanel
    eiv: float
    mean_regret: float
    activation_rate: float
    conditional_activation_value: float
    false_positive_loss: float
    false_negative_loss: float
    mae: float
    rmse: float
    sign_accuracy: float
    precision: float
    recall: float
    f1: float
    eiv_ci_lower: float
    eiv_ci_upper: float


@dataclass(frozen=True)
class OneStepCalibrationResult:
    """Complete local result, retaining prediction provenance by model."""

    records: tuple[dict[str, object], ...]
    splits: dict[str, tuple[str, ...]]
    predictions: dict[str, tuple[float, ...]]
    model_backends: dict[str, ModelBackend]
    metrics: tuple[PolicyMetric, ...]
    region_metrics: tuple[dict[str, object], ...]
    comparisons: tuple[dict[str, object], ...]


def run_one_step_calibration(
    config: OneStepLearningConfig | None = None,
) -> OneStepCalibrationResult:
    """Fit declared calibration policies and evaluate their frozen panel once."""

    resolved = config or OneStepLearningConfig()
    records = tuple(example.as_record() for example in materialize_one_step_dataset())
    _validate_records(records)
    splits = build_grouped_splits(records, config=resolved)
    features = _feature_matrix(records)
    targets = np.asarray(
        [_as_float(record["net_refinement_value"]) for record in records],
        dtype=np.float64,
    )
    indices = _indices_by_id(records)
    train_indices = np.asarray(
        [indices[row_id] for row_id in splits["train"]], dtype=np.int64
    )
    validation_indices = np.asarray(
        [indices[row_id] for row_id in splits["validation"]], dtype=np.int64
    )

    predictions: dict[str, np.ndarray] = {
        "never_refine": np.full(len(records), -1.0),
        "always_refine": np.full(len(records), 1.0),
        "linear_value_model": _ridge_predict(
            features[train_indices], targets[train_indices], features
        ),
        "formula_coo_oracle_statistic": _formula_oracle_prediction(records),
        "oracle_coo": targets.copy(),
    }
    predictions["mlp_value_model"], mlp_backend = _learned_predictions(
        records,
        features,
        targets,
        train_indices,
        validation_indices,
        resolved,
        graph=False,
    )
    predictions["gnn_coo"], gnn_backend = _learned_predictions(
        records,
        features,
        targets,
        train_indices,
        validation_indices,
        resolved,
        graph=True,
    )
    frozen_indices = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if record["experiment_panel"] == "frozen_test"
        ],
        dtype=np.int64,
    )
    learned_rate = float((predictions["gnn_coo"][frozen_indices] > 0.0).mean())
    predictions["budget_matched_random"] = _random_predictions(
        len(records), learned_rate, seed=resolved.seed + 91
    )
    independent_indices = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if record["experiment_panel"] == "independent_confirmation"
        ],
        dtype=np.int64,
    )
    historical_metrics, historical_regions, historical_comparisons = _evaluate_policies(
        records,
        predictions,
        evaluation_indices=frozen_indices,
        evaluation_panel="historical_frozen_test",
        config=resolved,
    )
    independent_metrics, independent_regions, independent_comparisons = (
        _evaluate_policies(
            records,
            predictions,
            evaluation_indices=independent_indices,
            evaluation_panel="independent_confirmation",
            config=resolved,
        )
    )
    return OneStepCalibrationResult(
        records=records,
        splits=splits,
        predictions={
            name: tuple(float(value) for value in values)
            for name, values in predictions.items()
        },
        model_backends={"mlp_value_model": mlp_backend, "gnn_coo": gnn_backend},
        metrics=historical_metrics + independent_metrics,
        region_metrics=historical_regions + independent_regions,
        comparisons=historical_comparisons + independent_comparisons,
    )


def build_grouped_splits(
    records: tuple[dict[str, object], ...], *, config: OneStepLearningConfig
) -> dict[str, tuple[str, ...]]:
    """Create graph-disjoint learning, historical, and fresh allocations."""

    frozen = tuple(
        str(record["instance_id"])
        for record in records
        if record["experiment_panel"] == "frozen_test"
    )
    independent_confirmation = tuple(
        str(record["instance_id"])
        for record in records
        if record["experiment_panel"] == "independent_confirmation"
    )
    learning = tuple(
        record
        for record in records
        if record["experiment_panel"] not in {"frozen_test", "independent_confirmation"}
    )
    groups = sorted({str(record["base_graph_id"]) for record in learning})
    random.Random(config.seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * config.validation_fraction))
    validation_groups = frozenset(groups[:validation_count])
    splits = {
        "train": tuple(
            sorted(
                str(record["instance_id"])
                for record in learning
                if str(record["base_graph_id"]) not in validation_groups
            )
        ),
        "validation": tuple(
            sorted(
                str(record["instance_id"])
                for record in learning
                if str(record["base_graph_id"]) in validation_groups
            )
        ),
        "frozen_test": tuple(sorted(frozen)),
        "independent_confirmation": tuple(sorted(independent_confirmation)),
    }
    _assert_split_integrity(records, splits)
    return splits


def write_one_step_calibration(
    output_dir: str | Path, config: OneStepLearningConfig | None = None
) -> tuple[Path, ...]:
    """Write raw inputs, predictions, metrics, and split provenance locally."""

    resolved = config or OneStepLearningConfig()
    directory = Path(output_dir)
    data_path, data_manifest_path = write_one_step_dataset(directory)
    result = run_one_step_calibration(resolved)
    predictions_path = directory / "one_step_predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": record["instance_id"],
                    "is_frozen_test": record["experiment_panel"] == "frozen_test",
                    "is_independent_confirmation": record["experiment_panel"]
                    == "independent_confirmation",
                    **{
                        policy: result.predictions[policy][index]
                        for policy in sorted(result.predictions)
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for index, record in enumerate(result.records)
        ),
        encoding="utf-8",
    )
    metrics_path = directory / "one_step_policy_metrics.jsonl"
    metrics_path.write_text(
        "".join(
            json.dumps(asdict(metric), sort_keys=True) + "\n"
            for metric in result.metrics
        ),
        encoding="utf-8",
    )
    regions_path = directory / "one_step_region_metrics.jsonl"
    regions_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in result.region_metrics
        ),
        encoding="utf-8",
    )
    comparisons_path = directory / "one_step_bootstrap_comparisons.jsonl"
    comparisons_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result.comparisons),
        encoding="utf-8",
    )
    seed_metrics_path = directory / "one_step_training_seed_metrics.jsonl"
    seed_summary_path = directory / "one_step_training_seed_summary.json"
    seed_results = [(resolved.seed, result)]
    for seed in resolved.robustness_seeds:
        if seed == resolved.seed:
            continue
        seed_results.append(
            (
                seed,
                run_one_step_calibration(
                    replace(resolved, seed=seed, robustness_seeds=(seed,))
                ),
            )
        )
    seed_metric_rows = [
        {"training_seed": seed, **asdict(metric)}
        for seed, seed_result in seed_results
        for metric in seed_result.metrics
        if metric.policy in {"linear_value_model", "mlp_value_model", "gnn_coo"}
    ]
    seed_metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_metric_rows),
        encoding="utf-8",
    )
    independent_by_seed = {
        seed: {
            metric.policy: metric
            for metric in seed_result.metrics
            if metric.evaluation_panel == "independent_confirmation"
        }
        for seed, seed_result in seed_results
    }
    seed_summary = {
        "schema_version": "causal-observation-reproduction.one-step-seed-robustness.v1",
        "evidence_status": "independent_confirmation_pending_review",
        "training_seeds": [seed for seed, _ in seed_results],
        "training_seed_count": len(seed_results),
        "independent_confirmation_row_count": len(
            result.splits["independent_confirmation"]
        ),
        "linear_mae_below_mlp_count": sum(
            int(metrics["linear_value_model"].mae < metrics["mlp_value_model"].mae)
            for metrics in independent_by_seed.values()
        ),
        "linear_mae_below_gnn_count": sum(
            int(metrics["linear_value_model"].mae < metrics["gnn_coo"].mae)
            for metrics in independent_by_seed.values()
        ),
        "paper_level_conclusion": False,
    }
    seed_summary_path.write_text(
        json.dumps(seed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = directory / "one_step_calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.one-step-calibration.v1",
                "config": asdict(resolved),
                "feature_allowlist": FEATURE_NAMES,
                "label_denylist": sorted(LABEL_FIELDS),
                "split_group_key": "base_graph_id",
                "splits": result.splits,
                "model_backends": result.model_backends,
                "training_seed_robustness": seed_summary,
                "learning_row_count": len(result.splits["train"])
                + len(result.splits["validation"]),
                "historical_frozen_test_row_count": len(result.splits["frozen_test"]),
                "independent_confirmation_row_count": len(
                    result.splits["independent_confirmation"]
                ),
                "paper_design_status": "learning_count_reconciled",
                "evidence_status": "completed_with_independent_confirmation_pending_review",
                "historical_protocol_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        data_path,
        data_manifest_path,
        predictions_path,
        metrics_path,
        regions_path,
        comparisons_path,
        seed_metrics_path,
        seed_summary_path,
        manifest_path,
    )


def _validate_records(records: tuple[dict[str, object], ...]) -> None:
    if not records:
        _raise_invalid("one-step calibration corpus is empty")
    if set(FEATURE_NAMES) & LABEL_FIELDS:
        message = "label field present in public feature allowlist"
        raise AssertionError(message)
    for record in records:
        expected = (
            _as_float(record["gross_refinement_value"])
            - _as_float(record["observation_cost"])
            - _as_float(record["probe_cost"])
        )
        if not math.isclose(_as_float(record["net_refinement_value"]), expected):
            _raise_invalid(
                "net refinement value must subtract declared costs exactly once"
            )


def _assert_split_integrity(
    records: tuple[dict[str, object], ...], splits: dict[str, tuple[str, ...]]
) -> None:
    owner: dict[str, str] = {}
    group_by_id = {
        str(record["instance_id"]): str(record["base_graph_id"]) for record in records
    }
    for split, row_ids in splits.items():
        for row_id in row_ids:
            group = group_by_id[row_id]
            previous = owner.setdefault(group, split)
            if previous != split:
                _raise_invalid(f"group leakage across splits: {group}")


def _indices_by_id(records: tuple[dict[str, object], ...]) -> dict[str, int]:
    return {str(record["instance_id"]): index for index, record in enumerate(records)}


def _feature_matrix(records: tuple[dict[str, object], ...]) -> np.ndarray:
    return np.asarray(
        [[_as_float(record[name]) for name in FEATURE_NAMES] for record in records],
        dtype=np.float64,
    )


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    message = f"expected a numeric record value, received {type(value).__name__}"
    raise TypeError(message)


def _standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    return mean, np.where(scale < 1e-9, 1.0, scale)


def _ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray
) -> np.ndarray:
    mean, scale = _standardize_fit(x_train)
    train = (x_train - mean) / scale
    all_values = (x_all - mean) / scale
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    return np.asarray(
        np.column_stack((np.ones(len(all_values)), all_values)) @ coefficients,
        dtype=np.float64,
    )


def _learned_predictions(
    records: tuple[dict[str, object], ...],
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: OneStepLearningConfig,
    *,
    graph: bool,
) -> tuple[np.ndarray, ModelBackend]:
    try:
        torch = import_module("torch")
    except ImportError:
        return _ridge_predict(
            features[train_indices], targets[train_indices], features
        ), "ridge_fallback"
    return _torch_predictions(
        torch,
        records,
        features,
        targets,
        train_indices,
        validation_indices,
        config,
        graph=graph,
    ), "torch"


def _torch_predictions(
    torch: Any,
    records: tuple[dict[str, object], ...],
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: OneStepLearningConfig,
    *,
    graph: bool,
) -> np.ndarray:
    nn = torch.nn
    torch.manual_seed(config.seed)
    mean, scale = _standardize_fit(features[train_indices])
    normalized = ((features - mean) / scale).astype(np.float32)
    if graph:
        normalized = np.column_stack(
            (
                normalized,
                _graph_embeddings(records, layers=config.message_passing_layers),
            )
        ).astype(np.float32)
    all_features = torch.as_tensor(normalized)
    target_tensor = torch.as_tensor(targets.astype(np.float32))
    input_size = int(normalized.shape[1])
    head = nn.Sequential(
        nn.Linear(input_size, config.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.hidden_dim, 1),
    )
    sign = nn.Linear(input_size, 1)
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + list(sign.parameters()), lr=config.learning_rate
    )
    best_state: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_regret = math.inf
    wait = 0
    for _ in range(config.max_epochs):
        head.train()
        sign.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = head(all_features[train_indices]).squeeze(-1)
        sign_logits = sign(all_features[train_indices]).squeeze(-1)
        train_tensor = torch.as_tensor(train_indices)
        loss = torch.nn.functional.huber_loss(prediction, target_tensor[train_tensor])
        loss = (
            loss
            + config.auxiliary_sign_loss_weight
            * torch.nn.functional.binary_cross_entropy_with_logits(
                sign_logits, (target_tensor[train_tensor] > 0.0).float()
            )
        )
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            validation = np.asarray(
                [
                    float(value)
                    for value in head(all_features[validation_indices]).squeeze(-1)
                ],
                dtype=np.float64,
            )
        value = targets[validation_indices]
        regret = float(
            np.mean(np.maximum(0.0, value) - np.where(validation > 0.0, value, 0.0))
        )
        if regret + 1e-12 < best_regret:
            best_regret = regret
            wait = 0
            best_state = (deepcopy(head.state_dict()), deepcopy(sign.state_dict()))
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state[0])
        sign.load_state_dict(best_state[1])
    head.eval()
    with torch.no_grad():
        return np.asarray(
            [float(value) for value in head(all_features).squeeze(-1)], dtype=np.float64
        )


def _graph_tensors(
    records: tuple[dict[str, object], ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    role_index = {"action": 0, "context": 1, "mediator": 2, "proxy": 3, "outcome": 4}
    tensors: list[tuple[np.ndarray, np.ndarray]] = []
    for record in records:
        graph = json.loads(str(record["graph_json"]))
        nodes = list(graph["nodes"])
        node_index = {str(node["id"]): index for index, node in enumerate(nodes)}
        features = np.zeros((len(nodes), 7), dtype=np.float32)
        for index, node in enumerate(nodes):
            features[index, role_index.get(str(node.get("role")), 5)] = 1.0
            features[index, -1] = float(bool(node.get("query_member")))
        adjacency = np.eye(len(nodes), dtype=np.float32)
        for edge in graph["edges"]:
            source, target = str(edge["source"]), str(edge["target"])
            if source in node_index and target in node_index:
                adjacency[node_index[source], node_index[target]] = 1.0
                adjacency[node_index[target], node_index[source]] = 1.0
        tensors.append((features, adjacency))
    return tuple(tensors)


def _graph_embeddings(
    records: tuple[dict[str, object], ...], *, layers: int
) -> np.ndarray:
    """Produce a local, role-aware message-passing graph representation."""

    embeddings: list[np.ndarray] = []
    for node_features, adjacency in _graph_tensors(records):
        degree = np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
        hidden = node_features
        for _ in range(layers + 1):
            hidden = np.maximum(0.0, (adjacency / degree) @ hidden)
        embeddings.append(np.asarray(hidden.mean(axis=0), dtype=np.float64))
    return np.asarray(embeddings, dtype=np.float64)


def _formula_oracle_prediction(records: tuple[dict[str, object], ...]) -> np.ndarray:
    return np.asarray(
        [
            _as_float(record["observation_reliability"])
            - _as_float(record["observation_cost"])
            if bool(record["incompatible_optimal_actions"])
            else -_as_float(record["observation_cost"])
            for record in records
        ],
        dtype=np.float64,
    )


def _policy_metric(values: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actions = predictions > 0.0
    policy_value = np.where(actions, values, 0.0)
    positive = values > 0.0
    false_positive = actions & ~positive
    false_negative = ~actions & positive
    true_positive = int(np.sum(actions & positive))
    false_positive_count = int(np.sum(false_positive))
    false_negative_count = int(np.sum(false_negative))
    precision = true_positive / max(1, true_positive + false_positive_count)
    recall = true_positive / max(1, true_positive + false_negative_count)
    return {
        "eiv": float(policy_value.mean()),
        "mean_regret": float((np.maximum(0.0, values) - policy_value).mean()),
        "activation_rate": float(actions.mean()),
        "conditional_activation_value": float(policy_value[actions].mean())
        if actions.any()
        else 0.0,
        "false_positive_loss": float((-values[false_positive]).sum() / len(values)),
        "false_negative_loss": float(values[false_negative].sum() / len(values)),
        "mae": float(np.abs(predictions - values).mean()),
        "rmse": float(np.sqrt(np.mean((predictions - values) ** 2))),
        "sign_accuracy": float((actions == positive).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
    }


def _cluster_bootstrap_difference(
    records: tuple[dict[str, object], ...],
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    groups = sorted({str(record["base_graph_id"]) for record in records})
    values = np.asarray(
        [_as_float(record["net_refinement_value"]) for record in records],
        dtype=np.float64,
    )
    candidate = np.where(candidate_predictions > 0.0, values, 0.0)
    reference = np.where(reference_predictions > 0.0, values, 0.0)
    indices = {
        group: np.asarray(
            [
                index
                for index, record in enumerate(records)
                if str(record["base_graph_id"]) == group
            ],
            dtype=np.int64,
        )
        for group in groups
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([indices[str(group)] for group in sampled])
        estimates.append(float((candidate[rows] - reference[rows]).mean()))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _random_predictions(size: int, activation_rate: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(size) < activation_rate, 1.0, -1.0)


def _evaluate_policies(
    records: tuple[dict[str, object], ...],
    predictions: dict[str, np.ndarray],
    *,
    evaluation_indices: np.ndarray,
    evaluation_panel: EvaluationPanel,
    config: OneStepLearningConfig,
) -> tuple[
    tuple[PolicyMetric, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    evaluated = tuple(records[int(index)] for index in evaluation_indices)
    values = np.asarray(
        [_as_float(record["net_refinement_value"]) for record in evaluated],
        dtype=np.float64,
    )
    metrics: list[PolicyMetric] = []
    region_metrics: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    reference = predictions["never_refine"][evaluation_indices]
    for name in sorted(predictions):
        model_prediction = predictions[name][evaluation_indices]
        values_by_metric = _policy_metric(values, model_prediction)
        lower, upper = _cluster_bootstrap_difference(
            evaluated,
            model_prediction,
            reference,
            samples=config.bootstrap_samples,
            seed=config.seed
            + (0 if evaluation_panel == "historical_frozen_test" else 1_000),
        )
        metrics.append(
            PolicyMetric(
                policy=name,
                evaluation_panel=evaluation_panel,
                eiv_ci_lower=lower,
                eiv_ci_upper=upper,
                **values_by_metric,
            )
        )
        for region in ("failure", "null", "positive"):
            region_indices = np.asarray(
                [
                    index
                    for index, record in enumerate(evaluated)
                    if record["registered_region"] == region
                ],
                dtype=np.int64,
            )
            if len(region_indices):
                region_metrics.append(
                    {
                        "policy": name,
                        "evaluation_panel": evaluation_panel,
                        "region": region,
                        **_policy_metric(
                            values[region_indices], model_prediction[region_indices]
                        ),
                    }
                )
        if name != "never_refine":
            lower, upper = _cluster_bootstrap_difference(
                evaluated,
                model_prediction,
                reference,
                samples=config.bootstrap_samples,
                seed=config.seed
                + (3 if evaluation_panel == "historical_frozen_test" else 1_003),
            )
            comparisons.append(
                {
                    "policy": name,
                    "reference": "never_refine",
                    "evaluation_panel": evaluation_panel,
                    "eiv_difference_ci_lower": lower,
                    "eiv_difference_ci_upper": upper,
                }
            )
    return tuple(metrics), tuple(region_metrics), tuple(comparisons)


__all__ = [
    "FEATURE_NAMES",
    "LABEL_FIELDS",
    "ModelBackend",
    "OneStepCalibrationResult",
    "OneStepLearningConfig",
    "PolicyMetric",
    "build_grouped_splits",
    "run_one_step_calibration",
    "write_one_step_calibration",
]

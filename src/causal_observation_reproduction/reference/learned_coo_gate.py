"""Bounded learned approximation to the COO coarse-versus-refine decision.

This module deliberately models one static decision slice.  It rematerializes the
deterministic theorem worlds when the publication registry has only the run
configuration, but never treats theorem-only labels as controller features.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from causal_observation_reproduction.reference.sequential_theorem_benchmark import (
    NULL_PANEL,
    PATHOLOGY_PANEL,
    POSITIVE_PANEL,
    build_world,
    evaluate_world,
)
from causal_observation_reproduction.reference.topology import (
    BOUNDARY_MEDIATOR_MOTIF,
    COLLIDER_CONFOUNDED_MOTIF,
    ENDPOINT_DIRECT_MOTIF,
    causal_observability_topology_by_motif,
)

SCHEMA_VERSION = "causal_observation_reproduction.learned_coo_gate.v1"
UPSTREAM_REVISION = "b4c468ef4dfac5eeab8b11a9e7f990ba36824748"
TARGET_COLUMN = "net_refinement_value"
LABEL_DENYLIST = frozenset(
    {
        "V_coarse",
        "V_refined",
        "gross_refinement_value",
        "net_refinement_value",
        "sequential_value_gap_delta",
        "oracle_action",
        "registered_region",
        "fine_observation_value",
        "continuation_action_gap",
        "incompatible_optimal_actions",
        "positive_reachability",
    }
)
MODEL_FEATURE_ALLOWLIST = (
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
PANEL_TO_REGION = {
    POSITIVE_PANEL: "positive",
    NULL_PANEL: "null",
    PATHOLOGY_PANEL: "failure",
}
PANEL_TO_MOTIF = {
    POSITIVE_PANEL: BOUNDARY_MEDIATOR_MOTIF,
    NULL_PANEL: ENDPOINT_DIRECT_MOTIF,
    PATHOLOGY_PANEL: COLLIDER_CONFOUNDED_MOTIF,
}


@dataclass(frozen=True)
class LearnedCOOGateConfig:
    output_dir: str = "artifacts/learned_coo"
    seed: int = 17
    development_seeds: tuple[int, ...] = tuple(range(1001, 1061))
    validation_fraction: float = 0.2
    frozen_test_seeds: tuple[int, ...] = tuple(range(2001, 2021))
    historical_primary_seeds: tuple[int, ...] = tuple(range(1, 9))
    historical_confirmation_seeds: tuple[int, ...] = tuple(range(101, 109))
    null_primary_seeds: tuple[int, ...] = tuple(range(1, 7))
    null_confirmation_seeds: tuple[int, ...] = tuple(range(201, 207))
    pathology_primary_seeds: tuple[int, ...] = tuple(range(1, 7))
    rollouts: int = 512
    hidden_dim: int = 24
    message_passing_layers: int = 2
    dropout: float = 0.0
    learning_rate: float = 0.01
    max_epochs: int = 160
    early_stopping_patience: int = 24
    auxiliary_sign_loss_weight: float = 0.1
    bootstrap_samples: int = 400
    device: str = "cpu"
    claim_id: str = "foundation_h2_sparse_poscm_rl_identifiability"
    experiment_id: str = "learned_coo_gate"
    claim_text: str = (
        "A query-conditioned learned model can approximate cost-adjusted COO "
        "refinement decisions under the declared theorem-POSCM design distribution."
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _observable_graph(panel: str) -> dict[str, Any]:
    """Serialize only nodes and edges present in the declared coarse observation map."""

    spec = causal_observability_topology_by_motif(PANEL_TO_MOTIF[panel])
    observed_nodes = [node for node in spec.graph.nodes if node.observed]
    observed_ids = {node.node_id for node in observed_nodes}
    return {
        "nodes": [
            {
                "id": node.node_id,
                "role": node.role,
                "query_member": node.role == "action",
            }
            for node in observed_nodes
        ],
        "edges": [
            {"source": edge.source, "target": edge.target, "type": "causal"}
            for edge in spec.graph.edges
            if edge.source in observed_ids and edge.target in observed_ids
        ],
    }


def _graph_summary(graph: Mapping[str, Any]) -> dict[str, float]:
    nodes = list(graph.get("nodes") or [])
    roles = [str(node.get("role") or "") for node in nodes]
    return {
        "observable_node_count": float(len(nodes)),
        "observable_edge_count": float(len(list(graph.get("edges") or []))),
        "observable_action_count": float(sum(role == "action" for role in roles)),
        "observable_mediator_count": float(sum(role == "mediator" for role in roles)),
        "observable_proxy_count": float(sum(role == "proxy" for role in roles)),
        "observable_outcome_count": float(sum(role == "outcome" for role in roles)),
    }


class LearnedCOODatasetBuilder:
    """Materializes paired coarse/refined theorem evaluations into an auditable dataset."""

    def __init__(self, config: LearnedCOOGateConfig) -> None:
        self.config = config

    def materialize(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        panels: list[tuple[str, str, tuple[int, ...]]] = [
            ("development", POSITIVE_PANEL, self.config.development_seeds),
            ("development", NULL_PANEL, self.config.development_seeds),
            ("development", PATHOLOGY_PANEL, self.config.development_seeds),
            ("primary", POSITIVE_PANEL, self.config.historical_primary_seeds),
            ("primary", NULL_PANEL, self.config.null_primary_seeds),
            ("primary", PATHOLOGY_PANEL, self.config.pathology_primary_seeds),
            ("confirmation", POSITIVE_PANEL, self.config.historical_confirmation_seeds),
            ("confirmation", NULL_PANEL, self.config.null_confirmation_seeds),
            ("frozen_test", POSITIVE_PANEL, self.config.frozen_test_seeds),
            ("frozen_test", NULL_PANEL, self.config.frozen_test_seeds),
            ("frozen_test", PATHOLOGY_PANEL, self.config.frozen_test_seeds),
        ]
        rows: list[dict[str, Any]] = []
        graph_rows: dict[str, Any] = {}
        for experiment_panel, theorem_panel, seeds in panels:
            graph = _observable_graph(theorem_panel)
            graph_id = f"observable-{PANEL_TO_MOTIF[theorem_panel]}"
            graph_rows[graph_id] = graph
            for seed in seeds:
                world = build_world(theorem_panel, int(seed))
                result = evaluate_world(world, rollouts=self.config.rollouts)
                gross = float(result.exact_gross_gap)
                net = float(result.exact_net_gap)
                metric_is_net = theorem_panel == PATHOLOGY_PANEL
                raw_metric = net if metric_is_net else gross
                features = {
                    "observation_reliability": float(world.observation_reliability),
                    "observation_cost": float(world.observation_cost),
                    "probe_cost": 0.0,
                    "observation_regime_coarse": 1.0,
                    "query_continuation_action": 1.0,
                    **_graph_summary(graph),
                }
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "instance_id": f"{experiment_panel}:{theorem_panel}:seed-{seed}",
                    "base_graph_id": f"{experiment_panel}:{theorem_panel}:seed-{seed}",
                    "source_graph_id": graph_id,
                    "target_graph_id": graph_id,
                    "perturbation_family_id": theorem_panel,
                    "motif_family": PANEL_TO_MOTIF[theorem_panel],
                    "generator_seed": int(seed),
                    "query_id": "continuation_action",
                    "observation_regime_id": "coarse",
                    "experiment_panel": experiment_panel,
                    "eligibility_status": "eligible",
                    "graph_json": json.dumps(graph, sort_keys=True),
                    **features,
                    # Label-side fields.  These must never be model inputs.
                    "V_coarse": float(result.coarse_value),
                    "V_refined": float(result.fine_gross_value),
                    "gross_refinement_value": gross,
                    "observation_cost_included_in_raw_metric": metric_is_net,
                    "probe_cost_included_in_raw_metric": False,
                    "net_refinement_value": net,
                    "oracle_action": "refine" if net > 0.0 else "coarse",
                    "registered_region": PANEL_TO_REGION[theorem_panel],
                    "sequential_value_gap_delta": raw_metric,
                    "sequential_metric_name": (
                        "sequential_value_gap_delta_pathology"
                        if metric_is_net
                        else "sequential_value_gap_delta"
                    ),
                    "rollout_count": int(self.config.rollouts),
                    "continuation_action_gap": float(result.continuation_action_gap),
                    "positive_reachability": bool(result.positive_reachability),
                    "incompatible_optimal_actions": bool(
                        result.incompatible_optimal_actions
                    ),
                }
                rows.append(row)
        frame = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
        self._validate_frame(frame)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_revision": UPSTREAM_REVISION,
            "source_artifacts": [
                "configs/reference/one_step_gate.json",
                "reference/sequential_theorem_benchmark.py",
            ],
            "source_artifact_status": "final run configuration and deterministic evaluator; historical per-instance result artifacts absent from registry",
            "materialization_case": "B",
            "cost_semantics": {
                "sequential_value_gap_delta": "gross_refinement_value",
                "sequential_value_gap_delta_pathology": "net_refinement_value",
                "net_refinement_value": "gross_refinement_value - observation_cost - probe_cost",
            },
            "feature_allowlist": list(MODEL_FEATURE_ALLOWLIST),
            "label_denylist": sorted(LABEL_DENYLIST),
            "row_count": len(frame),
            "graph_count": len(graph_rows),
            "counts_by_region": frame["registered_region"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "counts_by_family": frame["motif_family"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "config_hash": _sha256(asdict(self.config)),
        }
        manifest["graphs"] = graph_rows
        return frame, manifest

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("learned COO dataset is empty")
        expected = (
            frame["gross_refinement_value"]
            - frame["observation_cost"]
            - frame["probe_cost"]
        )
        if not np.allclose(frame["net_refinement_value"], expected):
            raise ValueError(
                "net refinement value does not subtract costs exactly once"
            )
        for raw in frame.to_dict("records"):
            graph = json.loads(str(raw["graph_json"]))
            if any("latent" in str(node["id"]).lower() for node in graph["nodes"]):
                raise ValueError("latent node leaked into observable graph")


def build_grouped_splits(
    frame: pd.DataFrame, *, config: LearnedCOOGateConfig
) -> dict[str, list[str]]:
    frozen = frame.loc[
        frame["experiment_panel"] == "frozen_test", "instance_id"
    ].tolist()
    development = frame.loc[frame["experiment_panel"] == "development"].copy()
    groups = sorted(development["base_graph_id"].unique().tolist())
    rng = random.Random(config.seed)
    rng.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * config.validation_fraction)))
    validation_groups = set(groups[:validation_count])
    validation = development.loc[
        development["base_graph_id"].isin(validation_groups), "instance_id"
    ].tolist()
    train = development.loc[
        ~development["base_graph_id"].isin(validation_groups), "instance_id"
    ].tolist()
    splits = {
        "train": sorted(train),
        "validation": sorted(validation),
        "frozen_test": sorted(frozen),
    }
    assert_split_overlap_free(frame, splits)
    return splits


def assert_split_overlap_free(
    frame: pd.DataFrame, splits: Mapping[str, Iterable[str]]
) -> None:
    ownership: dict[str, str] = {}
    indexed = frame.set_index("instance_id")
    for split, ids in splits.items():
        for instance_id in ids:
            group = str(indexed.loc[str(instance_id), "base_graph_id"])
            previous = ownership.setdefault(group, str(split))
            if previous != str(split):
                raise ValueError(
                    f"group leakage across splits: {group} in {previous} and {split}"
                )


def model_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    unexpected = set(MODEL_FEATURE_ALLOWLIST) & LABEL_DENYLIST
    if unexpected:
        raise AssertionError(
            f"label columns in feature allowlist: {sorted(unexpected)}"
        )
    return cast(
        "np.ndarray",
        frame.loc[:, list(MODEL_FEATURE_ALLOWLIST)].to_numpy(dtype=np.float64),
    )


def _standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    return mean, np.where(scale < 1e-9, 1.0, scale)


def _ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    mean, scale = _standardize_fit(x_train)
    train = (x_train - mean) / scale
    all_values = (x_all - mean) / scale
    design = np.column_stack([np.ones(len(train)), train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
    return cast(
        "np.ndarray",
        np.column_stack([np.ones(len(all_values)), all_values]) @ coefficients,
    )


def _graph_tensors(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    role_index = {"action": 0, "context": 1, "mediator": 2, "proxy": 3, "outcome": 4}
    tensors: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_graph in frame["graph_json"]:
        graph = json.loads(str(raw_graph))
        nodes = list(graph["nodes"])
        node_ids = [str(node["id"]) for node in nodes]
        index = {node_id: i for i, node_id in enumerate(node_ids)}
        x = np.zeros((len(nodes), len(role_index) + 2), dtype=np.float32)
        for i, node in enumerate(nodes):
            x[i, role_index.get(str(node.get("role")), 5)] = 1.0
            x[i, -1] = float(bool(node.get("query_member")))
        adjacency = np.eye(len(nodes), dtype=np.float32)
        for edge in graph["edges"]:
            source, target = str(edge["source"]), str(edge["target"])
            if source in index and target in index:
                adjacency[index[source], index[target]] = 1.0
                adjacency[index[target], index[source]] = 1.0
        tensors.append((x, adjacency))
    return tensors


def _torch_predictions(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    config: LearnedCOOGateConfig,
    *,
    graph: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from torch import nn
    except ImportError:
        return _ridge_predict(x[train_idx], y[train_idx], x), {
            "backend": "ridge_fallback",
            "reason": "torch_unavailable",
        }
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cpu" if config.device == "auto" else config.device)
    mean, scale = _standardize_fit(x[train_idx])
    all_x = torch.as_tensor(((x - mean) / scale).astype(np.float32), device=device)
    targets = torch.as_tensor(y.astype(np.float32), device=device)
    tensors = _graph_tensors(frame) if graph else []

    class Encoder(nn.Module):
        def __init__(self, node_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.Linear(node_dim if i == 0 else hidden_dim, hidden_dim)
                    for i in range(config.message_passing_layers + 1)
                ]
            )

        def forward(self, node_x: Any, adjacency: Any) -> Any:
            degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
            hidden = node_x
            for layer in self.layers:
                hidden = torch.relu(layer((adjacency / degree) @ hidden))
            return hidden.mean(dim=0)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder(7, config.hidden_dim) if graph else None
            input_dim = x.shape[1] + (config.hidden_dim if graph else 0)
            self.head = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
            self.sign = nn.Linear(input_dim, 1)

        def forward(self, index: int) -> tuple[Any, Any]:
            global_x = all_x[index]
            if self.encoder is None:
                features = global_x
            else:
                node_x, adjacency = tensors[index]
                embedding = self.encoder(
                    torch.as_tensor(node_x, device=device),
                    torch.as_tensor(adjacency, device=device),
                )
                features = torch.cat([embedding, global_x])
            return self.head(features).squeeze(-1), self.sign(features).squeeze(-1)

    model = Net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, Any] | None = None
    best_regret = math.inf
    wait = 0
    loss_history: list[float] = []
    for _ in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted, logits = zip(*(model(int(i)) for i in train_idx), strict=True)
        pred = torch.stack(predicted)
        sign_logits = torch.stack(logits)
        loss = torch.nn.functional.huber_loss(
            pred, targets[torch.as_tensor(train_idx, device=device)]
        )
        loss = (
            loss
            + config.auxiliary_sign_loss_weight
            * torch.nn.functional.binary_cross_entropy_with_logits(
                sign_logits,
                (targets[torch.as_tensor(train_idx, device=device)] > 0).float(),
            )
        )
        loss.backward()  # type: ignore[no-untyped-call]  # torch lacks complete typing
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_predictions = np.asarray(
                [float(model(int(i))[0].cpu()) for i in validation_idx]
            )
        regret = float(
            np.mean(
                np.maximum(0.0, y[validation_idx])
                - np.where(validation_predictions > 0.0, y[validation_idx], 0.0)
            )
        )
        if regret + 1e-12 < best_regret:
            best_regret, wait = regret, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = np.asarray([float(model(i)[0].cpu()) for i in range(len(frame))])
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / ("gnn_coo.pt" if graph else "mlp_value_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_allowlist": MODEL_FEATURE_ALLOWLIST,
        },
        checkpoint_path,
    )
    return prediction, {
        "backend": "torch",
        "epochs": len(loss_history),
        "best_validation_regret": best_regret,
        "loss_final": loss_history[-1] if loss_history else 0.0,
        "checkpoint": str(checkpoint_path),
    }


def _formula_oracle_prediction(frame: pd.DataFrame) -> np.ndarray:
    """Theorem-oracle comparator; it intentionally uses withheld diagnostic labels."""

    # This is not a deployed feature path. It represents the theorem condition
    # already evaluated by the simulator and is labelled oracle in every artifact.
    compatible = frame["incompatible_optimal_actions"].astype(bool).to_numpy()
    return cast(
        "np.ndarray",
        np.where(
            compatible,
            frame["observation_reliability"].to_numpy()
            - frame["observation_cost"].to_numpy(),
            -frame["observation_cost"].to_numpy(),
        ),
    )


def policy_metrics(
    values: np.ndarray, predictions: np.ndarray, *, threshold: float = 0.0
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    actions = predictions > threshold
    policy_value = np.where(actions, values, 0.0)
    positive = values > 0.0
    false_positive = actions & ~positive
    false_negative = ~actions & positive
    tp, fp, fn = (
        int(np.sum(actions & positive)),
        int(np.sum(false_positive)),
        int(np.sum(false_negative)),
    )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "eiv": float(policy_value.mean()),
        "mean_regret": float((np.maximum(0.0, values) - policy_value).mean()),
        "activation_rate": float(actions.mean()),
        "conditional_activation_value": float(policy_value[actions].mean())
        if actions.any()
        else 0.0,
        "false_positive_loss": float((-values[false_positive]).sum() / len(values)),
        "false_positive_loss_conditional": float((-values[false_positive]).mean())
        if false_positive.any()
        else 0.0,
        "false_negative_loss": float(values[false_negative].sum() / len(values)),
        "false_negative_loss_conditional": float(values[false_negative].mean())
        if false_negative.any()
        else 0.0,
        "mae": float(np.abs(predictions - values).mean()),
        "rmse": float(np.sqrt(np.mean((predictions - values) ** 2))),
        "sign_accuracy": float((actions == positive).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
    }


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    groups = frame["base_graph_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    candidate = np.where(
        candidate_predictions > 0.0, frame[TARGET_COLUMN].to_numpy(), 0.0
    )
    reference = np.where(
        reference_predictions > 0.0, frame[TARGET_COLUMN].to_numpy(), 0.0
    )
    indices = {
        group: np.flatnonzero(frame["base_graph_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    estimates = []
    for _ in range(max(1, samples)):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([indices[str(group)] for group in chosen])
        estimates.append(float((candidate[rows] - reference[rows]).mean()))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _random_predictions(size: int, activation_rate: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(size) < activation_rate, 1.0, -1.0)


class LearnedCOOGateExperiment:
    def __init__(self, config: LearnedCOOGateConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)

    def audit_inputs(self) -> dict[str, Any]:
        return {
            "source_revision": UPSTREAM_REVISION,
            "evaluator": "reference/sequential_theorem_benchmark.py",
            "metric_semantics": "positive/null sequential_value_gap_delta is gross; pathology metric is net",
            "per_instance_graphs_stored": False,
            "paired_values_stored": False,
            "rerun_case": "B",
            "paper_stale_pilot": True,
        }

    def materialize_dataset(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        return LearnedCOODatasetBuilder(self.config).materialize()

    def build_grouped_splits(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        return build_grouped_splits(frame, config=self.config)

    def fit_baselines(
        self, frame: pd.DataFrame, splits: Mapping[str, list[str]]
    ) -> dict[str, np.ndarray]:
        indexed = {value: index for index, value in enumerate(frame["instance_id"])}
        train_idx = np.asarray(
            [indexed[value] for value in splits["train"]], dtype=np.int64
        )
        x, y = (
            model_feature_matrix(frame),
            frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
        )
        linear = _ridge_predict(x[train_idx], y[train_idx], x)
        return {
            "never_refine": np.full(len(frame), -1.0),
            "always_refine": np.full(len(frame), 1.0),
            "linear_value_model": linear,
            "formula_coo_oracle_statistic": _formula_oracle_prediction(frame),
            "oracle_coo": y.copy(),
        }

    def fit_gnn(
        self, frame: pd.DataFrame, splits: Mapping[str, list[str]], *, graph: bool
    ) -> tuple[np.ndarray, dict[str, Any]]:
        indexed = {value: index for index, value in enumerate(frame["instance_id"])}
        train_idx = np.asarray(
            [indexed[value] for value in splits["train"]], dtype=np.int64
        )
        validation_idx = np.asarray(
            [indexed[value] for value in splits["validation"]], dtype=np.int64
        )
        return _torch_predictions(
            frame,
            model_feature_matrix(frame),
            frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
            train_idx,
            validation_idx,
            self.config,
            graph=graph,
        )

    def evaluate_policies(
        self, frame: pd.DataFrame, predictions: Mapping[str, np.ndarray]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test = frame.loc[frame["experiment_panel"] == "frozen_test"].reset_index(
            drop=True
        )
        test_indices = frame.index[
            frame["experiment_panel"] == "frozen_test"
        ].to_numpy()
        values = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
        table_rows: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        region_rows: list[dict[str, Any]] = []
        learned_rate = float((predictions["gnn_coo"][test_indices] > 0).mean())
        predictions = dict(predictions)
        predictions["budget_matched_random"] = _random_predictions(
            len(frame), learned_rate, seed=self.config.seed + 91
        )
        for name, all_predictions in predictions.items():
            pred = np.asarray(all_predictions)[test_indices]
            metrics = policy_metrics(values, pred)
            lower, upper = cluster_bootstrap_difference(
                test,
                pred,
                predictions["never_refine"][test_indices],
                samples=self.config.bootstrap_samples,
                seed=self.config.seed,
            )
            table_rows.append(
                {
                    "policy": name,
                    **metrics,
                    "eiv_ci_lower": lower,
                    "eiv_ci_upper": upper,
                }
            )
            for region, subset in test.groupby("registered_region", sort=True):
                subset_idx = subset.index.to_numpy()
                region_rows.append(
                    {
                        "policy": name,
                        "region": region,
                        **policy_metrics(values[subset_idx], pred[subset_idx]),
                    }
                )
            if name != "never_refine":
                lower, upper = cluster_bootstrap_difference(
                    test,
                    pred,
                    predictions["never_refine"][test_indices],
                    samples=self.config.bootstrap_samples,
                    seed=self.config.seed + 3,
                )
                comparisons.append(
                    {
                        "policy": name,
                        "reference": "never_refine",
                        "eiv_difference_ci_lower": lower,
                        "eiv_difference_ci_upper": upper,
                    }
                )
        return (
            pd.DataFrame(table_rows),
            pd.DataFrame(region_rows),
            pd.DataFrame(comparisons),
        )

    def generate_artifacts(
        self,
        frame: pd.DataFrame,
        manifest: Mapping[str, Any],
        splits: Mapping[str, list[str]],
        predictions: Mapping[str, np.ndarray],
        metrics: pd.DataFrame,
        region_metrics: pd.DataFrame,
        comparisons: pd.DataFrame,
        model_info: Mapping[str, Any],
    ) -> None:
        datasets = self.output_dir / "datasets"
        (
            splits_dir,
            metrics_dir,
            figures_dir,
            tables_dir,
            manifests_dir,
            predictions_dir,
        ) = (
            self.output_dir / name
            for name in (
                "splits",
                "metrics",
                "figures",
                "tables",
                "manifests",
                "predictions",
            )
        )
        for path in (
            datasets,
            splits_dir,
            metrics_dir,
            figures_dir,
            tables_dir,
            manifests_dir,
            predictions_dir,
            self.output_dir / "configs",
            self.output_dir / "checkpoints",
        ):
            path.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(datasets / "learned_coo_examples.parquet", index=False)
        (datasets / "learned_coo_graphs.json").write_text(
            json.dumps(manifest["graphs"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        schema = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        (datasets / "dataset_schema.json").write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_payload = dict(manifest)
        manifest_payload.pop("graphs", None)
        (datasets / "dataset_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_manifest = {
            "group_key": "base_graph_id",
            "seed": self.config.seed,
            "splits": splits,
            "hash": _sha256(splits),
        }
        (splits_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for split, split_values in splits.items():
            (splits_dir / f"{split}_ids.json").write_text(
                json.dumps(split_values, indent=2) + "\n", encoding="utf-8"
            )
        leakage = {
            "feature_allowlist": list(MODEL_FEATURE_ALLOWLIST),
            "label_denylist": sorted(LABEL_DENYLIST),
            "intersection": sorted(set(MODEL_FEATURE_ALLOWLIST) & LABEL_DENYLIST),
            "observable_graph_only": True,
            "split_overlap_free": True,
            "status": "passed",
        }
        (self.output_dir / "leakage_audit.json").write_text(
            json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, prediction_values in predictions.items():
            pd.DataFrame(
                {
                    "instance_id": frame["instance_id"],
                    "prediction": prediction_values,
                    "is_frozen_test": frame["experiment_panel"].eq("frozen_test"),
                }
            ).to_parquet(predictions_dir / f"{name}.parquet", index=False)
        metrics.to_json(metrics_dir / "policy_metrics.json", orient="records", indent=2)
        metrics.to_csv(metrics_dir / "policy_metrics.csv", index=False)
        region_metrics.to_csv(metrics_dir / "region_metrics.csv", index=False)
        region_metrics.rename(columns={"region": "family"}).to_csv(
            metrics_dir / "family_metrics.csv", index=False
        )
        comparisons.to_csv(metrics_dir / "bootstrap_comparisons.csv", index=False)
        metrics.to_csv(tables_dir / "learned_coo_policy_table.csv", index=False)
        (self.output_dir / "configs" / "learned_coo_gate_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (manifests_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "model_info": dict(model_info),
                    "dataset_hash": _sha256(manifest_payload),
                    "source_revision": UPSTREAM_REVISION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_figure(
            frame, predictions, figures_dir / "learned_coo_gate_main.png"
        )
        self._write_report(metrics, model_info)

    def _write_figure(
        self, frame: pd.DataFrame, predictions: Mapping[str, np.ndarray], output: Path
    ) -> None:
        import matplotlib.pyplot as plt

        test = frame[frame["experiment_panel"] == "frozen_test"]
        indices = test.index.to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
        axes[0].scatter(
            predictions["gnn_coo"][indices],
            test[TARGET_COLUMN],
            s=18,
            alpha=0.75,
            color="#1f4e79",
        )
        limits = [
            min(
                float(test[TARGET_COLUMN].min()),
                float(predictions["gnn_coo"][indices].min()),
            ),
            max(
                float(test[TARGET_COLUMN].max()),
                float(predictions["gnn_coo"][indices].max()),
            ),
        ]
        axes[0].plot(limits, limits, color="#555555", linewidth=1)
        axes[0].set(
            xlabel="Predicted net refinement value",
            ylabel="Realized net refinement value",
            title="Held-out calibration",
        )
        names = [
            "never_refine",
            "always_refine",
            "budget_matched_random",
            "linear_value_model",
            "mlp_value_model",
            "formula_coo_oracle_statistic",
            "gnn_coo",
            "oracle_coo",
        ]
        values = []
        for name in names:
            prediction = predictions.get(name)
            if prediction is None:
                continue
            values.append(
                (
                    name,
                    policy_metrics(test[TARGET_COLUMN].to_numpy(), prediction[indices])[
                        "eiv"
                    ],
                )
            )
        axes[1].bar(
            [name.replace("_", " ") for name, _ in values],
            [value for _, value in values],
            color="#3a7d44",
        )
        axes[1].tick_params(axis="x", rotation=60, labelsize=7)
        axes[1].set(ylabel="Expected intervention value", title="Held-out policy value")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

    def _write_report(
        self, metrics: pd.DataFrame, model_info: Mapping[str, Any]
    ) -> None:
        by_policy = metrics.set_index("policy")
        gnn = by_policy.loc["gnn_coo"]
        linear = by_policy.loc["linear_value_model"]
        report = f"""# Learned COO Gate Report

## Question

Can an observable-graph, query, and cost conditioned model approximate the static coarse-versus-refine COO action on frozen theorem-POSCM families?

## Target and Boundary

The regression target is `net_refinement_value = gross_refinement_value - observation_cost - probe_cost`. Model features are restricted to the coarse observable graph, query, observation reliability, and declared costs. The raw positive/null `sequential_value_gap_delta` is gross; the pathology raw metric is already net.

## Result

Frozen-panel GNN EIV: `{gnn.eiv:.6f}` (paired cluster bootstrap difference from never-refine [{gnn.eiv_ci_lower:.6f}, {gnn.eiv_ci_upper:.6f}]). GNN regret: `{gnn.mean_regret:.6f}`. Linear-model EIV: `{linear.eiv:.6f}`. GNN training backend: `{model_info.get("gnn", {}).get("backend", "unknown")}`. The GNN does not improve EIV over the linear value model in this deterministic panel.

## Scope

This is a static learned gate, not a sequential meta-RL policy, causal-discovery system, or base-RL sample-efficiency result. The formula comparator is explicitly theorem-oracle because it uses withheld continuation-action compatibility.

## Reproduction

`./scripts/reproduce.sh one-step` or
`powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 one-step`
"""
        (self.output_dir / "learned_coo_gate_report.md").write_text(
            report, encoding="utf-8"
        )
        audit = """# Learned COO Artifact Audit

## Final-result status

The source-derived configuration defines primary positive seeds 1--8 and confirmation seeds 101--108, together with null and pathology controls. The public run rematerializes those deterministic theorem-world rows locally.

## Metric semantics

`sequential_value_gap_delta` in `sequential_theorem_benchmark.py` is `exact_gross_gap = fine_gross_value - coarse_value`; observation cost is not included. `sequential_value_gap_delta_pathology` is `exact_net_gap`, which has already subtracted observation cost. This experiment preserves the raw metric and constructs canonical gross/net fields with a double-counting assertion.

## Rerun scope

This is Case B: the final configuration and deterministic paired evaluator are durable, but per-instance payloads are not. Only per-instance theorem-world materialization is rerun. Lean, MiniGrid, historical transfer suites, and sequential probing are not rerun. A disjoint learned-gate frozen panel is generated from new deterministic seeds.
"""
        (self.output_dir / "learned_coo_audit.md").write_text(audit, encoding="utf-8")
        (self.output_dir / "rerun_scope_report.md").write_text(
            "# Rerun Scope\n\nOnly deterministic paired coarse/refined theorem-world evaluation was rematerialized. No Lean proof, MiniGrid run, or historical transfer suite was rerun.\n",
            encoding="utf-8",
        )

    def write_evidence(self, metrics: pd.DataFrame) -> dict[str, Any]:
        gnn_backend = str(
            json.loads(
                (self.output_dir / "manifests" / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            .get("model_info", {})
            .get("gnn", {})
            .get("backend")
            or "unknown"
        )
        gnn_eiv = float(metrics.loc[metrics.policy.eq("gnn_coo"), "eiv"].iloc[0])
        gnn_ci_lower = float(
            metrics.loc[metrics.policy.eq("gnn_coo"), "eiv_ci_lower"].iloc[0]
        )
        claim_status = (
            "supported_with_scope"
            if gnn_backend == "torch" and gnn_eiv > 0.0 and gnn_ci_lower > 0.0
            else "exploratory_not_supported"
        )
        evidence = {
            "schema_version": "causal_observation_reproduction.learned_coo_gate_evidence.v1",
            "source_revision": UPSTREAM_REVISION,
            "claim_id": self.config.claim_id,
            "experiment_id": self.config.experiment_id,
            "claim": self.config.claim_text,
            "claim_status": claim_status,
            "limitations": [
                "synthetic inspectable POSCM setting",
                "static coarse-versus-refine slice",
                "no sequential active probing",
                "no base-RL sample-efficiency claim",
            ],
            "evidence_records": [
                {
                    "role": "frozen_heldout_metrics",
                    "gnn_backend": gnn_backend,
                    "policy_metrics": metrics.to_dict("records"),
                }
            ],
        }
        path = self.output_dir / "learned_coo_claim_evidence.json"
        path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return evidence

    def run(self) -> dict[str, Any]:
        frame, manifest = self.materialize_dataset()
        splits = self.build_grouped_splits(frame)
        predictions = self.fit_baselines(frame, splits)
        predictions["mlp_value_model"], mlp_info = self.fit_gnn(
            frame, splits, graph=False
        )
        predictions["gnn_coo"], gnn_info = self.fit_gnn(frame, splits, graph=True)
        metrics, region_metrics, comparisons = self.evaluate_policies(
            frame, predictions
        )
        self.generate_artifacts(
            frame,
            manifest,
            splits,
            predictions,
            metrics,
            region_metrics,
            comparisons,
            {"mlp": mlp_info, "gnn": gnn_info},
        )
        evidence = self.write_evidence(metrics)
        return {
            "output_dir": str(self.output_dir),
            "row_count": len(frame),
            "policy_metrics": metrics.to_dict("records"),
            "evidence": evidence,
        }

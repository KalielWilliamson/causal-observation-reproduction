import json

from causal_observation_reproduction.reference.quotient_recovery.statistics import (
    clustered_noninferiority,
)
from causal_observation_reproduction.reference.sequential_experiment import (
    POLICIES,
    ConditionalEfficiencyConfirmationConfig,
    ConditionalEfficiencyPilotConfig,
    run_conditional_efficiency_confirmation,
    run_conditional_efficiency_pilot,
)


def test_clustered_noninferiority_resamples_topology_instances_not_episode_rows():
    result = clustered_noninferiority(
        differences=[0.10, 0.10, -0.01, -0.01],
        clusters=["topology-a", "topology-a", "topology-b", "topology-b"],
        margin=0.05,
        bootstrap_repetitions=64,
        seed=7,
    )
    assert result["cluster_count"] == 2
    assert result["paired_episode_count"] == 4
    assert result["noninferiority_margin"] == 0.05
    assert isinstance(result["noninferior"], bool)


def test_conditional_efficiency_pilot_is_paired_and_reports_declared_compute(tmp_path):
    result = run_conditional_efficiency_pilot(
        config=ConditionalEfficiencyPilotConfig(
            training_seeds=(1, 2, 3, 4),
            evaluation_seeds=(31, 32),
            rollout_budget=8,
            output_dir=str(tmp_path),
        )
    )
    assert {row["policy"] for row in result.rows} == set(POLICIES)
    assert {
        row["family_id"] for row in result.rows if row["policy"] == POLICIES[0]
    } == {row["family_id"] for row in result.rows if row["policy"] == POLICIES[1]}
    assert all(
        row["rollout_count"] == 0
        for row in result.rows
        if row["policy"] != "generic_bounded_rollout_voi"
    )
    assert all(
        row["rollout_count"] > 0
        for row in result.rows
        if row["policy"] == "generic_bounded_rollout_voi"
    )
    assert result.models["learned_causal_quotient"].uses_quotient
    assert not result.models["capacity_matched_no_quotient"].uses_quotient
    assert (
        result.models["learned_causal_quotient"].metadata()["parameter_count"]
        == result.models["capacity_matched_no_quotient"].metadata()["parameter_count"]
    )
    assert {row["policy"] for row in result.rows} >= {
        "never_refine",
        "always_refine",
        "information_gain",
        "oracle",
    }
    assert all(
        not row["uses_oracle"] for row in result.rows if row["policy"] != "oracle"
    )
    assert all(row["uses_oracle"] for row in result.rows if row["policy"] == "oracle")
    assert all(path.endswith(".json") for path in result.artifacts.values())


def test_generic_voi_counts_only_distinct_public_rollout_branches(tmp_path):
    result = run_conditional_efficiency_pilot(
        config=ConditionalEfficiencyPilotConfig(
            training_seeds=(1, 2),
            evaluation_seeds=(31,),
            rollout_budget=8,
            output_dir=str(tmp_path),
        )
    )
    rows = [
        row
        for row in result.aggregate
        if row["policy"] == "generic_bounded_rollout_voi"
        and row["regime"] == "positive"
    ]
    by_distractors = {
        row["distractor_count"]: row["mean_ambient_branch_count"] for row in rows
    }
    # Branch counts come from unique public observation traces actually
    # evaluated, not a synthetic 2**distractor_count calculation.
    # The pilot horizon is three; each decision may evaluate acquire and
    # coarse traces for at most the eight requested public rollout seeds.
    assert all(1 <= count <= 3 * 2 * 8 for count in by_distractors.values())


def test_confirmation_matrix_holds_out_topology_and_emits_locked_manifest(tmp_path):
    result = run_conditional_efficiency_confirmation(
        config=ConditionalEfficiencyConfirmationConfig(
            training_topology_families=("positive_boundary_advantage",),
            heldout_topology_families=("observability_scope_qualifier",),
            regimes=("positive", "null"),
            distractor_counts=(0, 2),
            horizons=(3,),
            probe_costs=(0.1,),
            generic_rollout_budgets=(1, 4),
            primary_generic_rollout_budget=4,
            topology_instances_per_family=2,
            cluster_bootstrap_repetitions=16,
            training_seeds=(1, 2, 5, 7),
            evaluation_seeds=(31, 32),
            output_dir=str(tmp_path),
        )
    )
    assert {row["split"] for row in result.rows} == {"heldout_topology"}
    assert {row["topology_family"] for row in result.rows} == {
        "observability_scope_qualifier"
    }
    assert len({row["topology_instance_id"] for row in result.rows}) == 2
    assert {
        row["generic_rollout_budget"]
        for row in result.rows
        if row["policy"] == "generic_bounded_rollout_voi"
    } == {1, 4}
    assert all(
        row["generic_rollout_budget"] == 0
        for row in result.rows
        if row["policy"] != "generic_bounded_rollout_voi"
    )
    assert {
        cost
        for _, cost, _ in result.models["learned_causal_quotient"].gain_by_feature_cost
    } == {0.1}
    assert tmp_path.joinpath("locked_confirmation_manifest.json").exists()
    assert tmp_path.joinpath("clustered_noninferiority.json").exists()
    assert tmp_path.joinpath("policy_metadata.json").exists()
    projection = json.loads(
        tmp_path.joinpath("conditional_efficiency_confirmation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert projection["schema_version"].endswith(".v1")
    assert {row["budget"] for row in projection["comparisons"]} == {1, 4}
    split = json.loads(
        tmp_path.joinpath("split_manifest.json").read_text(encoding="utf-8")
    )
    assert split["cluster_key"] == "topology_instance_id"
    assert split["training_topology_families"] == ["positive_boundary_advantage"]
    assert all(
        instance.startswith("observability_scope_qualifier:")
        for instance in split["heldout_topology_instance_ids"]
    )
    evidence = json.loads(
        tmp_path.joinpath("confirmation_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["assessment"] == "requires_frozen_review"
    assert evidence["claim_id"] == "causal_quotient_conditional_voi_efficiency"

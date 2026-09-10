// Outcome-blind contract for the finite certified quotient-cache control.
package cachecontrol

#CacheControl: {
	schema_version:         "causal-observation-reproduction.cache-control.v1"
	nuisance_cardinalities: [1, 4, 16]
	acquisition_cost:       0.25
	tie_policy:             "abstain"
	evaluation_domain:      "finite_planner_candidate_set"
	cache_reuse_policy:     "both_arms_cache_only_key_differs"
	valid_projection:       "signal"
	positive_evidence_requires: ["projection_certified", "decision_preserved"]
	invalid_control:        "same_projection_with_nuisance_dependent_action_values"
	dataset_layout: {
		manifest: "manifest.json"
		rows:     "rows.jsonl"
		summary:  "summary.json"
	}
}

cache_control_contract: #CacheControl

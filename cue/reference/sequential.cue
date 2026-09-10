package reference

// Static validation schema for the source-derived sequential JSON config.
#SequentialConfirmation: {
	training_topology_families: [...string] & [string, ...string]
	heldout_topology_families:  [...string] & [string, ...string]
	regimes: [...("positive" | "null" | "invalid")] & [_, ..._]
	distractor_counts: [...int & >=0] & [_, ..._]
	horizons: [...int & >=1] & [_, ..._]
	probe_costs: [...number & >=0] & [_, ..._]
	generic_rollout_budgets: [...int & >=1] & [_, ..._]
	primary_generic_rollout_budget: int & >=1
	topology_instances_per_family:  int & >=2
	noninferiority_margin:           number & >=0
	cluster_bootstrap_repetitions:   int & >=1
	policy_arms: [...string] & [string, ...string]
	cache_policy: string
	training_seeds: [...int] & [int, ...int]
	evaluation_seeds: [...int] & [int, ...int]
	output_dir: string
}

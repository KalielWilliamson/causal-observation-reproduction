package reference

// Static validation schema for the source-derived one-step JSON config.
#OneStepGate: {
	output_dir: string
	seed:       int
	development_seeds: [...int] & [int, ...int]
	validation_fraction: number & >0 & <1
	frozen_test_seeds: [...int] & [int, ...int]
	historical_primary_seeds: [...int] & [int, ...int]
	historical_confirmation_seeds: [...int] & [int, ...int]
	null_primary_seeds: [...int] & [int, ...int]
	null_confirmation_seeds: [...int] & [int, ...int]
	pathology_primary_seeds: [...int] & [int, ...int]
	rollouts:                 int & >=1
	hidden_dim:               int & >=1
	message_passing_layers:   int & >=1
	dropout:                  number & >=0 & <1
	learning_rate:             number & >0
	max_epochs:                int & >=1
	early_stopping_patience:   int & >=1
	auxiliary_sign_loss_weight: number & >=0
	bootstrap_samples:          int & >=1
	device:                     string
	claim_id:                   string
	experiment_id:              string
	claim_text:                 string
}

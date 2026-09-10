# Reference data

`minigrid_external_control.parquet` is the analysis-ready table for the 16
evaluated rows reported in the manuscript appendix. The equivalent JSON Lines
file is included for dependency-free inspection and the JSON summary is
recomputed from those rows in tests.

The source object SHA-256 is
`4bf02d1a5297792b66a6fed49171aae7c76a04b67f56434797a2acdd45031361`. The public
table retains the 42 scientific design and outcome fields needed to audit the
result and omits registry, storage, and orchestration columns.

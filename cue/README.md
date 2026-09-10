# CUE validation schemas

CUE is an optional static checker, not a configuration or orchestration layer.
The exact source-derived experiment values live in the checked JSON files under
`configs/reference/`; Python dataclasses define the runtime types.

`reference/sequential.cue` and `reference/one_step.cue` validate those JSON
shapes and scientific-domain constraints. `cache_control/protocol.cue` retains
the outcome-blind contract for the explicitly post-manuscript cache-control
extension.

Run the pinned CUE image used by CI:

```sh
bash scripts/reproduce.sh contract-check
```

There is deliberately no export step: generating a second copy of source
defaults would create an unnecessary authority and drift risk.

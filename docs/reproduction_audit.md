# Reproduction fidelity audit

## Conclusion

The earlier public implementations were scientifically useful clean-room models,
but they did not exactly match the implementation that generated the manuscript.
The canonical runner now executes a pinned extraction of the scientific source.
The clean-room and strengthened variants remain available only as explicitly
named extensions.

## Resolved discrepancies

| Area                   | Earlier public behavior                                                         | Pinned manuscript behavior                                                                      | Resolution                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Sequential matrix      | Several independently designed 5,760- or 49,152-row bundles                     | 152,064 emitted policy episodes; 4,608 paired gate/comparator executions per budget-regime cell | `sequential` now calls the exact source-derived grid and fail-closed parity verifier                                   |
| Sequential algorithm   | Full history/belief quotient planner was described as the paper implementation  | Six-cell threshold gate using a supplied query-disagreement Boolean; bounded rollout comparator | Exact gate is canonical; full planner is `extension-sequential-robustness`                                             |
| One-step cardinality   | Public repair produced 274 learning plus 60 historical and 300 independent rows | Source executable produces 214 non-test plus 60 frozen-test rows (274 total)                    | Exact source is canonical; repair is `extension-one-step-strengthened`; mismatch is documented                         |
| MiniGrid               | Recreated design and local PPO/DQN panels                                       | Exact 16 evaluated external-control rows in the paper artifact                                  | Exact rows are packaged and recomputed by `minigrid-audit`; training variants are extensions                           |
| Formal theory          | Smaller independent Bellman specifications                                      | Full manuscript-cited Lean closure plus transitive imports and compatibility checks             | Complete source corpus copied with namespace-only changes and pinned Mathlib                                           |
| Provenance             | Behavioral comparison without a source map                                      | Pinned revision plus source blob/tree hashes                                                    | `provenance/source_manifest.json` records every extracted module and allowed refactor                                  |
| Compute summary schema | Older paper summary omitted two non-applicable generic-gate counters            | Pinned implementation emits both counters explicitly as zero                                    | Parity projects onto all published counters and permits only additional zero-valued fields; any nonzero addition fails |

## Verification layers

1. Ported source tests exercise artifacts, domains, interventional signatures,
   SCM complexity, semantic worlds, quotient generation, sequential sensing,
   theorem panels, the contract, the locked experiment, and the learned gate.
2. Configuration tests compare checked JSON directly with typed dataclass
   defaults.
3. The sequential full runner compares every paper-facing cell and primary
   information action against immutable reference data. A complete local run
   regenerated all 152,064 rows and passed this check.
4. The one-step verifier checks the qualitative outcome statements and records
   the manuscript/source corpus-cardinality contradiction as a failed protocol
   claim rather than masking it.
5. MiniGrid summaries are recomputed from the exact 16 raw evaluated rows.
6. `lake build CausalObservationReproduction` checks the complete formal corpus.
7. The release scan forbids private infrastructure references, credentials,
   temporary extraction files, and legacy interface identifiers.

## Remaining publication boundary

The code and locally generated evidence can be made public without cloud
credentials. A durable data deposit and DOI are separate release actions: the
archive should contain the exact raw run directory, environment lockfiles,
source-provenance manifest, and evidence manifest. Until those files are
uploaded, `release/manifest.json` must keep the archive URL and DOI unset.

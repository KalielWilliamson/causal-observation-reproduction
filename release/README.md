# Release evidence boundary

This directory identifies the exact protocol and provenance inputs expected in
an archival evidence deposit. It deliberately leaves `archive_url`, `doi`, and
`evidence_sha256` unset until a real deposit exists.

Generate and verify the complete local bundle:

```sh
bash scripts/reproduce.sh paper
uv run python tools/release_tools.py verify-evidence \
  --source artifacts/runs/paper
```

Copy the verified tree into an empty upload directory:

```sh
uv run python tools/release_tools.py package-evidence \
  --source artifacts/runs/paper \
  --destination <archive-upload-directory>
```

The verifier independently recomputes sequential parity from raw rows, checks
the one-step outcome conclusions while requiring the exact documented corpus
cardinality mismatch, checks MiniGrid cardinality, verifies
configuration/provenance hashes, and records a hash over the complete evidence
tree. Upload that tree to a durable archive, then record its real URL, DOI, and
evidence hash in `release/manifest.json`.

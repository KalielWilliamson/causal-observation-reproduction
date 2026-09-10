# Public release checklist

## 1. Validate the source tree

```sh
bash scripts/reproduce.sh lint
bash scripts/reproduce.sh test
bash scripts/reproduce.sh docs
bash scripts/reproduce.sh formal
bash scripts/reproduce.sh contract-check
```

The CUE check requires Docker in Linux-container mode. CUE is optional for the
experiment runtime but remains part of the release quality gate.

## 2. Generate exact evidence

```sh
bash scripts/reproduce.sh paper
uv run python tools/release_tools.py verify-evidence \
  --source artifacts/runs/paper
```

The verifier reads the raw sequential rows rather than trusting the generated
parity report, independently checks the one-step outcome conclusions and known
corpus-cardinality contradiction, validates MiniGrid scope, checks frozen
protocol and source-provenance hashes, and computes the evidence-tree SHA-256.

## 3. Export without private history

Do not make the development checkout public in place. When the reviewed change
is already committed, export that Git tree into an empty directory:

```sh
uv run python tools/release_tools.py snapshot \
  --ref HEAD \
  --destination <empty-public-repository>
```

The exporter includes no Git history and rejects private-source or legacy

For a reviewed working tree that has not yet been committed, use:

```sh
uv run python tools/release_tools.py snapshot-worktree \
  --destination <empty-public-repository>
```

This includes current tracked modifications and non-ignored untracked files,
omits tracked deletions and ignored files, includes no Git metadata, and rejects
private-source identifiers, legacy interface identifiers, private keys, and
common credential-like values. Review `PUBLIC_SNAPSHOT.md`, run the quality
checks in that directory, then initialize and publish a new repository.

## 4. Deposit data separately

```sh
uv run python tools/release_tools.py package-evidence \
  --source artifacts/runs/paper \
  --destination <empty-archive-upload-directory>
```

Upload the complete packaged directory to a durable archive. Only after the
record exists should `release/manifest.json` receive the actual archive URL,
DOI, and evidence hash. The DOI should resolve to the data/code evidence bundle
used for the paper, not merely to a mutable repository branch.

## 5. Final review

- confirm the source revision in the provenance, result, and release manifests;
- check that raw files, configs, summaries, and parity report are all present;
- retain the finite-grid and descriptive-control limitations;
- ensure every optional result is labelled `extension-`;
- run a secrets scan on both the public snapshot and archive directory; and
- tag the public software revision corresponding to the deposit.

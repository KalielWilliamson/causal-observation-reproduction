# Certified quotient-cache control

This finite control compares two cache keys over the same exact action-value
evaluator and the same full structural history inputs. The full-history arm keys
results by `(signal, nuisance)`; the quotient arm keys by the declared
decision-relevant `signal` only.

The valid panel exhaustively certifies that each quotient class preserves the
complete action-value vector and selected action. Its invalid panel uses the
same proposed projection while making nuisance alter values, so it must fail
certification and is never positive compression evidence.

Run it with:

```powershell
bash scripts/reproduce.sh contract-check
bash scripts/reproduce.sh extension-certified-cache-control
```

## Dataset package

The output is a compact research data package, not an opaque binary artifact:

| File            | Purpose                                                  |
| --------------- | -------------------------------------------------------- |
| `manifest.json` | Contract identity, configuration, and provenance digest. |
| `rows.jsonl`    | One flat row per frozen regime and nuisance cardinality. |
| `summary.json`  | Reader-facing preservation and evaluator-call checks.    |

JSONL is the right primary format for this small, auditable panel: it can be
streamed, diffed, and read without a platform-specific reader. If the panel
grows to a large parameter sweep, publish a Parquet mirror alongside this
canonical manifest-and-JSONL package; do not replace it.

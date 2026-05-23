# Final Runs

`final_runs/` is reserved for polished, presentation-ready experiment outputs. Do not use it for smoke tests, preflights, partial gates, or scratch reruns.

## Directory contract

Use one subdirectory per final result family:

```text
final_runs/
  scienceworld/
  alfworld/
  intercode_sql/
  webshop/
```

Each final run directory should include:

- `MANIFEST.md`: run date, owner, environment, model, seed policy, task split, task count, and exact command(s).
- `suite_config.json`: immutable runner configuration.
- `summaries/`: final CSV, JSON, and Markdown summaries.
- `reports/`: human-readable readouts and paired-delta analysis.
- `artifacts/`: trajectories, metrics, retrieved memories, logs, and any other raw evidence needed to audit the summary.
- `PROVENANCE.md`: links back to source KBs, prompts, code commit, and any prior exploratory run that motivated the final run.

## Promotion rule

A run can be copied or regenerated here only when it is intended to support paper, slide, or final decision claims. Exploratory outputs should stay in the domain-specific run trees such as `scienceworld_runs/`, `alfworld_runs/`, `intercode_sql_runs/`, and `webshop_runs/`.

## Preservation rule

Never delete or overwrite old run artifacts when promoting a final run. If a final run needs correction, create a new dated subdirectory and mark the older one as superseded in that run family's `MANIFEST.md` or `PROVENANCE.md`.

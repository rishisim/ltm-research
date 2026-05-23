# KB Integration Checkpoint

Date: 2026-05-23

Branch: `final/integration-kb-state`

## Integrated

- `infra/final-run-audits`
- `final/sql-kb`

## Safe Final KBs

- SQL sanitized train KB:
  `final_runs/kb/sql_train_reflexion_trials3_sanitized/knowledge_base.sql_sanitized.json`

## Preserved But Not Promoted

- ALFWorld Phase 2 audit report:
  `experiment_reports/final/alfworld_kb_report.md`
- ALFWorld final path contains only a blocking marker:
  `final_runs/kb/alfworld_train_reflexion_trials3/BLOCKED_DO_NOT_USE.md`
- ScienceWorld 10-category diagnostic report:
  `experiment_reports/final/scienceworld_kb_10cat_diagnostic_report.md`
- ScienceWorld 10-category diagnostic manifest:
  `final_runs/manifests/scienceworld_train_reflexion_trials3_va80_10cat_diagnostic.json`

## Reserved Rebuild Paths

- ALFWorld full train KB:
  `final_runs/kb/alfworld_train_reflexion_trials3/`
- ScienceWorld full 30-category train KB:
  `final_runs/kb/scienceworld_train_reflexion_trials3_va80_30cat/`

## Decision

Do not start full Phase 3 until ALFWorld is rebuilt and the full
ScienceWorld 30-category KB is available. SQL eval may proceed independently
as background work because its sanitized KB passed Phase 2 audit.

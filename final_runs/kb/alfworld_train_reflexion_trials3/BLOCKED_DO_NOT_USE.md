# ALFWorld Final KB Promotion Blocked

No `knowledge_base.json` has been promoted in this directory.

The existing source KB at `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json` is train-only against the official ALFWorld split map, but it is not a clean `alfworld_train_reflexion_trials3` artifact:

- It covers 115 unique trajectory task IDs, not the intended 120 train games.
- Its raw trajectories/reflexions run through trial 7, and 85 KB entries cite trials after trial 3.
- Source trajectories do not record split tags, seed, or a stable 120-game task manifest.
- The KB contains malformed/empty rows, including one blank `valid_level`.

See `final_runs/manifests/alfworld_train_reflexion_trials3.json` and `experiment_reports/final/alfworld_kb_report.md` for audit evidence.

This path is not safe for final eval use until a clean train-only ALFWorld KB is rebuilt with `max_trials=3` and promoted here.

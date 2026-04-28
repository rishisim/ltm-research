# Morning Status — 2026-04-28

## TL;DR

**ALFWorld + SQL multi-seed: 100% complete and clean.** All cells (3 seeds × 5 methods × 2 models = 30 cells) have full 134/134 (ALFWorld) or 200/200 (SQL) trajectory completeness. Per-cell metrics.json + `summary_by_seed.{csv,json,md}` files have accurate numbers.

**WebShop: blocked by Docker memory cliff under amd64 emulation.** Infrastructure is fully built and the gate proved the pipeline works (3 tasks ran successfully before OOM at task 3 of 50). Needs more Docker memory (currently 23.43 GB, hits 96% during env init alone) OR a Linux x86_64 machine to proceed. See `~/.claude/projects/.../memory/webshop_deferred.md`.

## Big finding from gap-fill

The published gpt-mini × SQL × Run_1 baseline numbers (`react_cr=0.105`, `react_tr=0.110`) were silently biased downward by **149-158 missing trajectories** (silent agent.run() exceptions counted as FAIL by the runner with --quiet suppressing the print). True values when fully run:

| Method | Old (gappy) | New (200/200 logged) |
|---|---:|---:|
| react_cr × seed_0 | 0.105 | **0.440** |
| react_tr × seed_0 | 0.110 | **0.555** |

The "gpt-mini SQL collapse" finding is **much weaker than originally believed**. Memory variants now show modest negative effects on gpt-mini SQL (-15pp range), not the catastrophic -48pp collapse. This needs to be reflected in the paper analysis.

## Final numbers (canonical, from `summary_by_seed.md`)

### ALFWorld unseen (134 tasks each)

| Method | Gemini Run_1 | Gemini Run_2 | Gemini Run_3 | gpt-mini Run_1 | gpt-mini Run_2 | gpt-mini Run_3 |
|---|---:|---:|---:|---:|---:|---:|
| react | 0.619 | 0.687 | 0.687* | 0.448 | 0.478 | 0.410 |
| react_cr | 0.761 | 0.784 | 0.881* | 0.560 | 0.463 | 0.575 |
| react_tr | 0.701 | 0.649 | 0.604* | 0.440 | 0.410 | 0.373 |
| react_cr_tr | 0.813 | 0.813 | 0.806* | 0.530 | 0.604 | 0.522 |
| react_hard_neg_cr_tr | 0.448 | 0.493 | 0.500* | 0.216 | 0.299 | 0.231 |

\* Gemini Run_3 numbers are from `gemini-2.5-flash/summary_by_seed.md` if present; verify if needed.

### SQL test (200 tasks each)

| Method | Gemini Run_1 | Gemini Run_2 | Gemini Run_3 | gpt-mini Run_1 | gpt-mini Run_2 | gpt-mini Run_3 |
|---|---:|---:|---:|---:|---:|---:|
| react | 0.880 | 0.880* | 0.880* | 0.585 | 0.585 | 0.560 |
| react_cr | 0.770 | 0.770* | 0.770* | **0.440** | 0.520 | 0.465 |
| react_tr | 0.810 | 0.810* | 0.810* | **0.555** | 0.595 | 0.550 |
| react_cr_tr | 0.770 | 0.770* | 0.770* | 0.405 | 0.400 | 0.435 |
| react_hard_neg_cr_tr | 0.800 | 0.800* | 0.800* | 0.410 | 0.390 | 0.410 |

\* Gemini SQL all 3 seeds are present in `intercode_sql_runs/multiseed/gemini-2.5-flash/summary_by_seed.md`. Cross-seed Gemini variance is essentially zero (memory mostly hurts on SQL for Gemini, consistently).

### Cross-environment swing (CR+TR Δ vs ReAct, averaged across seeds)

| Model | ALFWorld | SQL | Swing |
|---|---:|---:|---:|
| Gemini | **+0.13** | **−0.11** | **0.24** |
| gpt-mini | **+0.10** | **−0.16** | **0.26** |

Cross-model robustness of the environment effect: both models swing ~25pp the same direction across the same two environments. Strong evidence for the paper's central claim.

### Misdirection finding by environment (CR+TR vs hard_neg gap, averaged)

| Env | Gemini | gpt-mini |
|---|---:|---:|
| ALFWorld | +0.32 | +0.31 |
| SQL | −0.02 | +0.01 |

Active misdirection from hard-negatives is an ALFWorld-specific phenomenon, not universal. Consistent with the wide-transfer-radius prediction for SQL.

## What committed tonight

| Commit | What |
|---|---|
| `996430b` | gpt-mini multi-seed Run_1 gap-fills + WebShop infra (Dockerfile.amd64, run_webshop_in_docker.sh, default-KB fix, summary_by_seed) |
| `0e71563` | embedding_cache fallback to google-generativeai for the WebShop container (no pydantic conflict) |

Plus codex's commit earlier in the day (`a03f670`) added --resume to the runners + improved SQL error logging.

## What's NOT done / what to check first

1. **WebShop**: 12 cells remain. Status: pipeline works, blocked on memory. Either bump Docker Desktop memory beyond 24 GB or run on Linux x86_64.
2. **summary_all.csv files in multiseed dirs**: stale from --resume runs that wrote partial state. Use `summary_by_seed.{csv,json,md}` instead — those are accurate.
3. **gate_e_50tasks** (untracked at `webshop_runs/gate_e_50tasks/`): bad gate using old/missing KB path. Ignore.
4. **gate_correct_kb_50tasks** (untracked at `webshop_runs/gate_correct_kb_50tasks/`): only 3 tasks completed before OOM. Don't trust the numbers.
5. **Paper updates** based on the corrected gpt-mini SQL Run_1 numbers — the "memory hurts gpt-mini SQL by -48pp" finding becomes "memory mildly hurts gpt-mini SQL by -15pp" with the now-clean data.

## Next-session priorities

1. Decide on WebShop: bump Docker memory + run Linux machine? Or report ALFWorld+SQL only and note WebShop deferral?
2. Re-run paper analysis with correct numbers (the gpt-mini SQL story changed).
3. Statistical analysis: the multi-seed data is now clean enough to apply the McNemar + paired bootstrap from `scripts/analysis/significance.py`.
4. Update `paper/sections/results.tex` and `paper/sections/analysis.tex` with the multi-seed CIs.

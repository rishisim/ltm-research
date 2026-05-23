# ALFWorld final KB rebuild blocked

This directory is not safe for final evaluation. The final train KB was not rebuilt.

## Blockers

- Python module 'alfworld' is not importable.
- No ALFWorld task data root found. Set ALFWORLD_DATA or pass --alfworld-data to a root containing json_2.1.1/train or train/.
- LTM_OPENROUTER_API_KEY is missing for gemini chat calls routed through OpenRouter.
- GEMINI_API_KEY is missing for gemini-embedding-001 cache construction.

## Rebuild command to run after blockers are fixed

```bash
python3 scripts/alfworld_final_kb_builder.py --mode run --final-dir final_runs/kb/alfworld_train_reflexion_trials3 --manifest final_runs/manifests/alfworld_train_reflexion_trials3.json --expected-tasks 120 --max-trials 3 --model gemini-2.5-flash --embedding-provider gemini --seed 0
```

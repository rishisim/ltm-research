#!/bin/bash
source .venv/bin/activate
cd alfworld_runs
python main.py \
    --num_trials 1 \
    --num_envs 1 \
    --run_name "test_run_gemini" \
    --model "gemini-2.5-flash"

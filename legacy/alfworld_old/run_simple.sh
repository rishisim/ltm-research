#!/bin/bash
source ../.venv/bin/activate
python main.py \
        --num_trials 1 \
        --num_envs 10 \
        --run_name "base_run_logs_gemini" \
        --model "gemini-2.5-flash"


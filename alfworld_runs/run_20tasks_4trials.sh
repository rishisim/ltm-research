#!/bin/bash
source .alfworld_venv/bin/activate
python main.py \
        --num_trials 4 \
        --num_envs 20 \
        --run_name "run_20tasks_4trials" \
        --use_memory \
        --model "gemini-2.5-flash"

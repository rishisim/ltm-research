#!/usr/bin/env bash
set -e

cd /Users/rishisim/Documents/research/ltm-research

PYTHON=webshop/.webshop_venv/bin/python

$PYTHON scripts/legacy/run_webshop_suite.py --splits dev,test --num-tasks 200 --frameworks react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr,react_reflexion --memory-bank webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train/knowledge_base.json --resume --workers 16

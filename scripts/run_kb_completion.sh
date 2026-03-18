#!/usr/bin/env bash
set -e

cd /Users/rishisim/Documents/research/ltm-research

PYTHON=webshop/.webshop_venv/bin/python

echo "=== Step 1: KB extraction on existing 50 train trajectories ==="
$PYTHON -m src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script --log_dir webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train --env webshop --resume

echo "=== Step 2: Generate trajectories for tasks 150-200 ==="
$PYTHON scripts/legacy/run_webshop_suite.py --splits train --frameworks react_reflexion --start-idx 150 --num-tasks 50 --runs-root webshop_runs/memory_retrieval_v2/memory_agent_runs

echo "=== Step 3: KB extraction on new trajectories ==="
$PYTHON -m src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script --log_dir webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train --env webshop --resume

echo "=== All 3 steps complete! ==="

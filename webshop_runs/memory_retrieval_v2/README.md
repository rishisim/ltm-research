# WebShop Memory Retrieval v2 - Experiment Directory

Organized structure for memory-augmented learning experiments on WebShop tasks.
Mirrors the ALFWorld experiment structure in `alfworld_runs/memory_retrieval_v2/`.

## Directory Structure

```
memory_retrieval_v2/
├── knowledge_base/                 # Knowledge base from train split trajectories
│   ├── knowledge_base.json         # Raw knowledge base entries
│   ├── knowledge_base.csv          # CSV export of KB
│   └── *.embeddings_cache.json     # Cached embeddings
│
├── memory_agent_runs/              # Evaluation run outputs
│   ├── manifests/                  # Task manifests
│   ├── summaries/                  # Per-split and combined results
│   └── {framework}_{split}/        # Per-framework, per-split runs
│
├── misc/                           # Archives and miscellaneous
│   └── old_runs/                   # Archived previous runs
│
└── README.md                       # This file
```

## Workflow

1. **Generate mini dataset**: `python scripts/create_webshop_mini.py`
2. **Run ReAct on train**: `python scripts/run_webshop_suite.py --splits train --frameworks react`
3. **Run Reflexion on train**: `python scripts/run_webshop_suite.py --splits train --frameworks react_reflexion`
4. **Generate KB**: `python -m src.frameworks.memory_retrieval_v2.preprocessing.knowledge_base_v2.script --log_dir webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train --env webshop`
5. **Run all frameworks on dev/test**: `python scripts/run_webshop_suite.py --splits dev,test`

## Key Differences from ALFWorld

- **Task IDs**: Integer indices (0-12086) instead of directory paths
- **Reward**: Continuous 0-1 score based on attribute match, not binary
- **Actions**: `search[query]` and `click[element]` instead of navigation/manipulation
- **Single task type**: Find and buy a product (vs 6 ALFWorld task types)
- **KB schema**: Uses `product_category`, `action_type`, `task_phase` instead of `obj_type`, `verbs`, `goal_phase`

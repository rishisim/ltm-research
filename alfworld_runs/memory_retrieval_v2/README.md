# Memory Retrieval v2 - Experiment Directory

Organized structure for memory-augmented learning experiments on ALFWorld tasks.

## Directory Structure

```
memory_retrieval_v2/
├── knowledge_base/                 # Knowledge base and embeddings
│   ├── knowledge_base.json         # Raw knowledge base entries (~400 items)
│   ├── knowledge_base.csv          # CSV export of KB
│   ├── knowledge_base.mem_learning_counts.json  # Cached aggregation
│   └── knowledge_base_copy.*       # Backup copies
│
├── agents/                         # Agent implementations
│   └── dedicated_task_agent.py    # Hard-coded context agent for specific tasks
│
├── results/                        # Experiment results and logs
│   ├── trajectories_dedicated_task.json      # Task trajectories
│   ├── trajectories_dedicated_task_CR.json   # CR variant trajectories
│   ├── run_summary.json           # Experiment summary
│   ├── run_summary_CR.json        # CR variant summary
│   └── knowledge_retrieval_bases.json        # Retrieved contexts by task_id
│
├── utils/                          # Utility scripts
│   └── extract_trajectories_csv.py # Convert trajectories to CSV
│
├── analysis/                       # Analysis and metadata
│   ├── alfworld_mini_tasks_sumry.csv        # Mini dataset summary
│   ├── trajectories_summary.csv            # Trajectory statistics
│   ├── knowledge_base_dedup.csv            # Deduplicated KB analysis
│   └── knowledge_base_progress.json        # KB construction progress
│
├── memory_allocation_runs/         # Legacy experiment data (keep for reference)
│   ├── reflexions.json
│   ├── trajectories.json
│   └── world.log
│
├── memory_agent_runs/              # Legacy agent runs (keep for reference)
│   └── [old run data]
│
└── README.md                       # This file
```

## Key Files

### knowledge_base/ 
The foundation of the memory system - curated issue-learning pairs extracted from past task trajectories.

**knowledge_base.json format:**
```json
{
  "task_desc": "put a cool apple in microwave.",
  "obj_type": "apple, microwave",
  "verbs": "put, heat",
  "goal_phase": "PLACE",
  "issue_text": "Failed to put apple into microwave.",
  "learning_text": "Heat the item directly while holding it, not in the microwave.",
  "valid_level": "VALID_SAME_TRIAL",
  "issue_ref": {"task_id": "...", "trial_num": 1, "step_range": [5, 8]},
  "evidence_ref": {"task_id": "...", "trial_num": 1, "step_range": [10, 10]}
}
```

**Caches:**
- `.mem_learning_counts.json` - Aggregated: unique task_desc + learning counts + embeddings

### results/
Stores experiment results indexed by task_id.

**knowledge_retrieval_bases.json:**
```json
{
  "task_001": {
    "query": "put a cool apple in microwave.",
    "metadata": {
      "pick_learning_count": 17,
      "max_learning_count": 11,
      "knowledge_retrieval_base_count": 30,
      "top_similar_tasks": [...]
    },
    "knowledge_retrieval_base": [...]
  }
}
```

This allows you to:
- Track what context was retrieved for each task
- Analyze retrieval effectiveness
- Debug and iterate on retrieval algorithms

### agents/
Agent implementations used in experiments.

**dedicated_task_agent.py**: Hard-coded learnings for a specific task (ablation study)

### analysis/
Analysis and summary files.

- **alfworld_mini_tasks_sumry.csv** - Summary of mini dataset (20 games per task type)
- **trajectories_summary.csv** - Statistics across all trajectories
- **knowledge_base_dedup.csv** - Deduplicated KB entries for analysis

## Usage

### Run a single task with context retrieval

```python
from src.frameworks.memory_allocation.retrieval.core import retrieve_context

result = retrieve_context(
    new_task_desc="put a cool apple in microwave.",
    memory_bank_path="memory_retrieval_v2/knowledge_base/knowledge_base.json",
    task_id="task_001",
    log_dir="memory_retrieval_v2/results",
    top_k_similar_tasks=5
)

print(f"Retrieved {result['metadata']['actual_selected_count']} learnings")
```

### Run full experiment with MemoryAgent

```bash
python scripts/run_memory_agent.py \
    --num-tasks 10 \
    --task-type look_at \
    --memory-bank memory_retrieval_v2/knowledge_base/knowledge_base.json \
    --log-dir memory_retrieval_v2/results
```

### Analyze results

```bash
python scripts/extract_valid_seen_csv.py \
    --trajectories memory_retrieval_v2/results/trajectories_*.json \
    --output memory_retrieval_v2/analysis/results_analysis.csv
```

## Experiment Flow

1. **Knowledge Base Construction**
   - Extract issue-learning pairs from trajectories (`knowledge_base.json`)
   - Validate learnings across multiple trials

2. **Embedding & Caching**
   - Compute embeddings for KB entries
   - Cache embeddings and learning counts
   - `embedding_cache.py` and `learning_counts.py` handle this

3. **Context Retrieval**
   - When a new task arrives, embed its description
   - Find top-5 similar tasks from mem_learning_counts
   - Retrieve and rank relevant learnings
   - Save full retrieval to `knowledge_retrieval_bases.json`

4. **Agent Execution**
   - Agent receives context learnings in prompt
   - Agent can call help tool during execution for additional learnings
   - Both tracked in trajectory

5. **Analysis**
   - Compare success rates with/without context
   - Analyze what learnings were most useful
   - Refine knowledge base and retrieval parameters

## Performance Notes

- **Retrieval time**: ~0.8 seconds (cached)
- **Embedding cache size**: ~1MB
- **Learning counts cache size**: ~1MB
- **Knowledge retrieval log size**: Grows with number of tasks (~10KB per task)

## Future Improvements

- [ ] Implement query expansion for better similarity matching
- [ ] Add learned weights to retrieval ranking
- [ ] Support for negative examples (what NOT to do)
- [ ] Multi-hop reasoning over learnings
- [ ] Periodic knowledge base cleanup and deduplication

## References

- Context retrieval: `src/frameworks/memory_allocation/retrieval/core/context_retrieval.py`
- Learning counts: `src/frameworks/memory_allocation/retrieval/core/learning_counts.py`
- Embedding cache: `src/frameworks/memory_allocation/retrieval/core/embedding_cache.py`
- Agents: `src/frameworks/memory_allocation/agents/`

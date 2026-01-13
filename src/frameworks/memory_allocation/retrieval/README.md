# Memory Allocation Retrieval Module

This module implements memory-augmented learning for ALFWorld agents with optimized embedding caching and dynamic knowledge base retrieval.

## Directory Structure

```
retrieval/
├── core/                        # Core retrieval modules
│   ├── embedding_cache.py      # Persistent embedding storage and caching
│   ├── learning_counts.py      # Aggregated learning counts by task description
│   ├── context_retrieval.py    # Main context retrieval with optimized algorithm
│   └── __init__.py             # Public API exports
│
├── variants/                    # Alternative retrieval strategies
│   ├── hard_neg_context_retrieval.py     # Hard negative example retrieval
│   ├── hard_neg_tool_retrieval.py        # Hard negative help tool
│   ├── rag_context_retrieval.py          # RAG-based retrieval from trajectories
│   ├── trajectory_context_retrieval.py   # Raw trajectory-based retrieval
│   └── __init__.py
│
├── tool_retrieval.py           # Help tool retrieval during task execution
├── __init__.py                 # Package initialization
└── README.md                   # This file
```

## Core Modules

### 1. embedding_cache.py
Manages persistent caching of embeddings to avoid re-computing on every task run.

**Key functions:**
- `create_knowledge_base_embeddings(kb_path)` - Embeds all KB entries, saves to cache
- `load_embeddings_cache(kb_path)` - Loads or creates cache with validation
- `cosine_similarity_batch()` - Efficient batch similarity computation

**Cache files:**
- `.embeddings_cache.json` - Cached embeddings for all KB entries
- `.mem_learning_counts.json` - Aggregated task descriptions with counts

### 2. learning_counts.py
Builds and manages the `mem_learning_counts` table - an aggregated view of the knowledge base grouped by task description.

**Key functions:**
- `build_learning_counts_table(kb_path)` - Groups KB by task_desc, counts learnings
- `load_learning_counts(kb_path)` - Loads the aggregation with embeddings
- `get_top_similar_tasks(query_emb, entries, embeddings, top_k)` - Similarity search

**Data structure:**
```python
{
  "task_desc": "put a cool apple in microwave.",
  "task_desc_embedding": [0.123, ...],  # 768-dim vector
  "learning_count": 5,
  "validated_learning_count": 3,
  "entry_indices": [0, 3, 7, 12, 15]
}
```

### 3. context_retrieval.py (OPTIMIZED)
Main retrieval module with new algorithm:

1. **Load/create** `mem_learning_counts` cache (embeddings + counts)
2. **Embed** query task_desc (1 API call per task)
3. **Similarity search** against cached task embeddings → top-5 similar tasks
4. **Calculate** `pick_learning_count = ceil(1.5 × max_learning_count)`
5. **Left join** top-5 tasks with full KB → `knowledge_retrieval_base`
6. **Re-rank**: validated entries first, CANDIDATE entries at bottom
7. **Select** top `pick_learning_count` learnings for LLM
8. **Log** full `knowledge_retrieval_base` to separate file by task_id

**Key functions:**
- `retrieve_context(task_desc, kb_path)` - Main retrieval function
- `format_learnings_for_prompt(learnings)` - Format learnings for LLM prompt
- `retrieve_learnings_only(task_desc, kb_path)` - Quick convenience function

**Output format:**
```python
{
  "query": "put a cool apple in microwave.",
  "metadata": {
    "pick_learning_count": 17,
    "max_learning_count": 11,
    "actual_selected_count": 17,
    "knowledge_retrieval_base_count": 30,
    "top_similar_tasks": [...]
  },
  "knowledge_retrieval_base": [...],     # Full table (all 30 rows)
  "selected_learnings": [...]            # Top 17 formatted for LLM
}
```

### 4. tool_retrieval.py
Handles dynamic retrieval during task execution via the `help["query"]` action.

**Key functions:**
- `help_tool(query, kb_path)` - Retrieve learnings mid-execution
- `format_help_response(learnings)` - Format help response for agent

## Variant Modules

Located in `variants/` - alternative retrieval strategies:

- **hard_neg_context_retrieval.py** - Hard negative example retrieval
- **hard_neg_tool_retrieval.py** - Hard negative help tool
- **rag_context_retrieval.py** - RAG-based retrieval from trajectory CSV
- **trajectory_context_retrieval.py** - Raw trajectory-based retrieval

## Usage Examples

### Basic Context Retrieval

```python
from src.frameworks.memory_allocation.retrieval.core import retrieve_context

result = retrieve_context(
    new_task_desc="put a cool apple in microwave.",
    memory_bank_path="path/to/knowledge_base.json",
    task_id="task_001",           # Optional, for logging
    log_dir="path/to/logs",       # Optional, saves knowledge_retrieval_base
    top_k_similar_tasks=5         # Default: 5
)

# Use learnings in prompt
learnings = result["selected_learnings"]
prompt_text = f"Context: {result['metadata']['actual_selected_count']} relevant learnings found"
```

### With Agent

```python
from src.frameworks.memory_allocation.agents import MemoryAgent

agent = MemoryAgent(model="gemini-2.5-flash")
history, success = agent.run(
    env=env,
    base_prompt=base_prompt,
    memory=[],
    task_id="task_001",
    task_desc="put a cool apple in microwave.",
    memory_bank_path="path/to/knowledge_base.json",
    log_dir="path/to/logs"
)
```

## Caching Strategy

The system uses two-level caching for optimal performance:

1. **Embedding Cache** (`.embeddings_cache.json`)
   - Caches embeddings of ALL knowledge base entries
   - Rebuilt only if source KB changes (validated by MD5 hash)
   - ~1MB for ~400 KB entries

2. **Learning Counts Cache** (`.mem_learning_counts.json`)
   - Aggregated view: unique task descriptions + learning counts + embeddings
   - Rebuilt only if source KB changes
   - ~1MB for ~100 unique tasks

3. **Knowledge Retrieval Log** (`knowledge_retrieval_bases.json`)
   - Separate file tracking what was retrieved for each task_id
   - Appends new entries as tasks are run
   - Useful for analysis and debugging

## Performance

- **First run**: ~5 seconds (embeddings computed and cached)
- **Subsequent runs**: ~0.8 seconds (uses cached embeddings)
- **Query embedding**: 1 API call per task
- **Similarity search**: O(n) where n = unique task descriptions (~100)

## Configuration

Key parameters in `retrieve_context()`:
- `top_k_similar_tasks` (default: 5) - Number of similar tasks to retrieve
- `pick_learning_count` formula: `ceil(1.5 × max_learning_count)` - Dynamic selection
- Validation ranking: VALID_NEXT_TRIAL > VALID_SAME_TRIAL > CANDIDATE

## Validation Levels

- **VALID_NEXT_TRIAL** (priority 3): Learning validated in a later trial
- **VALID_SAME_TRIAL** (priority 2): Learning validated in the same trial
- **CANDIDATE** (priority 1): Learning not yet validated

## Integration Points

- **Agents**: `src/frameworks/memory_allocation/agents/`
  - `MemoryAgent` - Full context + help tool
  - `ContextOnlyAgent` - Context only (no help)
  - `ToolOnlyAgent` - Help tool only (no initial context)

- **Task Logging**: Saved to `knowledge_retrieval_bases.json` by task_id
- **Trajectory**: Learnings also included in trajectory JSON

## Related Files

- Knowledge base: `alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json`
- Results: `alfworld_runs/memory_retrieval_v2/results/`
- Analysis: `alfworld_runs/memory_retrieval_v2/analysis/`

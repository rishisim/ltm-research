# Hard Negative CR+TR Agent (v2)

Implementation of hard negative retrieval variants for memory-augmented agents.

## Overview

This variant uses **bottom-k** (least similar) retrieval instead of top-k, with inverted validation ranking to test the agent's performance with deliberately irrelevant context.

## Implementation Details

### 1. Hard Negative Context Retrieval
**File**: `src/frameworks/memory_retrieval_v2/retrieval/variants/hard_neg_context_retrieval.py`

**Changes from normal retrieval**:
- ✅ Uses `get_bottom_similar_tasks()` to retrieve LOWEST similarity matches
- ✅ Sorts similarities **ascending** instead of descending
- ✅ Inverted validation priority:
  - `CANDIDATE`: 3 (highest priority - was lowest)
  - `VALID_SAME_TRIAL`: 2 (middle)
  - `VALID_NEXT_TRIAL`: 1 (lowest priority - was highest)
- ✅ Re-ranking puts CANDIDATES first, validated entries at bottom
- ✅ Logs to `knowledge_retrieval_bases_hard_neg.json`

### 2. Hard Negative Tool Retrieval
**File**: `src/frameworks/memory_retrieval_v2/retrieval/variants/hard_neg_tool_retrieval.py`

**Changes from normal retrieval**:
- ✅ Uses inverted multiplicative scoring: `(1.0 - similarity) × valid_weight`
- ✅ Inverted validation weights:
  - `CANDIDATE`: 1.0 (highest weight)
  - `VALID_SAME_TRIAL`: 0.5 (lower weight)
  - `VALID_NEXT_TRIAL`: 0.5 (lower weight)
- ✅ Sorts by inverted score descending (highest = least similar)
- ✅ Logs to `help_retrieval_step_{N}_hard_neg.json`

### 3. Hard Negative Memory Agent
**File**: `src/frameworks/memory_retrieval_v2/agents/hard_neg_memory_agent.py`

Inherits from `ReAct`, uses hard negative retrieval variants for both:
- Context retrieval at task start
- Help tool during execution

Adds `retrieval_type: "hard_negative"` to trajectory logs.

## Usage

### Command Line

```bash
python scripts/run_hard_neg_memory_agent_v2.py \
  --task-file path/to/task_list.json \
  --memory-bank alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json \
  --output-dir alfworld_runs/memory_retrieval_v2/results/hard_neg_CR_TR \
  --model gemini-2.5-flash \
  --bottom-k 5
```

### Arguments

- `--task-file`: JSON file with task specifications (required)
- `--memory-bank`: Path to knowledge_base.json (default: v2 knowledge base)
- `--output-dir`: Output directory (default: `results/hard_neg_CR_TR`)
- `--model`: LLM model to use (default: `gemini-2.5-flash`)
- `--bottom-k`: Number of bottom-k tasks to retrieve (default: 5)
- `--quiet`: Suppress verbose output

### Task File Format

```json
{
  "tasks": [
    "train:pick_and_place_simple-Mug-None-Desk-308/trial_T20190909_044019_803145",
    "valid_seen:pick_cool_then_place_in_recep-Apple-None-Fridge-424/trial_T20190909_055144_168648"
  ]
}
```

## Expected Behavior

The hard negative agent should perform **worse** than the normal memory agent, as it receives:
- Least similar task context at the start
- Least relevant help during execution
- Unvalidated candidates prioritized over proven solutions

This serves as a **control experiment** to validate that similarity-based retrieval is actually beneficial.

## Output Files

```
alfworld_runs/memory_retrieval_v2/results/hard_neg_CR_TR/
├── world.log                                    # Task results summary
├── trajectories.json                            # Complete trajectory logs
└── knowledge_retrieval_bases_hard_neg.json     # Retrieved context per task
```

## Comparison with Normal Agent

| Feature | Normal CR+TR | Hard Neg CR+TR |
|---------|-------------|----------------|
| Context similarity | Top-k (highest) | Bottom-k (lowest) |
| Validation priority | Validated first | Candidates first |
| Tool retrieval score | `similarity × weight` | `(1-similarity) × weight` |
| Expected performance | Higher | Lower (control) |

## Architecture

```
src/frameworks/memory_retrieval_v2/
├── retrieval/
│   ├── core/                    # Normal retrieval (top-k)
│   │   ├── context_retrieval.py
│   │   └── tool_retrieval.py
│   └── variants/                # Hard negative variants (bottom-k)
│       ├── hard_neg_context_retrieval.py  ← NEW
│       └── hard_neg_tool_retrieval.py     ← NEW
└── agents/
    ├── memory_agent.py                      # Normal agent
    └── hard_neg_memory_agent.py            ← NEW
```

## Caching

Both hard negative variants reuse the same embedding caches as normal retrieval:
- `knowledge_base.embeddings.json` (issue_text embeddings)
- `knowledge_base.mem_learning_counts.json` (task_desc embeddings)

No additional caching infrastructure needed.

## Validation

To verify the implementation is working:

1. Check similarity scores in logs (should be < 0.5)
2. Verify CANDIDATE entries appear first in retrieval base
3. Compare accuracy with normal memory agent (should be lower)
4. Inspect retrieved learnings (should be topically different from task)

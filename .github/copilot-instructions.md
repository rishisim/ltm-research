# Copilot Instructions for LTM-Research

This repository implements **Reflexion** (NeurIPS 2023) - a framework for language agents with verbal reinforcement learning, extended with novel memory allocation mechanisms for ALFWorld tasks.

## Architecture Overview

```
src/
├── core/           # Base abstractions: BaseEnv, Framework, EnvironmentHistory, LLM wrappers
├── envs/           # Environment implementations (AlfworldEnv wraps alfworld package)
└── frameworks/     # Agent frameworks extending ReAct base
    ├── react.py           # Base ReAct agent with action-observation loops
    ├── reflexion.py       # Adds self-reflection after failures
    └── memory_allocation/ # Extended agents with retrieval-augmented memory
        ├── agents/        # MemoryAgent, ContextOnlyAgent, ToolOnlyAgent, etc.
        ├── retrieval/     # context_retrieval.py (task-start), tool_retrieval.py (help["query"])
        └── preprocessing/ # Knowledge base construction utilities
```

**Key inheritance chain**: `ReAct` → `Reflexion` → `MemoryAgent` (each extends the previous)

## Environment & Setup

- **Virtual environment**: Always use `.venv/bin/python` - never system Python
- **API keys**: Store `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env` at project root
- **ALFWorld data**: Requires `$ALFWORLD_DATA` env var pointing to game files
- **Config**: [data/alfworld/base_config.yaml](data/alfworld/base_config.yaml) controls task types and dataset paths

## Running Experiments

```bash
# Main entry point for ALFWorld experiments
python experiments/alfworld/main.py --framework react --run_name my_test --num_envs 5

# Memory agent runner (most commonly used)
python scripts/run_memory_agent.py --num-tasks 10 --task-type look_at --memory-bank path/to/knowledge_base.json
```

Key CLI flags: `--framework {react,reflexion,in-trajectory}`, `--use_memory`, `--dataset {official,mini}`

## Framework Patterns

### Agent `run()` Signature
All agents implement this pattern (see [src/core/base.py](src/core/base.py)):
```python
def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "") -> Tuple[EnvironmentHistory, bool]:
```
Memory agents extend this with: `task_id`, `task_desc`, `memory_bank_path`, `log_dir`

### LLM Calls
Use `get_chat()` from [src/core/llm.py](src/core/llm.py) - handles both OpenAI and Gemini models:
```python
from src.core.llm import get_chat, Model
response = get_chat(prompt, model="gemini-2.5-flash", temperature=0.0, stop_strs=["\n"])
```
Models: `"gpt-4"`, `"gpt-3.5-turbo"`, `"gemini-2.0-flash"`, `"gemini-2.5-flash"`

### Memory Retrieval
Two retrieval mechanisms in `memory_allocation/retrieval/`:
1. **Context retrieval** (`context_retrieval.py`): Called at task start, uses embedding similarity on `task_desc`
2. **Tool retrieval** (`tool_retrieval.py`): Called via `help["query"]` action during execution

Both use Google's `text-embedding-004` model and cosine similarity with validation-level ranking.

### Memory Bank Schema (`knowledge_base.json`)
Each entry in the knowledge base contains:
```json
{
  "task_desc": "put a mug in desk.",           // Task description for similarity matching
  "obj_type": "mug",                            // Target object type
  "goal_phase": "SEARCH",                       // Phase: SEARCH, ACQUIRE, TRANSFORM, PLACE, RECOVER
  "issue_text": "Failed to open drawer 2...",   // The problem encountered
  "learning_text": "If a container cannot...",  // The learned solution
  "valid_level": "VALID_SAME_TRIAL",            // Validation status (see below)
  "trigger": {"key": "NO_EFFECT", "verb": "open", "obs": "Nothing happens."},
  "evidence_ref": {"task_id": "...", "trial_num": 1, "step_range": [8, 8]}
}
```

**Validation Levels** (priority for retrieval ranking):
- `VALID_NEXT_TRIAL` (3): Validated by success in subsequent trial
- `VALID_SAME_TRIAL` (2): Validated within the same trial
- `CANDIDATE` (1): Not yet validated

### Goal Phases
Defined in [context_retrieval.py](src/frameworks/memory_allocation/retrieval/context_retrieval.py#L35):
- **SEARCH**: Finding/locating objects (keywords: `find`, `look`, `where`, `cannot find`)
- **ACQUIRE**: Picking up objects (keywords: `pick`, `take`, `grab`, `hold`)
- **TRANSFORM**: Applying operations (keywords: `heat`, `cool`, `clean`, `microwave`, `fridge`)
- **PLACE**: Putting objects in locations (keywords: `put`, `place`, `drop`)
- **RECOVER**: Handling errors/stuck states (keywords: `stuck`, `nothing happens`, `wrong`)

## ALFWorld Task Types

Defined in [experiments/alfworld/main.py](experiments/alfworld/main.py#L17-L24):
- `pick_and_place` → `put`, `pick_clean_then_place` → `clean`, `pick_heat_then_place` → `heat`
- `pick_cool_then_place` → `cool`, `look_at_obj` → `examine`, `pick_two_obj` → `puttwo`

Prompts loaded from [data/alfworld/prompts/alfworld_3prompts.json](data/alfworld/prompts/alfworld_3prompts.json)

## Code Conventions

- **Logging**: Experiments write to `alfworld_runs/<run_name>/` with `world.log`, `trial_N.log`, and `trajectories/` JSON files
- **History tracking**: Use `EnvironmentHistory.add("action"|"observation", value)` - tracks exhaustion via repeated actions
- **Action parsing**: Clean actions with `.strip()`, handle `Action:` and `>` prefixes (see [react.py](src/frameworks/react.py#L37-L41))

## Known Issues

- **Gemini syntax mismatch**: Gemini models output `put X in Y` but ALFWorld requires `put X in/on Y` - causes "Nothing happens" (see [experiment_reports/react_baseline_report.md](experiment_reports/react_baseline_report.md))
- **Stop sequences**: LLM calls use `stop_strs=["\n"]` which can truncate multi-line thoughts

## Mini Dataset

`alfworld_mini/` contains a curated subset for faster iteration. Set `--dataset mini` or modify config:
```yaml
dataset:
  data_path: 'alfworld_mini/train'
```

## Utility Scripts

Key scripts in `scripts/` for common workflows:

| Script | Purpose |
|--------|---------|
| `run_memory_agent.py` | Main runner for memory-augmented agents with `--task-type`, `--memory-bank` flags |
| `run_memory_agent_batch.py` | Batch execution across multiple task types |
| `run_*_ablation.py` | Ablation studies (context-only, tool-only variants) |
| `create_alfworld_mini.py` | Generates the mini dataset (20 games/type) from full ALFWorld |
| `alfworld_log_parser.py` | Parses trial logs into structured trajectory JSON |
| `extract_*_csv.py` | Export results to CSV for analysis (valid_seen, valid_unseen, ablations) |

**Ablation agents** in `memory_allocation/agents/`:
- `context_only_agent.py`: Only context retrieval at task start, no help tool
- `tool_only_agent.py`: Only help tool during execution, no initial context
- `trajectory_context_agent.py`: Uses raw trajectories instead of extracted learnings

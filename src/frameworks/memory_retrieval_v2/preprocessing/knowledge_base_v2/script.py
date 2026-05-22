"""
Single Table Knowledge Base Generator

Generates a unified knowledge base table directly from raw trajectories.json.
Each row contains: obj_type, verbs, goal_phase, trigger_key, issue_text, 
learning_text, valid_level, evidence_ref, issue_ref
"""

import json
import os
import argparse
import csv
from typing import List, Dict, Any, Set
from collections import defaultdict
from src.core.llm import get_chat

# Constants
MODEL_NAME = "gemini-2.5-flash"

ALFWORLD_SYSTEM_PROMPT = """You are analyzing raw task trajectories to extract issue-learning pairs for a knowledge base.

Input: A list of trajectories for the same task_id across multiple trials.
Each trajectory has: task_id, task_desc, trial_num, steps (action/observation pairs), success boolean.

Task template format: "pick_cool_then_place_in_recep-Tomato-None-Microwave-10"
- Verbs: pick, cool, place
- Objects: Tomato, Microwave

For each issue encountered during the trajectories, extract an entry. Issues include but not limited to:
- Actions that fail or have no effect but were later resolved
- Repeated failed attempts (loops) that were eventually broken
- Missing preconditions (not at location, container closed, etc.) that were addressed
- Using wrong verbs or commands that were corrected later
- Objects not found in expected locations

valid_level meanings:
- VALID_SAME_TRIAL: Issue was fixed later in the same trial
- VALID_NEXT_TRIAL: Issue was fixed in a later trial
- CANDIDATE: Issue was never fixed across all trials

evidence_ref meanings:
- Where the learning/solution was validated (i.e., what worked and fixed the issue)

issue_ref meanings:
- Where the issue/problem was first observed (i.e., what went wrong initially)

Learning Text Guidelines:
- Good: Actionable verbs, specific strategies, can be specific to objects/locations if those worked
- Bad: Passive observations, describing what happened without the action taken

Rules:
- Look across trials to find what eventually worked
- learning_text MUST describe the corrective action/strategy, not just what happened
- Consolidate duplicate issues within the same trial: if the same issue occurs across multiple consecutive or nearby steps, create ONE entry with an expanded step_range
- Only create separate entries for the same issue type within the same task if the learnings are different
- If no issues found, return {"entries": []}


Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "entries": [
    {
      "task_desc": string (the task description, e.g., "put a cool tomato in microwave."),
      "obj_type": string (e.g., "tomato, fridge"),
      "verbs": string (e.g., "cool"),
      "goal_phase": "SEARCH" | "ACQUIRE" | "TRANSFORM" | "PLACE" | "RECOVER",
      "issue_text": string (<30 words, what went wrong),
      "issue_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      }
      "learning_text": string (<40 words, the specific ACTION/STRATEGY that fixed the issue.),
      "evidence_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "valid_level": "VALID_SAME_TRIAL" | "VALID_NEXT_TRIAL" | "CANDIDATE",
    }
  ]
}

<TRAJECTORIES_JSON>"""

# Kept as alias for backward compatibility
SYSTEM_PROMPT = ALFWORLD_SYSTEM_PROMPT

WEBSHOP_SYSTEM_PROMPT = """You are analyzing raw task trajectories from a web shopping environment to extract issue-learning pairs for a knowledge base.

Input: A list of trajectories for the same task_id across multiple trials.
Each trajectory has: task_id, task_desc, trial_num, steps (action/observation pairs), success boolean, reward (0-1 score).

The environment is WebShop, where an agent navigates a simulated e-commerce website to find and purchase a product matching a natural language instruction. The agent uses two actions:
- search[query]: search for products
- click[element]: click on page elements (product links, attribute options like size/color, "Buy Now", etc.)

For each issue encountered during the trajectories, extract an entry. Issues include but not limited to:
- Search queries that returned irrelevant results
- Clicking on wrong products that don't match the instruction
- Forgetting to select required attributes (size, color, etc.) before buying
- Buying a product that exceeds the price constraint
- Not exploring enough results (missing better-matching products on other pages)
- Selecting wrong attribute options

valid_level meanings:
- VALID_SAME_TRIAL: Issue was fixed later in the same trial
- VALID_NEXT_TRIAL: Issue was fixed in a later trial
- CANDIDATE: Issue was never fixed across all trials

evidence_ref meanings:
- Where the learning/solution was validated (i.e., what worked and fixed the issue)

issue_ref meanings:
- Where the issue/problem was first observed (i.e., what went wrong initially)

Learning Text Guidelines:
- Good: Specific search strategies, attribute selection approaches, price-checking habits
- Bad: Passive observations, describing what happened without actionable advice

Rules:
- Look across trials to find what eventually worked
- learning_text MUST describe the corrective action/strategy, not just what happened
- Consolidate duplicate issues within the same trial
- Only create separate entries for the same issue type if the learnings are different
- If no issues found, return {"entries": []}

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "entries": [
    {
      "task_desc": string (the shopping instruction, e.g., "Find me a grey queen size quilt set under $80"),
      "obj_type": string (the product and attributes, e.g., "bedding, quilt sets, size, options"),
      "verbs": string (e.g., "search, click, buy"),
      "goal_phase": "SEARCH" | "BROWSE" | "SELECT" | "PURCHASE",
      "issue_text": string (<30 words, what went wrong),
      "issue_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "learning_text": string (<40 words, the specific ACTION/STRATEGY that fixed the issue),
      "evidence_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "valid_level": "VALID_SAME_TRIAL" | "VALID_NEXT_TRIAL" | "CANDIDATE"
    }
  ]
}

<TRAJECTORIES_JSON>"""


INTERCODE_SQL_SYSTEM_PROMPT = """You are analyzing raw task trajectories from an interactive SQL coding environment to extract issue-learning pairs for a knowledge base.

Input: A list of trajectories for the same task_id across multiple trials.
Each trajectory has: task_id, task_desc, trial_num, steps (action/observation pairs), success boolean.

The environment is InterCode-SQL, where an agent interacts with a MySQL database to answer natural language questions by writing SQL queries. The agent executes SQL statements and receives query results or error messages as observations. The agent submits when confident in its answer.

For each issue encountered during the trajectories, extract an entry. Issues include but not limited to:
- SQL syntax errors that were corrected
- Wrong table or column references that were fixed
- Missing JOIN conditions or incorrect JOIN types
- Incorrect WHERE clauses or filtering logic
- Aggregation errors (wrong GROUP BY, missing HAVING, etc.)
- Subquery errors that were resolved
- Schema misunderstanding (wrong column names, types, etc.)
- Incorrect use of SQL functions (COUNT, AVG, etc.)
- Not exploring the schema before querying (leading to errors)

valid_level meanings:
- VALID_SAME_TRIAL: Issue was fixed later in the same trial
- VALID_NEXT_TRIAL: Issue was fixed in a later trial
- CANDIDATE: Issue was never fixed across all trials

evidence_ref meanings:
- Where the learning/solution was validated (i.e., what worked and fixed the issue)

issue_ref meanings:
- Where the issue/problem was first observed (i.e., what went wrong initially)

Learning Text Guidelines:
- Good: Specific SQL patterns, schema exploration strategies, query construction approaches
- Bad: Passive observations, vague advice like "write better queries"

Rules:
- Look across trials to find what eventually worked
- learning_text MUST describe the corrective action/strategy, not just what happened
- Consolidate duplicate issues within the same trial
- Only create separate entries for the same issue type if the learnings are different
- If no issues found, return {"entries": []}

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "entries": [
    {
      "task_desc": string (the natural language question, e.g., "How many singers do we have?"),
      "obj_type": string (database objects involved, e.g., "singer table, Singer_ID column"),
      "verbs": string (SQL operations, e.g., "SELECT, COUNT, JOIN"),
      "goal_phase": "EXPLORE" | "QUERY" | "REFINE" | "VERIFY",
      "issue_text": string (<30 words, what went wrong),
      "issue_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "learning_text": string (<40 words, the specific ACTION/STRATEGY that fixed the issue),
      "evidence_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "valid_level": "VALID_SAME_TRIAL" | "VALID_NEXT_TRIAL" | "CANDIDATE"
    }
  ]
}

<TRAJECTORIES_JSON>"""


SCIENCEWORLD_SYSTEM_PROMPT = """You are analyzing raw task trajectories from ScienceWorld to extract issue-learning pairs for a knowledge base.

Input: A list of trajectories for the same task_id across multiple trials.
Each trajectory has: task_id, task_desc, trial_num, steps (action/observation pairs), success boolean, and reward/score when available.

ScienceWorld is a text-based science-procedure environment. Agents solve tasks by inspecting rooms and objects, collecting materials, using tools, measuring observations, changing temperature, mixing substances, moving between rooms, and focusing on or submitting answers when required.

For each issue encountered during the trajectories, extract an entry. Issues include but are not limited to:
- Skipping orientation actions such as look around, inventory, or examine
- Using the wrong room, container, tool, or substance
- Trying a procedure step before satisfying a prerequisite
- Heating, cooling, mixing, pouring, measuring, or focusing on the wrong object
- Failing to verify a measurement or observation before answering
- Repeated valid-looking actions that produced no progress

valid_level meanings:
- VALID_SAME_TRIAL: Issue was fixed later in the same trial
- VALID_NEXT_TRIAL: Issue was fixed in a later trial
- CANDIDATE: Issue was never fixed across all trials

evidence_ref meanings:
- Where the learning/solution was validated (i.e., what worked and fixed the issue)

issue_ref meanings:
- Where the issue/problem was first observed (i.e., what went wrong initially)

Learning Text Guidelines:
- Good: Specific ScienceWorld action strategies, prerequisite checks, measurement/verification habits, and recovery steps
- Bad: Passive observations, vague advice like "be careful", or restating the task goal without an action

Rules:
- Look across trials to find what eventually worked
- learning_text MUST describe the corrective action/strategy, not just what happened
- Consolidate duplicate issues within the same trial
- Only create separate entries for the same issue type if the learnings are different
- Prefer reusable object categories (container, thermometer, heat source, substance) over variation-specific names
- If no issues found, return {"entries": []}

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "entries": [
    {
      "task_desc": string (the ScienceWorld task description),
      "obj_type": string (objects/materials/tools involved, e.g., "substance, beaker, thermometer"),
      "verbs": string (actions involved, e.g., "examine, take, heat, measure"),
      "goal_phase": "ORIENT" | "SEARCH" | "ACQUIRE" | "PROCEDURE" | "MEASURE" | "VERIFY" | "ANSWER" | "RECOVER",
      "issue_text": string (<30 words, what went wrong),
      "issue_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "learning_text": string (<40 words, the specific ACTION/STRATEGY that fixed the issue),
      "evidence_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      },
      "valid_level": "VALID_SAME_TRIAL" | "VALID_NEXT_TRIAL" | "CANDIDATE"
    }
  ]
}

<TRAJECTORIES_JSON>"""


def get_system_prompt(env: str = "alfworld") -> str:
    """Return the appropriate system prompt for the given environment."""
    if env == "webshop":
        return WEBSHOP_SYSTEM_PROMPT
    if env == "intercode_sql":
        return INTERCODE_SQL_SYSTEM_PROMPT
    if env == "scienceworld":
        return SCIENCEWORLD_SYSTEM_PROMPT
    return ALFWORLD_SYSTEM_PROMPT


def load_json(path: str) -> List[Dict[str, Any]]:
    """Safe JSON loader."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {path}")
        return []


def save_json(path: str, data: Any):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_progress(path: str) -> Set[str]:
    """Load processed task_ids from a progress file."""
    data = load_json(path)
    if isinstance(data, list):
        return set(str(x) for x in data)
    return set()


def save_progress(path: str, task_ids: Set[str]):
    """Persist processed task_ids to a progress file."""
    save_json(path, sorted(task_ids))


def processed_task_ids_from_entries(entries: List[Dict[str, Any]]) -> Set[str]:
    """Extract processed task_ids from existing knowledge base entries."""
    processed = set()
    for entry in entries:
        issue = entry.get("issue_ref") or {}
        tid = issue.get("task_id")
        if tid:
            processed.add(tid)
    return processed


def group_trajectories_by_task(trajectories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group trajectories by task_id."""
    grouped = defaultdict(list)
    for traj in trajectories:
        task_id = traj.get("task_id", "")
        grouped[task_id].append(traj)
    
    # Sort each group by trial_num
    for task_id in grouped:
        grouped[task_id].sort(key=lambda x: x.get("trial_num", 0))
    
    return dict(grouped)


def process_task_trajectories(task_id: str, trajectories: List[Dict[str, Any]], model: str, env: str = "alfworld") -> List[Dict[str, Any]]:
    """Process all trajectories for a single task to extract knowledge base entries."""
    
    traj_str = json.dumps(trajectories, indent=2)
    system_prompt = get_system_prompt(env)
    prompt = system_prompt.replace("<TRAJECTORIES_JSON>", traj_str)
    
    print(f"Processing task: {task_id} ({len(trajectories)} trials)...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # reasoning={"effort": "none"} is a Gemini-specific OpenRouter extension.
            # Pass it only for Gemini models so non-Gemini models don't fail.
            _reasoning = {"effort": "none"} if model.startswith("gemini") else None
            response_text, _usage = get_chat(prompt, model=model, max_tokens=2048, reasoning=_reasoning, request_timeout=180)
            
            # Clean up response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            if isinstance(result, list):
                entries = result
            elif isinstance(result, dict):
                entries = result.get("entries", [])
            else:
                entries = []
            
            print(f"  Extracted {len(entries)} entries")
            return entries
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: JSON parse error - {e}")
            else:
                print(f"Error processing task {task_id}: {e}")
                return []
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            return []
    
    return []


def convert_to_csv(entries: List[Dict[str, Any]], output_path: str):
    """Convert entries to CSV format, flattening nested structures."""
    if not entries:
        print("No entries to convert to CSV")
        return
    
    # Flattened fieldnames matching actual JSON structure
    fieldnames = [
        "unique_id", "task_desc", "obj_type", "verbs", "goal_phase",
        "issue_text", "learning_text", "valid_level",
        "issue_task_id", "issue_trial_num", "issue_step_range",
        "evidence_task_id", "evidence_trial_num", "evidence_step_range"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            # Flatten evidence_ref and issue_ref objects; tolerate missing/None values
            evidence = entry.get("evidence_ref") or {}
            issue = entry.get("issue_ref") or {}
            evidence_step_range = evidence.get("step_range") or []
            issue_step_range = issue.get("step_range") or []

            if not isinstance(evidence_step_range, list):
                evidence_step_range = []
            if not isinstance(issue_step_range, list):
                issue_step_range = []
            
            row = {
                "unique_id": entry.get("unique_id", ""),
                "task_desc": entry.get("task_desc", ""),
                "obj_type": entry.get("obj_type", ""),
                "verbs": entry.get("verbs", ""),
                "goal_phase": entry.get("goal_phase", ""),
                "issue_text": entry.get("issue_text", ""),
                "learning_text": entry.get("learning_text", ""),
                "valid_level": entry.get("valid_level", ""),
                "issue_task_id": issue.get("task_id", ""),
                "issue_trial_num": issue.get("trial_num", ""),
                "issue_step_range": f"{issue_step_range[0]}-{issue_step_range[1]}" if len(issue_step_range) == 2 else "",
                "evidence_task_id": evidence.get("task_id", ""),
                "evidence_trial_num": evidence.get("trial_num", ""),
                "evidence_step_range": f"{evidence_step_range[0]}-{evidence_step_range[1]}" if len(evidence_step_range) == 2 else ""
            }
            writer.writerow(row)
    
    print(f"Saved CSV to {output_path}")


def generate_knowledge_base(
    log_dir: str,
    task_ids: List[str] = None,
    reuse_json: bool = False,
    resume: bool = False,
    progress_path: str = None,
    env: str = "alfworld",
    allow_eval_trajectories: bool = False,
):
    """Main logic to generate knowledge base from trajectories."""

    trajectories_path = os.path.join(log_dir, "trajectories.json")
    json_output_path = os.path.join(log_dir, "knowledge_base.json")
    csv_output_path = os.path.join(log_dir, "knowledge_base.csv")
    progress_path = progress_path or os.path.join(log_dir, "knowledge_base_progress.json")

    # Load existing entries if present
    existing_entries: List[Dict[str, Any]] = []
    if os.path.exists(json_output_path):
        existing_entries = load_json(json_output_path)

    # When reuse_json without resume, just convert existing entries
    if reuse_json and not resume:
        if not existing_entries:
            print(f"No existing knowledge base at {json_output_path} to reuse")
            return
        print(f"Reusing existing knowledge base at {json_output_path}")
        convert_to_csv(existing_entries, csv_output_path)
        return

    # Resume mode: skip already processed task_ids
    processed_tasks = processed_task_ids_from_entries(existing_entries)
    processed_tasks |= load_progress(progress_path)

    if not os.path.exists(trajectories_path):
        print(f"No trajectories found at {trajectories_path}")
        return

    trajectories = load_json(trajectories_path)
    if not trajectories:
        print("No trajectories to process")
        return

    # --- Eval-into-memory leakage guard ---
    # Trajectories tagged with split="dev" or split="test" must not be ingested
    # into the knowledge base (they are evaluation data, not training data).
    # Pass --allow-eval-trajectories only for ablations / intentional overrides.
    if not allow_eval_trajectories:
        eval_splits_found = set()
        missing_split_count = 0
        for t in trajectories:
            s = t.get("split", "")
            if s in ("dev", "test"):
                eval_splits_found.add(s)
            if not s:
                missing_split_count += 1
        if eval_splits_found:
            raise ValueError(
                f"Eval-into-memory leakage detected: trajectories.json contains entries "
                f"with split={sorted(eval_splits_found)}. "
                f"Only 'train' split trajectories should be ingested into the KB. "
                f"Pass --allow-eval-trajectories to suppress this check (for ablations only)."
            )
        total = len(trajectories)
        missing_fraction = missing_split_count / total if total > 0 else 0.0
        if missing_fraction > 0.05:
            raise ValueError(
                f"Eval-into-memory leakage guard: {missing_split_count}/{total} trajectory "
                f"entries ({missing_fraction:.1%}) are missing the 'split' field. "
                f"A missing split field means the guard cannot verify these trajectories "
                f"are safe to ingest. Ensure all trajectories are written with a 'split' "
                f"field. Pass --allow-eval-trajectories to suppress this check (for "
                f"ablations only)."
            )
    # Group by task_id
    grouped = group_trajectories_by_task(trajectories)
    print(f"Found {len(grouped)} unique tasks")
    
    # Filter to specific task_ids if provided
    if task_ids:
        grouped = {tid: grouped[tid] for tid in task_ids if tid in grouped}
        print(f"Filtering to {len(grouped)} specified tasks")

    # Apply resume filter
    if resume and processed_tasks:
        original_count = len(grouped)
        grouped = {tid: trajs for tid, trajs in grouped.items() if tid not in processed_tasks}
        print(f"Resume enabled: skipping {original_count - len(grouped)} already processed tasks")

    # If nothing to do, still refresh CSV from existing entries
    if not grouped:
        print("No new tasks to process; refreshing CSV from existing entries")
        convert_to_csv(existing_entries, csv_output_path)
        return

    # Process each task group
    new_entries: List[Dict[str, Any]] = []
    for task_id, task_trajs in grouped.items():
        entries = process_task_trajectories(task_id, task_trajs, MODEL_NAME, env=env)
        new_entries.extend(entries)
        processed_tasks.add(task_id)

    # Merge and save JSON
    all_entries = existing_entries + new_entries
    
    # Assign sequential unique_ids
    for i, entry in enumerate(all_entries):
        entry["unique_id"] = str(i + 1)
        
    save_json(json_output_path, all_entries)
    print(f"Saved {len(all_entries)} total entries to {json_output_path}")

    # Persist progress for resume (includes tasks with zero extracted entries)
    save_progress(progress_path, processed_tasks)

    # Convert to CSV
    convert_to_csv(all_entries, csv_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate knowledge base from raw trajectories."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing trajectories.json"
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific task_ids to process (space-separated)"
    )
    parser.add_argument(
        "--reuse_json",
        action="store_true",
        help="If set, reuse existing knowledge_base.json in log_dir instead of regenerating"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks already present in knowledge_base.json or progress file and append new ones"
    )
    parser.add_argument(
        "--progress_path",
        type=str,
        default=None,
        help="Optional path for progress file (default: knowledge_base_progress.json in log_dir)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default="alfworld",
        choices=["alfworld", "webshop", "intercode_sql", "scienceworld"],
        help="Environment type for prompt selection (default: alfworld)"
    )

    parser.add_argument(
        "--allow-eval-trajectories",
        action="store_true",
        help=(
            "Suppress the eval-into-memory leakage check. "
            "Use ONLY for ablations where you intentionally want to process "
            "dev/test split trajectories."
        )
    )

    args = parser.parse_args()
    generate_knowledge_base(
        args.log_dir,
        args.task_ids,
        args.reuse_json,
        args.resume,
        args.progress_path,
        env=args.env,
        allow_eval_trajectories=args.allow_eval_trajectories,
    )

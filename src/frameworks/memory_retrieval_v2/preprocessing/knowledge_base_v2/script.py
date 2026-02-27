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

SYSTEM_PROMPT = """You are analyzing raw task trajectories to extract issue-learning pairs for a knowledge base.

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


def process_task_trajectories(task_id: str, trajectories: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Process all trajectories for a single task to extract knowledge base entries."""
    
    traj_str = json.dumps(trajectories, indent=2)
    prompt = SYSTEM_PROMPT.replace("<TRAJECTORIES_JSON>", traj_str)
    
    print(f"Processing task: {task_id} ({len(trajectories)} trials)...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_text, _usage = get_chat(prompt, model=model, max_tokens=16384)
            
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
        entries = process_task_trajectories(task_id, task_trajs, MODEL_NAME)
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
    
    args = parser.parse_args()
    generate_knowledge_base(
        args.log_dir,
        args.task_ids,
        args.reuse_json,
        args.resume,
        args.progress_path,
    )

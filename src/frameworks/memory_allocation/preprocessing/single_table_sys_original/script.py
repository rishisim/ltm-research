"""
Single Table Knowledge Base Generator

Generates a unified knowledge base table directly from raw trajectories.json.
Each row contains: obj_type, verbs, goal_phase, trigger_key, issue_text, 
learning_text, valid_level, evidence_ref
"""

import json
import os
import argparse
import csv
from typing import List, Dict, Any
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

For each issue encountered during the trajectories, extract an entry. Issues include:
- Actions that fail or have no effect
- Repeated failed attempts (loops)
- Missing preconditions (not at location, container closed, etc.)
- Using wrong verbs or commands
- Objects not found in expected locations

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "entries": [
    {
      "task_desc": string (the task description, e.g., "put a cool tomato in microwave."),
      "obj_type": string (lowercase, e.g., "tomato"),
      "verbs": string (single verb, e.g., "cool"),
      "goal_phase": "SEARCH" | "ACQUIRE" | "TRANSFORM" | "PLACE" | "RECOVER",
      "issue_text": string (<30 words, what went wrong),
      "learning_text": string (<40 words, how to fix based on what worked later),
      "trigger": {
        "key": "WRONG_VERB" | "PRECONDITION" | "NO_EFFECT" | "CONTAINER_STATE" | "LOOP" | "NOT_FOUND" | "OTHER",
        "verb": string (the action verb that failed, e.g., "put"),
        "target_type": string (lowercase object type, e.g., "microwave"),
        "obs": string (the observation that indicated the issue)
      },
      "valid_level": "VALID_SAME_TRIAL" | "VALID_NEXT_TRIAL" | "CANDIDATE",
      "evidence_ref": {
        "task_id": string,
        "trial_num": number,
        "step_range": [start_step, end_step]
      }
    }
  ]
}

trigger.key meanings:
- WRONG_VERB: Used wrong action verb (e.g., "put" instead of "move")
- PRECONDITION: Agent not at correct location or prerequisite not met
- NO_EFFECT: Action simply didn't work for unknown reason
- CONTAINER_STATE: Container open/closed/occupied issue blocking action
- LOOP: Agent stuck in repeated no-op actions
- NOT_FOUND: Object not found in expected locations
- OTHER: Doesn't fit above categories

valid_level meanings:
- VALID_SAME_TRIAL: Issue was fixed later in the same trial
- VALID_NEXT_TRIAL: Issue was fixed in a later trial
- CANDIDATE: Issue was never fixed across all trials

Rules:
- Each row must have exactly ONE obj_type (lowercase)
- Each row must have exactly ONE verb (not multiple verbs)
- If an issue involves multiple verbs, create separate rows for each
- Look across trials to find what eventually worked
- If no issues found, return {"entries": []}

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
            response_text = get_chat(prompt, model=model, max_tokens=16384)
            
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
    
    # Flattened fieldnames
    fieldnames = [
        "task_desc", "obj_type", "verbs", "goal_phase",
        "trigger_key", "trigger_verb", "trigger_target_type", "trigger_obs",
        "issue_text", "learning_text", "valid_level",
        "evidence_task_id", "evidence_trial_num", "evidence_step_range"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            # Flatten trigger object
            trigger = entry.get("trigger", {})
            evidence = entry.get("evidence_ref", {})
            step_range = evidence.get("step_range", [])
            
            row = {
                "task_desc": entry.get("task_desc", ""),
                "obj_type": entry.get("obj_type", ""),
                "verbs": entry.get("verbs", ""),
                "goal_phase": entry.get("goal_phase", ""),
                "trigger_key": trigger.get("key", ""),
                "trigger_verb": trigger.get("verb", ""),
                "trigger_target_type": trigger.get("target_type", ""),
                "trigger_obs": trigger.get("obs", ""),
                "issue_text": entry.get("issue_text", ""),
                "learning_text": entry.get("learning_text", ""),
                "valid_level": entry.get("valid_level", ""),
                "evidence_task_id": evidence.get("task_id", ""),
                "evidence_trial_num": evidence.get("trial_num", ""),
                "evidence_step_range": f"{step_range[0]}-{step_range[1]}" if len(step_range) == 2 else ""
            }
            writer.writerow(row)
    
    print(f"Saved CSV to {output_path}")


def generate_knowledge_base(log_dir: str):
    """Main logic to generate knowledge base from trajectories."""
    
    trajectories_path = os.path.join(log_dir, "trajectories.json")
    json_output_path = os.path.join(log_dir, "knowledge_base.json")
    csv_output_path = os.path.join(log_dir, "knowledge_base.csv")
    
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
    
    # Process each task group
    all_entries = []
    for task_id, task_trajs in grouped.items():
        entries = process_task_trajectories(task_id, task_trajs, MODEL_NAME)
        all_entries.extend(entries)
    
    # Save JSON
    save_json(json_output_path, all_entries)
    print(f"Saved {len(all_entries)} entries to {json_output_path}")
    
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
    
    args = parser.parse_args()
    generate_knowledge_base(args.log_dir)

#!/usr/bin/env python3
"""
Generate a CSV of all task_ids in the alfworld-mini dataset.
Columns: task_type, task_id, split, task_desc, already_in_memory
"""

import os
import json
import csv

# Paths
ALFWORLD_MINI_DIR = "/Users/rishisim/Documents/research/ltm-research/alfworld_mini"
TRAJECTORIES_PATH = "/Users/rishisim/Documents/research/ltm-research/alfworld_runs/memory_allocation_test/trajectories.json"
OUTPUT_CSV = "/Users/rishisim/Documents/research/ltm-research/alfworld_runs/memory_allocation_test/alfworld_mini_tasks.csv"

# Task types (from create_alfworld_mini.py)
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place"
]

def get_task_type(task_name):
    """Determine task type from the task name."""
    for prefix in TASK_TYPES:
        if task_name.startswith(prefix):
            return prefix
    return "unknown"

def get_task_desc_from_traj_data(traj_data_path):
    """Extract task_desc from traj_data.json file."""
    try:
        with open(traj_data_path, "r") as f:
            data = json.load(f)
        # Task description is in turk_annotations.anns[0].task_desc
        anns = data.get("turk_annotations", {}).get("anns", [])
        if anns and len(anns) > 0:
            return anns[0].get("task_desc", "")
        return ""
    except Exception as e:
        print(f"Warning: Could not read {traj_data_path}: {e}")
        return ""

def get_task_ids_from_trajectories(trajectories_path):
    """Extract unique task_ids from trajectories.json."""
    task_ids = set()
    with open(trajectories_path, "r") as f:
        data = json.load(f)
    for entry in data:
        task_id = entry.get("task_id", "")
        if task_id:
            task_ids.add(task_id)
    return task_ids

def get_task_ids_from_alfworld_mini(alfworld_dir):
    """Scan the alfworld_mini directory to extract all task_ids with their splits and task_desc."""
    tasks = []
    for split in ["train", "valid_seen", "valid_unseen"]:
        split_dir = os.path.join(alfworld_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for task_name in os.listdir(split_dir):
            task_path = os.path.join(split_dir, task_name)
            if not os.path.isdir(task_path):
                continue
            task_type = get_task_type(task_name)
            # Each task folder contains trial subdirectories
            for trial in os.listdir(task_path):
                trial_path = os.path.join(task_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                # The task_id is typically: task_name/trial
                task_id = f"{task_name}/{trial}"
                
                # Get task_desc from traj_data.json
                traj_data_path = os.path.join(trial_path, "traj_data.json")
                task_desc = ""
                if os.path.exists(traj_data_path):
                    task_desc = get_task_desc_from_traj_data(traj_data_path)
                
                tasks.append({
                    "task_type": task_type,
                    "task_id": task_id,
                    "split": split,
                    "task_desc": task_desc
                })
    return tasks

def main():
    # Get existing task_ids from trajectories.json
    print("Loading existing task_ids from trajectories.json...")
    existing_task_ids = get_task_ids_from_trajectories(TRAJECTORIES_PATH)
    print(f"Found {len(existing_task_ids)} unique task_ids in trajectories.json")
    
    # Identify task types in trajectories.json
    task_types_in_memory = set()
    for task_id in existing_task_ids:
        task_name = task_id.split("/")[0]
        task_type = get_task_type(task_name)
        task_types_in_memory.add(task_type)
    print(f"\nTask types in memory_allocation_test/trajectories.json:")
    for tt in sorted(task_types_in_memory):
        print(f"  - {tt}")
    
    # Get all task_ids from alfworld_mini
    print("\nScanning alfworld_mini directory...")
    all_tasks = get_task_ids_from_alfworld_mini(ALFWORLD_MINI_DIR)
    print(f"Found {len(all_tasks)} total tasks in alfworld_mini")
    
    # Sort tasks by split, then task_type, then task_id
    all_tasks.sort(key=lambda x: (x["split"], x["task_type"], x["task_id"]))
    
    # Add already_in_memory column
    for task in all_tasks:
        task["already_in_memory"] = task["task_id"] in existing_task_ids
    
    # Write to CSV
    print(f"\nWriting CSV to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "task_id", "split", "task_desc", "already_in_memory"])
        writer.writeheader()
        writer.writerows(all_tasks)
    
    # Summary
    in_memory_count = sum(1 for t in all_tasks if t["already_in_memory"])
    print(f"\nSummary:")
    print(f"  Total tasks in alfworld_mini: {len(all_tasks)}")
    print(f"  Tasks already in memory: {in_memory_count}")
    print(f"  Tasks not in memory: {len(all_tasks) - in_memory_count}")
    
    # Breakdown by split
    print(f"\nBreakdown by split:")
    for split in ["train", "valid_seen", "valid_unseen"]:
        split_tasks = [t for t in all_tasks if t["split"] == split]
        in_mem = sum(1 for t in split_tasks if t["already_in_memory"])
        print(f"  {split}: {len(split_tasks)} tasks ({in_mem} in memory)")
    
    # Breakdown by task type
    print(f"\nBreakdown by task type:")
    for tt in TASK_TYPES:
        tt_tasks = [t for t in all_tasks if t["task_type"] == tt]
        in_mem = sum(1 for t in tt_tasks if t["already_in_memory"])
        print(f"  {tt}: {len(tt_tasks)} tasks ({in_mem} in memory)")

if __name__ == "__main__":
    main()

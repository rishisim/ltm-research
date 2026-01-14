
import json
import csv
import os
import argparse
from typing import List, Dict, Any

def flatten_chunk_data(task_id: str, trial_num: int, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens the chunk data into a single dictionary for CSV writing.
    Nested objects like 'state_before' are converted to JSON strings.
    """
    flat_data = {
        "task_id": task_id,
        "trial_num": trial_num,
        "chunk_id": chunk.get("chunk_id"),
        "step_range": str(chunk.get("step_range")),  # Convert list to string
        "phase": chunk.get("phase"),
        "goal": chunk.get("goal"),
        # Convert nested dicts/lists to JSON strings for CSV compatibility
        "state_before": json.dumps(chunk.get("state_before", {})),
        "actions": chunk.get("actions"),
        "key_action": chunk.get("key_action"),
        "result": chunk.get("result"),
        "failure_type": chunk.get("failure_type"),
        "state_after_delta": chunk.get("state_after_delta"),
        "progress_flag": chunk.get("progress_flag")
    }
    return flat_data

def convert_to_csv(log_dir: str):
    """
    Reads trajectory_chunks.json and writes trajectory_chunks.csv
    """
    json_path = os.path.join(log_dir, "trajectory_chunks.json")
    csv_path = os.path.join(log_dir, "trajectory_chunks.csv")
    
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_path}")
        return

    csv_rows = []
    
    # Process all chunks
    for entry in data:
        task_id = entry.get("task_id")
        trial_num = entry.get("trial_num")
        chunks = entry.get("chunks", [])
        
        for chunk in chunks:
            csv_rows.append(flatten_chunk_data(task_id, trial_num, chunk))

    if not csv_rows:
        print("No chunks found to convert.")
        return

    # Get headers from the first row keys
    headers = list(csv_rows[0].keys())

    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Successfully converted {len(csv_rows)} chunks to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trajectory_chunks.json to CSV.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing trajectory_chunks.json")
    
    args = parser.parse_args()
    
    convert_to_csv(args.log_dir)

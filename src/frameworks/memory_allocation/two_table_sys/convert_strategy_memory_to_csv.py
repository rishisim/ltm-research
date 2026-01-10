
import json
import csv
import os
import argparse
from typing import List, Dict, Any


def flatten_strategy_data(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens the strategy data into a single dictionary for CSV writing.
    Complex objects are converted to JSON strings.
    """
    flat_data = {
        "strategy_id": strategy.get("strategy_id"),
        "polarity": strategy.get("polarity"),
        "task_type": strategy.get("task_type"),
        "phase": strategy.get("phase"),
        "trigger_signature": json.dumps(strategy.get("trigger_signature", {})),
        "patch_type": strategy.get("patch_type"),
        "patch": strategy.get("patch"),
        "origin": strategy.get("origin"),
        "source_task_ids": json.dumps(strategy.get("source_task_ids", [])),
        "source_trials": json.dumps(strategy.get("source_trials", [])),
        "validated": strategy.get("validated"),
        "validation_score": strategy.get("validation_score"),
        "validation_evidence": strategy.get("validation_evidence", "")
    }
    return flat_data


def convert_to_csv(log_dir: str):
    """
    Reads strategy_memory.json and writes strategy_memory.csv
    """
    json_path = os.path.join(log_dir, "strategy_memory.json")
    csv_path = os.path.join(log_dir, "strategy_memory.csv")
    
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_path}")
        return

    strategies = data.get("strategies", [])
    
    if not strategies:
        print("No strategies found to convert.")
        return

    csv_rows = [flatten_strategy_data(s) for s in strategies]

    # Get headers from the first row keys
    headers = list(csv_rows[0].keys())

    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Successfully converted {len(csv_rows)} strategies to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert strategy_memory.json to CSV."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing strategy_memory.json"
    )
    
    args = parser.parse_args()
    
    convert_to_csv(args.log_dir)

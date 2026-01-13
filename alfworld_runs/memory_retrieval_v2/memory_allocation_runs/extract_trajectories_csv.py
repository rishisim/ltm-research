#!/usr/bin/env python3
"""
Extract trajectory data from trajectories.json and output as CSV.
"""

import json
import csv
import os

def main():
    # Path to the trajectories.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "trajectories.json")
    csv_path = os.path.join(script_dir, "trajectories_summary.csv")
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        trajectories = json.load(f)
    
    # Extract the required fields
    rows = []
    for entry in trajectories:
        task_id = entry.get("task_id", "")
        task_type = entry.get("task_type", "")
        task_desc = entry.get("task_desc", "")
        trial_num = entry.get("trial_num", "")
        num_steps = len(entry.get("steps", []))
        success = entry.get("success", False)
        result = "Success" if success else "Failure"
        
        rows.append({
            "Task Type": task_type,
            "Task ID": task_id,
            "Task Desc": task_desc,
            "Trial Num": trial_num,
            "Num Steps": num_steps,
            "Result": result
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ["Task Type", "Task ID", "Task Desc", "Trial Num", "Num Steps", "Result"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV saved to: {csv_path}")
    print(f"\nTotal entries: {len(rows)}")
    
    # Also print to stdout
    print("\n" + "-" * 120)
    print("Task Type | Task ID | Task Desc | Trial Num | Num Steps | Result")
    print("-" * 120)
    for row in rows:
        print(f"{row['Task Type']} | {row['Task ID']} | {row['Task Desc'][:30]}... | {row['Trial Num']} | {row['Num Steps']} | {row['Result']}")

if __name__ == "__main__":
    main()

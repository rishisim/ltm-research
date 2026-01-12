#!/usr/bin/env python3
"""
Extract trajectory data from plain_traj_CR trajectories_valid_unseen.json and output as CSV.
Uses the correct keys for plain trajectory context retrieval.
"""

import json
import csv
import os

def main():
    # Path to the trajectories.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "trajectories_valid_unseen.json")
    csv_path = os.path.join(script_dir, "plain_traj_CR_unseen.csv")
    
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
        result = "SUCCESS" if success else "FAIL"
        
        # Context source
        context_source = entry.get("context_source", "")
        
        # Retrieval metadata (nested object)
        retrieval_metadata = entry.get("retrieval_metadata", {})
        source_task_id = retrieval_metadata.get("source_task_id", "")
        source_task_desc = retrieval_metadata.get("source_task_desc", "")
        similarity_score = retrieval_metadata.get("similarity_score", 0)
        source_success = retrieval_metadata.get("source_success", False)
        word_count_limit = retrieval_metadata.get("word_count_limit", 0)
        
        # Help calls
        help_calls = entry.get("help_calls", [])
        help_calls_json = json.dumps(help_calls)
        help_call_count = entry.get("help_call_count", 0)
        
        rows.append({
            "task_type": task_type,
            "task_id": task_id,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "num_steps": num_steps,
            "result": result,
            "context_source": context_source,
            "source_task_id": source_task_id,
            "source_task_desc": source_task_desc,
            "similarity_score": similarity_score,
            "source_success": source_success,
            "word_count_limit": word_count_limit,
            "help_calls": help_calls_json,
            "help_call_count": help_call_count
        })
    
    # Write to CSV with pipe delimiter and proper quoting
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            "task_type", "task_id", "task_desc", "trial_num", "num_steps", "result",
            "context_source", "source_task_id", "source_task_desc", "similarity_score",
            "source_success", "word_count_limit", "help_calls", "help_call_count"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|', 
                                quoting=csv.QUOTE_MINIMAL, lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV saved to: {csv_path}")
    print(f"\nTotal entries: {len(rows)}")
    
    # Calculate and print summary stats
    successes = sum(1 for r in rows if r["result"] == "SUCCESS")
    failures = len(rows) - successes
    accuracy = successes / len(rows) if rows else 0
    
    print(f"\n--- Summary ---")
    print(f"SUCCESS: {successes}/{len(rows)} ({accuracy:.1%})")
    print(f"FAIL: {failures}/{len(rows)}")
    
    # Print similarity score summary
    all_sims = [r["similarity_score"] for r in rows]
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
    max_sim = max(all_sims) if all_sims else 0
    min_sim = min(all_sims) if all_sims else 0
    
    print(f"\n--- Similarity Scores (Plain Traj CR) ---")
    print(f"Avg Similarity: {avg_sim:.4f}")
    print(f"Max Similarity: {max_sim:.4f}")
    print(f"Min Similarity: {min_sim:.4f}")
    
    # Count source successes
    source_successes = sum(1 for r in rows if r["source_success"])
    print(f"\n--- Source Trajectory Stats ---")
    print(f"Source Successes: {source_successes}/{len(rows)}")

if __name__ == "__main__":
    main()

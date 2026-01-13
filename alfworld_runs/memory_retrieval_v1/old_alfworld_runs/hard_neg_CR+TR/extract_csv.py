#!/usr/bin/env python3
"""
Extract trajectory data from hard_neg trajectories.json and output as CSV.
Matches the format used in the with +LTM folder.
"""

import json
import csv
import os

def main():
    # Path to the trajectories.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "trajectories.json")
    csv_path = os.path.join(script_dir, "hard_neg.csv")
    
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
        
        # Context retrieval info
        context_from_retrieval = entry.get("context_from_retrieval", [])
        context_json = json.dumps(context_from_retrieval)
        
        # Calculate similarity score stats from context
        similarity_scores = [c.get("similarity_score", 0) for c in context_from_retrieval]
        max_similarity = max(similarity_scores) if similarity_scores else 0
        min_similarity = min(similarity_scores) if similarity_scores else 0
        num_retrievals = len(context_from_retrieval)
        num_candidates = sum(1 for c in context_from_retrieval if c.get("valid_level") == "CANDIDATE")
        
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
            "context_from_retrieval": context_json,
            "max_similarity_score": max_similarity,
            "min_similarity_score": min_similarity,
            "num_retrievals": num_retrievals,
            "num_candidates": num_candidates,
            "help_calls": help_calls_json,
            "help_call_count": help_call_count
        })
    
    # Write to CSV with pipe delimiter (matching with +LTM format)
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            "task_type", "task_id", "task_desc", "trial_num", "num_steps", "result",
            "context_from_retrieval", "max_similarity_score", "min_similarity_score",
            "num_retrievals", "num_candidates", "help_calls", "help_call_count"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
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
    
    # Print similarity score summary (for hard neg, these should be low)
    all_max_sims = [r["max_similarity_score"] for r in rows]
    all_min_sims = [r["min_similarity_score"] for r in rows]
    avg_max = sum(all_max_sims) / len(all_max_sims) if all_max_sims else 0
    avg_min = sum(all_min_sims) / len(all_min_sims) if all_min_sims else 0
    overall_max = max(all_max_sims) if all_max_sims else 0
    overall_min = min(all_min_sims) if all_min_sims else 0
    
    print(f"\n--- Similarity Scores (Hard Neg) ---")
    print(f"Avg Max Similarity: {avg_max:.4f}")
    print(f"Avg Min Similarity: {avg_min:.4f}")
    print(f"Overall Max: {overall_max:.4f}")
    print(f"Overall Min: {overall_min:.4f}")

if __name__ == "__main__":
    main()

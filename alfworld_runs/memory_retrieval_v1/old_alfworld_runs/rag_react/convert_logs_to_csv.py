
import json
import csv
import os
import sys

def main():
    # Define paths
    project_root = os.getcwd()
    log_dir = os.path.join(project_root, "alfworld_runs/memory_agent_test/rag_react/logs")
    json_path = os.path.join(log_dir, "trajectories.json")
    csv_path = os.path.join(log_dir, "trajectories.csv")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    print(f"Reading from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            trajectories = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    print(f"Found {len(trajectories)} trajectories.")
    
    # Define CSV columns
    fieldnames = [
        "task_id",
        "task_type",
        "trial_num",
        "success",
        "steps_count",
        "task_desc",
        "context_source",
        "retrieved_trajectory_context",
        "full_trajectory_log"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()
        
        for traj in trajectories:
            # Format retrieved context
            retrieval_meta = traj.get("retrieval_metadata", {})
            retrieved_context = retrieval_meta.get("retrieved_trajectory", "")
            
            # Format full trajectory log (steps) as a single string
            steps_log = ""
            for step in traj.get("steps", []):
                steps_log += f"Step {step['step']}: Action: {step['action']}, Obs: {step['observation']}\\n"
            
            # Sanitize text fields: replace newlines with literal \n and pipes with [PIPE]
            def sanitize(text):
                if not isinstance(text, str):
                    return text
                return text.replace('\n', '\\n').replace('\r', '').replace('|', '[PIPE]')
            
            writer.writerow({
                "task_id": traj.get("task_id", ""),
                "task_type": traj.get("task_type", ""),
                "trial_num": traj.get("trial_num", ""),
                "success": traj.get("success", False),
                "steps_count": len(traj.get("steps", [])),
                "task_desc": sanitize(traj.get("task_desc", "")),
                "context_source": traj.get("context_source", ""),
                "retrieved_trajectory_context": sanitize(retrieved_context),
                "full_trajectory_log": sanitize(steps_log.strip())
            })
            
    print(f"Successfully converted to CSV at {csv_path}")

if __name__ == "__main__":
    main()

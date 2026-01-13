import json
import csv
import os

# Define input and output paths
INPUT_FILE = '/Users/rishisim/Documents/research/ltm-research/alfworld_runs/memory_allocation_test/trajectories.json'
OUTPUT_DIR = '/Users/rishisim/Documents/research/ltm-research/alfworld_runs/memory_agent_test/rag_react'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'truncated_trajectories.csv')

def process_trajectories():
    # Ensure output directory exists (though user said it does, good practice)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Reading from: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    extracted_data = []
    
    for entry in data:
        task_id = entry.get('task_id', '')
        task_desc = entry.get('task_desc', '')
        steps = entry.get('steps', [])
        
        # Format steps
        step_strings = []
        for s in steps:
            step_num = s.get('step', '')
            action = s.get('action', '')
            observation = s.get('observation', '')
            step_str = f"Step {step_num}: action: {action}, observation: {observation}"
            step_strings.append(step_str)
            
        full_trajectory = " ".join(step_strings)
        
        # Truncate to first 130 words
        words = full_trajectory.split()
        if len(words) > 130:
            truncated_trajectory = " ".join(words[:130]) + "..."
        else:
            truncated_trajectory = full_trajectory
            
        extracted_data.append({
            'task_id': task_id,
            'task_desc': task_desc,
            'trunc_trajectory': truncated_trajectory
        })
        
    print(f"Processed {len(extracted_data)} entries.")
    
    # Write to CSV
    print(f"Writing to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['task_id', 'task_desc', 'trunc_trajectory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)
            
    print("Done.")

if __name__ == "__main__":
    process_trajectories()

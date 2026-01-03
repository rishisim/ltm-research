import os
import re
import json
import argparse

def parse_log_file(log_path, output_dir, trial_idx):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by environment separator
    # The logs are separated by "\n#####\n\nEnvironment #{z}: ...\n\n#####\n"
    # We want to capture the content between these markers.
    
    # First, let's normalize headers to make splitting easier if needed, 
    # but strictly speaking the format is fixed in alfworld_trial.py
    
    # We can split by the start marker
    chunks = content.split('#####\n\nEnvironment #')
    
    # The first chunk is usually empty or specific header info, skip it if it doesn't look like an env block
    for chunk in chunks[1:]:
        # chunk starts with "{z}:\n{history}\n\nSTATUS: {OK/FAIL}\n\n"
        
        # Extract Env ID
        try:
            env_id_str, rest = chunk.split(':', 1)
            env_id = int(env_id_str)
        except ValueError:
            print(f"Skipping malformed chunk start: {chunk[:20]}")
            continue

        # Extract History and Status
        # The block ends with "\n\nSTATUS: ...\n\n"
        # Let's find the status line
        status_match = re.search(r'\n\nSTATUS: (OK|FAIL)\n\n', rest)
        if not status_match:
            # Maybe the file is incomplete/corrupted or formatted differently
            print(f"Could not find STATUS in env {env_id} chunk")
            continue
            
        history_text = rest[:status_match.start()].strip()
        
        # Now parse the history text into actions and observations
        # Format in env_history.py:
        # > action
        # observation
        # > action
        # observation
        
        trajectory = []
        lines = history_text.split('\n')
        
        current_observation = []
        
        # The prompt/task description is at the top. 
        # Usually it starts with "Interact with a household..." or "Here is the task:"
        # But in the log file, it just dumps str(env_history).
        # str(env_history) starts with the query (prompt + memory + task).
        
        # Heuristic: Actions start with '> '. Everything else is part of the previous observation or the initial prompt.
        # However, the initial prompt is technically the "observation" of the 0th step (or pre-history).
        # Let's treat everything before the first '> ' as the initial observation/context.
        
        for line in lines:
            if line.startswith('> '):
                # If we have accumulated observation text, dump it
                if current_observation:
                    obs_text = '\n'.join(current_observation).strip()
                    if obs_text: # Filter empty observations if any
                        # If this is the VERY first block, it's the context/prompt
                        # If we already have actions, it's an observation
                        trajectory.append({'label': 'observation', 'value': obs_text})
                    current_observation = []
                
                # Add the action
                action_text = line[2:].strip()
                trajectory.append({'label': 'action', 'value': action_text})
            else:
                current_observation.append(line)
        
        # Append remaining observation after last action
        if current_observation:
             obs_text = '\n'.join(current_observation).strip()
             if obs_text:
                 trajectory.append({'label': 'observation', 'value': obs_text})

        # Save to JSON
        filename = f'env_{env_id}_trial_{trial_idx}.json'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(trajectory, f, indent=4)
        
        print(f"Saved {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--num_trials', type=int, default=5)
    args = parser.parse_args()

    log_dir = args.run_name
    trajectory_dir = os.path.join(log_dir, 'trajectories')
    os.makedirs(trajectory_dir, exist_ok=True)

    for i in range(args.num_trials):
        log_file = os.path.join(log_dir, f'trial_{i}.log')
        if os.path.exists(log_file):
            print(f"Parsing Trial {i}...")
            parse_log_file(log_file, trajectory_dir, i)
        else:
            print(f"Trial {i} log not found (yet).")

if __name__ == "__main__":
    main()

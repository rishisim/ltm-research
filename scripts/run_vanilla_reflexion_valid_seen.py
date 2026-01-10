#!/usr/bin/env python3
"""
Run vanilla Reflexion on valid_seen split of alfworld-mini dataset.
Uses MemoryAllocationReflexion framework (no LTM/memory retrieval).

Output files:
  - trajectories_valid_seen.json
  - reflexions.json
  - world.log
"""

import os
import sys
import yaml
import json
import glob

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_allocation import MemoryAllocationReflexion

# Configuration
NUM_TRIALS = 7
TASKS_CONFIG_PATH = "alfworld_runs/memory_agent_test/vanilla reflexion run/valid_seen_tasks.json"


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_tasks(base_dir, config_path):
    """
    Load task IDs from the config JSON file.
    
    Returns:
        List of task info dicts with path, id, file, and split
    """
    full_path = os.path.join(base_dir, config_path)
    with open(full_path, 'r') as f:
        config = json.load(f)
    
    task_files = []
    
    for task_entry in config['tasks']:
        # Parse task entry format: "valid_seen:task_name/trial_name"
        split, task_path = task_entry.split(':', 1)
        task_name, trial_name = task_path.rsplit('/', 1)
        
        # Build full path
        trial_dir = os.path.join(base_dir, 'alfworld_mini', split, task_name, trial_name)
        game_file = os.path.join(trial_dir, 'game.tw-pddl')
        
        if os.path.exists(game_file):
            task_files.append({
                "path": trial_dir,
                "id": f"{task_name}/{trial_name}",
                "file": game_file,
                "split": split
            })
        else:
            print(f"Warning: Could not find game.tw-pddl in {trial_dir}")
    
    return task_files


def run_vanilla_reflexion():
    # Setup paths - base_dir should be project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'data', 'alfworld', 'base_config.yaml')
    log_dir = os.path.join(base_dir, 'alfworld_runs', 'memory_agent_test', 'vanilla reflexion run')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set output filename to match naming convention
    trajectories_filename = "trajectories_valid_seen.json"
    
    # Clean up old log files for a fresh run
    # Note: trajectories_valid_unseen.json is already renamed, so we don't delete it
    for old_file in ['trajectories.json', 'reflexions.json', 'world.log']:
        old_path = os.path.join(log_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Load config
    config = load_config(config_path)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Load tasks from config file
    task_files = load_tasks(base_dir, TASKS_CONFIG_PATH)
    print(f"Loaded {len(task_files)} tasks from {TASKS_CONFIG_PATH}")

    if not task_files:
        print("No tasks found. Exiting.")
        return

    # Initialize Framework
    framework = MemoryAllocationReflexion(model="gemini-2.5-flash", to_print=True)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    # Initialize env configs
    env_configs = []
    for i, task_info in enumerate(task_files):
        env_configs.append({
            'name': task_info['id'],
            'memory': [],
            'is_success': False,
            'skip': False
        })

    world_log_path = os.path.join(log_dir, 'world.log')

    print(f"""
    -----
    Starting Vanilla Reflexion Run (valid_seen):
    Number of trials: {NUM_TRIALS}
    Number of tasks: {len(task_files)}
    Log directory: {log_dir}
    Output file: {trajectories_filename}
    -----
    """)

    # Trial loop
    for trial_idx in range(NUM_TRIALS):
        # Log trial start
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')
        
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_idx + 1}/{NUM_TRIALS}")
        print(f"{'='*60}")

        trial_successes = 0
        trial_failures = 0
        additional_successes = 0

        # Run each task
        for i, task_info in enumerate(task_files):
            # Skip if already successful
            if env_configs[i]['is_success'] or env_configs[i]['skip']:
                with open(world_log_path, 'a') as wf:
                    wf.write(f"Task #{i} Trial #{trial_idx}: SUCCESS (previous)\n")
                trial_successes += 1
                print(f"\nTask {i}: {task_info['id']} - SKIPPED (already successful)")
                continue

            print(f"\nRunning Task {i}: {task_info['id']}")
            
            current_task_config = config.copy()
            current_task_config['dataset']['eval_ood_data_path'] = task_info['path']
            current_task_config['general']['evaluate']['batch_size'] = 1
            
            env = AlfworldEnv(current_task_config, split='eval_out_of_distribution')
            
            try:
                ob, info = env.reset()
            except IndexError:
                print(f"Error resetting environment for {task_info['id']}. Skipping.")
                env_configs[i]['skip'] = True
                continue
            
            # Extract task description
            task_desc = ""
            if "Your task is to:" in ob:
                task_desc = ob.split("Your task is to:")[-1].strip().split("\n")[0].strip()
            elif ob.strip():
                for line in ob.split("\n"):
                    if line.strip():
                        task_desc = line.strip()
                        break
            
            # Select prompt based on task name
            short_task_name = task_info['id'].split('-')[0]
            prompt_key = "react_put_0"  # Default
            
            if "pick_cool" in short_task_name:
                prompt_key = "react_cool_0"
            elif "pick_heat" in short_task_name:
                prompt_key = "react_heat_0"
            elif "pick_clean" in short_task_name:
                prompt_key = "react_clean_0"
            elif "pick_two" in short_task_name:
                prompt_key = "react_puttwo_0"
            elif "look_at" in short_task_name:
                prompt_key = "react_examine_0"
            elif "pick_and_place" in short_task_name:
                prompt_key = "react_put_0"
                
            prompt = prompts.get(prompt_key, "")
            if not prompt:
                print(f"Warning: Prompt not found for key {prompt_key}. Using default.")
                prompt = prompts.get("react_put_0", "")

            # Get memory for this task
            memory = env_configs[i]['memory']
            
            try:
                history, success = framework.run(
                    env=env,
                    base_prompt=prompt,
                    memory=memory,
                    start_ob=ob,
                    task_id=task_info['id'],
                    trial_num=trial_idx + 1,
                    log_dir=log_dir,
                    task_desc=task_desc
                )
                
                env_configs[i]['is_success'] = success
                
                status = "SUCCESS" if success else "FAIL"
                with open(world_log_path, 'a') as wf:
                    wf.write(f"Task #{i} Trial #{trial_idx}: {status}\n")
                
                if success:
                    trial_successes += 1
                    additional_successes += 1
                    print(f"Task {task_info['id']} - SUCCESS!")
                else:
                    trial_failures += 1
                    print(f"Task {task_info['id']} - FAIL")
                    
            except Exception as e:
                print(f"Error running task {task_info['id']}: {e}")
                import traceback
                traceback.print_exc()
                trial_failures += 1
                with open(world_log_path, 'a') as wf:
                    wf.write(f"Task #{i} Trial #{trial_idx}: FAIL (error)\n")
            finally:
                env.close()

        # Generate reflexions for failed tasks (after all tasks in trial)
        for i, task_info in enumerate(task_files):
            if not env_configs[i]['is_success'] and not env_configs[i]['skip']:
                # Read the latest reflexion from the file
                reflexions_path = os.path.join(log_dir, 'reflexions.json')
                if os.path.exists(reflexions_path):
                    with open(reflexions_path, 'r') as f:
                        reflexions = json.load(f)
                        for entry in reversed(reflexions):
                            if entry['task_id'] == task_info['id'] and entry['trial_num'] == trial_idx + 1:
                                if entry['reflexion']:
                                    env_configs[i]['memory'].append(entry['reflexion'])
                                break

        # Log trial summary
        total_tasks = len(task_files)
        accuracy = trial_successes / total_tasks if total_tasks > 0 else 0
        
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n-----\n')
            wf.write(f'SUCCESS: {trial_successes}\n')
            wf.write(f'ADDITIONAL SUCCESS: {additional_successes}\n')
            wf.write(f'FAIL: {trial_failures}\n')
            wf.write(f'TOTAL: {total_tasks}\n')
            wf.write(f'ACCURACY: {accuracy:.2f}\n')
            wf.write(f'-----\n')
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        print(f"\n--- Trial {trial_idx + 1} Summary ---")
        print(f"SUCCESS: {trial_successes}, FAIL: {trial_failures}, ACCURACY: {accuracy:.2f}")

        # Check if all tasks are done
        if all(ec['is_success'] or ec['skip'] for ec in env_configs):
            print(f"\nAll tasks completed successfully after {trial_idx + 1} trials!")
            break

    # Rename trajectories.json to trajectories_valid_seen.json
    old_traj_path = os.path.join(log_dir, 'trajectories.json')
    new_traj_path = os.path.join(log_dir, trajectories_filename)
    if os.path.exists(old_traj_path):
        os.rename(old_traj_path, new_traj_path)
        print(f"\nRenamed trajectories.json to {trajectories_filename}")

    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Final results saved to: {log_dir}")


if __name__ == "__main__":
    run_vanilla_reflexion()

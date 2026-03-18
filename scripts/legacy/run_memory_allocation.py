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
NUM_TASKS = 120  # Number of tasks to run
TASK_TYPE_FILTER = None  # Filter for specific task type (None for all)
SPLIT_FILTER = "train"  # Filter for specific split (None for all, or 'train', 'valid_seen', 'valid_unseen')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def discover_tasks(base_dir, task_type_filter=None, num_tasks=20, split_filter=None):
    """
    Dynamically discover tasks from alfworld_mini dataset.
    
    Args:
        base_dir: Project root directory
        task_type_filter: Filter tasks by type (e.g., 'pick_and_place', 'pick_heat', etc.)
        num_tasks: Maximum number of tasks to return
        split_filter: Filter tasks by split (e.g., 'train', 'valid_seen', 'valid_unseen')
    
    Returns:
        List of task names
    """
    tasks = []
    if split_filter:
        splits = [split_filter]
    else:
        splits = ['train', 'valid_seen', 'valid_unseen']
    
    for split in splits:
        split_dir = os.path.join(base_dir, 'alfworld_mini', split)
        if not os.path.exists(split_dir):
            continue
            
        for task_name in sorted(os.listdir(split_dir)):
            task_path = os.path.join(split_dir, task_name)
            if not os.path.isdir(task_path):
                continue
                
            # Apply task type filter
            if task_type_filter and not task_name.startswith(task_type_filter):
                continue
            
            # Check if there's a valid trial directory with game.tw-pddl
            trial_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            if trial_dirs:
                tasks.append((split, task_name))
                
            if len(tasks) >= num_tasks:
                break
        
        if len(tasks) >= num_tasks:
            break
    
    return tasks[:num_tasks]

def run_memory_allocation_test():
    # Setup paths - base_dir should be project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'data', 'alfworld', 'base_config.yaml')
    log_dir = os.path.join(base_dir, 'alfworld_runs', 'memory_allocation_test')
    os.makedirs(log_dir, exist_ok=True)
    
    # Clean up old log files for a fresh run
    for old_file in ['trajectories.json', 'reflexions.json', 'world.log', 'trajectories.jsonl', 'reflexions.jsonl']:
        old_path = os.path.join(log_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Load config
    config = load_config(config_path)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Discover tasks dynamically
    discovered_tasks = discover_tasks(base_dir, task_type_filter=TASK_TYPE_FILTER, num_tasks=NUM_TASKS, split_filter=SPLIT_FILTER)
    print(f"Discovered {len(discovered_tasks)} tasks of type '{TASK_TYPE_FILTER}'")
    
    # Search for the task json files
    task_files = []
    
    for split, task_name in discovered_tasks:
        task_dir = os.path.join(base_dir, 'alfworld_mini', split, task_name)
        
        # Check if task dir exists
        if not os.path.exists(task_dir):
            print(f"Warning: Task directory not found: {task_dir}")
            continue

        # Find the first trial directory
        trial_dirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not trial_dirs:
            print(f"Warning: No trial directories found in {task_dir}")
            continue
            
        # Sort to ensure reproducibility if needed, though picking first is fine
        trial_dirs.sort()
        trial_dir = os.path.join(task_dir, trial_dirs[0])
        game_file = os.path.join(trial_dir, 'game.tw-pddl')
        
        if os.path.exists(game_file):
            task_files.append({
                "path": trial_dir, # Alfworld expects the directory containing the .tw-pddl file
                "id": f"{task_name}/{trial_dirs[0]}",
                "file": game_file,
                "split": split
            })
        else:
            print(f"Warning: Could not find game.tw-pddl in {trial_dir}")

    if not task_files:
        print("No tasks found. Exiting.")
        return

    # Initialize Framework
    framework = MemoryAllocationReflexion(model="gemini-2.5-flash", to_print=True)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    # Initialize env configs (following legacy/alfworld_old/main.py structure)
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
    Starting Memory Allocation Reflexion Run:
    Number of trials: {NUM_TRIALS}
    Number of tasks: {len(task_files)}
    Log directory: {log_dir}
    -----
    """)

    # Trial loop (following legacy/alfworld_old/main.py structure)
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
            
            # Extract task description from the first line of observation
            # Format: "Your task is to: <task description>"
            task_desc = ""
            if "Your task is to:" in ob:
                task_desc = ob.split("Your task is to:")[-1].strip().split("\n")[0].strip()
            elif ob.strip():
                # Fallback: use first non-empty line as task desc
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

            # Get memory for this task (use last 3 reflexions as in original)
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
                # Reflexion is already generated in framework.run() and logged
                # We just need to add it to memory for next trial
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

    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Final results saved to: {log_dir}")

if __name__ == "__main__":
    run_memory_allocation_test()

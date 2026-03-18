"""
Runner script for Memory Agent

This script runs the memory-augmented agent on ALFWorld tasks.
The agent:
1. Receives context from previous similar tasks at the start
2. Can call help["query"] during execution to get targeted guidance
"""

import os
import sys
import yaml
import json
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_tasks(base_dir, task_type_filter=None, num_tasks=20, splits=None):
    """
    Dynamically discover tasks from alfworld_mini dataset.
    
    Args:
        base_dir: Project root directory
        task_type_filter: Filter tasks by type (e.g., 'pick_and_place', 'pick_heat', etc.)
        num_tasks: Maximum number of tasks to return
        splits: List of splits to search (default: ['valid_unseen', 'valid_seen', 'train'])
    
    Returns:
        List of (split, task_name) tuples
    """
    tasks = []
    if splits is None:
        splits = ['valid_unseen', 'valid_seen', 'train']
    
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


def run_memory_agent():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Memory Agent on ALFWorld tasks")
    parser.add_argument(
        "--num-tasks", 
        type=int, 
        default=5,
        help="Number of tasks to run (default: 5)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Task type filter (e.g., 'pick_and_place', 'pick_heat', 'pick_cool', 'look_at', 'pick_two', 'pick_clean')"
    )
    parser.add_argument(
        "--memory-bank",
        type=str,
        default=None,
        help="Path to knowledge_base.json (default: alfworld_runs/memory_allocation_test/knowledge_base.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for logs (default: alfworld_runs/memory_agent_test)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of specific tasks to run in format 'split:task_id', e.g., 'train:pick_and_place_simple-Book-None-Sofa-202/trial_T20190907_043652_096323'"
    )
    
    args = parser.parse_args()
    
    # Setup paths - base_dir should be project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'data', 'alfworld', 'base_config.yaml')
    
    # Default output directory with timestamped subfolder
    if args.output_dir:
        # User provided explicit path - use as-is
        log_dir = args.output_dir
    else:
        # Auto-generate timestamped subfolder under memory_agent_runs
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_dir, 'alfworld_runs', 'memory_retrieval_v2', 'memory_agent_runs', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Default memory bank path
    if args.memory_bank:
        memory_bank_path = args.memory_bank
    else:
        memory_bank_path = os.path.join(
            base_dir, 'alfworld_runs', 'memory_allocation_test', 'knowledge_base.json'
        )
    
    # Verify memory bank exists
    if not os.path.exists(memory_bank_path):
        print(f"Error: Memory bank not found at {memory_bank_path}")
        print("Please run the memory allocation script first to generate the knowledge base.")
        return
    
    # Clean up old log files for a fresh run
    for old_file in ['trajectories.json', 'world.log']:
        old_path = os.path.join(log_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Load config
    config = load_config(config_path)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Build task list: either from explicit --tasks argument or via dynamic discovery
    task_files = []
    
    if args.tasks:
        # Parse explicit task specifications: "split:task_id,split:task_id,..."
        task_specs = [t.strip() for t in args.tasks.split(',')]
        print(f"Running {len(task_specs)} explicitly specified task(s)")
        
        for spec in task_specs:
            if ':' not in spec:
                print(f"Warning: Invalid task spec '{spec}'. Expected format 'split:task_id'. Skipping.")
                continue
            
            split, task_id = spec.split(':', 1)
            # task_id is in format "task_name/trial_name"
            if '/' in task_id:
                task_name, trial_name = task_id.rsplit('/', 1)
            else:
                # If no trial specified, we'll find the first one
                task_name = task_id
                trial_name = None
            
            task_dir = os.path.join(base_dir, 'alfworld_mini', split, task_name)
            
            if not os.path.exists(task_dir):
                print(f"Warning: Task directory not found: {task_dir}")
                continue
            
            if trial_name:
                trial_dir = os.path.join(task_dir, trial_name)
            else:
                # Find the first trial directory
                trial_dirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
                if not trial_dirs:
                    print(f"Warning: No trial directories found in {task_dir}")
                    continue
                trial_dirs.sort()
                trial_name = trial_dirs[0]
                trial_dir = os.path.join(task_dir, trial_name)
            
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
    else:
        # Discover tasks dynamically
        discovered_tasks = discover_tasks(
            base_dir, 
            task_type_filter=args.task_type, 
            num_tasks=args.num_tasks
        )
        
        task_type_str = args.task_type if args.task_type else "all"
        print(f"Discovered {len(discovered_tasks)} tasks of type '{task_type_str}'")
        
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
                
            # Sort to ensure reproducibility
            trial_dirs.sort()
            trial_dir = os.path.join(task_dir, trial_dirs[0])
            game_file = os.path.join(trial_dir, 'game.tw-pddl')
            
            if os.path.exists(game_file):
                task_files.append({
                    "path": trial_dir,
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
    framework = MemoryAgent(model=args.model, to_print=not args.quiet)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    world_log_path = os.path.join(log_dir, 'world.log')

    print(f"""
    -----
    Starting Memory Agent Run:
    Number of tasks: {len(task_files)}
    Memory bank: {memory_bank_path}
    Log directory: {log_dir}
    Model: {args.model}
    -----
    """)

    # Results tracking
    successes = 0
    failures = 0
    total_help_calls = 0

    # Run each task (single trial per task for evaluation)
    for i, task_info in enumerate(task_files):
        print(f"\n{'='*60}")
        print(f"TASK {i + 1}/{len(task_files)}: {task_info['id']}")
        print(f"{'='*60}")
        
        current_task_config = config.copy()
        current_task_config['dataset']['eval_ood_data_path'] = task_info['path']
        current_task_config['general']['evaluate']['batch_size'] = 1
        
        env = AlfworldEnv(current_task_config, split='eval_out_of_distribution')
        
        try:
            ob, info = env.reset()
        except IndexError:
            print(f"Error resetting environment for {task_info['id']}. Skipping.")
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

        try:
            history, success = framework.run(
                env=env,
                base_prompt=prompt,
                memory=[],  # No reflexion memory for memory agent
                start_ob=ob,
                task_id=task_info['id'],
                trial_num=1,
                log_dir=log_dir,
                task_desc=task_desc,
                memory_bank_path=memory_bank_path
            )
            
            status = "SUCCESS" if success else "FAIL"
            with open(world_log_path, 'a') as wf:
                wf.write(f"Task #{i}: {task_info['id']} - {status}\n")
            
            if success:
                successes += 1
                print(f"\n✓ Task SUCCESS!")
            else:
                failures += 1
                print(f"\n✗ Task FAIL")
                
        except Exception as e:
            print(f"Error running task {task_info['id']}: {e}")
            import traceback
            traceback.print_exc()
            failures += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f"Task #{i}: {task_info['id']} - FAIL (error)\n")
        finally:
            env.close()

    # Final summary
    total = successes + failures
    accuracy = successes / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print("RUN COMPLETE")
    print(f"{'='*60}")
    print(f"SUCCESS: {successes}/{total} ({accuracy:.1%})")
    print(f"Results saved to: {log_dir}")
    
    # Write final summary to world.log
    with open(world_log_path, 'a') as wf:
        wf.write(f"\n-----\n")
        wf.write(f"FINAL SUMMARY\n")
        wf.write(f"SUCCESS: {successes}\n")
        wf.write(f"FAIL: {failures}\n")
        wf.write(f"TOTAL: {total}\n")
        wf.write(f"ACCURACY: {accuracy:.2f}\n")
        wf.write(f"-----\n")


if __name__ == "__main__":
    run_memory_agent()

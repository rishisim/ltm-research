"""
Runner script for Hard Negative Memory Agent with batch task file support

This script runs the HARD NEGATIVE memory agent on ALFWorld tasks.
The hard negative agent retrieves LEAST relevant learnings (bottom similarity).
"""

import os
import sys
import yaml
import json
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_allocation.hard_neg_memory_agent import HardNegMemoryAgent


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_hard_neg_memory_agent_batch():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Hard Negative Memory Agent on ALFWorld tasks")
    parser.add_argument(
        "--task-file", 
        type=str, 
        required=True,
        help="Path to JSON file containing task list"
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
        help="Output directory for logs (default: alfworld_runs/memory_agent_test/hard_neg_CR+TR)"
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
    
    args = parser.parse_args()
    
    # Setup paths - base_dir should be project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'data', 'alfworld', 'base_config.yaml')
    
    # Load task file
    with open(args.task_file, 'r') as f:
        task_data = json.load(f)
    
    task_specs = task_data.get('tasks', [])
    print(f"Loaded {len(task_specs)} tasks from {args.task_file}")
    
    # Default output directory - HARD NEG specific
    if args.output_dir:
        log_dir = args.output_dir
    else:
        log_dir = os.path.join(base_dir, 'alfworld_runs', 'memory_agent_test', 'hard_neg_CR+TR')
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

    # Build task list from task file
    task_files = []
    
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

    if not task_files:
        print("No tasks found. Exiting.")
        return

    # Initialize HARD NEGATIVE Framework
    framework = HardNegMemoryAgent(model=args.model, to_print=not args.quiet)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    world_log_path = os.path.join(log_dir, 'world.log')

    print(f"""
    -----
    Starting HARD NEGATIVE Memory Agent Batch Run:
    Number of tasks: {len(task_files)}
    Memory bank: {memory_bank_path}
    Log directory: {log_dir}
    Model: {args.model}
    NOTE: This uses BOTTOM similarity (least relevant learnings)
    -----
    """)

    # Results tracking
    successes = 0
    failures = 0

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
    print("HARD NEGATIVE RUN COMPLETE")
    print(f"{'='*60}")
    print(f"SUCCESS: {successes}/{total} ({accuracy:.1%})")
    print(f"Results saved to: {log_dir}")
    
    # Write final summary to world.log
    with open(world_log_path, 'a') as wf:
        wf.write(f"\n-----\n")
        wf.write(f"HARD NEGATIVE FINAL SUMMARY\n")
        wf.write(f"SUCCESS: {successes}\n")
        wf.write(f"FAIL: {failures}\n")
        wf.write(f"TOTAL: {total}\n")
        wf.write(f"ACCURACY: {accuracy:.2f}\n")
        wf.write(f"-----\n")


if __name__ == "__main__":
    run_hard_neg_memory_agent_batch()

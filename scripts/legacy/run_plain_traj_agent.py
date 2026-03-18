"""
Runner script for Plain Trajectory Agent (Ablation Study)

This script runs the plain trajectory agent on ALFWorld tasks.
The agent:
1. Retrieves top-K similar trajectories from training data
2. Provides truncated (130 words each) raw trajectory text as context
3. Does NOT have access to help["query"] during execution

This is an ablation to test whether raw trajectories provide better context
than extracted knowledge base learnings.
"""

import os
import sys
import yaml
import json
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_retrieval_v2.agents.plain_traj_agent import PlainTrajAgent


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_plain_traj_agent():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Plain Trajectory Agent on ALFWorld tasks")
    parser.add_argument(
        "--trajectories-path",
        type=str,
        default=None,
        help="Path to training trajectories.json (default: alfworld_runs/memory_retrieval_v2/knowledge_base/memory_allocation_runs/trajectories.json)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run (default: all tasks in valid_unseen)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Filter to specific task type (e.g., 'look_at_obj', 'pick_and_place')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top similar trajectories to retrieve (default: 1)"
    )
    parser.add_argument(
        "--word-limit",
        type=int,
        default=130,
        help="Maximum word count for each trajectory (default: 130)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: alfworld_runs/plain_traj_agent/valid_unseen)"
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
        "--split",
        type=str,
        default="valid_unseen",
        choices=["valid_seen", "valid_unseen"],
        help="Dataset split to use (default: valid_unseen)"
    )
    
    args = parser.parse_args()
    
    # Setup paths - base_dir should be project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'data', 'alfworld', 'base_config.yaml')
    
    # Default trajectories path
    if args.trajectories_path:
        trajectories_path = args.trajectories_path
    else:
        trajectories_path = os.path.join(
            base_dir, 'alfworld_runs', 'memory_retrieval_v2', 'knowledge_base', 
            'memory_allocation_runs', 'trajectories.json'
        )
    
    # Verify trajectories file exists
    if not os.path.exists(trajectories_path):
        print(f"Error: Trajectories file not found at {trajectories_path}")
        print("Please specify a valid --trajectories-path")
        return
    
    # Setup output directory
    if args.output_dir:
        log_dir = args.output_dir
    else:
        log_dir = os.path.join(base_dir, 'alfworld_runs', 'plain_traj_agent', args.split)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Clean up old log files for a fresh run
    for old_file in ['trajectories.json', 'world.log']:
        old_path = os.path.join(log_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Load config
    config = load_config(config_path)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Get all tasks from the dataset
    dataset_path = os.path.join(base_dir, 'alfworld_mini', args.split)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Collect all task files
    task_files = []
    for task_name in sorted(os.listdir(dataset_path)):
        task_dir = os.path.join(dataset_path, task_name)
        
        if not os.path.isdir(task_dir):
            continue
        
        # Filter by task type if specified
        if args.task_type and not task_name.startswith(args.task_type):
            continue
        
        # Find trial directories
        for trial_name in sorted(os.listdir(task_dir)):
            trial_dir = os.path.join(task_dir, trial_name)
            
            if not os.path.isdir(trial_dir):
                continue
            
            game_file = os.path.join(trial_dir, 'game.tw-pddl')
            
            if os.path.exists(game_file):
                task_files.append({
                    "path": trial_dir,
                    "id": f"{task_name}/{trial_name}",
                    "file": game_file,
                    "split": args.split
                })

    if not task_files:
        print("No tasks found. Exiting.")
        return
    
    # Limit number of tasks if specified
    if args.num_tasks:
        task_files = task_files[:args.num_tasks]

    # Initialize Framework
    framework = PlainTrajAgent(model=args.model, to_print=not args.quiet)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    world_log_path = os.path.join(log_dir, 'world.log')

    print(f"""
    -----
    Starting Plain Trajectory Agent Run:
    Number of tasks: {len(task_files)}
    Trajectories file: {trajectories_path}
    Top-K: {args.top_k}
    Word limit per trajectory: {args.word_limit}
    Log directory: {log_dir}
    Model: {args.model}
    Split: {args.split}
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

        # Create task-specific log directory
        task_log_dir = os.path.join(log_dir, task_info['id'].replace('/', '_'))
        os.makedirs(task_log_dir, exist_ok=True)

        try:
            history, success = framework.run(
                env=env,
                base_prompt=prompt,
                memory=[],  # No reflexion memory for trajectory agent
                start_ob=ob,
                task_id=task_info['id'],
                trial_num=1,
                log_dir=task_log_dir,
                task_desc=task_desc,
                trajectories_path=trajectories_path,
                top_k=args.top_k,
                word_limit=args.word_limit
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
        wf.write(f"FINAL SUMMARY (Plain Trajectory Agent)\n")
        wf.write(f"Top-K: {args.top_k}\n")
        wf.write(f"Word limit: {args.word_limit}\n")
        wf.write(f"SUCCESS: {successes}\n")
        wf.write(f"FAIL: {failures}\n")
        wf.write(f"TOTAL: {total}\n")
        wf.write(f"ACCURACY: {accuracy:.2f}\n")
        wf.write(f"-----\n")


if __name__ == "__main__":
    run_plain_traj_agent()

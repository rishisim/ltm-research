"""
Runner script for Tool Retrieval Only Agent (Ablation Study) - Valid Seen Split

This script runs the tool-retrieval-only agent on ALFWorld valid_seen tasks.
The agent:
1. Does NOT receive context from previous similar tasks at the start
2. HAS access to help["query"] during execution

This is an ablation to measure the impact of tool retrieval alone.
"""

import os
import sys
import yaml
import json
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_allocation.tool_only_agent import ToolRetrievalOnlyAgent


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_tool_retrieval_only_ablation_seen():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Tool Retrieval Only Agent on ALFWorld valid_seen tasks (ablation)")
    parser.add_argument(
        "--memory-bank",
        type=str,
        default=None,
        help="Path to knowledge_base.json (default: alfworld_runs/memory_allocation_test/knowledge_base.json)"
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
    
    # Output to "with TR" folder
    log_dir = os.path.join(base_dir, 'alfworld_runs', 'memory_agent_test', 'with TR')
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
        return
    
    # Clean up old log files for a fresh run
    for old_file in ['trajectories_valid_seen.json', 'world_seen.log']:
        old_path = os.path.join(log_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Load config
    config = load_config(config_path)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Load task IDs from valid_seen_batch_tasks.json
    tasks_file = os.path.join(
        base_dir, 'alfworld_runs', 'memory_agent_test', 'vanilla reflexion run', 'valid_seen_batch_tasks.json'
    )
    
    with open(tasks_file, 'r') as f:
        tasks_data = json.load(f)
    
    task_specs = tasks_data.get('tasks', [])
    
    # Build task files list from task specs
    task_files = []
    for spec in task_specs:
        if ':' not in spec:
            continue
        
        split, task_id = spec.split(':', 1)
        if '/' in task_id:
            task_name, trial_name = task_id.rsplit('/', 1)
        else:
            task_name = task_id
            trial_name = None
        
        task_dir = os.path.join(base_dir, 'alfworld_mini', split, task_name)
        if not os.path.exists(task_dir):
            continue
        
        if trial_name:
            trial_dir = os.path.join(task_dir, trial_name)
        else:
            trial_dirs = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
            if not trial_dirs:
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

    # Initialize Framework
    framework = ToolRetrievalOnlyAgent(model=args.model, to_print=not args.quiet)
    
    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    world_log_path = os.path.join(log_dir, 'world_seen.log')

    print(f"""
    -----
    Starting Tool Retrieval Only Ablation Run (NO CONTEXT RETRIEVAL) - VALID SEEN:
    Number of tasks: {len(task_files)}
    Memory bank: {memory_bank_path}
    Log directory: {log_dir}
    Model: {args.model}
    -----
    """)

    successes = 0
    failures = 0

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
        except Exception as e:
            print(f"Error resetting env: {e}")
            continue
        
        task_desc = ""
        if "Your task is to:" in ob:
            task_desc = ob.split("Your task is to:")[-1].strip().split("\n")[0].strip()
        
        short_task_name = task_info['id'].split('-')[0]
        prompt_key = "react_put_0" 
        
        if "pick_cool" in short_task_name: prompt_key = "react_cool_0"
        elif "pick_heat" in short_task_name: prompt_key = "react_heat_0"
        elif "pick_clean" in short_task_name: prompt_key = "react_clean_0"
        elif "pick_two" in short_task_name: prompt_key = "react_puttwo_0"
        elif "look_at" in short_task_name: prompt_key = "react_examine_0"
            
        prompt = prompts.get(prompt_key, prompts.get("react_put_0", ""))

        try:
            history, success = framework.run(
                env=env,
                base_prompt=prompt,
                memory=[], 
                start_ob=ob,
                task_id=task_info['id'],
                trial_num=1,
                log_dir=log_dir,
                task_desc=task_desc,
                memory_bank_path=memory_bank_path,
                trajectory_file="trajectories_valid_seen.json"
            )
            
            if success: successes += 1
            else: failures += 1
                
        except Exception as e:
            print(f"Error running task: {e}")
            failures += 1
        finally:
            env.close()

    total = successes + failures
    accuracy = successes / total if total > 0 else 0
    
    print(f"\nFinal Accuracy: {successes}/{total} ({accuracy:.1%})")


if __name__ == "__main__":
    run_tool_retrieval_only_ablation_seen()

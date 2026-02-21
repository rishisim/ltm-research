#!/usr/bin/env python3
"""
Runner script for DedicatedTaskAgent on a specific task
"""

import os
import sys
import json
import yaml
import importlib.util

# Add project root to sys.path
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, base_dir)

from src.envs.alfworld_env import AlfworldEnv

# Import the DedicatedTaskAgent directly from file
spec = importlib.util.spec_from_file_location(
    "dedicated_task_agent",
    os.path.join(base_dir, "alfworld_runs/memory_retrieval_v2/memory_agent_runs/dedicated_task_ agent.py")
)
dedicated_task_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedicated_task_module)
DedicatedTaskAgent = dedicated_task_module.DedicatedTaskAgent

def main():
    # Setup directories
    run_dir = os.path.join(base_dir, "alfworld_runs", "memory_retrieval_v2", "memory_agent_runs")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(base_dir, "data", "alfworld", "base_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set ALFWORLD_DATA environment variable
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')
    
    # Load prompts
    prompts_path = os.path.join(base_dir, "data", "alfworld", "prompts", "alfworld_3prompts.json")
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    
    # Target task specification
    target_split = "train"
    task_name = "pick_heat_then_place_in_recep-Plate-None-CounterTop-1"
    trial_name = "trial_T20190909_115633_911483"
    target_task_id = f"{task_name}/{trial_name}"
    
    print(f"\n{'='*60}")
    print(f"Running DedicatedTaskAgent on task:")
    print(f"  Task ID: {target_task_id}")
    print(f"  Split: {target_split}")
    print(f"{'='*60}\n")
    
    # Build task path
    task_dir = os.path.join(base_dir, "data", "alfworld_mini", target_split, task_name)
    trial_dir = os.path.join(task_dir, trial_name)
    game_file = os.path.join(trial_dir, 'game.tw-pddl')
    
    print(f"Task directory: {task_dir}")
    print(f"Trial directory: {trial_dir}")
    print(f"Game file: {game_file}\n")
    
    if not os.path.exists(game_file):
        print(f"ERROR: Game file not found: {game_file}")
        return
    
    # Initialize environment with task-specific config
    task_config = config.copy()
    task_config['dataset']['eval_ood_data_path'] = trial_dir
    
    env = AlfworldEnv(task_config, split='eval_out_of_distribution')
    
    # Reset environment
    try:
        ob, info = env.reset()
    except Exception as e:
        print(f"ERROR resetting environment: {e}")
        return
    
    # Extract task description
    task_desc = ""
    if "Your task is to:" in ob:
        task_desc = ob.split("Your task is to:")[-1].strip().split("\n")[0].strip()
    elif ob.strip():
        for line in ob.split("\n"):
            if line.strip() and not line.startswith('-'):
                task_desc = line.strip()
                break
    
    if not task_desc:
        task_desc = ob[:100]
    
    print(f"Task description: {task_desc}\n")
    
    # Load the appropriate prompt for heat task
    base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + prompts['react_heat_1'] + prompts['react_heat_0']
    
    # Initialize agent
    agent = DedicatedTaskAgent(model="gemini-2.5-flash", to_print=True)
    
    # Run agent
    env_history, is_success = agent.run(
        env=env,
        base_prompt=base_prompt,
        memory=[],
        start_ob=ob,
        task_id=task_name,
        trial_num=1,
        log_dir=run_dir,
        task_desc=task_desc,
        memory_bank_path="",  # Not using memory bank
        trajectory_file="trajectories_dedicated_task.json"
    )
    
    print(f"\n{'='*60}")
    print(f"Task Result: {'SUCCESS' if is_success else 'FAILURE'}")
    print(f"{'='*60}\n")
    
    # Save summary
    summary = {
        "task_id": target_task_id,
        "task_desc": task_desc,
        "success": is_success,
        "agent": "DedicatedTaskAgent",
        "model": "gemini-2.5-flash"
    }
    
    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Logs saved to: {run_dir}")
    print(f"  - trajectories_dedicated_task.json")
    print(f"  - run_summary.json")

if __name__ == "__main__":
    main()

import os
import sys
import json
import argparse
import yaml
from typing import Any, List, Dict

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.react import ReAct
from src.frameworks.reflexion import Reflexion
from src.frameworks.memory_allocation import MemoryAllocationReflexion

# Task types mapping
PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--use_memory", action='store_true', help="Use standard Reflexion (memory across trials)")
    parser.add_argument("--framework", type=str, choices=['react', 'reflexion', 'in-trajectory'], default='react')
    parser.add_argument("--dataset", type=str, choices=['official', 'mini'], default='official')
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    return parser.parse_args()

def main(args):
    # Setup directories
    run_dir = os.path.join("alfworld_runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    traj_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    world_log_path = os.path.join(run_dir, 'world.log')
    
    # Load config and prompts
    config_path = "data/alfworld/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.dataset == 'mini':
        config['dataset']['data_path'] = 'alfworld_mini/train'
        config['dataset']['eval_id_data_path'] = 'alfworld_mini/valid_seen'
        config['dataset']['eval_ood_data_path'] = 'alfworld_mini/valid_unseen'

    prompts_path = "data/alfworld/prompts/alfworld_3prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    # Initialize environment configs
    env_configs = []
    for i in range(args.num_envs):
        env_configs.append({'name': f'env_{i}', 'memory': [], 'is_success': False})

    # Selection logic for frameworks
    if args.framework == 'in-trajectory':
        agent = MemoryAllocationReflexion(model=args.model)
    elif args.framework == 'reflexion' or args.use_memory:
        agent = Reflexion(model=args.model)
    else:
        agent = ReAct(model=args.model)

    # Initialize environment
    env = AlfworldEnv(config)

    for trial_idx in range(args.num_trials):
        print(f"\n***** Start Trial #{trial_idx} *****\n")
        trial_log_path = os.path.join(run_dir, f'trial_{trial_idx}.log')
        
        for z, env_config in enumerate(env_configs):
            if env_config["is_success"]:
                continue

            # Reset environment to get a task
            ob, info = env.reset()
            game_file = info['extra.gamefile'][0]
            name = '/'.join(game_file.split('/')[-3:-1])
            
            # Identify task type
            found_type = None
            for k in PREFIXES.keys():
                if name.startswith(k):
                    found_type = k
                    break
            
            if not found_type:
                continue # or handle error

            v = PREFIXES[found_type]
            base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + prompts[f'react_{v}_1'] + prompts[f'react_{v}_0']
            
            print(f"Executing Env #{z}: {name}")
            run_kwargs = {
                "env": env,
                "base_prompt": base_prompt,
                "memory": env_config["memory"],
                "start_ob": ob
            }
            if isinstance(agent, MemoryAllocationReflexion):
                run_kwargs.update({
                    "task_id": name,
                    "task_type": found_type,
                    "env_id": env_config["name"],
                    "log_dir": run_dir
                })
            
            history, is_success = agent.run(**run_kwargs)
            
            # Update results
            if is_success:
                env_config['is_success'] = True
                status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            else:
                status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'

            # Logging
            with open(world_log_path, 'a') as f:
                f.write(status_str + '\n')
            
            with open(trial_log_path, 'a') as f:
                f.write(f"\n#####\n\nEnvironment #{z}:\n{str(history)}\n\nSTATUS: {'OK' if is_success else 'FAIL'}\n\n#####\n")

            # Save trajectory
            traj_path = os.path.join(traj_dir, f'env_{z}_trial_{trial_idx}.json')
            with open(traj_path, 'w') as f:
                json.dump(history.to_json(), f, indent=4)

        # Post-trial memory update
        if args.use_memory:
            env_configs = agent.update_memory(env_configs, trial_log_path)
        
        # Save env_configs snapshot
        snapshot_path = os.path.join(run_dir, f'env_results_trial_{trial_idx}.json')
        with open(snapshot_path, 'w') as f:
            json.dump(env_configs, f, indent=4)

    env.close()
    print("\nRun complete.")

if __name__ == "__main__":
    main(get_args())

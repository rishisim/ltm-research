
import os
import sys
import yaml
import json
import alfworld.agents.environment as envs

# Ensure src is in path
sys.path.append(os.getcwd())

from src.frameworks.memory_allocation.rag_react import RAGReAct

class SingleEnvWrapper:
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        obs, scores, dones, infos = self.env.step([action])
        # infos is dict where values are lists
        single_info = {}
        for k, v in infos.items():
            if isinstance(v, list) and len(v) > 0:
                single_info[k] = v[0]
            else:
                single_info[k] = v
        return obs[0], scores[0], dones[0], single_info

    def reset(self):
        obs, infos = self.env.reset()
        single_info = {}
        for k, v in infos.items():
            if isinstance(v, list) and len(v) > 0:
                single_info[k] = v[0]
            else:
                single_info[k] = v
        return obs[0], single_info

def main():
    # Set up environment variables
    project_root = os.getcwd()
    os.environ["ALFWORLD_DATA"] = project_root
    
    # Load config
    config_path = os.path.join(project_root, "data/alfworld/base_config.yaml")
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Modify config for valid_unseen
    config['dataset']['eval_id_data_path'] = None # Disable valid_seen
    config['dataset']['data_path'] = os.path.join(project_root, 'alfworld_mini/train')
    config['dataset']['eval_ood_data_path'] = os.path.join(project_root, 'alfworld_mini/valid_unseen')
    config['dataset']['num_eval_games'] = -1
    
    # Update logic file paths
    alfworld_pkg_path = "/Users/rishisim/Documents/research/ltm-research/alfworld_runs/.alfworld_venv/lib/python3.9/site-packages/alfworld"
    config['logic']['domain'] = os.path.join(alfworld_pkg_path, "data/alfred.pddl")
    config['logic']['grammar'] = os.path.join(alfworld_pkg_path, "data/alfred.twl2")
    
    # Initialize environment
    print("Initializing environment...")
    env_type = config['env']['type']
    env_class = envs.get_environment(env_type)
    alfred_env = env_class(config, train_eval='eval_out_of_distribution')
    
    # Randomly select ONE game
    import random
    all_games = alfred_env.game_files
    num_games = len(all_games)
    print(f"Total available games: {num_games}")
    
    selected_game = random.choice(all_games)
    print(f"Selected random game: {selected_game}")
    
    # Override env game files
    alfred_env.game_files = [selected_game]
    alfred_env.num_games = 1
    
    batch_env = alfred_env.init_env(batch_size=1)
    env = SingleEnvWrapper(batch_env)
    
    # Initialize Agent
    agent = RAGReAct(to_print=True)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    # Extract task info
    # reset() guarantees extra.gamefile is present in AlfredInfos wrapper
    current_game_file = info.get('extra.gamefile', 'Unknown')
    print(f"Current task file matching: {current_game_file}")
    
    # Parse Task Description
    task_desc = "Unknown Task"
    match_start = "Here is the task:\n"
    if match_start in obs:
        start_idx = obs.find(match_start) + len(match_start)
        end_idx = obs.find("\n", start_idx)
        if end_idx == -1:
             task_desc = obs[start_idx:].strip()
        else:
             task_desc = obs[start_idx:end_idx].strip()
    elif "Your task is to:" in obs:
        parts = obs.split("Your task is to:")
        task_desc = parts[1].strip().split('\n')[0].strip()
        if task_desc.endswith('.'):
            task_desc = task_desc[:-1] # Remove trailing period

    
    print(f"Task Description: {task_desc}")
    
    # Define base prompt and memory
    base_prompt = "Interact with a household to solve a task."
    memory = []
    
    # Run Agent
    log_dir = os.path.join(project_root, "alfworld_runs/memory_agent_test/rag_react/logs")
    
    # Extract simple task id
    task_id = os.path.basename(os.path.dirname(current_game_file)) if '/' in current_game_file else "task_random"
    
    print(f"Running agent on task: {task_id}")
    
    agent.run(
        env=env,
        base_prompt=base_prompt,
        memory=memory,
        start_ob=obs,
        task_desc=task_desc,
        task_id=task_id,
        trial_num=1,
        log_dir=log_dir
    )
    
    print(f"Run completed. Logs saved to {log_dir}")
    # Don't need loop logic anymore as we just picked one
    return

if __name__ == "__main__":
    main()

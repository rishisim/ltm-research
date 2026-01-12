
import os
import sys
import yaml
import json
import alfworld.agents.environment as envs
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.getcwd())

from src.frameworks.memory_allocation.agents.rag_react import RAGReAct

class SingleEnvWrapper:
    def __init__(self, env):
        self.env = env
        
    def step(self, action):
        obs, scores, dones, infos = self.env.step([action])
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
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Modify config for valid_unseen
    config['dataset']['eval_id_data_path'] = None 
    config['dataset']['data_path'] = os.path.join(project_root, 'alfworld_mini/train')
    config['dataset']['eval_ood_data_path'] = os.path.join(project_root, 'alfworld_mini/valid_unseen')
    config['dataset']['num_eval_games'] = -1
    
    # Update logic file paths
    alfworld_pkg_path = "/Users/rishisim/Documents/research/ltm-research/alfworld_runs/.alfworld_venv/lib/python3.9/site-packages/alfworld"
    config['logic']['domain'] = os.path.join(alfworld_pkg_path, "data/alfred.pddl")
    config['logic']['grammar'] = os.path.join(alfworld_pkg_path, "data/alfred.twl2")

    # Load Prompts
    prompts_path = os.path.join(project_root, "data/alfworld/prompts/alfworld_3prompts.json")
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)

    # Load Batch Tasks
    batch_file = os.path.join(project_root, "alfworld_runs/memory_agent_test/rag_react/valid_unseen_batch1_tasks.json")
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    target_tasks = batch_data['tasks'] # List of strings like "valid_unseen:task/trial..."
    print(f"Loaded {len(target_tasks)} tasks from batch file.")
    
    # Initialize environment class
    print("Initializing environment class...")
    env_type = config['env']['type']
    env_class = envs.get_environment(env_type)
    alfred_env = env_class(config, train_eval='eval_out_of_distribution')
    
    # Filter Game Files
    all_games = alfred_env.game_files
    filtered_games = []
    
    # Create a set of target trial IDs for faster lookup
    # The format in JSON is "valid_unseen:TaskName/TrialID"
    # We can extract the "TaskName/TrialID" part
    target_suffixes = []
    for t in target_tasks:
        if ":" in t:
            target_suffixes.append(t.split(":")[1])
        else:
            target_suffixes.append(t)
            
    print(f"Filtering {len(all_games)} available games against {len(target_suffixes)} targets...")
    
    for game_path in all_games:
        # Check if game_path ends with any of the target suffixes
        # game_path is absolute: .../TaskName/TrialID/game.tw-pddl
        # target is TaskName/TrialID
        # So we check if target in game_path
        for suffix in target_suffixes:
            if suffix in game_path:
                filtered_games.append(game_path)
                break
    
    print(f"Found {len(filtered_games)} matching games to run.")
    
    if not filtered_games:
        print("No matching games found! Check paths.")
        return

    # Override env game files
    alfred_env.game_files = filtered_games
    alfred_env.num_games = len(filtered_games)
    
    # Init Env
    batch_env = alfred_env.init_env(batch_size=1)
    env = SingleEnvWrapper(batch_env)
    
    # Initialize Agent
    agent = RAGReAct(to_print=True)
    
    log_dir = os.path.join(project_root, "alfworld_runs/memory_agent_test/rag_react/logs")
    
    # Run Loop
    print(f"Starting execution of {len(filtered_games)} tasks...")
    
    for _ in tqdm(range(len(filtered_games))):
        obs, info = env.reset()
        
        current_game_file = info.get('extra.gamefile', 'Unknown')
        print(f"\nCurrent task: {current_game_file}")
        
        # Parse Description
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
                task_desc = task_desc[:-1]

        # Determine Task Type and Construct Base Prompt
        game_dir = os.path.dirname(current_game_file)
        game_name = os.path.basename(game_dir)
        
        prompt_key = ""
        if 'pick_and_place' in game_name:
            prompt_key = 'put'
        elif 'pick_clean' in game_name:
            prompt_key = 'clean'
        elif 'pick_heat' in game_name:
            prompt_key = 'heat'
        elif 'pick_cool' in game_name:
            prompt_key = 'cool'
        elif 'look_at_obj' in game_name:
            prompt_key = 'examine'
        elif 'pick_two_obj' in game_name:
            prompt_key = 'puttwo'
        
        base_prompt = "Interact with a household to solve a task." # Fallback
        if prompt_key:
            # Concatenate the 3 few-shot examples
            base_prompt = prompts[f"react_{prompt_key}_0"] + "\n" + prompts[f"react_{prompt_key}_1"] + "\n" + prompts[f"react_{prompt_key}_2"]

        # ID
        task_id = os.path.basename(os.path.dirname(current_game_file)) if '/' in current_game_file else "task_unknown"
        
        # Run
        # We need trial_num? Just use 1 for now or parse from logs if needed uniqueness
        try:
            agent.run(
                env=env,
                base_prompt=base_prompt,
                memory=[],
                start_ob=obs,
                task_desc=task_desc,
                task_id=task_id,
                trial_num=1,
                log_dir=log_dir
            )
        except Exception as e:
            print(f"Error running task {task_id}: {e}")
            # Continue to next task
            continue

    print(f"Batch run completed. Logs saved to {log_dir}")

if __name__ == "__main__":
    main()

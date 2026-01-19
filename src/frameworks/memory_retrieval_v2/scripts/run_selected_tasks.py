
import os
import sys
import yaml
import json
from typing import List

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from src.envs.alfworld_env import AlfworldEnv
from src.frameworks.memory_retrieval_v2.agents.memory_agent import MemoryAgent

def run_selected_tasks():
    # 1. Configuration
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    
    base_config_path = os.path.join(os.path.dirname(__file__), "../../../../data/alfworld/base_config.yaml")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output directory
    # log_dir = os.path.join(os.path.dirname(__file__), "../../alfworld_runs/memory_retrieval_v2/memory_agent_runs/selected_tasks")
    log_dir = os.path.join(base_dir, "alfworld_runs/memory_retrieval_v2/memory_agent_runs/selected_tasks")
    os.makedirs(log_dir, exist_ok=True)
    
    # Memory Bank Path
    memory_bank_path = os.path.join(os.path.dirname(__file__), "../../alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json")

    # 2. Target Tasks (split, task_dir, trial_dir)
    target_tasks = [
        # Easy
        ("valid_unseen", "look_at_obj_in_light-AlarmClock-None-DeskLamp-308", "trial_T20190908_222917_366542"),
    ]

    print(f"Targeting {len(target_tasks)} tasks:")
    for t in target_tasks:
        print(f" - {t}")

    # Set ALFWORLD_DATA
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Memory Bank Path
    memory_bank_path = os.path.join(base_dir, "alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json")

    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
    else:
        print(f"Warning: Prompts file not found at {prompts_path}")
        prompts = {}

    # 3. Initialize Agent
    agent = MemoryAgent(model="gemini-2.5-flash", to_print=True)

    # 4. Run Loop
    tasks_run = 0
    
    for split, task_name, trial_name in target_tasks:
        task_id = f"{split}:{task_name}/{trial_name}"
        
        # Construct path to task in alfworld_mini
        task_path = os.path.join(base_dir, 'data', 'alfworld_mini', split, task_name, trial_name)
        
        if not os.path.exists(task_path):
            print(f"Warning: Task path not found: {task_path}")
            continue
            
        print(f"\n>>> RUNNING TASK: {task_id}")
        
        # Configure env for this specific task
        task_config = config.copy()
        task_config['dataset']['eval_ood_data_path'] = task_path
        task_config['general']['evaluate']['batch_size'] = 1
        
        try:
            env = AlfworldEnv(task_config, split='eval_out_of_distribution')
            obs, info = env.reset()
            
            # Get task description from observation
            task_desc = ""
            if "Your task is to:" in obs:
                # Split by "Your task is to:" and take the last part
                task_desc = obs.split("Your task is to:")[-1].strip()
                # If there are subsequent newlines, take only the first line of the task desc
                if "\n" in task_desc:
                    task_desc = task_desc.split("\n")[0].strip()
            else:
                 # Fallback: take the last line if "Your task is to:" is not found explicitly
                 # or try to find a line that looks like a task.
                 lines = obs.split('\n')
                 if lines:
                     task_desc = lines[-1].strip()

            print(f"Task Desc: {task_desc}")
            
            # Select prompt based on task name (simple heuristic)
            if "cool" in task_name:
                prompt = prompts.get("react_cool_0", "")
            elif "heat" in task_name:
                prompt = prompts.get("react_heat_0", "")
            elif "clean" in task_name:
                prompt = prompts.get("react_clean_0", "")
            elif "pick_two" in task_name:
                prompt = prompts.get("react_puttwo_0", "")
            elif "look_at" in task_name:
                prompt = prompts.get("react_examine_0", "")
            else:
                prompt = prompts.get("react_put_0", "")

            # Run Agent
            agent.run(
                env=env,
                base_prompt=prompt, 
                memory=[],
                start_ob=obs,
                task_id=task_id,
                trial_num=1,
                log_dir=log_dir,
                task_desc=task_desc,
                memory_bank_path=memory_bank_path
            )
            
            tasks_run += 1
            env.close()
            
        except Exception as e:
            print(f"Error running task {task_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nExecution complete. Ran {tasks_run}/{len(target_tasks)} tasks.")

if __name__ == "__main__":
    run_selected_tasks()

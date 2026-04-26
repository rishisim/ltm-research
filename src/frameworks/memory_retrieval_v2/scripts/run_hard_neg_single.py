
import argparse
import os
import sys
import yaml
import json
from typing import List

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from src.envs.alfworld_env import AlfworldEnv
# IMPORT HARD NEGATIVE AGENT
from src.frameworks.memory_retrieval_v2.agents.hard_neg_memory_agent import HardNegMemoryAgent

def _auto_embedding_provider(model: str) -> str:
    """Auto-pair embedding provider: gemini-* → gemini, claude-* → openai."""
    if model.startswith("claude"):
        return "openai"
    return "gemini"

def run_hard_neg_single(model: str = "gemini-2.5-flash", embedding_provider: str = ""):
    # Set embedding provider env var before importing retrieval modules.
    provider = embedding_provider if embedding_provider else _auto_embedding_provider(model)
    os.environ["LTM_EMBEDDING_PROVIDER"] = provider
    print(f"Using model={model}, embedding_provider={provider}")

    # 1. Configuration
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    
    base_config_path = os.path.join(base_dir, "data/alfworld/base_config.yaml")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output directory: hard_neg
    log_dir = os.path.join(base_dir, "alfworld_runs/memory_retrieval_v2/memory_agent_runs/hard_neg")
    os.makedirs(log_dir, exist_ok=True)
    
    # Memory Bank Path
    memory_bank_path = os.path.join(base_dir, "alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json")

    # 2. Target Task
    target_tasks = [
        ("valid_unseen", "look_at_obj_in_light-Mug-None-DeskLamp-308", "trial_T20190908_161733_213242"),
    ]

    print(f"Targeting {len(target_tasks)} tasks for HARD NEGATIVE run:")
    for t in target_tasks:
        print(f" - {t}")

    # Set ALFWORLD_DATA
    os.environ['ALFWORLD_DATA'] = os.path.join(base_dir, 'data')

    # Load Prompts
    prompts_path = os.path.join(base_dir, 'data', 'alfworld', 'prompts', 'alfworld_3prompts.json')
    if os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
    else:
        print(f"Warning: Prompts file not found at {prompts_path}")
        prompts = {}

    # 3. Initialize Agent (HardNegMemoryAgent)
    agent = HardNegMemoryAgent(model=model, to_print=True)

    # 4. Run Loop
    tasks_run = 0
    
    for split, task_name, trial_name in target_tasks:
        task_id = f"{split}:{task_name}/{trial_name}"
        
        # Construct path to task in alfworld_mini
        task_path = os.path.join(base_dir, 'data', 'alfworld_mini', split, task_name, trial_name)
        
        if not os.path.exists(task_path):
            print(f"Warning: Task path not found: {task_path}")
            # Try finding it in the full alfworld data if mini missing? 
            # Assuming user provided provided path exists as per request.
            # Attempt to verify trial dir exists if not full path
            continue
            
        print(f"\n>>> RUNNING HARD NEGATIVE AGENT ON TASK: {task_id}")
        
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
                task_desc = obs.split("Your task is to:")[-1].strip()
                if "\n" in task_desc:
                    task_desc = task_desc.split("\n")[0].strip()
            else:
                 lines = obs.split('\n')
                 if lines:
                     task_desc = lines[-1].strip()

            print(f"Task Desc: {task_desc}")
            
            # Select prompt based on task name heuristic
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
    parser = argparse.ArgumentParser(description="Run a single hard-negative ALFWorld task with HardNegMemoryAgent")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Chat model name (default: gemini-2.5-flash). Use 'claude-haiku-4-5' for Claude.",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="",
        choices=["", "gemini", "openai"],
        help=(
            "Embedding provider (default: auto-pair). "
            "Auto-pair: gemini-* models → gemini, claude-* models → openai."
        ),
    )
    args = parser.parse_args()
    run_hard_neg_single(model=args.model, embedding_provider=args.embedding_provider)

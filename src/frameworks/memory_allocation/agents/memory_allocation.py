import sys
import os
import json
from typing import List, Tuple, Any, Dict
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct


class MemoryAllocationReflexion(ReAct):
    """
    A framework that logs complete trajectories and generates reflexions for failed tasks.
    """
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        # Load few-shot examples for reflexion
        self.few_shot_examples = ""
        few_shot_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'alfworld', 'reflexion_few_shot_examples.txt')
        try:
            with open(few_shot_path, 'r') as f:
                self.few_shot_examples = f.read()
        except FileNotFoundError:
            print(f"Warning: Few-shot examples not found at {few_shot_path}")

    def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "", 
            task_id: str = "", trial_num: int = 1, log_dir: str = "", task_desc: str = "") -> Tuple[EnvironmentHistory, bool]:
        """
        Run the agent on the environment and log the trajectory.
        
        Args:
            env: The environment to run on
            base_prompt: The base prompt for the agent
            memory: List of previous reflexions/memories
            start_ob: Starting observation
            task_id: Unique identifier for the task
            trial_num: Trial number for this task
            log_dir: Directory to save logs
            task_desc: The task description text
            
        Returns:
            Tuple of (environment history, success boolean)
        """
        # Collect steps as (action, observation) pairs
        steps: List[Dict[str, str]] = []
        
        env_history = EnvironmentHistory(base_prompt, start_ob, memory[-3:] if len(memory) > 3 else memory)
        
        if self.to_print:
            print(start_ob)
            sys.stdout.flush()

        cur_step = 0
        reward = 0

        while cur_step < 49:
            # Choose action
            action = self._llm(str(env_history) + "Action:", stop=['\n']).strip()
            
            # Clean up action
            if action.startswith('Action:'):
                action = action[7:].strip()
            if action.startswith('>'):
                action = action[1:].strip()

            env_history.add("action", action)
            
            observation, reward, done, info = env.step(action)
            
            if action.startswith('think:'):
                observation = 'OK.'

            env_history.add("observation", observation)
            
            # Record the step
            steps.append({
                "step": cur_step + 1,
                "action": action,
                "observation": observation
            })
            
            if self.to_print:
                print(f'Action: {action}\nObs: {observation}')
                sys.stdout.flush()
            
            if done:
                break
            elif env_history.check_is_exhausted():
                break
            
            cur_step += 1

        # Determine success
        is_success = reward > 0
        
        if self.to_print:
            print(f"\nTask {'SUCCESS' if is_success else 'FAILURE'}")
        
        # Log trajectory and reflexion
        if log_dir:
            self._log_trajectory(log_dir, task_id, trial_num, steps, is_success, task_desc)
            
            # Generate reflexion if task failed, otherwise use empty string
            reflexion = ""
            if not is_success:
                reflexion = self._generate_reflexion(str(env_history), memory)
            
            # Log reflexion for all trials (success or failure)
            self._log_reflexion(log_dir, task_id, trial_num, reflexion, is_success)
            
            if self.to_print:
                print(f"\nReflexion: {reflexion}")

        return env_history, is_success

    def _log_trajectory(self, log_dir: str, task_id: str, trial_num: int, 
                        steps: List[Dict[str, str]], success: bool, task_desc: str = "") -> None:
        """Log the complete trajectory to trajectories.json"""
        # Extract task_type from task_id (e.g., 'pick_and_place_simple' from 'pick_and_place_simple-Mug-None-Desk-308/...')
        task_type = task_id.split('-')[0] if task_id else ""
        
        trajectory = {
            "task_id": task_id,
            "task_type": task_type,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "steps": steps,
            "success": success
        }
        
        trajectories_path = os.path.join(log_dir, "trajectories.json")
        
        # Read existing trajectories, append new one, write back
        trajectories = []
        if os.path.exists(trajectories_path):
            with open(trajectories_path, 'r') as f:
                trajectories = json.load(f)
        
        trajectories.append(trajectory)
        
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=2)

    def _log_reflexion(self, log_dir: str, task_id: str, trial_num: int, 
                       reflexion: str, is_success: bool) -> None:
        """Log the reflexion to reflexions.json"""
        reflexion_entry = {
            "task_id": task_id,
            "trial_num": trial_num,
            "is_success": is_success,
            "reflexion": reflexion
        }
        
        reflexions_path = os.path.join(log_dir, "reflexions.json")
        
        # Read existing reflexions, append new one, write back
        reflexions = []
        if os.path.exists(reflexions_path):
            with open(reflexions_path, 'r') as f:
                reflexions = json.load(f)
        
        reflexions.append(reflexion_entry)
        
        with open(reflexions_path, 'w') as f:
            json.dump(reflexions, f, indent=2)

    def _generate_reflexion(self, history_str: str, memory: List[str]) -> str:
        """
        Generate a reflexion for a failed task using the original Reflexion prompt.
        This is the exact prompt from src/frameworks/reflexion.py
        """
        scenario = history_str.split("Here is the task:")[-1].strip()
        
        query = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{self.few_shot_examples}

{scenario}"""

        if memory:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(memory):
                query += f'Trial #{i}: {m}\n'

        query += '\n\nNew plan:'
        return get_chat(query, model=self.model)

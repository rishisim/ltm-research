"""
Trajectory Context Agent - Ablation using raw trajectory text instead of knowledge base

This agent is an ablation of the ContextOnlyAgent that:
1. Does NOT use the knowledge base (processed issue + learning pairs)
2. Instead, retrieves the most similar training trajectory (by task_desc similarity)
3. Provides raw trajectory text cut at a word count limit

This tests whether the knowledge base's structured format provides value over raw trajectories.
"""

import sys
import os
import json
from typing import List, Tuple, Any, Dict, Optional

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

# Import trajectory-based retrieval module
from src.frameworks.memory_allocation.retrieval.trajectory_context_retrieval import (
    retrieve_trajectory_context
)


class TrajectoryContextAgent(ReAct):
    """
    A trajectory-based context agent that:
    1. Retrieves the most similar training trajectory at task start
    2. Provides raw trajectory text (cut at word limit) as context
    3. Does NOT provide a help tool during execution
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.training_trajectories_path: Optional[str] = None

    def run(
        self, 
        env: BaseEnv, 
        base_prompt: str, 
        memory: List[str], 
        start_ob: str = "",
        task_id: str = "",
        trial_num: int = 1,
        log_dir: str = "",
        task_desc: str = "",
        training_trajectories_path: str = "",
        word_count_limit: int = 135,  # Default based on analysis (avg context length)
        trajectory_file: str = "trajectories.json"
    ) -> Tuple[EnvironmentHistory, bool]:
        """
        Run the trajectory context agent on the environment.
        
        Args:
            env: The environment to run on
            base_prompt: The base prompt for the agent
            memory: List of previous reflexions/memories
            start_ob: Starting observation
            task_id: Unique identifier for the task
            trial_num: Trial number for this task
            log_dir: Directory to save logs
            task_desc: The task description text
            training_trajectories_path: Path to the training trajectories.json file
            word_count_limit: Maximum word count for trajectory context (default: 135)
            trajectory_file: Name of output trajectory file
            
        Returns:
            Tuple of (environment history, success boolean)
        """
        self.training_trajectories_path = training_trajectories_path
        
        # Collect steps as (action, observation) pairs
        steps: List[Dict[str, Any]] = []
        retrieval_metadata: Dict[str, Any] = {}  # Store retrieval info for logging
        
        # Step 1: Retrieve trajectory context from training trajectories
        trajectory_context = ""
        if training_trajectories_path and task_desc:
            try:
                trajectory_context, retrieval_metadata = retrieve_trajectory_context(
                    new_task_desc=task_desc,
                    trajectories_path=training_trajectories_path,
                    word_count_limit=word_count_limit
                )
                if self.to_print and trajectory_context:
                    print("\n" + "="*60)
                    print("CONTEXT FROM SIMILAR TRAJECTORY:")
                    print("="*60)
                    print(f"Source: {retrieval_metadata.get('source_task_id', 'Unknown')}")
                    print(f"Similarity: {retrieval_metadata.get('similarity_score', 0):.4f}")
                    print("-"*60)
                    print(trajectory_context)
                    print("="*60 + "\n")
            except Exception as e:
                if self.to_print:
                    print(f"Warning: Could not retrieve trajectory context: {e}")
        
        # Step 2: Build enhanced prompt with trajectory context
        enhanced_prompt = base_prompt
        
        # Add trajectory context if available
        if trajectory_context:
            enhanced_prompt = f"""[CONTEXT FROM A SIMILAR TASK]
Below is a partial trajectory from a similar task. Use this as a reference for how to approach your current task:

{trajectory_context}

[END CONTEXT]

{base_prompt}"""
        
        # Initialize environment history with enhanced prompt
        env_history = EnvironmentHistory(
            enhanced_prompt, 
            start_ob, 
            memory[-3:] if len(memory) > 3 else memory
        )
        
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
            
            # Normal environment interaction (no help tool)
            observation, reward, done, info = env.step(action)
            
            if action.startswith('think:'):
                observation = 'OK.'

            env_history.add("observation", observation)
            
            # Record the step
            steps.append({
                "step": cur_step + 1,
                "action": action,
                "observation": observation,
                "is_help_call": False
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
        
        # Log trajectory
        if log_dir:
            self._log_trajectory(
                log_dir, task_id, trial_num, steps, 
                is_success, task_desc, retrieval_metadata, 
                trajectory_context, trajectory_file
            )

        return env_history, is_success
    
    def _log_trajectory(
        self, 
        log_dir: str, 
        task_id: str, 
        trial_num: int,
        steps: List[Dict[str, Any]], 
        success: bool, 
        task_desc: str = "",
        retrieval_metadata: Dict[str, Any] = None,
        context_text: str = "",
        trajectory_file: str = "trajectories.json"
    ) -> None:
        """Log the complete trajectory to trajectories.json"""
        # Extract task_type from task_id
        task_type = task_id.split('-')[0] if task_id else ""
        
        trajectory = {
            "task_id": task_id,
            "task_type": task_type,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "context_source": "trajectory",  # Indicate this is trajectory-based
            "retrieval_metadata": retrieval_metadata or {},
            "context_sent_to_agent": context_text,  # The actual raw trajectory text sent to LLM
            "steps": steps,
            "success": success,
            "help_calls": [],  # Always empty for this agent
            "help_call_count": 0
        }
        
        trajectories_path = os.path.join(log_dir, trajectory_file)
        
        # Read existing trajectories, append new one, write back
        trajectories = []
        if os.path.exists(trajectories_path):
            with open(trajectories_path, 'r') as f:
                trajectories = json.load(f)
        
        trajectories.append(trajectory)
        
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=2)

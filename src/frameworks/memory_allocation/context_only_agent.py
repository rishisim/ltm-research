"""
Context-Only Agent - Memory agent with context retrieval only (no help tool)

This agent is an ablation of the full MemoryAgent that:
1. Context retrieval at task start - Uses context_retrieval to provide relevant 
   learnings from previous similar tasks
2. NO help tool during execution - Agent cannot call help["query"]

This is used for ablation studies to measure the impact of context retrieval alone.
"""

import sys
import os
import json
from typing import List, Tuple, Any, Dict, Optional

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

# Import retrieval modules (only context retrieval, no tool retrieval)
from src.frameworks.memory_allocation.context_retrieval import (
    retrieve_learnings_only,
    format_learnings_for_prompt
)


class ContextOnlyAgent(ReAct):
    """
    A context-retrieval-only agent that:
    1. Retrieves relevant context at task start using context_retrieval
    2. Does NOT provide a help tool during execution (ablation)
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.memory_bank_path: Optional[str] = None

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
        memory_bank_path: str = "",
        trajectory_file: str = "trajectories_valid_unseen.json"
    ) -> Tuple[EnvironmentHistory, bool]:
        """
        Run the context-only agent on the environment.
        
        Args:
            env: The environment to run on
            base_prompt: The base prompt for the agent
            memory: List of previous reflexions/memories
            start_ob: Starting observation
            task_id: Unique identifier for the task
            trial_num: Trial number for this task
            log_dir: Directory to save logs
            task_desc: The task description text
            memory_bank_path: Path to the knowledge_base.json file
            
        Returns:
            Tuple of (environment history, success boolean)
        """
        self.memory_bank_path = memory_bank_path
        
        # Collect steps as (action, observation) pairs
        steps: List[Dict[str, Any]] = []
        retrieved_learnings: List[Dict[str, str]] = []  # Store raw learnings for logging
        
        # Step 1: Retrieve context from memory bank at task start
        context_learnings = ""
        if memory_bank_path and task_desc:
            try:
                learnings = retrieve_learnings_only(
                    new_task_desc=task_desc,
                    memory_bank_path=memory_bank_path,
                    top_k_similar=20,
                    top_per_phase=2
                )
                if learnings:
                    retrieved_learnings = learnings  # Store for trajectory logging
                    context_learnings = format_learnings_for_prompt(learnings)
                    if self.to_print:
                        print("\n" + "="*60)
                        print("CONTEXT FROM PREVIOUS EXPERIENCES:")
                        print("="*60)
                        print(context_learnings)
                        print("="*60 + "\n")
            except Exception as e:
                if self.to_print:
                    print(f"Warning: Could not retrieve context: {e}")
        
        # Step 2: Build enhanced prompt with context (NO help instructions - ablation)
        enhanced_prompt = base_prompt
        
        # Add context learnings if available
        if context_learnings:
            enhanced_prompt = f"""[CONTEXT FROM PREVIOUS SIMILAR TASKS]
The following learnings are from previous tasks similar to yours. Use them to avoid common mistakes:

{context_learnings}

[END CONTEXT]

{base_prompt}"""
        
        # NO help tool instructions added (this is the ablation difference)
        
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
            
            # NO help action parsing (this is the ablation difference)
            
            # Normal environment interaction
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
                is_success, task_desc, retrieved_learnings, trajectory_file
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
        context_learnings: List[Dict[str, str]] = None,
        trajectory_file: str = "trajectories_valid_unseen.json"
    ) -> None:
        """Log the complete trajectory to trajectories.json"""
        # Extract task_type from task_id (e.g., 'pick_and_place_simple' from 'pick_and_place_simple-Mug-None-Desk-308/...')
        task_type = task_id.split('-')[0] if task_id else ""
        
        trajectory = {
            "task_id": task_id,
            "task_type": task_type,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "context_from_retrieval": context_learnings or [],
            "steps": steps,
            "success": success,
            "help_calls": [],  # Always empty for context-only agent
            "help_call_count": 0  # Always zero for context-only agent
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

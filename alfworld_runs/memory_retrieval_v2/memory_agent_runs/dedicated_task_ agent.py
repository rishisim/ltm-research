"""
Dedicated Task Agent - Hard-coded learnings for specific task

This agent is configured for a single specific task:
  Task: pick_heat_then_place_in_recep-Plate-None-CounterTop-1

It uses hard-coded learnings extracted from previous experiences to guide the agent,
without any dynamic retrieval or help tool during execution.

Useful for testing whether pre-computed context helps on a specific task.
"""

import sys
import os
import json
from typing import List, Tuple, Any, Dict, Optional

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct


class DedicatedTaskAgent(ReAct):
    """
    A hard-coded context agent for a specific task:
    - Uses pre-computed learnings at task start (no dynamic retrieval)
    - Does NOT provide a help tool during execution
    - Configured for: pick_heat_then_place_in_recep-Plate-None-CounterTop-1
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.memory_bank_path: Optional[str] = None
    
    def _get_hardcoded_learnings(self) -> List[Dict[str, str]]:
        """
        Returns hard-coded learnings for the pick_heat_then_place task.
        These are extracted from previous similar successful trials.
        """
        return [
            {
                "issue_text": "Heated apple became un-interactable after successful heating in microwave.",
                "learning_text": "No successful action or strategy was found to resolve this issue."
            },
            {
                "issue_text": "Agent failed to heat cup using the stoveburner.",
                "learning_text": "Investigate alternative heating methods if stoveburner fails to heat the item."
            },
            {
                "issue_text": "Cannot remove blocking item (pan) from stoveburner.",
                "learning_text": "Some items on stoveburners are irremovable, blocking their use for heating."
            },
            {
                "issue_text": "Attempted to heat mug directly on stoveburner without a container.",
                "learning_text": "Heating on a stoveburner requires placing the item within a suitable container like a pan."
            },
            {
                "issue_text": "Cannot take pan from countertop, it is irremovable.",
                "learning_text": "Some objects on countertops are irremovable and cannot be acquired."
            },
            {
                "issue_text": "Cannot pick up a new mug while already holding an item.",
                "learning_text": "Release the currently held item before attempting to pick up a new one."
            },
            {
                "issue_text": "Unable to put down or drop the currently held mug on any surface.",
                "learning_text": "Verify environmental conditions or alternative drop points if an item cannot be released."
            },
            {
                "issue_text": "Unable to place a plate on a stoveburner that is already occupied.",
                "learning_text": "Find an empty stoveburner or clear an occupied one to place the plate."
            },
            {
                "issue_text": "Unable to place a plate inside an open microwave.",
                "learning_text": "Plates cannot be placed inside the microwave; they must be heated while held."
            },
            {
                "issue_text": "After heating a plate while holding it, the plate becomes unplaceable.",
                "learning_text": "Avoid heating plates while holding them, as it leads to an unrecoverable state where they cannot be placed."
            },
            {
                "issue_text": "Unable to pick up new items while holding an unplaceable heated plate.",
                "learning_text": "Ensure the currently held item can be placed before attempting to acquire new items."
            },
            {
                "issue_text": "Unable to take items from occupied stoveburners while holding another item.",
                "learning_text": "Place down the currently held item before attempting to clear an occupied stoveburner."
            },
            {
                "issue_text": "Unable to close the microwave door while holding an item.",
                "learning_text": "Place down the held item before attempting to close the microwave."
            },
            {
                "issue_text": "Heating a plate with a stoveburner while holding it fails.",
                "learning_text": "Place the plate on an empty stoveburner before heating it."
            },
            {
                "issue_text": "Failed to activate stoveburner, suggesting it is not an activatable appliance for plates.",
                "learning_text": "The stoveburner is not an activatable appliance for heating plates; do not attempt to use it."
            }
        ]
    
    def _format_hardcoded_context(self, learnings: List[Dict[str, str]]) -> str:
        """
        Format hard-coded learnings into a prompt section.
        Matches the format from context_retrieval.format_learnings_for_prompt()
        """
        if not learnings:
            return ""
        
        formatted_lines = []
        for i, learning in enumerate(learnings, 1):
            formatted_lines.append(
                f"\n{i}. Issue: {learning['issue_text']}"
                f"\n   Learning: {learning['learning_text']}"
            )
        
        return "".join(formatted_lines)

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
        
        # Step 1: Get hard-coded learnings (no dynamic retrieval)
        learnings = self._get_hardcoded_learnings()
        retrieved_learnings = learnings  # Store for trajectory logging
        context_learnings = self._format_hardcoded_context(learnings)
        
        if self.to_print:
            print("\n" + "="*60)
            print("CONTEXT FROM PREVIOUS SIMILAR TASKS:")
            print("="*60)
            print(context_learnings)
            print("="*60 + "\n")
            sys.stdout.flush()
        
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

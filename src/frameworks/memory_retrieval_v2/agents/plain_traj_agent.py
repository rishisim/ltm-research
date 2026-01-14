"""
Plain Trajectory Agent - Uses raw trajectory examples as context

This agent extends the base ReAct agent with:
1. Trajectory retrieval at task start - Retrieves top-K similar trajectories
   from training data and provides truncated trajectory text as context
2. No help tool - This is a context-only ablation variant

The goal is to test whether raw trajectory examples (vs. extracted learnings)
provide useful context for task execution.
"""

import sys
import os
import json
from typing import List, Tuple, Any, Dict, Optional

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

# Import plain trajectory retrieval module
from src.frameworks.memory_retrieval_v2.retrieval.core.plain_traj_retrieval import (
    retrieve_plain_trajectories,
    format_context_for_prompt
)


class PlainTrajAgent(ReAct):
    """
    A trajectory-based context agent that:
    1. Retrieves top-K most similar training trajectories at task start
    2. Provides truncated raw trajectory text as context (130 words each)
    3. Does NOT provide a help tool during execution (context-only)
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.trajectories_path: Optional[str] = None

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
        trajectories_path: str = "",
        top_k: int = 1,
        word_limit: int = 130,
        trajectory_file: str = "trajectories.json"
    ) -> Tuple[EnvironmentHistory, bool]:
        """
        Run the plain trajectory agent on the environment.
        
        Args:
            env: The environment to run on
            base_prompt: The base prompt for the agent
            memory: List of previous reflexions/memories
            start_ob: Starting observation
            task_id: Unique identifier for the task
            trial_num: Trial number for this task
            log_dir: Directory to save logs
            task_desc: The task description text
            trajectories_path: Path to the training trajectories.json file
            top_k: Number of top similar trajectories to retrieve
            word_limit: Maximum word count for each trajectory (default: 130)
            trajectory_file: Name of output trajectory file
            
        Returns:
            Tuple of (environment history, success boolean)
        """
        self.trajectories_path = trajectories_path
        
        # Collect steps as (action, observation) pairs
        steps: List[Dict[str, Any]] = []
        retrieval_metadata: List[Dict[str, Any]] = []  # Store retrieval info for logging
        
        # Step 1: Retrieve trajectory context from training trajectories
        trajectory_context = ""
        if trajectories_path and task_desc:
            try:
                raw_context, retrieval_metadata = retrieve_plain_trajectories(
                    task_desc=task_desc,
                    trajectories_path=trajectories_path,
                    top_k=top_k,
                    word_limit=word_limit
                )
                
                if raw_context:
                    trajectory_context = format_context_for_prompt(raw_context)
                    
                    if self.to_print:
                        print("\n" + "="*60)
                        print("CONTEXT FROM SIMILAR TRAJECTORIES:")
                        print("="*60)
                        for meta in retrieval_metadata:
                            print(f"Example {meta['rank']}: {meta['source_task_id']} (similarity: {meta['similarity_score']:.4f})")
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
            enhanced_prompt = f"""{trajectory_context}

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
            
            # Execute action in environment
            observation, reward, done, info = env.step(action)
            
            if self.to_print:
                print(f'Action: {action}')
                print(f'Observation: {observation}')
                sys.stdout.flush()
            
            # Record the step
            steps.append({
                "step": cur_step + 1,
                "action": action,
                "observation": observation
            })
            
            env_history.add("observation", observation)
            
            cur_step += 1
            
            # Check if done
            if done:
                break
            
            # Check for exhaustion (repeated actions)
            if env_history.check_is_exhausted():
                if self.to_print:
                    print("Agent is exhausted (repeated actions). Stopping.")
                break

        # Step 3: Save trajectory data with retrieval metadata
        if log_dir and task_id:
            os.makedirs(log_dir, exist_ok=True)
            traj_data_path = os.path.join(log_dir, f"traj_data.json")
            
            # Count words in context
            context_word_count = len(trajectory_context.split()) if trajectory_context else 0
            
            traj_data = {
                "task_id": task_id,
                "trial_num": trial_num,
                "task_desc": task_desc,
                "steps": steps,
                "success": reward > 0,
                "num_steps": len(steps),
                "retrieved_trajectories": retrieval_metadata,
                "retrieval_context_word_count": context_word_count,
                "top_k_used": top_k,
                "word_limit_per_traj": word_limit
            }
            
            with open(traj_data_path, 'w') as f:
                json.dump(traj_data, f, indent=2)
        
        # Return whether the task was successful
        return env_history, reward > 0

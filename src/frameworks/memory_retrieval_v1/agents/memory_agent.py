"""
Memory Agent - A memory-augmented agent for ALFWorld

This agent extends the base ReAct agent with:
1. Context retrieval at task start - Uses context_retrieval to provide relevant 
   learnings from previous similar tasks
2. Help tool during execution - Allows the agent to call help["query"] at any 
   point to retrieve targeted learnings via tool_retrieval
"""

import sys
import os
import re
import json
from typing import List, Tuple, Any, Dict, Optional

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

# Import retrieval modules
# Import retrieval modules
from src.frameworks.memory_allocation.retrieval.core.context_retrieval import (
    retrieve_learnings_only,
    format_learnings_for_prompt
)
from src.frameworks.memory_allocation.retrieval.core.tool_retrieval import (
    help_tool,
    format_help_response
)


# Regex pattern to match help["..."] or help['...']
HELP_PATTERN = re.compile(r'help\s*\[\s*["\'](.+?)["\']\s*\]', re.IGNORECASE)


class MemoryAgent(ReAct):
    """
    A memory-augmented agent that:
    1. Retrieves relevant context at task start using context_retrieval
    2. Provides a help["query"] tool that agents can call during execution
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.memory_bank_path: Optional[str] = None
        
    def _get_help_instructions(self) -> str:
        """
        Returns instructions for the agent on how to use the help tool.
        """
        return """
IMPORTANT: You have access to a help tool when you're struggling or need guidance.
To use it, output an action in this format:
help["your issue here"]

CRITICAL: Your query must be SHORT - maximum ONE sentence. Be concise.

Examples of good queries:
help["cannot find the mug"]
help["how to heat something in microwave"]
help["stuck after opening drawer"]

Do NOT write long queries like "I have been searching for the mug for a long time and checked many locations but still cannot find it" - instead write: help["cannot find the mug"]

Use this tool when you:
- Can't find an object after searching
- Are unsure about the next step
- Keep encountering the same error
"""

    def _parse_help_action(self, action: str) -> Optional[str]:
        """
        Parse an action to check if it's a help call.
        
        Args:
            action: The action string from the agent
            
        Returns:
            The help query if action is a help call, None otherwise
        """
        match = HELP_PATTERN.search(action)
        if match:
            return match.group(1)
        return None
    
    def _execute_help_tool(self, query: str) -> str:
        """
        Execute the help tool and return formatted response.
        
        Args:
            query: The help query from the agent
            
        Returns:
            Formatted help response string
        """
        if not self.memory_bank_path:
            return "Error: Memory bank path not configured. Cannot provide help."
        
        try:
            result = help_tool(
                issue=query,
                memory_bank_path=self.memory_bank_path,
                top_k=3
            )
            return format_help_response(result)
        except Exception as e:
            return f"Error retrieving help: {str(e)}"

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
        memory_bank_path: str = ""
    ) -> Tuple[EnvironmentHistory, bool]:
        """
        Run the memory-augmented agent on the environment.
        
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
        help_calls: List[Dict[str, str]] = []
        retrieved_learnings: List[Dict[str, str]] = []  # Store raw learnings for logging
        
        # Step 1: Retrieve context from memory bank at task start
        context_learnings = ""
        if memory_bank_path and task_desc:
            try:
                learnings = retrieve_learnings_only(
                    new_task_desc=task_desc,
                    memory_bank_path=memory_bank_path,
                    top_k_similar_tasks=5,
                    log_dir=log_dir,
                    task_id=task_id
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
        
        # Step 2: Build enhanced prompt with context and help instructions
        enhanced_prompt = base_prompt
        
        # Add context learnings if available
        if context_learnings:
            enhanced_prompt = f"""[CONTEXT FROM PREVIOUS SIMILAR TASKS]
The following learnings are from previous tasks similar to yours. Use them to avoid common mistakes:

{context_learnings}

[END CONTEXT]

{base_prompt}"""
        
        # Add help tool instructions
        help_instructions = self._get_help_instructions()
        enhanced_prompt = f"{enhanced_prompt}\n\n{help_instructions}"
        
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
            
            # Check if this is a help action
            help_query = self._parse_help_action(action)
            
            if help_query:
                # Execute help tool
                observation = self._execute_help_tool(help_query)
                
                # Log the help call
                help_calls.append({
                    "step": cur_step + 1,
                    "query": help_query,
                    "response": observation
                })
                
                if self.to_print:
                    print(f'Action: {action}')
                    print(f'Help Response:\n{observation}')
                    sys.stdout.flush()
                
                # Record the step with help flag
                steps.append({
                    "step": cur_step + 1,
                    "action": action,
                    "observation": observation,
                    "is_help_call": True
                })
                
                env_history.add("observation", observation)
                
                # Don't interact with environment for help calls
                cur_step += 1
                continue
            
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
            if help_calls:
                print(f"Help tool was called {len(help_calls)} time(s) during this task.")
        
        # Log trajectory
        if log_dir:
            self._log_trajectory(
                log_dir, task_id, trial_num, steps, 
                is_success, task_desc, help_calls, retrieved_learnings
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
        help_calls: List[Dict[str, str]] = None,
        context_learnings: List[Dict[str, str]] = None
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
            "help_calls": help_calls or [],
            "help_call_count": len(help_calls) if help_calls else 0,
            "step_num": len(steps)
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

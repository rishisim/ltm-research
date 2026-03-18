"""
Hard Negative Memory Agent

This agent uses the Hard Negative (least similar) retrieval variants for both
Context Retrieval (at start) and Tool Retrieval (help tool).

It serves as a baseline to demonstrate that retrieval quality matters.
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

# Import HARD NEGATIVE retrieval modules
from src.frameworks.memory_retrieval_v2.retrieval.variants.hard_neg_context_retrieval import (
    retrieve_learnings_only,
    format_learnings_for_prompt
)
from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import (
    get_active_embedding_model,
)
from src.frameworks.memory_retrieval_v2.retrieval.variants.hard_neg_tool_retrieval import (
    help_tool,
    format_help_response
)


# Regex pattern to match help["..."] or help['...']
HELP_PATTERN = re.compile(r'help\s*\[\s*["\'](.+?)["\']\s*\]', re.IGNORECASE)


class HardNegMemoryAgent(ReAct):
    """
    A memory-augmented agent that uses HARD NEGATIVE retrieval logic.
    
    1. Retrieves IRRELEVANT context at task start
    2. Provides IRRELEVANT help advice when queried
    """
    
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        self.memory_bank_path: Optional[str] = None
        
    def _get_help_instructions(self) -> str:
        """
        Returns instructions for the agent on how to use the help tool.
        (Instructions remain the same, only the retrieval quality changes invisibly to the agent)
        """
        return """
IMPORTANT: You have access to a help tool when you're struggling or need guidance.
To use it, output an action in this format:
help["your issue here"]

CRITICAL: Your query must match the style of issues in your memory bank to get the best results.
Query Style Guidelines:
1. Describe the FAILURE or OBSTACLE, not just what you want to do.
2. Mention the ACTION that failed (e.g. "put command failed").
3. Mention MISSING PRECONDITIONS (e.g. "object not found", "container closed").
4. KEY: Do not include numbers or specific identifiers (e.g. "sidetable 1", "mug 2") - instead use general objects (e.g. "sidetable", "mug").

Examples of GOOD queries:
help["cannot find the mug on the table"]
help["put command failed when placing egg on sidetable"]
help["container is closed and cannot put object inside"]
help["repeatedly failing to go to the fridge"]

Examples of BAD queries:
help["how to put mug"] (Too vague)
help["cannot find mug 1"] (Contains ID '1')
help["what do i do next"] (Not specific to an issue)

Use this tool when you:
- Can't find an object after searching
- Are unsure about the next step
- Keep encountering the same error
"""

    def _parse_help_action(self, action: str) -> Optional[str]:
        match = HELP_PATTERN.search(action)
        if match:
            return match.group(1)
        return None
    
    def _execute_help_tool(self, query: str) -> str:
        if not self.memory_bank_path:
            return "Error: Memory bank path not configured. Cannot provide help."
        
        try:
            # Calls the Hard Negative help_tool
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
        self.memory_bank_path = memory_bank_path
        
        steps: List[Dict[str, Any]] = []
        help_calls: List[Dict[str, str]] = []
        retrieved_learnings: List[Dict[str, str]] = []
        context_retrieval_error = ""
        context_retrieval_status = "empty"
        embedding_model_used = get_active_embedding_model()
        
        # Step 1: Retrieve context (Hard Negative)
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
                    retrieved_learnings = learnings
                    context_learnings = format_learnings_for_prompt(learnings)
                    context_retrieval_status = "ok"
                    if self.to_print:
                        print("\n" + "="*60)
                        print("CONTEXT FROM PREVIOUS EXPERIENCES (Hard Negative):")
                        print("="*60)
                        print(context_learnings)
                        print("="*60 + "\n")
            except Exception as e:
                context_retrieval_error = str(e)
                context_retrieval_status = "error"
                if self.to_print:
                    print(f"Warning: Could not retrieve context: {e}")

        if log_dir:
            world_log_path = os.path.join(log_dir, "world.log")
            try:
                with open(world_log_path, "a") as wf:
                    wf.write(
                        f"Context retrieval [{task_id or 'unknown_task'}]: {context_retrieval_status}\n"
                    )
                    if context_retrieval_error:
                        wf.write(f"Context retrieval error: {context_retrieval_error}\n")
            except Exception:
                pass
        
        # Step 2: Build enhanced prompt
        enhanced_prompt = base_prompt
        
        if context_learnings:
            enhanced_prompt = f"""[CONTEXT FROM PREVIOUS SIMILAR TASKS]
The following learnings are from previous tasks similar to yours. Use them to avoid common mistakes:

{context_learnings}

[END CONTEXT]

{base_prompt}"""
        
        help_instructions = self._get_help_instructions()
        enhanced_prompt = f"{enhanced_prompt}\n\n{help_instructions}"
        
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
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        while cur_step < 49:
            action_text, usage = self._llm(str(env_history) + "Action:", stop=['\n'])
            action = action_text.strip()
            
            step_input_tokens = usage.get("input_tokens", 0)
            step_output_tokens = usage.get("output_tokens", 0)
            step_total_tokens = usage.get("total_tokens", 0)
            
            total_input_tokens += step_input_tokens
            total_output_tokens += step_output_tokens
            total_tokens += step_total_tokens
            
            if action.startswith('Action:'):
                action = action[7:].strip()
            if action.startswith('>'):
                action = action[1:].strip()

            env_history.add("action", action)
            
            help_query = self._parse_help_action(action)
            
            if help_query:
                observation = self._execute_help_tool(help_query)
                
                help_calls.append({
                    "step": cur_step + 1,
                    "query": help_query,
                    "response": observation
                })
                
                if self.to_print:
                    print(f'Action: {action}')
                    print(f'Help Response:\n{observation}')
                    sys.stdout.flush()
                
                steps.append({
                    "step": cur_step + 1,
                    "action": action,
                    "observation": observation,
                    "is_help_call": True,
                    "token_usage": {
                        "input_tokens": step_input_tokens,
                        "output_tokens": step_output_tokens,
                        "total_tokens": step_total_tokens
                    }
                })
                
                env_history.add("observation", observation)
                cur_step += 1
                continue
            
            observation, reward, done, info = env.step(action)
            
            if action.startswith('think:'):
                observation = 'OK.'

            env_history.add("observation", observation)
            
            steps.append({
                "step": cur_step + 1,
                "action": action,
                "observation": observation,
                "is_help_call": False,
                "token_usage": {
                    "input_tokens": step_input_tokens,
                    "output_tokens": step_output_tokens,
                    "total_tokens": step_total_tokens
                }
            })
            
            if self.to_print:
                print(f'Action: {action}\nObs: {observation}')
                sys.stdout.flush()
            
            if done:
                break
            elif env_history.check_is_exhausted():
                break
            
            cur_step += 1

        is_success = reward > 0
        
        if self.to_print:
            print(f"\nTask {'SUCCESS' if is_success else 'FAILURE'}")
            if help_calls:
                print(f"Help tool was called {len(help_calls)} time(s) during this task.")
        
        if log_dir:
            self._log_trajectory(
                log_dir, task_id, trial_num, steps, 
                is_success, task_desc, help_calls, retrieved_learnings,
                total_input_tokens, total_output_tokens, total_tokens,
                context_retrieval_error=context_retrieval_error,
                embedding_model_used=embedding_model_used,
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
        context_learnings: List[Dict[str, str]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        context_retrieval_error: str = "",
        embedding_model_used: str = "",
    ) -> None:
        """Log the complete trajectory to trajectories.json"""
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
            "step_num": len(steps),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "context_retrieval_error": context_retrieval_error,
            "embedding_model_used": embedding_model_used,
            "agent_type": "hard_negative"
        }
        
        # Write to append-only JSONL to avoid race conditions with the suite runner
        import json as _json
        jsonl_path = os.path.join(log_dir, "agent_trajectories.jsonl")
        line = _json.dumps(trajectory, separators=(",", ":")) + "\n"
        with open(jsonl_path, "a") as f:
            f.write(line)

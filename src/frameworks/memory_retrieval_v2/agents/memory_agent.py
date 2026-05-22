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
from src.frameworks.react import (
    ReAct,
    action_stop_sequences,
    clean_action_text,
    should_auto_submit_repeated_sql,
)

# Import retrieval modules
from src.frameworks.memory_retrieval_v2.retrieval.core.context_retrieval import (
    retrieve_learnings_only,
    format_learnings_for_prompt
)
from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import (
    get_active_embedding_model,
)
from src.frameworks.memory_retrieval_v2.retrieval.core.tool_retrieval import (
    help_tool,
    format_help_response
)


# Regex pattern to match help["..."] or help['...']
HELP_PATTERN = re.compile(r'help\s*\[\s*["\'](.+?)["\']\s*\]', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Per-environment help-tool instruction strings
# ---------------------------------------------------------------------------

_HELP_INSTRUCTIONS_ALFWORLD = """
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

_HELP_INSTRUCTIONS_WEBSHOP = """
IMPORTANT: You have access to a help tool when you're struggling or need guidance.
To use it, output an action in this format:
help["your issue here"]

CRITICAL: Your query must match the style of issues in your memory bank to get the best results.
Query Style Guidelines:
1. Describe the FAILURE or OBSTACLE, not just what you want to do.
2. Mention the product type and key constraints (e.g. "price", "color", "size").
3. Do not include specific product names or SKUs — use general categories.

Examples of GOOD queries:
help["search returns no products matching price constraint"]
help["clicked buy but forgot to select size attribute first"]
help["product page does not show color options"]
help["search query too specific and returns empty results"]

Examples of BAD queries:
help["how to buy"] (Too vague)
help["B08XYZ product is wrong"] (Contains specific product ID)
help["what do i do next"] (Not specific to an issue)

Use this tool when you:
- Cannot find a product matching all constraints
- Are unsure which attribute to select
- Keep getting wrong search results
"""

_HELP_INSTRUCTIONS_SQL = """
IMPORTANT: You have access to a help tool when you're struggling or need guidance.
To use it, output an action in this format:
help["your issue here"]

CRITICAL: Your query must match the style of issues in your memory bank to get the best results.
Query Style Guidelines:
1. Describe the SQL ERROR or LOGICAL MISTAKE, not just what you want to query.
2. Mention the SQL operation that failed (e.g. "JOIN", "GROUP BY", "subquery").
3. Mention the MISSING INFORMATION (e.g. "unknown column", "missing FK", "wrong aggregation").
4. Do not include specific table names or column values — use general SQL concepts.

Examples of GOOD queries:
help["missing FK column to join two tables on"]
help["GROUP BY clause missing after using COUNT aggregation"]
help["column name does not exist in table, need to check schema"]
help["subquery returns multiple rows but scalar expected"]
help["wrong aggregation column causes incorrect total"]

Examples of BAD queries:
help["how to write SQL"] (Too vague)
help["table singer_id is wrong"] (Too specific to one schema)
help["what do i do next"] (Not specific to an issue)
help["schema exploration needed before writing the query"] (Routine schema inspection is not a failure)

Use this tool when you:
- Get a SQL error or unexpected result
- Are unsure about column names or table relationships
- Keep getting the wrong result set

Do not use help for routine schema exploration at the start of a task. First
inspect tables normally with SHOW TABLES / SHOW COLUMNS, and call help only
after a concrete SQL error, logical mismatch, or repeated wrong result.
"""

_HELP_INSTRUCTIONS_SCIENCEWORLD = """
IMPORTANT: You have access to a help tool when you're struggling or need guidance.
To use it, output an action in this format:
help["your issue here"]

CRITICAL: Your query must match the style of issues in your memory bank to get the best results.
Query Style Guidelines:
1. Describe the FAILURE or OBSTACLE, not just the science goal.
2. Mention the procedure step or action that failed (e.g. "heating", "mixing", "measuring", "placing").
3. Mention the missing precondition or uncertainty (e.g. "container closed", "wrong substance", "need thermometer reading").
4. Do not include specific object IDs or variation details; use general objects and materials.

Examples of GOOD queries:
help["cannot heat the substance because container is missing"]
help["thermometer reading is needed before comparing melting points"]
help["mixing materials gives no reaction and may need a different container"]
help["plant growth task stalls after watering and needs next procedure step"]

Examples of BAD queries:
help["how to win"] (Too vague)
help["what do i do next"] (Not specific to an issue)
help["substance 7 in variation 42 is wrong"] (Too specific)

Use this tool when you:
- Are unsure which science procedure step comes next
- Need help recovering from a failed action
- Keep getting observations that show no progress
"""


class MemoryAgent(ReAct):
    """
    A memory-augmented agent that:
    1. Retrieves relevant context at task start using context_retrieval
    2. Provides a help["query"] tool that agents can call during execution
    """

    # Supported env_kind values: "alfworld", "webshop", "intercode_sql", "scienceworld"
    def __init__(
        self,
        model: Model = "gemini-2.5-flash",
        to_print: bool = True,
        env_kind: str = "alfworld",
    ):
        super().__init__(model, to_print)
        self.memory_bank_path: Optional[str] = None
        self.env_kind: str = env_kind
        self.min_valid_level: Optional[str] = None

    def _get_help_instructions(self) -> str:
        """
        Returns env-appropriate instructions for the agent on how to use the
        help tool.  Dispatches on self.env_kind so that SQL agents receive SQL-
        flavoured examples rather than ALFWorld household-object examples.
        """
        if self.env_kind == "intercode_sql":
            return _HELP_INSTRUCTIONS_SQL
        if self.env_kind == "webshop":
            return _HELP_INSTRUCTIONS_WEBSHOP
        if self.env_kind == "scienceworld":
            return _HELP_INSTRUCTIONS_SCIENCEWORLD
        return _HELP_INSTRUCTIONS_ALFWORLD

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
                top_k=3,
                min_valid_level=self.min_valid_level,
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
        memory_bank_path: str = "",
        max_learnings: int = 25,
        min_valid_level: str = "",
        split: str = "",
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
            max_learnings: Hard cap on retrieved learnings (default: 25)
            min_valid_level: Minimum validation level filter (default: "" = no filter)

        Returns:
            Tuple of (environment history, success boolean)
        """
        self.memory_bank_path = memory_bank_path
        self.min_valid_level = min_valid_level or None
        
        # Collect steps as (action, observation) pairs
        steps: List[Dict[str, Any]] = []
        help_calls: List[Dict[str, str]] = []
        retrieved_learnings: List[Dict[str, str]] = []  # Store raw learnings for logging
        context_retrieval_error = ""
        context_retrieval_status = "empty"
        embedding_model_used = get_active_embedding_model()
        
        # Step 1: Retrieve context from memory bank at task start
        context_learnings = ""
        if memory_bank_path and task_desc:
            try:
                learnings = retrieve_learnings_only(
                    new_task_desc=task_desc,
                    memory_bank_path=memory_bank_path,
                    top_k_similar_tasks=5,
                    log_dir=log_dir,
                    task_id=task_id,
                    max_learnings=max_learnings,
                    min_valid_level=min_valid_level or None,
                )
                if learnings:
                    retrieved_learnings = learnings  # Store for trajectory logging
                    context_learnings = format_learnings_for_prompt(learnings)
                    context_retrieval_status = "ok"
                    if self.to_print:
                        print("\n" + "="*60)
                        print("CONTEXT FROM PREVIOUS EXPERIENCES:")
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

        # Keep InterCode SQL prompt placement identical to the ReAct baseline.
        # GPT-style SQL models are sensitive to the examples living in the
        # system role: they start emitting multiple SQL commands per action and
        # skip the submit action. Other environments keep the cached prefix path.
        use_system_prompt_cache = self.env_kind != "intercode_sql"
        self._stable_system_prompt = enhanced_prompt if use_system_prompt_cache else None

        env_history = EnvironmentHistory(
            "" if use_system_prompt_cache else enhanced_prompt,
            start_ob,
            memory[-3:] if len(memory) > 3 else memory
        )
        
        if self.to_print:
            print(start_ob)
            sys.stdout.flush()

        cur_step = 0
        reward = 0

        # Token usage tracking
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_cached_tokens = 0

        while cur_step < 49:
            # Choose action — pass the stable system prefix for prompt caching.
            # The stable prefix (enhanced_prompt) is byte-identical across all
            # turns; the growing trajectory in str(env_history) is the variable
            # suffix and is NOT cached.
            allow_newlines = self.env_kind == "intercode_sql"
            action_text, usage = self._llm(
                str(env_history) + "Action:",
                stop=action_stop_sequences(allow_newlines),
                system_prompt=self._stable_system_prompt if use_system_prompt_cache else None,
            )
            action = clean_action_text(action_text, allow_newlines=allow_newlines)
            
            # Update token usage
            step_input_tokens = usage.get("input_tokens", 0)
            step_output_tokens = usage.get("output_tokens", 0)
            step_total_tokens = usage.get("total_tokens", 0)
            step_cached_tokens = usage.get("cached_tokens", 0)

            total_input_tokens += step_input_tokens
            total_output_tokens += step_output_tokens
            total_tokens += step_total_tokens
            total_cached_tokens += step_cached_tokens
            
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
                    "is_help_call": True,
                    "token_usage": {
                        "input_tokens": step_input_tokens,
                        "output_tokens": step_output_tokens,
                        "total_tokens": step_total_tokens,
                        "cached_tokens": step_cached_tokens,
                    }
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
                "is_help_call": False,
                "token_usage": {
                    "input_tokens": step_input_tokens,
                    "output_tokens": step_output_tokens,
                    "total_tokens": step_total_tokens,
                    "cached_tokens": step_cached_tokens,
                }
            })
            
            if self.to_print:
                print(f'Action: {action}\nObs: {observation}')
                sys.stdout.flush()
            
            if done:
                break
            elif env_history.check_is_exhausted():
                if should_auto_submit_repeated_sql(action, allow_newlines):
                    observation, reward, done, info = env.step("submit")
                    env_history.add("action", "submit")
                    env_history.add("observation", observation)
                    steps.append({
                        "step": cur_step + 2,
                        "action": "submit",
                        "observation": observation,
                        "is_help_call": False,
                        "token_usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "cached_tokens": 0,
                        }
                    })
                    if self.to_print:
                        print(f'Action: submit\nObs: {observation}')
                        sys.stdout.flush()
                break
            
            cur_step += 1

        # Determine success. SQL/WebShop can return partial rewards, so treat
        # only full reward as success in trajectory logs used by resume.
        is_success = reward >= 1.0 if self.env_kind in {"intercode_sql", "webshop", "scienceworld"} else reward > 0
        
        if self.to_print:
            print(f"\nTask {'SUCCESS' if is_success else 'FAILURE'}")
            if help_calls:
                print(f"Help tool was called {len(help_calls)} time(s) during this task.")
        
        # Log trajectory
        if log_dir:
            self._log_trajectory(
                log_dir, task_id, trial_num, steps,
                is_success, task_desc, help_calls, retrieved_learnings,
                total_input_tokens, total_output_tokens, total_tokens,
                context_retrieval_error=context_retrieval_error,
                embedding_model_used=embedding_model_used,
                cached_tokens=total_cached_tokens,
                split=split,
                reward=reward,
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
        cached_tokens: int = 0,
        split: str = "",
        reward: float = 0.0,
    ) -> None:
        """Log the complete trajectory to trajectories.json"""
        # Extract task_type from task_id (e.g., 'pick_and_place_simple' from 'pick_and_place_simple-Mug-None-Desk-308/...')
        task_type = task_id.split('-')[0] if task_id else ""

        trajectory = {
            "task_id": task_id,
            "task_type": task_type,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "split": split,
            "context_from_retrieval": context_learnings or [],
            "steps": steps,
            "success": success,
            "reward": float(reward),
            "help_calls": help_calls or [],
            "help_call_count": len(help_calls) if help_calls else 0,
            "step_num": len(steps),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "context_retrieval_error": context_retrieval_error,
            "embedding_model_used": embedding_model_used,
        }
        
        # Write to append-only JSONL to avoid race conditions with the suite runner
        import json as _json
        jsonl_path = os.path.join(log_dir, "agent_trajectories.jsonl")
        line = _json.dumps(trajectory, separators=(",", ":")) + "\n"
        with open(jsonl_path, "a") as f:
            f.write(line)

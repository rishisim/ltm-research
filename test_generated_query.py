
import os
import sys
import json
from src.core.llm import get_chat
from src.frameworks.memory_retrieval_v2.retrieval.core.tool_retrieval import help_tool, format_help_response

# Simulated Trajectory Context (Failure to find knife)
# Scenario: Agent checks typical locations (cabinet, drawer, fridge) but fails to find knife.
TRAJECTORY_OBSERVATIONS = [
    "You are in the kitchen.",
    "Action: open cabinet 1",
    "Obs: The cabinet 1 is open. In it, you see a plate 1.",
    "Action: open drawer 1",
    "Obs: The drawer 1 is open. In it, you see a fork 1.",
    "Action: open fridge 1",
    "Obs: The fridge 1 is open. In it, you see a apple 1.",
    "Action: think: I have looked in the cabinet, drawer, and fridge but still cannot find the knife.",
    "Obs: OK."
]

HISTORY_STR = "\n".join(TRAJECTORY_OBSERVATIONS)

# New Prompt Instructions (from our update)
HELP_INSTRUCTIONS = """
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
"""

FULL_PROMPT = f"""
You are an intelligent agent interacting with an environment.
Your history of observations and actions is:

{HISTORY_STR}

You are stuck and need help. Based on the history above and the help tool instructions below, generate the BEST help query to use.

{HELP_INSTRUCTIONS}

Output ONLY the help tool call, e.g. help["..."]
"""

def run_test():
    print("--- 1. Generating Query using LLM ---")
    print(f"Context:\n{HISTORY_STR}\n")
    
    generated_action = get_chat(FULL_PROMPT, model="gemini-2.5-flash", max_tokens=1024).strip()
    print(f"DEBUG LLM Output: {generated_action}")
    
    # Parse query
    import re
    # Try multiple regexes to catch common formats
    match = re.search(r'help\s*\[\s*["\'](.+?)["\']\s*\]', generated_action, re.IGNORECASE)
    if not match:
        # Fallback: maybe it didn't use brackets?
        match = re.search(r'help\s*["\'](.+?)["\']', generated_action, re.IGNORECASE)
        
    if not match:
        print(f"Failed to generate valid help call. Output: {generated_action}")
        return

    query = match.group(1)
    print(f"Generated Query: '{query}'")
    
    print("\n--- 2. Running Tool Retrieval ---")
    kb_path = "alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json"
    
    result = help_tool(query, kb_path, top_k=5)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_test()

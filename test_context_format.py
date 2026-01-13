"""
Quick test to show context retrieval formatting without running a task
"""

import json
from src.frameworks.memory_allocation.retrieval.context_retrieval import (
    retrieve_context,
    format_learnings_for_prompt
)

# Test with a sample task
test_task_desc = "put a cool apple in microwave."
memory_bank_path = "alfworld_runs/memory_retrieval_v2/memory_allocation_runs/knowledge_base.json"

print("="*80)
print("TESTING CONTEXT RETRIEVAL FORMATTING")
print("="*80)
print(f"\nTest Task: {test_task_desc}\n")

# Retrieve context
result = retrieve_context(
    new_task_desc=test_task_desc,
    memory_bank_path=memory_bank_path,
    top_k_similar_tasks=5,
    force_rebuild_cache=False
)

print("\n" + "="*80)
print("1. WHAT GETS LOGGED TO TRAJECTORY (knowledge_retrieval_base)")
print("="*80)
print("\nFirst 3 rows from knowledge_retrieval_base:\n")
for i, row in enumerate(result["knowledge_retrieval_base"][:3], 1):
    print(f"Row {i}:")
    print(json.dumps(row, indent=2))
    print()

print(f"Total rows in knowledge_retrieval_base: {len(result['knowledge_retrieval_base'])}")

print("\n" + "="*80)
print("2. WHAT GETS SENT TO THE LLM (formatted prompt)")
print("="*80)

# Format learnings for LLM
llm_prompt_section = format_learnings_for_prompt(result["selected_learnings"])

# Show the actual prompt structure that would be sent
full_prompt = f"""[CONTEXT FROM PREVIOUS SIMILAR TASKS]
The following learnings are from previous tasks similar to yours. Use them to avoid common mistakes:

{llm_prompt_section}

[END CONTEXT]

[BASE_PROMPT_WOULD_GO_HERE]"""

print("\n" + full_prompt)

print("\n" + "="*80)
print("3. METADATA")
print("="*80)
print(json.dumps(result["metadata"], indent=2))

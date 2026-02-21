
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.frameworks.memory_retrieval_v2.retrieval.variants.hard_neg_context_retrieval import retrieve_context
from src.frameworks.memory_retrieval_v2.retrieval.variants.hard_neg_tool_retrieval import help_tool

KB_PATH = "alfworld_runs/memory_retrieval_v2/knowledge_base/knowledge_base.json"
KB_ABS_PATH = os.path.abspath(KB_PATH)

def test_hard_neg_cr():
    print("\n--- Testing Hard Negative Context Retrieval ---")
    task_desc = "examine clean mug with desklamp"
    
    try:
        result = retrieve_context(
            new_task_desc=task_desc,
            memory_bank_path=KB_ABS_PATH,
            top_k_similar_tasks=5
        )
        
        print(f"Query: {task_desc}")
        print(f"Selected Learnings: {len(result.get('selected_learnings', []))}")
        
        print("Top 3 Retrieved Tasks (should be low similarity):")
        for t in result.get("metadata", {}).get("top_similar_tasks", [])[:3]:
            print(f"  - [{t['similarity_score']:.4f}] {t['task_desc']}")
            
        print("Top 3 Learnings (should be sorted by low ranking score):")
        for l in result.get("selected_learnings", [])[:3]:
            print(f"  - [Score: {l.get('ranking_score', 0):.4f}] {l.get('issue')}")
            
    except Exception as e:
        print(f"CR Error: {e}")
        import traceback
        traceback.print_exc()

def test_hard_neg_tr():
    print("\n--- Testing Hard Negative Tool Retrieval ---")
    issue = "cannot find mug on desk"
    
    try:
        result = help_tool(
            issue=issue,
            memory_bank_path=KB_ABS_PATH,
            top_k=3
        )
        
        print(f"Issue: {issue}")
        print("Results (should be low scores):")
        for i, res in enumerate(result.get("results", []), 1):
             print(f"  {i}. [Score: {res['TR_rank_score']:.4f}] {res['issue']}")
             
    except Exception as e:
        print(f"TR Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(KB_ABS_PATH):
        print(f"Error: Knowledge base not found at {KB_ABS_PATH}")
    else:
        test_hard_neg_cr()
        test_hard_neg_tr()

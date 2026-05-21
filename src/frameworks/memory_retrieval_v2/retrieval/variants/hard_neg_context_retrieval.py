"""
Hard Negative Context Retrieval Module (Variant)

Retrieves the LEAST similar tasks (distractors) to serve as a control group/baseline.
"Hard Negative" in this context refers to retrieving irrelevant or low-similarity items.
"""

import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# Import from core modules
from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import (
    get_embedding,
    cosine_similarity,
    cosine_similarity_batch,
    load_embeddings_cache,
    create_knowledge_base_embeddings,
    get_batch_embeddings
)
from src.frameworks.memory_retrieval_v2.retrieval.core.learning_counts import (
    load_learning_counts,
    build_learning_counts_table,
    # get_top_similar_tasks - We will implement a LOCAL version for hard negatives
)
from src.frameworks.memory_retrieval_v2.retrieval.core.context_retrieval import (
    load_knowledge_base,
    _format_selected_learnings,
    _save_knowledge_retrieval_base,
    _empty_result,
    format_learnings_for_prompt,
    VALID_LEVEL_PRIORITY,
)

# Reuse core functionality by importing, but we will redefine the main logic flow
# to use hard negative selection.

# -----------------------------------------------------------------------------
# Hard Negative Logic
# -----------------------------------------------------------------------------

def get_hard_neg_tasks(
    query_embedding: np.ndarray,
    learning_counts_entries: List[Dict[str, Any]],
    learning_counts_embeddings: List[np.ndarray],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find top-k LEAST similar tasks (Hard Negatives).
    
    Args:
        query_embedding: Embedding of the query task description
        learning_counts_entries: List of learning counts table entries
        learning_counts_embeddings: List of corresponding embeddings
        top_k: Number of tasks to return
        
    Returns:
        List of entries with similarity scores, sorted by similarity ASCENDING.
    """
    if not learning_counts_entries or not learning_counts_embeddings:
        return []
    
    # Compute similarities
    similarities = []
    for i, emb in enumerate(learning_counts_embeddings):
        norm1 = np.linalg.norm(query_embedding)
        norm2 = np.linalg.norm(emb)
        
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = float(np.dot(query_embedding, emb) / (norm1 * norm2))
            
        similarities.append((i, sim))
    
    # Sort by similarity ASCENDING (Lowest similarity first)
    # We want the "worst" matches that are still valid tasks.
    similarities.sort(key=lambda x: x[1], reverse=False)
    
    # Get bottom-k entries (which are now at the top of the list)
    top_entries = []
    for idx, sim_score in similarities[:top_k]:
        entry = learning_counts_entries[idx].copy()
        entry["similarity_score"] = sim_score
        # Remove embedding
        entry.pop("task_desc_embedding", None)
        top_entries.append(entry)
    
    return top_entries


def _build_knowledge_retrieval_base(
    top_similar_tasks: List[Dict[str, Any]],
    knowledge_base: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build knowledge_retrieval_base by left joining top similar tasks with knowledge_base.
    (Copied from core to ensure compatibility, though logic is identical)
    """
    # Build lookup for top tasks by task_desc
    task_lookup = {t["task_desc"]: t for t in top_similar_tasks}
    
    # Build index of knowledge_base entries by task_desc
    kb_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for i, entry in enumerate(knowledge_base):
        task_desc = entry.get("task_desc", "")
        if task_desc not in kb_by_task:
            kb_by_task[task_desc] = []
        kb_by_task[task_desc].append((i, entry))
    
    # Perform left join
    retrieval_base = []
    
    for task in top_similar_tasks:
        task_desc = task["task_desc"]
        similarity_score = task["similarity_score"]
        learning_count = task["learning_count"]
        
        # Get all matching KB entries for this task_desc
        kb_entries = kb_by_task.get(task_desc, [])
        
        for kb_index, kb_entry in kb_entries:
            # Build joined row - handle None values
            issue_ref = kb_entry.get("issue_ref") or {}
            evidence_ref = kb_entry.get("evidence_ref") or {}
            
            # Fallback
            if not issue_ref:
                issue_ref = {
                    "task_id": kb_entry.get("issue_task_id", ""),
                    "trial_num": kb_entry.get("issue_trial_num", ""),
                    "step_range": kb_entry.get("issue_step_range", [])
                }
            if not evidence_ref:
                evidence_ref = {
                    "task_id": kb_entry.get("evidence_task_id", ""),
                    "trial_num": kb_entry.get("evidence_trial_num", ""),
                    "step_range": kb_entry.get("evidence_step_range", [])
                }
            
            issue_text = issue_ref.get("text", "") or kb_entry.get("issue_text", "")

            prefixed_issue_ref = {
                "issue_text": issue_text,
                "issue_task_id": issue_ref.get("task_id", ""),
                "issue_trial_num": issue_ref.get("trial_num", ""),
                "issue_step_range": issue_ref.get("step_range", []),
            }

            prefixed_evidence_ref = {
                "evidence_task_id": evidence_ref.get("task_id", ""),
                "evidence_trial_num": evidence_ref.get("trial_num", ""),
                "evidence_step_range": evidence_ref.get("step_range", []),
            }

            row = {
                "task_desc": task_desc,
                "similarity_score": similarity_score,
                "learning_count": learning_count,
                "kb_index": kb_index,
                "learning_text": kb_entry.get("learning_text", ""),
                "valid_level": kb_entry.get("valid_level", "CANDIDATE"),
                "obj_type": kb_entry.get("obj_type", ""),
                "verbs": kb_entry.get("verbs", ""),
                "goal_phase": kb_entry.get("goal_phase", ""),
                "unique_id": kb_entry.get("unique_id", ""),
                "issue_ref": prefixed_issue_ref,
                "evidence_ref": prefixed_evidence_ref,
            }
            retrieval_base.append(row)
    
    return retrieval_base


def _rerank_by_lowest_similarity(retrieval_base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank by lowest task similarity while preserving validation quality.

    The hard-negative control should differ from regular CR in relevance, not
    by preferentially selecting lower-validation CANDIDATE memories. Validation
    level is therefore used only as a tie-breaker after the low-similarity
    ranking direction has been established.
    """
    for row in retrieval_base:
        similarity = row.get("similarity_score", 0.0)
        row["ranking_score"] = similarity
    
    retrieval_base.sort(
        key=lambda x: (
            x["ranking_score"],
            -VALID_LEVEL_PRIORITY.get(x.get("valid_level", "CANDIDATE"), 0),
        )
    )
    
    return retrieval_base


def retrieve_context(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    max_learnings: int = 25,
    min_valid_level: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve Hard Negative learnings (Least similar).
    """
    # Step 1: Load cached learning counts table
    learning_counts_entries, learning_counts_embeddings = load_learning_counts(
        memory_bank_path, 
        force_rebuild=force_rebuild_cache
    )
    
    if not learning_counts_entries:
        return _empty_result(new_task_desc, log_dir, task_id)
    
    # Step 2: Embed query
    query_embedding = get_embedding(new_task_desc)
    
    # Step 3.1: Find BOTTOM-k similar tasks (Hard Negatives)
    top_similar_tasks = get_hard_neg_tasks(
        query_embedding,
        learning_counts_entries,
        learning_counts_embeddings,
        top_k=top_k_similar_tasks
    )
    
    if not top_similar_tasks:
        return _empty_result(new_task_desc, log_dir, task_id)
    
    # Step 3.2: Calculate counts (same logic, just based on the returned tasks)
    learning_counts = [task["learning_count"] for task in top_similar_tasks]
    max_learning_count = max(learning_counts)
    pick_learning_count = min(math.ceil(1.5 * max_learning_count), max_learnings)
    
    # Step 3.3: Build base
    knowledge_base = load_knowledge_base(memory_bank_path)
    knowledge_retrieval_base = _build_knowledge_retrieval_base(
        top_similar_tasks, 
        knowledge_base
    )
    
    # Step 3.4: Filter by minimum validation level before ranking so the hard
    # negative baseline remains comparable to regular retrieval.
    if min_valid_level is not None:
        min_priority = VALID_LEVEL_PRIORITY.get(min_valid_level, 0)
        knowledge_retrieval_base = [
            row for row in knowledge_retrieval_base
            if VALID_LEVEL_PRIORITY.get(row.get("valid_level", "CANDIDATE"), 0) >= min_priority
        ]

    # Step 3.5: Re-rank ASCENDING by relevance only.
    knowledge_retrieval_base = _rerank_by_lowest_similarity(knowledge_retrieval_base)

    # Step 3.6: Deduplicate by learning_text like the regular CR path.
    seen_learning_texts = set()
    deduped_base = []
    for row in knowledge_retrieval_base:
        learning_text = row.get("learning_text", "")
        if learning_text not in seen_learning_texts:
            seen_learning_texts.add(learning_text)
            deduped_base.append(row)
    knowledge_retrieval_base = deduped_base

    # Step 3.7: Select top pick_learning_count (the lowest-similarity rows)
    selected_rows = knowledge_retrieval_base[:pick_learning_count]
    
    # Format
    selected_learnings = _format_selected_learnings(selected_rows)
    
    metadata = {
        "pick_learning_count": pick_learning_count,
        "max_learning_count": max_learning_count,
        "actual_selected_count": len(selected_learnings),
        "knowledge_retrieval_base_count": len(knowledge_retrieval_base),
        "top_similar_tasks": [
            {
                "task_desc": t["task_desc"],
                "similarity_score": t["similarity_score"],
                "learning_count": t["learning_count"],
                "validated_learning_count": t.get("validated_learning_count", 0)
            }
            for t in top_similar_tasks
        ]
    }
    
    result = {
        "query": new_task_desc,
        "metadata": metadata,
        "knowledge_retrieval_base": knowledge_retrieval_base,
        "selected_learnings": selected_learnings
    }
    
    if log_dir and task_id:
        _save_knowledge_retrieval_base(result, log_dir, task_id)
    
    return result

def retrieve_learnings_only(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    max_learnings: int = 25,
    min_valid_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for retrieve_context.
    """
    result = retrieve_context(
        new_task_desc,
        memory_bank_path,
        top_k_similar_tasks=top_k_similar_tasks,
        force_rebuild_cache=force_rebuild_cache,
        log_dir=log_dir,
        task_id=task_id,
        max_learnings=max_learnings,
        min_valid_level=min_valid_level,
    )
    return result["selected_learnings"]

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from src.frameworks.memory_retrieval_v2.retrieval.core.context_retrieval import save_as_csv
    
    parser = argparse.ArgumentParser(description="Hard Negative Context Retrieval")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--memory-bank", type=str, required=True, help="Path to knowledge_base.json")
    parser.add_argument("--top-k-tasks", type=int, default=5, help="Number of bottom tasks (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild cache")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Root directory for default output (default: auto-derived from --memory-bank parent). "
            "E.g. 'intercode_sql_runs/hard_neg_test' or 'alfworld_runs/hard_neg_test'."
        ),
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    if args.output_root:
        default_output_dir = project_root / args.output_root / "context_retrieval"
    else:
        # Derive from the memory-bank path to avoid hardcoding alfworld_runs
        memory_bank_parent = Path(args.memory_bank).resolve().parent
        default_output_dir = memory_bank_parent / "hard_neg_test" / "context_retrieval"
    
    print("Using HARD NEGATIVE retrieval algorithm (Least similar)...")
    result = retrieve_context(
        new_task_desc=args.task,
        memory_bank_path=args.memory_bank,
        top_k_similar_tasks=args.top_k_tasks,
        force_rebuild_cache=args.rebuild_cache
    )
    
    if args.output:
        output_path = Path(args.output)
    else:
        safe_task_name = "".join(c if c.isalnum() else "_" for c in args.task).strip("_")[:50]
        filename = f"hard_neg_retrieval_{safe_task_name}.json"
        output_path = default_output_dir / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"JSON Results saved to {output_path}")
    
    csv_output_path = output_path.with_suffix('.csv')
    try:
        save_as_csv(result, csv_output_path)
        print(f"CSV Results saved to {csv_output_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
    
    metadata = result.get("metadata", {})
    print(f"\n=== Hard Negative Retrieval Summary ===")
    print(f"Query: {result.get('query', '')}")
    print(f"Actual selected: {metadata.get('actual_selected_count', len(result.get('selected_learnings', [])))}")
    
    print(f"\nBottom similar tasks (should have low scores):")
    for t in metadata.get("top_similar_tasks", []):
        print(f"  - [{t['similarity_score']:.4f}] {t['task_desc']}")
    
    print(f"\nSelected learnings (should be irrelevant):")
    learnings = result.get("selected_learnings", [])
    for i, l in enumerate(learnings[:10], 1):
        print(f"  {i}. [{l.get('ranking_score', 0):.4f}] {l.get('learning', '')[:80]}...")

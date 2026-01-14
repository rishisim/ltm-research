"""
Context Retrieval Module for Memory Bank (Optimized Version)

Given a new task, performs optimized similarity search using cached embeddings
and a dynamic learning count-based selection strategy.

Algorithm:
1. Load/create cached embeddings for knowledge_base and mem_learning_counts table
2. Embed query task_desc, find top-5 similar tasks from mem_learning_counts
3. Calculate pick_learning_count = ceil(1.5 * max(learning_counts of top-5))
4. Left join top-5 tasks with full knowledge_base to get knowledge_retrieval_base
5. Re-rank: validated entries first, CANDIDATE entries at bottom
6. Select top pick_learning_count learnings for LLM prompt
7. Save full retrieval table to trajectory log
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
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Import from local modules
from .embedding_cache import (
    get_embedding,
    cosine_similarity,
    cosine_similarity_batch,
    load_embeddings_cache,
    create_knowledge_base_embeddings,
    get_batch_embeddings
)
from .learning_counts import (
    load_learning_counts,
    build_learning_counts_table,
    get_top_similar_tasks
)

# Validation level priority (higher is better)
VALID_LEVEL_PRIORITY = {
    "VALID_NEXT_TRIAL": 3,  # Validated by subsequent trial
    "VALID_SAME_TRIAL": 2,  # Validated within same trial
    "CANDIDATE": 1,         # Not yet validated
}

# Columns to extract from knowledge_base for knowledge_retrieval_base
KB_COLUMNS = [
    "issue_text", "learning_text", "valid_level", "obj_type", "verbs", "goal_phase",
    "issue_ref", "evidence_ref"
]


def load_knowledge_base(knowledge_base_path: str) -> List[Dict[str, Any]]:
    """
    Load the knowledge base (raw data) from JSON file.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        
    Returns:
        List of knowledge base entries
    """
    with open(knowledge_base_path, 'r') as f:
        kb = json.load(f)

    # Normalize reference structure to nested issue_ref / evidence_ref
    for entry in kb:
        _normalize_refs(entry)

    return kb


def _normalize_refs(entry: Dict[str, Any]) -> None:
    """Ensure refs are stored in nested objects, preserving flat fields if present."""
    if not entry.get("issue_ref"):
        entry["issue_ref"] = {
            "task_id": entry.get("issue_task_id", ""),
            "trial_num": entry.get("issue_trial_num", ""),
            "step_range": entry.get("issue_step_range", [])
        }
    if not entry.get("evidence_ref"):
        entry["evidence_ref"] = {
            "task_id": entry.get("evidence_task_id", ""),
            "trial_num": entry.get("evidence_trial_num", ""),
            "step_range": entry.get("evidence_step_range", [])
        }


def retrieve_context(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve relevant learnings from the memory bank for a new task.
    
    Uses optimized caching and dynamic learning count selection:
    1. Find top-5 similar tasks from mem_learning_counts table
    2. Calculate pick_learning_count = ceil(1.5 * max_learning_count)
    3. Left join with knowledge_base to build knowledge_retrieval_base
    4. Re-rank with CANDIDATE entries at bottom
    5. Select top pick_learning_count learnings
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        top_k_similar_tasks: Number of top similar tasks to retrieve (default: 5)
        force_rebuild_cache: If True, rebuild caches even if valid
        log_dir: Optional directory to save retrieval files
        task_id: Optional task ID for logging knowledge_retrieval_base to separate file
        
    Returns:
        Dictionary containing:
        - query: The original task description
        - metadata: pick_learning_count, max_learning_count, top tasks info
        - knowledge_retrieval_base: Full retrieval table for logging
        - selected_learnings: Top learnings formatted for LLM
    """
    # Step 1: Load cached learning counts table
    learning_counts_entries, learning_counts_embeddings = load_learning_counts(
        memory_bank_path, 
        force_rebuild=force_rebuild_cache
    )
    
    if not learning_counts_entries:
        return _empty_result(new_task_desc, log_dir, task_id)
    
    # Step 2: Embed query task description
    query_embedding = get_embedding(new_task_desc)
    
    # Step 3.1: Find top-k similar tasks from mem_learning_counts
    top_similar_tasks = get_top_similar_tasks(
        query_embedding,
        learning_counts_entries,
        learning_counts_embeddings,
        top_k=top_k_similar_tasks
    )
    
    if not top_similar_tasks:
        return _empty_result(new_task_desc, log_dir, task_id)
    
    # Step 3.2: Calculate max_learning_count and pick_learning_count
    learning_counts = [task["learning_count"] for task in top_similar_tasks]
    max_learning_count = max(learning_counts)
    pick_learning_count = math.ceil(1.5 * max_learning_count)
    
    # Step 3.3 & 3.4: Build knowledge_retrieval_base via left join
    knowledge_base = load_knowledge_base(memory_bank_path)
    knowledge_retrieval_base = _build_knowledge_retrieval_base(
        top_similar_tasks, 
        knowledge_base
    )
    
    # Step 3.5 & 3.6: Re-rank with CANDIDATE at bottom, order by similarity
    knowledge_retrieval_base = _rerank_by_validation(knowledge_retrieval_base)
    # Enforce similarity < 1.0 on output rows
    knowledge_retrieval_base = [
        row for row in knowledge_retrieval_base
        if row.get("similarity_score", 0.0) < 1.0
    ]
    
    # Step 3.7: Select top pick_learning_count rows
    selected_rows = knowledge_retrieval_base[:pick_learning_count]
    
    # Format selected learnings for LLM
    selected_learnings = _format_selected_learnings(selected_rows)
    
    # Build metadata
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
    
    # Save knowledge_retrieval_base to separate file if log_dir and task_id provided
    if log_dir and task_id:
        _save_knowledge_retrieval_base(result, log_dir, task_id)
    
    return result


def _empty_result(query: str, log_dir: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
    """Return empty result structure."""
    result = {
        "query": query,
        "metadata": {
            "pick_learning_count": 0,
            "max_learning_count": 0,
            "actual_selected_count": 0,
            "knowledge_retrieval_base_count": 0,
            "top_similar_tasks": []
        },
        "knowledge_retrieval_base": [],
        "selected_learnings": []
    }
    
    if log_dir and task_id:
        _save_knowledge_retrieval_base(result, log_dir, task_id)
    
    return result


def _build_knowledge_retrieval_base(
    top_similar_tasks: List[Dict[str, Any]],
    knowledge_base: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build knowledge_retrieval_base by left joining top similar tasks with knowledge_base.
    
    Each row from top_similar_tasks is joined with all matching knowledge_base entries
    on task_desc. Results are ordered by similarity score descending, grouped by task_desc.
    
    Args:
        top_similar_tasks: Top-k similar tasks from mem_learning_counts
        knowledge_base: Full knowledge base entries
        
    Returns:
        List of joined entries with all relevant columns
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
            # Build joined row - handle None values for refs
            issue_ref = kb_entry.get("issue_ref") or {}
            evidence_ref = kb_entry.get("evidence_ref") or {}

            # Fallback to flattened ref fields when nested refs are absent
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
            
            # Build nested refs with prefixed keys and no duplicated flat fields
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
                # From left table (mem_learning_counts)
                "task_desc": task_desc,
                "similarity_score": similarity_score,
                "learning_count": learning_count,
                "kb_index": kb_index,
                # From right table (knowledge_base)
                "learning_text": kb_entry.get("learning_text", ""),
                "valid_level": kb_entry.get("valid_level", "CANDIDATE"),
                "obj_type": kb_entry.get("obj_type", ""),
                "verbs": kb_entry.get("verbs", ""),
                "goal_phase": kb_entry.get("goal_phase", ""),
                # Nested refs only (prefixed keys)
                "issue_ref": prefixed_issue_ref,
                "evidence_ref": prefixed_evidence_ref,
            }
            retrieval_base.append(row)
    
    # Sort by similarity score descending (grouped by task_desc inherently since we iterate by task)
    # Already ordered by task similarity from outer loop, but let's ensure stable sort
    retrieval_base.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return retrieval_base


def _rerank_by_validation(retrieval_base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank retrieval base: validated entries first, CANDIDATE at bottom.
    
    Maintains similarity score ordering within each validation group.
    
    Args:
        retrieval_base: The knowledge_retrieval_base table
        
    Returns:
        Re-ranked table
    """
    # Separate validated and candidate entries
    validated = []
    candidates = []
    
    for row in retrieval_base:
        if row["valid_level"] == "CANDIDATE":
            candidates.append(row)
        else:
            validated.append(row)
    
    # Sort validated by validation priority (descending), then similarity (descending)
    validated.sort(
        key=lambda x: (VALID_LEVEL_PRIORITY.get(x["valid_level"], 0), x["similarity_score"]),
        reverse=True
    )
    
    # Candidates keep similarity ordering
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Concatenate: validated first, then candidates
    return validated + candidates


def _format_selected_learnings(selected_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format selected rows into learnings for LLM prompt.
    
    Args:
        selected_rows: Top rows from knowledge_retrieval_base
        
    Returns:
        List of formatted learning dictionaries
    """
    learnings = []
    
    for row in selected_rows:
        # Pull issue text from nested ref (new structure), fallback to legacy top-level key
        nested_issue = (row.get("issue_ref") or {}).get("issue_text", "")
        issue_text = nested_issue or row.get("issue_text", "")

        learning = {
            "issue": issue_text,
            "learning": row.get("learning_text", ""),
            "valid_level": row.get("valid_level", ""),
            "goal_phase": row.get("goal_phase", ""),
            "task_desc": row.get("task_desc", ""),
            "similarity_score": row.get("similarity_score", 0.0)
        }
        learnings.append(learning)
    
    return learnings


def _save_retrieval_log(result: Dict[str, Any], log_dir: str) -> None:
    """
    Save the full retrieval result as JSON in the trajectory log directory.
    
    Args:
        result: The complete retrieval result
        log_dir: Directory to save the log
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"context_retrieval_{timestamp}.json"
    output_path = log_path / filename
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Retrieval log saved to {output_path}")


def _save_knowledge_retrieval_base(result: Dict[str, Any], log_dir: str, task_id: str) -> None:
    """
    Save knowledge_retrieval_base indexed by task_id to a separate file.
    Appends new task entries to the file.
    
    File format:
    {
      "task_id_1": {
        "query": "...",
        "metadata": {...},
        "knowledge_retrieval_base": [...]
      },
      "task_id_2": {...}
    }
    
    Args:
        result: The complete retrieval result
        log_dir: Directory to save the file
        task_id: Task identifier to use as key
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    retrieval_file = log_path / "knowledge_retrieval_bases.json"
    
    # Load existing data or create new
    if retrieval_file.exists():
        with open(retrieval_file, 'r') as f:
            all_retrievals = json.load(f)
    else:
        all_retrievals = {}
    
    # Add or update this task's retrieval
    all_retrievals[task_id] = {
        "query": result["query"],
        "metadata": result["metadata"],
        "knowledge_retrieval_base": result["knowledge_retrieval_base"]
    }
    
    # Save back
    with open(retrieval_file, 'w') as f:
        json.dump(all_retrievals, f, indent=2)
    
    print(f"  Knowledge retrieval base saved to {retrieval_file} (key: {task_id})")


def format_learnings_for_prompt(learnings: List[Dict[str, Any]]) -> str:
    """
    Format the learnings list as a string suitable for including in a prompt.
    
    Args:
        learnings: List of learning dictionaries (from selected_learnings)
        
    Returns:
        Formatted string of learnings
    """
    if not learnings:
        return "No relevant learnings found."
    
    formatted_lines = [f"Relevant learnings from {len(learnings)} similar past tasks:"]
    
    for i, learning in enumerate(learnings, 1):
        issue = learning.get("issue", "")
        learning_text = learning.get("learning", "")
        
        formatted_lines.append(
            f"\n{i}. Issue: {issue}"
            f"\n   Learning: {learning_text}"
        )
    
    return "\n".join(formatted_lines)


def retrieve_learnings_only(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve only the selected learnings list.
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        top_k_similar_tasks: Number of top similar tasks (default: 5)
        force_rebuild_cache: If True, rebuild caches
        log_dir: Optional directory to save retrieval files
        task_id: Optional task ID for logging
        
    Returns:
        List of selected learning dictionaries
    """
    result = retrieve_context(
        new_task_desc,
        memory_bank_path,
        top_k_similar_tasks=top_k_similar_tasks,
        force_rebuild_cache=force_rebuild_cache,
        log_dir=log_dir,
        task_id=task_id
    )
    return result["selected_learnings"]


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

# Goal phases we want to retrieve from (for legacy function)
GOAL_PHASES = ["SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "RECOVER"]


def retrieve_context_legacy(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar: int = 20,
    top_per_phase: int = 2
) -> Dict[str, Any]:
    """
    Legacy retrieval function for backward compatibility.
    
    Uses the old algorithm: top-k similarity → rank by validation → top per phase.
    """
    # Load knowledge base
    with open(memory_bank_path, 'r') as f:
        memory_bank = json.load(f)
    
    if not memory_bank:
        return {"query": new_task_desc, "retrieved_entries": [], "learnings": []}
    
    task_descs = [entry["task_desc"] for entry in memory_bank]
    query_embedding = get_embedding(new_task_desc)
    bank_embeddings = get_batch_embeddings(task_descs)
    
    similarities = [cosine_similarity(query_embedding, e) for e in bank_embeddings]
    indexed_entries = [(i, sim, memory_bank[i]) for i, sim in enumerate(similarities)]
    indexed_entries.sort(key=lambda x: x[1], reverse=True)
    top_similar = indexed_entries[:top_k_similar]
    
    def get_validation_priority(entry):
        return VALID_LEVEL_PRIORITY.get(entry.get("valid_level", "CANDIDATE"), 0)
    
    top_similar.sort(key=lambda x: (get_validation_priority(x[2]), x[1]), reverse=True)
    
    phase_entries = {phase: [] for phase in GOAL_PHASES}
    
    for idx, similarity, entry in top_similar:
        goal_phase = entry.get("goal_phase", "")
        if goal_phase in phase_entries and len(phase_entries[goal_phase]) < top_per_phase:
            entry_copy = entry.copy()
            entry_copy["_similarity_score"] = float(similarity)
            phase_entries[goal_phase].append(entry_copy)
    
    retrieved_entries = []
    learnings = []
    for phase in GOAL_PHASES:
        for entry in phase_entries[phase]:
            retrieved_entries.append(entry)
            if entry.get("learning_text"):
                learnings.append({
                    "goal_phase": entry.get("goal_phase", ""),
                    "valid_level": entry.get("valid_level", ""),
                    "issue": entry.get("issue_text", ""),
                    "learning": entry.get("learning_text", ""),
                    "task_desc": entry.get("task_desc", ""),
                    "similarity_score": entry.get("_similarity_score", 0.0)
                })
    
    return {"query": new_task_desc, "retrieved_entries": retrieved_entries, "learnings": learnings}


# =============================================================================
# CSV Export
# =============================================================================

import csv

def save_as_csv(data: Dict[str, Any], output_path: Path):
    """
    Save retrieval results as CSV.
    
    Args:
        data: Retrieval result dictionary
        output_path: Path to save CSV
    """
    entries = data.get("knowledge_retrieval_base", [])
    if not entries:
        # Try legacy format
        entries = data.get("retrieved_entries", [])
    
    if not entries:
        return
    
    # Collect all unique keys
    all_keys = set()
    for entry in entries:
        all_keys.update(entry.keys())
    
    # Define column order
    priority_cols = [
        "similarity_score", "valid_level", "goal_phase", "task_desc",
        "learning_text", "issue_text", "learning_count"
    ]
    sorted_priority_cols = [c for c in priority_cols if c in all_keys]
    other_cols = sorted([c for c in all_keys if c not in priority_cols])
    fieldnames = sorted_priority_cols + other_cols
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in entries:
            row = {}
            for k, v in entry.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            writer.writerow(row)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Context retrieval from memory bank")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The new task description to retrieve context for"
    )
    parser.add_argument(
        "--memory-bank",
        type=str,
        required=True,
        help="Path to the knowledge_base.json file"
    )
    parser.add_argument(
        "--top-k-tasks",
        type=int,
        default=5,
        help="Number of top similar tasks from learning counts table (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild embedding caches"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy retrieval algorithm"
    )
    
    args = parser.parse_args()
    
    # Define default output directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    default_output_dir = project_root / "alfworld_runs/memory_allocation_test/tests/context_retrieval"
    
    # Retrieve context
    if args.legacy:
        print("Using legacy retrieval algorithm...")
        result = retrieve_context_legacy(
            new_task_desc=args.task,
            memory_bank_path=args.memory_bank
        )
    else:
        print("Using optimized retrieval algorithm...")
        result = retrieve_context(
            new_task_desc=args.task,
            memory_bank_path=args.memory_bank,
            top_k_similar_tasks=args.top_k_tasks,
            force_rebuild_cache=args.rebuild_cache
        )
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        safe_task_name = "".join(c if c.isalnum() else "_" for c in args.task).strip("_")[:50]
        filename = f"retrieval_{safe_task_name}.json"
        output_path = default_output_dir / filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"JSON Results saved to {output_path}")
    
    # Save CSV
    csv_output_path = output_path.with_suffix('.csv')
    try:
        save_as_csv(result, csv_output_path)
        print(f"CSV Results saved to {csv_output_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
    
    # Print summary
    metadata = result.get("metadata", {})
    print(f"\n=== Retrieval Summary ===")
    print(f"Query: {result.get('query', '')}")
    print(f"Max learning count (from top-5): {metadata.get('max_learning_count', 'N/A')}")
    print(f"Pick learning count (1.5x): {metadata.get('pick_learning_count', 'N/A')}")
    print(f"Actual selected: {metadata.get('actual_selected_count', len(result.get('learnings', [])))}")
    
    print(f"\nTop similar tasks:")
    for t in metadata.get("top_similar_tasks", []):
        print(f"  - [{t['similarity_score']:.4f}] ({t['learning_count']} learnings) {t['task_desc']}")
    
    print(f"\nSelected learnings:")
    learnings = result.get("selected_learnings", result.get("learnings", []))
    for i, l in enumerate(learnings[:10], 1):  # Show first 10
        print(f"  {i}. [{l.get('goal_phase', '')}] {l.get('learning', '')[:80]}...")
    if len(learnings) > 10:
        print(f"  ... and {len(learnings) - 10} more")


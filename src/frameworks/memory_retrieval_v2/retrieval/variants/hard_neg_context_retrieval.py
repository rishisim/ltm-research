"""
Hard Negative Context Retrieval Module for Memory Bank (v2 Optimized)

This is a hard negative version that retrieves the LEAST relevant learnings using
bottom-k similarity search with inverted validation ranking.

Algorithm:
1. Load/create cached embeddings for knowledge_base and mem_learning_counts table
2. Embed query task_desc, find BOTTOM-k similar tasks from mem_learning_counts (LOWEST similarity)
3. Calculate pick_learning_count = ceil(1.5 * max(learning_counts of bottom-k))
4. Left join bottom-k tasks with full knowledge_base to get knowledge_retrieval_base
5. Re-rank: CANDIDATE entries first (highest priority), validated entries at bottom
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
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Import from local modules
from ..core.embedding_cache import (
    get_embedding,
    cosine_similarity,
    cosine_similarity_batch,
    load_embeddings_cache,
    create_knowledge_base_embeddings,
    get_batch_embeddings
)
from ..core.learning_counts import (
    load_learning_counts,
    build_learning_counts_table
)

# INVERTED Validation level priority (CANDIDATES are now prioritized)
VALID_LEVEL_PRIORITY_HARD_NEG = {
    "CANDIDATE": 3,            # Highest priority (was lowest)
    "VALID_SAME_TRIAL": 2,    # Middle
    "VALID_NEXT_TRIAL": 1,    # Lowest priority (was highest)
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


def get_bottom_similar_tasks(
    query_embedding: np.ndarray,
    learning_counts_entries: List[Dict[str, Any]],
    learning_counts_embeddings: List[np.ndarray],
    bottom_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find BOTTOM-k LEAST similar tasks from the learning counts table (HARD NEGATIVE).
    
    Args:
        query_embedding: Embedding of the query task description
        learning_counts_entries: List of learning counts table entries
        learning_counts_embeddings: List of corresponding embeddings
        bottom_k: Number of bottom similar tasks to return
        
    Returns:
        List of bottom-k entries with similarity scores added (lowest similarity first)
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
        
        # Skip degenerate self-matches (similarity >= 1.0)
        if sim >= 1.0:
            continue

        similarities.append((i, sim))
    
    # Sort by similarity ASCENDING (lowest first) - HARD NEGATIVE
    similarities.sort(key=lambda x: x[1])
    
    # Get bottom-k entries with scores
    bottom_entries = []
    for idx, sim_score in similarities[:bottom_k]:
        entry = learning_counts_entries[idx].copy()
        entry["similarity_score"] = sim_score
        # Remove embedding from output (not needed downstream)
        entry.pop("task_desc_embedding", None)
        bottom_entries.append(entry)
    
    return bottom_entries


def retrieve_context(
    new_task_desc: str,
    memory_bank_path: str,
    bottom_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve LEAST relevant learnings from the memory bank for a new task (HARD NEGATIVE).
    
    Uses optimized caching and dynamic learning count selection with inverted ranking:
    1. Find BOTTOM-k LEAST similar tasks from mem_learning_counts table (lowest similarity)
    2. Calculate pick_learning_count = ceil(1.5 * max_learning_count)
    3. Left join with knowledge_base to build knowledge_retrieval_base
    4. Re-rank with CANDIDATE entries first, validated at bottom
    5. Select top pick_learning_count learnings
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        bottom_k_similar_tasks: Number of bottom similar tasks to retrieve (default: 5)
        force_rebuild_cache: If True, rebuild caches even if valid
        log_dir: Optional directory to save retrieval files
        task_id: Optional task ID for logging knowledge_retrieval_base to separate file
        
    Returns:
        Dictionary containing:
        - query: The original task description
        - metadata: pick_learning_count, max_learning_count, bottom tasks info
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
    
    # Step 3.1: Find BOTTOM-k LEAST similar tasks from mem_learning_counts (HARD NEGATIVE)
    bottom_similar_tasks = get_bottom_similar_tasks(
        query_embedding,
        learning_counts_entries,
        learning_counts_embeddings,
        bottom_k=bottom_k_similar_tasks
    )
    
    if not bottom_similar_tasks:
        return _empty_result(new_task_desc, log_dir, task_id)
    
    # Step 3.2: Calculate max_learning_count and pick_learning_count
    learning_counts = [task["learning_count"] for task in bottom_similar_tasks]
    max_learning_count = max(learning_counts)
    pick_learning_count = math.ceil(1.5 * max_learning_count)
    
    # Step 3.3 & 3.4: Build knowledge_retrieval_base via left join
    knowledge_base = load_knowledge_base(memory_bank_path)
    knowledge_retrieval_base = _build_knowledge_retrieval_base(
        bottom_similar_tasks, 
        knowledge_base
    )
    
    # Step 3.5 & 3.6: Re-rank with CANDIDATE first, validated at bottom (HARD NEGATIVE)
    knowledge_retrieval_base = _rerank_by_validation_hard_neg(knowledge_retrieval_base)
    
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
        "bottom_similar_tasks": [
            {
                "task_desc": t["task_desc"],
                "similarity_score": t["similarity_score"],
                "learning_count": t["learning_count"],
                "validated_learning_count": t.get("validated_learning_count", 0)
            }
            for t in bottom_similar_tasks
        ],
        "retrieval_type": "hard_negative"
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
            "bottom_similar_tasks": [],
            "retrieval_type": "hard_negative"
        },
        "knowledge_retrieval_base": [],
        "selected_learnings": []
    }
    
    if log_dir and task_id:
        _save_knowledge_retrieval_base(result, log_dir, task_id)
    
    return result


def _build_knowledge_retrieval_base(
    bottom_similar_tasks: List[Dict[str, Any]],
    knowledge_base: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build knowledge_retrieval_base by left joining bottom similar tasks with knowledge_base.
    
    Each row from bottom_similar_tasks is joined with all matching knowledge_base entries
    on task_desc. Results are ordered by similarity score ascending (lowest first).
    
    Args:
        bottom_similar_tasks: Bottom-k LEAST similar tasks from mem_learning_counts
        knowledge_base: Full knowledge base entries
        
    Returns:
        List of joined entries with all relevant columns
    """
    # Build lookup for bottom tasks by task_desc
    task_lookup = {t["task_desc"]: t for t in bottom_similar_tasks}
    
    # Build index of knowledge_base entries by task_desc
    kb_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for i, entry in enumerate(knowledge_base):
        task_desc = entry.get("task_desc", "")
        if task_desc not in kb_by_task:
            kb_by_task[task_desc] = []
        kb_by_task[task_desc].append((i, entry))
    
    # Perform left join
    retrieval_base = []
    
    for task in bottom_similar_tasks:
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
            
            # Build nested refs with prefixed keys
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
    
    # Sort by similarity score ASCENDING (lowest first) - HARD NEGATIVE
    retrieval_base.sort(key=lambda x: x["similarity_score"])
    
    return retrieval_base


def _rerank_by_validation_hard_neg(retrieval_base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank retrieval base for HARD NEGATIVE: CANDIDATE entries first, validated at bottom.
    
    Maintains similarity score ordering (ascending - lowest first) within each validation group.
    
    Args:
        retrieval_base: The knowledge_retrieval_base table
        
    Returns:
        Re-ranked table with CANDIDATES first
    """
    # Separate candidates and validated entries
    candidates = []
    validated = []
    
    for row in retrieval_base:
        if row["valid_level"] == "CANDIDATE":
            candidates.append(row)
        else:
            validated.append(row)
    
    # Sort candidates by similarity ASCENDING (lowest first)
    candidates.sort(key=lambda x: x["similarity_score"])
    
    # Sort validated by INVERTED validation priority (ascending), then similarity (ascending)
    # Lower priority number = later in output (VALID_NEXT_TRIAL = 1 is last)
    validated.sort(
        key=lambda x: (VALID_LEVEL_PRIORITY_HARD_NEG.get(x["valid_level"], 0), x["similarity_score"])
    )
    
    # Concatenate: CANDIDATES first, then validated (HARD NEGATIVE)
    return candidates + validated


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
    
    retrieval_file = log_path / "knowledge_retrieval_bases_hard_neg.json"
    
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
    
    print(f"  Hard negative knowledge retrieval base saved to {retrieval_file} (key: {task_id})")


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
    bottom_k_similar_tasks: int = 5,
    force_rebuild_cache: bool = False,
    log_dir: Optional[str] = None,
    task_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve only the selected learnings list (HARD NEGATIVE).
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        bottom_k_similar_tasks: Number of bottom similar tasks (default: 5)
        force_rebuild_cache: If True, rebuild caches
        log_dir: Optional directory to save retrieval files
        task_id: Optional task ID for logging
        
    Returns:
        List of selected learning dictionaries
    """
    result = retrieve_context(
        new_task_desc,
        memory_bank_path,
        bottom_k_similar_tasks=bottom_k_similar_tasks,
        force_rebuild_cache=force_rebuild_cache,
        log_dir=log_dir,
        task_id=task_id
    )
    return result["selected_learnings"]

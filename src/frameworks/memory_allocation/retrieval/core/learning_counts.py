"""
Learning Counts Module for Memory Bank

Builds and manages the mem_learning_counts table - an aggregated view of 
the knowledge base grouped by task_desc with learning counts and embeddings.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .embedding_cache import (
    get_file_hash,
    get_batch_embeddings,
    get_embedding,
    EMBEDDING_MODEL
)

# Cache file suffix
LEARNING_COUNTS_CACHE_SUFFIX = ".mem_learning_counts.json"


def get_learning_counts_cache_path(knowledge_base_path: str) -> Path:
    """
    Get the learning counts cache file path for a knowledge base.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        
    Returns:
        Path to the mem_learning_counts cache file
    """
    kb_path = Path(knowledge_base_path)
    return kb_path.parent / f"{kb_path.stem}{LEARNING_COUNTS_CACHE_SUFFIX}"


def is_learning_counts_cache_valid(knowledge_base_path: str, cache_path: Path) -> bool:
    """
    Check if the learning counts cache is valid.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        cache_path: Path to the cache file
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        current_hash = get_file_hash(knowledge_base_path)
        cached_hash = cache_data.get("source_hash", "")
        
        return current_hash == cached_hash
    except (json.JSONDecodeError, IOError):
        return False


def build_learning_counts_table(knowledge_base_path: str, force: bool = False) -> Dict[str, Any]:
    """
    Build the mem_learning_counts table from the knowledge base.
    
    Groups entries by task_desc and counts learnings per task. Each unique
    task_desc gets an embedding for similarity search.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        force: If True, rebuild cache even if valid
        
    Returns:
        Dictionary containing the learning counts table and metadata
    """
    cache_path = get_learning_counts_cache_path(knowledge_base_path)
    
    # Check if cache is valid
    if not force and is_learning_counts_cache_valid(knowledge_base_path, cache_path):
        print(f"Loading existing learning counts cache from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"Building learning counts table for {knowledge_base_path}...")
    
    # Load knowledge base
    with open(knowledge_base_path, 'r') as f:
        knowledge_base = json.load(f)
    
    if not knowledge_base:
        cache_data = {
            "source_hash": get_file_hash(knowledge_base_path),
            "source_path": str(knowledge_base_path),
            "created_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "unique_task_count": 0,
            "total_learning_count": 0,
            "entries": []
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        return cache_data
    
    # Group entries by task_desc
    task_groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "learning_count": 0,
        "entry_indices": [],
        "valid_levels": [],
        "goal_phases": []
    })
    
    for i, entry in enumerate(knowledge_base):
        task_desc = entry.get("task_desc", "")
        if task_desc:
            task_groups[task_desc]["learning_count"] += 1
            task_groups[task_desc]["entry_indices"].append(i)
            task_groups[task_desc]["valid_levels"].append(entry.get("valid_level", "CANDIDATE"))
            task_groups[task_desc]["goal_phases"].append(entry.get("goal_phase", ""))
    
    # Get unique task descriptions
    unique_task_descs = list(task_groups.keys())
    print(f"  Found {len(unique_task_descs)} unique task descriptions")
    
    # Get embeddings for unique task descriptions
    print(f"  Embedding {len(unique_task_descs)} unique task descriptions...")
    embeddings = get_batch_embeddings(unique_task_descs)
    
    # Build the table entries
    table_entries = []
    for task_desc, embedding in zip(unique_task_descs, embeddings):
        group = task_groups[task_desc]
        
        # Count validated learnings
        validated_count = sum(
            1 for vl in group["valid_levels"] 
            if vl in ("VALID_SAME_TRIAL", "VALID_NEXT_TRIAL")
        )
        
        table_entries.append({
            "task_desc": task_desc,
            "task_desc_embedding": embedding.tolist(),
            "learning_count": group["learning_count"],
            "validated_learning_count": validated_count,
            "candidate_count": group["learning_count"] - validated_count,
            "entry_indices": group["entry_indices"],
            "goal_phases": list(set(group["goal_phases"])),  # Unique phases
        })
    
    # Sort by learning count descending for reference
    table_entries.sort(key=lambda x: x["learning_count"], reverse=True)
    
    cache_data = {
        "source_hash": get_file_hash(knowledge_base_path),
        "source_path": str(knowledge_base_path),
        "created_at": datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "unique_task_count": len(table_entries),
        "total_learning_count": len(knowledge_base),
        "entries": table_entries
    }
    
    # Save cache
    print(f"  Saving learning counts cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    
    print(f"  Successfully built learning counts table with {len(table_entries)} unique tasks")
    return cache_data


def load_learning_counts(knowledge_base_path: str, force_rebuild: bool = False) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Load learning counts table and return entries with numpy embeddings.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        force_rebuild: If True, rebuild cache even if valid
        
    Returns:
        Tuple of (list of table entries, list of numpy embedding arrays)
    """
    cache_data = build_learning_counts_table(knowledge_base_path, force=force_rebuild)
    
    entries = cache_data.get("entries", [])
    embeddings = [
        np.array(entry["task_desc_embedding"]) 
        for entry in entries
    ]
    
    return entries, embeddings


def get_top_similar_tasks(
    query_embedding: np.ndarray,
    learning_counts_entries: List[Dict[str, Any]],
    learning_counts_embeddings: List[np.ndarray],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find top-k most similar tasks from the learning counts table.
    
    Args:
        query_embedding: Embedding of the query task description
        learning_counts_entries: List of learning counts table entries
        learning_counts_embeddings: List of corresponding embeddings
        top_k: Number of top similar tasks to return
        
    Returns:
        List of top-k entries with similarity scores added
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
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k entries with scores
    top_entries = []
    for idx, sim_score in similarities[:top_k]:
        entry = learning_counts_entries[idx].copy()
        entry["similarity_score"] = sim_score
        # Remove embedding from output (not needed downstream)
        entry.pop("task_desc_embedding", None)
        top_entries.append(entry)
    
    return top_entries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build learning counts table for knowledge base")
    parser.add_argument(
        "--knowledge-base",
        type=str,
        required=True,
        help="Path to knowledge_base.json"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild cache even if valid"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print cache info without rebuilding"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N tasks by learning count"
    )
    
    args = parser.parse_args()
    
    cache_path = get_learning_counts_cache_path(args.knowledge_base)
    
    if args.info:
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            print(f"Cache file: {cache_path}")
            print(f"Source: {cache_data.get('source_path', 'N/A')}")
            print(f"Created: {cache_data.get('created_at', 'N/A')}")
            print(f"Unique tasks: {cache_data.get('unique_task_count', 0)}")
            print(f"Total learnings: {cache_data.get('total_learning_count', 0)}")
            print(f"\nTop {args.top} tasks by learning count:")
            for i, entry in enumerate(cache_data.get("entries", [])[:args.top]):
                print(f"  {i+1}. [{entry['learning_count']}] {entry['task_desc']}")
        else:
            print(f"No cache found at {cache_path}")
    else:
        cache_data = build_learning_counts_table(
            args.knowledge_base, 
            force=args.rebuild
        )
        print(f"\nLearning counts table ready:")
        print(f"  Unique tasks: {cache_data.get('unique_task_count', 0)}")
        print(f"  Total learnings: {cache_data.get('total_learning_count', 0)}")
        print(f"\nTop {args.top} tasks by learning count:")
        for i, entry in enumerate(cache_data.get("entries", [])[:args.top]):
            validated = entry.get('validated_learning_count', 0)
            candidate = entry.get('candidate_count', 0)
            print(f"  {i+1}. [{entry['learning_count']} total, {validated} validated, {candidate} candidate] {entry['task_desc']}")

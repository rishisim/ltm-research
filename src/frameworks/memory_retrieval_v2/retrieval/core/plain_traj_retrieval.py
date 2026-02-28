"""
Plain Trajectory Retrieval Module

Retrieves top-K similar trajectories based on task_desc embedding similarity
and provides truncated (130 words) raw trajectory context for the agent.

Algorithm:
1. Load training trajectories from trajectories.json
2. Perform similarity search on task_desc field against all trajectories
3. Retrieve top-K matches by similarity
4. Format each trajectory as action/observation steps
5. Truncate each to 130 words maximum

OPTIMIZATION: Embeddings for training trajectories are cached to avoid
recomputing them for every query (reduces API calls significantly).
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .embedding_cache import (
    get_embedding,
    get_batch_embeddings,
    cosine_similarity,
)

# ============================================================================
# EMBEDDING CACHE - stores precomputed embeddings for training trajectories
# ============================================================================
_embedding_cache: Dict[str, Dict[str, Any]] = {}
# Structure: {trajectories_path: {"task_descs": [...], "embeddings": [...], "trajectories": [...]}}


def load_training_trajectories(trajectories_path: str) -> List[Dict[str, Any]]:
    """
    Load the training trajectories from a JSON file.
    """
    with open(trajectories_path, 'r') as f:
        return json.load(f)


def _get_cached_embeddings(trajectories_path: str) -> Tuple[List[str], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Get cached embeddings for training trajectories. Computes and caches if not already cached.
    
    Returns:
        Tuple of (task_descriptions, embeddings, trajectories)
    """
    global _embedding_cache
    
    # Check if already cached
    if trajectories_path in _embedding_cache:
        cache = _embedding_cache[trajectories_path]
        return cache["task_descs"], cache["embeddings"], cache["trajectories"]
    
    # Load trajectories and compute embeddings
    print(f"[Cache] Computing embeddings for training trajectories (one-time)...")
    trajectories = load_training_trajectories(trajectories_path)
    task_descs = [traj.get("task_desc", "") for traj in trajectories]
    
    # Compute embeddings in batches to avoid API limits
    BATCH_SIZE = 100
    all_embeddings = []
    
    for i in range(0, len(task_descs), BATCH_SIZE):
        batch = task_descs[i:i + BATCH_SIZE]
        batch_embeddings = get_batch_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"[Cache] Computed embeddings {i+1}-{min(i+BATCH_SIZE, len(task_descs))} of {len(task_descs)}")
    
    # Cache the results
    _embedding_cache[trajectories_path] = {
        "task_descs": task_descs,
        "embeddings": all_embeddings,
        "trajectories": trajectories
    }
    
    print(f"[Cache] Cached {len(trajectories)} trajectory embeddings")
    return task_descs, all_embeddings, trajectories


def format_single_trajectory(trajectory: Dict[str, Any], word_limit: int = 130) -> str:
    """
    Format a single trajectory as raw text, truncated at word_limit.
    
    Format:
    Task: {task_desc}
    Step 1:
      Action: {action}
      Observation: {observation}
    Step 2:
      ...
    """
    lines = []
    lines.append(f"Task: {trajectory.get('task_desc', '')}")
    
    steps = trajectory.get("steps", [])
    for step in steps:
        step_num = step.get("step", "")
        action = step.get("action", "")
        observation = step.get("observation", "")
        
        lines.append(f"Step {step_num}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
    
    # Join and cut at word limit
    full_text = "\n".join(lines)
    words = full_text.split()
    
    if len(words) > word_limit:
        # Cut at word limit and add indicator
        truncated_words = words[:word_limit]
        return " ".join(truncated_words) + "\n[... trajectory truncated ...]"
    
    return full_text


def retrieve_plain_trajectories(
    task_desc: str,
    trajectories_path: str,
    top_k: int = 1,
    word_limit: int = 130
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve top-K most similar trajectories and format them for the prompt.
    
    Uses cached embeddings for training trajectories to minimize API calls.
    
    Args:
        task_desc: The description of the new task
        trajectories_path: Path to the training trajectories.json file
        top_k: Number of top similar trajectories to retrieve
        word_limit: Maximum word count for each trajectory text
        
    Returns:
        Tuple of (formatted context string, metadata list with source info and similarity)
    """
    # Get cached embeddings (computes once, reuses for subsequent calls)
    task_descs, traj_embeddings, trajectories = _get_cached_embeddings(trajectories_path)
    
    if not trajectories:
        return "", []
    
    # Get embedding for new task query (only API call per task)
    query_embedding = get_embedding(task_desc)
    
    # Calculate similarity scores using cached embeddings
    similarities = [
        cosine_similarity(query_embedding, traj_emb)
        for traj_emb in traj_embeddings
    ]
    
    # Get top-K indices
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build formatted context and metadata
    context_parts = []
    metadata = []
    
    for rank, idx in enumerate(top_k_indices, 1):
        traj = trajectories[idx]
        similarity = similarities[idx]
        
        # Format this trajectory
        formatted_traj = format_single_trajectory(traj, word_limit)
        
        # Add header with similarity score
        header = f"=== Similar Task Example {rank} (similarity: {similarity:.2f}) ==="
        context_parts.append(header)
        context_parts.append(formatted_traj)
        context_parts.append("")  # Blank line between examples
        
        # Store metadata
        metadata.append({
            "rank": rank,
            "source_task_id": traj.get("task_id", "unknown"),
            "similarity_score": float(similarity),
            "task_desc": traj.get("task_desc", ""),
            "success": traj.get("success", False)
        })
    
    formatted_context = "\n".join(context_parts)
    return formatted_context, metadata


def format_context_for_prompt(context: str) -> str:
    """
    Wrap the retrieved context in a prompt-ready format.
    
    Args:
        context: The formatted trajectory context from retrieve_plain_trajectories
        
    Returns:
        Prompt-ready context string
    """
    if not context:
        return ""
    
    return f"""
=== EXAMPLES FROM SIMILAR PAST TASKS ===
{context}
=== END OF EXAMPLES ===
"""

"""
Trajectory Context Retrieval Module for Plain Trajectory CR Ablation Study

Instead of retrieving from the knowledge base (processed issue+learning pairs),
this module retrieves raw trajectory text from training trajectories.

Algorithm:
1. Load training trajectories from memory_allocation_test/trajectories.json
2. Perform similarity search on task_desc field against all trajectories  
3. Retrieve the top match by similarity (regardless of success/failure)
4. Format the trajectory steps as text (action/observation pairs)
5. Cut off at word_count_limit words

OPTIMIZATION: Embeddings for training trajectories are cached to avoid
recomputing them for every query (reduces API calls by ~97.5%).
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Initialize the client with API key from environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-004"

# ============================================================================
# EMBEDDING CACHE - stores precomputed embeddings for training trajectories
# ============================================================================
_embedding_cache: Dict[str, Dict[str, Any]] = {}
# Structure: {trajectories_path: {"task_descs": [...], "embeddings": [...], "trajectories": [...]}}


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a text string using Google's embedding model.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return np.array(result.embeddings[0].values)


def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts using batch API when possible.
    """
    if not texts:
        return []
    
    # Use batch embedding API for efficiency
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    return [np.array(emb.values) for emb in result.embeddings]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


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


def format_trajectory_for_prompt(trajectory: Dict[str, Any], word_count_limit: int) -> str:
    """
    Format a trajectory as raw text for the prompt, cut off at word_count_limit.
    
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
    
    if len(words) > word_count_limit:
        # Cut at word limit and add indicator
        truncated_words = words[:word_count_limit]
        return " ".join(truncated_words) + "\n[... trajectory truncated ...]"
    
    return full_text


def retrieve_trajectory_context(
    new_task_desc: str,
    trajectories_path: str,
    word_count_limit: int = 135  # Default based on average from analysis
) -> Tuple[str, Dict[str, Any]]:
    """
    Retrieve the most similar trajectory and format it for the prompt.
    
    Uses cached embeddings for training trajectories to minimize API calls.
    
    Args:
        new_task_desc: The description of the new task
        trajectories_path: Path to the training trajectories.json file
        word_count_limit: Maximum word count for the trajectory text
        
    Returns:
        Tuple of (formatted trajectory text, metadata dict with source_task_id and similarity)
    """
    # Get cached embeddings (computes once, reuses for subsequent calls)
    task_descs, traj_embeddings, trajectories = _get_cached_embeddings(trajectories_path)
    
    if not trajectories:
        return "", {"source_task_id": None, "similarity_score": 0.0}
    
    # Get embedding for new task query (only API call per task)
    query_embedding = get_embedding(new_task_desc)
    
    # Calculate similarity scores using cached embeddings
    similarities = [
        cosine_similarity(query_embedding, traj_emb)
        for traj_emb in traj_embeddings
    ]
    
    # Find the top match
    best_idx = int(np.argmax(similarities))
    best_similarity = similarities[best_idx]
    best_trajectory = trajectories[best_idx]
    
    # Format the trajectory
    formatted_text = format_trajectory_for_prompt(best_trajectory, word_count_limit)
    
    metadata = {
        "source_task_id": best_trajectory.get("task_id", ""),
        "source_task_desc": best_trajectory.get("task_desc", ""),
        "similarity_score": best_similarity,
        "source_success": best_trajectory.get("success", None),
        "word_count_limit": word_count_limit
    }
    
    return formatted_text, metadata


def clear_embedding_cache():
    """Clear the embedding cache (useful for testing or memory management)."""
    global _embedding_cache
    _embedding_cache = {}
    print("[Cache] Embedding cache cleared")


if __name__ == "__main__":
    # Test the retrieval
    import os
    
    # Path to training trajectories
    base_dir = Path(__file__).parent.parent.parent.parent.parent  # ltm-research/
    traj_path = base_dir / "alfworld_runs" / "memory_allocation_test" / "trajectories.json"
    
    if traj_path.exists():
        # Test multiple queries to demonstrate caching
        test_tasks = [
            "put a book in sofa.",
            "examine the alarmclock with the desklamp.",
            "heat some potato and put it in fridge."
        ]
        
        for i, test_task in enumerate(test_tasks):
            print("\n" + "=" * 60)
            print(f"Query {i+1}: {test_task}")
            print("=" * 60)
            
            context, metadata = retrieve_trajectory_context(
                test_task, 
                str(traj_path),
                word_count_limit=135
            )
            
            print(f"Source Task ID: {metadata['source_task_id']}")
            print(f"Source Task Desc: {metadata['source_task_desc']}")
            print(f"Similarity: {metadata['similarity_score']:.4f}")
            print(f"Source Success: {metadata['source_success']}")
    else:
        print(f"Training trajectories not found at: {traj_path}")

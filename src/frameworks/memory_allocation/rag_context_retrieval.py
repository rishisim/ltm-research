"""
RAG Context Retrieval Module

This module implements context retrieval from a CSV of truncated trajectories.
It computes and caches embeddings for the task descriptions in the CSV,
and performs similarity search to retrieve the most relevant trajectory 
for a given new task description.
"""

import os
import json
import csv
import numpy as np
import csv
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Reuse embedding utilities from existing context_retrieval
from src.frameworks.memory_allocation.context_retrieval import (
    get_embedding,
    get_batch_embeddings,
    cosine_similarity,
    genai_client,
    EMBEDDING_MODEL
)

def load_trajectories_with_embeddings(
    csv_path: str, 
    cache_path: str
) -> List[Dict[str, Any]]:
    """
    Load trajectories from CSV and their embeddings (from cache or compute new).
    
    Args:
        csv_path: Path to the truncated_trajectories.csv file
        cache_path: Path to the JSON file for caching embeddings
        
    Returns:
        List of dictionaries with 'task_desc', 'trunc_trajectory', and 'embedding'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectories CSV not found at {csv_path}")

    # Load CSV using standard library
    raw_data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append(row)
    
    # Check if cache exists
    embeddings_map = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                # Cache format: {"task_desc": [embedding_values]}
                embeddings_map = json.load(f)
            # Convert lists back to numpy arrays
            for k, v in embeddings_map.items():
                embeddings_map[k] = np.array(v)
        except Exception as e:
            print(f"Warning: Failed to load embedding cache: {e}")
            embeddings_map = {}
            
    # Compute missing embeddings
    descriptions_to_embed = []
    indices_to_embed = []
    
    data = []
    
    for idx, row in enumerate(raw_data):
        task_desc = row['task_desc']
        trajectory = row['trunc_trajectory']
        
        entry = {
            'task_desc': task_desc,
            'trunc_trajectory': trajectory,
            'embedding': None
        }
        
        if task_desc in embeddings_map:
            entry['embedding'] = embeddings_map[task_desc]
        else:
            descriptions_to_embed.append(task_desc)
            indices_to_embed.append(len(data))
            
        data.append(entry)
        
    # Batch process missing embeddings
    if descriptions_to_embed:
        print(f"Computing embeddings for {len(descriptions_to_embed)} new task descriptions...")
        
        batch_size = 50  # Safe batch size under 100
        new_embeddings = []
        
        for i in range(0, len(descriptions_to_embed), batch_size):
            batch_texts = descriptions_to_embed[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{len(descriptions_to_embed)//batch_size + 1}...")
            try:
                batch_embs = get_batch_embeddings(batch_texts)
                new_embeddings.extend(batch_embs)
            except Exception as e:
                print(f"Error embedding batch: {e}")
                # Fallback or partial failure handling could go here
                # For now we'll just append None or zeroes to keep alignment if needed, 
                # but get_batch_embeddings generally returns all or fails.
                # If it fails, we might just stop and save what we have or re-raise.
                raise e
        
        # Update data and cache map
        for i, embedding in enumerate(new_embeddings):
            data_idx = indices_to_embed[i]
            data[data_idx]['embedding'] = embedding
            embeddings_map[descriptions_to_embed[i]] = embedding.tolist() # Store as list for JSON
            
        # Save updated cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(embeddings_map, f)
            print(f"Updated embedding cache saved to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save embedding cache: {e}")
            
    return data

def retrieve_similar_trajectory(
    query_task_desc: str,
    csv_path: str,
    cache_path: str
) -> Optional[str]:
    """
    Retrieve the truncated trajectory for the task most similar to the query.
    
    Args:
        query_task_desc: The description of the current task
        csv_path: Path to the truncated_trajectories.csv
        cache_path: Path to the embeddings cache JSON
        
    Returns:
        The truncated trajectory string of the best match, or None
    """
    # Load data with embeddings
    try:
        data = load_trajectories_with_embeddings(csv_path, cache_path)
    except Exception as e:
        print(f"Error loading RAG data: {e}")
        return None
        
    if not data:
        return None
        
    # Embed query
    try:
        query_embedding = get_embedding(query_task_desc)
    except Exception as e:
        print(f"Error embedding query task: {e}")
        return None
        
    # Find best match
    best_similarity = -1.0
    best_trajectory = None
    
    for entry in data:
        if entry['embedding'] is None:
            continue
            
        sim = cosine_similarity(query_embedding, entry['embedding'])
        
        if sim > best_similarity:
            best_similarity = sim
            best_trajectory = entry['trunc_trajectory']
            
    print(f"RAG Retrieval: Best match similarity = {best_similarity:.4f}")
    
    return best_trajectory

if __name__ == "__main__":
    # Test block
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    csv_file = project_root / "alfworld_runs/memory_agent_test/rag_react/truncated_trajectories.csv"
    cache_file = project_root / "alfworld_runs/memory_agent_test/rag_react/trajectory_embeddings.json"
    
    test_query = "examine the mug with the desklamp"
    
    if os.path.exists(csv_file):
        result = retrieve_similar_trajectory(test_query, str(csv_file), str(cache_file))
        print(f"Query: {test_query}")
        print(f"Result: {result[:100]}..." if result else "No result")
    else:
        print(f"CSV file not found at {csv_file}")

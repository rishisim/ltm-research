"""
Hard Negative Tool Retrieval Module for Agent Tool Calls (v2 Optimized)

This is a hard negative version that retrieves the LEAST relevant learnings using
bottom similarity search with inverted validation scoring.

Algorithm:
1. Takes the issue description from the agent
2. Performs similarity search on issue_text to find LOWEST similarity matches
3. Ranks by inverted score: (1.0 - similarity) * validation_weight
   - CANDIDATE: 1.0 weight (highest priority)
   - VALID_SAME_TRIAL: 0.5 weight (lower priority)
   - VALID_NEXT_TRIAL: 0.5 weight (lower priority)
4. Returns bottom K issues and learnings as JSON
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path

from ..core.embedding_cache import (
    get_embedding,
    cosine_similarity,
    load_embeddings_cache,
    cosine_similarity_batch
)

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# INVERTED Validation level weights for multiplicative scoring (HARD NEGATIVE)
# Higher weight = higher priority
VALID_LEVEL_WEIGHT_HARD_NEG = {
    "CANDIDATE": 1.0,            # Highest weight (prioritize unvalidated)
    "VALID_SAME_TRIAL": 0.85,    # Lower weight (de-prioritize validated)
    "VALID_NEXT_TRIAL": 0.6,    # Lower weight (de-prioritize validated)
}

# Global cache for issue_text embeddings (loaded once per knowledge base)
_issue_embeddings_cache = None
_cached_kb_path = None


def load_memory_bank(memory_bank_path: str) -> List[Dict[str, Any]]:
    """
    Load the memory bank (knowledge base) from a JSON file.
    """
    with open(memory_bank_path, 'r') as f:
        return json.load(f)


def load_issue_embeddings(memory_bank_path: str, force_rebuild: bool = False) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Load cached issue_text embeddings for the knowledge base.
    
    Uses the global cache to avoid reloading on every help_tool call within the same session.
    
    Args:
        memory_bank_path: Path to knowledge_base.json
        force_rebuild: If True, rebuild cache even if valid
        
    Returns:
        Tuple of (cache entries with metadata, list of numpy embedding arrays)
    """
    global _issue_embeddings_cache, _cached_kb_path
    
    # Check if we already have the cache loaded for this KB
    if not force_rebuild and _cached_kb_path == memory_bank_path and _issue_embeddings_cache is not None:
        return _issue_embeddings_cache
    
    # Load from disk (or create if doesn't exist)
    print(f"Loading issue_text embeddings cache for {memory_bank_path}...")
    cache_data, embeddings = load_embeddings_cache(
        memory_bank_path, 
        force_rebuild=force_rebuild,
        embed_field="issue_text"
    )
    
    # Store in global cache
    _cached_kb_path = memory_bank_path
    _issue_embeddings_cache = (cache_data.get("entries", []), embeddings)
    
    return _issue_embeddings_cache


def help_tool(
    issue: str,
    memory_bank_path: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    HARD NEGATIVE help tool: Retrieves LEAST relevant learnings when agent is struggling.
    
    Uses inverted scoring to prioritize low similarity and unvalidated candidates.
    
    Args:
        issue: The issue description from the agent (e.g., "cannot find tomato")
        memory_bank_path: Path to the knowledge_base.json file
        top_k: Number of bottom similar issues to retrieve (default: 3)
        
    Returns:
        Dictionary containing:
        - query_issue: The original issue
        - results: List of bottom matching issues with learnings, ranked by inverted score
    """
    # Load memory bank
    memory_bank = load_memory_bank(memory_bank_path)
    
    # Load cached issue_text embeddings
    cached_entries, all_issue_embeddings = load_issue_embeddings(memory_bank_path)
    
    if not memory_bank:
        return {
            "query_issue": issue,
            "message": "Memory bank is empty",
            "results": []
        }
    
    # Step 1: Embed query issue
    query_embedding = get_embedding(issue)
    
    # Step 2: Calculate similarity scores with cached embeddings
    similarities = cosine_similarity_batch(query_embedding, all_issue_embeddings)
    
    # Step 3: Calculate INVERTED scores (HARD NEGATIVE)
    # Score = (1.0 - similarity) * validation_weight
    # Higher score = less similar + prioritize CANDIDATES
    scored_entries = []
    for i, (sim, cache_entry) in enumerate(zip(similarities, cached_entries)):
        # Get the full entry from the original knowledge base using the index
        entry_index = cache_entry.get("index", i)
        if entry_index < len(memory_bank):
            full_entry = memory_bank[entry_index]
            valid_level = full_entry.get("valid_level", "CANDIDATE")
            valid_weight = VALID_LEVEL_WEIGHT_HARD_NEG.get(valid_level, 1.0)
            
            # INVERTED score: (1.0 - similarity) * validation_weight
            # This prioritizes LOW similarity and CANDIDATES
            inverted_score = (1.0 - sim) * valid_weight
            
            scored_entries.append((inverted_score, sim, full_entry))
    
    # Sort by INVERTED score (descending) - highest inverted score = least similar
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    
    # Step 4: Take top_k results (which are actually the LEAST similar)
    top_results = []
    for inverted_score, sim, entry in scored_entries[:top_k]:
        result = {
            "TR_rank_score": float(inverted_score),  # Inverted score
            "similarity_score": float(sim),   # Original similarity (will be low)
            "issue": entry.get("issue_text") or entry.get("issue_ref", {}).get("text", ""),
            "learning": entry.get("learning_text", ""),
            "valid_level": entry.get("valid_level", ""),
            "unique_id": entry.get("unique_id", ""),  # Added unique_id
            "trigger": entry.get("trigger", {}),
            "obj_type": entry.get("obj_type", ""),
            "verbs": entry.get("verbs", "")
        }
        top_results.append(result)
    
    return {
        "query_issue": issue,
        "results": top_results,
        "retrieval_type": "hard_negative"
    }


def format_help_response(response: Dict[str, Any]) -> str:
    """
    Format the help tool response as a string suitable for the agent to read.
    
    Args:
        response: The response dictionary from help_tool
        
    Returns:
        Formatted string response
    """
    if "error" in response:
        return f"Error: {response['error']}"
    
    if not response.get("results"):
        return f"No relevant learnings found for: {response['query_issue']}"
    
    retrieval_type = response.get("retrieval_type", "normal")
    type_label = "HARD NEGATIVE" if retrieval_type == "hard_negative" else "Relevant"
    
    lines = [
        f"Help for issue: \"{response['query_issue']}\"",
        f"\n{type_label} learnings from past experiences:"
    ]
    
    for i, result in enumerate(response["results"], 1):
        lines.append(f"\n{i}. Similar Issue: {result['issue']}")
        lines.append(f"   RELEVANT ISSUE LEARNINGS:")
        lines.append(f"   unique_id: {result.get('unique_id', 'N/A')}")
        lines.append(f"   Issue: {result['issue']}")
        lines.append(f"   Learning: {result['learning']}")
        lines.append(f"   (Validation: {result['valid_level']}, Score: {result.get('TR_rank_score', 0):.2f}, Similarity: {result['similarity_score']:.2f})")
    
    return "\n".join(lines)


import csv

def save_as_csv(data: Dict[str, Any], output_path: Path):
    """
    Save tool retrieval results as CSV using standard csv library.
    Flattens nested dicts/lists into JSON strings.
    """
    results = data.get("results", [])
    if not results:
        return
        
    # Collect all unique keys
    all_keys = set()
    for entry in results:
        all_keys.update(entry.keys())
    
    all_keys = sorted(all_keys)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        
        for entry in results:
            row = {}
            for key in all_keys:
                value = entry.get(key, "")
                # Convert dicts/lists to JSON strings
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
                else:
                    row[key] = value
            writer.writerow(row)
    
    print(f"Saved results to {output_path}")


def save_help_retrieval_log(response: Dict[str, Any], log_dir: str, step_num: int) -> None:
    """
    Save the help tool retrieval result to a log file.
    
    Args:
        response: The response from help_tool
        log_dir: Directory to save the log
        step_num: The step number when help was called
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with step number
    filename = f"help_retrieval_step_{step_num}_hard_neg.json"
    output_path = log_path / filename
    
    with open(output_path, 'w') as f:
        json.dump(response, f, indent=2)
    
    print(f"  Help retrieval log saved to {output_path}")

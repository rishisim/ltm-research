"""
Tool Retrieval Module for Agent Tool Calls

This module provides a help_tool that agents can call during trajectory execution
when they are struggling. The tool:
1. Takes the issue description from the agent
2. Performs similarity search on issue_text across all memory bank entries
3. Ranks by similarity and validation level
4. Returns top K issues and learnings as JSON
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path

from .embedding_cache import (
    get_embedding,
    cosine_similarity,
    load_embeddings_cache
)
from rank_bm25 import BM25Okapi

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Validation level weights for multiplicative scoring
VALID_LEVEL_WEIGHT = {
    "VALID_NEXT_TRIAL": 1.0,
    "VALID_SAME_TRIAL": 0.9,
    "CANDIDATE": 0.6,
}

# Global cache for issue_text embeddings (loaded once per knowledge base)
_issue_embeddings_cache = None
_cached_kb_path = None
_bm25_cache = None

_cached_kb_path = None
_bm25_cache = None


def clean_query(query: str) -> str:
    """
    Clean the query by removing digits and extra whitespace.
    This helps BM25 focus on objects/actions rather than specific instance IDs.
    """
    # Remove digits
    cleaned = re.sub(r'\d+', '', query)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a text string using Google's embedding model.
    Imported from embedding_cache module.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of the embedding vector
    """
    # This is now imported from embedding_cache, but keeping function for backward compatibility
    from .embedding_cache import get_embedding as _get_embedding
    return _get_embedding(text)


def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts.
    Imported from embedding_cache module.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of numpy arrays of embedding vectors
    """
    # This is now imported from embedding_cache, but keeping function for backward compatibility
    from .embedding_cache import get_batch_embeddings as _get_batch_embeddings
    return _get_batch_embeddings(texts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Imported from embedding_cache module.
    """
    # This is now imported from embedding_cache, but keeping function for backward compatibility
    from .embedding_cache import cosine_similarity as _cosine_similarity
    return _cosine_similarity(vec1, vec2)


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
    global _issue_embeddings_cache, _cached_kb_path, _bm25_cache
    
    # Check if we already have the cache loaded for this KB
    if not force_rebuild and _cached_kb_path == memory_bank_path and _issue_embeddings_cache is not None:
        return _issue_embeddings_cache
    
    # Reset BM25 cache if we are switching KBs
    if _cached_kb_path != memory_bank_path:
        _bm25_cache = None

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
    Help tool for agents to retrieve relevant learnings when struggling.
    
    Args:
        issue: The issue description from the agent (e.g., "cannot find tomato")
        memory_bank_path: Path to the knowledge_base.json file
        top_k: Number of top similar issues to retrieve (default: 3)
        
    Returns:
        Dictionary containing:
        - query_issue: The original issue
        - results: List of top matching issues with learnings, ranked by similarity and validation
    """
    global _bm25_cache

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
    
    # --- Prepare BM25 ---
    if _bm25_cache is None:
        print("Building BM25 index...")
        corpus = []
        for entry in memory_bank:
            issue_txt = (entry.get("issue_text") or entry.get("issue_ref", {}).get("text", "")).lower()
            learning_txt = (entry.get("learning_text", "") or "").lower()
            obj_type = (entry.get("obj_type", "") or "").lower()
            verbs = (entry.get("verbs", "") or "").lower()
            
            # New structure: obj_type + verbs + issue_text + learning_text
            doc_text = f"{obj_type} {verbs} {issue_txt} {learning_txt}"
            corpus.append(doc_text.split())
        _bm25_cache = BM25Okapi(corpus)
    
    # --- Pre-process Query ---
    # Clean digits from query for retrieval (but keep original for response)
    search_issue = clean_query(issue)
    
    # --- Step 1: Embedding Similarity ---
    query_embedding = get_embedding(search_issue)
    if all_issue_embeddings:
        query_dim = int(query_embedding.shape[0])
        cache_dim = int(all_issue_embeddings[0].shape[0])
        if query_dim != cache_dim:
            cached_entries, all_issue_embeddings = load_issue_embeddings(
                memory_bank_path,
                force_rebuild=True
            )
            if not all_issue_embeddings:
                return {
                    "query_issue": issue,
                    "message": "Issue embedding cache is empty after rebuild",
                    "results": []
                }

            rebuilt_cache_dim = int(all_issue_embeddings[0].shape[0])
            if query_dim != rebuilt_cache_dim:
                raise ValueError(
                    f"Issue embedding dimension mismatch after forced rebuild: "
                    f"query={query_dim}, cache={rebuilt_cache_dim}"
                )

    from .embedding_cache import cosine_similarity_batch
    # Calculate all cosine similarities
    cos_similarities = cosine_similarity_batch(query_embedding, all_issue_embeddings)
    
    # --- Step 2: BM25 Similarity ---
    tokenized_query = search_issue.lower().split()
    bm25_scores = _bm25_cache.get_scores(tokenized_query)
    
    # --- Step 3: Candidate Selection (Top 100 BM25 U Top 100 Dense) ---
    top_n_candidates = 100
    
    # Get indices of top N cosine scores
    top_cos_indices = np.argsort(cos_similarities)[-top_n_candidates:]
    
    # Get indices of top N BM25 scores
    top_bm25_indices = np.argsort(bm25_scores)[-top_n_candidates:]
    
    # Union of candidates
    candidate_indices = set(top_cos_indices) | set(top_bm25_indices)
    
    # --- Step 4: Normalization and Ranking ---
    # We only care about statistics within the candidate set for BM25 normalization
    candidate_bm25_scores = [bm25_scores[i] for i in candidate_indices]
    min_bm25 = min(candidate_bm25_scores) if candidate_bm25_scores else 0
    max_bm25 = max(candidate_bm25_scores) if candidate_bm25_scores else 1
    if max_bm25 == min_bm25:
        max_bm25 = min_bm25 + 1e-6 # Avoid div/0
        
    scored_entries = []
    
    for i in candidate_indices:
        # Get raw scores
        raw_cos = cos_similarities[i]
        raw_bm25 = bm25_scores[i]
        
        # Normalize Cosine: Clamp(cos, 0, 1)
        cos_norm = max(0.0, min(1.0, float(raw_cos)))
        
        # Normalize BM25: (score - min) / (max - min)
        bm25_norm = (raw_bm25 - min_bm25) / (max_bm25 - min_bm25)
        
        # Get validation weight
        # Ensure we map cache index to memory bank index correctly (should be 1-to-1 but strictly cache_entry has 'index')
        cache_entry = cached_entries[i] if i < len(cached_entries) else {}
        entry_index = cache_entry.get("index", i)
        
        if entry_index < len(memory_bank):
            full_entry = memory_bank[entry_index]
            valid_level = full_entry.get("valid_level", "CANDIDATE")
            valid_weight = VALID_LEVEL_WEIGHT.get(valid_level, 1)
            
            # Final Score
            tr_rank_score = (0.7 * cos_norm + 0.3 * bm25_norm) * valid_weight
            
            result_entry = {
                "TR_rank_score": tr_rank_score,
                "similarity_score": float(raw_cos), # Keep original for reference
                "bm25_score": float(raw_bm25),
                "cos_norm": cos_norm,
                "bm25_norm": bm25_norm,
                "issue": full_entry.get("issue_text") or full_entry.get("issue_ref", {}).get("text", ""),
                "learning": full_entry.get("learning_text", ""),
                "valid_level": valid_level,
                "unique_id": full_entry.get("unique_id", ""),
                "trigger": full_entry.get("trigger", {}),
                "obj_type": full_entry.get("obj_type", ""),
                "verbs": full_entry.get("verbs", "")
            }
            scored_entries.append(result_entry)

    # Sort by TR_rank_score (descending)
    scored_entries.sort(key=lambda x: x["TR_rank_score"], reverse=True)
    
    # Step 5: Return top_k
    return {
        "query_issue": issue,
        "results": scored_entries[:top_k]
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
    
    lines = [
        f"Help for issue: \"{response['query_issue']}\"",
        "Relevant learnings from past experiences:"
    ]
    
    for i, result in enumerate(response["results"], 1):
        lines.append(f"\nLEARNING {i}:")
        lines.append(f"   unique_id: {result.get('unique_id', 'N/A')}")
        lines.append(f"   Issue: {result['issue']}")
        lines.append(f"   Learning: {result['learning']}")
        lines.append(f"   (Validation: {result['valid_level']}, Score: {result.get('TR_rank_score', 0):.2f})") 
        lines.append(f"   (Metrics: Cos={result.get('similarity_score',0):.2f}, BM25={result.get('bm25_score',0):.2f}, CosNorm={result.get('cos_norm',0):.2f}, BM25Norm={result.get('bm25_norm',0):.2f})")
    
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
    
    # Define column order with priority columns first
    priority_cols = ["TR_rank_score", "similarity_score", "valid_level", "issue", "learning"]
    sorted_priority_cols = [c for c in priority_cols if c in all_keys]
    other_cols = [c for c in sorted(list(all_keys)) if c not in priority_cols]
    
    fieldnames = sorted_priority_cols + other_cols
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in results:
            row = {}
            for k, v in entry.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            writer.writerow(row)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Help tool for agent retrieval")
    parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="The issue the agent is facing"
    )
    parser.add_argument(
        "--memory-bank",
        type=str,
        required=True,
        help="Path to the knowledge_base.json file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to retrieve (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-save to tests folder)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Define default output directory relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    default_output_dir = project_root / "alfworld_runs/memory_allocation_test/tests/tool_retrieval"
    
    # Call help tool
    result = help_tool(
        issue=args.issue,
        memory_bank_path=args.memory_bank,
        top_k=args.top_k
    )
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        # Sanitize issue for filename
        safe_issue_name = "".join(c if c.isalnum() else "_" for c in args.issue).strip("_")
        safe_issue_name = safe_issue_name[:50]
        filename = f"tool_retrieval_{safe_issue_name}.json"
        output_path = default_output_dir / filename
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format output
    if args.format == "json":
        output = json.dumps(result, indent=2)
    else:
        output = format_help_response(result)
    
    # Write JSON output
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"JSON Results saved to {output_path}")
    
    # Write CSV output
    csv_output_path = output_path.with_suffix('.csv')
    try:
        save_as_csv(result, csv_output_path)
        print(f"CSV Results saved to {csv_output_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

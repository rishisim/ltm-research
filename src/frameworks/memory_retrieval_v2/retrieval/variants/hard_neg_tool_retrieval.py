"""
Hard Negative Tool Retrieval Module (Variant)

Retrieves the LEAST similar issues/learnings (irrelevant ones) as a control/baseline.
"Hard Negative" here means retrieving the issues with the lowest similarity scores.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path
from rank_bm25 import BM25Okapi

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Import from core
from src.frameworks.memory_retrieval_v2.retrieval.core.embedding_cache import (
    get_embedding,
    cosine_similarity_batch,
    load_embeddings_cache,
    cosine_similarity
)
from src.frameworks.memory_retrieval_v2.retrieval.core.tool_retrieval import (
    clean_query,
    load_memory_bank,
    load_issue_embeddings,
    format_help_response,
    save_as_csv,
    VALID_LEVEL_WEIGHT
)

# Global cache for BM25 (re-implementing here to avoid sharing state with core if run in same process)
_bm25_cache_hn = None
_cached_kb_path_hn = None


def help_tool(
    issue: str,
    memory_bank_path: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Hard Negative Help Tool.
    
    Retrieves the LEAST relevant learnings (lowest similarity).
    Strategically selects "bad" advice.
    
    Args:
        issue: The issue description
        memory_bank_path: Path to knowledge_base.json
        top_k: Number of results to return
        
    Returns:
        Dictionary with query and results (irrelevant ones)
    """
    global _bm25_cache_hn, _cached_kb_path_hn
    
    # Load memory bank
    memory_bank = load_memory_bank(memory_bank_path)
    
    # Load cached issue_text embeddings (we can reuse the core cache loader as it's just raw data)
    cached_entries, all_issue_embeddings = load_issue_embeddings(memory_bank_path)
    
    if not memory_bank:
        return {
            "query_issue": issue,
            "message": "Memory bank is empty",
            "results": []
        }
    
    # --- Prepare BM25 ---
    if _cached_kb_path_hn != memory_bank_path or _bm25_cache_hn is None:
        print("Building BM25 index for Hard Negative...")
        corpus = []
        for entry in memory_bank:
            issue_txt = (entry.get("issue_text") or entry.get("issue_ref", {}).get("text", "")).lower()
            learning_txt = (entry.get("learning_text", "") or "").lower()
            obj_type = (entry.get("obj_type", "") or "").lower()
            verbs = (entry.get("verbs", "") or "").lower()
            
            doc_text = f"{obj_type} {verbs} {issue_txt} {learning_txt}"
            corpus.append(doc_text.split())
        _bm25_cache_hn = BM25Okapi(corpus)
        _cached_kb_path_hn = memory_bank_path
        
    # --- Pre-process Query ---
    search_issue = clean_query(issue)
    
    # --- Step 1: Embedding Similarity ---
    query_embedding = get_embedding(search_issue)
    cos_similarities = cosine_similarity_batch(query_embedding, all_issue_embeddings)
    
    # --- Step 2: BM25 Similarity ---
    tokenized_query = search_issue.lower().split()
    bm25_scores = _bm25_cache_hn.get_scores(tokenized_query)
    
    # --- Step 3: Hard Negative Candidate Selection ---
    # We want the BOTTOM candidates.
    # Logic: Get indices of BOTTOM N cosine scores and BOTTOM N BM25 scores.
    # Note: "Bottom" usually means avoiding 0.0s that are completely unrelated? 
    # Or strictly the lowest scores? 
    # User said: "pick the lowest scores."
    
    top_n_candidates = 100
    
    # Sort indices by score ASCENDING (lowest first)
    sorted_cos_indices = np.argsort(cos_similarities)
    sorted_bm25_indices = np.argsort(bm25_scores)
    
    # Pick lowest scoring indices
    bottom_cos_indices = sorted_cos_indices[:top_n_candidates]
    bottom_bm25_indices = sorted_bm25_indices[:top_n_candidates]
    
    # Union
    candidate_indices = set(bottom_cos_indices) | set(bottom_bm25_indices)
    
    # --- Step 4: Normalization and Ranking ---
    # We normalize based on the full range (0 to max) to keep scores comparable
    # But strictly we just compute scores and sort ASCENDING.
    
    # For normalization context, let's look at the candidate set
    candidate_bm25_scores = [bm25_scores[i] for i in candidate_indices]
    min_bm25 = min(candidate_bm25_scores) if candidate_bm25_scores else 0
    max_bm25 = max(candidate_bm25_scores) if candidate_bm25_scores else 1
    if max_bm25 == min_bm25:
        max_bm25 = min_bm25 + 1e-6
        
    scored_entries = []
    
    for i in candidate_indices:
        raw_cos = cos_similarities[i]
        raw_bm25 = bm25_scores[i]
        
        cos_norm = max(0.0, min(1.0, float(raw_cos)))
        bm25_norm = (raw_bm25 - min_bm25) / (max_bm25 - min_bm25)
        
        # Validation weight
        cache_entry = cached_entries[i] if i < len(cached_entries) else {}
        entry_index = cache_entry.get("index", i)
        
        if entry_index < len(memory_bank):
            full_entry = memory_bank[entry_index]
            valid_level = full_entry.get("valid_level", "CANDIDATE")
            valid_weight = VALID_LEVEL_WEIGHT.get(valid_level, 1.0)
            
            # Weighted Score
            # We use the SAME formula as core
            tr_rank_score = (0.7 * cos_norm + 0.3 * bm25_norm) * valid_weight
            
            result_entry = {
                "TR_rank_score": tr_rank_score,
                "similarity_score": float(raw_cos),
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
            
    # Sort by TR_rank_score ASCENDING (Lowest score first)
    scored_entries.sort(key=lambda x: x["TR_rank_score"], reverse=False)
    
    # Step 5: Return top_k (which are the lowest scoring ones)
    return {
        "query_issue": issue,
        "results": scored_entries[:top_k]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hard Negative Help Tool")
    parser.add_argument("--issue", type=str, required=True, help="Issue description")
    parser.add_argument("--memory-bank", type=str, required=True, help="Path to knowledge_base.json")
    parser.add_argument("--top-k", type=int, default=3, help="Number of bad results")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--format", type=str, choices=["json", "text"], default="json")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    default_output_dir = project_root / "alfworld_runs/hard_neg_test/tool_retrieval"
    
    print("Using HARD NEGATIVE tool retrieval (Bottom 100 candidates, Lowest score first)...")
    result = help_tool(
        issue=args.issue,
        memory_bank_path=args.memory_bank,
        top_k=args.top_k
    )
    
    if args.output:
        output_path = Path(args.output)
    else:
        safe_issue_name = "".join(c if c.isalnum() else "_" for c in args.issue).strip("_")[:50]
        filename = f"hard_neg_tool_retrieval_{safe_issue_name}.json"
        output_path = default_output_dir / filename
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "json":
        output = json.dumps(result, indent=2)
    else:
        output = format_help_response(result)
        
    with open(output_path, 'w') as f:
        f.write(output)
    print(f"Results saved to {output_path}")
    
    # Print summary
    print(f"\nHard Negative Results for '{args.issue}':")
    for i, res in enumerate(result.get("results", []), 1):
        print(f"{i}. [Score: {res['TR_rank_score']:.4f}] {res['issue']} -> {res['learning'][:50]}...")

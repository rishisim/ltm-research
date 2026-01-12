"""
Context Retrieval Module for Memory Bank

Given a new task, performs similarity search on task_desc key in memory bank
to retrieve relevant learnings. The retrieval algorithm:
1. Get top 20 task descriptions by similarity to the new task
2. Rank by validation (validated ones on top)
3. Get top 2 rows per each goal phase
4. Output all learnings in JSON format
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
from google import genai

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Initialize Google GenAI client for embeddings
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-004"

# Validation level priority (higher is better)
VALID_LEVEL_PRIORITY = {
    "VALID_NEXT_TRIAL": 3,  # Validated by subsequent trial
    "VALID_SAME_TRIAL": 2,  # Validated within same trial
    "CANDIDATE": 1,         # Not yet validated
}

# Goal phases we want to retrieve from
GOAL_PHASES = ["SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "RECOVER"]


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a text string using Google's embedding model.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of the embedding vector
    """
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return np.array(result.embeddings[0].values)


def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of numpy arrays of embedding vectors
    """
    if not texts:
        return []
    
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [np.array(emb.values) for emb in result.embeddings]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def load_memory_bank(memory_bank_path: str) -> List[Dict[str, Any]]:
    """
    Load the memory bank (knowledge base) from a JSON file.
    
    Args:
        memory_bank_path: Path to the knowledge_base.json file
        
    Returns:
        List of memory bank entries
    """
    with open(memory_bank_path, 'r') as f:
        return json.load(f)


def retrieve_context(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar: int = 20,
    top_per_phase: int = 2
) -> Dict[str, Any]:
    """
    Retrieve relevant learnings from the memory bank for a new task.
    
    Algorithm:
    1. Similarity search on task_desc to get top_k_similar task descriptions
    2. Rank these by validation level (validated ones first)
    3. Get top_per_phase rows per each goal phase
    4. Return all learnings in JSON format
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        top_k_similar: Number of top similar entries to retrieve (default: 20)
        top_per_phase: Number of top entries to keep per goal phase (default: 2)
        
    Returns:
        Dictionary containing:
        - query: The original task description
        - retrieved_entries: List of selected entries with full info
        - learnings: List of just the learning texts
    """
    # Load memory bank
    memory_bank = load_memory_bank(memory_bank_path)
    
    if not memory_bank:
        return {
            "query": new_task_desc,
            "retrieved_entries": [],
            "learnings": []
        }
    
    # Step 1: Get embeddings for similarity search
    # Extract all unique task descriptions
    task_descs = [entry["task_desc"] for entry in memory_bank]
    
    # Get embedding for new task
    query_embedding = get_embedding(new_task_desc)
    
    # Get embeddings for all task descriptions in memory bank
    bank_embeddings = get_batch_embeddings(task_descs)
    
    # Calculate similarity scores
    similarities = [
        cosine_similarity(query_embedding, bank_emb)
        for bank_emb in bank_embeddings
    ]
    
    # Create list of (index, similarity, entry) tuples
    indexed_entries = [
        (i, sim, memory_bank[i])
        for i, sim in enumerate(similarities)
    ]
    
    # Sort by similarity (descending) and take top_k_similar
    indexed_entries.sort(key=lambda x: x[1], reverse=True)
    top_similar = indexed_entries[:top_k_similar]
    
    # Step 2: Rank by validation level (higher priority first)
    def get_validation_priority(entry: Dict[str, Any]) -> int:
        valid_level = entry.get("valid_level", "CANDIDATE")
        return VALID_LEVEL_PRIORITY.get(valid_level, 0)
    
    # Sort by validation priority (descending), then by similarity (descending)
    top_similar.sort(
        key=lambda x: (get_validation_priority(x[2]), x[1]),
        reverse=True
    )
    
    # Step 3: Get top entries per goal phase
    phase_entries: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in GOAL_PHASES}
    
    for idx, similarity, entry in top_similar:
        goal_phase = entry.get("goal_phase", "")
        if goal_phase in phase_entries:
            if len(phase_entries[goal_phase]) < top_per_phase:
                # Add similarity score to entry for reference
                entry_with_score = entry.copy()
                entry_with_score["_similarity_score"] = float(similarity)
                phase_entries[goal_phase].append(entry_with_score)
    
    # Step 4: Collect all selected entries and extract learnings
    retrieved_entries = []
    learnings = []
    
    for phase in GOAL_PHASES:
        for entry in phase_entries[phase]:
            retrieved_entries.append(entry)
            learning_text = entry.get("learning_text", "")
            if learning_text:
                learnings.append({
                    "goal_phase": entry.get("goal_phase", ""),
                    "valid_level": entry.get("valid_level", ""),
                    "issue": entry.get("issue_text", ""),
                    "learning": learning_text,
                    "task_desc": entry.get("task_desc", ""),
                    "similarity_score": entry.get("_similarity_score", 0.0)
                })
    
    return {
        "query": new_task_desc,
        "retrieved_entries": retrieved_entries,
        "learnings": learnings
    }


def retrieve_learnings_only(
    new_task_desc: str,
    memory_bank_path: str,
    top_k_similar: int = 20,
    top_per_phase: int = 2
) -> List[Dict[str, str]]:
    """
    Convenience function to retrieve only the learnings list.
    
    Args:
        new_task_desc: The description of the new task
        memory_bank_path: Path to the knowledge_base.json file
        top_k_similar: Number of top similar entries to retrieve (default: 20)
        top_per_phase: Number of top entries to keep per goal phase (default: 2)
        
    Returns:
        List of learning dictionaries with goal_phase, valid_level, issue, and learning
    """
    result = retrieve_context(
        new_task_desc, 
        memory_bank_path, 
        top_k_similar, 
        top_per_phase
    )
    return result["learnings"]


def format_learnings_for_prompt(learnings: List[Dict[str, str]]) -> str:
    """
    Format the learnings list as a string suitable for including in a prompt.
    
    Args:
        learnings: List of learning dictionaries
        
    Returns:
        Formatted string of learnings
    """
    if not learnings:
        return "No relevant learnings found."
    
    formatted_lines = ["Relevant learnings from previous tasks:"]
    
    for i, learning in enumerate(learnings, 1):
        phase = learning.get("goal_phase", "Unknown")
        issue = learning.get("issue", "")
        learning_text = learning.get("learning", "")
        
        formatted_lines.append(
            f"\n{i}. Phase: [{phase}]"
            f"\n   Issue: {issue}"
            f"\n   Learning: {learning_text}"
        )
    
    return "\n".join(formatted_lines)


import csv

# ... (imports)

def save_as_csv(data: Dict[str, Any], output_path: Path):
    """
    Save retrieval results as CSV using standard csv library.
    Flattens nested dicts/lists into JSON strings.
    """
    entries = data.get("retrieved_entries", [])
    if not entries:
        return
        
    # Collect all unique keys
    all_keys = set()
    for entry in entries:
        all_keys.update(entry.keys())
    
    # Define column order with priority columns first
    priority_cols = [
        "_similarity_score", "valid_level", "goal_phase", 
        "task_desc", "learning_text", "issue_text"
    ]
    # keys that are in priority_cols
    sorted_priority_cols = [c for c in priority_cols if c in all_keys]
    # remaining keys
    other_cols = [c for c in sorted(list(all_keys)) if c not in priority_cols]
    
    fieldnames = sorted_priority_cols + other_cols
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in entries:
            # Create a row where complex objects are stringified
            row = {}
            for k, v in entry.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v
            writer.writerow(row)


if __name__ == "__main__":
    # Example usage and testing
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
        "--top-k",
        type=int,
        default=20,
        help="Number of top similar entries to retrieve (default: 20)"
    )
    parser.add_argument(
        "--per-phase",
        type=int,
        default=2,
        help="Number of top entries per goal phase (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Define default output directory relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    default_output_dir = project_root / "alfworld_runs/memory_allocation_test/tests/context_retrieval"
    
    # Retrieve context
    result = retrieve_context(
        new_task_desc=args.task,
        memory_bank_path=args.memory_bank,
        top_k_similar=args.top_k,
        top_per_phase=args.per_phase
    )
    
    # Generate output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        # Sanitize task description for filename
        safe_task_name = "".join(c if c.isalnum() else "_" for c in args.task).strip("_")
        # Limit length just in case
        safe_task_name = safe_task_name[:50]
        filename = f"retrieval_{safe_task_name}.json"
        output_path = default_output_dir / filename

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Output results to JSON
    output_json = json.dumps(result, indent=2)
    with open(output_path, 'w') as f:
        f.write(output_json)
    print(f"JSON Results saved to {output_path}")
    
    # Output results to CSV
    csv_output_path = output_path.with_suffix('.csv')
    try:
        save_as_csv(result, csv_output_path)
        print(f"CSV Results saved to {csv_output_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")


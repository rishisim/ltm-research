"""
Tool Retrieval Module for Agent Tool Calls

This module provides a help_tool that agents can call during trajectory execution
when they are struggling. The tool:
1. Takes the issue description and goal_phase from the agent
2. Filters the memory bank by goal_phase
3. Performs similarity search on issue_text within the filtered entries
4. Returns top 3 issues and learnings as JSON
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

# Valid goal phases
GOAL_PHASES = ["SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "RECOVER"]

# Validation level priority for ranking (higher is better)
VALID_LEVEL_PRIORITY = {
    "VALID_NEXT_TRIAL": 3,
    "VALID_SAME_TRIAL": 2,
    "CANDIDATE": 1,
}


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
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def load_memory_bank(memory_bank_path: str) -> List[Dict[str, Any]]:
    """
    Load the memory bank (knowledge base) from a JSON file.
    """
    with open(memory_bank_path, 'r') as f:
        return json.load(f)


def infer_goal_phase(issue_text: str) -> Optional[str]:
    """
    Infer the goal phase from the issue text using keyword matching.
    This is a simple heuristic - could be improved with LLM or better logic.
    
    Args:
        issue_text: The issue description from the agent
        
    Returns:
        Inferred goal phase or None if unable to infer
    """
    issue_lower = issue_text.lower()
    
    # Keywords associated with each phase
    phase_keywords = {
        "SEARCH": ["find", "found", "search", "locate", "look", "where", "not found", "cannot find"],
        "ACQUIRE": ["pick", "take", "grab", "get", "acquire", "hold", "carrying"],
        "TRANSFORM": ["heat", "cool", "clean", "slice", "cut", "cook", "transform", 
                      "microwave", "fridge", "stove", "sink"],
        "PLACE": ["put", "place", "drop", "set", "move to", "place in"],
        "RECOVER": ["stuck", "error", "nothing happens", "wrong", "undo", "close", "open"]
    }
    
    # Count keyword matches for each phase
    phase_scores = {}
    for phase, keywords in phase_keywords.items():
        score = sum(1 for kw in keywords if kw in issue_lower)
        if score > 0:
            phase_scores[phase] = score
    
    if phase_scores:
        return max(phase_scores.keys(), key=lambda x: phase_scores[x])
    
    return None


def help_tool(
    issue: str,
    memory_bank_path: str,
    goal_phase: Optional[str] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Help tool for agents to retrieve relevant learnings when struggling.
    
    Args:
        issue: The issue description from the agent (e.g., "cannot find tomato")
        memory_bank_path: Path to the knowledge_base.json file
        goal_phase: The current goal phase (SEARCH, ACQUIRE, TRANSFORM, PLACE, RECOVER).
                   If None, will attempt to infer from the issue text.
        top_k: Number of top similar issues to retrieve (default: 3)
        
    Returns:
        Dictionary containing:
        - query_issue: The original issue
        - goal_phase: The goal phase used for filtering
        - inferred_phase: Whether the phase was inferred (True) or provided (False)
        - results: List of top matching issues with learnings
    """
    # Load memory bank
    memory_bank = load_memory_bank(memory_bank_path)
    
    # Infer goal phase if not provided
    inferred_phase = False
    if goal_phase is None:
        goal_phase = infer_goal_phase(issue)
        inferred_phase = True
    
    # Validate goal phase
    if goal_phase and goal_phase not in GOAL_PHASES:
        return {
            "query_issue": issue,
            "goal_phase": goal_phase,
            "inferred_phase": inferred_phase,
            "error": f"Invalid goal_phase. Must be one of: {GOAL_PHASES}",
            "results": []
        }
    
    # Step 1: Filter memory bank by goal_phase
    if goal_phase:
        filtered_entries = [
            entry for entry in memory_bank
            if entry.get("goal_phase") == goal_phase
        ]
    else:
        # If no phase, search all entries
        filtered_entries = memory_bank
    
    if not filtered_entries:
        return {
            "query_issue": issue,
            "goal_phase": goal_phase,
            "inferred_phase": inferred_phase,
            "message": f"No entries found for goal_phase: {goal_phase}",
            "results": []
        }
    
    # Step 2: Similarity search on issue_text
    # Extract issue texts from filtered entries
    issue_texts = [entry.get("issue_text", "") for entry in filtered_entries]
    
    # Get embedding for query issue
    query_embedding = get_embedding(issue)
    
    # Get embeddings for all issue texts
    issue_embeddings = get_batch_embeddings(issue_texts)
    
    # Calculate similarity scores
    similarities = [
        cosine_similarity(query_embedding, issue_emb)
        for issue_emb in issue_embeddings
    ]
    
    # Create list of (similarity, entry) tuples and add validation priority
    scored_entries = []
    for i, (sim, entry) in enumerate(zip(similarities, filtered_entries)):
        valid_level = entry.get("valid_level", "CANDIDATE")
        valid_priority = VALID_LEVEL_PRIORITY.get(valid_level, 0)
        scored_entries.append((sim, valid_priority, entry))
    
    # Sort by similarity first, then by validation priority for ties
    scored_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Step 3: Take top_k results
    top_results = []
    for sim, valid_priority, entry in scored_entries[:top_k]:
        result = {
            "similarity_score": float(sim),
            "issue": entry.get("issue_text", ""),
            "learning": entry.get("learning_text", ""),
            "valid_level": entry.get("valid_level", ""),
            "trigger": entry.get("trigger", {}),
            "obj_type": entry.get("obj_type", ""),
            "verbs": entry.get("verbs", "")
        }
        top_results.append(result)
    
    return {
        "query_issue": issue,
        "goal_phase": goal_phase,
        "inferred_phase": inferred_phase,
        "results": top_results
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
        f"Goal Phase: {response['goal_phase']}",
        "\nRelevant learnings from past experiences:"
    ]
    
    for i, result in enumerate(response["results"], 1):
        lines.append(f"\n{i}. Similar Issue: {result['issue']}")
        lines.append(f"   Learning: {result['learning']}")
        lines.append(f"   (Validation: {result['valid_level']}, Similarity: {result['similarity_score']:.2f})")
    
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
    priority_cols = ["similarity_score", "valid_level", "issue", "learning"]
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
        "--goal-phase",
        type=str,
        default=None,
        choices=GOAL_PHASES,
        help="Current goal phase (if not provided, will infer from issue)"
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
        goal_phase=args.goal_phase,
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


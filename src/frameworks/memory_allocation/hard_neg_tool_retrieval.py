"""
Hard Negative Tool Retrieval Module for Agent Tool Calls

This is a hard negative version that retrieves the LEAST relevant learnings:
1. Takes the issue description and goal_phase from the agent
2. Filters the memory bank by goal_phase
3. Performs similarity search to find LOWEST similarity matches
4. Returns bottom 3 issues and learnings as JSON
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

# REVERSED Validation level priority (CANDIDATE is now "best" for hard negative)
VALID_LEVEL_PRIORITY = {
    "VALID_NEXT_TRIAL": 1,  # Now lowest priority
    "VALID_SAME_TRIAL": 2,  # Middle
    "CANDIDATE": 3,         # Now highest priority
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
    HARD NEGATIVE: Help tool that retrieves LEAST relevant learnings.
    
    Args:
        issue: The issue description from the agent (e.g., "cannot find tomato")
        memory_bank_path: Path to the knowledge_base.json file
        goal_phase: The current goal phase (SEARCH, ACQUIRE, TRANSFORM, PLACE, RECOVER).
                   If None, will attempt to infer from the issue text.
        top_k: Number of bottom similar issues to retrieve (default: 3)
        
    Returns:
        Dictionary containing:
        - query_issue: The original issue
        - goal_phase: The goal phase used for filtering
        - inferred_phase: Whether the phase was inferred (True) or provided (False)
        - results: List of LOWEST matching issues with learnings
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
    
    # Create list of (similarity, valid_priority, entry) tuples
    scored_entries = []
    for i, (sim, entry) in enumerate(zip(similarities, filtered_entries)):
        valid_level = entry.get("valid_level", "CANDIDATE")
        valid_priority = VALID_LEVEL_PRIORITY.get(valid_level, 0)
        scored_entries.append((sim, valid_priority, entry))
    
    # HARD NEG: Sort by similarity ASCENDING (lowest first), then by validation priority
    # Lower similarity is "better" for hard negative
    scored_entries.sort(key=lambda x: (x[0], -x[1]), reverse=False)
    
    # Step 3: Take top_k results (which are now the LOWEST similarity)
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

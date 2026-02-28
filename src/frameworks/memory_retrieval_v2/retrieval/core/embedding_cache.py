"""
Embedding Cache Module for Knowledge Base

Manages persistent storage of embeddings to avoid re-computing on every task.
Embeddings are cached alongside the knowledge_base.json file.
"""

import os
import json
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# Initialize Google GenAI client for embeddings
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Embedding model to use (intentionally pinned)
EMBEDDING_MODEL = "gemini-embedding-001"
EMPTY_EMBEDDING_PLACEHOLDER = "[EMPTY_TEXT]"

# Cache file suffix
EMBEDDINGS_CACHE_SUFFIX = ".embeddings_cache.json"


def get_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a file for change detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cache_path(knowledge_base_path: str, embed_field: str = "task_desc") -> Path:
    """
    Get the cache file path for a knowledge base.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        embed_field: Which field is embedded - "task_desc" or "issue_text"
        
    Returns:
        Path to the embeddings cache file
    """
    kb_path = Path(knowledge_base_path)
    if embed_field == "issue_text":
        return kb_path.parent / f"{kb_path.stem}.issue_embeddings_cache.json"
    else:
        return kb_path.parent / f"{kb_path.stem}{EMBEDDINGS_CACHE_SUFFIX}"


def get_active_embedding_model() -> str:
    """
    Return the active embedding model used by retrieval.
    """
    return EMBEDDING_MODEL


def normalize_text_for_embedding(text: Any) -> str:
    """
    Normalize text content so embedding calls never receive empty payloads.
    """
    if text is None:
        return EMPTY_EMBEDDING_PLACEHOLDER
    if not isinstance(text, str):
        text = str(text)
    normalized = text.strip()
    return normalized if normalized else EMPTY_EMBEDDING_PLACEHOLDER


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a single text string.
    
    Args:
        text: The text to embed
        
    Returns:
        numpy array of the embedding vector
    """
    normalized_text = normalize_text_for_embedding(text)
    result = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=normalized_text,
    )
    return np.array(result.embeddings[0].values)


def get_batch_embeddings(texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts with batching to handle API limits.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call (default: 100)
        
    Returns:
        List of numpy arrays of embedding vectors
    """
    if not texts:
        return []

    normalized_texts = [normalize_text_for_embedding(t) for t in texts]
    
    def _embed_with_backoff(items: List[str], max_retries: int = 3) -> List[np.ndarray]:
        """
        Embed a list with graceful fallback:
        - First try as a single batch.
        - If it fails and batch has multiple items, split recursively.
        - If single item fails, retry with exponential backoff then raise.
        """
        if not items:
            return []

        try:
            result = genai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=items,
            )
            return [np.array(emb.values) for emb in result.embeddings]
        except Exception:
            if len(items) == 1:
                for retry in range(max_retries):
                    try:
                        result = genai_client.models.embed_content(
                            model=EMBEDDING_MODEL,
                            contents=items,
                        )
                        return [np.array(result.embeddings[0].values)]
                    except Exception:
                        if retry == max_retries - 1:
                            raise
                        time.sleep(0.5 * (2 ** retry))

            mid = len(items) // 2
            left = _embed_with_backoff(items[:mid], max_retries=max_retries)
            right = _embed_with_backoff(items[mid:], max_retries=max_retries)
            return left + right

    all_embeddings = []
    for i in range(0, len(normalized_texts), batch_size):
        batch = normalized_texts[i:i + batch_size]
        batch_embeddings = _embed_with_backoff(batch)
        all_embeddings.extend(batch_embeddings)

        if i + batch_size < len(normalized_texts):
            print(f"  Embedded {i + batch_size}/{len(normalized_texts)} texts...")

    return all_embeddings


def is_cache_valid(knowledge_base_path: str, cache_path: Path, embed_field: str = "task_desc") -> bool:
    """
    Check if the embeddings cache is valid (exists and matches source file).
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        cache_path: Path to the cache file
        embed_field: Which field is embedded (for validation)
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check if source file hash matches
        current_hash = get_file_hash(knowledge_base_path)
        cached_hash = cache_data.get("source_hash", "")
        
        # Also verify embed_field and embedding model match.
        cached_field = cache_data.get("embed_field", "task_desc")
        cached_model = cache_data.get("embedding_model", "")

        return (
            current_hash == cached_hash
            and cached_field == embed_field
            and cached_model == EMBEDDING_MODEL
        )
    except (json.JSONDecodeError, IOError):
        return False


def create_knowledge_base_embeddings(
    knowledge_base_path: str, 
    force: bool = False,
    embed_field: str = "task_desc"
) -> Dict[str, Any]:
    """
    Create embeddings for all entries in the knowledge base and cache them.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        force: If True, rebuild cache even if valid
        embed_field: Which field to embed - "task_desc" or "issue_text" (default: "task_desc")
        
    Returns:
        Dictionary containing embeddings cache data
    """
    cache_path = get_cache_path(knowledge_base_path, embed_field=embed_field)
    
    # Check if cache is valid
    if not force and is_cache_valid(knowledge_base_path, cache_path, embed_field=embed_field):
        print(f"Loading existing {embed_field} embeddings cache from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"Creating new {embed_field} embeddings cache for {knowledge_base_path}...")
    
    # Load knowledge base
    with open(knowledge_base_path, 'r') as f:
        knowledge_base = json.load(f)
    
    if not knowledge_base:
        cache_data = {
            "source_hash": get_file_hash(knowledge_base_path),
            "source_path": str(knowledge_base_path),
            "created_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "embed_field": embed_field,
            "entry_count": 0,
            "entries": []
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        return cache_data
    
    # Extract all texts to embed based on embed_field
    texts_to_embed = []
    empty_source_indices = []
    for entry in knowledge_base:
        if embed_field == "issue_text":
            # Fallback for issue_ref if issue_text is missing
            text = entry.get("issue_text") or entry.get("issue_ref", {}).get("text", "")
        else:
            text = entry.get(embed_field, "")
        texts_to_embed.append(text)

    sanitized_texts_to_embed = []
    for idx, text in enumerate(texts_to_embed):
        if text is None or (isinstance(text, str) and not text.strip()):
            empty_source_indices.append(idx)
        sanitized_texts_to_embed.append(normalize_text_for_embedding(text))

    if empty_source_indices:
        preview = empty_source_indices[:10]
        print(
            f"  Normalized {len(empty_source_indices)} empty {embed_field} entries "
            f"to placeholder for embedding (sample indices: {preview})"
        )
    
    print(f"  Embedding {len(texts_to_embed)} {embed_field} entries...")
    embeddings = get_batch_embeddings(sanitized_texts_to_embed)
    
    # Build cache data with embeddings
    entries_with_embeddings = []
    for i, (entry, embedding) in enumerate(zip(knowledge_base, embeddings)):
        entries_with_embeddings.append({
            "index": i,
            embed_field: entry.get(embed_field, ""),
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON
            # Include key fields for quick access
            "valid_level": entry.get("valid_level", "CANDIDATE"),
            "goal_phase": entry.get("goal_phase", ""),
        })
    
    cache_data = {
        "source_hash": get_file_hash(knowledge_base_path),
        "source_path": str(knowledge_base_path),
        "created_at": datetime.now().isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "embed_field": embed_field,
        "entry_count": len(entries_with_embeddings),
        "entries": entries_with_embeddings
    }
    
    # Save cache
    print(f"  Saving cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    
    print(f"  Successfully cached {len(entries_with_embeddings)} embeddings")
    return cache_data


def load_embeddings_cache(
    knowledge_base_path: str, 
    force_rebuild: bool = False,
    embed_field: str = "task_desc"
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """
    Load or create embeddings cache and return both cache data and numpy embeddings.
    
    Args:
        knowledge_base_path: Path to knowledge_base.json
        force_rebuild: If True, rebuild cache even if valid
        embed_field: Which field to embed - "task_desc" or "issue_text"
        
    Returns:
        Tuple of (cache_data dict, list of numpy embedding arrays)
    """
    cache_data = create_knowledge_base_embeddings(
        knowledge_base_path, 
        force=force_rebuild,
        embed_field=embed_field
    )
    
    # Convert embeddings back to numpy arrays
    embeddings = [
        np.array(entry["embedding"]) 
        for entry in cache_data.get("entries", [])
    ]
    
    return cache_data, embeddings


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
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_similarity_batch(query_vec: np.ndarray, embeddings: List[np.ndarray]) -> List[float]:
    """
    Compute cosine similarity between a query vector and a batch of embeddings.
    
    Args:
        query_vec: Query embedding vector
        embeddings: List of embedding vectors to compare against
        
    Returns:
        List of similarity scores
    """
    if not embeddings:
        return []
    
    # Stack embeddings into a matrix for efficient computation
    embedding_matrix = np.vstack(embeddings)
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return [0.0] * len(embeddings)
    query_normalized = query_vec / query_norm
    
    # Normalize all embeddings
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embedding_matrix / norms
    
    # Compute similarities
    similarities = np.dot(embeddings_normalized, query_normalized)
    
    return similarities.tolist()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage embeddings cache for knowledge base")
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
    
    args = parser.parse_args()
    
    cache_path = get_cache_path(args.knowledge_base)
    
    if args.info:
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            print(f"Cache file: {cache_path}")
            print(f"Source: {cache_data.get('source_path', 'N/A')}")
            print(f"Created: {cache_data.get('created_at', 'N/A')}")
            print(f"Model: {cache_data.get('embedding_model', 'N/A')}")
            print(f"Entries: {cache_data.get('entry_count', 0)}")
            print(f"Valid: {is_cache_valid(args.knowledge_base, cache_path)}")
        else:
            print(f"No cache found at {cache_path}")
    else:
        cache_data = create_knowledge_base_embeddings(
            args.knowledge_base, 
            force=args.rebuild
        )
        print(f"\nCache ready with {cache_data.get('entry_count', 0)} entries")

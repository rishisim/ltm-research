"""
Embedding Cache Module for Knowledge Base

Manages persistent storage of embeddings to avoid re-computing on every task.
Embeddings are cached alongside the knowledge_base.json file.
"""

import os
import json
import hashlib
import threading
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# NOTE: google.genai is NOT imported at module level.  It is imported lazily
# inside _get_genai_client() so that containers / environments that don't have
# the google-genai package (e.g. the WebShop amd64 Docker image) can import
# this module without errors.  The import will only fail if someone actually
# tries to *use* Gemini embeddings in an environment where google-genai is not
# installed — which will never happen in WebShop runs.

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# ---------------------------------------------------------------------------
# Embedding provider factory
# ---------------------------------------------------------------------------
# Set LTM_EMBEDDING_PROVIDER=openai to use OpenAI text-embedding-3-large.
# Default is "gemini" (gemini-embedding-001), which is the research default.
# Auto-pairing: runners set this env var based on --embedding-provider flag,
# or auto-pair (gemini-* chat → gemini, claude-* chat → openai).
# ---------------------------------------------------------------------------

_EMBEDDING_PROVIDER = os.environ.get("LTM_EMBEDDING_PROVIDER", "gemini").lower()

# Gemini client — only instantiated when needed (lazy import of google.genai)
# Falls back to google.generativeai (older SDK, pydantic 1.x compat) when the
# newer google-genai package is not installed (e.g., inside the WebShop amd64
# Docker container which is pinned to spacy 3.3 / pydantic 1.8).
_genai_client: Optional[Any] = None


class _OldSDKClientWrapper:
    """Adapter that lets google.generativeai (old SDK) speak the new SDK's
    .models.embed_content() shape used throughout this module.

    The new SDK returns: response.embeddings = [Emb(values=[...]), ...]
    The old SDK returns: {"embedding": [...]} for single, or {"embedding": [[...], ...]} for batch.
    This wrapper normalizes both into the new-SDK shape.
    """

    class _Models:
        def embed_content(self, model: str, contents):
            import google.generativeai as _old_sdk  # type: ignore
            # Old SDK requires a "models/" prefix and returns a different schema.
            old_model = model if model.startswith("models/") else f"models/{model}"
            class _Emb:  # noqa: D401
                pass
            class _Resp:  # noqa: D401
                pass
            resp = _Resp()
            if isinstance(contents, list):
                # Old SDK accepts a list and returns {"embedding": [[...], [...]]}
                r = _old_sdk.embed_content(model=old_model, content=contents)
                resp.embeddings = []
                for vec in r["embedding"]:
                    e = _Emb()
                    e.values = vec
                    resp.embeddings.append(e)
            else:
                r = _old_sdk.embed_content(model=old_model, content=contents)
                e = _Emb()
                e.values = r["embedding"]
                resp.embeddings = [e]
            return resp

    def __init__(self):
        self.models = self._Models()


def _get_genai_client() -> Any:
    global _genai_client
    if _genai_client is None:
        # Prefer the newer google-genai SDK (production default on the host).
        try:
            from google import genai as _genai_mod  # type: ignore
            _genai_client = _genai_mod.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            return _genai_client
        except ImportError:
            pass
        # Fall back to google.generativeai (old SDK, pydantic-1 compatible).
        try:
            import google.generativeai as _old_sdk  # type: ignore
            _old_sdk.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            _genai_client = _OldSDKClientWrapper()
            return _genai_client
        except ImportError as exc:
            raise ImportError(
                "Neither google-genai nor google-generativeai is installed.\n"
                "Install one of them: pip install google-genai (preferred), or\n"
                "pip install google-generativeai (older SDK, pydantic-1 compatible).\n"
                "If running inside the WebShop container with neither, set\n"
                "LTM_EMBEDDING_PROVIDER=openai instead (note: this changes the\n"
                "embedding model and will mismatch a Gemini-embedded KB)."
            ) from exc
    return _genai_client

# OpenAI client — only instantiated when needed
_openai_client: Optional[Any] = None

def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        import openai as _openai_mod  # type: ignore
        # Support both openai>=1.0 (new) and openai==0.27 (legacy pinned in requirements.txt).
        # For the embedding path we use the REST API directly to avoid SDK version issues.
        _openai_client = _openai_mod
    return _openai_client

_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# Expose as module-level constant for backward compat (always the active model).
# Use get_active_embedding_model() for a function-based accessor.
EMBEDDING_MODEL: str  # assigned below via get_active_embedding_model()

def get_active_embedding_model() -> str:
    """Return the embedding model name for the active provider."""
    provider = os.environ.get("LTM_EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "openai":
        return _OPENAI_EMBEDDING_MODEL
    return _GEMINI_EMBEDDING_MODEL

# Set module-level constant once at import time (stable for the process lifetime).
EMBEDDING_MODEL = get_active_embedding_model()

EMPTY_EMBEDDING_PLACEHOLDER = "[EMPTY_TEXT]"

# Cache file suffix
EMBEDDINGS_CACHE_SUFFIX = ".embeddings_cache.json"

# Thread-safety lock for cache creation
_cache_lock = threading.Lock()


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


def _gemini_embed_single(normalized_text: str) -> np.ndarray:
    """Embed a single text via Google GenAI."""
    model = get_active_embedding_model()
    result = _get_genai_client().models.embed_content(
        model=model,
        contents=normalized_text,
    )
    return np.array(result.embeddings[0].values)


def _openai_embed_single(normalized_text: str) -> np.ndarray:
    """Embed a single text via OpenAI REST API (provider-agnostic, no SDK version lock)."""
    import requests as _req
    api_key = os.environ.get("OPENAI_API_KEY", "")
    resp = _req.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": _OPENAI_EMBEDDING_MODEL, "input": normalized_text},
        timeout=60,
    )
    resp.raise_for_status()
    return np.array(resp.json()["data"][0]["embedding"])


def _openai_embed_batch(normalized_texts: List[str]) -> List[np.ndarray]:
    """Embed a batch of texts via OpenAI REST API."""
    import requests as _req
    api_key = os.environ.get("OPENAI_API_KEY", "")
    resp = _req.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": _OPENAI_EMBEDDING_MODEL, "input": normalized_texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # OpenAI returns results in the same order as input
    return [np.array(item["embedding"]) for item in data]


def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a single text string using the active provider.

    Args:
        text: The text to embed

    Returns:
        numpy array of the embedding vector
    """
    normalized_text = normalize_text_for_embedding(text)
    provider = os.environ.get("LTM_EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "openai":
        return _openai_embed_single(normalized_text)
    return _gemini_embed_single(normalized_text)


def get_batch_embeddings(texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
    """
    Get embeddings for a batch of texts with batching to handle API limits.

    Uses the active provider (gemini or openai) as set by LTM_EMBEDDING_PROVIDER.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call (default: 100 for Gemini; OpenAI
                    handles up to 2048 items per call but we keep the same batch_size
                    for rate-limit safety)

    Returns:
        List of numpy arrays of embedding vectors
    """
    if not texts:
        return []

    normalized_texts = [normalize_text_for_embedding(t) for t in texts]
    provider = os.environ.get("LTM_EMBEDDING_PROVIDER", "gemini").lower()

    if provider == "openai":
        return _get_batch_embeddings_openai(normalized_texts, batch_size)
    return _get_batch_embeddings_gemini(normalized_texts, batch_size)


def _get_batch_embeddings_gemini(normalized_texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
    """Batch embedding via Google GenAI with recursive split-on-failure backoff."""

    model = get_active_embedding_model()

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
            result = _get_genai_client().models.embed_content(
                model=model,
                contents=items,
            )
            return [np.array(emb.values) for emb in result.embeddings]
        except Exception:
            if len(items) == 1:
                for retry in range(max_retries):
                    try:
                        result = _get_genai_client().models.embed_content(
                            model=model,
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

    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(normalized_texts), batch_size):
        batch = normalized_texts[i:i + batch_size]
        batch_embeddings = _embed_with_backoff(batch)
        all_embeddings.extend(batch_embeddings)

        if i + batch_size < len(normalized_texts):
            print(f"  Embedded {i + batch_size}/{len(normalized_texts)} texts...")

    return all_embeddings


def _get_batch_embeddings_openai(normalized_texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
    """Batch embedding via OpenAI REST API with simple retry on failure."""

    all_embeddings: List[np.ndarray] = []
    max_retries = 3

    for i in range(0, len(normalized_texts), batch_size):
        batch = normalized_texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                embeddings = _openai_embed_batch(batch)
                all_embeddings.extend(embeddings)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))

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
            and cached_model == get_active_embedding_model()
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

    with _cache_lock:
        # Re-check inside lock to avoid redundant rebuilds
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
                "embedding_model": get_active_embedding_model(),
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
                "embedding": embedding.tolist(),
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

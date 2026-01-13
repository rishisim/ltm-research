"""Core retrieval modules"""

from .embedding_cache import (
    get_embedding,
    get_batch_embeddings,
    cosine_similarity,
    cosine_similarity_batch,
    load_embeddings_cache,
    create_knowledge_base_embeddings,
    get_file_hash,
    is_cache_valid
)

from .learning_counts import (
    load_learning_counts,
    build_learning_counts_table,
    get_top_similar_tasks
)

from .context_retrieval import (
    retrieve_context,
    retrieve_learnings_only,
    format_learnings_for_prompt,
    retrieve_context_legacy
)

__all__ = [
    "get_embedding",
    "get_batch_embeddings",
    "cosine_similarity",
    "cosine_similarity_batch",
    "load_embeddings_cache",
    "create_knowledge_base_embeddings",
    "get_file_hash",
    "is_cache_valid",
    "load_learning_counts",
    "build_learning_counts_table",
    "get_top_similar_tasks",
    "retrieve_context",
    "retrieve_learnings_only",
    "format_learnings_for_prompt",
    "retrieve_context_legacy"
]

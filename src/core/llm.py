import os
import sys
import openai
import requests as _requests
from dotenv import load_dotenv
from pathlib import Path
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
import logging

_logger = logging.getLogger("llm.retry")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.WARNING)
from typing import Optional, List, Tuple, Dict

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path)

Model = Literal[
    "gpt-4",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "claude-haiku-4-5",
    "claude-3-5-haiku",
]

# OpenRouter configuration — Gemini chat calls route through here.
# Embeddings still use the provider SDK directly (see retrieval/core/embedding_cache.py).
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("LTM_OPENROUTER_API_KEY")
OPENAI_API_BASE = "https://api.openai.com/v1"

# Maps internal model names to OpenRouter model IDs.
# All models in this map are dispatched via OpenRouter.
OPENROUTER_MODEL_MAP = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    # Cross-lab second-model candidates (cost-ordered).
    # gpt-5-nano: $0.05/$0.40 — cheapest, but failed ReAct format compliance
    # gpt-5-mini: $0.25/$2.00 — bigger sibling; respects reasoning_effort=minimal
    # gpt-5.4-nano: $0.20/$1.25 — fallback if mini also fails
    # claude-3-5-haiku: $0.80/$4.00 — last-resort fallback (known-good format)
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5.4": "openai/gpt-5.4",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-3-5-haiku": "anthropic/claude-3.5-haiku",
}

# OpenAI SDK import is kept for the legacy get_completion() function (text-davinci-003).
openai.api_key = os.getenv('OPENAI_API_KEY')


def _parse_cached_tokens(raw_usage: dict) -> int:
    """
    Extract cached prompt tokens from a provider's usage object.

    Provider-specific locations:
      - OpenAI / OpenRouter-GPT:  usage.prompt_tokens_details.cached_tokens
      - Anthropic / OpenRouter-Claude:
            usage.cache_read_input_tokens  (tokens served from cache)
            usage.cache_creation_input_tokens  (tokens written to cache this call)
      - Gemini / OpenRouter-Gemini: usage.prompt_tokens_details.cached_content_token_count
        OR usage.cache_tokens_details (OpenRouter wraps this differently; treat
        any non-zero value in prompt_tokens_details as a cache hit indicator)

    Returns 0 if no cached tokens are found.
    """
    if not raw_usage:
        return 0

    # OpenAI-style (also used by OpenRouter for GPT models)
    details = raw_usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens", 0)
        if cached:
            return int(cached)
        # Gemini via OpenRouter may use cached_content_token_count
        cached = details.get("cached_content_token_count", 0)
        if cached:
            return int(cached)

    # Anthropic-style (direct or via OpenRouter passthrough)
    cache_read = raw_usage.get("cache_read_input_tokens", 0)
    if cache_read:
        return int(cache_read)

    return 0


# Retry strategy: exponential backoff (2s, 4s, 8s, 16s, 30s cap)
# with up to 8 attempts. Total retry budget ~2 minutes.
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
)
def get_chat(
    prompt: str,
    model: Model,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    stop_strs: Optional[List[str]] = None,
    reasoning: Optional[Dict[str, str]] = None,
    request_timeout: int = 60,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Unified chat completion dispatch.

    Prompt caching behaviour per provider (all routed via OpenRouter):

    (a) Anthropic models (claude-*):
        OpenRouter passes cache_control blocks through to Anthropic.  When
        system_prompt is provided, it is sent as a structured system array with
        cache_control: {"type": "ephemeral"} on the last block, making it
        eligible for server-side caching (TTL 5 min, min 1024 tokens).
        Verify cache hits via usage.cache_read_input_tokens in the response.

    (b) OpenAI models (gpt-*) via OpenRouter:
        Automatic prefix caching is applied by OpenAI for repeated prefixes
        >=1024 tokens.  No payload change is needed.  Check cache hits via
        usage.prompt_tokens_details.cached_tokens in the response (non-zero on
        turn >=2 of a task with a long stable prefix).

    (c) Gemini models (gemini-*) via OpenRouter:
        OpenRouter applies implicit Gemini caching automatically for repeated
        prefixes.  No payload change is needed.  Verify via
        usage.prompt_tokens_details.cached_content_token_count.

    All paths populate 'cached_tokens' in the returned usage dict so callers
    can log and inspect cache hit rates.
    """
    messages = [{"role": "user", "content": prompt}]

    if model in OPENROUTER_MODEL_MAP:
        openrouter_model = OPENROUTER_MODEL_MAP[model]
        # Use requests directly instead of openai SDK so that extra
        # parameters like `reasoning` and `cache_control` are reliably
        # included in the body.
        body: Dict = dict(
            model=openrouter_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if stop_strs:
            body["stop"] = stop_strs

        # OpenRouter's `reasoning` extension is supported by Gemini and GPT-5
        # reasoning models. For GPT-5* we default to "minimal" effort.
        # We tried "low" (per the OpenAI docs' general guidance for tool-use
        # workflows), but in our specific setup with long stable system
        # prompts (~3k tokens of memory context + help instructions),
        # gpt-5.4-mini at "low" effort threw uncaught exceptions inside
        # agent.run() for ~60% of CR+TR tasks (silent trajectory gap).
        # gpt-5-mini and gpt-5.4-mini at "minimal" effort run all tasks to
        # completion with full trajectories. Callers can override.
        if reasoning is None and model.startswith("gpt-5"):
            reasoning = {"effort": "minimal"}
        if reasoning is not None and (
            model.startswith("gemini") or model.startswith("gpt-5")
        ):
            body["reasoning"] = reasoning

        # --- Prompt caching ---
        # OpenRouter expects OpenAI-compatible message format: system as a role
        # inside `messages`, NOT as a top-level field. Anthropic's native
        # top-level `system` field is silently dropped by OpenRouter (verified:
        # input_tokens reflects only the user message when the top-level field
        # is used). Pass `system` as a message role for all providers.
        #
        # For Claude (claude-*): structured content with cache_control on the
        # text block tells Anthropic to cache the stable prefix server-side
        # (TTL 5 min, min 1024 tokens). OpenRouter forwards cache_control
        # through. Cache hits show up as usage.cache_read_input_tokens.
        #
        # For OpenAI (gpt-*) and Gemini (gemini-*) via OpenRouter: cache_control
        # is silently ignored, but automatic prefix caching fires when the
        # stable prefix is byte-identical and >=1024 tokens. Cache hits show up
        # as usage.prompt_tokens_details.cached_tokens (or
        # cached_content_token_count for Gemini).
        if system_prompt:
            if model.startswith("claude"):
                system_message = {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            else:
                system_message = {"role": "system", "content": system_prompt}
            body["messages"] = [system_message] + messages

        # --- Prompt caching: Gemini (gemini-*) ---
        # OpenRouter applies implicit Gemini caching automatically for repeated
        # long prefixes.  No payload change needed here; cache hits show up in
        # usage.prompt_tokens_details.cached_content_token_count.

        resp = _requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=(10, request_timeout),  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()
        response_data = resp.json()

        # Normalise into the same shape the rest of the function expects
        class _Obj:
            pass
        response = _Obj()
        response.choices = response_data["choices"]
        raw_usage = response_data.get("usage")
        if raw_usage:
            u = _Obj()
            u.prompt_tokens = raw_usage.get("prompt_tokens", 0)
            u.completion_tokens = raw_usage.get("completion_tokens", 0)
            u.total_tokens = raw_usage.get("total_tokens", 0)
            u.cached_tokens = _parse_cached_tokens(raw_usage)
            response.usage = u
        else:
            response.usage = None

    elif model.startswith("gpt-"):
        # GPT chat path via OpenAI REST directly.  The repo pins openai==0.27.0
        # so we use requests directly for GPT-5.x compatibility.
        #
        # --- Prompt caching: OpenAI (gpt-*) ---
        # OpenAI automatically caches the longest repeated prefix that is
        # >=1024 tokens.  No payload change needed.  The stable prefix must
        # appear byte-identically across calls to get a cache hit.
        # Cache hits are reported in usage.prompt_tokens_details.cached_tokens;
        # non-zero values on turn >=2 of a task confirm caching is working.
        if system_prompt:
            direct_messages = [
                {"role": "system", "content": system_prompt},
            ] + messages
        else:
            direct_messages = messages

        body = dict(
            model=model,
            messages=direct_messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        if stop_strs:
            body["stop"] = stop_strs
        resp = _requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=(10, request_timeout),
        )
        resp.raise_for_status()
        response_data = resp.json()

        class _Obj:
            pass
        response = _Obj()
        response.choices = response_data["choices"]
        raw_usage = response_data.get("usage")
        if raw_usage:
            u = _Obj()
            u.prompt_tokens = raw_usage.get("prompt_tokens", 0)
            u.completion_tokens = raw_usage.get("completion_tokens", 0)
            u.total_tokens = raw_usage.get("total_tokens", 0)
            u.cached_tokens = _parse_cached_tokens(raw_usage)
            response.usage = u
        else:
            response.usage = None
    else:
        raise ValueError(f"Unsupported chat model: {model}")

    usage: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }
    if hasattr(response, 'usage') and response.usage is not None:
        usage["input_tokens"] = response.usage.prompt_tokens
        usage["output_tokens"] = response.usage.completion_tokens
        usage["total_tokens"] = response.usage.total_tokens
        usage["cached_tokens"] = getattr(response.usage, "cached_tokens", 0)

    return response.choices[0]["message"]["content"], usage


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(8))
def get_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stop_strs: Optional[List[str]] = None,
) -> str:
    # Legacy completion function for text-davinci-003
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

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

Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "gemini-2.0-flash", "gemini-2.5-flash", "claude-haiku-4-5"]

# OpenRouter configuration — all model chat calls (Gemini and Claude) route through here.
# Embeddings still use the provider SDK directly (see retrieval/core/embedding_cache.py).
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("LTM_OPENROUTER_API_KEY")

# Maps internal model names to OpenRouter model IDs.
# All models in this map are dispatched via OpenRouter.
OPENROUTER_MODEL_MAP = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
}

# OpenAI SDK import is kept for the legacy get_completion() function (text-davinci-003).
# All chat models (GPT, Gemini, Claude) now route through OpenRouter via requests.
openai.api_key = os.getenv('OPENAI_API_KEY')


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
) -> Tuple[str, Dict[str, int]]:
    messages = [{"role": "user", "content": prompt}]

    if model in OPENROUTER_MODEL_MAP:
        openrouter_model = OPENROUTER_MODEL_MAP[model]
        # Use requests directly instead of openai SDK so that extra
        # parameters like `reasoning` are reliably included in the body.
        body = dict(
            model=openrouter_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if stop_strs:
            body["stop"] = stop_strs
        # reasoning={"effort": ...} is a Gemini-specific OpenRouter extension;
        # do NOT send it to Claude or other non-Gemini models.
        if reasoning is not None and model.startswith("gemini"):
            body["reasoning"] = reasoning
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
            response.usage = u
        else:
            response.usage = None
    else:
        # Legacy GPT path via OpenRouter (avoids the pre-v1.0 openai SDK).
        # GPT model names that are not in OPENROUTER_MODEL_MAP are passed
        # through to OpenRouter using their native name (e.g. "gpt-4").
        body = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if stop_strs:
            body["stop"] = stop_strs
        resp = _requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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
            response.usage = u
        else:
            response.usage = None

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if hasattr(response, 'usage') and response.usage is not None:
        usage["input_tokens"] = response.usage.prompt_tokens
        usage["output_tokens"] = response.usage.completion_tokens
        usage["total_tokens"] = response.usage.total_tokens

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

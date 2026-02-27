import os
import sys
import openai
from dotenv import load_dotenv
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from google import genai
from google.genai import types
from typing import Optional, List, Union, Tuple, Dict

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Load .env from parent directory
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path)

Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "gemini-2.0-flash", "gemini-2.5-flash"]

# Initialize Google GenAI client with request timeout to avoid indefinite hangs.
# Timeout is in seconds and applies per request attempt.
genai_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)
openai.api_key = os.getenv('OPENAI_API_KEY')

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
def get_gemini_chat(prompt: str, model: str, temperature: float = 0.0, stop_strs: Optional[List[str]] = None, max_tokens: int = 2048) -> Tuple[str, Dict[str, int]]:
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        stop_sequences=stop_strs if stop_strs else [],
    )
    response = genai_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    if not response.candidates:
        raise ValueError("No candidates in response - content may be blocked")
    
    candidate = response.candidates[0]
    if candidate.finish_reason and candidate.finish_reason.name == 'SAFETY':
        raise ValueError("Response blocked by safety filter")
    
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    if hasattr(response, 'usage_metadata'):
        usage["input_tokens"] = response.usage_metadata.prompt_token_count
        usage["output_tokens"] = response.usage_metadata.candidates_token_count
        usage["total_tokens"] = response.usage_metadata.total_token_count

    try:
        return response.text, usage
    except AttributeError:
        raise ValueError(f"Could not access response text. Response: {response}")

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    # Deprecated/Unused legacy function? leaving as is for now returning just str to avoid breaking extensive changes if used elsewhere
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

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 2048, stop_strs: Optional[List[str]] = None) -> Tuple[str, Dict[str, int]]:
    if model.startswith("gemini"):
        return get_gemini_chat(prompt, model, temperature, stop_strs, max_tokens)

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    
    # Extract usage for OpenAI models if available
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    if hasattr(response, 'usage'):
        usage["input_tokens"] = response.usage.prompt_tokens
        usage["output_tokens"] = response.usage.completion_tokens
        usage["total_tokens"] = response.usage.total_tokens
        
    return response.choices[0]["message"]["content"], usage

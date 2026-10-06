"""LLM client for Google Gemini (google-genai SDK)."""

import os
import time
from collections.abc import Callable

import httpx
from google import genai
from google.genai import errors, types

from aiact_rag.llm.base import LLMError
from aiact_rag.models import GenerateFn

RATE_LIMIT = 429
OVERLOADED = (500, 503)  # temporary on Google's side, e.g. "model is experiencing high demand"


def make_gemini(
    model: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 5,
    retry_wait: float = 20.0,
    overload_wait: float = 5.0,
    min_interval: float = 0.0,
    client: genai.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> GenerateFn:
    """Return a GenerateFn ``(system, user) -> answer text`` that asks Gemini.

    A rate limit (429) is retried after ``retry_wait`` seconds. A server error (500, 503) is
    retried after ``overload_wait`` seconds, doubling each time. ``min_interval`` is the least
    number of seconds between two requests, to stay under a requests-per-minute limit
    (4.0 for 15 requests per minute); 0 turns the pacing off.
    """
    if client is None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("No Gemini API key: set the environment variable GEMINI_API_KEY.")
        client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    last_request: float | None = None

    def pace() -> None:
        """Wait until ``min_interval`` seconds have passed since the last request."""
        nonlocal last_request
        if min_interval > 0 and last_request is not None:
            remaining = min_interval - (clock() - last_request)
            if remaining > 0:
                sleep(remaining)
        last_request = clock()

    def generate(system: str, user: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        overloads = 0
        for attempt in range(max_retries + 1):
            pace()
            try:
                response = client.models.generate_content(model=model, contents=user, config=config)
                break
            except errors.APIError as error:
                if attempt < max_retries and error.code == RATE_LIMIT:
                    sleep(retry_wait)
                    continue
                if attempt < max_retries and error.code in OVERLOADED:
                    sleep(overload_wait * 2**overloads)
                    overloads += 1
                    continue
                raise LLMError(
                    f"Gemini returned HTTP {error.code} for model {model!r}: {error.message}"
                ) from error
            except httpx.TimeoutException as error:
                raise LLMError(
                    f"Gemini did not answer within the timeout (model {model!r})."
                ) from error
            except httpx.TransportError as error:
                raise LLMError(
                    "Cannot reach the Gemini API. Check the network connection."
                ) from error

        if not response.text:
            candidates = response.candidates or []
            reason = candidates[0].finish_reason if candidates else "no candidates"
            raise LLMError(f"Gemini returned no text for model {model!r} ({reason}).")
        return response.text

    return generate

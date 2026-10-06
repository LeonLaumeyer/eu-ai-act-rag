"""LLM client for a local Ollama server."""

import httpx

from aiact_rag.llm.base import LLMError
from aiact_rag.models import GenerateFn


def make_ollama(
    model: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> GenerateFn:
    """Return a GenerateFn ``(system, user) -> answer text`` that asks Ollama."""
    if client is None:
        client = httpx.Client(timeout=timeout)

    def generate(system: str, user: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise LLMError(
                f"Ollama returned HTTP {error.response.status_code} for model {model!r}: "
                f"{error.response.text}"
            ) from error
        except httpx.TimeoutException as error:
            raise LLMError(
                f"Ollama at {base_url} did not answer within the timeout (model {model!r})."
            ) from error
        except httpx.TransportError as error:
            raise LLMError(
                f"Cannot reach Ollama at {base_url}. Is it running? Start it with 'ollama serve'."
            ) from error
        return response.json()["message"]["content"]

    return generate

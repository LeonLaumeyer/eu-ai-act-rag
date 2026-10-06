"""LLM client for Claude via the Anthropic API."""

import anthropic

from aiact_rag.llm.base import LLMError
from aiact_rag.models import GenerateFn


def make_claude(model: str = "claude-sonnet-5", max_tokens: int = 1024, client=None) -> GenerateFn:
    """Return a GenerateFn ``(system, user) -> answer text`` that asks Claude."""
    if client is None:
        client = anthropic.Anthropic()

    def generate_fn(system: str, user: str) -> str:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError as error:
            raise LLMError(f"Claude request failed for model {model}") from error
        return message.content[0].text

    return generate_fn

"""Shared error type of the LLM clients."""


class LLMError(RuntimeError):
    """The LLM could not be reached or returned an error."""

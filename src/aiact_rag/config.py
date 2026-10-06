"""Typed settings loaded from config.yaml."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from aiact_rag.models import ROOT
from aiact_rag.retrieve.hybrid import MODES

CHUNKINGS = ("article", "fixed")
PROVIDERS = ("ollama", "anthropic", "gemini")


def _check(key: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{key} must be one of {tuple(allowed)}, got {value!r}")


@dataclass(frozen=True)
class RetrievalConfig:
    chunking: str = "article"
    mode: str = "hybrid"
    rerank: bool = True
    top_k: int = 5

    def __post_init__(self) -> None:
        _check("retrieval.chunking", self.chunking, CHUNKINGS)
        _check("retrieval.mode", self.mode, MODES)
        if not isinstance(self.rerank, bool):
            raise ValueError(f"retrieval.rerank must be true or false, got {self.rerank!r}")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"retrieval.top_k must be a whole number >= 1, got {self.top_k!r}")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = ""
    base_url: str = "http://localhost:11434"  # only used by ollama
    max_tokens: int = 1024  # only used by anthropic
    min_interval: float = 0.0  # only used by gemini: least seconds between two requests

    def __post_init__(self) -> None:
        _check("llm.provider", self.provider, PROVIDERS)
        if not self.model:
            raise ValueError(f"llm.{self.provider}.model is missing")
        if (
            isinstance(self.min_interval, bool)
            or not isinstance(self.min_interval, int | float)
            or self.min_interval < 0
        ):
            raise ValueError(
                f"llm.{self.provider}.min_interval must be a number >= 0, got {self.min_interval!r}"
            )


@dataclass(frozen=True)
class Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="llama3.2"))


def llm_config(llm_section: dict, provider: str | None = None) -> LLMConfig:
    """Pick the block of one provider out of the ``llm`` section of the file."""
    provider = provider or llm_section.get("provider", "ollama")
    _check("llm.provider", provider, PROVIDERS)
    return LLMConfig(provider=provider, **(llm_section.get(provider) or {}))


def load_llm_config(provider: str | None = None, path: Path = ROOT / "config.yaml") -> LLMConfig:
    """The settings of one LLM provider from the file (None: the active one)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return llm_config(raw.get("llm") or {}, provider)


def load_config(path: Path = ROOT / "config.yaml") -> Config:
    """Read ``config.yaml``. Missing keys get the defaults of the dataclasses."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
        llm=llm_config(raw.get("llm") or {}),
    )

"""Shared data types."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Chunk:
    """One line of ``data/processed/chunks_{strategy}_{lang}.jsonl`` (written by chunk.py)."""

    chunk_id: str  # "en-article-art6-2", "en-article-art5-1#0", "de-fixed-17"
    lang: str  # "en" | "de"
    strategy: str  # "article" | "fixed"
    kind: str  # "article" | "annex" | "recital"
    unit_ids: list[str]  # ["art6-2"]; a fixed chunk covers several units
    citation: str  # "Art. 6(2)"
    text: str  # what the LLM gets to read
    embed_text: str  # header + text: what gets embedded, searched and reranked


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


EmbedFn = Callable[[list[str]], np.ndarray]
ScoreFn = Callable[[list[tuple[str, str]]], list[float]]
GenerateFn = Callable[[str, str], str]


def load_chunks(
    strategy: str, langs: Sequence[str] = ("en", "de"), processed_dir: Path = PROCESSED
) -> list[Chunk]:
    """Read the chunk files of one strategy, all ``langs`` in one list."""
    chunks = []
    for lang in langs:
        with open(processed_dir / f"chunks_{strategy}_{lang}.jsonl", encoding="utf-8") as f:
            chunks += [Chunk(**json.loads(line)) for line in f]
    return chunks

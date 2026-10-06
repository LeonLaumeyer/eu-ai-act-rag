"""Shared test helpers."""

import re
import zlib

import numpy as np

from aiact_rag.models import Chunk, SearchResult

DIM = 64


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedding: bag of words hashed into DIM buckets, normalized."""
    vectors = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % DIM] += 1.0
        if not vectors[row].any():
            vectors[row, 0] = 1.0
        vectors[row] /= np.linalg.norm(vectors[row])
    return vectors


class CountingEmbed:
    """``fake_embed`` that remembers every call: ``.calls`` is a list of text lists."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return fake_embed(texts)


def chunk(
    unit_id: str,
    text: str = "",
    lang: str = "en",
    kind: str = "article",
    citation: str | None = None,
    unit_ids: list[str] | None = None,
    piece: int | None = None,
) -> Chunk:
    """Short way to build a Chunk.

    ``chunk("art6-2", "some text")`` -> chunk_id ``en-article-art6-2``, unit_ids
    ``["art6-2"]``, citation ``C:art6-2``, embed_text ``[C:art6-2]\\nsome text``.
    ``piece=1`` adds ``#1`` to the chunk_id (a piece of a split unit).
    """
    citation = citation or f"C:{unit_id}"
    chunk_id = f"{lang}-article-{unit_id}" + ("" if piece is None else f"#{piece}")
    return Chunk(
        chunk_id=chunk_id,
        lang=lang,
        strategy="article",
        kind=kind,
        unit_ids=unit_ids or [unit_id],
        citation=citation,
        text=text,
        embed_text=f"[{citation}]\n{text}",
    )


def r(unit_id: str, score: float = 0.0, **kwargs) -> SearchResult:
    """Short way to build a SearchResult: ``r("art6-2", 0.9, text="...")``."""
    return SearchResult(chunk(unit_id, **kwargs), score)


def ids(results: list[SearchResult]) -> list[str]:
    """The chunk ids with the ``en-article-`` prefix removed: ``["art6-2", "art5-1#1"]``."""
    return [result.chunk.chunk_id.split("-", 2)[2] for result in results]


class FakeIndex:
    """Stands in for VectorStore / BM25Index: returns canned results, records calls."""

    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.calls: list[tuple] = []

    def search(self, query, lang, top_k=5, kinds=None) -> list[SearchResult]:
        self.calls.append((query, lang, top_k, kinds))
        return self.results[:top_k]


class FakeRetriever:
    """Stands in for Retriever: ``retrieve`` returns canned results, records calls."""

    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query: str, lang: str) -> list[SearchResult]:
        self.calls.append((query, lang))
        return self.results


class FakeLLM:
    """A GenerateFn that returns a canned answer and records ``(system, user)`` calls."""

    def __init__(self, answer: str = "", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.answer

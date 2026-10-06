"""BM25 keyword retrieval with per-language stemming."""

import re
from collections.abc import Sequence

import Stemmer
from rank_bm25 import BM25Okapi

from aiact_rag.models import Chunk, SearchResult

STEMMERS = {"en": Stemmer.Stemmer("english"), "de": Stemmer.Stemmer("german")}


def tokenize(text: str, lang: str) -> list[str]:
    """Lowercase ``text``, split it into words and stem every word."""
    if lang not in STEMMERS:
        raise ValueError(f"Language {lang!r} is not supported, expected one of {list(STEMMERS)}")
    return STEMMERS[lang].stemWords(re.findall(r"\w+", text.lower()))


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        """Build one ``BM25Okapi`` per language from ``chunk.embed_text``."""
        tokens: dict[str, list[list[str]]] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            tokens.setdefault(chunk.lang, []).append(tokenize(chunk.embed_text, chunk.lang))
            self.chunks.setdefault(chunk.lang, []).append(chunk)
        self.bm25 = {lang: BM25Okapi(corpus) for lang, corpus in tokens.items()}

    def search(
        self, query: str, lang: str, top_k: int = 5, kinds: Sequence[str] | None = None
    ) -> list[SearchResult]:
        """The ``top_k`` best chunks of language ``lang`` for ``query``, best first."""
        query_tokens = tokenize(query, lang)
        if lang not in self.bm25:
            return []
        scores = self.bm25[lang].get_scores(query_tokens)
        results = [
            SearchResult(chunk, float(score))
            for chunk, score in zip(self.chunks[lang], scores)
            if score > 0 and (kinds is None or chunk.kind in kinds)
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

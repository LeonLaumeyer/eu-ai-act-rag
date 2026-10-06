"""Retriever with dense, BM25 and hybrid (RRF) modes and optional reranking."""

from collections.abc import Sequence
from dataclasses import replace

from aiact_rag.models import ScoreFn, SearchResult
from aiact_rag.retrieve.rerank import rerank

MODES = ("dense", "bm25", "hybrid")


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Reciprocal Rank Fusion: a chunk scores the sum of 1 / (k + rank) over the lists."""
    scores: dict[str, float] = {}
    by_id: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            by_id[chunk_id] = result
    fused = [replace(result, score=scores[chunk_id]) for chunk_id, result in by_id.items()]
    return sorted(fused, key=lambda result: result.score, reverse=True)


def dedupe_by_unit(results: list[SearchResult]) -> list[SearchResult]:
    """Drop results that cover the same units as a better-ranked result."""
    seen = set()
    unique = []
    for result in results:
        key = tuple(result.chunk.unit_ids)
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique


class Retriever:
    """Retrieves with dense search, BM25, or both fused; then dedupes and optionally reranks."""

    def __init__(
        self,
        dense,
        sparse,
        mode: str = "hybrid",
        top_k: int = 5,
        candidate_k: int = 50,
        rerank_k: int = 30,
        rrf_k: int = 60,
        score_fn: ScoreFn | None = None,
        kinds: Sequence[str] | None = ("article", "annex"),
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Mode {mode!r} is not supported, expected one of {MODES}")
        self.dense = dense
        self.sparse = sparse
        self.mode = mode
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.score_fn = score_fn
        self.kinds = kinds

    def retrieve(self, query: str, lang: str) -> list[SearchResult]:
        """Return the best ``top_k`` results for ``query`` in language ``lang``."""
        if self.mode == "dense":
            candidates = self.dense.search(query, lang, self.candidate_k, self.kinds)
        elif self.mode == "bm25":
            candidates = self.sparse.search(query, lang, self.candidate_k, self.kinds)
        else:
            dense_results = self.dense.search(query, lang, self.candidate_k, self.kinds)
            sparse_results = self.sparse.search(query, lang, self.candidate_k, self.kinds)
            candidates = reciprocal_rank_fusion([dense_results, sparse_results], self.rrf_k)

        candidates = dedupe_by_unit(candidates)
        if self.score_fn is not None:
            return rerank(query, candidates[: self.rerank_k], self.score_fn, self.top_k)
        return candidates[: self.top_k]

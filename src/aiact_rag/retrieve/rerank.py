"""Cross-encoder reranking."""

from dataclasses import replace

from aiact_rag.models import ScoreFn, SearchResult


def make_cross_encoder(
    model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512
) -> ScoreFn:
    """Load a sentence-transformers ``CrossEncoder`` once and return a ScoreFn."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=max_length)

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        return model.predict(pairs).tolist()

    return score


def rerank(
    query: str, results: list[SearchResult], score_fn: ScoreFn, top_k: int
) -> list[SearchResult]:
    """Re-order ``results`` by ``score_fn`` and keep the best ``top_k``."""
    if not results:
        return []
    scores = score_fn([(query, result.chunk.embed_text) for result in results])
    rescored = [replace(result, score=float(score)) for result, score in zip(results, scores)]
    return sorted(rescored, key=lambda result: result.score, reverse=True)[:top_k]

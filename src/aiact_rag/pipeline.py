"""Builds retrievers, LLM clients and the answer function from a config."""

from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from aiact_rag.config import LLMConfig, RetrievalConfig
from aiact_rag.generate import Answer, answer_question
from aiact_rag.index.bm25 import BM25Index
from aiact_rag.index.embed import embed_cached, make_bge_m3_embedder
from aiact_rag.index.vector_store import VectorStore
from aiact_rag.llm.anthropic_llm import make_claude
from aiact_rag.llm.gemini import make_gemini
from aiact_rag.llm.ollama import make_ollama
from aiact_rag.models import PROCESSED, ROOT, EmbedFn, GenerateFn, ScoreFn, load_chunks
from aiact_rag.retrieve.hybrid import Retriever
from aiact_rag.retrieve.rerank import make_cross_encoder

CACHE = ROOT / "data" / "embeddings"


class Resources:
    """The slow things, created on first use and then reused."""

    def __init__(
        self,
        embed: EmbedFn | None = None,
        score_fn: ScoreFn | None = None,
        processed_dir: Path = PROCESSED,
        cache_dir: Path = CACHE,
    ) -> None:
        self._embed = embed
        self._score_fn = score_fn
        self.processed_dir = processed_dir
        self.cache_dir = cache_dir
        self._client = QdrantClient(":memory:")
        self._indexes: dict[str, tuple[VectorStore, BM25Index]] = {}

    @cached_property
    def embed(self) -> EmbedFn:
        return self._embed or make_bge_m3_embedder()

    @cached_property
    def score_fn(self) -> ScoreFn:
        return self._score_fn or make_cross_encoder()

    def indexes(self, chunking: str) -> tuple[VectorStore, BM25Index]:
        """The dense and the keyword index over the chunks of one strategy."""
        if chunking not in self._indexes:
            chunks = load_chunks(chunking, processed_dir=self.processed_dir)
            vectors = np.asarray(
                embed_cached(
                    self.embed,
                    [c.embed_text for c in chunks],
                    self.cache_dir / f"{chunking}.npz",
                )
            )
            store = VectorStore(self._client, f"aiact_{chunking}", self.embed, dim=vectors.shape[1])
            store.build(chunks, vectors)
            self._indexes[chunking] = (store, BM25Index(chunks))
        return self._indexes[chunking]


def build_retriever(
    config: RetrievalConfig, resources: Resources, top_k: int | None = None
) -> Retriever:
    """A ``Retriever`` for one config; ``top_k`` overrides ``config.top_k``."""
    dense, sparse = resources.indexes(config.chunking)
    return Retriever(
        dense,
        sparse,
        mode=config.mode,
        top_k=top_k or config.top_k,
        score_fn=resources.score_fn if config.rerank else None,
    )


def build_llm(config: LLMConfig) -> GenerateFn:
    load_dotenv(ROOT / ".env")
    if config.provider == "gemini":
        return make_gemini(model=config.model, min_interval=config.min_interval)
    if config.provider == "anthropic":
        return make_claude(model=config.model, max_tokens=config.max_tokens)
    return make_ollama(model=config.model, base_url=config.base_url)


def build_answer_fn(retriever, generate: GenerateFn) -> Callable[[str, str], Answer]:
    """Bind a retriever and an LLM into ``answer_fn(question, lang) -> Answer``."""

    def answer_fn(question: str, lang: str) -> Answer:
        return answer_question(question, lang, retriever, generate)

    return answer_fn

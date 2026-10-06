"""Dense retrieval with Qdrant, filtered by language and kind."""

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
from qdrant_client import QdrantClient, models

from aiact_rag.models import Chunk, EmbedFn, SearchResult


class VectorStore:
    def __init__(
        self, client: QdrantClient, collection: str, embed: EmbedFn, dim: int = 1024
    ) -> None:
        self.client = client
        self.collection = collection
        self.embed = embed
        self.dim = dim

    def build(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        """(Re)create the collection and upload all chunks; embeds them if no vectors are given."""
        if vectors is not None and len(chunks) != len(vectors):
            raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        if vectors is None:
            vectors = self.embed([chunk.embed_text for chunk in chunks])
        points = [
            models.PointStruct(id=i, vector=vector.tolist(), payload=asdict(chunk))
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection, wait=True, points=points)

    def count(self) -> int:
        """Number of points in the collection."""
        return self.client.count(collection_name=self.collection, exact=True).count

    def search(
        self, query: str, lang: str, top_k: int = 5, kinds: Sequence[str] | None = None
    ) -> list[SearchResult]:
        """The ``top_k`` chunks of language ``lang`` most similar to ``query``, best first."""
        conditions = [models.FieldCondition(key="lang", match=models.MatchValue(value=lang))]
        if kinds is not None:
            conditions.append(
                models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds)))
            )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embed([query])[0].tolist(),
            query_filter=models.Filter(must=conditions),
            with_payload=True,
            limit=top_k,
        ).points
        return [SearchResult(chunk=Chunk(**hit.payload), score=hit.score) for hit in hits]

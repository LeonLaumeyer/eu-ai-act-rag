"""bge-m3 embeddings with an on-disk cache."""

import hashlib
from pathlib import Path

import numpy as np

from aiact_rag.models import EmbedFn


def make_bge_m3_embedder(
    model_name: str = "BAAI/bge-m3", batch_size: int = 16, max_length: int = 1024
) -> EmbedFn:
    """Load bge-m3 once and return an EmbedFn that uses it."""
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_name, use_fp16=False)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, max_length=max_length)["dense_vecs"]

    return embed


def embed_cached(embed: EmbedFn, texts: list[str], cache_path: Path) -> np.ndarray:
    """Return the vectors for ``texts``, from the cache file if it holds exactly these texts."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    key = digest.hexdigest()

    if cache_path.is_file():
        with np.load(cache_path) as data:
            if key in data.files:
                return data[key]

    vectors = embed(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **{key: vectors})
    return vectors

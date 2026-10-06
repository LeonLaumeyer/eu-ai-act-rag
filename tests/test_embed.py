import numpy as np
import pytest

from aiact_rag.index.embed import embed_cached, make_bge_m3_embedder
from helpers import DIM, CountingEmbed, fake_embed

TEXTS = ["high-risk AI systems", "prohibited practices", "transparency obligations"]


class TestEmbedCached:
    def test_first_call_embeds_and_saves(self, tmp_path):
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        vectors = embed_cached(embed, TEXTS, path)
        assert embed.calls == [TEXTS]
        assert vectors.shape == (3, DIM)
        np.testing.assert_allclose(vectors, fake_embed(TEXTS))
        assert path.exists()

    def test_second_call_uses_the_cache(self, tmp_path):
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        first = embed_cached(embed, TEXTS, path)
        second = embed_cached(embed, TEXTS, path)
        assert len(embed.calls) == 1
        np.testing.assert_allclose(second, first)

    def test_cache_survives_a_new_embed_function(self, tmp_path):
        path = tmp_path / "vectors.npz"
        embed_cached(CountingEmbed(), TEXTS, path)
        fresh = CountingEmbed()
        embed_cached(fresh, TEXTS, path)
        assert fresh.calls == []

    def test_changed_text_same_count_embeds_again(self, tmp_path):
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        embed_cached(embed, TEXTS, path)
        changed = TEXTS[:2] + ["something else entirely"]
        vectors = embed_cached(embed, changed, path)
        assert len(embed.calls) == 2
        np.testing.assert_allclose(vectors, fake_embed(changed))

    def test_changed_order_embeds_again(self, tmp_path):
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        embed_cached(embed, TEXTS, path)
        vectors = embed_cached(embed, TEXTS[::-1], path)
        assert len(embed.calls) == 2
        np.testing.assert_allclose(vectors, fake_embed(TEXTS[::-1]))

    def test_cache_is_updated_after_a_change(self, tmp_path):
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        changed = ["a new text"]
        embed_cached(embed, TEXTS, path)
        embed_cached(embed, changed, path)
        embed_cached(embed, changed, path)
        assert len(embed.calls) == 2

    def test_creates_missing_folders(self, tmp_path):
        path = tmp_path / "data" / "embeddings" / "vectors.npz"
        embed_cached(CountingEmbed(), TEXTS, path)
        assert path.exists()

    def test_texts_that_only_differ_in_where_they_are_split(self, tmp_path):
        # ["ab", "c"] and ["a", "bc"] must not produce the same cache key.
        embed, path = CountingEmbed(), tmp_path / "vectors.npz"
        embed_cached(embed, ["ab", "c"], path)
        embed_cached(embed, ["a", "bc"], path)
        assert len(embed.calls) == 2


@pytest.mark.slow
class TestBgeM3:
    def test_shape_and_meaning_across_languages(self):
        embed = make_bge_m3_embedder()
        vectors = embed(
            [
                "Which AI practices are prohibited?",
                "Welche KI-Praktiken sind verboten?",
                "The committee meets twice a year in Brussels.",
            ]
        )
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 1024)
        question, german, unrelated = vectors
        assert question @ german > question @ unrelated

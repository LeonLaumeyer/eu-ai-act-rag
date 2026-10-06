import pytest
from qdrant_client import QdrantClient

from aiact_rag.index.vector_store import VectorStore
from aiact_rag.models import SearchResult
from helpers import DIM, CountingEmbed, chunk, fake_embed, ids

CHUNKS = [
    chunk("art5-1", "prohibited AI practices manipulation social scoring"),
    chunk("art6-2", "high-risk AI systems listed in the annex"),
    chunk("art50-1", "transparency obligations for providers of chatbots"),
    chunk("anx3-4", "high-risk employment recruitment systems", kind="annex"),
    chunk("rct27", "high-risk systems trustworthy human oversight recital", kind="recital"),
    chunk("art6-2", "Hochrisiko KI Systeme im Anhang", lang="de"),
    chunk("art5-1", "verbotene KI Praktiken Manipulation", lang="de"),
]


@pytest.fixture
def embed():
    return CountingEmbed()


@pytest.fixture
def store(embed):
    s = VectorStore(QdrantClient(":memory:"), "test", embed, dim=DIM)
    s.build(CHUNKS)
    return s


class TestBuild:
    def test_count(self, store):
        assert store.count() == len(CHUNKS)

    def test_building_twice_leaves_no_duplicates(self, store):
        store.build(CHUNKS)
        assert store.count() == len(CHUNKS)

    def test_rebuild_replaces_old_chunks(self, store):
        store.build(CHUNKS[:2])
        assert store.count() == 2

    def test_embeds_embed_text_in_one_call(self, embed, store):
        assert embed.calls == [[c.embed_text for c in CHUNKS]]

    def test_given_vectors_are_used(self, embed):
        s = VectorStore(QdrantClient(":memory:"), "test", embed, dim=DIM)
        s.build(CHUNKS, vectors=fake_embed([c.embed_text for c in CHUNKS]))
        assert embed.calls == []
        assert s.count() == len(CHUNKS)

    def test_given_vectors_decide_the_ranking(self, embed):
        # Deliberately "wrong" vectors: chunk 0 gets the vector of the query.
        vectors = fake_embed(["zebra"] + ["unrelated filler"] * (len(CHUNKS) - 1))
        s = VectorStore(QdrantClient(":memory:"), "test", embed, dim=DIM)
        s.build(CHUNKS, vectors=vectors)
        assert ids(s.search("zebra", "en", top_k=1)) == ["art5-1"]

    def test_wrong_number_of_vectors(self, embed):
        s = VectorStore(QdrantClient(":memory:"), "test", embed, dim=DIM)
        with pytest.raises(ValueError):
            s.build(CHUNKS, vectors=fake_embed(["only one"]))

    def test_wrong_number_of_vectors_keeps_the_old_collection(self, store):
        with pytest.raises(ValueError):
            store.build(CHUNKS[:2], vectors=fake_embed(["only one"]))
        assert store.count() == len(CHUNKS)


class TestSearch:
    def test_best_match_first(self, store):
        results = store.search("prohibited practices", "en", top_k=3)
        assert ids(results)[0] == "art5-1"
        assert len(results) == 3

    def test_returns_search_results_with_full_chunk(self, store):
        result = store.search("transparency obligations chatbots", "en", top_k=1)[0]
        assert isinstance(result, SearchResult)
        assert result.chunk == CHUNKS[2]
        assert isinstance(result.score, float)

    def test_scores_descend(self, store):
        scores = [x.score for x in store.search("high-risk systems", "en", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_only_the_asked_language(self, store):
        assert {x.chunk.lang for x in store.search("high-risk KI Systeme", "en", top_k=10)} == {
            "en"
        }
        german = store.search("Hochrisiko KI Systeme", "de", top_k=10)
        assert {x.chunk.lang for x in german} == {"de"}
        assert len(german) == 2

    def test_same_unit_in_two_languages_is_two_chunks(self, store):
        assert (
            store.search("verbotene Praktiken", "de", top_k=1)[0].chunk.chunk_id
            == "de-article-art5-1"
        )

    def test_kinds_none_returns_all_kinds(self, store):
        kinds = {x.chunk.kind for x in store.search("high-risk systems", "en", top_k=10)}
        assert kinds == {"article", "annex", "recital"}

    def test_kinds_filter(self, store):
        results = store.search("high-risk systems", "en", top_k=10, kinds=("article", "annex"))
        assert {x.chunk.kind for x in results} == {"article", "annex"}
        assert len(results) == 4

    def test_single_kind(self, store):
        results = store.search("high-risk systems", "en", top_k=10, kinds=["recital"])
        assert ids(results) == ["rct27"]

    def test_top_k(self, store):
        assert len(store.search("AI", "en", top_k=2)) == 2

    def test_query_is_embedded_as_one_text(self, embed, store):
        embed.calls.clear()
        store.search("prohibited practices", "en")
        assert embed.calls == [["prohibited practices"]]

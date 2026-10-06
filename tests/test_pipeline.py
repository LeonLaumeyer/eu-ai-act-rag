import json
from dataclasses import asdict

import pytest

from aiact_rag.config import LLMConfig, RetrievalConfig
from aiact_rag.generate import Answer
from aiact_rag.llm.base import LLMError
from aiact_rag.pipeline import Resources, build_answer_fn, build_llm, build_retriever
from helpers import CountingEmbed, FakeLLM, FakeRetriever, chunk, ids, r

ARTICLE_CHUNKS = [
    chunk("art5-1", "prohibited AI practices such as social scoring"),
    chunk("art6-2", "systems listed in Annex III are high-risk"),
    chunk("art99-3", "administrative fines of up to 35 000 000 EUR"),
    chunk("art50-1", "chatbots must inform natural persons"),
    chunk("art113", "entry into force and application dates"),
    chunk("rct1", "a recital about administrative fines", kind="recital"),
    chunk("art99-3", "Geldbußen von bis zu 35 000 000 EUR", lang="de"),
    chunk("art5-1", "verbotene KI Praktiken wie soziale Bewertung", lang="de"),
    # BM25 gives a word that is in half of the documents a weight of 0, so the
    # German side needs more than two chunks for a keyword to count.
    chunk("art6-2", "Systeme in Anhang III gelten als hochriskant", lang="de"),
    chunk("art113", "Inkrafttreten und Geltungsbeginn", lang="de"),
]


def write_chunks(folder, strategy, chunks):
    for lang in ("en", "de"):
        with open(folder / f"chunks_{strategy}_{lang}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                if c.lang == lang:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


@pytest.fixture
def embed():
    return CountingEmbed()


@pytest.fixture
def resources(tmp_path, embed):
    processed = tmp_path / "processed"
    processed.mkdir()
    write_chunks(processed, "article", ARTICLE_CHUNKS)
    write_chunks(processed, "fixed", ARTICLE_CHUNKS[:3])
    return Resources(
        embed=embed,
        score_fn=lambda pairs: [float(len(text)) for _, text in pairs],
        processed_dir=processed,
        cache_dir=tmp_path / "cache",
    )


class TestResources:
    def test_indexes_hold_all_chunks(self, resources):
        store, _ = resources.indexes("article")
        assert store.count() == len(ARTICLE_CHUNKS)

    def test_chunks_are_embedded_once_per_strategy(self, resources, embed):
        resources.indexes("article")
        resources.indexes("article")
        assert len(embed.calls) == 1
        assert len(embed.calls[0]) == len(ARTICLE_CHUNKS)

    def test_same_objects_are_returned(self, resources):
        assert resources.indexes("article") is resources.indexes("article")

    def test_strategies_are_separate(self, resources):
        article, _ = resources.indexes("article")
        fixed, _ = resources.indexes("fixed")
        assert (article.count(), fixed.count()) == (len(ARTICLE_CHUNKS), 3)

    def test_vectors_come_from_the_cache_in_a_new_run(self, resources, tmp_path):
        resources.indexes("article")
        fresh_embed = CountingEmbed()
        fresh = Resources(
            embed=fresh_embed,
            processed_dir=resources.processed_dir,
            cache_dir=tmp_path / "cache",
        )
        fresh.indexes("article")
        assert fresh_embed.calls == []

    def test_given_models_are_used(self, resources, embed):
        assert resources.embed is embed
        assert resources.score_fn([("q", "abc")]) == [3.0]


class TestBuildRetriever:
    def test_bm25(self, resources):
        retriever = build_retriever(RetrievalConfig(mode="bm25", rerank=False), resources)
        assert ids(retriever.retrieve("fines", "en")) == ["art99-3"]

    def test_language(self, resources):
        retriever = build_retriever(RetrievalConfig(mode="bm25", rerank=False), resources)
        found = retriever.retrieve("Geldbußen", "de")
        assert [x.chunk.chunk_id for x in found] == ["de-article-art99-3"]

    def test_dense_and_hybrid_find_the_obvious_chunk(self, resources):
        for mode in ("dense", "hybrid"):
            retriever = build_retriever(RetrievalConfig(mode=mode, rerank=False), resources)
            assert ids(retriever.retrieve("administrative fines EUR", "en"))[0] == "art99-3"

    def test_recitals_are_left_out(self, resources):
        retriever = build_retriever(RetrievalConfig(mode="bm25", rerank=False, top_k=5), resources)
        assert "rct1" not in ids(retriever.retrieve("administrative fines", "en"))

    def test_top_k_from_config_and_override(self, resources):
        config = RetrievalConfig(mode="dense", rerank=False, top_k=2)
        assert len(build_retriever(config, resources).retrieve("AI", "en")) == 2
        assert len(build_retriever(config, resources, top_k=4).retrieve("AI", "en")) == 4

    def test_rerank_setting_decides_if_the_reranker_is_used(self, resources):
        without = build_retriever(RetrievalConfig(mode="dense", rerank=False), resources)
        with_reranker = build_retriever(RetrievalConfig(mode="dense", rerank=True), resources)
        assert without.score_fn is None
        # the fake reranker scores by text length, so the scores become lengths
        results = with_reranker.retrieve("AI", "en")
        assert results[0].score == max(
            len(c.embed_text) for c in ARTICLE_CHUNKS if c.lang == "en" and c.kind != "recital"
        )

    def test_fixed_chunking_uses_the_other_file(self, resources):
        retriever = build_retriever(
            RetrievalConfig(chunking="fixed", mode="dense", rerank=False, top_k=10),
            resources,
        )
        assert len(retriever.retrieve("AI", "en")) == 3

    def test_no_reranker_is_loaded_without_rerank(self, tmp_path, embed):
        processed = tmp_path / "processed"
        processed.mkdir()
        write_chunks(processed, "article", ARTICLE_CHUNKS)
        resources = Resources(embed=embed, processed_dir=processed, cache_dir=tmp_path / "cache")
        build_retriever(RetrievalConfig(rerank=False), resources)
        assert "score_fn" not in vars(resources)  # the cached_property was never touched


class TestBuildLLM:
    def test_ollama(self):
        generate = build_llm(
            LLMConfig(provider="ollama", model="gemma", base_url="http://127.0.0.1:1")
        )
        with pytest.raises(LLMError, match="127.0.0.1:1"):  # nothing listens on port 1
            generate("s", "u")

    def test_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
        assert callable(build_llm(LLMConfig(provider="anthropic", model="claude-test")))


class TestBuildAnswerFn:
    def test_answers_with_retriever_and_llm(self):
        sources = [
            r(
                "art6-2",
                0.9,
                citation="Art. 6(2)",
                text="Systems in Annex III are high-risk.",
            )
        ]
        retriever, llm = FakeRetriever(sources), FakeLLM("High-risk [Art. 6(2)].")
        answer = build_answer_fn(retriever, llm)("Is it high-risk?", "en")
        assert isinstance(answer, Answer)
        assert answer.valid_citations == ["Art. 6(2)"]
        assert retriever.calls == [("Is it high-risk?", "en")]
        assert len(llm.calls) == 1

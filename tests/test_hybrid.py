import pytest

from aiact_rag.index.bm25 import BM25Index
from aiact_rag.retrieve.hybrid import Retriever, dedupe_by_unit, reciprocal_rank_fusion
from helpers import FakeIndex, chunk, ids, r

DENSE = [
    r("a", 0.9, text="text a"),
    r("b", 0.8, text="text b"),
    r("c", 0.7, text="text c"),
]
SPARSE = [
    r("c", 12.0, text="text c"),
    r("a", 7.5, text="text a"),
    r("d", 3.0, text="text d"),
]


class TestRRF:
    def test_single_list_keeps_order(self):
        fused = reciprocal_rank_fusion([DENSE], k=60)
        assert ids(fused) == ["a", "b", "c"]
        assert [x.score for x in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])

    def test_two_lists(self):
        fused = reciprocal_rank_fusion([DENSE, SPARSE], k=60)
        assert ids(fused) == ["a", "c", "b", "d"]
        assert [x.score for x in fused] == pytest.approx(
            [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
        )

    def test_keeps_the_chunk(self):
        fused = reciprocal_rank_fusion([DENSE, SPARSE])
        assert fused[0].chunk == DENSE[0].chunk

    def test_same_unit_in_two_languages_is_not_the_same_chunk(self):
        fused = reciprocal_rank_fusion([[r("a")], [r("a", lang="de")]])
        assert len(fused) == 2

    def test_ties_first_seen_wins(self):
        a, b = [r("a", 0.1)], [r("b", 0.9)]
        assert ids(reciprocal_rank_fusion([a, b])) == ["a", "b"]
        assert ids(reciprocal_rank_fusion([b, a])) == ["b", "a"]

    @pytest.mark.parametrize("lists", [[], [[]], [[], []]])
    def test_empty(self, lists):
        assert reciprocal_rank_fusion(lists) == []

    def test_does_not_modify_inputs(self):
        dense, sparse = list(DENSE), list(SPARSE)
        reciprocal_rank_fusion([dense, sparse])
        assert dense == DENSE and sparse == SPARSE


class TestDedupe:
    def test_pieces_of_one_unit(self):
        results = [
            r("art5-1", 0.9, piece=2),
            r("art6-2", 0.8),
            r("art5-1", 0.7, piece=0),
        ]
        assert ids(dedupe_by_unit(results)) == ["art5-1#2", "art6-2"]

    def test_no_duplicates_changes_nothing(self):
        assert dedupe_by_unit(DENSE) == DENSE

    def test_partial_overlap_is_not_a_duplicate(self):
        results = [
            r("x", unit_ids=["art5-1", "art5-2"]),
            r("y", unit_ids=["art5-2", "art5-3"]),
            r("z", unit_ids=["art5-1", "art5-2"]),
        ]
        assert ids(dedupe_by_unit(results)) == ["x", "y"]

    def test_empty(self):
        assert dedupe_by_unit([]) == []

    def test_does_not_modify_input(self):
        results = [r("a", piece=0), r("a", piece=1)]
        dedupe_by_unit(results)
        assert len(results) == 2


class TestRetriever:
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            Retriever(FakeIndex([]), FakeIndex([]), mode="magic")

    def test_dense_mode_only_uses_dense(self):
        dense, sparse = FakeIndex(DENSE), FakeIndex(SPARSE)
        out = Retriever(dense, sparse, mode="dense", top_k=2).retrieve("q", "en")
        assert ids(out) == ["a", "b"]
        assert sparse.calls == []

    def test_bm25_mode_only_uses_sparse(self):
        dense, sparse = FakeIndex(DENSE), FakeIndex(SPARSE)
        out = Retriever(dense, sparse, mode="bm25", top_k=2).retrieve("q", "en")
        assert ids(out) == ["c", "a"]
        assert dense.calls == []

    def test_hybrid_fuses_both(self):
        out = Retriever(FakeIndex(DENSE), FakeIndex(SPARSE), mode="hybrid", top_k=3).retrieve(
            "q", "en"
        )
        assert ids(out) == ["a", "c", "b"]
        assert out[0].score == pytest.approx(1 / 61 + 1 / 62)

    def test_hybrid_is_the_default_mode(self):
        out = Retriever(FakeIndex(DENSE), FakeIndex(SPARSE)).retrieve("q", "en")
        assert ids(out) == ["a", "c", "b", "d"]

    def test_search_arguments_are_passed_on(self):
        dense, sparse = FakeIndex(DENSE), FakeIndex(SPARSE)
        Retriever(dense, sparse, top_k=2, candidate_k=17, kinds=("annex",)).retrieve("Frage", "de")
        assert dense.calls == [("Frage", "de", 17, ("annex",))]
        assert sparse.calls == [("Frage", "de", 17, ("annex",))]

    def test_default_kinds_leave_out_recitals(self):
        dense = FakeIndex(DENSE)
        Retriever(dense, FakeIndex([]), mode="dense").retrieve("q", "en")
        assert set(dense.calls[0][3]) == {"article", "annex"}

    def test_rrf_k_is_used(self):
        out = Retriever(FakeIndex(DENSE), FakeIndex(SPARSE), rrf_k=1).retrieve("q", "en")
        assert out[0].score == pytest.approx(1 / 2 + 1 / 3)

    def test_dedupes_before_cutting_to_top_k(self):
        dense = FakeIndex([r("art5-1", 0.9, piece=0), r("art5-1", 0.8, piece=1), r("art6-2", 0.7)])
        out = Retriever(dense, FakeIndex([]), mode="dense", top_k=2).retrieve("q", "en")
        assert ids(out) == ["art5-1#0", "art6-2"]

    def test_reranker_reorders_and_cuts(self):
        order = {"text a": 1.0, "text b": 3.0, "text c": 2.0}

        def score_fn(pairs):
            return [order[text.split("\n")[1]] for _, text in pairs]

        out = Retriever(
            FakeIndex(DENSE), FakeIndex([]), mode="dense", top_k=2, score_fn=score_fn
        ).retrieve("q", "en")
        assert ids(out) == ["b", "c"]
        assert [x.score for x in out] == [3.0, 2.0]

    def test_reranker_only_sees_rerank_k_candidates(self):
        seen = []

        def spy(pairs):
            seen.append(len(pairs))
            return [float(i) for i in range(len(pairs))]

        out = Retriever(
            FakeIndex(DENSE), FakeIndex(SPARSE), top_k=1, rerank_k=2, score_fn=spy
        ).retrieve("q", "en")
        assert seen == [2]
        assert ids(out) == ["c"]  # fused order is a, c, b, d; the spy prefers the later one

    def test_reranker_gets_the_query(self):
        seen = []

        def spy(pairs):
            seen.extend(query for query, _ in pairs)
            return [0.0] * len(pairs)

        Retriever(FakeIndex(DENSE), FakeIndex([]), mode="dense", score_fn=spy).retrieve(
            "my question", "en"
        )
        assert set(seen) == {"my question"}

    def test_nothing_found(self):
        def boom(pairs):
            raise AssertionError("nothing to rerank")

        assert Retriever(FakeIndex([]), FakeIndex([]), score_fn=boom).retrieve("q", "en") == []

    def test_with_a_real_bm25_index(self):
        chunks = [
            chunk("art99-3", "administrative fines of up to 35 000 000 EUR"),
            chunk("art5-1", "prohibited practices such as social scoring"),
            chunk("art6-2", "systems in Annex III are high-risk"),
            chunk("art113", "entry into force and application"),
            chunk(
                "rct1",
                "administrative fines are mentioned in this recital",
                kind="recital",
            ),
        ]
        retriever = Retriever(FakeIndex([]), BM25Index(chunks), mode="bm25")
        assert ids(retriever.retrieve("fines", "en")) == ["art99-3"]

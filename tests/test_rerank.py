import pytest

from aiact_rag.retrieve.rerank import make_cross_encoder, rerank
from helpers import ids, r

RESULTS = [
    r("art1-1", 0.9, text="scope of the regulation"),
    r("art5-1", 0.8, text="prohibited practices"),
    r("art6-2", 0.7, text="high-risk systems"),
]


def length_score(pairs):
    """Fake ScoreFn: longer text = higher score."""
    return [float(len(text)) for _, text in pairs]


class TestRerank:
    def test_reorders_by_new_score(self):
        scores = {"art1-1": 0.1, "art5-1": 0.9, "art6-2": 0.5}
        out = rerank(
            "q",
            RESULTS,
            lambda pairs: [scores[t.split("]")[0][3:]] for _, t in pairs],
            top_k=3,
        )
        assert ids(out) == ["art5-1", "art6-2", "art1-1"]
        assert [x.score for x in out] == [0.9, 0.5, 0.1]

    def test_top_k(self):
        out = rerank("q", RESULTS, length_score, top_k=2)
        assert len(out) == 2

    def test_scores_query_with_embed_text(self):
        seen = []

        def spy(pairs):
            seen.append(pairs)
            return [0.0] * len(pairs)

        rerank("my question", RESULTS, spy, top_k=3)
        assert seen == [[("my question", x.chunk.embed_text) for x in RESULTS]]

    def test_equal_scores_keep_input_order(self):
        out = rerank("q", RESULTS, lambda pairs: [1.0] * len(pairs), top_k=3)
        assert ids(out) == ["art1-1", "art5-1", "art6-2"]

    def test_scores_are_floats(self):
        import numpy as np

        out = rerank("q", RESULTS, lambda pairs: np.array([3, 1, 2]), top_k=3)
        assert all(type(x.score) is float for x in out)
        assert ids(out) == ["art1-1", "art6-2", "art5-1"]

    def test_empty_does_not_call_score_fn(self):
        def boom(pairs):
            raise AssertionError("score_fn must not be called for empty results")

        assert rerank("q", [], boom, top_k=5) == []

    def test_does_not_modify_input(self):
        before = [(x.chunk.chunk_id, x.score) for x in RESULTS]
        results = list(RESULTS)
        rerank("q", results, length_score, top_k=1)
        assert [(x.chunk.chunk_id, x.score) for x in results] == before


@pytest.mark.slow
class TestCrossEncoder:
    def test_scores_relevant_text_higher_in_both_languages(self):
        score = make_cross_encoder()
        assert score([]) == []
        scores = score(
            [
                (
                    "Which AI practices are prohibited?",
                    "The following AI practices shall be prohibited: social scoring.",
                ),
                (
                    "Which AI practices are prohibited?",
                    "This Regulation shall enter into force after publication.",
                ),
                (
                    "Welche KI-Praktiken sind verboten?",
                    "Die folgenden KI-Praktiken sind verboten: soziale Bewertung.",
                ),
                (
                    "Welche KI-Praktiken sind verboten?",
                    "Diese Verordnung tritt nach ihrer Veröffentlichung in Kraft.",
                ),
            ]
        )
        assert all(type(s) is float for s in scores)
        assert scores[0] > scores[1]
        assert scores[2] > scores[3]

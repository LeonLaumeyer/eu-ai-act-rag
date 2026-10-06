import pytest

from aiact_rag.evaluate import Question
from aiact_rag.retrieval_eval import (
    BASELINE,
    SETTINGS,
    Setting,
    best_setting,
    breakdown_table,
    evaluate_setting,
    failures_markdown,
    load_known_units,
    report,
    results_table,
    summarize,
)
from helpers import r

QUESTIONS = [
    Question("q1", "en", "Which practices are banned?", ["art5-1"], True, "factual"),
    Question("q2", "de", "Was ist ein Betreiber?", ["art3-p4"], True, "definition"),
    Question(
        "q3",
        "en",
        "Is hiring AI high-risk?",
        ["anx3-4", "art6-2"],
        True,
        "cross_reference",
    ),
    Question("q4", "en", "What about crypto?", [], False, "trick"),
]


class CannedRetriever:
    """Returns the results prepared for each question text."""

    def __init__(self, by_question):
        self.by_question = by_question
        self.calls = []

    def retrieve(self, query, lang):
        self.calls.append((query, lang))
        return self.by_question.get(query, [])


GOOD = CannedRetriever(
    {
        "Which practices are banned?": [r("art5-1", citation="Art. 5(1)"), r("art5-2")],
        "Was ist ein Betreiber?": [r("art3-p3"), r("art3-p4", citation="Art. 3 Nr. 4")],
        "Is hiring AI high-risk?": [r("art6-2"), r("art1-1"), r("anx3-4")],
    }
)
BAD = CannedRetriever(
    {
        "Which practices are banned?": [r("art1-1"), r("art5-2", citation="Art. 5(2)")],
        "Was ist ein Betreiber?": [],
        "Is hiring AI high-risk?": [r("art1-1"), r("art6-2")],
    }
)


@pytest.fixture
def good_rows():
    return evaluate_setting("good", GOOD, QUESTIONS)


@pytest.fixture
def bad_rows():
    return evaluate_setting("bad", BAD, QUESTIONS)


class TestSettings:
    def test_names(self):
        assert Setting("article", "hybrid", True).name == "article + hybrid + rerank"
        assert BASELINE.name == "fixed + dense"

    def test_grid(self):
        assert len(SETTINGS) == 7
        assert len({s.name for s in SETTINGS}) == 7
        assert sum(s.rerank for s in SETTINGS) == 3
        assert [s for s in SETTINGS if s.chunking == "fixed"] == [BASELINE]


class TestEvaluateSetting:
    def test_unanswerable_questions_are_left_out(self, good_rows):
        assert [row.question_id for row in good_rows] == ["q1", "q2", "q3"]
        assert ("What about crypto?", "en") not in GOOD.calls

    def test_question_and_language_go_to_the_retriever(self):
        retriever = CannedRetriever({})
        evaluate_setting("x", retriever, QUESTIONS)
        assert retriever.calls[:2] == [
            ("Which practices are banned?", "en"),
            ("Was ist ein Betreiber?", "de"),
        ]

    def test_metrics_of_single_questions(self, good_rows):
        q1, q2, q3 = good_rows
        assert (q1.hit1, q1.hit5, q1.rr10) == (1.0, 1.0, 1.0)
        assert (q2.hit1, q2.hit5, q2.rr10) == (0.0, 1.0, 0.5)
        assert (q3.hit1, q3.recall5) == (1.0, 1.0)

    def test_article_level_and_recall(self, bad_rows):
        q1, q2, q3 = bad_rows
        assert (q1.hit5, q1.hit5_article) == (0.0, 1.0)  # art5-2 instead of art5-1
        assert (q2.hit5, q2.hit5_article, q2.rr10) == (0.0, 0.0, 0.0)
        assert (q3.hit5, q3.recall5, q3.rr10) == (1.0, 0.5, 0.5)

    def test_row_keeps_what_is_needed_to_look_at_a_case(self, bad_rows):
        row = bad_rows[0]
        assert (row.setting, row.lang, row.type) == ("bad", "en", "factual")
        assert row.gold == ["art5-1"]
        assert row.top5 == ["C:art1-1", "Art. 5(2)"]
        assert row.top5_units == [["art1-1"], ["art5-2"]]
        assert row.seconds >= 0


class TestSummarize:
    def test_means(self, good_rows, bad_rows):
        good, bad = summarize(good_rows), summarize(bad_rows)
        assert good["n"] == 3
        assert good["hit1"] == pytest.approx(2 / 3)
        assert good["hit5"] == 1.0
        assert good["mrr10"] == pytest.approx((1 + 0.5 + 1) / 3)
        assert bad["hit5"] == pytest.approx(1 / 3)
        assert bad["hit5_article"] == pytest.approx(2 / 3)
        assert bad["recall5"] == pytest.approx(0.5 / 3)

    def test_interval_surrounds_the_mean(self, bad_rows):
        s = summarize(bad_rows)
        assert s["hit5_low"] <= s["hit5"] <= s["hit5_high"]


class TestTables:
    def test_results_table_has_a_line_per_setting(self, good_rows, bad_rows):
        table = results_table({"good": good_rows, "bad": bad_rows})
        lines = table.splitlines()
        assert len(lines) == 4
        assert lines[2].startswith("| good | 67% | 100% (100%–100%)")
        assert lines[3].startswith("| bad | 0% | 33% (")

    def test_breakdown_by_language(self, good_rows, bad_rows):
        table = breakdown_table({"good": good_rows, "bad": bad_rows}, "lang")
        assert "| Setting | de (n=1) | en (n=2) |" in table
        assert "| good | 100% | 100% |" in table
        assert "| bad | 0% | 50% |" in table

    def test_breakdown_by_type(self, bad_rows):
        table = breakdown_table({"bad": bad_rows}, "type")
        assert "cross_reference (n=1)" in table and "definition (n=1)" in table

    def test_best_setting(self, good_rows, bad_rows):
        assert best_setting({"bad": bad_rows, "good": good_rows}) == "good"

    def test_best_setting_tie_is_decided_by_mrr(self, good_rows):
        worse_rank = [type(row)(**{**vars(row), "rr10": row.rr10 / 2}) for row in good_rows]
        assert best_setting({"a": worse_rank, "b": good_rows}) == "b"

    def test_report_compares_baseline_and_best(self, good_rows, bad_rows):
        text = report({BASELINE.name: bad_rows, "article + hybrid": good_rows})
        assert "Best setting: **article + hybrid**" in text
        assert "Paired bootstrap" in text and "p = " in text
        assert "## Hit@5 by language" in text and "## Hit@5 by question type" in text

    def test_report_without_baseline(self, good_rows):
        assert "Paired bootstrap" not in report({"good": good_rows})


class TestFailures:
    def test_lists_only_the_misses(self, bad_rows):
        text = failures_markdown("bad", bad_rows, QUESTIONS)
        assert "2 of 3 questions" in text
        assert "## q1 (en, factual)" in text and "## q2 (de, definition)" in text
        assert "## q3" not in text

    def test_shows_question_gold_and_what_came_back(self, bad_rows):
        text = failures_markdown("bad", bad_rows, QUESTIONS)
        assert "**Question:** Which practices are banned?" in text
        assert "**Gold:** art5-1" in text
        assert "**Top 5:** C:art1-1; Art. 5(2)" in text
        assert "**Right article in top 5:** yes" in text
        assert "**Top 5:** nothing found" in text

    def test_no_misses(self, good_rows):
        assert "0 of 3 questions" in failures_markdown("good", good_rows, QUESTIONS)


def test_known_units_come_from_the_parsed_act():
    units = load_known_units()
    assert len(units) == 808
    assert {"art6-2", "art3-p12", "anx3-4", "rct27", "art113"} <= units

import json

import pytest

from aiact_rag.answer_eval import (
    answers_table,
    cites_gold,
    evaluate_answers,
    load_rows,
    row_line,
    rows_to_resume,
    summarize_answers,
)
from aiact_rag.evaluate import Question
from aiact_rag.generate import REFUSAL, Answer
from aiact_rag.judge import (
    JUDGE_SYSTEM,
    build_judge_prompt,
    faithfulness,
    parse_verdicts,
)
from helpers import FakeLLM, r

SOURCES = [
    r("art6-2", 0.9, citation="Art. 6(2)", text="Systems in Annex III are high-risk."),
    r(
        "anx3-4",
        0.8,
        citation="Annex III, point 4",
        text="Employment and recruitment.",
        kind="annex",
    ),
    r("art5-1", 0.7, citation="Art. 5(1)", text="Social scoring is prohibited."),
]
GOOD = Answer("High-risk [Art. 6(2)].", SOURCES, ["Art. 6(2)"], [], False)
MADE_UP = Answer("Fined [Art. 99(3)].", SOURCES, [], ["Art. 99(3)"], False)
OFF_TARGET = Answer("Banned [Art. 5(1)].", SOURCES, ["Art. 5(1)"], [], False)
NO_CITATION = Answer("It is high-risk.", SOURCES, [], [], False)
REFUSED = Answer(REFUSAL["en"], SOURCES, [], [], True)


def verdicts(*supported):
    return json.dumps(
        {"claims": [{"claim": f"claim {i}", "supported": s} for i, s in enumerate(supported)]}
    )


class TestParseVerdicts:
    def test_plain_json(self):
        assert parse_verdicts(verdicts(True, False, True)) == [True, False, True]

    def test_json_in_a_code_fence_with_text_around(self):
        reply = (
            "Here is my assessment:\n```json\n" + verdicts(True, False) + "\n```\nHope this helps."
        )
        assert parse_verdicts(reply) == [True, False]

    @pytest.mark.parametrize(
        "reply",
        [
            "",
            "All claims are supported.",
            '{"claims": [{"claim": "a", "supported": true}',  # cut off
            '{"claims": []}',
            '{"verdict": "fine"}',
            '{"claims": "all good"}',
            '{"claims": [{"claim": "a", "supported": "yes"}]}',
            '{"claims": [{"claim": "a"}]}',
            '{"claims": ["a", "b"]}',
            "[true, false]",
        ],
    )
    def test_unusable_replies(self, reply):
        assert parse_verdicts(reply) is None


class TestJudgePrompt:
    def test_contains_sources_and_answer(self):
        system, user = build_judge_prompt("High-risk [Art. 6(2)].", SOURCES)
        assert system == JUDGE_SYSTEM
        assert "[Art. 6(2)]\nSystems in Annex III are high-risk." in user
        assert user.index("Social scoring is prohibited.") < user.index("High-risk [Art. 6(2)].")


class TestFaithfulness:
    def test_share_of_supported_claims(self):
        assert faithfulness(GOOD, FakeLLM(verdicts(True, True, False, True))) == 0.75

    def test_all_supported(self):
        assert faithfulness(GOOD, FakeLLM(verdicts(True))) == 1.0

    def test_judge_gets_the_judge_prompt(self):
        judge = FakeLLM(verdicts(True))
        faithfulness(GOOD, judge)
        assert judge.calls == [build_judge_prompt(GOOD.text, SOURCES)]

    def test_refusal_is_not_judged(self):
        judge = FakeLLM(verdicts(True))
        assert faithfulness(REFUSED, judge) is None
        assert judge.calls == []

    def test_answer_without_sources_is_not_judged(self):
        judge = FakeLLM(verdicts(True))
        assert faithfulness(Answer("Something.", [], [], [], False), judge) is None
        assert judge.calls == []

    def test_unreadable_reply_gives_none_not_zero(self):
        assert faithfulness(GOOD, FakeLLM("I think it is fine.")) is None


class TestCitesGold:
    def test_valid_citation_of_a_gold_unit(self):
        assert cites_gold(GOOD, ["art6-2"]) is True

    def test_valid_citation_of_another_unit(self):
        assert cites_gold(OFF_TARGET, ["art6-2"]) is False

    def test_gold_was_retrieved_but_not_cited(self):
        assert cites_gold(NO_CITATION, ["art6-2"]) is False

    def test_any_gold_unit_counts(self):
        assert cites_gold(GOOD, ["anx3-4", "art6-2"]) is True

    def test_unanswerable_question(self):
        assert cites_gold(GOOD, []) is False


QUESTIONS = [
    Question("q1", "en", "Is hiring AI high-risk?", ["art6-2"], True, "factual"),
    Question("q2", "en", "What is the fine?", ["art99-3"], True, "factual"),
    Question("q3", "de", "Ist das verboten?", ["art6-2"], True, "factual"),
    Question("q4", "en", "A question refused by mistake?", ["art6-2"], True, "factual"),
    Question("q5", "en", "What about crypto?", [], False, "trick"),
    Question("q6", "en", "What about salaries?", [], False, "trick"),
]
ANSWERS = {
    "q1": GOOD,
    "q2": MADE_UP,
    "q3": OFF_TARGET,
    "q4": REFUSED,
    "q5": REFUSED,
    "q6": NO_CITATION,
}


@pytest.fixture
def answer_fn():
    by_text = {q.question: ANSWERS[q.id] for q in QUESTIONS}
    calls = []

    def fn(question, lang):
        calls.append((question, lang))
        return by_text[question]

    fn.calls = calls
    return fn


class TestEvaluateAnswers:
    def test_every_question_is_answered(self, answer_fn):
        rows = evaluate_answers(QUESTIONS, answer_fn)
        assert [row.question_id for row in rows] == ["q1", "q2", "q3", "q4", "q5", "q6"]
        assert answer_fn.calls[2] == ("Ist das verboten?", "de")

    def test_row_fields(self, answer_fn):
        q1, q2, *_ = evaluate_answers(QUESTIONS, answer_fn)
        assert (
            q1.refused,
            q1.valid_citations,
            q1.invalid_citations,
            q1.cites_gold,
        ) == (False, ["Art. 6(2)"], [], True)
        assert (q2.valid_citations, q2.invalid_citations, q2.cites_gold) == (
            [],
            ["Art. 99(3)"],
            False,
        )
        assert q1.answer == "High-risk [Art. 6(2)]."
        assert q1.sources == ["Art. 6(2)", "Annex III, point 4", "Art. 5(1)"]
        assert q1.faithfulness is None

    def test_judge_is_only_called_for_real_answers(self, answer_fn):
        judge = FakeLLM(verdicts(True, False))
        rows = evaluate_answers(QUESTIONS, answer_fn, judge)
        assert [row.faithfulness for row in rows] == [0.5, 0.5, 0.5, None, None, 0.5]
        assert len(judge.calls) == 4

    def test_rows_can_be_saved_as_json(self, answer_fn):
        from dataclasses import asdict

        rows = evaluate_answers(QUESTIONS, answer_fn, FakeLLM(verdicts(True)))
        assert json.loads(json.dumps(asdict(rows[0]), ensure_ascii=False))["question_id"] == "q1"


SETTINGS = {"model": "gemma", "judge": None, "mode": "dense"}


class TestResume:
    @pytest.fixture
    def run(self, tmp_path, answer_fn):
        """A run that stopped after the first three questions."""
        rows_path, settings_path = tmp_path / "answers.jsonl", tmp_path / "answers.run.json"
        rows = evaluate_answers(QUESTIONS[:3], answer_fn)
        rows_path.write_text("".join(row_line(row) for row in rows), encoding="utf-8")
        settings_path.write_text(json.dumps(SETTINGS), encoding="utf-8")
        answer_fn.calls.clear()
        return rows_path, settings_path

    def test_done_questions_are_not_asked_again(self, answer_fn):
        first = evaluate_answers(QUESTIONS[:3], answer_fn)
        answer_fn.calls.clear()
        rows = evaluate_answers(QUESTIONS, answer_fn, done={row.question_id: row for row in first})
        assert [question for question, _ in answer_fn.calls] == [q.question for q in QUESTIONS[3:]]
        assert [row.question_id for row in rows] == ["q1", "q2", "q3", "q4", "q5", "q6"]
        assert rows[:3] == first

    def test_on_row_gets_every_new_row_as_soon_as_it_is_made(self, answer_fn):
        first = evaluate_answers(QUESTIONS[:3], answer_fn)
        seen = []
        evaluate_answers(
            QUESTIONS, answer_fn, done={row.question_id: row for row in first}, on_row=seen.append
        )
        assert [row.question_id for row in seen] == ["q4", "q5", "q6"]

    def test_rows_before_a_crash_are_kept(self, answer_fn):
        def crashing(question, lang):
            if question == QUESTIONS[2].question:
                raise RuntimeError("the LLM is down")
            return answer_fn(question, lang)

        seen = []
        with pytest.raises(RuntimeError):
            evaluate_answers(QUESTIONS, crashing, on_row=seen.append)
        assert [row.question_id for row in seen] == ["q1", "q2"]

    def test_load_rows_round_trip(self, tmp_path, answer_fn):
        rows = evaluate_answers(QUESTIONS[:3], answer_fn)
        path = tmp_path / "answers.jsonl"
        path.write_text("".join(row_line(row) for row in rows), encoding="utf-8")
        assert load_rows(path) == {row.question_id: row for row in rows}

    def test_load_rows_without_a_file(self, tmp_path):
        assert load_rows(tmp_path / "nothing.jsonl") == {}

    def test_a_line_cut_off_by_a_crash_is_skipped(self, run):
        rows_path, _ = run
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write('{"question_id": "q4", "lang": "en", "ty')
        assert list(load_rows(rows_path)) == ["q1", "q2", "q3"]

    def test_resume_keeps_the_rows_of_the_same_settings(self, run):
        rows_path, settings_path = run
        done = rows_to_resume(rows_path, settings_path, SETTINGS, QUESTIONS)
        assert list(done) == ["q1", "q2", "q3"]

    def test_resume_cleans_the_file_so_rows_can_be_appended(self, run):
        rows_path, settings_path = run
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write('{"question_id": "q4", "lang": "en", "ty')
        rows_to_resume(rows_path, settings_path, SETTINGS, QUESTIONS)
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2", "q3"]

    def test_resume_drops_rows_of_questions_that_are_not_asked(self, run):
        rows_path, settings_path = run
        done = rows_to_resume(rows_path, settings_path, SETTINGS, QUESTIONS[:2])
        assert list(done) == ["q1", "q2"]

    def test_resume_refuses_other_settings(self, run):
        rows_path, settings_path = run
        with pytest.raises(ValueError, match="other settings"):
            rows_to_resume(rows_path, settings_path, {**SETTINGS, "mode": "hybrid"}, QUESTIONS)
        assert len(load_rows(rows_path)) == 3  # nothing was thrown away

    def test_resume_refuses_rows_without_recorded_settings(self, run):
        rows_path, settings_path = run
        settings_path.unlink()
        with pytest.raises(ValueError, match="other settings"):
            rows_to_resume(rows_path, settings_path, SETTINGS, QUESTIONS)

    def test_nothing_to_resume(self, tmp_path):
        assert (
            rows_to_resume(tmp_path / "a.jsonl", tmp_path / "a.run.json", SETTINGS, QUESTIONS) == {}
        )


class TestSummarizeAnswers:
    @pytest.fixture
    def summary(self, answer_fn):
        return summarize_answers(
            evaluate_answers(QUESTIONS, answer_fn, FakeLLM(verdicts(True, False)))
        )

    def test_refusals(self, summary):
        assert summary["refused_unanswerable"] == 0.5  # q5 yes, q6 no
        assert summary["refused_answerable"] == 0.25  # q4 of q1..q4

    def test_citation_numbers_only_count_real_answers_to_answerable_questions(self, summary):
        assert summary["n_answered"] == 3  # q1, q2, q3
        assert summary["has_citation"] == 1.0
        assert summary["all_citations_valid"] == pytest.approx(2 / 3)  # q2 made one up
        assert summary["cites_gold"] == pytest.approx(1 / 3)  # only q1

    def test_faithfulness(self, summary):
        assert summary["faithfulness"] == 0.5
        assert summary["n_judged"] == 4

    def test_without_judge(self, answer_fn):
        summary = summarize_answers(evaluate_answers(QUESTIONS, answer_fn))
        assert summary["faithfulness"] is None and summary["n_judged"] == 0

    def test_nothing_to_measure_gives_none(self, answer_fn):
        only_tricks = [q for q in QUESTIONS if not q.answerable]
        summary = summarize_answers(evaluate_answers(only_tricks, answer_fn))
        assert summary["refused_answerable"] is None
        assert summary["cites_gold"] is None

    def test_table(self, summary):
        table = answers_table(summary, "ollama: gemma")
        assert "| Measure | ollama: gemma |" in table
        assert "| Unanswerable questions refused | 50% |" in table
        assert "| Answerable questions refused by mistake | 25% |" in table
        assert "| Faithfulness, judged on 4 answers | 50% |" in table

    def test_table_shows_a_dash_for_missing_values(self, answer_fn):
        table = answers_table(summarize_answers(evaluate_answers(QUESTIONS, answer_fn)), "x")
        assert "| Faithfulness, judged on 0 answers | – |" in table

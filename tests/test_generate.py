import pytest

from aiact_rag.generate import (
    REFUSAL,
    Answer,
    answer_question,
    build_prompt,
    check_citations,
    extract_citations,
    is_refusal,
    normalize,
)
from helpers import FakeLLM, FakeRetriever, r

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


class TestNormalize:
    def test_example(self):
        assert normalize("  Art.  6(2)\n") == "art. 6(2)"

    def test_tabs_and_newlines(self):
        assert normalize("Annex\tIII,\n point 4") == "annex iii, point 4"

    def test_german_sharp_s(self):
        assert normalize("Maßnahme") == normalize("MASSNAHME")


class TestBuildPrompt:
    def test_returns_two_strings(self):
        system, user = build_prompt("Is recruitment AI high-risk?", SOURCES, "en")
        assert isinstance(system, str) and isinstance(user, str)

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_system_contains_the_refusal_sentence(self, lang):
        system, _ = build_prompt("q", SOURCES, lang)
        assert REFUSAL[lang] in system

    def test_system_does_not_contain_the_question_or_sources(self):
        system, _ = build_prompt("Is recruitment AI high-risk?", SOURCES, "en")
        assert "recruitment AI" not in system
        assert "Social scoring is prohibited." not in system

    def test_user_has_label_then_text_for_each_source(self):
        _, user = build_prompt("q", SOURCES, "en")
        assert "[Art. 6(2)]\nSystems in Annex III are high-risk." in user
        assert "[Annex III, point 4]\nEmployment and recruitment." in user
        assert "[Art. 5(1)]\nSocial scoring is prohibited." in user

    def test_sources_in_ranking_order(self):
        _, user = build_prompt("q", SOURCES, "en")
        positions = [user.index(f"[{x.chunk.citation}]") for x in SOURCES]
        assert positions == sorted(positions)

    def test_question_comes_after_the_last_source(self):
        _, user = build_prompt("Is recruitment AI high-risk?", SOURCES, "en")
        assert user.index("Is recruitment AI high-risk?") > user.index(
            "Social scoring is prohibited."
        )
        assert user.rstrip().endswith("Is recruitment AI high-risk?")

    def test_uses_text_not_embed_text(self):
        # helpers.chunk builds embed_text as "[citation]\ntext", so a label must appear exactly once.
        _, user = build_prompt("q", SOURCES, "en")
        assert user.count("[Art. 6(2)]") == 1

    def test_does_not_modify_results(self):
        results = list(SOURCES)
        build_prompt("q", results, "en")
        assert results == SOURCES


class TestExtractCitations:
    def test_example_from_docstring(self):
        answer = "High-risk [Art. 6(2)]. See [ Annex III, point 4 ] and [Art. 6(2)]."
        assert extract_citations(answer) == ["Art. 6(2)", "Annex III, point 4"]

    def test_no_citations(self):
        assert extract_citations("No brackets here (only parentheses).") == []

    def test_two_brackets_in_a_row(self):
        assert extract_citations("Yes [Art. 5(1)][Art. 6(2)].") == [
            "Art. 5(1)",
            "Art. 6(2)",
        ]

    def test_empty_brackets_are_ignored(self):
        assert extract_citations("Nothing [] and [   ] here.") == []

    def test_different_writing_is_kept_as_written(self):
        assert extract_citations("[Art. 6(2)] and [art. 6(2)]") == [
            "Art. 6(2)",
            "art. 6(2)",
        ]

    def test_german(self):
        assert extract_citations("Ja [Art. 6 Abs. 2] und [Anhang III Nummer 4].") == [
            "Art. 6 Abs. 2",
            "Anhang III Nummer 4",
        ]


class TestCheckCitations:
    def test_all_valid(self):
        answer = "It is high-risk [Art. 6(2)] because of recruitment [Annex III, point 4]."
        assert check_citations(answer, SOURCES) == (
            ["Art. 6(2)", "Annex III, point 4"],
            [],
        )

    def test_made_up_citation(self):
        answer = "It is high-risk [Art. 6(2)] and fined [Art. 99(3)]."
        assert check_citations(answer, SOURCES) == (["Art. 6(2)"], ["Art. 99(3)"])

    def test_close_is_not_valid(self):
        assert check_citations("See [Art. 6(1)] and [Art. 6].", SOURCES) == (
            [],
            ["Art. 6(1)", "Art. 6"],
        )

    def test_citing_a_point_inside_a_source_is_valid(self):
        answer = "Banned [Art. 5(1)(d)] and [Art. 5(1), point (e)]; see [Annex III, point 4(a)]."
        assert check_citations(answer, SOURCES) == (
            ["Art. 5(1)", "Annex III, point 4"],
            [],
        )

    def test_a_longer_number_is_not_a_point_inside(self):
        sources = SOURCES + [r("art4", citation="Art. 4", text="AI literacy.")]
        valid, invalid = check_citations(
            "See [Art. 40], [Art. 5(10)] and [Art. 4 sentence 1].", sources
        )
        assert valid == ["Art. 4"]
        assert invalid == ["Art. 40", "Art. 5(10)"]

    def test_german_sub_citation(self):
        sources = [r("art5-1", citation="Art. 5 Abs. 1", text="Verboten.")]
        assert check_citations("Ja [Art. 5 Abs. 1 Buchst. d].", sources) == (
            ["Art. 5 Abs. 1"],
            [],
        )

    def test_case_and_spaces_do_not_matter(self):
        valid, invalid = check_citations("See [art.  6(2)] and [ANNEX III, POINT 4].", SOURCES)
        assert valid == ["Art. 6(2)", "Annex III, point 4"]  # as written in the source
        assert invalid == []

    def test_same_source_cited_in_two_writings_counts_once(self):
        valid, _ = check_citations("[Art. 6(2)] ... [art. 6(2)]", SOURCES)
        assert valid == ["Art. 6(2)"]

    def test_order_of_first_appearance_in_the_answer(self):
        valid, invalid = check_citations("[Art. 5(1)] [x] [Art. 6(2)] [y] [x]", SOURCES)
        assert valid == ["Art. 5(1)", "Art. 6(2)"]
        assert invalid == ["x", "y"]

    def test_no_citations(self):
        assert check_citations("I just know it.", SOURCES) == ([], [])

    def test_no_sources(self):
        assert check_citations("See [Art. 6(2)].", []) == ([], ["Art. 6(2)"])


class TestIsRefusal:
    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_exact_sentence(self, lang):
        assert is_refusal(REFUSAL[lang], lang) is True

    def test_inside_a_longer_answer_and_other_case(self):
        assert is_refusal("Sorry.\n" + REFUSAL["en"].upper() + "  ", "en") is True

    def test_normal_answer(self):
        assert is_refusal("It is high-risk [Art. 6(2)].", "en") is False

    def test_refusal_in_the_other_language_does_not_count(self):
        assert is_refusal(REFUSAL["de"], "en") is False


class TestAnswerQuestion:
    def test_normal_answer(self):
        retriever = FakeRetriever(SOURCES)
        llm = FakeLLM("It is high-risk [Art. 6(2)], see also [Art. 99(3)].")
        answer = answer_question("Is recruitment AI high-risk?", "en", retriever, llm)
        assert isinstance(answer, Answer)
        assert answer.text == "It is high-risk [Art. 6(2)], see also [Art. 99(3)]."
        assert answer.sources == SOURCES
        assert answer.valid_citations == ["Art. 6(2)"]
        assert answer.invalid_citations == ["Art. 99(3)"]
        assert answer.refused is False

    def test_passes_question_and_lang_to_the_retriever(self):
        retriever = FakeRetriever(SOURCES)
        answer_question("Ist das hochriskant?", "de", retriever, FakeLLM("Ja."))
        assert retriever.calls == [("Ist das hochriskant?", "de")]

    def test_llm_gets_the_built_prompt(self):
        llm = FakeLLM("ok")
        answer_question("Is recruitment AI high-risk?", "en", FakeRetriever(SOURCES), llm)
        assert llm.calls == [build_prompt("Is recruitment AI high-risk?", SOURCES, "en")]

    def test_llm_refuses(self):
        answer = answer_question("q", "en", FakeRetriever(SOURCES), FakeLLM(REFUSAL["en"]))
        assert answer.refused is True
        assert answer.sources == SOURCES
        assert answer.valid_citations == [] and answer.invalid_citations == []

    @pytest.mark.parametrize("lang", ["en", "de"])
    def test_nothing_retrieved_skips_the_llm(self, lang):
        llm = FakeLLM("should never be used")
        answer = answer_question("q", lang, FakeRetriever([]), llm)
        assert llm.calls == []
        assert answer == Answer(REFUSAL[lang], [], [], [], True)

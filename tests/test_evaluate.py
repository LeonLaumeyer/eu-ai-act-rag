import pytest

from aiact_rag.evaluate import (
    Question,
    bootstrap_ci,
    hit_at_k,
    load_questions,
    paired_bootstrap_p,
    recall_at_k,
    reciprocal_rank,
    to_article,
    units_of,
)
from helpers import r

RESULTS = [r("art5-1"), r("art6-1"), r("art6-2"), r("anx3-4"), r("art50-1")]
# Fixed-size chunks cover several units each.
FIXED = [
    r("f0", unit_ids=["art5-1", "art5-2"]),
    r("f1", unit_ids=["art5-2", "art5-3", "art6-1"]),
    r("f2", unit_ids=["art6-1", "art6-2", "art6-3"]),
]

GOOD_YAML = """
- id: q001
  lang: en
  question: "Which AI practices are prohibited?"
  gold: [art5-1]
  type: factual
- id: q002
  lang: de
  question: "Ab wann gilt ein KI-System als hochriskant?"
  gold: [art6-2, anx3-4]
  answerable: true
  type: cross_reference
  reference_answer: "Wenn es in Anhang III steht."
- id: q003
  lang: en
  question: "What fines does the AI Act set for GDPR violations?"
  answerable: false
  type: trick
"""


def write(tmp_path, text):
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadQuestions:
    def test_loads_all_fields(self, tmp_path):
        questions = load_questions(write(tmp_path, GOOD_YAML))
        assert questions[0] == Question(
            "q001",
            "en",
            "Which AI practices are prohibited?",
            ["art5-1"],
            True,
            "factual",
            "",
        )
        assert questions[1].gold == ["art6-2", "anx3-4"]
        assert questions[1].reference_answer == "Wenn es in Anhang III steht."
        assert questions[2] == Question(
            "q003",
            "en",
            "What fines does the AI Act set for GDPR violations?",
            [],
            False,
            "trick",
            "",
        )

    def test_umlauts_survive(self, tmp_path):
        path = write(
            tmp_path,
            '- {id: q1, lang: de, question: "Wie hoch sind die Geldbußen?", gold: [art99-3]}',
        )
        assert load_questions(path)[0].question == "Wie hoch sind die Geldbußen?"

    def test_empty_file(self, tmp_path):
        assert load_questions(write(tmp_path, "")) == []

    def test_known_units_ok(self, tmp_path):
        known = {"art5-1", "art6-2", "anx3-4"}
        assert len(load_questions(write(tmp_path, GOOD_YAML), known_units=known)) == 3

    def test_unknown_gold_unit(self, tmp_path):
        with pytest.raises(ValueError, match="q002"):
            load_questions(write(tmp_path, GOOD_YAML), known_units={"art5-1", "art6-2"})

    @pytest.mark.parametrize(
        "bad, question_id",
        [
            (
                "- {id: q1, lang: en, question: a, gold: [art5-1]}\n- {id: q1, lang: en, question: b, gold: [art5-1]}",
                "q1",
            ),
            ("- {id: q7, lang: fr, question: a, gold: [art5-1]}", "q7"),
            ("- {id: q8, lang: en, question: a}", "q8"),
            ("- {id: q9, lang: en, question: a, gold: []}", "q9"),
            (
                "- {id: q5, lang: en, question: a, gold: [art5-1], answerable: false}",
                "q5",
            ),
        ],
        ids=[
            "duplicate id",
            "unknown lang",
            "gold missing",
            "gold empty",
            "unanswerable with gold",
        ],
    )
    def test_invalid_data(self, tmp_path, bad, question_id):
        with pytest.raises(ValueError, match=question_id):
            load_questions(write(tmp_path, bad))


class TestToArticle:
    @pytest.mark.parametrize(
        "unit, article",
        [
            ("art6-2", "art6"),
            ("art3-p12", "art3"),
            ("art4", "art4"),
            ("art113-3", "art113"),
            ("anx3-4", "anx3"),
            ("anx8-s1", "anx8"),
            ("anx2", "anx2"),
            ("rct27", "rct27"),
        ],
    )
    def test_examples(self, unit, article):
        assert to_article(unit) == article


class TestUnitsOf:
    def test_unit_level(self):
        assert units_of(FIXED[1]) == {"art5-2", "art5-3", "art6-1"}

    def test_article_level(self):
        assert units_of(FIXED[1], "article") == {"art5", "art6"}

    def test_single_unit(self):
        assert units_of(r("art6-2")) == {"art6-2"}

    def test_unknown_level(self):
        with pytest.raises(ValueError):
            units_of(r("art6-2"), "chapter")


class TestHitAtK:
    def test_hit_inside_k(self):
        assert hit_at_k(RESULTS, ["art6-2"], k=3) is True

    def test_gold_just_outside_k(self):
        assert hit_at_k(RESULTS, ["art6-2"], k=2) is False

    def test_any_gold_unit_is_enough(self):
        assert hit_at_k(RESULTS, ["art99-1", "anx3-4"], k=5) is True

    def test_miss(self):
        assert hit_at_k(RESULTS, ["art99-1"], k=5) is False

    def test_k_larger_than_results(self):
        assert hit_at_k(RESULTS, ["art50-1"], k=100) is True

    def test_no_results(self):
        assert hit_at_k([], ["art6-2"], k=5) is False

    def test_fixed_chunk_covers_gold(self):
        assert hit_at_k(FIXED, ["art5-3"], k=2) is True
        assert hit_at_k(FIXED, ["art5-3"], k=1) is False

    def test_article_level_is_more_forgiving(self):
        assert hit_at_k(RESULTS, ["art6-3"], k=5) is False
        assert hit_at_k(RESULTS, ["art6-3"], k=5, level="article") is True
        assert hit_at_k(RESULTS, ["art6-3"], k=1, level="article") is False

    def test_article_level_does_not_mix_up_numbers(self):
        # art5 must not match art50
        assert hit_at_k([r("art50-1")], ["art5-1"], k=1, level="article") is False

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            hit_at_k(RESULTS, [], k=5)


class TestReciprocalRank:
    @pytest.mark.parametrize("gold, expected", [("art5-1", 1.0), ("art6-1", 0.5), ("anx3-4", 0.25)])
    def test_rank(self, gold, expected):
        assert reciprocal_rank(RESULTS, [gold]) == pytest.approx(expected)

    def test_first_gold_unit_counts(self):
        assert reciprocal_rank(RESULTS, ["art50-1", "art6-2"]) == pytest.approx(1 / 3)

    def test_miss_is_zero(self):
        assert reciprocal_rank(RESULTS, ["art99-1"]) == 0.0

    def test_outside_k_is_zero(self):
        assert reciprocal_rank(RESULTS, ["art50-1"], k=4) == 0.0
        assert reciprocal_rank(RESULTS, ["art50-1"], k=5) == pytest.approx(0.2)

    def test_article_level(self):
        assert reciprocal_rank(RESULTS, ["art6-3"], level="article") == pytest.approx(0.5)

    def test_fixed_chunks(self):
        assert reciprocal_rank(FIXED, ["art6-1"]) == pytest.approx(0.5)

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            reciprocal_rank(RESULTS, [])


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(RESULTS, ["art6-2", "anx3-4"], k=5) == 1.0

    def test_half_found(self):
        assert recall_at_k(RESULTS, ["art6-2", "anx3-4"], k=3) == 0.5

    def test_none_found(self):
        assert recall_at_k(RESULTS, ["art99-1", "art99-3"], k=5) == 0.0

    def test_one_chunk_can_cover_two_gold_units(self):
        assert recall_at_k(FIXED, ["art5-1", "art5-2"], k=1) == 1.0

    def test_gold_found_twice_counts_once(self):
        assert recall_at_k(FIXED, ["art6-1", "art99-1"], k=3) == 0.5

    def test_article_level(self):
        # gold articles: art6 and art99; art6 is found
        assert recall_at_k(RESULTS, ["art6-3", "art6-4", "art99-1"], k=5, level="article") == 0.5

    def test_empty_gold(self):
        with pytest.raises(ValueError):
            recall_at_k(RESULTS, [], k=5)


HITS_80 = [1.0] * 40 + [0.0] * 10  # 50 questions, 80% hit rate


class TestBootstrapCI:
    def test_interval_contains_the_mean(self):
        low, high = bootstrap_ci(HITS_80)
        assert low < 0.8 < high

    def test_width_for_50_questions(self):
        low, high = bootstrap_ci(HITS_80)
        assert 0.15 < high - low < 0.30  # about +-11 points: 50 questions are not many

    def test_more_questions_give_a_narrower_interval(self):
        small = bootstrap_ci(HITS_80)
        large = bootstrap_ci(HITS_80 * 20)
        assert large[1] - large[0] < (small[1] - small[0]) / 2

    def test_smaller_alpha_gives_a_wider_interval(self):
        ci95 = bootstrap_ci(HITS_80, alpha=0.05)
        ci99 = bootstrap_ci(HITS_80, alpha=0.01)
        assert ci99[0] <= ci95[0] and ci99[1] >= ci95[1]
        assert ci99 != ci95

    def test_all_equal_values(self):
        assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)

    def test_same_seed_same_result(self):
        assert bootstrap_ci(HITS_80, seed=3) == bootstrap_ci(HITS_80, seed=3)

    def test_other_seed_other_result(self):
        values = [i / 37 for i in range(37)]  # many different values, so the quantiles can move
        assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)

    def test_plain_floats(self):
        low, high = bootstrap_ci(HITS_80)
        assert type(low) is float and type(high) is float

    def test_stays_inside_the_data_range(self):
        low, high = bootstrap_ci([0.0, 1.0, 1.0])
        assert 0.0 <= low <= high <= 1.0

    def test_empty(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])


class TestPairedBootstrap:
    def test_clear_improvement(self):
        baseline = [0.0] * 20 + [1.0] * 30
        improved = [1.0] * 15 + [0.0] * 5 + [1.0] * 30  # fixes 15 questions, breaks none
        assert paired_bootstrap_p(baseline, improved) < 0.01

    def test_identical_systems(self):
        assert paired_bootstrap_p(HITS_80, HITS_80) == 1.0

    def test_worse_system(self):
        baseline = [1.0] * 30 + [0.0] * 20
        worse = [0.0] * 10 + [1.0] * 20 + [0.0] * 20
        assert paired_bootstrap_p(baseline, worse) > 0.99

    def test_small_difference_is_not_significant(self):
        baseline = [1.0] * 35 + [0.0] * 15
        improved = [1.0] * 36 + [0.0] * 14  # one more hit
        assert paired_bootstrap_p(baseline, improved) > 0.1

    def test_pairing_matters(self):
        # Same hit rates (50% vs 60%) in both cases; only the pairing differs.
        baseline = [1.0, 0.0] * 25
        consistent = [1.0, 0.0] * 20 + [1.0, 1.0] * 5  # fixes 5, breaks none
        noisy = [0.0, 1.0] * 10 + [1.0, 1.0] * 5 + [1.0, 0.0] * 10  # fixes 15, breaks 10
        assert sum(consistent) == sum(noisy)
        assert paired_bootstrap_p(baseline, consistent) < paired_bootstrap_p(baseline, noisy)

    def test_same_seed_same_result(self):
        a, b = [0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0]
        assert paired_bootstrap_p(a, b, seed=5) == paired_bootstrap_p(a, b, seed=5)

    def test_plain_float(self):
        assert type(paired_bootstrap_p([0.0, 1.0], [1.0, 1.0])) is float

    @pytest.mark.parametrize("a, b", [([], []), ([1.0], [1.0, 0.0])])
    def test_invalid_input(self, a, b):
        with pytest.raises(ValueError):
            paired_bootstrap_p(a, b)

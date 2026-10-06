import pytest

from aiact_rag.index.bm25 import BM25Index, tokenize
from aiact_rag.models import SearchResult
from helpers import chunk, ids

CHUNKS = [
    chunk(
        "art5-1",
        "The following AI practices shall be prohibited: manipulation and social scoring.",
    ),
    chunk(
        "art6-2",
        "AI systems referred to in Annex III shall be considered to be high-risk.",
    ),
    chunk(
        "art50-1",
        "Providers shall ensure that chatbots inform natural persons about the interaction.",
    ),
    chunk(
        "art99-3",
        "Non-compliance shall be subject to administrative fines of up to 35 000 000 EUR.",
    ),
    chunk(
        "art113",
        "This Regulation shall enter into force on the twentieth day after publication.",
    ),
    chunk(
        "art26-1",
        "Deployers shall take appropriate technical measures and follow the instructions.",
    ),
    chunk(
        "anx3-4",
        "Employment: systems used for recruitment and the selection of natural persons.",
        kind="annex",
    ),
    chunk(
        "rct27",
        "Recital on recruitment, trustworthy systems and oversight by natural persons.",
        kind="recital",
    ),
    chunk(
        "art5-1",
        "Die folgenden KI-Praktiken sind verboten: Manipulation und soziale Bewertung.",
        lang="de",
    ),
    chunk(
        "art6-2",
        "Die in Anhang III genannten KI-Systeme gelten als hochriskant.",
        lang="de",
    ),
    chunk(
        "art50-1",
        "Anbieter stellen sicher, dass Chatbots natürliche Personen informieren.",
        lang="de",
    ),
    chunk(
        "art99-3",
        "Bei Verstößen werden Geldbußen von bis zu 35 000 000 EUR verhängt.",
        lang="de",
    ),
    chunk(
        "art113",
        "Diese Verordnung tritt am zwanzigsten Tag nach ihrer Veröffentlichung in Kraft.",
        lang="de",
    ),
    chunk(
        "art26-1",
        "Betreiber treffen geeignete technische Maßnahmen und befolgen die Anweisungen.",
        lang="de",
    ),
]


@pytest.fixture(scope="module")
def index():
    return BM25Index(CHUNKS)


class TestTokenize:
    def test_example_from_docstring(self):
        assert tokenize("Prohibited AI practices", "en") == [
            "prohibit",
            "ai",
            "practic",
        ]

    def test_stemming_makes_word_forms_equal(self):
        assert tokenize("prohibited", "en") == tokenize("PROHIBITION", "en") == ["prohibit"]

    def test_punctuation_is_dropped(self):
        assert tokenize("high-risk, (AI) systems.", "en") == [
            "high",
            "risk",
            "ai",
            "system",
        ]

    def test_german_stemmer(self):
        assert tokenize("Systeme Systemen", "de") == ["system", "system"]
        assert tokenize("Anbietern", "de") == tokenize("Anbieter", "de")

    def test_german_umlauts_stay_one_word(self):
        assert len(tokenize("Geldbußen für Verstöße", "de")) == 3

    def test_language_matters(self):
        assert tokenize("verbotene", "de") != tokenize("verbotene", "en")

    def test_empty(self):
        assert tokenize("", "en") == []
        assert tokenize(" ... ", "de") == []

    def test_unknown_language(self):
        with pytest.raises(ValueError):
            tokenize("bonjour", "fr")


class TestSearch:
    def test_exact_keyword(self, index):
        results = index.search("administrative fines", "en")
        assert ids(results) == ["art99-3"]
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].score, float)
        assert results[0].score > 0

    def test_stemming_connects_query_and_text(self, index):
        # "prohibition" is not in the text, "prohibited" is.
        assert ids(index.search("prohibition of manipulation", "en"))[0] == "art5-1"

    def test_only_matching_chunks_are_returned(self, index):
        assert ids(index.search("chatbots", "en", top_k=10)) == ["art50-1"]

    def test_no_match(self, index):
        assert index.search("cryptocurrency", "en") == []

    def test_empty_query(self, index):
        assert index.search("", "en") == []
        assert index.search("?!", "en") == []

    def test_sorted_best_first(self, index):
        results = index.search("recruitment selection natural persons", "en", top_k=10)
        scores = [x.score for x in results]
        assert scores == sorted(scores, reverse=True)
        assert ids(results)[0] == "anx3-4"

    def test_top_k(self, index):
        assert len(index.search("natural persons", "en", top_k=2)) == 2

    def test_german_query_searches_german_chunks(self, index):
        results = index.search("Geldbußen", "de")
        assert [x.chunk.chunk_id for x in results] == ["de-article-art99-3"]

    def test_german_stemming(self, index):
        # "Anbietern" (dative) must find "Anbieter".
        assert ids(index.search("Pflichten von Anbietern", "de"))[0] == "art50-1"

    def test_languages_are_separate(self, index):
        assert index.search("Geldbußen", "en") == []
        assert index.search("fines", "de") == []

    def test_header_is_searched_too(self, index):
        # helpers.chunk puts "[C:art113]" into embed_text; "art113" is not in the text.
        assert ids(index.search("art113", "en")) == ["art113"]

    def test_kinds_filter(self, index):
        everything = index.search("recruitment", "en", top_k=10)
        assert {x.chunk.kind for x in everything} == {"annex", "recital"}
        assert ids(index.search("recruitment", "en", top_k=10, kinds=("article", "annex"))) == [
            "anx3-4"
        ]

    def test_kinds_filter_happens_before_top_k(self, index):
        results = index.search("recruitment natural persons", "en", top_k=1, kinds=["recital"])
        assert ids(results) == ["rct27"]

    def test_language_without_chunks(self):
        only_english = BM25Index([c for c in CHUNKS if c.lang == "en"])
        assert only_english.search("Geldbußen", "de") == []

    def test_unknown_language(self, index):
        with pytest.raises(ValueError):
            index.search("amendes", "fr")

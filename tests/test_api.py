import pytest
from fastapi.testclient import TestClient

from aiact_rag.api import create_app, detect_lang
from aiact_rag.generate import REFUSAL, Answer
from aiact_rag.llm.base import LLMError
from helpers import r

SOURCES = [
    r("art6-2", 0.9, citation="Art. 6(2)", text="Systems in Annex III are high-risk."),
    r("art5-1", 0.5, citation="Art. 5(1)", text="Social scoring is prohibited."),
]
ANSWER = Answer(
    "It is high-risk [Art. 6(2)] [Art. 99].", SOURCES, ["Art. 6(2)"], ["Art. 99"], False
)


class FakeAnswerFn:
    def __init__(self, answer=ANSWER, error=None):
        self.answer, self.error = answer, error
        self.calls = []

    def __call__(self, question, lang):
        self.calls.append((question, lang))
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def answer_fn():
    return FakeAnswerFn()


@pytest.fixture
def client(answer_fn):
    return TestClient(create_app(answer_fn))


class TestDetectLang:
    @pytest.mark.parametrize(
        "question",
        [
            "Which AI practices are prohibited?",
            "What is the fine for non-compliance?",
            "Is a chatbot a high-risk system under the AI Act?",
            "GPAI obligations",
            "",
        ],
    )
    def test_english(self, question):
        assert detect_lang(question) == "en"

    @pytest.mark.parametrize(
        "question",
        [
            "Welche KI-Praktiken sind verboten?",
            "Was ist ein Hochrisiko-KI-System?",
            "Wann gilt die Verordnung?",
            "Wie hoch sind die Geldbußen?",
            "MÜSSEN ANBIETER DAS MELDEN",
        ],
    )
    def test_german(self, question):
        assert detect_lang(question) == "de"


class TestHealth:
    def test_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAsk:
    def test_response(self, client):
        response = client.post(
            "/ask", json={"question": "Is recruitment AI high-risk?", "lang": "en"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "answer": "It is high-risk [Art. 6(2)] [Art. 99].",
            "lang": "en",
            "refused": False,
            "citations": ["Art. 6(2)"],
            "invalid_citations": ["Art. 99"],
            "sources": [
                {
                    "citation": "Art. 6(2)",
                    "text": "Systems in Annex III are high-risk.",
                    "score": 0.9,
                },
                {
                    "citation": "Art. 5(1)",
                    "text": "Social scoring is prohibited.",
                    "score": 0.5,
                },
            ],
        }

    def test_answer_fn_gets_question_and_lang(self, client, answer_fn):
        client.post("/ask", json={"question": "Is recruitment AI high-risk?", "lang": "en"})
        assert answer_fn.calls == [("Is recruitment AI high-risk?", "en")]

    def test_language_is_detected_when_missing(self, client, answer_fn):
        response = client.post("/ask", json={"question": "Welche KI-Praktiken sind verboten?"})
        assert answer_fn.calls == [("Welche KI-Praktiken sind verboten?", "de")]
        assert response.json()["lang"] == "de"

    def test_given_language_wins_over_detection(self, client, answer_fn):
        client.post(
            "/ask",
            json={"question": "Welche KI-Praktiken sind verboten?", "lang": "en"},
        )
        assert answer_fn.calls[0][1] == "en"

    def test_refusal(self):
        refusal = Answer(REFUSAL["en"], [], [], [], True)
        client = TestClient(create_app(FakeAnswerFn(refusal)))
        body = client.post("/ask", json={"question": "What about crypto?"}).json()
        assert body["refused"] is True
        assert body["sources"] == [] and body["citations"] == []

    @pytest.mark.parametrize(
        "body",
        [{}, {"question": ""}, {"question": "ok", "lang": "fr"}, {"lang": "en"}],
    )
    def test_invalid_request(self, client, answer_fn, body):
        assert client.post("/ask", json=body).status_code == 422
        assert answer_fn.calls == []

    def test_llm_error_gives_503(self):
        client = TestClient(create_app(FakeAnswerFn(error=LLMError("Is Ollama running?"))))
        response = client.post("/ask", json={"question": "q"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Is Ollama running?"}

    def test_each_app_has_its_own_answer_fn(self):
        one, two = FakeAnswerFn(), FakeAnswerFn()
        TestClient(create_app(one)).post("/ask", json={"question": "q"})
        TestClient(create_app(two))
        assert len(one.calls) == 1 and two.calls == []

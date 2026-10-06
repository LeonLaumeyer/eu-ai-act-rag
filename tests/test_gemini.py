from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from aiact_rag.config import LLMConfig
from aiact_rag.llm.base import LLMError
from aiact_rag.llm.gemini import make_gemini
from aiact_rag.pipeline import build_llm


def reply(text, finish_reason="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)])


def api_error(code, message="something went wrong"):
    return errors.APIError(code, {"error": {"message": message, "status": "ERROR"}})


class FakeGenai:
    """Looks like ``genai.Client`` as far as make_gemini needs it.

    ``outcomes`` are returned (or raised, if they are exceptions) one per call; the last one repeats.
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def gemini(client, **kwargs):
    return make_gemini("gemini-test", client=client, **kwargs)


class TestGemini:
    def test_returns_the_answer_text(self):
        assert gemini(FakeGenai(reply("Hello!")))("be brief", "say hello") == "Hello!"

    def test_request(self):
        client = FakeGenai(reply("ok"))
        gemini(client)("the system text", "the user text")

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["model"] == "gemini-test"
        assert call["contents"] == "the user text"
        assert call["config"].system_instruction == "the system text"
        assert call["config"].temperature == 0

    def test_rate_limit_is_retried_after_a_wait(self):
        client = FakeGenai(api_error(429), api_error(429), reply("finally"))
        waits = []
        assert gemini(client, retry_wait=7.0, sleep=waits.append)("s", "u") == "finally"
        assert waits == [7.0, 7.0]
        assert len(client.calls) == 3

    def test_rate_limit_gives_up_after_max_retries(self):
        client = FakeGenai(api_error(429, "quota exceeded"))
        waits = []
        with pytest.raises(LLMError, match="429") as info:
            gemini(client, max_retries=2, sleep=waits.append)("s", "u")
        assert len(client.calls) == 3 and len(waits) == 2
        assert "quota exceeded" in str(info.value)
        assert isinstance(info.value.__cause__, errors.APIError)

    @pytest.mark.parametrize("code", [500, 503])
    def test_server_error_is_retried_with_a_growing_wait(self, code):
        client = FakeGenai(api_error(code), api_error(code), api_error(code), reply("finally"))
        waits = []
        assert gemini(client, overload_wait=5.0, sleep=waits.append)("s", "u") == "finally"
        assert waits == [5.0, 10.0, 20.0]
        assert len(client.calls) == 4

    def test_server_error_gives_up_after_max_retries(self):
        client = FakeGenai(api_error(503, "high demand"))
        waits = []
        with pytest.raises(LLMError, match="503") as info:
            gemini(client, max_retries=2, overload_wait=1.0, sleep=waits.append)("s", "u")
        assert len(client.calls) == 3 and waits == [1.0, 2.0]
        assert "high demand" in str(info.value)

    def test_rate_limit_and_server_error_share_the_retries(self):
        client = FakeGenai(api_error(503), api_error(429), api_error(503), reply("finally"))
        waits = []
        answer = gemini(client, retry_wait=20.0, overload_wait=5.0, sleep=waits.append)("s", "u")
        assert answer == "finally"
        assert waits == [5.0, 20.0, 10.0]

    def test_other_api_errors_are_not_retried(self):
        client = FakeGenai(api_error(404, "model not found"))
        with pytest.raises(LLMError, match="404") as info:
            gemini(client, sleep=lambda seconds: None)("s", "u")
        assert len(client.calls) == 1
        assert "gemini-test" in str(info.value)

    @pytest.mark.parametrize("text", [None, ""])
    def test_blocked_or_empty_answer(self, text):
        with pytest.raises(LLMError, match="SAFETY"):
            gemini(FakeGenai(reply(text, finish_reason="SAFETY")))("s", "u")

    def test_no_candidates(self):
        with pytest.raises(LLMError, match="no candidates"):
            gemini(FakeGenai(SimpleNamespace(text=None, candidates=None)))("s", "u")

    def test_timeout_and_connection_errors(self):
        with pytest.raises(LLMError, match="timeout"):
            gemini(FakeGenai(httpx.ReadTimeout("too slow")))("s", "u")
        with pytest.raises(LLMError, match="Cannot reach"):
            gemini(FakeGenai(httpx.ConnectError("no network")))("s", "u")

    def test_other_errors_are_not_hidden(self):
        with pytest.raises(KeyError):
            gemini(FakeGenai(KeyError("a bug in our code")))("s", "u")

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            make_gemini("gemini-test")

    def test_key_from_environment_or_argument(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        assert callable(make_gemini("gemini-test"))
        monkeypatch.delenv("GEMINI_API_KEY")
        assert callable(make_gemini("gemini-test", api_key="given-key"))


class FakeTime:
    """A clock that only moves when someone sleeps (or the test moves it)."""

    def __init__(self):
        self.now = 100.0
        self.waits = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class TestPacing:
    def paced(self, client, fake_time, **kwargs):
        return gemini(client, sleep=fake_time.sleep, clock=fake_time.clock, **kwargs)

    def test_no_wait_before_the_first_request(self):
        fake_time = FakeTime()
        self.paced(FakeGenai(reply("ok")), fake_time, min_interval=4.0)("s", "u")
        assert fake_time.waits == []

    def test_requests_right_after_each_other_are_spaced(self):
        fake_time = FakeTime()
        generate = self.paced(FakeGenai(reply("ok")), fake_time, min_interval=4.0)
        for _ in range(3):
            generate("s", "u")
        assert fake_time.waits == [4.0, 4.0]

    def test_only_the_rest_of_the_interval_is_waited(self):
        fake_time = FakeTime()
        generate = self.paced(FakeGenai(reply("ok")), fake_time, min_interval=4.0)
        generate("s", "u")
        fake_time.now += 1.5  # the answer took a while, or retrieval ran in between
        generate("s", "u")
        assert fake_time.waits == [2.5]

    def test_no_wait_when_the_interval_has_passed(self):
        fake_time = FakeTime()
        generate = self.paced(FakeGenai(reply("ok")), fake_time, min_interval=4.0)
        generate("s", "u")
        fake_time.now += 10.0
        generate("s", "u")
        assert fake_time.waits == []

    def test_a_retry_wait_counts_towards_the_interval(self):
        fake_time = FakeTime()
        client = FakeGenai(api_error(429), reply("ok"))
        self.paced(client, fake_time, min_interval=4.0, retry_wait=20.0)("s", "u")
        assert fake_time.waits == [20.0]

    def test_off_by_default(self):
        fake_time = FakeTime()
        generate = self.paced(FakeGenai(reply("ok")), fake_time)
        generate("s", "u")
        generate("s", "u")
        assert fake_time.waits == []


def test_build_llm_makes_a_gemini_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    assert callable(build_llm(LLMConfig(provider="gemini", model="gemini-test")))

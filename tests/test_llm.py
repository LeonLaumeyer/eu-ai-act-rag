import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from aiact_rag.llm.anthropic_llm import make_claude
from aiact_rag.llm.base import LLMError
from aiact_rag.llm.ollama import make_ollama


def fake_ollama_client(handler) -> httpx.Client:
    """An httpx client that sends every request to ``handler`` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOllama:
    def test_returns_the_answer_text(self):
        def handler(request):
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello!"}})

        generate = make_ollama("gemma", client=fake_ollama_client(handler))
        assert generate("be brief", "say hello") == "Hello!"

    def test_request(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"message": {"content": "ok"}})

        generate = make_ollama(
            "gemma", base_url="http://myhost:1234", client=fake_ollama_client(handler)
        )
        generate("the system text", "the user text")

        assert len(seen) == 1
        assert seen[0].method == "POST"
        assert str(seen[0].url) == "http://myhost:1234/api/chat"
        body = json.loads(seen[0].content)
        assert body["model"] == "gemma"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert body["messages"] == [
            {"role": "system", "content": "the system text"},
            {"role": "user", "content": "the user text"},
        ]

    def test_can_be_called_many_times(self):
        answers = iter(["one", "two"])

        def handler(request):
            return httpx.Response(200, json={"message": {"content": next(answers)}})

        generate = make_ollama("gemma", client=fake_ollama_client(handler))
        assert [generate("s", "u"), generate("s", "u")] == ["one", "two"]

    def test_server_not_running(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        generate = make_ollama(
            "gemma", base_url="http://myhost:1234", client=fake_ollama_client(handler)
        )
        with pytest.raises(LLMError, match="myhost:1234") as info:
            generate("s", "u")
        assert isinstance(info.value.__cause__, httpx.ConnectError)

    def test_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(LLMError, match="timeout"):
            make_ollama("gemma", client=fake_ollama_client(handler))("s", "u")

    def test_unknown_model(self):
        def handler(request):
            return httpx.Response(404, json={"error": "model 'nope' not found"})

        generate = make_ollama("nope", client=fake_ollama_client(handler))
        with pytest.raises(LLMError, match="404") as info:
            generate("s", "u")
        assert "nope" in str(info.value)

    def test_making_the_function_sends_nothing(self):
        def handler(request):
            raise AssertionError("no request expected")

        make_ollama("gemma", client=fake_ollama_client(handler))


class FakeAnthropic:
    """Looks like ``anthropic.Anthropic()`` as far as make_claude needs it."""

    def __init__(self, answer="Hello!", error=None):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._answer, self._error = answer, error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._answer)])


class TestClaude:
    def test_returns_the_answer_text(self):
        generate = make_claude(client=FakeAnthropic("Hello!"))
        assert generate("be brief", "say hello") == "Hello!"

    def test_request(self):
        client = FakeAnthropic()
        make_claude(model="claude-test", max_tokens=77, client=client)(
            "the system text", "the user text"
        )
        assert client.calls == [
            {
                "model": "claude-test",
                "max_tokens": 77,
                "system": "the system text",
                "messages": [{"role": "user", "content": "the user text"}],
            }
        ]

    def test_api_error_becomes_llm_error(self):
        error = anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com")
        )
        generate = make_claude(model="claude-test", client=FakeAnthropic(error=error))
        with pytest.raises(LLMError, match="claude-test") as info:
            generate("s", "u")
        assert info.value.__cause__ is error

    def test_other_errors_are_not_hidden(self):
        generate = make_claude(client=FakeAnthropic(error=KeyError("a bug in our code")))
        with pytest.raises(KeyError):
            generate("s", "u")

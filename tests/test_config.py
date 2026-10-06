from dataclasses import replace

import pytest

from aiact_rag.config import (
    Config,
    LLMConfig,
    RetrievalConfig,
    llm_config,
    load_config,
    load_llm_config,
)

YAML = """
llm:
  provider: ollama
  ollama:  { base_url: "http://myhost:1234", model: "gemma" }
  anthropic: { model: "claude-test", max_tokens: 500 }
retrieval:
  chunking: fixed
  mode: bm25
  rerank: false
  top_k: 3
"""


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_all_values(self, tmp_path):
        config = load_config(write(tmp_path, YAML))
        assert config.retrieval == RetrievalConfig(
            chunking="fixed", mode="bm25", rerank=False, top_k=3
        )
        assert config.llm == LLMConfig(
            provider="ollama", model="gemma", base_url="http://myhost:1234"
        )

    def test_the_active_provider_decides_the_llm_block(self, tmp_path):
        config = load_config(
            write(tmp_path, YAML.replace("provider: ollama", "provider: anthropic"))
        )
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-test"
        assert config.llm.max_tokens == 500

    def test_missing_retrieval_section_uses_defaults(self, tmp_path):
        config = load_config(write(tmp_path, "llm: {provider: ollama, ollama: {model: gemma}}"))
        assert config.retrieval == RetrievalConfig()

    def test_the_real_config_file_is_valid(self):
        assert isinstance(load_config(), Config)

    @pytest.mark.parametrize(
        "old, new, key",
        [
            ("mode: bm25", "mode: magic", "retrieval.mode"),
            ("chunking: fixed", "chunking: sentence", "retrieval.chunking"),
            ("rerank: false", "rerank: maybe", "retrieval.rerank"),
            ("top_k: 3", "top_k: 0", "retrieval.top_k"),
            ("provider: ollama", "provider: gpt", "llm.provider"),
            ('model: "gemma"', 'model: ""', "llm.ollama.model"),
        ],
    )
    def test_invalid_value_names_the_key(self, tmp_path, old, new, key):
        assert old in YAML
        with pytest.raises(ValueError, match=key):
            load_config(write(tmp_path, YAML.replace(old, new)))

    def test_unknown_key_is_an_error(self, tmp_path):
        with pytest.raises(TypeError, match="topk"):
            load_config(write(tmp_path, YAML.replace("top_k: 3", "topk: 3")))


class TestChangingValues:
    def test_replace_makes_a_changed_copy(self):
        base = RetrievalConfig()
        dense = replace(base, mode="dense", rerank=False)
        assert (dense.mode, dense.rerank, dense.chunking) == ("dense", False, "article")
        assert base.mode == "hybrid"

    def test_replace_is_checked_too(self):
        with pytest.raises(ValueError, match="retrieval.mode"):
            replace(RetrievalConfig(), mode="magic")

    def test_a_config_cannot_be_changed_in_place(self):
        with pytest.raises(AttributeError):
            RetrievalConfig().mode = "dense"


class TestLLMConfig:
    SECTION = {
        "provider": "ollama",
        "ollama": {"model": "gemma"},
        "anthropic": {"model": "claude-test"},
    }

    def test_active_provider(self):
        assert llm_config(self.SECTION) == LLMConfig(provider="ollama", model="gemma")

    def test_other_provider_for_the_judge(self):
        assert llm_config(self.SECTION, "anthropic") == LLMConfig(
            provider="anthropic", model="claude-test"
        )

    def test_provider_without_a_block(self):
        with pytest.raises(ValueError, match="llm.anthropic.model"):
            llm_config({"provider": "ollama", "ollama": {"model": "gemma"}}, "anthropic")

    def test_min_interval_of_gemini(self):
        section = {"gemini": {"model": "gemini-test", "min_interval": 4}}
        assert llm_config(section, "gemini").min_interval == 4
        assert LLMConfig(provider="gemini", model="gemini-test").min_interval == 0.0

    @pytest.mark.parametrize("value", [-1, "4", True, None])
    def test_min_interval_must_be_a_number_from_zero(self, value):
        with pytest.raises(ValueError, match="llm.gemini.min_interval"):
            LLMConfig(provider="gemini", model="gemini-test", min_interval=value)

    def test_load_llm_config(self, tmp_path):
        path = write(tmp_path, YAML)
        assert load_llm_config(path=path).model == "gemma"
        assert load_llm_config("anthropic", path=path).model == "claude-test"

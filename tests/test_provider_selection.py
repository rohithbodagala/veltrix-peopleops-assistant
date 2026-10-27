"""LLM_PROVIDER=auto: a configured key wins; with no key at all the offline stand-in runs."""
from app.config import load_settings
from app.llm.client import build_provider


def test_auto_without_any_key_runs_the_offline_stand_in():
    assert build_provider(load_settings({"LLM_PROVIDER": "auto"})).name == "fake"


def test_auto_prefers_a_configured_key_over_the_stand_in():
    provider = build_provider(load_settings({"LLM_PROVIDER": "auto", "OPENAI_API_KEY": "sk-test"}))
    assert provider.name == "openai"


def test_named_provider_without_its_key_does_not_fall_back():
    assert build_provider(load_settings({"LLM_PROVIDER": "openai"})) is None

"""
Tests fuer die einmalige temperature-Capability-Pruefung.

Anforderung: Ob das LLM den Parameter `temperature` akzeptiert, wird GENAU
EINMAL geprueft - noch vor der Relevanzpruefung. Ab dann laufen alle Requests
konsequent mit oder konsequent ohne `temperature` (kein Ausprobieren pro Call).
"""

import asyncio
from types import SimpleNamespace

import pytest

from QCA_AID_assets.utils.llm.capabilities import (
    PROBE_PROMPT,
    ensure_temperature_capability,
    get_temperature_support,
    reset_temperature_probe_cache,
)
from QCA_AID_assets.utils.tracking.token_tracker import get_global_token_counter


class FakeProvider:
    """Minimaler Provider-Ersatz.

    Bildet das Verhalten der echten Provider nach: bei Erfolg mit
    `temperature` wird die Capability auf True gesetzt; bei einem
    temperature-Fehler auf False.
    """

    def __init__(self, supports_temperature=True, error=None, records=True):
        self.supports_temperature = supports_temperature
        self.error = error
        self.call_count = 0
        self.calls = []
        self.records = records

    async def create_completion(self, model, messages, temperature=0.7, **kwargs):
        self.call_count += 1
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
        )

        if "temperature" in str(self.error).lower():
            # Modell lehnt temperature ab -> Capability False (wie echte Provider)
            if self.records:
                get_global_token_counter().model_capabilities[model] = False
            raise self.error

        if self.error is not None:
            # Fehler ohne temperature-Bezug -> Ergebnis bleibt offen
            raise self.error

        if self.records and temperature is not None:
            get_global_token_counter().model_capabilities[model] = True
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture(autouse=True)
def _clean_capability_state():
    reset_temperature_probe_cache()
    get_global_token_counter().model_capabilities.clear()
    yield
    reset_temperature_probe_cache()
    get_global_token_counter().model_capabilities.clear()


def test_probe_runs_exactly_once_for_supported_model():
    provider = FakeProvider(supports_temperature=True)

    first = asyncio.run(
        ensure_temperature_capability(provider, "test-model-supported")
    )
    second = asyncio.run(
        ensure_temperature_capability(provider, "test-model-supported")
    )

    assert first is True
    assert second is True
    assert provider.call_count == 1, "temperature darf nur einmal geprueft werden"
    assert provider.calls[0]["messages"][0]["content"] == PROBE_PROMPT
    assert provider.calls[0]["temperature"] == 0.0


def test_probe_detects_unsupported_model_once():
    provider = FakeProvider(
        supports_temperature=False,
        error=TypeError("Unsupported parameter: 'temperature' is not supported"),
    )

    first = asyncio.run(ensure_temperature_capability(provider, "test-model-no-temp"))
    second = asyncio.run(ensure_temperature_capability(provider, "test-model-no-temp"))

    assert first is False
    assert second is False
    assert provider.call_count == 1
    assert get_temperature_support("test-model-no-temp") is False


def test_cached_capability_skips_probe_entirely():
    get_global_token_counter().model_capabilities["test-model-cached"] = False
    provider = FakeProvider(supports_temperature=True)

    result = asyncio.run(ensure_temperature_capability(provider, "test-model-cached"))

    assert result is False
    assert provider.call_count == 0


def test_inconclusive_probe_defaults_to_temperature_and_is_cached():
    provider = FakeProvider(error=RuntimeError("connection reset"))

    result = asyncio.run(ensure_temperature_capability(provider, "test-model-broken"))

    assert result is True, "Fallback ohne klares Ergebnis: weiter MIT temperature"
    assert get_temperature_support("test-model-broken") is True
    assert provider.call_count == 1

    # Zweiter Aufruf prueft nicht erneut
    asyncio.run(ensure_temperature_capability(provider, "test-model-broken"))
    assert provider.call_count == 1


def test_missing_provider_returns_unknown_without_crashing(caplog):
    result = asyncio.run(ensure_temperature_capability(None, "test-model-noprovider"))

    assert result is None
    assert any("kein LLM-Provider" in r.getMessage() for r in caplog.records)


def test_force_allows_second_probe():
    provider = FakeProvider(supports_temperature=True)

    asyncio.run(ensure_temperature_capability(provider, "test-model-force"))
    asyncio.run(ensure_temperature_capability(provider, "test-model-force", force=True))

    assert provider.call_count == 2


def test_analysis_manager_probe_is_wired_before_relevance():
    """
    `_ensure_temperature_capability()` nutzt den Provider des
    RelevanceCheckers und setzt `temperature_supported` fuer den Rest des Laufs.
    """
    from QCA_AID_assets.analysis.analysis_manager import IntegratedAnalysisManager

    provider = FakeProvider(supports_temperature=True)
    fake_self = SimpleNamespace(
        model_name="test-model-wiring",
        relevance_checker=SimpleNamespace(llm_provider=provider),
        temperature_supported=None,
    )

    supported = asyncio.run(
        IntegratedAnalysisManager._ensure_temperature_capability(fake_self)
    )

    assert supported is True
    assert fake_self.temperature_supported is True
    assert provider.call_count == 1

    # Zweiter Aufruf (z. B. Retry des Laufs) nutzt den Cache
    asyncio.run(IntegratedAnalysisManager._ensure_temperature_capability(fake_self))
    assert provider.call_count == 1


class _RejectingCompletions:
    """OpenAI-kompatibler Client, der 'temperature' wie ein Reasoning-Modell ablehnt."""

    def __init__(self):
        self.requests = []

    async def create(self, **params):
        self.requests.append(dict(params))
        if "temperature" in params:
            raise TypeError(
                "Error code: 400 - Unsupported parameter: 'temperature' is not "
                "supported with this model."
            )
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}


def test_real_provider_switches_to_without_temperature_after_probe():
    """
    Nach der einmaligen Pruefung muss der echte Provider-Codepfad
    'ohne temperature' weiterlaufen (weiche mit/ohne).
    """
    from QCA_AID_assets.utils.llm.openai_provider import OpenAIProvider

    provider = object.__new__(OpenAIProvider)
    completions = _RejectingCompletions()
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    provider.model_name = "test-provider-no-temp"

    supported = asyncio.run(
        ensure_temperature_capability(provider, "test-provider-no-temp")
    )
    assert supported is False
    # Pruefung: erster Call mit temperature (abgelehnt), Retry ohne temperature
    assert "temperature" in completions.requests[0]
    assert "temperature" not in completions.requests[1]

    # Alle weiteren Calls: temperature wird gar nicht mehr mitgeschickt
    completions.requests.clear()
    asyncio.run(
        provider.create_completion(
            model="test-provider-no-temp",
            messages=[{"role": "user", "content": "hallo"}],
            temperature=0.7,
        )
    )

    assert len(completions.requests) == 1
    assert "temperature" not in completions.requests[0]

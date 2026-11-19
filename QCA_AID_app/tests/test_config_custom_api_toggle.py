"""
Tests fuer den Custom-API-Schalter in der Konfigurations-UI.

Anforderung: Beim Provider 'OpenAI / Custom API Provider' wird eine Custom
Base URL nur uebernommen, wenn das Haekchen 'Custom API Base URL verwenden'
gesetzt ist. Ohne Haekchen gilt der Standard-Cloud-Provider und eine
gespeicherte Custom-Einstellung wird zurueckgesetzt.

Getestet werden die reinen Helfer aus `webapp_components.config_ui`
(Streamlit-frei); das Rendering selbst wird nicht ausgefuehrt.
"""

from types import SimpleNamespace

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp_components.config_ui import (
    CUSTOM_ENDPOINT_PROVIDERS,
    is_valid_api_base_url,
    normalize_api_base_url,
    provider_supports_custom_endpoint,
    reset_custom_api_settings,
)
from webapp_models.config_data import ConfigData


def _config(**kwargs) -> ConfigData:
    base = dict(
        model_provider="openai",
        model_name="gpt-4o-mini",
        data_dir="input",
        output_dir="output",
        chunk_size=1200,
        chunk_overlap=50,
        batch_size=8,
        code_with_context=False,
        multiple_codings=True,
        multiple_coding_threshold=0.65,
        analysis_mode="deductive",
        review_mode="consensus",
        attribute_labels={},
        coder_settings=[],
    )
    base.update(kwargs)
    return ConfigData(**base)


def test_custom_endpoint_only_for_openai_and_local():
    assert provider_supports_custom_endpoint("openai") is True
    # Alias, den die Factory ebenfalls akzeptiert
    assert provider_supports_custom_endpoint("gpt") is True
    assert provider_supports_custom_endpoint("local") is True

    for provider in ("anthropic", "mistral", "openrouter"):
        assert provider_supports_custom_endpoint(provider) is False

    assert set(CUSTOM_ENDPOINT_PROVIDERS) == {"openai", "gpt", "local"}


def test_normalize_api_base_url():
    assert normalize_api_base_url(None) is None
    assert normalize_api_base_url("") is None
    assert normalize_api_base_url("   ") is None
    assert (
        normalize_api_base_url("  https://chat-ai.academiccloud.de/v1  ")
        == "https://chat-ai.academiccloud.de/v1"
    )


def test_is_valid_api_base_url():
    assert is_valid_api_base_url("https://example.org/v1")
    assert is_valid_api_base_url("http://localhost:1234/v1")
    assert not is_valid_api_base_url("chat-ai.academiccloud.de/v1")
    assert not is_valid_api_base_url("")
    assert not is_valid_api_base_url(None)


def test_reset_custom_api_settings_clears_url_and_env():
    config = _config(
        api_base_url="https://chat-ai.academiccloud.de/v1",
        custom_api_key_env="GWDG_API_KEY",
    )

    assert reset_custom_api_settings(config) is True
    assert config.api_base_url is None
    assert config.custom_api_key_env is None

    # Persistiert als Standard-Cloud-Provider (kein Custom-Endpoint)
    as_dict = config.to_dict()
    assert as_dict["api_base_url"] is None
    assert as_dict["custom_api_key_env"] is None


def test_reset_clears_env_even_without_url():
    """Wird die URL geleert, darf auch die Custom-Env-Variable nicht stehen bleiben."""
    config = _config(api_base_url=None, custom_api_key_env="GWDG_API_KEY")

    assert reset_custom_api_settings(config) is True
    assert config.api_base_url is None
    assert config.custom_api_key_env is None


def test_reset_custom_api_settings_is_idempotent():
    already_default = _config(api_base_url=None, custom_api_key_env=None)
    assert reset_custom_api_settings(already_default) is False

    # Beide Felder bleiben unveraendert
    assert already_default.api_base_url is None
    assert already_default.custom_api_key_env is None


def test_reset_works_with_legacy_objects_without_fields():
    """Alte Session-State-Objekte (SimpleNamespace) duerfen nicht crashen."""
    legacy = SimpleNamespace()
    assert reset_custom_api_settings(legacy) is False

    legacy_with_url = SimpleNamespace(api_base_url="https://x/v1")
    assert reset_custom_api_settings(legacy_with_url) is True
    assert legacy_with_url.api_base_url is None
    assert legacy_with_url.custom_api_key_env is None


def test_checked_custom_endpoint_is_stored_untouched():
    """Mit Haekchen: Base URL wird unveraendert (getrimmt) uebernommen."""
    config = _config(api_base_url=None)
    checked = True

    if checked:  # Verhalten der UI bei gesetztem Haekchen
        config.api_base_url = normalize_api_base_url(
            " https://chat-ai.academiccloud.de/v1 "
        )
    else:  # pragma: no cover - nur zur Dokumentation
        reset_custom_api_settings(config)

    assert config.api_base_url == "https://chat-ai.academiccloud.de/v1"

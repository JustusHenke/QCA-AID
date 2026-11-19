"""
Streamlit-Rendertests fuer den Custom-API-Schalter in den Modell-Einstellungen.

Nutzt `streamlit.testing.v1.AppTest`, um `render_model_settings()` headless zu
rendern (kein API-Call, Provider-Manager arbeitet aus dem lokalen Cache/den
mitgelieferten Configs).

Geprueft wird:
- OpenAI / Custom API Provider: Haekchen vorhanden, Default an, wenn eine
  Custom Base URL gespeichert ist.
- Haekchen entfernen -> `api_base_url` und `custom_api_key_env` werden
  zurueckgesetzt (Standard-Cloud-Provider), kein Render-Fehler.
- Andere Provider (z. B. Anthropic): keine Custom-API-Sektion, gespeicherte
  Einstellungen bleiben unangetastet.
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

SCRIPT_TEMPLATE = """
from types import SimpleNamespace
import streamlit as st
from webapp_models.config_data import ConfigData, CoderSetting
from webapp_components.config_ui import render_model_settings

if "project_manager" not in st.session_state:
    st.session_state.project_manager = SimpleNamespace(
        get_root_directory=lambda: ".",
        path_resolver=None,
    )
if "config_data" not in st.session_state:
    st.session_state.config_data = ConfigData(
        model_provider="{provider}", model_name="{model}", data_dir="input",
        output_dir="output", chunk_size=1200, chunk_overlap=50, batch_size=8,
        code_with_context=False, multiple_codings=True, multiple_coding_threshold=0.65,
        analysis_mode="deductive", review_mode="consensus", attribute_labels={{}},
        coder_settings=[CoderSetting(0.3, "auto_1")],
        api_base_url={base_url},
        custom_api_key_env={key_env},
    )
render_model_settings()
"""

CHECKBOX_LABEL = "Custom API Base URL verwenden"


def _run(provider: str, model: str, base_url: str, key_env: str) -> AppTest:
    script = SCRIPT_TEMPLATE.format(
        provider=provider,
        model=model,
        base_url=f'"{base_url}"' if base_url else "None",
        key_env=f'"{key_env}"' if key_env else "None",
    )
    return AppTest.from_string(script, default_timeout=180).run()


def _find_checkbox(app: AppTest):
    for checkbox in app.checkbox:
        if checkbox.label == CHECKBOX_LABEL:
            return checkbox
    return None


def test_openai_shows_checked_checkbox_and_resets_on_uncheck():
    app = _run(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://chat-ai.academiccloud.de/v1",
        key_env="GWDG_API_KEY",
    )
    assert not app.exception, [e.value for e in app.exception]

    checkbox = _find_checkbox(app)
    assert checkbox is not None, "Haekchen 'Custom API Base URL verwenden' fehlt"
    assert checkbox.value is True, "Bei gespeicherter Custom-URL muss es angehakt sein"

    # Haekchen entfernen -> Standard-Cloud-Provider, Custom-Einstellung weg
    _find_checkbox(app).uncheck().run()

    assert not app.exception, [e.value for e in app.exception]
    config = app.session_state["config_data"]
    assert config.api_base_url is None
    assert config.custom_api_key_env is None
    assert config.model_provider == "openai"
    assert "config_modified" in app.session_state
    assert app.session_state["config_modified"] is True


def test_openai_without_custom_url_has_unchecked_checkbox():
    app = _run(
        provider="openai",
        model="gpt-4o-mini",
        base_url="",
        key_env="",
    )
    assert not app.exception, [e.value for e in app.exception]

    checkbox = _find_checkbox(app)
    assert checkbox is not None
    assert checkbox.value is False

    config = app.session_state["config_data"]
    assert config.api_base_url is None
    assert config.custom_api_key_env is None


def test_other_providers_hide_custom_api_section_and_keep_settings():
    app = _run(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        base_url="https://example.org/v1",
        key_env="CUSTOM_API_KEY",
    )
    assert not app.exception, [e.value for e in app.exception]

    assert _find_checkbox(app) is None, "Custom-API-Haekchen darf hier nicht erscheinen"
    expander_labels = [e.label for e in app.expander]
    assert not any("Custom API Base URL" in label for label in expander_labels)
    assert not any("Server-URL" in label for label in expander_labels)

    # Gespeicherte Einstellung bleibt unangetastet (kein Loeschen im Fremd-Provider)
    config = app.session_state["config_data"]
    assert config.api_base_url == "https://example.org/v1"
    assert config.custom_api_key_env == "CUSTOM_API_KEY"

"""
Einmalige Capability-Prüfung für LLM-Modelle
=============================================

Manche Modelle (z. B. OpenAI-Reasoning-Modelle wie o1/o3 oder Modelle mit
fest verdrahteter Temperatur) lehnen den Parameter ``temperature`` mit einem
400er-Fehler ab. Bisher wurde das pro API-Call erneut "ausprobiert": jeder
Call mit unbekannter Capability schickte ``temperature`` mit, scheiterte
gegebenenfalls und wurde ohne ``temperature`` wiederholt.

Diese Prüfung passiert jetzt **genau einmal** vor der Relevanzprüfung
(:meth:`QCA_AID_assets.analysis.analysis_manager.IntegratedAnalysisManager.analyze_material`).
Danach steht das Ergebnis im globalen Capability-Cache
(``token_counter.model_capabilities``) und alle Provider nehmen entweder
konsequent mit oder konsequent ohne ``temperature`` - ohne weitere
Fehlversuche und Retries.
"""

import logging
from typing import Optional

from ..tracking.token_tracker import get_global_token_counter

logger = logging.getLogger(__name__)

# Minimaler Prompt: Die Antwort wird verworfen, es geht nur um die
# Parameter-Akzeptanz des Modells.
PROBE_PROMPT = "ping"

# Modelle, für die in diesem Prozess bereits geprüft wurde. Verhindert eine
# zweite Prüfung, auch wenn das Modell (z. B. nach einem Netzwerkfehler) keinen
# Eintrag im Capability-Cache hat.
_PROBED_MODELS: set = set()


def reset_temperature_probe_cache() -> None:
    """Setzt die Prüf-Marker zurück (nur für Tests/Neustart gedacht)."""
    _PROBED_MODELS.clear()


def get_temperature_support(model_name: str) -> Optional[bool]:
    """
    Liefert das gecachte Prüfergebnis für ein Modell.

    Returns:
        True  - Modell akzeptiert ``temperature``
        False - Modell lehnt ``temperature`` ab
        None  - noch nicht geprüft
    """
    return get_global_token_counter().model_capabilities.get(str(model_name))


def set_temperature_support(model_name: str, supported: bool) -> None:
    """Schreibt das Prüfergebnis in den globalen Capability-Cache."""
    get_global_token_counter().model_capabilities[str(model_name)] = bool(supported)


async def ensure_temperature_capability(
    provider,
    model_name: str,
    probe_temperature: float = 0.0,
    force: bool = False,
) -> Optional[bool]:
    """
    Prüft genau einmal, ob ``model_name`` den Parameter ``temperature`` akzeptiert.

    Der Test ist ein einzelner, minimaler Completion-Aufruf mit ``temperature``.
    Die Provider-Implementierungen setzen dabei selbst den Capability-Cache:
    Erfolg -> ``True``, temperature-Fehler -> ``False`` (inkl. Retry ohne
    ``temperature``).

    Args:
        provider: Initialisierter LLM-Provider (``LLMProvider``-Instanz).
        model_name: Name des zu prüfenden Modells.
        probe_temperature: Temperatur für den Testaufruf (Default 0.0 = deterministisch).
        force: Erzwingt eine erneute Prüfung, auch wenn bereits ein Ergebnis
            gecacht ist (nur für Tests/Diagnose).

    Returns:
        True/False wenn ermittelbar; None, wenn kein Provider übergeben wurde.
    """
    model_key = str(model_name)
    token_counter = get_global_token_counter()
    capabilities = token_counter.model_capabilities

    if not force and model_key in capabilities:
        return capabilities[model_key]

    if not force and model_key in _PROBED_MODELS:
        # Wurde in diesem Prozess bereits geprüft - nie ein zweites Mal.
        return capabilities.get(model_key)

    # Marker VOR dem Await setzen, damit parallele Aufrufe nicht doppelt prüfen.
    _PROBED_MODELS.add(model_key)

    if provider is None:
        logger.warning(
            "Temperature-Prüfung für '%s' übersprungen: kein LLM-Provider verfügbar.",
            model_key,
        )
        return capabilities.get(model_key)

    try:
        await provider.create_completion(
            model=model_key,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            temperature=probe_temperature,
        )
    except Exception as exc:
        # Häufigster Fall: Modell/Provider nicht erreichbar - dann ist das
        # Ergebnis unklar und wir fallen unten auf den Default zurück.
        logger.warning(
            "Temperature-Prüfung für '%s' fehlgeschlagen: %s", model_key, exc
        )

    supported = capabilities.get(model_key)

    if supported is None:
        # Kein eindeutiges Ergebnis (z. B. Netzwerk-/Auth-Fehler, oder der
        # Provider-Call lief mit temperature durch, ohne den Cache zu setzen).
        # Default: weiterhin MIT temperature arbeiten (bisheriges Verhalten),
        # aber den Wert cachen, damit nicht bei jedem Call neu "probiert" wird.
        supported = True
        capabilities[model_key] = supported
        logger.warning(
            "Temperature-Unterstützung für '%s' konnte nicht eindeutig ermittelt "
            "werden - Fallback: Calls MIT temperature.",
            model_key,
        )

    return supported


def describe_temperature_support(model_name: str) -> str:
    """Kurze, logbare Beschreibung des Prüfergebnisses für ein Modell."""
    supported = get_temperature_support(model_name)
    if supported is None:
        return "unbekannt (noch nicht geprüft)"
    return "mit temperature" if supported else "ohne temperature"


__all__ = [
    "PROBE_PROMPT",
    "ensure_temperature_capability",
    "get_temperature_support",
    "set_temperature_support",
    "describe_temperature_support",
    "reset_temperature_probe_cache",
]

"""
Data Models for LLM Provider Manager

Defines data structures for normalized models, cache metadata, and custom exceptions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class NormalizedModel:
    """
    Einheitliches Modell-Format für alle Provider.
    
    Normalisiert Modelle von verschiedenen Providern (OpenAI, Anthropic, Mistral,
    OpenRouter, lokale Modelle) in ein konsistentes Format.
    
    Attributes:
        provider: Provider-Name (z.B. 'openai', 'anthropic', 'mistral', 'openrouter', 'local')
        model_id: Eindeutige Modell-ID (z.B. 'gpt-4o-mini', 'claude-3-opus')
        model_name: Anzeigename des Modells (z.B. 'GPT-4o Mini', 'Claude 3 Opus')
        context_window: Maximale Anzahl von Tokens im Context Window (optional)
        cost_in: Kosten pro 1M Input-Tokens in USD (optional)
        cost_out: Kosten pro 1M Output-Tokens in USD (optional)
        options: Dictionary für zusätzliche Provider-spezifische Eigenschaften
    
    Example:
        >>> model = NormalizedModel(
        ...     provider='openai',
        ...     model_id='gpt-4o-mini',
        ...     model_name='GPT-4o Mini',
        ...     context_window=128000,
        ...     cost_in=0.15,
        ...     cost_out=0.60,
        ...     options={'supports_attachments': True, 'can_reason': False}
        ... )
    """
    provider: str
    model_id: str
    model_name: str
    context_window: Optional[int] = None
    cost_in: Optional[float] = None
    cost_out: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    def supports_capability(self, capability: str) -> bool:
        """
        Prüft ob Modell eine bestimmte Capability unterstützt.
        
        Sucht nach der Capability in options['capabilities'] Liste oder
        direkt als boolean Flag in options.
        
        Args:
            capability: Name der zu prüfenden Capability (z.B. 'can_reason', 'supports_attachments')
            
        Returns:
            bool: True wenn Capability unterstützt wird, sonst False
            
        Example:
            >>> model.supports_capability('can_reason')
            False
            >>> model.supports_capability('supports_attachments')
            True
        """
        # Prüfe ob capability in capabilities-Liste ist
        capabilities = self.options.get('capabilities', [])
        if isinstance(capabilities, list) and capability in capabilities:
            return True
        
        # Prüfe ob capability als direktes boolean Flag existiert
        return self.options.get(capability, False)


@dataclass
class CacheMetadata:
    """
    Metadaten für Cache-Verwaltung.
    
    Speichert Informationen über den Cache-Zustand, um TTL-Validierung
    und Cache-Versionierung zu ermöglichen.
    
    Attributes:
        timestamp: Unix-Timestamp der Cache-Erstellung
        version: Cache-Format-Version (für zukünftige Migrations-Kompatibilität)
        providers: Liste der im Cache enthaltenen Provider
    
    Example:
        >>> import time
        >>> metadata = CacheMetadata(
        ...     timestamp=time.time(),
        ...     version='1.0',
        ...     providers=['openai', 'anthropic', 'mistral']
        ... )
    """
    timestamp: float
    version: str
    providers: List[str]


# Custom Exceptions

class ProviderLoadError(Exception):
    """
    Exception für Fehler beim Laden von Provider-Konfigurationen.
    
    Wird ausgelöst wenn:
    - Catwalk-URL nicht erreichbar ist UND lokale Fallback-Datei fehlt
    - Netzwerk-Fehler auftreten
    - Provider-Konfiguration nicht geladen werden kann
    
    Example:
        >>> raise ProviderLoadError("Failed to load openai config from URL and local fallback")
    """
    pass


class ValidationError(Exception):
    """
    Exception für Validierungsfehler bei Modell-Daten.
    
    Wird ausgelöst wenn:
    - Erforderliche Felder in Modell-Daten fehlen
    - Datentypen ungültig sind
    - Modell-Struktur nicht dem erwarteten Format entspricht
    
    Example:
        >>> raise ValidationError("Model missing required field 'model_id'")
    """
    pass


class CacheError(Exception):
    """
    Exception für Cache-bezogene Fehler.
    
    Wird ausgelöst wenn:
    - Cache-Verzeichnis nicht erstellt werden kann
    - Cache-Datei nicht gelesen/geschrieben werden kann
    - Cache-Format ungültig ist
    - Cache-Metadaten korrupt sind
    
    Example:
        >>> raise CacheError("Failed to write cache file: Permission denied")
    """
    pass

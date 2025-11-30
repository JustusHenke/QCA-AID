"""
Model Registry

Verwaltet normalisierte Modelle von allen Providern.
Bietet Filter- und Such-API für Modelle.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .models import NormalizedModel, ValidationError


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Verwaltet normalisierte Modelle.
    
    Die Registry speichert alle Modelle in einem einheitlichen Format
    und bietet Filter- und Suchfunktionen. Bei duplicate model_ids
    wird der neuere Eintrag verwendet (last-write-wins).
    
    Attributes:
        _models: Dictionary mit model_id als Key und NormalizedModel als Value
        _pricing_overrides: Dictionary mit Preis-Overrides aus pricing_overrides.json
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_models('openai', raw_openai_models)
        >>> registry.register_models('anthropic', raw_anthropic_models)
        >>> all_models = registry.get_all()
        >>> gpt_models = registry.get_by_provider('openai')
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialisiert ModelRegistry mit leerem Model-Dictionary.
        
        Args:
            config_dir: Optionales Konfigurationsverzeichnis für pricing_overrides.json
                       Falls None, wird das aktuelle Verzeichnis verwendet.
        """
        self._models: Dict[str, NormalizedModel] = {}
        self._pricing_overrides: Dict[str, Dict] = {}
        self._config_dir = Path(config_dir).expanduser() if config_dir else Path.cwd()
        logger.debug("ModelRegistry initialized")
    
    def register_models(self, provider: str, raw_models: List[Dict]) -> None:
        """
        Registriert Modelle eines Providers.
        
        Normalisiert Modelle und fügt sie zur Registry hinzu.
        Bei duplicate IDs wird der neuere Eintrag verwendet (last-write-wins).
        Fehlerhafte Modelle werden ignoriert und geloggt.
        
        Args:
            provider: Provider-Name (z.B. 'openai', 'anthropic', 'openrouter', 'local')
            raw_models: Liste von Modell-Dicts im Provider-spezifischen Format
        
        Example:
            >>> raw_models = [
            ...     {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'cost_per_1m_in': 0.15, ...},
            ...     {'id': 'gpt-4o', 'name': 'GPT-4o', 'cost_per_1m_in': 2.50, ...}
            ... ]
            >>> registry.register_models('openai', raw_models)
        """
        if not raw_models:
            logger.warning(f"No models provided for provider '{provider}'")
            return
        
        if not isinstance(raw_models, list):
            logger.error(
                f"Invalid models data for provider '{provider}': "
                f"Expected list, got {type(raw_models).__name__}"
            )
            return
        
        registered_count = 0
        skipped_count = 0
        
        for idx, raw_model in enumerate(raw_models):
            try:
                # Validiere dass raw_model ein Dict ist
                if not isinstance(raw_model, dict):
                    logger.warning(
                        f"Skipping invalid model at index {idx} from provider '{provider}': "
                        f"Expected dict, got {type(raw_model).__name__}"
                    )
                    skipped_count += 1
                    continue
                
                # Normalisiere basierend auf Provider-Typ
                if provider == 'local':
                    normalized = self._normalize_local_model(raw_model)
                else:
                    # Catwalk-Provider (openai, anthropic, openrouter)
                    normalized = self._normalize_catwalk_model(provider, raw_model)
                
                # Prüfe auf Duplikate
                if normalized.model_id in self._models:
                    logger.debug(
                        f"Duplicate model_id '{normalized.model_id}' detected. "
                        f"Overwriting previous entry (last-write-wins)."
                    )
                
                # Registriere Modell (überschreibt bei Duplikaten)
                self._models[normalized.model_id] = normalized
                registered_count += 1
                
            except ValidationError as e:
                logger.warning(
                    f"Skipping invalid model at index {idx} from provider '{provider}': "
                    f"Validation error: {e}"
                )
                skipped_count += 1
                continue
            except (KeyError, TypeError, AttributeError) as e:
                logger.warning(
                    f"Skipping malformed model at index {idx} from provider '{provider}': "
                    f"{type(e).__name__}: {e}. Model data: {raw_model}"
                )
                skipped_count += 1
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error processing model at index {idx} from provider '{provider}': "
                    f"{type(e).__name__}: {e}. Model data: {raw_model}"
                )
                skipped_count += 1
                continue
        
        logger.info(
            f"Registered {registered_count} models from provider '{provider}' "
            f"({skipped_count} skipped due to errors)"
        )
    
    def _load_pricing_overrides(self, override_path: Optional[str] = None) -> Dict[str, Dict]:
        """
        Lädt Preis-Overrides aus pricing_overrides.json.
        
        Sucht nach pricing_overrides.json im Konfigurationsverzeichnis
        oder am angegebenen Pfad. Gibt leeres Dict zurück wenn Datei
        nicht existiert oder fehlerhaft ist.
        
        Args:
            override_path: Optionaler expliziter Pfad zur Override-Datei.
                          Falls None, wird im config_dir nach pricing_overrides.json gesucht.
        
        Returns:
            Dict[str, Dict]: Dictionary mit model_id als Key und Override-Dict als Value
                            Format: {'model_id': {'cost_in': float, 'cost_out': float}}
                            Leeres Dict wenn Datei nicht existiert oder fehlerhaft ist.
        
        Example:
            >>> overrides = registry._load_pricing_overrides()
            >>> # Returns: {'gpt-4o-mini': {'cost_in': 0.10, 'cost_out': 0.50}}
        """
        # Bestimme Pfad zur Override-Datei
        if override_path:
            override_file = Path(override_path).expanduser()
        else:
            override_file = self._config_dir / "pricing_overrides.json"
        
        # Prüfe ob Datei existiert
        if not override_file.exists():
            logger.debug(f"No pricing overrides file found at {override_file}")
            return {}
        
        # Lade und parse JSON
        try:
            with open(override_file, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
            
            # Validiere Format
            if not isinstance(overrides, dict):
                logger.warning(
                    f"Invalid pricing overrides format in {override_file}: "
                    f"Expected dict, got {type(overrides).__name__}"
                )
                return {}
            
            # Validiere Einträge
            valid_overrides = {}
            for model_id, override_data in overrides.items():
                if not isinstance(override_data, dict):
                    logger.warning(
                        f"Skipping invalid override for '{model_id}': "
                        f"Expected dict, got {type(override_data).__name__}"
                    )
                    continue
                
                # Prüfe ob mindestens cost_in oder cost_out vorhanden ist
                if 'cost_in' not in override_data and 'cost_out' not in override_data:
                    logger.warning(
                        f"Skipping override for '{model_id}': "
                        f"Neither 'cost_in' nor 'cost_out' specified"
                    )
                    continue
                
                valid_overrides[model_id] = override_data
            
            logger.info(
                f"Loaded {len(valid_overrides)} pricing overrides from {override_file}"
            )
            return valid_overrides
            
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse pricing overrides from {override_file}: {e}. "
                f"Continuing without overrides."
            )
            return {}
        except IOError as e:
            logger.warning(
                f"Failed to read pricing overrides from {override_file}: {e}. "
                f"Continuing without overrides."
            )
            return {}
        except Exception as e:
            logger.warning(
                f"Unexpected error loading pricing overrides from {override_file}: {e}. "
                f"Continuing without overrides."
            )
            return {}
    
    def apply_pricing_overrides(self, overrides: Optional[Dict[str, Dict]] = None) -> None:
        """
        Wendet Preis-Overrides auf Modelle an.
        
        Überschreibt cost_in und cost_out für Modelle die in der
        Override-Datei definiert sind. Overrides für nicht-existierende
        Modelle werden ignoriert.
        
        Falls keine Overrides übergeben werden, lädt die Methode automatisch
        die Overrides aus pricing_overrides.json.
        
        Args:
            overrides: Optional Dict mit model_id als Key und Override-Dict als Value
                      Format: {'model_id': {'cost_in': float, 'cost_out': float}}
                      Falls None, werden Overrides aus pricing_overrides.json geladen.
        
        Example:
            >>> # Explizite Overrides
            >>> overrides = {
            ...     'gpt-4o-mini': {'cost_in': 0.10, 'cost_out': 0.50},
            ...     'claude-3-opus': {'cost_in': 10.0, 'cost_out': 30.0}
            ... }
            >>> registry.apply_pricing_overrides(overrides)
            
            >>> # Automatisches Laden aus Datei
            >>> registry.apply_pricing_overrides()
        """
        # Lade Overrides aus Datei falls nicht übergeben
        if overrides is None:
            overrides = self._load_pricing_overrides()
        
        # Speichere Overrides
        self._pricing_overrides = overrides
        
        # Keine Overrides vorhanden
        if not overrides:
            logger.debug("No pricing overrides to apply")
            return
        
        applied_count = 0
        ignored_count = 0
        
        for model_id, override_data in overrides.items():
            if model_id in self._models:
                model = self._models[model_id]
                
                # Überschreibe Kosten falls vorhanden
                if 'cost_in' in override_data:
                    model.cost_in = override_data['cost_in']
                if 'cost_out' in override_data:
                    model.cost_out = override_data['cost_out']
                
                applied_count += 1
                logger.debug(f"Applied pricing override for model '{model_id}'")
            else:
                ignored_count += 1
                logger.debug(
                    f"Ignoring pricing override for non-existent model '{model_id}'"
                )
        
        logger.info(
            f"Applied {applied_count} pricing overrides "
            f"({ignored_count} ignored for non-existent models)"
        )
    
    def get_all(self) -> List[NormalizedModel]:
        """
        Gibt alle Modelle zurück.
        
        Returns:
            List[NormalizedModel]: Liste aller registrierten Modelle
        
        Example:
            >>> all_models = registry.get_all()
            >>> print(f"Total models: {len(all_models)}")
        """
        return list(self._models.values())
    
    def get_by_provider(self, provider: str) -> List[NormalizedModel]:
        """
        Filtert Modelle nach Provider.
        
        Args:
            provider: Provider-Name (z.B. 'openai', 'anthropic', 'local')
        
        Returns:
            List[NormalizedModel]: Liste der Modelle des angegebenen Providers
        
        Example:
            >>> openai_models = registry.get_by_provider('openai')
            >>> local_models = registry.get_by_provider('local')
        """
        return [
            model for model in self._models.values()
            if model.provider == provider
        ]
    
    def get_by_id(self, model_id: str) -> Optional[NormalizedModel]:
        """
        Sucht Modell nach ID.
        
        Args:
            model_id: Eindeutige Modell-ID (z.B. 'gpt-4o-mini', 'claude-3-opus')
        
        Returns:
            Optional[NormalizedModel]: Modell mit der angegebenen ID oder None
        
        Example:
            >>> model = registry.get_by_id('gpt-4o-mini')
            >>> if model:
            ...     print(f"Found: {model.model_name}")
        """
        return self._models.get(model_id)
    
    def filter(self,
              provider: Optional[str] = None,
              capabilities: Optional[List[str]] = None,
              max_cost_in: Optional[float] = None,
              max_cost_out: Optional[float] = None,
              min_context: Optional[int] = None) -> List[NormalizedModel]:
        """
        Filtert Modelle nach mehreren Kriterien.
        
        Alle Filter werden als UND-Verknüpfung angewendet.
        Modelle müssen alle angegebenen Kriterien erfüllen.
        
        Args:
            provider: Filtert nach Provider-Name (optional)
            capabilities: Liste von erforderlichen Capabilities (optional)
            max_cost_in: Maximale Input-Kosten pro 1M Tokens (optional)
            max_cost_out: Maximale Output-Kosten pro 1M Tokens (optional)
            min_context: Minimales Context Window in Tokens (optional)
        
        Returns:
            List[NormalizedModel]: Liste der Modelle die alle Filter erfüllen
        
        Example:
            >>> # Finde günstige OpenAI-Modelle mit großem Context
            >>> models = registry.filter(
            ...     provider='openai',
            ...     max_cost_in=1.0,
            ...     min_context=100000
            ... )
        """
        results = list(self._models.values())
        
        # Filter nach Provider
        if provider is not None:
            results = [m for m in results if m.provider == provider]
        
        # Filter nach Capabilities
        if capabilities is not None:
            for capability in capabilities:
                results = [
                    m for m in results
                    if m.supports_capability(capability)
                ]
        
        # Filter nach max Input-Kosten
        if max_cost_in is not None:
            results = [
                m for m in results
                if m.cost_in is not None and m.cost_in <= max_cost_in
            ]
        
        # Filter nach max Output-Kosten
        if max_cost_out is not None:
            results = [
                m for m in results
                if m.cost_out is not None and m.cost_out <= max_cost_out
            ]
        
        # Filter nach min Context Window
        if min_context is not None:
            results = [
                m for m in results
                if m.context_window is not None and m.context_window >= min_context
            ]
        
        return results
    
    def _normalize_catwalk_model(self, provider: str, model: Dict) -> NormalizedModel:
        """
        Normalisiert Modell aus Catwalk-Format.
        
        Catwalk verwendet einheitliche Feldnamen:
        - id → model_id
        - name → model_name
        - cost_per_1m_in → cost_in
        - cost_per_1m_out → cost_out
        - context_window → context_window
        - Alle anderen Felder → options
        
        Args:
            provider: Provider-Name (openai, anthropic, openrouter)
            model: Raw model dict aus Catwalk config
        
        Returns:
            NormalizedModel: Normalisiertes Modell
        
        Raises:
            ValidationError: Wenn erforderliche Felder fehlen
            KeyError: Wenn 'id' oder 'name' fehlt
        
        Example:
            >>> raw = {
            ...     'id': 'gpt-4o-mini',
            ...     'name': 'GPT-4o Mini',
            ...     'cost_per_1m_in': 0.15,
            ...     'cost_per_1m_out': 0.60,
            ...     'context_window': 128000,
            ...     'supports_attachments': True
            ... }
            >>> normalized = registry._normalize_catwalk_model('openai', raw)
        """
        # Erforderliche Felder prüfen
        if 'id' not in model:
            raise ValidationError("Model missing required field 'id'")
        if 'name' not in model:
            raise ValidationError("Model missing required field 'name'")
        
        # Standard-Felder extrahieren
        model_id = model['id']
        model_name = model['name']
        context_window = model.get('context_window')
        cost_in = model.get('cost_per_1m_in')
        cost_out = model.get('cost_per_1m_out')
        
        # Alle anderen Felder in options sammeln
        standard_fields = {
            'id', 'name', 'context_window',
            'cost_per_1m_in', 'cost_per_1m_out'
        }
        options = {
            key: value
            for key, value in model.items()
            if key not in standard_fields
        }
        
        return NormalizedModel(
            provider=provider,
            model_id=model_id,
            model_name=model_name,
            context_window=context_window,
            cost_in=cost_in,
            cost_out=cost_out,
            options=options
        )
    
    def _normalize_local_model(self, model: Dict) -> NormalizedModel:
        """
        Normalisiert lokales Modell (LM Studio, Ollama).
        
        Lokale Modelle haben:
        - provider='local'
        - cost_in=None, cost_out=None (keine Kosten)
        - Verschiedene Formate je nach Quelle (LM Studio vs Ollama)
        
        Args:
            model: Raw model dict von LM Studio oder Ollama
        
        Returns:
            NormalizedModel: Normalisiertes lokales Modell
        
        Raises:
            ValidationError: Wenn erforderliche Felder fehlen
        
        Example:
            >>> # LM Studio Format (OpenAI-kompatibel)
            >>> raw_lm = {'id': 'llama-2-7b', 'object': 'model'}
            >>> normalized = registry._normalize_local_model(raw_lm)
            
            >>> # Ollama Format
            >>> raw_ollama = {'name': 'llama2:7b', 'size': 3826793677}
            >>> normalized = registry._normalize_local_model(raw_ollama)
        """
        # LM Studio verwendet 'id', Ollama verwendet 'name'
        model_id = model.get('id') or model.get('name')
        
        if not model_id:
            raise ValidationError(
                "Local model missing required field 'id' or 'name'"
            )
        
        # Model name ist gleich ID falls nicht anders angegeben
        model_name = model.get('name', model_id)
        
        # Context window aus verschiedenen möglichen Feldern
        context_window = model.get('context_window') or model.get('context_length')
        
        # Alle anderen Felder in options
        standard_fields = {'id', 'name', 'context_window', 'context_length'}
        options = {
            key: value
            for key, value in model.items()
            if key not in standard_fields
        }
        
        return NormalizedModel(
            provider='local',
            model_id=model_id,
            model_name=model_name,
            context_window=context_window,
            cost_in=None,  # Lokale Modelle haben keine Kosten
            cost_out=None,
            options=options
        )

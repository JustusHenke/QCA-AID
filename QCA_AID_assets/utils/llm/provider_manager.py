"""
LLM Provider Manager

Hauptklasse für die Verwaltung mehrerer LLM-Provider und deren Modelle.
Koordiniert das Laden, Cachen und Bereitstellen von Provider- und Modell-Informationen.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .cache_manager import CacheManager
from .provider_loader import ProviderLoader
from .model_registry import ModelRegistry
from .local_detector import LocalDetector
from .models import NormalizedModel, ProviderLoadError


logger = logging.getLogger(__name__)


class LLMProviderManager:
    """
    Hauptklasse für LLM-Provider-Verwaltung.
    
    Koordiniert das Laden, Cachen und Bereitstellen von Provider-
    und Modell-Informationen. Integriert CacheManager, ProviderLoader,
    ModelRegistry und LocalDetector zu einer einheitlichen Schnittstelle.
    
    Attributes:
        cache_manager: Verwaltet lokalen Cache mit TTL-Logik
        provider_loader: Lädt Provider-Configs von Catwalk oder lokalen Dateien
        model_registry: Verwaltet normalisierte Modelle und bietet Filter-API
        local_detector: Erkennt lokale Modelle (LM Studio, Ollama)
        _initialized: Flag ob Manager bereits initialisiert wurde
    
    Example:
        >>> manager = LLMProviderManager(cache_dir="~/.llm_cache")
        >>> await manager.initialize()
        >>> all_models = manager.get_all_models()
        >>> openai_models = manager.get_models_by_provider('openai')
    """
    
    def __init__(self, 
                 cache_dir: str = "~/.llm_cache",
                 fallback_dir: Optional[str] = None,
                 config_dir: Optional[str] = None):
        """
        Initialisiert LLMProviderManager mit allen Komponenten.
        
        Args:
            cache_dir: Verzeichnis für Cache-Dateien (Standard: ~/.llm_cache)
            fallback_dir: Verzeichnis für lokale Fallback-Configs (optional)
            config_dir: Verzeichnis für pricing_overrides.json (optional)
        
        Example:
            >>> manager = LLMProviderManager()
            >>> manager = LLMProviderManager(
            ...     cache_dir="/custom/cache",
            ...     fallback_dir="/custom/configs"
            ... )
        """
        logger.info("Initializing LLMProviderManager")
        
        # Initialisiere Komponenten
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        self.provider_loader = ProviderLoader(fallback_dir=fallback_dir)
        self.model_registry = ModelRegistry(config_dir=config_dir)
        self.local_detector = LocalDetector()
        
        self._initialized = False
        
        logger.debug("LLMProviderManager components initialized")
    
    async def initialize(self, force_refresh: bool = False) -> None:
        """
        Initialisiert Provider-Daten.
        
        Lädt Provider- und Modell-Informationen aus Cache oder von externen
        Quellen. Erkennt lokale Modelle und wendet Pricing-Overrides an.
        
        Workflow:
        1. Prüfe Cache-Gültigkeit
        2. Falls Cache gültig und nicht force_refresh: Lade aus Cache
        3. Sonst: Lade von externen Quellen (Catwalk + lokale Server)
        4. Registriere alle Modelle in Registry
        5. Wende Pricing-Overrides an
        6. Speichere im Cache
        
        Args:
            force_refresh: Erzwingt Neuladen trotz gültigem Cache (Standard: False)
        
        Raises:
            ProviderLoadError: Wenn keine Provider geladen werden konnten
        
        Example:
            >>> await manager.initialize()
            >>> await manager.initialize(force_refresh=True)  # Erzwingt Neuladen
        """
        if self._initialized and not force_refresh:
            logger.info("LLMProviderManager already initialized")
            return
        
        logger.info(f"Initializing provider data (force_refresh={force_refresh})")
        
        # Schritt 1: Prüfe Cache
        if not force_refresh:
            try:
                if self.cache_manager.is_valid():
                    logger.info("Loading provider data from cache")
                    cache_data = self.cache_manager.load()
                    
                    if cache_data and 'models' in cache_data:
                        # Lade Modelle aus Cache
                        cached_models = cache_data['models']
                        
                        if not isinstance(cached_models, list):
                            logger.warning(
                                f"Invalid cache format: 'models' should be list, "
                                f"got {type(cached_models).__name__}. Reloading from sources."
                            )
                        else:
                            # Konvertiere Dicts zurück zu NormalizedModel-Instanzen
                            loaded_count = 0
                            for model_dict in cached_models:
                                try:
                                    model = NormalizedModel(**model_dict)
                                    # Registriere direkt im internen Dictionary
                                    self.model_registry._models[model.model_id] = model
                                    loaded_count += 1
                                except (TypeError, ValueError) as e:
                                    logger.warning(
                                        f"Skipping invalid cached model: {e}. "
                                        f"Model data: {model_dict}"
                                    )
                                    continue
                            
                            if loaded_count > 0:
                                logger.info(f"Loaded {loaded_count} models from cache")
                                
                                # Wende Pricing-Overrides an
                                try:
                                    self.model_registry.apply_pricing_overrides()
                                except Exception as e:
                                    logger.warning(f"Failed to apply pricing overrides: {e}")
                                
                                self._initialized = True
                                return
                            else:
                                logger.warning("No valid models in cache, reloading from sources")
            except Exception as e:
                logger.warning(
                    f"Failed to load from cache: {type(e).__name__}: {e}. "
                    f"Loading from external sources."
                )
        
        # Schritt 2: Lade von externen Quellen
        logger.info("Loading provider data from external sources")
        
        # Lade Catwalk-Provider parallel
        try:
            provider_configs = await self.provider_loader.load_all_providers()
        except Exception as e:
            logger.error(
                f"Critical error loading providers: {type(e).__name__}: {e}"
            )
            provider_configs = {}
        
        if not provider_configs:
            logger.warning(
                "No Catwalk providers loaded successfully. "
                "Continuing with local models only."
            )
        
        # Registriere Modelle von Catwalk-Providern
        for provider_name, config in provider_configs.items():
            try:
                if not isinstance(config, dict):
                    logger.warning(
                        f"Invalid config format for provider '{provider_name}': "
                        f"Expected dict, got {type(config).__name__}"
                    )
                    continue
                
                models = config.get('models', [])
                if models:
                    self.model_registry.register_models(provider_name, models)
                    logger.info(f"Registered models from {provider_name}")
                else:
                    logger.warning(f"No models found in {provider_name} config")
            except Exception as e:
                logger.error(
                    f"Failed to register models from provider '{provider_name}': "
                    f"{type(e).__name__}: {e}"
                )
                continue
        
        # Schritt 3: Erkenne lokale Modelle
        local_models = []
        try:
            logger.info("Detecting local models")
            local_models = await self.local_detector.detect_all()
            
            if local_models:
                self.model_registry.register_models('local', local_models)
                logger.info(f"Registered {len(local_models)} local models")
            else:
                logger.info("No local models detected")
        except Exception as e:
            logger.warning(
                f"Local model detection failed: {type(e).__name__}: {e}. "
                f"Continuing without local models."
            )
        
        # Schritt 4: Wende Pricing-Overrides an
        try:
            self.model_registry.apply_pricing_overrides()
        except Exception as e:
            logger.warning(
                f"Failed to apply pricing overrides: {type(e).__name__}: {e}. "
                f"Continuing with default pricing."
            )
        
        # Schritt 5: Prüfe ob Modelle geladen wurden
        all_models = self.model_registry.get_all()
        if not all_models:
            error_msg = (
                "Failed to load any models from providers. "
                "Possible causes:\n"
                "  - Network connection issues (cannot reach Catwalk GitHub)\n"
                "  - Missing local fallback configuration files\n"
                "  - All provider configurations are invalid\n"
                "  - No local LLM servers (LM Studio, Ollama) are running\n"
                "Please check your network connection and ensure fallback files exist."
            )
            logger.error(error_msg)
            raise ProviderLoadError(error_msg)
        
        # Schritt 6: Speichere im Cache
        try:
            loaded_providers = list(provider_configs.keys())
            if local_models:
                loaded_providers.append('local')
            
            self.cache_manager.save(all_models, loaded_providers)
            logger.info("Provider data saved to cache")
        except Exception as e:
            logger.warning(
                f"Failed to save cache: {type(e).__name__}: {e}. "
                f"Cache will not be available for next initialization."
            )
        
        self._initialized = True
        logger.info(
            f"✓ Initialization complete: {len(all_models)} models from "
            f"{len(loaded_providers)} providers"
        )
    
    def get_all_models(self) -> List[NormalizedModel]:
        """
        Gibt alle verfügbaren Modelle zurück.
        
        Wrapper für ModelRegistry.get_all().
        
        Returns:
            List[NormalizedModel]: Liste aller registrierten Modelle
        
        Example:
            >>> all_models = manager.get_all_models()
            >>> print(f"Total models: {len(all_models)}")
        """
        self._ensure_initialized()
        return self.model_registry.get_all()
    
    def get_models_by_provider(self, provider: str) -> List[NormalizedModel]:
        """
        Filtert Modelle nach Provider.
        
        Wrapper für ModelRegistry.get_by_provider().
        
        Args:
            provider: Provider-Name (z.B. 'openai', 'anthropic', 'local')
        
        Returns:
            List[NormalizedModel]: Liste der Modelle des angegebenen Providers
        
        Example:
            >>> openai_models = manager.get_models_by_provider('openai')
            >>> local_models = manager.get_models_by_provider('local')
        """
        self._ensure_initialized()
        return self.model_registry.get_by_provider(provider)
    
    def get_model_by_id(self, model_id: str) -> Optional[NormalizedModel]:
        """
        Sucht Modell nach ID.
        
        Wrapper für ModelRegistry.get_by_id().
        
        Args:
            model_id: Eindeutige Modell-ID (z.B. 'gpt-4o-mini', 'claude-3-opus')
        
        Returns:
            Optional[NormalizedModel]: Modell mit der angegebenen ID oder None
        
        Example:
            >>> model = manager.get_model_by_id('gpt-4o-mini')
            >>> if model:
            ...     print(f"Found: {model.model_name}")
        """
        self._ensure_initialized()
        return self.model_registry.get_by_id(model_id)
    
    def filter_models(self,
                     provider: Optional[str] = None,
                     capabilities: Optional[List[str]] = None,
                     max_cost_in: Optional[float] = None,
                     max_cost_out: Optional[float] = None,
                     min_context: Optional[int] = None) -> List[NormalizedModel]:
        """
        Filtert Modelle nach mehreren Kriterien.
        
        Wrapper für ModelRegistry.filter().
        Alle Filter werden als UND-Verknüpfung angewendet.
        
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
            >>> models = manager.filter_models(
            ...     provider='openai',
            ...     max_cost_in=1.0,
            ...     min_context=100000
            ... )
        """
        self._ensure_initialized()
        return self.model_registry.filter(
            provider=provider,
            capabilities=capabilities,
            max_cost_in=max_cost_in,
            max_cost_out=max_cost_out,
            min_context=min_context
        )
    
    def get_supported_providers(self) -> List[str]:
        """
        Gibt Liste unterstützter Provider zurück.
        
        Sammelt alle eindeutigen Provider-Namen aus der Registry.
        
        Returns:
            List[str]: Liste der Provider-Namen (z.B. ['openai', 'anthropic', 'local'])
        
        Example:
            >>> providers = manager.get_supported_providers()
            >>> print(f"Supported providers: {', '.join(providers)}")
        """
        self._ensure_initialized()
        
        # Sammle alle eindeutigen Provider aus den Modellen
        all_models = self.model_registry.get_all()
        providers = sorted(set(model.provider for model in all_models))
        
        return providers
    
    def invalidate_cache(self) -> None:
        """
        Löscht Cache und erzwingt Neuladen.
        
        Wrapper für CacheManager.invalidate().
        Setzt _initialized auf False, sodass beim nächsten Zugriff
        neu geladen wird.
        
        Example:
            >>> manager.invalidate_cache()
            >>> await manager.initialize()  # Lädt neu von externen Quellen
        """
        logger.info("Invalidating cache")
        self.cache_manager.invalidate()
        self._initialized = False
        logger.info("Cache invalidated, manager needs re-initialization")
    
    async def add_provider_source(self, 
                                   provider_name: str, 
                                   url: str,
                                   update_cache: bool = True) -> None:
        """
        Fügt eine neue Provider-Quelle via URL hinzu.
        
        Lädt Provider-Konfiguration von der angegebenen URL, normalisiert
        die Modelle automatisch und integriert sie in die Registry.
        Optional wird der Cache aktualisiert.
        
        Args:
            provider_name: Name des neuen Providers (z.B. 'custom-provider')
            url: URL zur Provider-Config JSON-Datei
            update_cache: Ob Cache nach dem HinzuFügen aktualisiert werden soll (Standard: True)
        
        Raises:
            ProviderLoadError: Wenn URL nicht erreichbar oder JSON ungültig ist
            RuntimeError: Wenn Manager nicht initialisiert ist
        
        Example:
            >>> await manager.add_provider_source(
            ...     'custom-provider',
            ...     'https://example.com/custom-provider.json'
            ... )
        """
        self._ensure_initialized()
        
        logger.info(f"Adding new provider source '{provider_name}' from URL: {url}")
        
        try:
            # Lade Config von URL
            config = await self.provider_loader._load_from_url(url)
            
            # Validiere Config-Format
            if not isinstance(config, dict):
                raise ProviderLoadError(
                    f"Invalid config format from {url}: "
                    f"Expected dict, got {type(config).__name__}"
                )
            
            # Extrahiere Modelle
            models = config.get('models', [])
            if not models:
                logger.warning(f"No models found in config from {url}")
                return
            
            # Registriere Modelle (automatische Normalisierung)
            self.model_registry.register_models(provider_name, models)
            logger.info(f"✓ Successfully added provider '{provider_name}' from URL")
            
            # Aktualisiere Cache falls gewünscht
            if update_cache:
                try:
                    all_models = self.model_registry.get_all()
                    current_providers = self.get_supported_providers()
                    self.cache_manager.save(all_models, current_providers)
                    logger.info("Cache updated with new provider")
                except Exception as e:
                    logger.warning(f"Failed to update cache: {e}")
            
        except Exception as e:
            logger.error(
                f"Failed to add provider source '{provider_name}' from {url}: "
                f"{type(e).__name__}: {e}"
            )
            raise
    
    async def add_local_provider_config(self,
                                        provider_name: str,
                                        config_path: str,
                                        update_cache: bool = True) -> None:
        """
        Fügt eine neue Provider-Quelle via lokale JSON-Datei hinzu.
        
        Lädt Provider-Konfiguration aus einer lokalen Datei, normalisiert
        die Modelle automatisch und integriert sie in die Registry.
        Optional wird der Cache aktualisiert.
        
        Args:
            provider_name: Name des neuen Providers (z.B. 'custom-provider')
            config_path: Pfad zur lokalen Provider-Config JSON-Datei
            update_cache: Ob Cache nach dem HinzuFügen aktualisiert werden soll (Standard: True)
        
        Raises:
            ProviderLoadError: Wenn Datei nicht existiert oder JSON ungültig ist
            RuntimeError: Wenn Manager nicht initialisiert ist
        
        Example:
            >>> await manager.add_local_provider_config(
            ...     'custom-provider',
            ...     '/path/to/custom-provider.json'
            ... )
        """
        self._ensure_initialized()
        
        logger.info(
            f"Adding new provider source '{provider_name}' from local file: {config_path}"
        )
        
        try:
            # Lade Config von lokaler Datei
            config = self.provider_loader._load_from_file(config_path)
            
            # Validiere Config-Format
            if not isinstance(config, dict):
                raise ProviderLoadError(
                    f"Invalid config format in {config_path}: "
                    f"Expected dict, got {type(config).__name__}"
                )
            
            # Extrahiere Modelle
            models = config.get('models', [])
            if not models:
                logger.warning(f"No models found in config from {config_path}")
                return
            
            # Registriere Modelle (automatische Normalisierung)
            self.model_registry.register_models(provider_name, models)
            logger.info(f"✓ Successfully added provider '{provider_name}' from local file")
            
            # Aktualisiere Cache falls gewünscht
            if update_cache:
                try:
                    all_models = self.model_registry.get_all()
                    current_providers = self.get_supported_providers()
                    self.cache_manager.save(all_models, current_providers)
                    logger.info("Cache updated with new provider")
                except Exception as e:
                    logger.warning(f"Failed to update cache: {e}")
            
        except Exception as e:
            logger.error(
                f"Failed to add local provider config '{provider_name}' from {config_path}: "
                f"{type(e).__name__}: {e}"
            )
            raise
    
    def _ensure_initialized(self) -> None:
        """
        Stellt sicher dass Manager initialisiert ist.
        
        Wirft RuntimeError wenn initialize() noch nicht aufgerufen wurde.
        
        Raises:
            RuntimeError: Wenn Manager nicht initialisiert ist
        """
        if not self._initialized:
            raise RuntimeError(
                "LLMProviderManager not initialized. "
                "Call 'await manager.initialize()' first."
            )

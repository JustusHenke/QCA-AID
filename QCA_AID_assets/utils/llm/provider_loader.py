"""
Provider Loader

Lädt Provider-Konfigurationen von Catwalk GitHub oder lokalen Fallback-Dateien.
Unterstützt paralleles Laden mehrerer Provider mit automatischem Fallback.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import aiohttp

from .models import ProviderLoadError


logger = logging.getLogger(__name__)


class ProviderLoader:
    """
    Lädt Provider-Configs von Catwalk oder lokalen Dateien.
    
    Implementiert Fallback-Strategie: Versucht zuerst Catwalk-URL,
    fällt dann auf lokale Kopien zurück.
    
    Attributes:
        CATWALK_BASE_URL: Basis-URL für Catwalk Provider-Configs
        PROVIDER_CONFIGS: Mapping von Provider-Namen zu Config-Dateinamen
        fallback_dir: Verzeichnis für lokale Fallback-Configs
    
    Example:
        >>> loader = ProviderLoader(fallback_dir='./configs')
        >>> config = await loader.load_provider_config('openai')
        >>> all_configs = await loader.load_all_providers()
    """
    
    CATWALK_BASE_URL = "https://raw.githubusercontent.com/charmbracelet/catwalk/main/internal/providers/configs/"
    
    PROVIDER_CONFIGS = {
        'openai': 'openai.json',
        'anthropic': 'anthropic.json',
        'openrouter': 'openrouter.json'
    }
    
    def __init__(self, fallback_dir: Optional[str] = None):
        """
        Initialisiert ProviderLoader.
        
        Args:
            fallback_dir: Verzeichnis für lokale Fallback-Configs.
                         Falls None, wird 'QCA_AID_assets/utils/llm/configs/' verwendet.
        """
        if fallback_dir is None:
            # Default: configs Verzeichnis im llm Package
            fallback_dir = Path(__file__).parent / 'configs'
        
        self.fallback_dir = Path(fallback_dir)
        logger.info(f"ProviderLoader initialized with fallback_dir: {self.fallback_dir}")
    
    async def load_provider_config(self, provider: str) -> Dict:
        """
        Lädt Config für einen Provider.
        
        Versucht zuerst Catwalk-URL, dann lokalen Fallback.
        
        Args:
            provider: Provider-Name ('openai', 'anthropic', 'openrouter')
            
        Returns:
            Dict mit Provider-Config
            
        Raises:
            ProviderLoadError: Wenn beide Quellen fehlschlagen
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ProviderLoadError(
                f"Unknown provider: {provider}. "
                f"Supported providers: {list(self.PROVIDER_CONFIGS.keys())}"
            )
        
        config_filename = self.PROVIDER_CONFIGS[provider]
        url = f"{self.CATWALK_BASE_URL}{config_filename}"
        
        # Versuch 1: Von URL laden
        try:
            logger.info(f"Loading {provider} config from Catwalk URL: {url}")
            config = await self._load_from_url(url)
            logger.info(f"✓ Successfully loaded {provider} config from URL")
            return config
        except Exception as url_error:
            logger.warning(f"Failed to load {provider} from URL: {url_error}")
            
            # Versuch 2: Von lokalem Fallback laden
            try:
                fallback_path = self.fallback_dir / config_filename
                logger.info(f"Attempting fallback: {fallback_path}")
                config = self._load_from_file(str(fallback_path))
                logger.info(f"✓ Successfully loaded {provider} config from local fallback")
                return config
            except Exception as file_error:
                logger.error(f"Failed to load {provider} from fallback: {file_error}")
                raise ProviderLoadError(
                    f"Failed to load {provider} config from both URL and local fallback. "
                    f"URL error: {url_error}. File error: {file_error}"
                )
    
    async def load_all_providers(self) -> Dict[str, Dict]:
        """
        Lädt alle Provider-Configs parallel.
        
        Verwendet asyncio.gather() für paralleles Laden.
        Fehler bei einzelnen Providern werden geloggt, aber nicht propagiert.
        
        Returns:
            Dict mit Provider-Namen als Keys und Configs als Values.
            Provider die fehlschlagen werden nicht im Dict enthalten sein.
            
        Example:
            >>> configs = await loader.load_all_providers()
            >>> # configs = {'openai': {...}, 'anthropic': {...}}
        """
        logger.info(f"Loading all providers: {list(self.PROVIDER_CONFIGS.keys())}")
        
        # Erstelle Tasks für alle Provider
        tasks = []
        provider_names = []
        
        for provider in self.PROVIDER_CONFIGS.keys():
            tasks.append(self.load_provider_config(provider))
            provider_names.append(provider)
        
        # Lade alle parallel mit gather, return_exceptions=True
        # damit einzelne Fehler nicht das gesamte Laden abbrechen
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sammle erfolgreiche Ergebnisse
        configs = {}
        for provider_name, result in zip(provider_names, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {provider_name}: {result}")
            else:
                configs[provider_name] = result
                logger.info(f"✓ {provider_name} config loaded successfully")
        
        logger.info(f"Loaded {len(configs)}/{len(self.PROVIDER_CONFIGS)} providers successfully")
        return configs
    
    async def _load_from_url(self, url: str) -> Dict:
        """
        Lädt Config von URL.
        
        Verwendet aiohttp für asynchrone HTTP-Requests.
        
        Args:
            url: URL zur Provider-Config JSON-Datei
            
        Returns:
            Dict mit geparster JSON-Config
            
        Raises:
            aiohttp.ClientError: Bei Netzwerk-Fehlern
            json.JSONDecodeError: Bei ungültigem JSON
            Exception: Bei anderen Fehlern
        """
        timeout = aiohttp.ClientTimeout(total=10)  # 10 Sekunden Timeout
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()  # Raise für HTTP-Fehler (4xx, 5xx)
                    text = await response.text()
                    
                    # Parse JSON mit aussagekräftiger Fehlermeldung
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from {url}: {e}")
                        raise ProviderLoadError(f"Invalid JSON response from {url}: {e}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error loading from {url}: {e}")
            raise ProviderLoadError(f"Network error: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout loading from {url}")
            raise ProviderLoadError(f"Request timeout after 10 seconds")
        except Exception as e:
            logger.error(f"Unexpected error loading from {url}: {type(e).__name__}: {e}")
            raise ProviderLoadError(f"Unexpected error: {type(e).__name__}: {e}")
    
    def _load_from_file(self, filepath: str) -> Dict:
        """
        Lädt Config aus lokaler Datei.
        
        Args:
            filepath: Pfad zur lokalen Config-Datei
            
        Returns:
            Dict mit geparster JSON-Config
            
        Raises:
            FileNotFoundError: Wenn Datei nicht existiert
            json.JSONDecodeError: Bei ungültigem JSON
            Exception: Bei anderen Fehlern
        """
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"Config file not found: {filepath}")
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filepath}: {e}")
            raise ProviderLoadError(f"Invalid JSON in {filepath}: {e}")
        except IOError as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            raise ProviderLoadError(f"Failed to read {filepath}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {type(e).__name__}: {e}")
            raise ProviderLoadError(f"Unexpected error reading {filepath}: {e}")

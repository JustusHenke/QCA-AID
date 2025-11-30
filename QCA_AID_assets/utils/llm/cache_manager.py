"""
Cache Manager for LLM Provider Manager

Manages local caching of provider and model information with TTL-based validation.
Reduces network requests and improves performance by storing provider data locally.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from .models import CacheMetadata, CacheError, NormalizedModel

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Verwaltet lokalen Provider-Cache mit TTL-Logik.
    
    Der CacheManager speichert Provider- und Modell-Informationen lokal,
    um wiederholte Netzwerk-Anfragen zu vermeiden. Cache-Einträge haben
    eine Time-To-Live (TTL) von 24 Stunden.
    
    Attributes:
        CACHE_TTL_HOURS: Time-To-Live für Cache-Einträge in Stunden (24h)
        CACHE_VERSION: Version des Cache-Formats für Migrations-Kompatibilität
        cache_dir: Pfad zum Cache-Verzeichnis
        cache_file: Pfad zur Cache-Datei (providers.json)
    
    Example:
        >>> cache_mgr = CacheManager(cache_dir="~/.llm_cache")
        >>> if cache_mgr.is_valid():
        ...     data = cache_mgr.load()
        ... else:
        ...     # Load from external sources
        ...     cache_mgr.save(models, providers)
    """
    
    CACHE_TTL_HOURS = 24
    CACHE_VERSION = "1.0"
    
    def __init__(self, cache_dir: str = "~/.llm_cache"):
        """
        Initialisiert CacheManager mit Cache-Verzeichnis.
        
        Erstellt das Cache-Verzeichnis falls es nicht existiert.
        
        Args:
            cache_dir: Pfad zum Cache-Verzeichnis (Standard: ~/.llm_cache)
                      Tilde (~) wird automatisch expandiert.
        
        Raises:
            CacheError: Wenn Cache-Verzeichnis nicht erstellt werden kann
        
        Example:
            >>> cache_mgr = CacheManager()  # Uses default ~/.llm_cache
            >>> cache_mgr = CacheManager("/custom/cache/path")
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_file = self.cache_dir / "providers.json"
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """
        Erstellt Cache-Verzeichnis falls nicht vorhanden.
        
        Setzt Berechtigungen auf 700 (nur Owner) für Sicherheit.
        
        Raises:
            CacheError: Wenn Verzeichnis nicht erstellt werden kann
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Set permissions to 700 (owner only) for security
            self.cache_dir.chmod(0o700)
            logger.debug(f"Cache directory ensured: {self.cache_dir}")
        except Exception as e:
            raise CacheError(f"Failed to create cache directory {self.cache_dir}: {e}")
    
    def is_valid(self) -> bool:
        """
        Prüft ob Cache existiert und gültig ist (< 24h alt).
        
        Validiert:
        - Cache-Datei existiert
        - Cache-Datei ist lesbar
        - Metadaten sind vorhanden und gültig
        - Timestamp ist weniger als CACHE_TTL_HOURS alt
        
        Returns:
            bool: True wenn Cache gültig ist, sonst False
        
        Example:
            >>> if cache_mgr.is_valid():
            ...     print("Using cached data")
            ... else:
            ...     print("Cache expired, reloading")
        """
        if not self.cache_file.exists():
            logger.debug("Cache file does not exist")
            return False
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if 'metadata' not in data:
                logger.warning("Cache missing metadata")
                return False
            
            metadata = data['metadata']
            
            # Validate required metadata fields
            if 'timestamp' not in metadata or 'version' not in metadata:
                logger.warning("Cache metadata incomplete")
                return False
            
            # Check TTL
            cache_age_hours = (time.time() - metadata['timestamp']) / 3600
            is_valid = cache_age_hours < self.CACHE_TTL_HOURS
            
            if is_valid:
                logger.debug(f"Cache valid (age: {cache_age_hours:.1f}h)")
            else:
                logger.debug(f"Cache expired (age: {cache_age_hours:.1f}h)")
            
            return is_valid
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache validation failed: {e}")
            return False
    
    def load(self) -> Optional[Dict[str, Any]]:
        """
        Lädt Cache-Daten.
        
        Lädt und validiert Cache-Daten aus der Cache-Datei.
        Gibt None zurück wenn Cache ungültig ist oder Fehler auftreten.
        
        Returns:
            Optional[Dict]: Dictionary mit 'metadata' und 'models' Keys,
                          oder None wenn Cache ungültig/nicht vorhanden
        
        Structure:
            {
                'metadata': {
                    'timestamp': float,
                    'version': str,
                    'providers': List[str]
                },
                'models': List[Dict]  # Serialized NormalizedModel instances
            }
        
        Example:
            >>> data = cache_mgr.load()
            >>> if data:
            ...     models = data['models']
            ...     metadata = data['metadata']
        """
        if not self.is_valid():
            return None
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded cache with {len(data.get('models', []))} models "
                       f"from {len(data['metadata'].get('providers', []))} providers")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def save(self, models: List[NormalizedModel], providers: List[str]) -> None:
        """
        Speichert Modelle im Cache mit Metadaten.
        
        Serialisiert NormalizedModel-Instanzen und speichert sie zusammen
        mit Metadaten (Timestamp, Version, Provider-Liste) in der Cache-Datei.
        
        Args:
            models: Liste von NormalizedModel-Instanzen zum Speichern
            providers: Liste der Provider-Namen die im Cache enthalten sind
        
        Raises:
            CacheError: Wenn Cache nicht geschrieben werden kann
        
        Example:
            >>> models = [model1, model2, model3]
            >>> providers = ['openai', 'anthropic']
            >>> cache_mgr.save(models, providers)
        """
        try:
            # Create metadata
            metadata = CacheMetadata(
                timestamp=time.time(),
                version=self.CACHE_VERSION,
                providers=providers
            )
            
            # Serialize models (convert dataclass to dict)
            serialized_models = []
            for model in models:
                model_dict = {
                    'provider': model.provider,
                    'model_id': model.model_id,
                    'model_name': model.model_name,
                    'context_window': model.context_window,
                    'cost_in': model.cost_in,
                    'cost_out': model.cost_out,
                    'options': model.options
                }
                serialized_models.append(model_dict)
            
            # Create cache structure
            cache_data = {
                'metadata': {
                    'timestamp': metadata.timestamp,
                    'version': metadata.version,
                    'providers': metadata.providers
                },
                'models': serialized_models
            }
            
            # Write to file with pretty formatting
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(models)} models from {len(providers)} providers to cache")
            
        except Exception as e:
            raise CacheError(f"Failed to save cache: {e}")
    
    def invalidate(self) -> None:
        """
        Löscht Cache-Datei und erzwingt Neuladen.
        
        Entfernt die Cache-Datei vom Dateisystem. Beim nächsten Zugriff
        werden Provider-Informationen neu geladen.
        
        Example:
            >>> cache_mgr.invalidate()
            >>> # Next load() will return None, triggering reload
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("Cache invalidated")
            else:
                logger.debug("Cache file does not exist, nothing to invalidate")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")

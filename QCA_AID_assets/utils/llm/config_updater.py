"""
Config Updater

Automatisches Update von Provider-Configs von Catwalk GitHub.
Prüft das Alter der lokalen Configs und aktualisiert sie bei Bedarf.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import aiohttp


logger = logging.getLogger(__name__)


class ConfigUpdater:
    """
    Automatisches Update-System für Provider-Configs.
    
    Prüft das Alter der lokalen Config-Dateien und lädt bei Bedarf
    neue Versionen von Catwalk GitHub herunter.
    
    Attributes:
        CATWALK_BASE_URL: Basis-URL für Catwalk Provider-Configs
        MAX_AGE_DAYS: Maximales Alter der Configs in Tagen (default: 7)
        config_dir: Verzeichnis für Config-Dateien
    """
    
    CATWALK_BASE_URL = "https://raw.githubusercontent.com/charmbracelet/catwalk/main/internal/providers/configs/"
    
    PROVIDER_CONFIGS = {
        'openai': 'openai.json',
        'anthropic': 'anthropic.json',
        'openrouter': 'openrouter.json'
    }
    
    def __init__(self, config_dir: Optional[Path] = None, max_age_days: int = 7):
        """
        Initialisiert ConfigUpdater.
        
        Args:
            config_dir: Verzeichnis für Config-Dateien (default: llm/configs/)
            max_age_days: Maximales Alter der Configs in Tagen (default: 7)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / 'configs'
        
        self.config_dir = Path(config_dir)
        self.max_age_days = max_age_days
        self.metadata_file = self.config_dir / '.config_metadata.json'
        
        logger.info(f"ConfigUpdater initialized: dir={self.config_dir}, max_age={max_age_days} days")
    
    def check_and_update_if_needed(self) -> bool:
        """
        Synchrone Wrapper-Methode für async Update-Check.
        
        Prüft ob Configs älter als max_age_days sind und aktualisiert sie.
        
        Returns:
            True wenn Update durchgeführt wurde, False sonst
        """
        try:
            # Erstelle neuen Event Loop falls keiner existiert
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.check_and_update_if_needed_async())
        except Exception as e:
            logger.warning(f"Config update check failed: {e}")
            return False
    
    async def check_and_update_if_needed_async(self) -> bool:
        """
        Prüft ob Configs älter als max_age_days sind und aktualisiert sie.
        
        Returns:
            True wenn Update durchgeführt wurde, False sonst
        """
        try:
            # Prüfe ob Update nötig ist
            if not self._needs_update():
                logger.info("Configs are up-to-date, no update needed")
                return False
            
            logger.info(f"Configs are older than {self.max_age_days} days, updating...")
            
            # Lade neue Configs von Catwalk
            updated_count = await self.update_all_configs()
            
            if updated_count > 0:
                # Speichere Metadaten mit aktuellem Timestamp
                self._save_metadata()
                logger.info(f"✓ Successfully updated {updated_count} config(s)")
                return True
            else:
                logger.warning("No configs were updated")
                return False
                
        except Exception as e:
            logger.error(f"Config update failed: {e}")
            return False
    
    def _needs_update(self) -> bool:
        """
        Prüft ob ein Update der Configs nötig ist.
        
        Returns:
            True wenn Configs älter als max_age_days sind oder Metadaten fehlen
        """
        # Prüfe ob Metadaten-Datei existiert
        if not self.metadata_file.exists():
            logger.info("No metadata file found, update needed")
            return True
        
        try:
            # Lade Metadaten
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Prüfe letztes Update-Datum
            last_update_str = metadata.get('last_update')
            if not last_update_str:
                logger.info("No last_update in metadata, update needed")
                return True
            
            # Parse Datum
            last_update = datetime.fromisoformat(last_update_str)
            age = datetime.now() - last_update
            
            needs_update = age.days >= self.max_age_days
            
            if needs_update:
                logger.info(f"Configs are {age.days} days old (max: {self.max_age_days}), update needed")
            else:
                logger.info(f"Configs are {age.days} days old, still fresh")
            
            return needs_update
            
        except Exception as e:
            logger.warning(f"Error reading metadata: {e}, assuming update needed")
            return True
    
    def _save_metadata(self) -> None:
        """Speichert Metadaten mit aktuellem Timestamp."""
        metadata = {
            'last_update': datetime.now().isoformat(),
            'max_age_days': self.max_age_days,
            'providers': list(self.PROVIDER_CONFIGS.keys())
        }
        
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def update_all_configs(self) -> int:
        """
        Lädt alle Provider-Configs von Catwalk und speichert sie lokal.
        
        Returns:
            Anzahl der erfolgreich aktualisierten Configs
        """
        logger.info(f"Updating all provider configs from Catwalk...")
        
        # Erstelle Tasks für alle Provider
        tasks = []
        provider_names = []
        
        for provider, filename in self.PROVIDER_CONFIGS.items():
            tasks.append(self._update_single_config(provider, filename))
            provider_names.append(provider)
        
        # Lade alle parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Zähle erfolgreiche Updates
        success_count = 0
        for provider_name, result in zip(provider_names, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to update {provider_name}: {result}")
            elif result:
                success_count += 1
                logger.info(f"✓ {provider_name} config updated")
        
        return success_count
    
    async def _update_single_config(self, provider: str, filename: str) -> bool:
        """
        Lädt eine einzelne Config von Catwalk und speichert sie lokal.
        
        Args:
            provider: Provider-Name (z.B. 'openai')
            filename: Dateiname (z.B. 'openai.json')
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        url = f"{self.CATWALK_BASE_URL}{filename}"
        local_path = self.config_dir / filename
        
        try:
            # Lade Config von URL
            config = await self._download_config(url)
            
            # Speichere lokal
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {provider} config to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update {provider} config: {e}")
            return False
    
    async def _download_config(self, url: str) -> Dict:
        """
        Lädt Config von URL.
        
        Args:
            url: URL zur Config-Datei
            
        Returns:
            Dict mit geparster JSON-Config
            
        Raises:
            Exception: Bei Netzwerk- oder Parse-Fehlern
        """
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                text = await response.text()
                return json.loads(text)


def check_and_update_configs(max_age_days: int = 7) -> bool:
    """
    Convenience-Funktion für Config-Update-Check.
    
    Args:
        max_age_days: Maximales Alter der Configs in Tagen (default: 7)
        
    Returns:
        True wenn Update durchgeführt wurde, False sonst
    """
    updater = ConfigUpdater(max_age_days=max_age_days)
    return updater.check_and_update_if_needed()

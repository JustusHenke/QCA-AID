#!/usr/bin/env python3
"""
Version Checker
===============
Prüft ob eine neue Version von QCA-AID auf GitHub verfügbar ist.
"""

import requests
from typing import Optional, Tuple
from packaging import version
import streamlit as st
import sys
import os

# Add assets path to import version
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'QCA_AID_assets'))
try:
    from __version__ import __version__ as CURRENT_VERSION
except ImportError:
    # Fallback if import fails
    CURRENT_VERSION = "0.11.2"


class VersionChecker:
    """Prüft GitHub Releases auf neue Versionen."""
    
    GITHUB_API_URL = "https://api.github.com/repos/JustusHenke/QCA-AID/releases/latest"
    # CURRENT_VERSION wird jetzt dynamisch aus __version__.py geladen
    CACHE_KEY = "version_check_result"
    CACHE_DURATION = 3600  # 1 Stunde in Sekunden
    
    @classmethod
    def check_for_updates(cls) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Prüft ob eine neue Version verfügbar ist.
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (update_available, latest_version, error_message)
        """
        try:
            # GitHub API Request mit Timeout
            response = requests.get(
                cls.GITHUB_API_URL,
                timeout=5,
                headers={"Accept": "application/vnd.github.v3+json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get("tag_name", "").lstrip("v")
                
                if latest_version:
                    # Versionen vergleichen
                    try:
                        current = version.parse(CURRENT_VERSION)
                        latest = version.parse(latest_version)
                        
                        if latest > current:
                            return True, latest_version, None
                        else:
                            return False, latest_version, None
                    except Exception as e:
                        return False, None, f"Fehler beim Vergleichen der Versionen: {str(e)}"
                else:
                    return False, None, "Keine Version in GitHub Response gefunden"
            else:
                return False, None, f"GitHub API Fehler: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, None, "Timeout beim Abrufen der Version"
        except requests.exceptions.RequestException as e:
            return False, None, f"Netzwerkfehler: {str(e)}"
        except Exception as e:
            return False, None, f"Unerwarteter Fehler: {str(e)}"
    
    @classmethod
    def get_cached_check(cls) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Holt gecachtes Ergebnis aus Session State.
        
        Returns:
            Optional[Tuple[bool, Optional[str]]]: (update_available, latest_version) oder None
        """
        import time
        
        if cls.CACHE_KEY in st.session_state:
            cached_data = st.session_state[cls.CACHE_KEY]
            timestamp = cached_data.get("timestamp", 0)
            
            # Prüfe ob Cache noch gültig ist
            if time.time() - timestamp < cls.CACHE_DURATION:
                return (
                    cached_data.get("update_available", False),
                    cached_data.get("latest_version")
                )
        
        return None
    
    @classmethod
    def cache_check_result(cls, update_available: bool, latest_version: Optional[str]):
        """
        Speichert Prüfergebnis im Session State Cache.
        
        Args:
            update_available: Ob ein Update verfügbar ist
            latest_version: Die neueste Version
        """
        import time
        
        st.session_state[cls.CACHE_KEY] = {
            "update_available": update_available,
            "latest_version": latest_version,
            "timestamp": time.time()
        }
    
    @classmethod
    def render_version_info(cls):
        """
        Rendert Versionsinformationen in der Sidebar.
        Zeigt einen Hinweis an wenn eine neue Version verfügbar ist.
        """
        # Versuche gecachtes Ergebnis zu verwenden
        cached = cls.get_cached_check()
        
        if cached is not None:
            update_available, latest_version = cached
        else:
            # Führe neue Prüfung durch (nur beim ersten Laden)
            update_available, latest_version, error = cls.check_for_updates()
            
            # Cache nur bei erfolgreicher Prüfung
            if error is None:
                cls.cache_check_result(update_available, latest_version)
        
        # Zeige aktuelle Version
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            st.caption(f"Version {CURRENT_VERSION}")
        with col2:
            # Button zum manuellen Prüfen
            if st.button("🔄", key="check_version", help="Nach Updates suchen"):
                with st.spinner("Prüfe..."):
                    update_available, latest_version, error = cls.check_for_updates()
                    if error is None:
                        cls.cache_check_result(update_available, latest_version)
                        st.rerun()
        
        # Zeige Update-Hinweis wenn verfügbar
        if update_available and latest_version:
            st.sidebar.warning(f"🆕 Version {latest_version} verfügbar! - [Update](https://github.com/JustusHenke/QCA-AID/releases/latest)")


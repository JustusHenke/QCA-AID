#!/usr/bin/env python3
"""
Performance Monitor für QCA-AID Webapp
======================================
Überwacht und misst Startup- und Runtime-Performance.
"""

import time
import streamlit as st
from typing import Dict, Optional


class PerformanceMonitor:
    """
    Performance-Monitor für Webapp-Optimierung.
    
    Misst Startup-Zeiten und identifiziert Bottlenecks.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}
        self.enabled = False
    
    def checkpoint(self, name: str):
        """
        Setzt einen Performance-Checkpoint.
        
        Args:
            name: Name des Checkpoints
        """
        if self.enabled:
            self.checkpoints[name] = time.time() - self.start_time
    
    def get_duration(self, name: str) -> Optional[float]:
        """
        Gibt die Dauer bis zu einem Checkpoint zurück.
        
        Args:
            name: Name des Checkpoints
            
        Returns:
            Dauer in Sekunden oder None
        """
        return self.checkpoints.get(name)
    
    def get_total_time(self) -> float:
        """Gibt die Gesamtzeit seit Start zurück."""
        return time.time() - self.start_time
    
    def render_sidebar_debug(self):
        """
        Rendert Performance-Debug-Informationen in der Sidebar.
        """
        if st.sidebar.checkbox("🔍 Performance Debug", value=False):
            st.sidebar.markdown("### ⏱️ Startup Performance")
            
            total_time = self.get_total_time()
            st.sidebar.metric("Gesamtzeit", f"{total_time:.2f}s")
            
            if self.checkpoints:
                st.sidebar.markdown("**Checkpoints:**")
                for name, duration in self.checkpoints.items():
                    st.sidebar.text(f"• {name}: {duration:.2f}s")
            
            # Performance-Tipps
            st.sidebar.markdown("---")
            st.sidebar.markdown("**💡 Performance-Tipps:**")
            
            if total_time > 5:
                st.sidebar.warning("⚠️ Langsamer Start (>5s)")
                st.sidebar.caption("Prüfe Lazy Loading")
            elif total_time > 3:
                st.sidebar.info("ℹ️ Moderater Start (>3s)")
            else:
                st.sidebar.success("✅ Schneller Start (<3s)")
    
    def enable_monitoring(self):
        """Aktiviert Performance-Monitoring."""
        self.enabled = True
        self.start_time = time.time()
        self.checkpoints.clear()
    
    def disable_monitoring(self):
        """Deaktiviert Performance-Monitoring."""
        self.enabled = False


# Globale Monitor-Instanz
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Gibt die globale Performance-Monitor-Instanz zurück.
    
    Returns:
        PerformanceMonitor-Instanz
    """
    return _monitor


def checkpoint(name: str):
    """
    Convenience-Funktion für Performance-Checkpoints.
    
    Args:
        name: Name des Checkpoints
    """
    _monitor.checkpoint(name)


def enable_performance_monitoring():
    """Aktiviert Performance-Monitoring."""
    _monitor.enable_monitoring()


def render_performance_debug():
    """Rendert Performance-Debug-UI."""
    _monitor.render_sidebar_debug()
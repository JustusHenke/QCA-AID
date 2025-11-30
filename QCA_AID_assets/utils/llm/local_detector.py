"""
Local Detector for LLM Provider Manager

Detects and loads local LLM models from LM Studio and Ollama servers.
Provides automatic discovery of locally running models with graceful degradation.
"""

import asyncio
import json
import logging
import subprocess
from typing import List, Dict, Optional
import aiohttp

logger = logging.getLogger(__name__)


class LocalDetector:
    """
    Erkennt lokale LLM-Server (LM Studio, Ollama).
    
    Der LocalDetector prüft ob lokale LLM-Server laufen und lädt
    verfügbare Modelle. Unterstützt LM Studio (OpenAI-kompatible API)
    und Ollama (eigene API + CLI-Fallback).
    
    Attributes:
        LM_STUDIO_URL: URL für LM Studio API-Endpoint
        OLLAMA_API_URL: URL für Ollama API-Endpoint
    
    Example:
        >>> detector = LocalDetector()
        >>> models = await detector.detect_all()
        >>> # models = [{'id': 'llama2', 'name': 'Llama 2', ...}, ...]
    """
    
    LM_STUDIO_URL = "http://localhost:1234/v1/models"
    OLLAMA_API_URL = "http://localhost:11434/api/tags"
    
    async def detect_all(self) -> List[Dict]:
        """
        Erkennt alle verfügbaren lokalen Modelle.
        
        Führt parallele Erkennung von LM Studio und Ollama durch.
        Fehler bei einzelnen Servern werden geloggt, aber nicht propagiert.
        
        Returns:
            Liste von Modell-Dicts im Raw-Format.
            Leere Liste wenn keine lokalen Server erreichbar sind.
        
        Example:
            >>> models = await detector.detect_all()
            >>> print(f"Found {len(models)} local models")
        """
        logger.info("Detecting local models from LM Studio and Ollama")
        
        # Führe beide Erkennungen parallel aus
        results = await asyncio.gather(
            self.detect_lm_studio(),
            self.detect_ollama(),
            return_exceptions=True
        )
        
        # Sammle alle erfolgreichen Ergebnisse
        all_models = []
        
        for i, result in enumerate(results):
            server_name = ["LM Studio", "Ollama"][i]
            
            if isinstance(result, Exception):
                logger.warning(f"{server_name} detection failed: {result}")
            elif isinstance(result, list):
                all_models.extend(result)
                logger.info(f"✓ {server_name}: Found {len(result)} models")
            else:
                logger.warning(f"{server_name} returned unexpected result type: {type(result)}")
        
        logger.info(f"Total local models detected: {len(all_models)}")
        return all_models
    
    async def detect_lm_studio(self) -> List[Dict]:
        """
        Erkennt LM Studio Modelle.
        
        Ruft /v1/models Endpoint auf (OpenAI-kompatibel).
        Prüft zuerst ob Server erreichbar ist.
        
        Returns:
            Liste von Modell-Dicts im LM Studio Format.
            Leere Liste wenn Server nicht erreichbar.
        
        Raises:
            Exception: Bei Netzwerk- oder Parsing-Fehlern
        
        Example:
            >>> models = await detector.detect_lm_studio()
        """
        if not await self._is_server_running(self.LM_STUDIO_URL):
            logger.debug("LM Studio server not running")
            return []
        
        try:
            logger.info(f"Querying LM Studio at {self.LM_STUDIO_URL}")
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.LM_STUDIO_URL) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Validiere Response-Format
            if not isinstance(data, dict):
                logger.warning(
                    f"LM Studio returned invalid response format: "
                    f"Expected dict, got {type(data).__name__}"
                )
                return []
            
            # LM Studio verwendet OpenAI-kompatibles Format
            # Response: {"data": [{"id": "...", "object": "model", ...}]}
            models = data.get('data', [])
            
            if not isinstance(models, list):
                logger.warning(
                    f"LM Studio 'data' field has invalid format: "
                    f"Expected list, got {type(models).__name__}"
                )
                return []
            
            logger.info(f"LM Studio returned {len(models)} models")
            return models
            
        except aiohttp.ClientError as e:
            logger.warning(f"Network error querying LM Studio: {type(e).__name__}: {e}")
            return []
        except asyncio.TimeoutError:
            logger.warning(f"Timeout querying LM Studio (5 seconds)")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response from LM Studio: {e}")
            return []
        except Exception as e:
            logger.warning(
                f"Unexpected error detecting LM Studio models: "
                f"{type(e).__name__}: {e}"
            )
            return []
    
    async def detect_ollama(self) -> List[Dict]:
        """
        Erkennt Ollama Modelle.
        
        Versucht zuerst API, dann CLI-Fallback.
        Prüft zuerst ob Server erreichbar ist.
        
        Returns:
            Liste von Modell-Dicts im Ollama Format.
            Leere Liste wenn Server nicht erreichbar und CLI fehlschlägt.
        
        Example:
            >>> models = await detector.detect_ollama()
        """
        # Versuch 1: API
        if await self._is_server_running(self.OLLAMA_API_URL):
            try:
                logger.info(f"Querying Ollama API at {self.OLLAMA_API_URL}")
                
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(self.OLLAMA_API_URL) as response:
                        response.raise_for_status()
                        data = await response.json()
                
                # Validiere Response-Format
                if not isinstance(data, dict):
                    logger.warning(
                        f"Ollama API returned invalid response format: "
                        f"Expected dict, got {type(data).__name__}"
                    )
                    logger.debug("Trying CLI fallback")
                else:
                    # Ollama API Format: {"models": [{"name": "...", "model": "...", ...}]}
                    models = data.get('models', [])
                    
                    if not isinstance(models, list):
                        logger.warning(
                            f"Ollama API 'models' field has invalid format: "
                            f"Expected list, got {type(models).__name__}"
                        )
                        logger.debug("Trying CLI fallback")
                    else:
                        logger.info(f"Ollama API returned {len(models)} models")
                        return models
                
            except aiohttp.ClientError as e:
                logger.warning(
                    f"Network error querying Ollama API: {type(e).__name__}: {e}, "
                    f"trying CLI fallback"
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout querying Ollama API (5 seconds), trying CLI fallback")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response from Ollama API: {e}, trying CLI fallback")
            except Exception as e:
                logger.warning(
                    f"Unexpected error querying Ollama API: {type(e).__name__}: {e}, "
                    f"trying CLI fallback"
                )
        else:
            logger.debug("Ollama API server not running, trying CLI fallback")
        
        # Versuch 2: CLI-Fallback
        try:
            models = self._detect_ollama_cli()
            logger.info(f"Ollama CLI returned {len(models)} models")
            return models
        except Exception as e:
            logger.warning(f"Ollama CLI fallback failed: {type(e).__name__}: {e}")
            return []
    
    def _detect_ollama_cli(self) -> List[Dict]:
        """
        Fallback: Ollama via CLI.
        
        Führt 'ollama list' Kommando aus und parst die Ausgabe.
        
        Returns:
            Liste von Modell-Dicts im Ollama Format
        
        Raises:
            Exception: Wenn CLI-Kommando fehlschlägt oder nicht verfügbar ist
        
        Example:
            >>> models = detector._detect_ollama_cli()
        """
        try:
            logger.debug("Attempting Ollama CLI detection")
            
            # Führe 'ollama list' aus
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5,
                check=True
            )
            
            # Parse Ausgabe
            # Format:
            # NAME                    ID              SIZE    MODIFIED
            # llama2:latest          abc123def456    3.8 GB  2 weeks ago
            
            lines = result.stdout.strip().split('\n')
            
            if len(lines) < 2:  # Header + mindestens ein Modell
                logger.debug("No models found in ollama list output")
                return []
            
            models = []
            # Überspringe Header-Zeile
            for line_num, line in enumerate(lines[1:], start=2):
                try:
                    parts = line.split()
                    if len(parts) >= 1:
                        model_name = parts[0]
                        model_id = parts[1] if len(parts) >= 2 else model_name
                        
                        models.append({
                            'name': model_name,
                            'model': model_name,
                            'id': model_id
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to parse Ollama CLI output line {line_num}: {e}. "
                        f"Line: '{line}'"
                    )
                    continue
            
            logger.debug(f"Parsed {len(models)} models from CLI output")
            return models
            
        except FileNotFoundError:
            logger.debug("Ollama CLI not found in PATH")
            raise Exception("Ollama CLI not installed or not in PATH")
        except subprocess.TimeoutExpired:
            logger.warning("Ollama CLI command timed out after 5 seconds")
            raise Exception("Ollama CLI command timed out after 5 seconds")
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Ollama CLI command failed with exit code {e.returncode}: "
                f"stderr: {e.stderr}"
            )
            raise Exception(f"Ollama CLI command failed with exit code {e.returncode}")
        except Exception as e:
            logger.error(
                f"Unexpected error running Ollama CLI: {type(e).__name__}: {e}"
            )
            raise Exception(f"Unexpected error running Ollama CLI: {e}")
    
    async def _is_server_running(self, url: str) -> bool:
        """
        Prüft ob Server erreichbar ist.
        
        Sendet einfachen HEAD-Request um Erreichbarkeit zu prüfen.
        
        Args:
            url: URL des zu prüfenden Servers
        
        Returns:
            bool: True wenn Server erreichbar, sonst False
        
        Example:
            >>> is_running = await detector._is_server_running("http://localhost:1234")
        """
        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url) as response:
                    # Akzeptiere alle 2xx und 4xx Status-Codes
                    # (4xx bedeutet Server läuft, aber Endpoint existiert nicht)
                    is_running = response.status < 500
                    logger.debug(
                        f"Server check for {url}: "
                        f"{'running' if is_running else 'not running'} "
                        f"(status: {response.status})"
                    )
                    return is_running
        except aiohttp.ClientError as e:
            logger.debug(f"Server check for {url}: not running (network error: {e})")
            return False
        except asyncio.TimeoutError:
            logger.debug(f"Server check for {url}: not running (timeout)")
            return False
        except Exception as e:
            logger.debug(
                f"Server check for {url}: not running "
                f"(unexpected error: {type(e).__name__}: {e})"
            )
            return False

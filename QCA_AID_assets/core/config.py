"""
Konfiguration für QCA-AID
=========================
Enthält globale Einstellungen, Forschungsfrage, Kodierregeln und Kategorien.
"""

import os
from dotenv import load_dotenv


# ============================
# FORSCHUNGSFRAGE
# ============================

FORSCHUNGSFRAGE = "Wie gestaltet sich [Phänomen] im Kontext von [Setting] und welche [Aspekt] lassen sich dabei identifizieren?"


# ============================
# ALLGEMEINE KODIERREGELN
# ============================

KODIERREGELN = {
    "general": [
        "Kodiere nur manifeste, nicht latente Inhalte",
        "Berücksichtige den Kontext der Aussage",
        "Bei Unsicherheit dokumentiere die Gründe",
        "Kodiere vollständige Sinneinheiten",
        "Prüfe Überschneidungen zwischen Kategorien"
    ],
    "format": [
        "Markiere relevante Textstellen",
        "Dokumentiere Begründung der Zuordnung",
        "Gib Konfidenzwert (1-5) an",
        "Notiere eventuelle Querverbindungen zu anderen Kategorien"
    ]
}


# ============================
# DEDUKTIVE KATEGORIEN
# ============================

DEDUKTIVE_KATEGORIEN = {
    "Akteure": {
        "definition": "Erfasst alle handelnden Personen, Gruppen oder Institutionen sowie deren Rollen, Beziehungen und Interaktionen",
        "rules": "Codiere Aussagen zu: Individuen, Gruppen, Organisationen, Netzwerken",
        "subcategories": {
            "Individuelle_Akteure": "Einzelpersonen und deren Eigenschaften",
            "Kollektive_Akteure": "Gruppen, Organisationen, Institutionen",
            "Beziehungen": "Interaktionen, Hierarchien, Netzwerke",
            "Rollen": "Formelle und informelle Positionen"
        },
        "examples": {
            "Die Projektleiterin hat die Entscheidung eigenständig getroffen",
            "Die Arbeitsgruppe trifft sich wöchentlich zur Abstimmung",
            "Als Vermittler zwischen den Parteien konnte er den Konflikt lösen",
            "Die beteiligten Organisationen haben eine Kooperationsvereinbarung unterzeichnet"
        }
    },
    "Kontextfaktoren": {
        "definition": "Umfasst die strukturellen, zeitlichen und räumlichen Rahmenbedingungen des untersuchten Phänomens",
        "subcategories": {
            "Strukturell": "Organisatorische und institutionelle Bedingungen",
            "Zeitlich": "Historische Entwicklung, Zeitpunkte, Perioden",
            "Räumlich": "Geografische und sozialräumliche Aspekte",
            "Kulturell": "Normen, Werte, Traditionen"
        }
    },
    "Prozesse": {
        "definition": "Erfasst Abläufe, Entwicklungen und Veränderungen über Zeit",
        "subcategories": {
            "Entscheidungsprozesse": "Formelle und informelle Entscheidungsfindung",
            "Entwicklungsprozesse": "Veränderungen und Transformationen",
            "Interaktionsprozesse": "Kommunikation und Austausch",
            "Konfliktprozesse": "Aushandlungen und Konflikte"
        }
    },
    "Ressourcen": {
        "definition": "Materielle und immaterielle Mittel und Kapazitäten",
        "subcategories": {
            "Materiell": "Finanzielle und physische Ressourcen",
            "Immateriell": "Wissen, Kompetenzen, soziales Kapital",
            "Zugang": "verfügbarkeit und Verteilung",
            "Nutzung": "Einsatz und Verwertung"
        }
    },
    "Strategien": {
        "definition": "Handlungsmuster und -konzepte zur Zielerreichung",
        "subcategories": {
            "Formell": "Offizielle Strategien und Pläne",
            "Informell": "Ungeschriebene Praktiken",
            "Adaptiv": "Anpassungsstrategien",
            "Innovativ": "Neue Lösungsansätze"
        }
    },
    "Outcomes": {
        "definition": "Ergebnisse, Wirkungen und Folgen von Handlungen und Prozessen",
        "subcategories": {
            "Intendiert": "Beabsichtigte Wirkungen",
            "Nicht_intendiert": "Unbeabsichtigte Folgen",
            "Kurzfristig": "Unmittelbare Effekte",
            "Langfristig": "Nachhaltige Wirkungen"
        }
    },
    "Herausforderungen": {
        "definition": "Probleme, Hindernisse und Spannungsfelder",
        "subcategories": {
            "Strukturell": "Systemische Barrieren",
            "Prozessual": "Ablaufbezogene Schwierigkeiten",
            "Individuell": "Persönliche Herausforderungen",
            "Kontextuell": "Umfeldbezogene Probleme"
        }
    },
    "Legitimation": {
        "definition": "Begründungen, Rechtfertigungen und Deutungsmuster",
        "subcategories": {
            "Normativ": "Wertbasierte Begründungen",
            "Pragmatisch": "Praktische Rechtfertigungen",
            "Kognitiv": "Wissensbasierte Erklärungen",
            "Emotional": "Gefühlsbezogene Deutungen"
        }
    }
}


# ============================
# VALIDIERUNGS-SCHWELLENWERTE
# ============================

VALIDATION_THRESHOLDS = {
    'MIN_DEFINITION_WORDS': 15,
    'MIN_EXAMPLES': 2,
    'SIMILARITY_THRESHOLD': 0.7,
    'MIN_SUBCATEGORIES': 2,
    'MAX_NAME_LENGTH': 50,
    'MIN_NAME_LENGTH': 3
}


# ============================
# HAUPTKONFIGURATION
# ============================

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gehe ein Level höher
CONFIG = {
    'MODEL_PROVIDER': 'OpenAI',
    'MODEL_NAME': 'gpt-4o-mini',
    'DATA_DIR': 'input',  # Relativ zum Projektverzeichnis
    'OUTPUT_DIR': 'output',  # Relativ zum Projektverzeichnis
    'CHUNK_SIZE': 1200,
    'CHUNK_OVERLAP': 50,
    'BATCH_SIZE': 8,
    'CODE_WITH_CONTEXT': False,
    'CONTEXT_PARAPHRASE_COUNT': 3,  # Anzahl vorheriger Paraphrasen als Kontext (wenn CODE_WITH_CONTEXT aktiviert)
    'MULTIPLE_CODINGS': True, 
    'MULTIPLE_CODING_THRESHOLD': 0.85,  # Schwellenwert für zusätzliche Relevanz
    'RELEVANCE_THRESHOLD': 0.3,  # Mindest-Konfidenz für relevante Segmente (0.3 = LLM-Standard, höhere Werte = strenger, niedrigere Werte = inkludieren auch LLM-verworfene Segmente)
    'ENABLE_OPTIMIZATION': True,  # Feature Flag für neue optimierte Analyse (Batching, Caching)
    'ANALYSIS_MODE': 'deductive',
    'REVIEW_MODE': 'consensus',
    'ATTRIBUTE_LABELS': {
        'attribut1': 'Attribut1',
        'attribut2': 'Attribut2',
        'attribut3': 'Attribut3'  
    },
    'EXPORT_ANNOTATED_PDFS': True,  # Aktiviert/deaktiviert PDF-Annotation
    'PDF_ANNOTATION_FUZZY_THRESHOLD': 0.85,  # Schwellenwert für Fuzzy-Text-Matching
    'PDF_SIDEBAR_BAR_WIDTH': 8,  # Breite der Sidebar-Marker in Pixeln
    'PDF_SIDEBAR_SPACING': 2,  # Abstand zwischen Sidebar-Markern
    'INDUCTIVE_CODER_TEMPERATURE': 0.2,  # Temperature für den InductiveCoder
    'RELEVANCE_CHECK_TEMPERATURE': 0.3,  # Temperature für die Relevanzprüfung
    'CODER_SETTINGS': [
        {
            'temperature': 0.3,
            'coder_id': 'auto_1'
        },
        {
            'temperature': 0.5,
            'coder_id': 'auto_2'
        }
    ],
    
    # ============================
    # LLM PROVIDER MANAGER KONFIGURATION
    # ============================
    # Konfiguration für den erweiterten LLM Provider Manager
    # Der Provider Manager lädt automatisch Modell-Metadaten von mehreren
    # Providern (OpenAI, Anthropic, Mistral, OpenRouter, lokale Modelle)
    
    'PROVIDER_MANAGER': {
        # Cache-Verzeichnis für Provider- und Modell-Informationen
        # Modell-Metadaten werden hier gecacht um wiederholte Netzwerk-Anfragen zu vermeiden
        # Cache ist 24 Stunden gültig
        'CACHE_DIR': os.path.join(os.path.expanduser("~"), '.llm_cache'),
        
        # Verzeichnis für lokale Fallback-Konfigurationen
        # Wenn Catwalk GitHub nicht erreichbar ist, werden Provider-Configs aus diesem
        # Verzeichnis geladen. Standard: QCA_AID_assets/utils/llm/configs/
        'FALLBACK_DIR': os.path.join(SCRIPT_DIR, 'utils', 'llm', 'configs'),
        
        # Pfad zur pricing_overrides.json Datei
        # Ermöglicht das Überschreiben von Modell-Kosten mit eigenen Werten
        # Format: {"model_id": {"cost_in": 1.0, "cost_out": 2.0}}
        'CONFIG_DIR': os.path.join(SCRIPT_DIR, 'utils', 'llm'),
        
        # Automatische Initialisierung beim Import
        # Wenn True, wird der Provider Manager automatisch beim ersten Zugriff initialisiert
        # Wenn False, muss initialize() manuell aufgerufen werden
        'AUTO_INITIALIZE': False,
        
        # Erzwingt Neuladen von Provider-Daten beim Start
        # Wenn True, wird Cache ignoriert und Daten werden neu von externen Quellen geladen
        # Nützlich für Entwicklung oder wenn aktuelle Modell-Informationen benötigt werden
        'FORCE_REFRESH': False,
    }
}

# Verzeichnisse werden bei Bedarf von der Webapp erstellt
# (nicht beim Import, da Pfade relativ zum Projektverzeichnis sind)

# Lade Umgebungsvariablen
env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
load_dotenv(env_path)


# ============================
# PROVIDER MANAGER HILFSFUNKTIONEN
# ============================

def get_provider_manager_config():
    """
    Gibt die Provider Manager Konfiguration zurück.
    
    Diese Funktion bietet Zugriff auf die Provider Manager Einstellungen
    und stellt Backward-Compatibility sicher.
    
    Returns:
        dict: Provider Manager Konfiguration mit folgenden Keys:
            - cache_dir: Verzeichnis für Cache-Dateien
            - fallback_dir: Verzeichnis für lokale Fallback-Configs
            - config_dir: Verzeichnis für pricing_overrides.json
            - auto_initialize: Ob automatische Initialisierung aktiviert ist
            - force_refresh: Ob Cache beim Start ignoriert werden soll
    
    Example:
        >>> config = get_provider_manager_config()
        >>> print(config['cache_dir'])
        ~/.llm_cache
    """
    return CONFIG.get('PROVIDER_MANAGER', {
        'CACHE_DIR': os.path.join(os.path.expanduser("~"), '.llm_cache'),
        'FALLBACK_DIR': os.path.join(SCRIPT_DIR, 'utils', 'llm', 'configs'),
        'CONFIG_DIR': os.path.join(SCRIPT_DIR, 'utils', 'llm'),
        'AUTO_INITIALIZE': False,
        'FORCE_REFRESH': False,
    })


async def create_provider_manager():
    """
    Erstellt und initialisiert einen LLMProviderManager mit den Einstellungen aus CONFIG.
    
    Diese Convenience-Funktion erstellt einen Provider Manager mit den
    Konfigurationseinstellungen aus CONFIG['PROVIDER_MANAGER'] und initialisiert
    ihn optional automatisch.
    
    Returns:
        LLMProviderManager: Initialisierter Provider Manager
    
    Raises:
        ImportError: Wenn LLMProviderManager nicht importiert werden kann
        ProviderLoadError: Wenn keine Provider geladen werden konnten
    
    Example:
        >>> import asyncio
        >>> manager = asyncio.run(create_provider_manager())
        >>> all_models = manager.get_all_models()
        >>> print(f"Loaded {len(all_models)} models")
    
    Note:
        Diese Funktion ist async und muss mit await oder asyncio.run() aufgerufen werden.
        Der Provider Manager wird automatisch initialisiert wenn AUTO_INITIALIZE=True.
    """
    try:
        # Versuche relativen Import (wenn als Modul verwendet)
        try:
            from ..utils.llm.provider_manager import LLMProviderManager
        except (ImportError, ValueError):
            # Fallback auf absoluten Import (wenn direkt ausgeführt)
            from QCA_AID_assets.utils.llm.provider_manager import LLMProviderManager
    except ImportError as e:
        raise ImportError(
            f"Failed to import LLMProviderManager: {e}. "
            "Ensure the provider_manager module is available."
        )
    
    pm_config = get_provider_manager_config()
    
    # Erstelle Provider Manager mit Konfiguration
    manager = LLMProviderManager(
        cache_dir=pm_config['CACHE_DIR'],
        fallback_dir=pm_config['FALLBACK_DIR'],
        config_dir=pm_config['CONFIG_DIR']
    )
    
    # Initialisiere automatisch wenn konfiguriert
    if pm_config.get('AUTO_INITIALIZE', False):
        force_refresh = pm_config.get('FORCE_REFRESH', False)
        await manager.initialize(force_refresh=force_refresh)
    
    return manager


def get_legacy_provider_config():
    """
    Gibt die Legacy-Provider-Konfiguration zurück (Backward-Compatibility).
    
    Diese Funktion extrahiert die traditionellen Provider-Einstellungen
    aus CONFIG für bestehenden Code, der noch nicht auf den Provider Manager
    migriert wurde.
    
    Returns:
        dict: Legacy-Konfiguration mit Keys:
            - provider: Provider-Name (z.B. 'OpenAI', 'Anthropic')
            - model_name: Modell-Name (z.B. 'gpt-4o-mini')
    
    Example:
        >>> legacy_config = get_legacy_provider_config()
        >>> print(f"Using {legacy_config['provider']} with {legacy_config['model_name']}")
    
    Note:
        Diese Funktion ist für Backward-Compatibility gedacht.
        Neuer Code sollte den LLMProviderManager verwenden.
    """
    return {
        'provider': CONFIG.get('MODEL_PROVIDER', 'OpenAI'),
        'model_name': CONFIG.get('MODEL_NAME', 'gpt-4o-mini')
    }

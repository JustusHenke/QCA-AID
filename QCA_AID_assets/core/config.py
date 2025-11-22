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
            "Zugang": "Verfügbarkeit und Verteilung",
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
    'DATA_DIR': os.path.join(SCRIPT_DIR, 'input'),
    'OUTPUT_DIR': os.path.join(SCRIPT_DIR, 'output'),
    'CHUNK_SIZE': 1200,
    'CHUNK_OVERLAP': 50,
    'BATCH_SIZE': 8,
    'CODE_WITH_CONTEXT': False,
    'CONTEXT_PARAPHRASE_COUNT': 3,  # Anzahl vorheriger Paraphrasen als Kontext (wenn CODE_WITH_CONTEXT aktiviert)
    'MULTIPLE_CODINGS': True, 
    'MULTIPLE_CODING_THRESHOLD': 0.85,  # Schwellenwert für zusätzliche Relevanz
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
    'RELEVANCE_CHECK_TEMPERATURE': 0.3,  # Temperature für die RelevanzprÜfung
    'CODER_SETTINGS': [
        {
            'temperature': 0.3,
            'coder_id': 'auto_1'
        },
        {
            'temperature': 0.5,
            'coder_id': 'auto_2'
        }
    ]
}

# Stelle sicher, dass die Verzeichnisse existieren
os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# Lade Umgebungsvariablen
env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
load_dotenv(env_path)

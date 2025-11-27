# QCA-AID Explorer

## Überblick
QCA-AID Explorer ist ein leistungsstarkes Tool zur Analyse qualitativer Kodierungsdaten. Es ermöglicht die Visualisierung von Kodiernetzwerken mit Hauptkategorien, Subkategorien und Schlüsselwörtern sowie die automatisierte Zusammenfassung von kodierten Textsegmenten mithilfe von Large Language Models (LLMs).

**Hinweis:** QCA-AID Explorer folgt der Versionierung von QCA-AID. Änderungen und Updates werden im Haupt-CHANGELOG dokumentiert.

## Hauptfunktionen

- **Modulare Architektur**: Vollständiges Refactoring in eine modulare Struktur innerhalb von `QCA_AID_assets`
- **Minimales Launcher-Skript**: Das Hauptskript `QCA-AID-Explorer.py` ist nur noch ein schlanker Launcher (< 50 Zeilen)
- **JSON-Konfigurationsunterstützung**: Konfiguration wahlweise als JSON oder Excel-Datei
- **Automatische Synchronisation**: Bidirektionale Synchronisation zwischen Excel- und JSON-Konfigurationsdateien
- **Netzwerk-Visualisierung**: Visualisierung von Kodiernetzwerken mit ForceAtlas2-Layout
- **Heatmap-Analyse**: Visualisierung von Code-Häufigkeiten entlang Dokumentattributen
- **LLM-basierte Zusammenfassungen**: Automatisierte Zusammenfassungen von kodierten Textsegmenten
- **Sentiment-Analyse**: Schlüsselwort-basierte Sentiment-Analyse mit Bubble-Visualisierung
- **Vereinheitlichte LLM Provider**: Nutzt die ausgereiften LLM Provider aus QCA-AID mit Model Capability Detection
- **Robuste Fehlerbehandlung**: Automatische Normalisierung von Spaltennamen, Behandlung leerer Graphen
- **Keyword-Harmonisierung**: Automatische Vereinheitlichung ähnlicher Schlüsselwörter

## Architektur

### Modulare Struktur

QCA-AID Explorer wurde vollständig refaktoriert und folgt einer modularen Architektur. Die gesamte Funktionalität ist in logische Module innerhalb des `QCA_AID_assets` Pakets organisiert.

#### Projektstruktur

```
QCA-AID-Explorer.py              # Minimales Launcher-Skript (< 50 Zeilen)
QCA_AID_assets/
├── explorer.py                  # Hauptfunktion für QCA-AID Explorer
├── analysis/
│   ├── __init__.py
│   └── qca_analyzer.py         # QCAAnalyzer Klasse mit allen Analysemethoden
├── utils/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py           # ConfigLoader für Konfigurationsverwaltung
│   │   ├── converter.py        # Konvertierung zwischen Excel und JSON
│   │   └── synchronizer.py     # Synchronisation von Konfigurationsdateien
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── response.py         # LLMResponse Wrapper-Klasse
│   │   ├── factory.py          # LLM Provider Factory
│   │   ├── openai_provider.py  # OpenAI Provider
│   │   └── mistral_provider.py # Mistral Provider
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── layout.py           # Layout-Algorithmen (ForceAtlas2)
│   ├── prompts.py              # Standard-Prompts für Analysen
│   └── common.py               # Allgemeine Hilfsfunktionen
└── ...                         # Weitere QCA-AID Module
```

#### Vorteile der modularen Architektur

1. **Wartbarkeit**: Jedes Modul hat eine klar definierte Verantwortlichkeit
2. **Testbarkeit**: Einzelne Komponenten können isoliert getestet werden
3. **Wiederverwendbarkeit**: Module können in anderen Projekten verwendet werden
4. **Erweiterbarkeit**: Neue Funktionen können als separate Module hinzugefügt werden
5. **Lesbarkeit**: Kleinere, fokussierte Dateien sind einfacher zu verstehen

#### Verwendung der Module

Die Module können direkt importiert und verwendet werden:

```python
# Konfiguration laden
from QCA_AID_assets.utils.config.loader import ConfigLoader
config_loader = ConfigLoader("QCA-AID-Explorer-Config.xlsx")
base_config, analysis_configs = config_loader.load_config()

# Analyzer initialisieren
from QCA_AID_assets.analysis.qca_analyzer import QCAAnalyzer
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory

llm_provider = LLMProviderFactory.create_provider(
    provider_name="openai",
    model="gpt-4o-mini"
)
analyzer = QCAAnalyzer(excel_path, llm_provider, base_config)

# Analysen durchführen
filtered_df = analyzer.filter_data(filters)
analyzer.create_network_graph(filtered_df, "output.pdf")
```

#### Änderungen im Vergleich zur alten Struktur

**Vorher:**
- Monolithische Datei `QCA-AID-Explorer.py` mit ~2500 Zeilen
- Alle Klassen und Funktionen in einer Datei
- Schwierig zu testen und zu erweitern

**Nachher:**
- Minimales Launcher-Skript `QCA-AID-Explorer.py` mit < 50 Zeilen
- Alle Funktionalitäten in logische Module aufgeteilt
- Klare Trennung von Verantwortlichkeiten
- Vollständige Dokumentation mit Docstrings
- Einfach zu testen und zu erweitern

**Wichtig:** Die Funktionalität bleibt vollständig erhalten. Alle Features sind weiterhin verfügbar und funktionieren identisch.

## Installation

### Voraussetzungen
- Python 3.8 oder höher
- Pip (Python Package Manager)

### Benötigte Pakete installieren
Führen Sie den folgenden Befehl aus, um alle erforderlichen Abhängigkeiten zu installieren:

```bash
pip install -r requirements.txt
```

Alternativ können Sie die Pakete einzeln installieren:

```bash
pip install networkx reportlab scikit-learn pandas openpyxl matplotlib seaborn httpx python-dotenv openai mistralai python-docx numpy circlify
```

### API-Schlüssel konfigurieren
Das Tool unterstützt folgende LLM-Provider:
- OpenAI
- Mistral AI

Erstellen Sie eine Datei namens `.environ.env` in Ihrem Benutzerverzeichnis mit folgendem Inhalt:

```
OPENAI_API_KEY=ihr_openai_api_schlüssel
MISTRAL_API_KEY=ihr_mistral_api_schlüssel
```

## Einrichtung

1. **Legen Sie den Skriptordner an:**
   - Erstellen Sie einen Ordner für das QCA-AID Explorer Skript
   - Erstellen Sie einen Unterordner namens "output" für die Ausgabedateien

2. **Konfigurationsdatei vorbereiten:**
   - Erstellen Sie eine Excel-Datei namens "QCA-AID-Explorer-Config.xlsx" im Skriptordner
   - Fügen Sie ein Basis-Sheet und mindestens ein Analysesheet hinzu (siehe unten)

3. **Datendatei platzieren:**
   - Legen Sie Ihre Kodierungsdaten-Excel-Datei im "output"-Ordner ab
   - Die Datei sollte ein Tabellenblatt namens "Kodierte_Segmente" mit Ihren qualitativen Daten enthalten

## Konfiguration

Die Konfiguration erfolgt über eine Konfigurationsdatei, die entweder als Excel-Datei ("QCA-AID-Explorer-Config.xlsx") oder als JSON-Datei ("QCA-AID-Explorer-Config.json") vorliegen kann.

### JSON vs. Excel Konfiguration

Das Tool unterstützt beide Formate:

- **Excel-Format (.xlsx)**: Traditionelles Format, einfach zu bearbeiten mit Excel oder LibreOffice
- **JSON-Format (.json)**: Maschinenlesbarer, versionskontrollfreundlicher, schneller zu laden

**Automatische Synchronisation:**
- Beim ersten Start wird automatisch eine JSON-Datei aus der Excel-Konfiguration erstellt
- Bei jedem Start prüft das Tool auf Differenzen zwischen beiden Dateien
- Wenn Unterschiede gefunden werden, werden Sie gefragt, welche Version aktueller ist
- Die gewählte Version wird dann in beide Formate synchronisiert
- JSON wird bevorzugt geladen, wenn beide Dateien vorhanden und identisch sind

**Vorteile von JSON:**
- Schnelleres Laden der Konfiguration
- Bessere Unterstützung für Git und andere Versionskontrollsysteme
- Einfachere programmatische Bearbeitung
- UTF-8 Encoding ohne Probleme

### Konfigurationsstruktur

Die Konfiguration besteht aus folgenden Teilen:

### 1. Basis-Sheet
Das Blatt "Basis" enthält globale Parameter für alle Analysen:

| Parameter | Beschreibung | Beispielwert |
|-----------|--------------|--------------|
| provider | Name des LLM-Providers (openai oder mistral) | openai |
| model | Name des zu verwendenden Modells | gpt-4o-mini |
| temperature | Kreativitätsparameter (0.0 - 1.0) | 0.7 |
| script_dir | Pfad zum Skriptverzeichnis | C:/Projekte/QCA-Explorer |
| output_dir | Name des Ausgabeverzeichnisses | output |
| explore_file | Name der zu analysierenden Excel-Datei | Meine_Analyse_20250407.xlsx |
| clean_keywords | Harmonisierung von Schlüsselwörtern aktivieren | True |
| similarity_threshold | Schwellenwert für Ähnlichkeit bei Harmonisierung | 0.7 |

### 2. Analyseblätter
Jedes weitere Blatt definiert eine eigene Analyse und muss mindestens diese Parameter enthalten:

| Parameter | Beschreibung |
|-----------|--------------|
| analysis_type | Art der Analyse (netzwerk, heatmap, summary_paraphrase, summary_reasoning, custom_summary) |
| filter_* | Filter für die Datenauswahl (z.B. filter_Hauptkategorie, filter_Dokument) |

### Analysetypen und spezifische Parameter

#### Netzwerkanalyse (`analysis_type: netzwerk`)
Erstellt eine Netzwerkvisualisierung der Codes.

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| node_size_factor | Faktor für die Knotengröße | 1.0 |
| layout_iterations | Anzahl der Layout-Iterationen | 100 |
| gravity | Anziehungskraft zum Zentrum | 0.05 |
| scaling | Skalierungsfaktor für Abstände | 2.0 |

#### Heatmap (`analysis_type: heatmap`)
Erstellt eine Heatmap der Codehäufigkeiten.

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| x_attribute | Spalte für die X-Achse | Dokument |
| y_attribute | Spalte für die Y-Achse | Hauptkategorie |
| z_attribute | Werttyp (count oder percentage) | count |
| cmap | Farbpalette für die Heatmap | YlGnBu |
| figsize | Größe der Abbildung (Format: Breite,Höhe) | 14,10 |
| annot | Zeige Werte in der Heatmap | True |
| fmt | Formatierung der Zahlen | .0f |

#### Zusammenfassungen
Es gibt drei Arten von Zusammenfassungen:

1. **Paraphrasenzusammenfassung** (`analysis_type: summary_paraphrase`)
2. **Begründungszusammenfassung** (`analysis_type: summary_reasoning`)
3. **Benutzerdefinierte Zusammenfassung** (`analysis_type: custom_summary`)

Zusätzlicher Parameter für alle Zusammenfassungen:
| Parameter | Beschreibung |
|-----------|--------------|
| prompt_template | Angepasster Prompt für die KI (optional, Standard-Prompts werden verwendet, wenn nicht angegeben) |

## Ausführung

1. **Skript starten:**
   ```
   python qca-aid-explorer.py
   ```

2. **Ausgabedateien:**
   - Alle Ausgaben werden im "output"-Ordner gespeichert
   - Netzwerkvisualisierungen: PDF und SVG
   - Heatmaps: PDF und PNG
   - Netzwerk- und Heatmap-Daten: Excel-Dateien
   - Zusammenfassungen: Textdateien

## Tipps und Tricks

### Analysen überspringen
Es gibt drei Möglichkeiten, bestimmte Analysen zu überspringen:

1. **Sheets umbenennen:** Fügen Sie einen Unterstrich am Anfang des Sheet-Namens hinzu (z.B. "_Netzwerk1")
2. **Sheets ausblenden:** Blenden Sie Sheets aus, die nicht verarbeitet werden sollen
3. **Mehrere Konfigurationsdateien:** Erstellen Sie verschiedene Konfigurationsdateien für unterschiedliche Analysesätze

### Keyword-Harmonisierung
Die Keyword-Harmonisierung erkennt ähnliche Schlüsselwörter und vereinheitlicht sie:

- Aktivieren Sie diese Funktion mit `clean_keywords: True` im Basis-Sheet
- Passen Sie die Erkennungsschwelle mit `similarity_threshold` an (0.7 bedeutet 70% Ähnlichkeit)
- Die harmonisierten Begriffe werden in der Konsole ausgegeben

### Anpassung der Visualisierungen
- Experimentieren Sie mit verschiedenen Parametern für Netzwerke und Heatmaps
- Für Netzwerke: Erhöhen Sie `iterations` für bessere Layouts, passen Sie `gravity` für kompaktere/lockerere Netzwerke an
- Für Heatmaps: Probieren Sie verschiedene Farbpaletten mit `cmap` (z.B. "YlOrRd", "Blues", "viridis")

### Erstellen eigener Prompts
Bei benutzerdefinierten Zusammenfassungen können Sie Ihre eigenen Prompts erstellen:
- Verwenden Sie `{text}` als Platzhalter für die zu analysierenden Texte
- Verwenden Sie `{filters}` als Platzhalter für die aktiven Filter
- Strukturieren Sie den Prompt klar, um spezifischere Analysen zu erhalten

## Fehlerbehandlung

### Häufige Probleme

1. **API-Schlüssel nicht gefunden:**
   - Stellen Sie sicher, dass die Datei `.environ.env` im Benutzerverzeichnis existiert
   - Überprüfen Sie, ob die API-Schlüssel korrekt sind

2. **Excel-Datei nicht gefunden:**
   - Überprüfen Sie, ob der Pfad zur Konfigurationsdatei korrekt ist
   - Stellen Sie sicher, dass die Analyse-Datei im "output"-Ordner liegt

3. **Fehler bei der Keyword-Harmonisierung:**
   - Reduzieren Sie den Wert für `similarity_threshold`
   - Deaktivieren Sie die Harmonisierung mit `clean_keywords: False`

4. **Layoutprobleme bei Netzwerken:**
   - Versuchen Sie, `iterations` zu erhöhen (z.B. auf 200)
   - Passen Sie `gravity` und `scaling` an
   - Bei sehr großen Netzwerken: Filtern Sie die Daten stärker

## Beispiel für eine Konfigurationsdatei

### Excel-Format

#### Basis-Sheet
| Parameter | Wert |
|-----------|------|
| provider | openai |
| model | gpt-4o-mini |
| temperature | 0.7 |
| script_dir | C:/MeinProjekt |
| output_dir | output |
| explore_file | QCA-Analyse_2025.xlsx |
| clean_keywords | True |
| similarity_threshold | 0.7 |

### Netzwerkanalyse-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Hauptkategorie | Herausforderungen |
| filter_Dokument | Interview_01 |
| analysis_type | netzwerk |
| node_size_factor | 1.2 |
| gravity | 0.08 |

### Heatmap-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Hauptkategorie |  |
| analysis_type | heatmap |
| x_attribute | Dokument |
| y_attribute | Hauptkategorie |
| cmap | YlOrRd |

### Zusammenfassung-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Dokument | Interview_03 |
| analysis_type | summary_paraphrase |
| prompt_template | Bitte analysieren Sie die folgenden Texte und identifizieren Sie die Hauptthemen: {text} |

### JSON-Format

Die gleiche Konfiguration als JSON-Datei (`QCA-AID-Explorer-Config.json`):

```json
{
  "base_config": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "script_dir": "C:/MeinProjekt",
    "output_dir": "output",
    "explore_file": "QCA-Analyse_2025.xlsx",
    "clean_keywords": true,
    "similarity_threshold": 0.7
  },
  "analysis_configs": [
    {
      "name": "Netzwerkanalyse1",
      "filters": {
        "Hauptkategorie": "Herausforderungen",
        "Dokument": "Interview_01"
      },
      "params": {
        "active": true,
        "analysis_type": "netzwerk",
        "node_size_factor": 1.2,
        "gravity": 0.08
      }
    },
    {
      "name": "Heatmap1",
      "filters": {
        "Hauptkategorie": ""
      },
      "params": {
        "active": true,
        "analysis_type": "heatmap",
        "x_attribute": "Dokument",
        "y_attribute": "Hauptkategorie",
        "cmap": "YlOrRd"
      }
    },
    {
      "name": "Zusammenfassung1",
      "filters": {
        "Dokument": "Interview_03"
      },
      "params": {
        "active": true,
        "analysis_type": "summary_paraphrase",
        "prompt_template": "Bitte analysieren Sie die folgenden Texte und identifizieren Sie die Hauptthemen: {text}"
      }
    }
  ]
}
```

**Hinweise zum JSON-Format:**
- Die Datei muss UTF-8 kodiert sein
- Boolesche Werte werden als `true`/`false` geschrieben (nicht `True`/`False`)
- Leere Filter-Werte können als leerer String `""` angegeben werden
- Jede Analyse hat einen eindeutigen Namen im `name`-Feld
- Filter werden im `filters`-Objekt definiert
- Alle anderen Parameter kommen ins `params`-Objekt

## Modulreferenz (für Entwickler)

### QCA_AID_assets.explorer

**Modul:** `QCA_AID_assets.explorer`

**Hauptfunktion:** `main()`

Die Hauptfunktion orchestriert die gesamte Analyse-Pipeline:
1. Lädt die Konfiguration (Excel oder JSON)
2. Initialisiert den LLM Provider
3. Erstellt den QCAAnalyzer
4. Führt alle konfigurierten Analysen durch

**Verwendung:**
```python
import asyncio
from QCA_AID_assets.explorer import main

asyncio.run(main())
```

### QCA_AID_assets.analysis.qca_analyzer

**Modul:** `QCA_AID_assets.analysis.qca_analyzer`

**Klasse:** `QCAAnalyzer`

Die zentrale Klasse für alle Analysen. Sie bietet Methoden für:
- Datenfilterung
- Keyword-Harmonisierung
- Netzwerk-Visualisierung
- Heatmap-Erstellung
- Zusammenfassungen mit LLMs
- Sentiment-Analyse

**Wichtige Methoden:**
- `filter_data(filters)`: Filtert Daten nach Kriterien
- `harmonize_keywords(similarity_threshold)`: Vereinheitlicht ähnliche Keywords
- `create_network_graph(filtered_df, output_filename, params)`: Erstellt Netzwerk-Visualisierung
- `create_heatmap(filtered_df, output_filename, params)`: Erstellt Heatmap
- `create_custom_summary(filtered_df, prompt_template, output_filename, ...)`: Erstellt LLM-Zusammenfassung
- `create_sentiment_analysis(filtered_df, output_filename, params)`: Erstellt Sentiment-Analyse

**Verwendung:**
```python
from QCA_AID_assets.analysis.qca_analyzer import QCAAnalyzer

analyzer = QCAAnalyzer(excel_path, llm_provider, config)
filtered_df = analyzer.filter_data({"Hauptkategorie": "Herausforderungen"})
analyzer.create_network_graph(filtered_df, "netzwerk.pdf")
```

### QCA_AID_assets.utils.config.loader

**Modul:** `QCA_AID_assets.utils.config.loader`

**Klasse:** `ConfigLoader`

Lädt und verwaltet Konfigurationsdateien (Excel und JSON).

**Wichtige Methoden:**
- `load_config()`: Lädt die Konfiguration und gibt (base_config, analysis_configs) zurück
- `get_base_config()`: Gibt nur die Basis-Konfiguration zurück
- `get_analysis_configs()`: Gibt nur die Analyse-Konfigurationen zurück

**Verwendung:**
```python
from QCA_AID_assets.utils.config.loader import ConfigLoader

loader = ConfigLoader("QCA-AID-Explorer-Config.xlsx")
base_config, analysis_configs = loader.load_config()
```

### QCA_AID_assets.utils.llm.factory

**Modul:** `QCA_AID_assets.utils.llm.factory`

**Klasse:** `LLMProviderFactory`

Factory-Klasse zur Erstellung von LLM Providern.

**Wichtige Methoden:**
- `create_provider(provider_name, model, **kwargs)`: Erstellt einen LLM Provider

**Unterstützte Provider:**
- `openai`: OpenAI GPT-Modelle
- `mistral`: Mistral AI Modelle

**Verwendung:**
```python
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory

provider = LLMProviderFactory.create_provider(
    provider_name="openai",
    model="gpt-4o-mini",
    temperature=0.7
)
```

### QCA_AID_assets.utils.visualization.layout

**Modul:** `QCA_AID_assets.utils.visualization.layout`

**Funktion:** `create_forceatlas_like_layout(G, iterations, gravity, scaling)`

Erstellt ein ForceAtlas2-ähnliches Layout für Netzwerk-Visualisierungen.

**Parameter:**
- `G`: NetworkX Graph
- `iterations`: Anzahl der Layout-Iterationen (Standard: 100)
- `gravity`: Anziehungskraft zum Zentrum (Standard: 0.01)
- `scaling`: Skalierungsfaktor für Abstände (Standard: 10.0)

**Rückgabe:** Dictionary mit Knotenpositionen

**Verwendung:**
```python
from QCA_AID_assets.utils.visualization.layout import create_forceatlas_like_layout
import networkx as nx

G = nx.Graph()
# ... Graph aufbauen ...
pos = create_forceatlas_like_layout(G, iterations=150, gravity=0.05)
```

### QCA_AID_assets.utils.prompts

**Modul:** `QCA_AID_assets.utils.prompts`

**Funktion:** `get_default_prompts()`

Gibt Standard-Prompts für verschiedene Analysetypen zurück.

**Rückgabe:** Dictionary mit Prompt-Templates für:
- `summary_paraphrase`: Paraphrasierung von Textsegmenten
- `summary_reasoning`: Begründungsanalyse
- `sentiment_analysis`: Sentiment-Analyse

**Verwendung:**
```python
from QCA_AID_assets.utils.prompts import get_default_prompts

prompts = get_default_prompts()
paraphrase_prompt = prompts["summary_paraphrase"]
```

### QCA_AID_assets.utils.common

**Modul:** `QCA_AID_assets.utils.common`

**Funktion:** `create_filter_string(filters)`

Erstellt eine String-Repräsentation der Filter für Dateinamen.

**Parameter:**
- `filters`: Dictionary mit Filter-Parametern

**Rückgabe:** String-Repräsentation der Filter

**Verwendung:**
```python
from QCA_AID_assets.utils.common import create_filter_string

filters = {"Hauptkategorie": "Herausforderungen", "Dokument": "Interview_01"}
filter_str = create_filter_string(filters)
# Ergebnis: "Hauptkategorie_Herausforderungen_Dokument_Interview_01"
```

## Verwendung der Module (für Entwickler)

Wenn Sie den Code von QCA-AID Explorer in eigenen Projekten verwenden:

**Modulare Imports:**
```python
# Neue modulare Struktur
from QCA_AID_assets.analysis.qca_analyzer import QCAAnalyzer
from QCA_AID_assets.utils.config.loader import ConfigLoader
from QCA_AID_assets.utils.llm.factory import LLMProviderFactory
from QCA_AID_assets.utils.visualization.layout import create_forceatlas_like_layout
from QCA_AID_assets.utils.prompts import get_default_prompts
from QCA_AID_assets.utils.common import create_filter_string
```

**Vorteile für Entwickler:**
- Klare Import-Struktur
- Bessere IDE-Unterstützung (Autocomplete, Type Hints)
- Einfacheres Testen einzelner Komponenten
- Wiederverwendung einzelner Module in anderen Projekten

## Kontakt und Unterstützung

Bei Fragen oder Problemen wenden Sie sich bitte an den Entwickler oder öffnen Sie ein Issue im Quell-Repository.

---

**Hinweis:** Stellen Sie sicher, dass Sie über die erforderlichen Rechte für die verwendeten API-Schlüssel und Daten verfügen. QCA-AID Explorer ist ein Werkzeug zur qualitativen Analyse und übernimmt keine Verantwortung für die Ergebnisse oder den Inhalt der generierten Analysen.

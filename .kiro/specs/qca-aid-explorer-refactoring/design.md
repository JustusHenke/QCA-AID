# Design Document: QCA-AID-Explorer Refactoring

## Overview

Dieses Design-Dokument beschreibt die Architektur und Implementierungsdetails für das Refactoring von QCA-AID-Explorer. Das Hauptziel ist es, die monolithische `QCA-AID-Explorer.py` Datei in eine modulare Struktur innerhalb von `QCA_AID_assets` zu überführen, während ein minimales Launcher-Skript im Root-Verzeichnis verbleibt.

Das Refactoring folgt dem Prinzip der Separation of Concerns und organisiert den Code in logische Module basierend auf ihrer Funktionalität.

## Architecture

### Aktuelle Struktur

```
QCA-AID-Explorer.py (2503 Zeilen)
├── Imports
├── LLMResponse Klasse
├── ConfigLoader Klasse
├── QCAAnalyzer Klasse
│   ├── __init__
│   ├── filter_data
│   ├── harmonize_keywords
│   ├── create_network_graph
│   ├── create_heatmap
│   ├── create_custom_summary
│   ├── create_sentiment_analysis
│   └── _create_keyword_bubble_chart
├── create_forceatlas_like_layout Funktion
├── create_filter_string Funktion
├── get_default_prompts Funktion
└── main Funktion
```

### Ziel-Struktur

```
QCA-AID.py (unverändert)
└── Ruft QCA_AID_assets.main.main() auf (QCA-AID Hauptfunktion)

QCA-AID-Explorer.py (< 50 Zeilen, Launcher)
└── Ruft QCA_AID_assets.explorer.main() auf (Explorer Hauptfunktion)

QCA_AID_assets/
├── main.py (QCA-AID Hauptfunktion - UNVERÄNDERT)
├── explorer.py (NEU: QCA-AID-Explorer Hauptfunktion)
├── analysis/
│   ├── __init__.py
│   ├── qca_analyzer.py (NEU: QCAAnalyzer Klasse für Explorer)
│   └── ... (bestehende QCA-AID Module - UNVERÄNDERT)
├── utils/
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py (ConfigLoader - wird erweitert, nicht geändert)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── response.py (LLMResponse - wird erweitert, nicht geändert)
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── layout.py (NEU: create_forceatlas_like_layout)
│   ├── prompts.py (NEU: get_default_prompts)
│   └── common.py (create_filter_string wird hinzugefügt)
└── ... (bestehende QCA-AID Module - UNVERÄNDERT)
```

**Wichtig:** Alle mit "UNVERÄNDERT" markierten Module bleiben vollständig kompatibel mit QCA-AID.

## Components and Interfaces

### 1. Launcher-Skript (`QCA-AID-Explorer.py`)

**Verantwortlichkeit:** Minimaler Entry Point für die Explorer-Anwendung

**Schnittstelle:**
```python
# QCA-AID-Explorer.py
import asyncio
import os
from QCA_AID_assets.explorer import main

if __name__ == "__main__":
    try:
        # Windows-spezifische Event Loop Policy setzen
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Hauptprogramm ausführen
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet")
    except Exception as e:
        print(f"Fehler im Hauptprogramm: {str(e)}")
        raise
```

**Hinweis:** Der Import erfolgt aus `QCA_AID_assets.explorer`, NICHT aus `QCA_AID_assets.main` (das ist für QCA-AID reserviert).

### 2. Explorer Main Modul (`QCA_AID_assets/explorer.py`)

**Verantwortlichkeit:** Orchestrierung der Explorer Analyse-Pipeline

**Schnittstelle:**
```python
async def main() -> None:
    """
    Hauptfunktion für QCA-AID Explorer.
    
    Lädt Konfiguration, initialisiert Analyzer und führt alle konfigurierten
    Analysen durch.
    """
```

**Abhängigkeiten:**
- `QCA_AID_assets.utils.config.loader.ConfigLoader`
- `QCA_AID_assets.analysis.qca_analyzer.QCAAnalyzer`
- `QCA_AID_assets.utils.llm.factory.LLMProviderFactory`
- `QCA_AID_assets.utils.prompts.get_default_prompts`
- `QCA_AID_assets.utils.common.create_filter_string`

**Hinweis:** Dies ist eine NEUE Datei, die die main-Funktion aus QCA-AID-Explorer.py enthält. Die bestehende `QCA_AID_assets/main.py` (für QCA-AID) bleibt unverändert.

### 3. QCAAnalyzer Modul (`QCA_AID_assets/analysis/qca_analyzer.py`)

**Verantwortlichkeit:** Kern-Analyse-Logik für QCA-Daten

**Schnittstelle:**
```python
class QCAAnalyzer:
    def __init__(self, excel_path: str, llm_provider: LLMProvider, config: Dict[str, Any])
    def filter_data(self, filters: Dict[str, str]) -> pd.DataFrame
    def harmonize_keywords(self, similarity_threshold: float = 0.60) -> pd.DataFrame
    def needs_keyword_harmonization(self, analysis_type: str, params: Dict[str, Any] = None) -> bool
    def perform_selective_harmonization(self, analysis_configs: List[Dict[str, Any]], 
                                       clean_keywords: bool, 
                                       similarity_threshold: float) -> bool
    def validate_keyword_mapping(self) -> None
    def create_network_graph(self, filtered_df: pd.DataFrame, output_filename: str, 
                            params: Dict[str, Any] = None) -> None
    def create_heatmap(self, filtered_df: pd.DataFrame, output_filename: str, 
                      params: Dict[str, Any] = None) -> None
    async def create_custom_summary(self, filtered_df: pd.DataFrame, prompt_template: str, 
                                   output_filename: str, model: str, temperature: float = 0.7,
                                   filters: Dict[str, str] = None, 
                                   params: Dict[str, Any] = None) -> Optional[str]
    async def create_sentiment_analysis(self, filtered_df: pd.DataFrame, output_filename: str, 
                                       params: Dict[str, Any] = None) -> Optional[pd.DataFrame]
    def _create_keyword_bubble_chart(self, keyword_data: List[Dict[str, Any]], 
                                    output_filename: str, color_mapping: Dict[str, str],
                                    params: Dict[str, Any] = None) -> None
```

### 4. ConfigLoader Erweiterung (`QCA_AID_assets/utils/config/loader.py`)

**Verantwortlichkeit:** Laden und Verwalten von Konfigurationsdaten

Die Klasse `ConfigLoader` existiert bereits in diesem Modul. Sie wird aus `QCA-AID-Explorer.py` hierher verschoben und mit der bestehenden Funktionalität zusammengeführt.

### 5. LLMResponse Erweiterung (`QCA_AID_assets/utils/llm/response.py`)

**Verantwortlichkeit:** Wrapper für LLM-Provider-Responses

Die Klasse `LLMResponse` existiert bereits in diesem Modul. Sie wird aus `QCA-AID-Explorer.py` hierher verschoben und mit der bestehenden Funktionalität zusammengeführt.

### 6. Visualization Utils (`QCA_AID_assets/utils/visualization/layout.py`)

**Verantwortlichkeit:** Layout-Algorithmen für Visualisierungen

**Schnittstelle:**
```python
def create_forceatlas_like_layout(G: nx.Graph, iterations: int = 100, 
                                 gravity: float = 0.01, 
                                 scaling: float = 10.0) -> Dict[Any, Tuple[float, float]]:
    """
    Erzeugt ein ForceAtlas2-ähnliches Layout mit NetworkX und scikit-learn.
    
    Args:
        G: NetworkX Graph
        iterations: Anzahl Iterationen
        gravity: Stärke der Anziehung zum Zentrum
        scaling: Skalierungsfaktor für Knotenabstände
        
    Returns:
        Dictionary mit Knotenpositionen
    """
```

### 7. Prompts Utils (`QCA_AID_assets/utils/prompts.py`)

**Verantwortlichkeit:** Standard-Prompts für verschiedene Analysetypen

**Schnittstelle:**
```python
def get_default_prompts() -> Dict[str, str]:
    """
    Gibt die Standard-Prompts für verschiedene Analysetypen zurück.
    
    Returns:
        Dictionary mit Prompt-Templates für verschiedene Analysetypen
    """
```

### 8. Common Utils Erweiterung (`QCA_AID_assets/utils/common.py`)

**Verantwortlichkeit:** Allgemeine Hilfsfunktionen

Die Funktion `create_filter_string` wird zu diesem bereits existierenden Modul hinzugefügt.

**Schnittstelle:**
```python
def create_filter_string(filters: Dict[str, str]) -> str:
    """
    Erstellt eine String-Repräsentation der Filter für Dateinamen.
    
    Args:
        filters: Dictionary mit Filter-Parametern
        
    Returns:
        String-Repräsentation der Filter
    """
```

## Data Models

### Keine neuen Datenmodelle

Das Refactoring führt keine neuen Datenmodelle ein. Alle bestehenden Datenstrukturen bleiben unverändert:

- `pd.DataFrame` für Kodierungsdaten
- `Dict[str, Any]` für Konfigurationen
- `nx.Graph` für Netzwerk-Visualisierungen

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

Nach Durchsicht aller Prework-Analysen wurden folgende Redundanzen identifiziert:

1. **Strukturelle Anforderungen für Klassenpositionen (2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3)**: Diese können in eine einzige Property zusammengefasst werden, die prüft, ob alle Klassen und Funktionen an den richtigen Orten sind.

2. **Import-Anforderungen (5.1, 5.2, 5.3, 5.4, 5.5)**: Diese können in zwei Properties zusammengefasst werden: eine für die Korrektheit der Imports und eine für die erfolgreiche Ausführung.

3. **Funktionalitätserhaltung (1.4, 3.5, 6.1)**: Diese sind redundant und können in eine einzige Property zusammengefasst werden.

4. **Konfigurationsverarbeitung (4.4, 4.5, 6.2, 6.3, 6.4, 6.5)**: Diese können in eine Property zusammengefasst werden, die prüft, ob alle Konfigurationen korrekt verarbeitet werden.

### Property 1: Launcher-Skript ist minimal

*For any* refactored system, the launcher script should contain no more than 50 lines of code and should only import and call the main function from QCA_AID_assets.main

**Validates: Requirements 1.1, 1.2**

### Property 2: Keine Code-Duplikate

*For any* class or function in the codebase, it should exist in exactly one location (either in the old monolithic file or in the new modular structure, but not both)

**Validates: Requirements 1.3**

### Property 3: Alle Module sind an den richtigen Orten

*For any* class or function that was moved, it should exist in its designated module location as specified in the requirements

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3**

### Property 4: __init__.py Dateien exportieren korrekt

*For any* module that contains classes or functions, the corresponding __init__.py file should export those classes or functions

**Validates: Requirements 2.5**

### Property 5: Imports sind korrekt

*For any* module in the refactored system, all imports should be absolute (not relative) and should resolve without errors

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

### Property 6: Funktionalität bleibt erhalten

*For any* valid input configuration and data file, the refactored system should produce the same output files as the original system

**Validates: Requirements 1.4, 3.5, 6.1**

### Property 7: Alle Konfigurationen werden korrekt verarbeitet

*For any* valid configuration (XLSX or JSON), the system should load it correctly and execute all configured analyses (Netzwerk, Heatmap, Summary, Sentiment) with correct filtering and keyword harmonization

**Validates: Requirements 4.4, 4.5, 6.2, 6.3, 6.4, 6.5**

### Property 8: Docstrings sind vorhanden

*For any* public class or function in the new modules, it should have a docstring that describes its purpose and parameters

**Validates: Requirements 7.2**

## Error Handling

### Fehlerbehandlung bleibt unverändert

Das Refactoring ändert die Fehlerbehandlung nicht. Alle bestehenden try-except-Blöcke werden beibehalten:

1. **Import-Fehler**: Werden im Launcher-Skript abgefangen
2. **Konfigurations-Fehler**: Werden in ConfigLoader behandelt
3. **LLM-Fehler**: Werden in den Analyse-Methoden behandelt
4. **Datei-I/O-Fehler**: Werden in den entsprechenden Methoden behandelt

### Neue Fehlerbehandlung

Keine neuen Fehlerbehandlungsmechanismen sind erforderlich.

## Testing Strategy

### Unit Testing

Unit Tests werden für folgende Komponenten erstellt:

1. **Launcher-Skript**: Test, dass es die main-Funktion korrekt aufruft
2. **Module-Struktur**: Tests, dass alle Klassen und Funktionen an den richtigen Orten sind
3. **Imports**: Tests, dass alle Imports korrekt funktionieren
4. **Docstrings**: Tests, dass alle öffentlichen Klassen und Funktionen Docstrings haben

### Property-Based Testing

Property-Based Tests werden mit **Hypothesis** (Python) erstellt. Jeder Test sollte mindestens 100 Iterationen durchlaufen.

1. **Property 1: Launcher-Skript ist minimal**
   - Generator: Keine (statische Prüfung)
   - Test: Zähle Zeilen in QCA-AID-Explorer.py
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 1: Launcher-Skript ist minimal**`

2. **Property 2: Keine Code-Duplikate**
   - Generator: Keine (statische Code-Analyse)
   - Test: Suche nach duplizierten Klassen/Funktionen
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 2: Keine Code-Duplikate**`

3. **Property 3: Alle Module sind an den richtigen Orten**
   - Generator: Keine (statische Prüfung)
   - Test: Prüfe Existenz aller Klassen/Funktionen in den richtigen Modulen
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 3: Alle Module sind an den richtigen Orten**`

4. **Property 4: __init__.py Dateien exportieren korrekt**
   - Generator: Keine (statische Prüfung)
   - Test: Prüfe, ob alle Klassen/Funktionen in __init__.py exportiert werden
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 4: __init__.py Dateien exportieren korrekt**`

5. **Property 5: Imports sind korrekt**
   - Generator: Keine (statische Prüfung)
   - Test: Versuche, alle Module zu importieren
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 5: Imports sind korrekt**`

6. **Property 6: Funktionalität bleibt erhalten**
   - Generator: Generiere zufällige Konfigurationen und Testdaten
   - Test: Vergleiche Ausgaben vor und nach Refactoring
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 6: Funktionalität bleibt erhalten**`

7. **Property 7: Alle Konfigurationen werden korrekt verarbeitet**
   - Generator: Generiere zufällige Konfigurationen (XLSX und JSON)
   - Test: Prüfe, ob alle Analysen durchgeführt werden
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 7: Alle Konfigurationen werden korrekt verarbeitet**`

8. **Property 8: Docstrings sind vorhanden**
   - Generator: Keine (statische Prüfung)
   - Test: Prüfe, ob alle öffentlichen Klassen/Funktionen Docstrings haben
   - Tag: `**Feature: qca-aid-explorer-refactoring, Property 8: Docstrings sind vorhanden**`

### Integration Testing

Integration Tests werden durchgeführt, um sicherzustellen, dass:

1. Das Launcher-Skript die main-Funktion korrekt aufruft
2. Alle Module korrekt zusammenarbeiten
3. Die gesamte Analyse-Pipeline funktioniert

### Manuelle Tests

Manuelle Tests werden durchgeführt, um sicherzustellen, dass:

1. Die Dokumentation klar und verständlich ist
2. Die neue Struktur für Entwickler nachvollziehbar ist
3. Alle Ausgabedateien korrekt erzeugt werden

## Migration Strategy

### Schritt-für-Schritt Migration

1. **Backup erstellen**: Erstelle eine Kopie der aktuellen `QCA-AID-Explorer.py`
2. **Module erstellen**: Erstelle alle neuen Module und Verzeichnisse
3. **Code verschieben**: Verschiebe Klassen und Funktionen in die neuen Module
4. **Imports aktualisieren**: Aktualisiere alle Import-Statements
5. **Launcher erstellen**: Erstelle das minimale Launcher-Skript
6. **Tests durchführen**: Führe alle Tests durch
7. **Dokumentation aktualisieren**: Aktualisiere README und Docstrings
8. **Alte Datei entfernen**: Entferne die alte monolithische Datei (optional: als Backup behalten)

### Isolation von QCA-AID und QCA-AID-Explorer

**Wichtig:** Das Refactoring von QCA-AID-Explorer betrifft NUR die Explorer-Funktionalität und hat KEINE Auswirkungen auf QCA-AID:

1. **Separate Launcher-Skripte**: 
   - `QCA-AID.py` bleibt unverändert und ruft `QCA_AID_assets.main.main()` auf
   - `QCA-AID-Explorer.py` wird refaktoriert und ruft eine neue `QCA_AID_assets.explorer.main()` auf

2. **Separate Module**:
   - QCA-AID verwendet: `QCA_AID_assets.analysis.*`, `QCA_AID_assets.preprocessing.*`, etc.
   - QCA-AID-Explorer verwendet: `QCA_AID_assets.analysis.qca_analyzer` (neu), `QCA_AID_assets.utils.*`

3. **Gemeinsame Utils bleiben unverändert**:
   - `QCA_AID_assets.utils.config.*` wird von beiden verwendet
   - `QCA_AID_assets.utils.llm.*` wird von beiden verwendet
   - Diese Module werden nur erweitert, nicht geändert

4. **Keine Breaking Changes für QCA-AID**:
   - Alle bestehenden Imports in QCA-AID bleiben gültig
   - Keine Änderungen an bestehenden Klassen oder Funktionen, die von QCA-AID verwendet werden
   - Nur neue Module werden hinzugefügt

5. **Testabsicherung**:
   - Vor dem Refactoring: Führe QCA-AID Tests durch
   - Nach dem Refactoring: Führe QCA-AID Tests erneut durch
   - Stelle sicher, dass alle QCA-AID Tests weiterhin bestehen

### Rückwärtskompatibilität

Das Refactoring ist nicht rückwärtskompatibel auf Code-Ebene für QCA-AID-Explorer (Imports ändern sich), aber funktional identisch. Benutzer müssen keine Änderungen an ihren Konfigurationsdateien vornehmen.

**QCA-AID bleibt vollständig kompatibel** - keine Änderungen erforderlich.

## Performance Considerations

Das Refactoring sollte keine Performance-Auswirkungen haben, da:

1. Keine Algorithmen geändert werden
2. Nur die Code-Organisation geändert wird
3. Python-Imports zur Laufzeit gecacht werden

## Security Considerations

Das Refactoring hat keine Auswirkungen auf die Sicherheit, da:

1. Keine neuen externen Abhängigkeiten hinzugefügt werden
2. Keine Änderungen an der Datenverarbeitung vorgenommen werden
3. Keine neuen Netzwerk-Verbindungen erstellt werden

## Future Enhancements

Nach dem Refactoring können folgende Verbesserungen einfacher umgesetzt werden:

1. **Bessere Testabdeckung**: Durch die modulare Struktur können einzelne Komponenten isoliert getestet werden
2. **Erweiterbarkeit**: Neue Analysetypen können als separate Module hinzugefügt werden
3. **Wiederverwendbarkeit**: Einzelne Module können in anderen Projekten wiederverwendet werden
4. **Wartbarkeit**: Bugs können schneller gefunden und behoben werden

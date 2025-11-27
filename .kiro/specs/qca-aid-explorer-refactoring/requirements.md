# Requirements Document

## Introduction

Dieses Dokument beschreibt die Anforderungen für das Refactoring von QCA-AID-Explorer. Das Ziel ist es, alle Funktionen aus der monolithischen `QCA-AID-Explorer.py` Datei in die bestehende `QCA_AID_assets` Modulstruktur zu übertragen und nur ein minimales Launcher-Skript im Root-Verzeichnis zu belassen.

## Glossary

- **QCA-AID-Explorer**: Das Tool zur Analyse von qualitativen Kodierungsdaten
- **Launcher-Skript**: Eine minimale Python-Datei im Root-Verzeichnis, die nur die Hauptfunktion aufruft
- **QCA_AID_assets**: Das Modul-Verzeichnis, das alle Funktionalitäten enthält
- **Monolithische Datei**: Eine einzelne große Datei mit allen Funktionen (aktueller Zustand)
- **Modulare Struktur**: Eine organisierte Verzeichnisstruktur mit separaten Modulen für verschiedene Funktionalitäten

## Requirements

### Requirement 1

**User Story:** Als Entwickler möchte ich eine klare Trennung zwischen Launcher und Funktionalität, damit die Codebasis wartbarer und testbarer wird.

#### Acceptance Criteria

1. WHEN das Refactoring abgeschlossen ist THEN das System SHALL ein Launcher-Skript `QCA-AID-Explorer.py` im Root-Verzeichnis enthalten, das maximal 50 Zeilen Code umfasst
2. WHEN das Launcher-Skript ausgeführt wird THEN das System SHALL die Hauptfunktion aus dem `QCA_AID_assets` Modul importieren und aufrufen
3. WHEN alle Funktionen verschoben sind THEN das System SHALL keine Duplikate von Klassen oder Funktionen zwischen alter und neuer Struktur enthalten
4. WHEN das Refactoring abgeschlossen ist THEN das System SHALL die gleiche Funktionalität wie vorher bieten
5. WHEN das Launcher-Skript ausgeführt wird THEN das System SHALL die Windows-spezifische Event Loop Policy korrekt setzen

### Requirement 2

**User Story:** Als Entwickler möchte ich, dass alle Klassen in logische Module organisiert sind, damit ich den Code leichter verstehen und erweitern kann.

#### Acceptance Criteria

1. WHEN die Klasse `ConfigLoader` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.utils.config.loader` platzieren
2. WHEN die Klasse `QCAAnalyzer` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.analysis` platzieren
3. WHEN die Klasse `LLMResponse` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.utils.llm.response` platzieren
4. WHEN Hilfsfunktionen verschoben werden THEN das System SHALL sie in passende Utility-Module platzieren
5. WHEN alle Klassen verschoben sind THEN das System SHALL die `__init__.py` Dateien aktualisieren, um die Klassen exportierbar zu machen

### Requirement 3

**User Story:** Als Entwickler möchte ich, dass die Visualisierungsfunktionen in einem eigenen Modul organisiert sind, damit ich sie unabhängig testen und erweitern kann.

#### Acceptance Criteria

1. WHEN die Funktion `create_network_graph` verschoben wird THEN das System SHALL sie als Methode der `QCAAnalyzer` Klasse im `analysis` Modul behalten
2. WHEN die Funktion `create_heatmap` verschoben wird THEN das System SHALL sie als Methode der `QCAAnalyzer` Klasse im `analysis` Modul behalten
3. WHEN die Funktion `create_forceatlas_like_layout` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.utils.visualization` als eigenständige Funktion platzieren
4. WHEN die Funktion `_create_keyword_bubble_chart` verschoben wird THEN das System SHALL sie als Methode der `QCAAnalyzer` Klasse im `analysis` Modul behalten
5. WHEN Visualisierungsfunktionen aufgerufen werden THEN das System SHALL die gleichen Ausgabedateien wie vorher erzeugen

### Requirement 4

**User Story:** Als Entwickler möchte ich, dass die Hauptfunktion und Hilfsfunktionen in einem eigenen Modul organisiert sind, damit die Programmlogik klar strukturiert ist.

#### Acceptance Criteria

1. WHEN die `main` Funktion verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.main` platzieren
2. WHEN die Funktion `get_default_prompts` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.utils.prompts` platzieren
3. WHEN die Funktion `create_filter_string` verschoben wird THEN das System SHALL sie im Modul `QCA_AID_assets.utils.common` platzieren
4. WHEN die Hauptfunktion aufgerufen wird THEN das System SHALL alle Konfigurationen korrekt laden und verarbeiten
5. WHEN die Hauptfunktion ausgeführt wird THEN das System SHALL alle konfigurierten Analysen durchführen

### Requirement 5

**User Story:** Als Entwickler möchte ich, dass alle Imports korrekt aktualisiert werden, damit das refaktorierte System ohne Fehler läuft.

#### Acceptance Criteria

1. WHEN Module verschoben werden THEN das System SHALL alle relativen Imports in absolute Imports umwandeln
2. WHEN Klassen verschoben werden THEN das System SHALL alle Import-Statements in anderen Modulen aktualisieren
3. WHEN das Refactoring abgeschlossen ist THEN das System SHALL keine Import-Fehler beim Ausführen produzieren
4. WHEN externe Abhängigkeiten verwendet werden THEN das System SHALL diese am Anfang jedes Moduls importieren
5. WHEN das System gestartet wird THEN das System SHALL alle benötigten Module erfolgreich laden

### Requirement 6

**User Story:** Als Entwickler möchte ich, dass die bestehende Funktionalität vollständig erhalten bleibt, damit keine Features verloren gehen.

#### Acceptance Criteria

1. WHEN das refaktorierte System ausgeführt wird THEN das System SHALL die gleichen Ausgabedateien wie vorher erzeugen
2. WHEN Konfigurationsdateien geladen werden THEN das System SHALL sowohl XLSX als auch JSON Formate unterstützen
3. WHEN Analysen durchgeführt werden THEN das System SHALL alle konfigurierten Analysetypen unterstützen (Netzwerk, Heatmap, Summary, Sentiment)
4. WHEN Keyword-Harmonisierung aktiviert ist THEN das System SHALL diese korrekt durchführen
5. WHEN Filter angewendet werden THEN das System SHALL die Daten korrekt filtern und verarbeiten

### Requirement 7

**User Story:** Als Entwickler möchte ich, dass die Dokumentation aktualisiert wird, damit andere Entwickler die neue Struktur verstehen.

#### Acceptance Criteria

1. WHEN das Refactoring abgeschlossen ist THEN das System SHALL eine aktualisierte README-Datei für QCA-AID-Explorer enthalten
2. WHEN neue Module erstellt werden THEN das System SHALL Docstrings für alle öffentlichen Klassen und Funktionen enthalten
3. WHEN die Struktur geändert wird THEN das System SHALL die Änderungen in der Dokumentation beschreiben
4. WHEN Entwickler die Dokumentation lesen THEN das System SHALL klare Anweisungen zur Verwendung der neuen Struktur bieten
5. WHEN die Dokumentation aktualisiert wird THEN das System SHALL Beispiele für die Verwendung der neuen Module enthalten

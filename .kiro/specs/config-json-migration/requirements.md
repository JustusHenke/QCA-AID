# Requirements Document

## Introduction

Dieses Feature ermöglicht die Migration der QCA-AID-Explorer Konfiguration von Excel (XLSX) zu JSON. Es implementiert einen bidirektionalen Synchronisationsmechanismus, der Differenzen zwischen beiden Formaten erkennt und den Benutzer zur Auflösung von Konflikten auffordert. Nach erfolgreicher Migration wird JSON als primäres Konfigurationsformat verwendet.

## Glossary

- **ConfigLoader**: Die Klasse, die Konfigurationsdaten aus Dateien lädt
- **Basis-Konfiguration**: Globale Parameter im "Basis"-Sheet der Excel-Datei
- **Analyse-Konfiguration**: Spezifische Parameter für einzelne Auswertungen in separaten Sheets
- **Synchronisation**: Der Prozess des Abgleichs zwischen XLSX und JSON Dateien
- **Migration**: Die einmalige Umstellung von XLSX auf JSON als primäres Format

## Requirements

### Requirement 1

**User Story:** Als Entwickler möchte ich die Excel-Konfiguration in JSON umwandeln, so dass die Konfiguration maschinenlesbarer und versionskontrollfreundlicher wird.

#### Acceptance Criteria

1. WHEN das System startet THEN das System SHALL prüfen ob eine JSON-Konfigurationsdatei existiert
2. WHEN keine JSON-Konfiguration existiert THEN das System SHALL die Excel-Datei in JSON konvertieren
3. WHEN die JSON-Konvertierung erfolgt THEN das System SHALL die Struktur der Excel-Datei vollständig erhalten (Basis-Sheet und alle Analyse-Sheets)
4. WHEN die JSON-Datei erstellt wird THEN das System SHALL sie im gleichen Verzeichnis wie die Excel-Datei speichern mit dem Namen "QCA-AID-Explorer-Config.json"

### Requirement 2

**User Story:** Als Benutzer möchte ich bei Differenzen zwischen XLSX und JSON gefragt werden welche Version aktueller ist, so dass keine Daten verloren gehen.

#### Acceptance Criteria

1. WHEN sowohl XLSX als auch JSON existieren THEN das System SHALL beide Dateien auf inhaltliche Differenzen prüfen
2. WHEN Differenzen erkannt werden THEN das System SHALL in der Konsole anzeigen welche Parameter sich unterscheiden
3. WHEN Differenzen existieren THEN das System SHALL den Benutzer fragen welche Datei die aktuellere Version enthält
4. WHEN der Benutzer die XLSX als aktueller markiert THEN das System SHALL die JSON-Datei mit den XLSX-Inhalten aktualisieren
5. WHEN der Benutzer die JSON als aktueller markiert THEN das System SHALL die XLSX-Datei mit den JSON-Inhalten aktualisieren

### Requirement 3

**User Story:** Als Entwickler möchte ich dass der ConfigLoader JSON-Dateien lädt, so dass die Konfiguration schneller und effizienter verarbeitet wird.

#### Acceptance Criteria

1. WHEN eine JSON-Konfigurationsdatei existiert THEN der ConfigLoader SHALL diese anstelle der XLSX-Datei laden
2. WHEN der ConfigLoader JSON lädt THEN der ConfigLoader SHALL die gleiche Datenstruktur zurückgeben wie beim XLSX-Laden
3. WHEN die JSON-Datei fehlerhaft ist THEN der ConfigLoader SHALL auf die XLSX-Datei zurückfallen
4. WHEN auf XLSX zurückgefallen wird THEN das System SHALL eine Warnung in der Konsole ausgeben

### Requirement 4

**User Story:** Als Benutzer möchte ich dass die JSON-Struktur lesbar ist, so dass ich die Konfiguration manuell bearbeiten kann.

#### Acceptance Criteria

1. WHEN JSON geschrieben wird THEN das System SHALL die Datei mit Einrückung formatieren
2. WHEN JSON geschrieben wird THEN das System SHALL UTF-8 Encoding verwenden
3. WHEN die JSON-Struktur erstellt wird THEN das System SHALL eine klare Trennung zwischen Basis-Konfiguration und Analyse-Konfigurationen beibehalten

### Requirement 5

**User Story:** Als Entwickler möchte ich dass der Synchronisationsprozess vor dem Laden der Konfiguration abläuft, so dass immer die aktuellsten Daten verwendet werden.

#### Acceptance Criteria

1. WHEN das System die Konfiguration lädt THEN das System SHALL zuerst den Synchronisationsprozess durchführen
2. WHEN die Synchronisation abgeschlossen ist THEN das System SHALL die JSON-Datei für das Laden verwenden
3. WHEN keine Benutzereingabe erforderlich ist THEN der Synchronisationsprozess SHALL automatisch ablaufen
4. WHEN Benutzereingabe erforderlich ist THEN das System SHALL auf die Eingabe warten bevor es fortfährt

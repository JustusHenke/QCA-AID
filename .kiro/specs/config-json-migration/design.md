# Design Document: Config JSON Migration

## Overview

Dieses Design beschreibt die Implementierung eines bidirektionalen Synchronisationsmechanismus zwischen Excel (XLSX) und JSON Konfigurationsdateien für QCA-AID Explorer. Das System ermöglicht eine nahtlose Migration von XLSX zu JSON als primäres Konfigurationsformat, während Kompatibilität und Datensicherheit gewährleistet werden.

## Architecture

Das System besteht aus drei Hauptkomponenten:

1. **ConfigSynchronizer**: Verwaltet den Abgleich zwischen XLSX und JSON
2. **ConfigConverter**: Konvertiert zwischen XLSX und JSON Formaten
3. **ConfigLoader** (erweitert): Lädt Konfiguration aus JSON oder XLSX

Der Ablauf ist wie folgt:
```
Start → ConfigSynchronizer.sync() → Differenzen? → Ja → Benutzerabfrage → Aktualisierung
                                   ↓ Nein
                                   ↓
                        ConfigLoader.load_from_json() → Konfiguration geladen
```

## Components and Interfaces

### ConfigConverter

Verantwortlich für die Konvertierung zwischen Formaten.

```python
class ConfigConverter:
    @staticmethod
    def xlsx_to_json(xlsx_path: str) -> Dict[str, Any]:
        """Konvertiert XLSX zu JSON-Struktur"""
        pass
    
    @staticmethod
    def json_to_xlsx(json_data: Dict[str, Any], xlsx_path: str) -> None:
        """Schreibt JSON-Struktur in XLSX-Datei"""
        pass
    
    @staticmethod
    def save_json(json_data: Dict[str, Any], json_path: str) -> None:
        """Speichert JSON mit Formatierung"""
        pass
    
    @staticmethod
    def load_json(json_path: str) -> Dict[str, Any]:
        """Lädt JSON-Datei"""
        pass
```

### ConfigSynchronizer

Verwaltet den Synchronisationsprozess.

```python
class ConfigSynchronizer:
    def __init__(self, xlsx_path: str, json_path: str):
        self.xlsx_path = xlsx_path
        self.json_path = json_path
    
    def sync(self) -> str:
        """
        Führt Synchronisation durch
        Returns: Pfad zur zu verwendenden Datei ('json' oder 'xlsx')
        """
        pass
    
    def _detect_differences(self, xlsx_data: Dict, json_data: Dict) -> List[str]:
        """Erkennt Differenzen zwischen beiden Formaten"""
        pass
    
    def _prompt_user_choice(self, differences: List[str]) -> str:
        """Fragt Benutzer welche Version aktueller ist"""
        pass
    
    def _update_from_xlsx(self) -> None:
        """Aktualisiert JSON aus XLSX"""
        pass
    
    def _update_from_json(self) -> None:
        """Aktualisiert XLSX aus JSON"""
        pass
```

### ConfigLoader (erweitert)

Erweitert die bestehende Klasse um JSON-Support.

```python
class ConfigLoader:
    def __init__(self, config_path: str):
        """
        config_path kann .xlsx oder .json sein
        """
        self.config_path = config_path
        self.base_config = {}
        self.analysis_configs = []
        
        # Synchronisation vor dem Laden
        self._sync_configs()
        self._load_config()
    
    def _sync_configs(self) -> None:
        """Führt Synchronisation durch falls nötig"""
        pass
    
    def _load_config(self) -> None:
        """Lädt Config aus JSON oder XLSX"""
        pass
    
    def _load_from_json(self, json_path: str) -> None:
        """Lädt Konfiguration aus JSON"""
        pass
    
    def _load_from_xlsx(self) -> None:
        """Bestehende XLSX-Lade-Logik"""
        pass
```

## Data Models

### JSON-Struktur

```json
{
  "base_config": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "script_dir": "",
    "output_dir": "output",
    "explore_file": "example.xlsx",
    "clean_keywords": true,
    "similarity_threshold": 0.7
  },
  "analysis_configs": [
    {
      "name": "Netzwerk_Alle",
      "filters": {
        "Attribut_1": "value1"
      },
      "params": {
        "active": true,
        "analysis_type": "netzwerk",
        "layout": "forceatlas2"
      }
    }
  ]
}
```

### Differenz-Objekt

```python
@dataclass
class ConfigDifference:
    path: str  # z.B. "base_config.temperature" oder "analysis_configs[0].params.active"
    xlsx_value: Any
    json_value: Any
```

## Cor
rectness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

**Property Reflection:**

Nach Analyse der Prework wurden folgende Redundanzen identifiziert:
- Property 2.4 und 2.5 testen beide die Synchronisation, können zu einer umfassenderen Property kombiniert werden
- Property 4.1, 4.2 und 4.3 testen alle Aspekte der JSON-Formatierung, können kombiniert werden

**Property 1: Konvertierung erhält Datenstruktur (Round-Trip)**
*For any* Excel-Konfigurationsdatei mit Basis-Sheet und Analyse-Sheets, wenn diese zu JSON konvertiert und zurück zu Excel konvertiert wird, sollte die resultierende Struktur identisch zur ursprünglichen sein
**Validates: Requirements 1.3**

**Property 2: Differenzerkennung ist vollständig**
*For any* Paar von XLSX und JSON Konfigurationen mit unterschiedlichen Werten, sollte das System alle Differenzen korrekt identifizieren und in der Ausgabe auflisten
**Validates: Requirements 2.1, 2.2**

**Property 3: Synchronisation stellt Konsistenz her**
*For any* Paar von XLSX und JSON Konfigurationen, nach der Synchronisation (unabhängig von der gewählten Quelle) sollten beide Dateien identische Inhalte haben
**Validates: Requirements 2.4, 2.5**

**Property 4: JSON-Laden entspricht XLSX-Laden**
*For any* Konfigurationsdatei, das Laden über JSON sollte die gleiche Datenstruktur (base_config und analysis_configs) zurückgeben wie das Laden über XLSX
**Validates: Requirements 3.2**

**Property 5: JSON-Formatierung ist korrekt**
*For any* Konfigurationsdaten, die gespeicherte JSON-Datei sollte eingerückt sein, UTF-8 Encoding verwenden und die Keys "base_config" und "analysis_configs" auf oberster Ebene enthalten
**Validates: Requirements 4.1, 4.2, 4.3**

## Error Handling

### Fehlende Dateien
- Wenn XLSX fehlt aber JSON existiert: Warnung ausgeben, JSON verwenden
- Wenn beide fehlen: Klare Fehlermeldung mit Pfadangabe

### Korrupte Dateien
- JSON-Parse-Fehler: Auf XLSX zurückfallen, Warnung ausgeben
- XLSX-Lesefehler: Fehler werfen mit Details

### Synchronisationsfehler
- Schreibfehler bei Update: Fehler werfen, Original-Datei nicht überschreiben
- Benutzer-Abbruch: Synchronisation abbrechen, bestehende Dateien unverändert lassen

### Validierungsfehler
- Fehlende erforderliche Keys in JSON: Auf XLSX zurückfallen
- Ungültige Datentypen: Fehler mit Details welcher Parameter betroffen ist

## Testing Strategy

### Unit Tests

Unit Tests werden für spezifische Szenarien und Edge Cases geschrieben:

- Test für Dateierstellung am korrekten Pfad (1.4)
- Test für JSON-Präferenz wenn beide Dateien existieren (3.1)
- Test für Fallback bei korrupter JSON (3.3)
- Test für automatische Synchronisation ohne Differenzen (5.3)

### Property-Based Tests

Property-Based Tests werden mit **Hypothesis** (Python PBT-Framework) implementiert. Jeder Test sollte mindestens 100 Iterationen durchlaufen.

**Generators:**
- `config_dict()`: Generiert zufällige aber valide Konfigurationsstrukturen
- `base_config()`: Generiert Basis-Konfigurationen mit verschiedenen Parametern
- `analysis_config()`: Generiert Analyse-Konfigurationen mit Filtern und Parametern
- `excel_structure()`: Generiert Excel-kompatible Datenstrukturen

**Property Tests:**
1. Round-Trip Test für XLSX→JSON→XLSX Konvertierung (Property 1)
2. Differenzerkennung für modifizierte Konfigurationen (Property 2)
3. Synchronisations-Konsistenz für beliebige Config-Paare (Property 3)
4. Lade-Äquivalenz zwischen JSON und XLSX (Property 4)
5. JSON-Format-Validierung für alle Konfigurationen (Property 5)

Jeder Property-Test wird mit einem Kommentar markiert:
```python
# Feature: config-json-migration, Property 1: Konvertierung erhält Datenstruktur (Round-Trip)
```

## Implementation Notes

### Pfad-Handling
- Verwende `pathlib.Path` für plattformunabhängige Pfade
- JSON-Pfad wird aus XLSX-Pfad durch Ersetzen der Extension abgeleitet

### Excel-Handling
- Verwende `openpyxl` für XLSX-Schreiboperationen (pandas read_excel unterstützt kein Schreiben)
- Behalte Sheet-Reihenfolge bei

### Benutzerinteraktion
- Verwende `input()` für Konsolenabfragen
- Klare Optionen: "1" für XLSX, "2" für JSON, "q" für Abbruch
- Zeige Differenzen in lesbarem Format

### Backward Compatibility
- Bestehender Code der nur XLSX-Pfad übergibt sollte weiterhin funktionieren
- ConfigLoader erkennt automatisch ob JSON existiert und bevorzugt diese

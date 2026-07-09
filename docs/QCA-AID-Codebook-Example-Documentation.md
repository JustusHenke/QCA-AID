# QCA-AID Codebook JSON - Dokumentation

Diese Datei erklärt die Struktur und Verwendung der JSON-Konfigurationsdatei `QCA-AID-Codebook-Example.json`.

## Übersicht

Die JSON-Konfiguration ist eine Alternative zum Excel-Codebook und bietet die gleiche Funktionalität mit folgenden Vorteilen:
- Schnelleres Laden (ca. 10x schneller als Excel)
- Einfachere Versionskontrolle mit Git
- Bessere Lesbarkeit für technisch versierte Nutzer
- Automatische Synchronisation mit Excel-Dateien

## Hauptstruktur

Die JSON-Datei besteht aus vier Hauptabschnitten:

```json
{
  "forschungsfrage": "...",
  "kodierregeln": {...},
  "deduktive_kategorien": {...},
  "config": {...}
}
```

## 1. Forschungsfrage

**Typ:** String (erforderlich)

**Beschreibung:** Die zentrale Forschungsfrage Ihrer Untersuchung. Diese wird den KI-Codern als Kontext für die Kodierung bereitgestellt.

**Beispiel:**
```json
"forschungsfrage": "Wie gestaltet sich die digitale Transformation in deutschen Hochschulen und welche Herausforderungen und Chancen lassen sich dabei identifizieren?"
```

**Best Practices:**
- Formulieren Sie die Frage präzise und fokussiert
- Vermeiden Sie zu breite oder zu enge Fragestellungen
- Die Frage sollte zum Kategoriensystem passen

## 2. Kodierregeln

**Typ:** Object mit drei Arrays (erforderlich)

**Beschreibung:** Definiert die Regeln für die Kodierung auf drei Ebenen.

### Struktur:
```json
"kodierregeln": {
  "general": [...],    // Allgemeine Kodierregeln
  "format": [...],     // Formatierungsregeln
  "exclusion": [...]   // Ausschlusskriterien
}
```

### 2.1 General (Allgemeine Regeln)

**Typ:** Array von Strings

**Beschreibung:** Grundlegende Prinzipien für die Kodierung, die für alle Kategorien gelten.

**Beispiele:**
```json
"general": [
  "Kodiere nur explizite Aussagen, keine Interpretationen",
  "Berücksichtige den Kontext der Aussage",
  "Bei Unsicherheit dokumentiere die Gründe",
  "Kodiere vollständige Sinneinheiten"
]
```

### 2.2 Format (Formatierungsregeln)

**Typ:** Array von Strings

**Beschreibung:** Regeln für die formale Gestaltung der Kodierungen.

**Beispiele:**
```json
"format": [
  "Markiere relevante Textstellen vollständig",
  "Dokumentiere Begründung der Zuordnung",
  "Gib den Konfidenzwert an (0.0-1.0)"
]
```

### 2.3 Exclusion (Ausschlusskriterien)

**Typ:** Array von Strings

**Beschreibung:** Definiert, welche Textteile bei der Relevanzprüfung ausgeschlossen werden sollen.

**Beispiele:**
```json
"exclusion": [
  "Literaturverzeichnisse und Referenzlisten",
  "Tabellarische Datenaufstellungen ohne Interpretation",
  "Methodische Infoboxen",
  "Fußnoten mit reinen Quellenangaben"
]
```

## 3. Deduktive Kategorien

**Typ:** Object (erforderlich)

**Beschreibung:** Das Herzstück des Codebooks - definiert alle Hauptkategorien und deren Subkategorien.

### Struktur einer Kategorie:

```json
"Kategorienname": {
  "definition": "Beschreibung der Kategorie (min. 15 Wörter)",
  "rules": ["Regel 1", "Regel 2", ...],
  "examples": ["Beispiel 1", "Beispiel 2", ...],
  "subcategories": {
    "Subkategorie_1": "Beschreibung",
    "Subkategorie_2": "Beschreibung"
  }
}
```

### 3.1 Definition

**Typ:** String (erforderlich)

**Beschreibung:** Klare, präzise Definition der Kategorie. Sollte mindestens 15 Wörter umfassen.

**Best Practices:**
- Definieren Sie klar, was zur Kategorie gehört
- Grenzen Sie die Kategorie von ähnlichen Kategorien ab
- Verwenden Sie präzise Fachbegriffe

### 3.2 Rules

**Typ:** Array von Strings (optional, aber empfohlen)

**Beschreibung:** Spezifische Kodierregeln für diese Kategorie.

**Best Practices:**
- Formulieren Sie konkrete, operationalisierbare Regeln
- Geben Sie Hinweise für Grenzfälle
- Definieren Sie, was NICHT zur Kategorie gehört

### 3.3 Examples

**Typ:** Array von Strings (erforderlich, min. 2)

**Beschreibung:** Konkrete Textbeispiele, die zur Kategorie gehören.

**Best Practices:**
- Geben Sie mindestens 2-4 Beispiele
- Zeigen Sie verschiedene Facetten der Kategorie
- Verwenden Sie realistische, kontextnahe Beispiele
- Markieren Sie auch Grenzfälle

### 3.4 Subcategories

**Typ:** Object (erforderlich, min. 2)

**Beschreibung:** Unterkategorien mit jeweils einer kurzen Beschreibung.

**Format:**
```json
"subcategories": {
  "Unterkategorie_Name": "Kurze Beschreibung der Unterkategorie"
}
```

**Best Practices:**
- Verwenden Sie Unterstriche statt Leerzeichen in Namen
- Definieren Sie mindestens 2 Subkategorien
- Stellen Sie sicher, dass Subkategorien sich gegenseitig ausschließen
- Halten Sie Beschreibungen prägnant (1-2 Sätze)

## 4. Config (Konfigurationsparameter)

**Typ:** Object (erforderlich)

**Beschreibung:** Technische Einstellungen für die Analyse.

### 4.1 LLM-Einstellungen

```json
"MODEL_PROVIDER": "OpenAI",  // "OpenAI" oder "Mistral"
"MODEL_NAME": "gpt-4o-mini"  // z.B. "gpt-4o-mini", "gpt-4o", "mistral-large-latest"
```

**Optionen:**
- **OpenAI**: "gpt-4o-mini" (günstig, schnell), "gpt-4o" (leistungsstark), "gpt-3.5-turbo" (legacy)
- **Mistral**: "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"

### 4.2 Verzeichnisse

```json
"DATA_DIR": "input",      // Eingabeverzeichnis für Dokumente
"OUTPUT_DIR": "output"    // Ausgabeverzeichnis für Ergebnisse
```

**Hinweise:**
- Relative Pfade werden relativ zum Projektverzeichnis aufgelöst
- Absolute Pfade werden direkt verwendet
- Verzeichnisse werden automatisch erstellt, falls nicht vorhanden

### 4.3 Chunking-Parameter

```json
"CHUNK_SIZE": 1000,       // Größe der Textabschnitte in Zeichen
"CHUNK_OVERLAP": 40       // Überlappung zwischen Chunks in Zeichen
```

**Empfehlungen:**
- **Interviews**: CHUNK_SIZE: 1000, CHUNK_OVERLAP: 50
- **Längere Texte**: CHUNK_SIZE: 1500, CHUNK_OVERLAP: 100
- **Kurze Dokumente**: CHUNK_SIZE: 800, CHUNK_OVERLAP: 30

**Wichtig:** CHUNK_OVERLAP muss kleiner als CHUNK_SIZE sein!

### 4.4 Batch-Verarbeitung

```json
"BATCH_SIZE": 5  // Anzahl gleichzeitig zu verarbeitender Chunks (1-20)
```

**Empfehlungen:**
- **Standard**: 5-8 (gute Balance)
- **Schnell**: 10-15 (mehr Geschwindigkeit, weniger Präzision)
- **Präzise**: 3-4 (höhere Qualität, langsamer)

### 4.5 Kodierungsmodus

```json
"CODE_WITH_CONTEXT": true,        // Kontextuelle Kodierung aktivieren
"MULTIPLE_CODINGS": true,         // Mehrfachkodierung aktivieren
"MULTIPLE_CODING_THRESHOLD": 0.85 // Schwellwert für Mehrfachkodierung (0.0-1.0)
```

**CODE_WITH_CONTEXT:**
- `true`: Nutzt progressiven Dokumentkontext (empfohlen für komplexe Analysen)
- `false`: Jeder Chunk wird isoliert kodiert (schneller)

**MULTIPLE_CODINGS:**
- `true`: Erlaubt mehrere Kategorien pro Textstelle
- `false`: Nur eine Kategorie pro Textstelle

### 4.6 Analysemodus

```json
"ANALYSIS_MODE": "deductive"  // "full", "abductive", "deductive", "inductive", "grounded"
```

**Optionen:**
- **deductive**: Nur vordefinierte Kategorien verwenden
- **abductive**: Subkategorien können erweitert werden
- **inductive**: Neue Haupt- und Subkategorien können entstehen
- **full**: Vollständige induktive Erweiterung
- **grounded**: Grounded Theory Ansatz (schrittweise Kategorienbildung)

### 4.7 Review-Modus

```json
"REVIEW_MODE": "consensus"  // "auto", "manual", "consensus", "majority"
```

**Optionen:**
- **consensus**: Nur übereinstimmende Kodierungen (höchste Qualität)
- **majority**: Mehrheitsentscheidung bei mehreren Codern
- **manual**: Manuelle Kodierungen haben Vorrang
- **auto**: Automatische Kodierungen ohne Review

### 4.8 Attribute Labels

```json
"ATTRIBUTE_LABELS": {
  "attribut1": "Hochschultyp",
  "attribut2": "Position",
  "attribut3": "Fachbereich"
}
```

**Beschreibung:** Definiert, wie Attribute aus Dateinamen extrahiert werden.

**Beispiel:**
- Dateiname: `Universität_Professor_Informatik_Interview.txt`
- Wird extrahiert als:
  - attribut1: "Universität"
  - attribut2: "Professor"
  - attribut3: "Informatik"

**Hinweis:** Setzen Sie Werte auf `null`, wenn Sie weniger als 3 Attribute benötigen.

### 4.9 PDF-Export

```json
"EXPORT_ANNOTATED_PDFS": true,           // PDF-Annotation aktivieren
"PDF_ANNOTATION_FUZZY_THRESHOLD": 0.85   // Fuzzy-Matching Schwellwert (0.0-1.0)
```

**EXPORT_ANNOTATED_PDFS:**
- `true`: Erstellt annotierte PDFs mit Kodierungen
- `false`: Keine PDF-Annotation

### 4.10 Coder Settings

```json
"CODER_SETTINGS": [
  {
    "temperature": 0.3,
    "coder_id": "auto_1"
  },
  {
    "temperature": 0.5,
    "coder_id": "auto_2"
  }
]
```

**Beschreibung:** Definiert mehrere KI-Coder mit unterschiedlichen Temperaturen.

**Temperature:**
- **0.0-0.3**: Sehr konsistent, wenig kreativ (empfohlen für deduktive Kodierung)
- **0.4-0.6**: Ausgewogen (empfohlen für abduktive Kodierung)
- **0.7-1.0**: Kreativ, variabel (empfohlen für induktive Kodierung)

**Best Practices:**
- Verwenden Sie 2-3 Coder für Intercoder-Reliabilität
- Variieren Sie die Temperature leicht (z.B. 0.3 und 0.5)
- Geben Sie eindeutige coder_ids

### 4.11 Validation (Validierungsregeln)

```json
"VALIDATION": {
  "MIN_DEFINITION_WORDS": 15,
  "MIN_EXAMPLES": 2,
  "SIMILARITY_THRESHOLD": 0.7,
  "MIN_SUBCATEGORIES": 2,
  "MAX_NAME_LENGTH": 50,
  "MIN_NAME_LENGTH": 3,
  "ENGLISH_WORDS": {
    "research": true,
    "development": true
  },
  "MESSAGES": {
    "short_definition": "Definition zu kurz (min. {min_words} Wörter)",
    "few_examples": "Zu wenige Beispiele (min. {min_examples})"
  }
}
```

**Beschreibung:** Regeln zur Validierung des Kategoriensystems.

**Parameter:**
- **MIN_DEFINITION_WORDS**: Minimale Wortanzahl für Definitionen
- **MIN_EXAMPLES**: Minimale Anzahl von Beispielen
- **SIMILARITY_THRESHOLD**: Schwellwert für Ähnlichkeitsprüfung (0.0-1.0)
- **MIN_SUBCATEGORIES**: Minimale Anzahl von Subkategorien
- **MAX_NAME_LENGTH**: Maximale Länge von Kategorienamen
- **MIN_NAME_LENGTH**: Minimale Länge von Kategorienamen
- **ENGLISH_WORDS**: Erlaubte englische Begriffe in Kategorienamen
- **MESSAGES**: Fehlermeldungen (mit Platzhaltern)

## Datentypen

### Boolean-Werte

In JSON werden Boolean-Werte als `true` oder `false` geschrieben (ohne Anführungszeichen).

**Richtig:**
```json
"CODE_WITH_CONTEXT": true,
"EXPORT_ANNOTATED_PDFS": false
```

**Falsch:**
```json
"CODE_WITH_CONTEXT": "true",  // String statt Boolean
"EXPORT_ANNOTATED_PDFS": 1    // Number statt Boolean
```

### Numerische Werte

Zahlen werden ohne Anführungszeichen geschrieben.

**Richtig:**
```json
"CHUNK_SIZE": 1000,
"MULTIPLE_CODING_THRESHOLD": 0.85
```

**Falsch:**
```json
"CHUNK_SIZE": "1000",           // String statt Number
"MULTIPLE_CODING_THRESHOLD": "0.85"
```

### Null-Werte

Fehlende oder nicht gesetzte Werte werden als `null` dargestellt.

**Beispiel:**
```json
"ATTRIBUTE_LABELS": {
  "attribut1": "Hochschultyp",
  "attribut2": "Position",
  "attribut3": null  // Nicht verwendet
}
```

## Häufige Fehler

### 1. Fehlende Kommas

**Falsch:**
```json
{
  "CHUNK_SIZE": 1000
  "CHUNK_OVERLAP": 40  // Fehlendes Komma!
}
```

**Richtig:**
```json
{
  "CHUNK_SIZE": 1000,
  "CHUNK_OVERLAP": 40
}
```

### 2. Trailing Commas

**Falsch:**
```json
{
  "CHUNK_SIZE": 1000,
  "CHUNK_OVERLAP": 40,  // Komma am Ende!
}
```

**Richtig:**
```json
{
  "CHUNK_SIZE": 1000,
  "CHUNK_OVERLAP": 40
}
```

### 3. Falsche Anführungszeichen

**Falsch:**
```json
{
  'MODEL_PROVIDER': 'OpenAI'  // Einfache Anführungszeichen!
}
```

**Richtig:**
```json
{
  "MODEL_PROVIDER": "OpenAI"  // Doppelte Anführungszeichen
}
```

### 4. Kommentare in JSON

JSON unterstützt keine Kommentare! Verwenden Sie diese Dokumentationsdatei für Erklärungen.

**Falsch:**
```json
{
  // Dies ist ein Kommentar
  "CHUNK_SIZE": 1000
}
```

## Validierung

Sie können Ihre JSON-Datei online validieren:
- https://jsonlint.com/
- https://jsonformatter.org/

Oder mit Python:
```python
import json

with open('QCA-AID-Codebook.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print("✅ JSON ist gültig!")
```

## Migration von Excel zu JSON

QCA-AID konvertiert automatisch zwischen Excel und JSON. Siehe `MIGRATION_GUIDE.md` für Details.

## Weitere Ressourcen

- **README.md**: Allgemeine Dokumentation zu QCA-AID
- **MIGRATION_GUIDE.md**: Anleitung zur Migration zwischen Formaten
- **CHANGELOG.md**: Versionshistorie und Änderungen

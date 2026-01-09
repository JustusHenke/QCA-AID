# QCA-AID Nutzerhandbuch
## Qualitative Inhaltsanalyse mit KI-Unterstützung

![QCA-AID Banner](banner-qca-aid.png)

**Version:** 0.11.1  
**Zielgruppe:** Sozialwissenschaftler:innen mit Erfahrung in qualitativer Forschung  
**Autor:** Justus Henke, Institut für Hochschulforschung Halle-Wittenberg

---

## Inhaltsverzeichnis

1. [Einführung und Grundlagen](#1-einführung-und-grundlagen)
2. [Design-Prinzipien von QCA-AID](#2-design-prinzipien-von-qca-aid)
3. [Die vier Kodiermodi](#3-die-vier-kodiermodi)
4. [Rolle der KI in QCA-AID](#4-rolle-der-ki-in-qca-aid)
5. [Installation und Einrichtung](#5-installation-und-einrichtung)
6. [LLM-Anbieter und Modellauswahl](#6-llm-anbieter-und-modellauswahl)
7. [Konfigurationseinstellungen](#7-konfigurationseinstellungen)
8. [Codebook-Entwicklung und -Pflege](#8-codebook-entwicklung-und-pflege)
9. [Arbeiten mit der Webapp](#9-arbeiten-mit-der-webapp)
10. [Output-Sheets und Ergebnisinterpretation](#10-output-sheets-und-ergebnisinterpretation)
11. [Optimaler Kodiermodus nach Forschungszielen](#11-optimaler-kodiermodus-nach-forschungszielen)
12. [Best Practices und Qualitätssicherung](#12-best-practices-und-qualitätssicherung)
13. [Häufige Probleme und Lösungen](#13-häufige-probleme-und-lösungen)
14. [Anhang: Screenshots und Beispiele](#14-anhang-screenshots-und-beispiele)

---

## 1. Einführung und Grundlagen

### Was ist QCA-AID?

QCA-AID (Qualitative Content Analysis with AI-supported Discovery) ist ein innovatives Tool, das Mayrings Methode der deduktiven qualitativen Inhaltsanalyse mit induktiver Erweiterung durch KI-Unterstützung implementiert. Es kombiniert bewährte qualitative Forschungsmethoden mit modernen KI-Fähigkeiten.

**Wichtiger Hinweis:** QCA-AID ersetzt nicht die menschliche Analyse, sondern erweitert die Möglichkeiten für strukturierte Textanalysen und schafft mehr Zeit für Reflexion und Interpretation.

### Anwendungsmöglichkeiten

- **Skalierung:** Analyse größerer Dokumentenmengen als in herkömmlichen Verfahren
- **Qualitätssicherung:** Intercoder-Vergleiche mit KI-Codern zusätzlich zu menschlichen Codierern
- **Exploration:** Zusätzliche explorative Analysen ohne KI-Coder möglich
- **Effizienz:** Alternative zu kostenpflichtigen QDA-Programmen

### Grenzen und Risiken

- **Überkonfidenz:** Gefahr der unkritischen Übernahme automatisiert ermittelter Strukturen
- **Dokumentenanzahl:** Bei wenigen Dokumenten überwiegen Vorteile manueller Kodierung
- **Qualitätskontrolle:** Ergebnisse müssen stets manuell validiert werden

---

## 2. Design-Prinzipien von QCA-AID

### Methodische Fundierung

QCA-AID basiert auf etablierten Prinzipien der qualitativen Inhaltsanalyse:

1. **Regelgeleitetheit:** Systematische Anwendung expliziter Kodierregeln
2. **Theoriegeleitetheit:** Deduktive Kategorien basieren auf theoretischen Vorannahmen
3. **Induktive Offenheit:** Möglichkeit zur Erweiterung des Kategoriensystems
4. **Intersubjektivität:** Nachvollziehbare und überprüfbare Kodierungen

### Technische Architektur

- **Modularer Aufbau:** Getrennte Komponenten für verschiedene Funktionen
- **Flexibilität:** Unterstützung verschiedener LLM-Anbieter und Modelle
- **Skalierbarkeit:** Batch-Verarbeitung für große Datenmengen
- **Transparenz:** Vollständige Dokumentation aller Kodierentscheidungen

---

## 3. Die vier Kodiermodi

QCA-AID bietet vier verschiedene Analysemodi, die sich in ihrer Offenheit für neue Kategorien unterscheiden:

### 3.1 Deduktiver Modus (`deductive`)

**Prinzip:** Ausschließliche Verwendung vordefinierter Kategorien

**Anwendung:**
- Theorieprüfung mit feststehendem Kategoriensystem
- Replikationsstudien
- Standardisierte Inhaltsanalysen

**Vorteile:**
- Höchste Vergleichbarkeit
- Klare theoretische Fundierung
- Schnelle Verarbeitung

**Nachteile:**
- Keine neuen Erkenntnisse möglich
- Gefahr des "Übersehens" relevanter Aspekte

### 3.2 Abduktiver Modus (`abductive`)

**Prinzip:** Erweiterung nur auf Subkategorien-Ebene

**Anwendung:**
- Verfeinerung bestehender Theorien
- Detaillierung bekannter Phänomene
- Explorative Vertiefung

**Vorteile:**
- Balance zwischen Struktur und Offenheit
- Theoretische Kohärenz bleibt erhalten
- Moderate Komplexität

**Nachteile:**
- Hauptkategorien bleiben fix
- Begrenzte theoretische Innovation

### 3.3 Induktiver Modus (`full`)

**Prinzip:** Vollständige Erweiterung um neue Haupt- und Subkategorien

**Anwendung:**
- Theorieentwicklung
- Exploration neuer Phänomene
- Grounded Theory-Ansätze

**Vorteile:**
- Maximale Offenheit für Neues
- Theoretische Innovation möglich
- Umfassende Datenerschließung

**Nachteile:**
- Hohe Komplexität
- Gefahr der Überstrukturierung
- Aufwendige Nachbearbeitung

### 3.4 Grounded Theory Modus (`grounded`)

**Prinzip:** Schrittweise Sammlung von Subcodes mit späterer Hauptkategoriengenerierung

**Anwendung:**
- Reine Grounded Theory-Studien
- Explorative Vorstudien
- Theorieentwicklung aus den Daten

**Vorteile:**
- Maximale Datennähe
- Emergente Theoriebildung
- Minimale Vorannahmen

**Nachteile:**
- Sehr zeitaufwendig
- Hohe analytische Anforderungen
- Unvorhersagbare Ergebnisse

---

## 4. Rolle der KI in QCA-AID

### KI als Kodierungsassistent

Die KI in QCA-AID fungiert als:

1. **Systematischer Kodierer:** Konsistente Anwendung von Kodierregeln
2. **Mustererkenner:** Identifikation wiederkehrender Themen
3. **Kategorienentwickler:** Vorschläge für neue Kategorien (induktive Modi)
4. **Qualitätsprüfer:** Intercoder-Reliabilität durch mehrere KI-Codierer

### Grenzen der KI-Kodierung

- **Kontextverständnis:** Begrenzt auf explizite Textinhalte
- **Kulturelles Wissen:** Keine impliziten kulturellen Codes
- **Kreativität:** Keine echte theoretische Innovation
- **Subjektivität:** Keine Berücksichtigung von Forscherperspektiven

### Qualitätssicherung

- **Mehrfachkodierung:** Verschiedene KI-Codierer mit unterschiedlichen Parametern
- **Konsensbildung:** Automatische Identifikation übereinstimmender Kodierungen
- **Menschliche Kontrolle:** Manuelle Überprüfung und Korrektur möglich
- **Transparenz:** Vollständige Dokumentation aller Entscheidungen

---
## 5. Installation und Einrichtung

### 5.1 Systemvoraussetzungen

**Hardware:**
- Mindestens 4 GB RAM (8 GB empfohlen)
- 2 GB freier Festplattenspeicher
- Internetverbindung (für Cloud-Modelle)

**Software:**
- **Python 3.10 oder 3.11** (WICHTIG: Nicht Python 3.13!)
- Windows 10/11, macOS 10.14+, oder Linux
- Moderner Webbrowser (für Webapp)

### 5.2 Schritt-für-Schritt Installation

#### Schritt 1: Python installieren

**⚠️ Wichtiger Hinweis:** Verwenden Sie Python 3.11 oder älter, da QCA-AID derzeit nicht mit Python 3.13 kompatibel ist!

1. Download von [python.org](https://www.python.org/downloads/release/python-3110/)
2. Installation mit "Add to PATH" aktivieren
3. Überprüfung: `python --version` in der Kommandozeile

#### Schritt 2: QCA-AID herunterladen

**Option A: Git (empfohlen)**
```bash
git clone https://github.com/JustusHenke/QCA-AID.git
cd QCA-AID
```

**Option B: ZIP-Download**
1. GitHub-Repository besuchen
2. "Code" → "Download ZIP"
3. Entpacken und in Ordner wechseln

#### Schritt 3: Abhängigkeiten installieren

```bash
# Alle Pakete installieren
pip install -r requirements.txt

# Deutsches Sprachmodell für spaCy
python -m spacy download de_core_news_sm
```

**Windows-spezifisch:** Falls Fehler auftreten, installieren Sie die Microsoft Visual C++ Build Tools:
- Download: [Visual Studio Build Tools](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)
- Aktivieren Sie "C++ Build Tools" inklusive MSVC und Windows SDK

#### Schritt 4: Installation testen

```bash
# Webapp starten (einfachster Test)
python QCA_AID_app/start_webapp.py

# Oder Kommandozeilen-Version
python QCA-AID.py
```

### 5.3 Erste Konfiguration

#### API-Schlüssel einrichten

Erstellen Sie eine `.env`-Datei im QCA-AID-Verzeichnis:

```bash
# OpenAI (empfohlen für Einsteiger)
OPENAI_API_KEY=sk-proj-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Mistral
MISTRAL_API_KEY=...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
```

**Sicherheitshinweis:** Fügen Sie `.env` zu Ihrer `.gitignore` hinzu!

#### Verzeichnisstruktur erstellen

```
mein-projekt/
├── input/          # Ihre Textdateien (.txt, .pdf, .docx)
├── output/         # Analyseergebnisse
├── config/         # Konfigurationsdateien (optional)
└── codebooks/      # Codebook-Dateien (optional)
```

---

## 6. LLM-Anbieter und Modellauswahl

### 6.1 Übersicht der Anbieter

| Anbieter | Datenschutz | Kosten | Qualität | Einrichtung |
|----------|-------------|--------|----------|-------------|
| **Lokal** ⭐ | ✅ 100% privat | ✅ Kostenlos | ⭐⭐⭐ Gut | ⭐⭐ Mittel |
| **OpenAI** | ⚠️ Cloud | 💰💰 Moderat | ⭐⭐⭐⭐⭐ Exzellent | ⭐⭐⭐⭐⭐ Einfach |
| **Anthropic** | ⚠️ Cloud | 💰💰💰 Hoch | ⭐⭐⭐⭐⭐ Exzellent | ⭐⭐⭐⭐ Einfach |
| **Mistral** | ⚠️ Cloud | 💰 Günstig | ⭐⭐⭐⭐ Sehr gut | ⭐⭐⭐⭐ Einfach |

### 6.2 Lokale Modelle (Empfohlen für sensible Daten)

**Vorteile:**
- ✅ **100% Datenschutz** - Keine Datenübermittlung
- ✅ **Kostenlos** - Keine API-Gebühren
- ✅ **DSGVO-konform** - Ideal für Forschungsdaten
- ✅ **Offline-fähig** - Keine Internetverbindung nötig

**Einrichtung mit LM Studio (Empfohlen für Einsteiger):**

1. **Download:** [lmstudio.ai](https://lmstudio.ai/)
2. **Modell herunterladen:**
   - "Discover" Tab öffnen
   - Nach "Llama 3.1 8B" suchen
   - Download starten
3. **Server starten:**
   - "Local Server" Tab
   - Modell auswählen
   - "Start Server" (Port 1234)
4. **In QCA-AID verwenden:**
   - Webapp: "Local (LM Studio/Ollama)" wählen
   - "🔄 Erkennen" klicken
   - Modell auswählen

**Empfohlene lokale Modelle:**

| Modell | Größe | RAM-Bedarf | Geschwindigkeit | Qualität |
|--------|-------|------------|-----------------|----------|
| **Llama 3.1 8B** | 4.7 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Qwen 2.5 14B** | 8.5 GB | 16 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| **Mistral 7B** | 4.1 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐ |

### 6.3 Cloud-Modelle

**OpenAI (Empfohlen für höchste Qualität):**
- `gpt-4o-mini`: Günstig, schnell, gute Qualität
- `gpt-4o`: Teurer, beste Qualität
- `gpt-4-turbo`: Balance aus Geschwindigkeit und Qualität

**Anthropic (Claude):**
- `claude-3-5-sonnet`: Sehr gute Textanalyse
- `claude-3-opus`: Höchste Qualität, teuer

**Mistral:**
- `mistral-large-latest`: Beste Mistral-Qualität
- `mistral-small-latest`: Günstig, ausreichend

### 6.4 Modellauswahl-Empfehlungen

**Für Einsteiger:**
- Cloud: OpenAI `gpt-4o-mini`
- Lokal: Llama 3.1 8B

**Für sensible Daten:**
- Nur lokale Modelle verwenden
- Qwen 2.5 14B (beste Qualität)

**Für große Projekte:**
- Cloud: OpenAI `gpt-4o` (beste Qualität)
- Lokal: Llama 3.1 70B (falls genug RAM)

**Für Budgetbeschränkungen:**
- Cloud: Mistral `mistral-small-latest`
- Lokal: Mistral 7B

---

## 7. Konfigurationseinstellungen

### 7.1 Konfigurationsformate

QCA-AID unterstützt zwei Formate, die automatisch synchronisiert werden:

**Excel-Format (`QCA-AID-Codebook.xlsx`):**
- ✅ Vertraute Oberfläche
- ✅ Einfache Bearbeitung
- ❌ Langsamer beim Laden
- ❌ Schwieriger für Versionskontrolle

**JSON-Format (`QCA-AID-Codebook.json`):**
- ✅ 10x schneller beim Laden
- ✅ Ideal für Git-Versionskontrolle
- ✅ Bessere Performance
- ❌ Erfordert JSON-Kenntnisse

### 7.2 Grundkonfiguration

#### Modell-Einstellungen

```json
{
  "config": {
    "MODEL_PROVIDER": "OpenAI",        // "OpenAI", "Anthropic", "Mistral", "local"
    "MODEL_NAME": "gpt-4o-mini",       // Spezifisches Modell
    "DATA_DIR": "input",               // Eingabeverzeichnis
    "OUTPUT_DIR": "output"             // Ausgabeverzeichnis
  }
}
```

#### Chunk-Einstellungen

```json
{
  "CHUNK_SIZE": 1000,        // Textabschnittsgröße (800-1500 Z.)
  "CHUNK_OVERLAP": 50,       // Überlappung zwischen Chunks (30-100 Z.)
  "BATCH_SIZE": 5            // Parallel verarbeitete Chunks (3-12)
}
```

**Empfehlungen nach Dokumenttyp:**

| Dokumenttyp | CHUNK_SIZE | CHUNK_OVERLAP | BATCH_SIZE |
|-------------|------------|---------------|------------|
| **Interviews** | 1000 | 50 | 5 |
| **Lange Texte** | 1500 | 100 | 4 |
| **Kurze Dokumente** | 800 | 30 | 8 |
| **Akademische Papers** | 1200 | 60 | 5 |

### 7.3 Erweiterte Einstellungen

#### Analysemodus-Konfiguration

```json
{
  "ANALYSIS_MODE": "deductive",      // "deductive", "abductive", "grounded"
  "CODE_WITH_CONTEXT": true,         // Kontextuelle Kodierung
  "MULTIPLE_CODINGS": true,          // Mehrfachkodierungen erlauben
  "MULTIPLE_CODING_THRESHOLD": 0.85  // Schwellwert für Mehrfachkodierung
}
```

#### Coder-Einstellungen

```json
{
  "CODER_SETTINGS": [
    {
      "temperature": 0.3,    // Konsistenz (0.0-1.0)
      "coder_id": "auto_1"   // Eindeutige ID
    },
    {
      "temperature": 0.5,    // Etwas kreativer
      "coder_id": "auto_2"
    }
  ]
}
```

**Temperature-Empfehlungen:**
- **0.0-0.3:** Sehr konsistent (deduktive Kodierung)
- **0.4-0.6:** Ausgewogen (abduktive Kodierung)
- **0.7-1.0:** Kreativ (induktive Kodierung)

#### Qualitätssicherung

```json
{
  "REVIEW_MODE": "consensus",        // "auto", "consensus", "majority", "manual"
  "AUTO_SAVE_INTERVAL": 10,          // Automatische Sicherung (Minuten)
  "MANUAL_CODING_ENABLED": false     // Manuelle Kodierung aktivieren
}
```

### 7.4 Attribut-Extraktion

QCA-AID kann Metadaten aus Dateinamen extrahieren:

```json
{
  "ATTRIBUTE_LABELS": {
    "attribut1": "Hochschultyp",
    "attribut2": "Position", 
    "attribut3": "Fachbereich"
  }
}
```

**Beispiel:**
- Dateiname: `Universität_Professor_Informatik_Interview.txt`
- Extrahiert: Hochschultyp="Universität", Position="Professor", Fachbereich="Informatik"

### 7.5 Performance-Optimierung

#### Batch-Größe anpassen

```json
{
  "BATCH_SIZE": 8  // Erhöhen für mehr Geschwindigkeit, reduzieren für mehr Präzision
}
```

**Empfehlungen:**
- **Hohe Präzision:** 3-4 (langsamer, genauer)
- **Standard:** 5-8 (ausgewogen)
- **Hohe Geschwindigkeit:** 10-12 (schneller, weniger präzise)

#### Kontextuelle Kodierung

```json
{
  "CODE_WITH_CONTEXT": true  // Aktiviert progressive Dokumentzusammenfassung
}
```

**Vorteile:**
- Bessere Kontextsensitivität
- Konsistentere Kodierung innerhalb von Dokumenten

**Nachteile:**
- Langsamere Verarbeitung
- Höherer Token-Verbrauch

---
## 8. Codebook-Entwicklung und -Pflege

### 8.1 Struktur eines QCA-AID Codebooks

Ein vollständiges Codebook besteht aus vier Hauptkomponenten:

#### Forschungsfrage
```json
{
  "forschungsfrage": "Wie gestaltet sich die digitale Transformation in deutschen Hochschulen und welche Herausforderungen und Chancen lassen sich dabei identifizieren?"
}
```

**Best Practices:**
- Formulieren Sie präzise und fokussiert
- Vermeiden Sie zu breite oder zu enge Fragestellungen
- Die Frage sollte zum Kategoriensystem passen

#### Kodierregeln
```json
{
  "kodierregeln": {
    "general": [
      "Kodiere nur explizite Aussagen, keine Interpretationen",
      "Berücksichtige den Kontext der Aussage",
      "Bei Unsicherheit dokumentiere die Gründe"
    ],
    "format": [
      "Markiere relevante Textstellen vollständig",
      "Dokumentiere Begründung der Zuordnung"
    ],
    "exclusion": [
      "Literaturverzeichnisse und Referenzlisten",
      "Tabellarische Datenaufstellungen ohne Interpretation"
    ]
  }
}
```

### 8.2 Kategorienentwicklung

#### Hauptkategorien definieren

**Struktur einer Kategorie:**
```json
{
  "Kategorienname": {
    "definition": "Klare, präzise Definition (min. 15 Wörter)",
    "rules": ["Spezifische Kodierregeln für diese Kategorie"],
    "examples": ["Konkretes Beispiel 1", "Konkretes Beispiel 2"],
    "subcategories": {
      "Subkategorie_1": "Beschreibung der Subkategorie",
      "Subkategorie_2": "Beschreibung der Subkategorie"
    }
  }
}
```

**Beispiel einer gut definierten Kategorie:**
```json
{
  "Akteure": {
    "definition": "Erfasst alle handelnden Personen, Gruppen oder Institutionen sowie deren Rollen, Beziehungen und Interaktionen im Kontext der digitalen Transformation",
    "rules": [
      "Codiere Aussagen zu: Individuen, Gruppen, Organisationen, Netzwerken",
      "Berücksichtige sowohl formelle als auch informelle Akteure",
      "Achte auf Machtbeziehungen und Hierarchien"
    ],
    "examples": [
      "Die Projektleiterin hat die Entscheidung für das neue LMS eigenständig getroffen",
      "Die Arbeitsgruppe Digitalisierung trifft sich wöchentlich zur Abstimmung",
      "Als Vermittler zwischen IT-Abteilung und Fakultät konnte er den Konflikt lösen"
    ],
    "subcategories": {
      "Individuelle_Akteure": "Einzelpersonen wie Lehrende, Studierende, IT-Personal",
      "Kollektive_Akteure": "Gruppen, Organisationen, Institutionen wie Fakultäten",
      "Beziehungen": "Interaktionen, Hierarchien, Netzwerke zwischen Akteuren",
      "Rollen": "Formelle und informelle Positionen wie Innovationstreiber"
    }
  }
}
```

### 8.3 Qualitätskriterien für Kategorien

#### Definition (erforderlich)
- **Mindestlänge:** 15 Wörter
- **Klarheit:** Eindeutige Abgrenzung zu anderen Kategorien
- **Vollständigkeit:** Alle relevanten Aspekte erfasst
- **Operationalisierbarkeit:** Konkret anwendbar

#### Regeln (empfohlen)
- **Spezifität:** Konkrete Anweisungen für diese Kategorie
- **Grenzfälle:** Hinweise für schwierige Entscheidungen
- **Ausschlüsse:** Was NICHT zur Kategorie gehört

#### Beispiele (erforderlich, min. 2)
- **Vielfalt:** Verschiedene Facetten der Kategorie zeigen
- **Realitätsnähe:** Authentische, kontextnahe Beispiele
- **Grenzfälle:** Auch schwierige Fälle illustrieren

#### Subkategorien (erforderlich, min. 2)
- **Vollständigkeit:** Alle wichtigen Aspekte abdecken
- **Trennschärfe:** Klare Abgrenzung untereinander
- **Ausgewogenheit:** Ähnlicher Abstraktionsgrad

### 8.4 Codebook-Pflege und Iteration

#### Induktive Codes importieren

**[Screenshot-Platzhalter: Webapp Codebook-Tab mit Import-Button]**

1. **Automatische Erkennung:** Webapp scannt Output-Ordner nach induktiven Codes
2. **Import-Dialog:** Auswahl der Analyse-Datei mit gewünschten Codes
3. **Vorschau:** Überprüfung der zu importierenden Codes
4. **Konflikt-Behandlung:** Umbenennungsoptionen bei Namenskonflikten
5. **Integration:** Codes werden in separater Sektion angezeigt

#### Iterative Verfeinerung

**Workflow:**
```
Iteration 1: Basis-Codebook (5 deduktive Kategorien)
    ↓
Analyse mit abduktivem Modus
    ↓
Import neuer Subkategorien (8 Kategorien total)
    ↓
Iteration 2: Erweitertes Codebook
    ↓
Weitere Analyse
    ↓
Sättigung erreicht (keine neuen Kategorien)
```

#### Versionskontrolle

**Mit Git (empfohlen):**
```bash
# Änderungen verfolgen
git add QCA-AID-Codebook.json
git commit -m "Kategorien 'Technologien' erweitert um KI-Subkategorien"

# Versionen vergleichen
git diff HEAD~1 QCA-AID-Codebook.json
```

**Manuelle Dokumentation:**
- Änderungsprotokoll führen
- Begründungen für Anpassungen notieren
- Datum und Version dokumentieren

### 8.5 Validierung und Qualitätskontrolle

#### Automatische Validierung

QCA-AID prüft automatisch:
- Mindestlänge von Definitionen
- Anzahl der Beispiele und Subkategorien
- Ähnlichkeit zwischen Kategorien
- Namenskonventionen

#### Manuelle Überprüfung

**Checkliste für Kategorien:**
- [ ] Definition ist klar und abgrenzend
- [ ] Mindestens 2 aussagekräftige Beispiele
- [ ] Subkategorien decken Kategorie vollständig ab
- [ ] Keine Überschneidungen mit anderen Kategorien
- [ ] Regeln sind operationalisierbar

**Checkliste für Gesamtsystem:**
- [ ] Alle Kategorien auf ähnlichem Abstraktionsniveau
- [ ] System ist vollständig (alle relevanten Aspekte erfasst)
- [ ] System ist sparsam (keine redundanten Kategorien)
- [ ] Kategorien sind theoretisch fundiert

---

## 9. Arbeiten mit der Webapp

### 9.1 Webapp-Übersicht

Die QCA-AID Webapp bietet eine intuitive Benutzeroberfläche mit vier Hauptbereichen:

**[Screenshot-Platzhalter: Webapp-Hauptansicht mit Tabs]**

1. **Konfiguration:** Technische Einstellungen und Modellauswahl
2. **Codebook:** Kategorienentwicklung und -verwaltung
3. **Analyse:** Durchführung und Überwachung von Analysen
4. **Explorer:** Ergebnisvisualisierung und -export

### 9.2 Projekt-Management

#### Projekt-Root festlegen

**[Screenshot-Platzhalter: Projekt-Verzeichnis-Dialog]**

1. **Verzeichnis wählen:** Klick auf "📁 Projekt-Verzeichnis ändern"
2. **Ordner auswählen:** Navigation zum gewünschten Projektordner
3. **Automatische Speicherung:** Einstellungen werden in `.qca-aid-project.json` gespeichert

**Empfohlene Projektstruktur:**
```
mein-forschungsprojekt/
├── input/                    # Eingabedateien
│   ├── interviews/
│   ├── documents/
│   └── transcripts/
├── output/                   # Analyseergebnisse
├── config/                   # Konfigurationsdateien
├── codebooks/               # Codebook-Versionen
└── .qca-aid-project.json    # Projekt-Einstellungen
```

### 9.3 Konfiguration-Tab

**[Screenshot-Platzhalter: Konfiguration-Tab mit Einstellungen]**

#### Datei-Browser verwenden

1. **Konfiguration laden:** Klick auf 📁 neben Pfad-Eingabe
2. **Datei auswählen:** Navigation zu `.json` oder `.xlsx` Datei
3. **Automatische Erkennung:** Format wird automatisch erkannt
4. **Validierung:** Echtzeit-Überprüfung der Einstellungen

#### Modell-Einstellungen

**Cloud-Modelle:**
1. **Anbieter wählen:** OpenAI, Anthropic, Mistral
2. **Modell auswählen:** Dropdown zeigt verfügbare Modelle
3. **API-Key prüfen:** Automatische Validierung

**Lokale Modelle:**
1. **"Local" auswählen:** Provider auf "Local (LM Studio/Ollama)" setzen
2. **Erkennung starten:** Klick auf "🔄 Lokale Modelle erkennen"
3. **Modell wählen:** Aus erkannten Modellen auswählen

#### Performance-Einstellungen

**[Screenshot-Platzhalter: Performance-Einstellungen Panel]**

- **Chunk-Größe:** Schieberegler für Textabschnittsgröße
- **Batch-Größe:** Balance zwischen Geschwindigkeit und Qualität
- **Kontextuelle Kodierung:** Toggle für erweiterten Kontext

### 9.4 Codebook-Tab

**[Screenshot-Platzhalter: Codebook-Editor mit Kategorien]**

#### Kategorien bearbeiten

1. **Neue Kategorie:** Klick auf "➕ Kategorie hinzufügen"
2. **Felder ausfüllen:**
   - Name (ohne Leerzeichen, Unterstriche verwenden)
   - Definition (mindestens 15 Wörter)
   - Regeln (optional, aber empfohlen)
   - Beispiele (mindestens 2)
   - Subkategorien (mindestens 2)

3. **Validierung:** Echtzeit-Feedback bei Eingabe
4. **Speichern:** Automatische Validierung vor Speicherung

#### Induktive Codes importieren

**[Screenshot-Platzhalter: Import-Dialog für induktive Codes]**

1. **Benachrichtigung beachten:** Info über verfügbare Codes
2. **Import starten:** Klick auf "Induktive Codes importieren"
3. **Datei auswählen:** Analyse-Datei mit gewünschten Codes
4. **Vorschau prüfen:** Übersicht der zu importierenden Codes
5. **Konflikte lösen:** Umbenennungsoptionen bei Namenskonflikten
6. **Import bestätigen:** Codes werden in separater Sektion angezeigt

### 9.5 Analyse-Tab

**[Screenshot-Platzhalter: Analyse-Tab mit Fortschrittsanzeige]**

#### Eingabedateien verwalten

1. **Dateien überprüfen:** Liste aller Dateien im Input-Verzeichnis
2. **Vorschau anzeigen:** Klick auf Dateinamen für Textvorschau
3. **Attribute prüfen:** Automatische Extraktion aus Dateinamen

#### Analyse starten

1. **Konfiguration prüfen:** Grüner Haken bei gültiger Konfiguration
2. **Codebook validieren:** Grüner Haken bei gültigem Codebook
3. **Analyse starten:** Klick auf "🚀 Analyse starten"
4. **Fortschritt verfolgen:** Echtzeit-Updates und Logs

#### Analyse überwachen

**[Screenshot-Platzhalter: Fortschrittsbalken und Live-Logs]**

- **Fortschrittsbalken:** Visueller Fortschritt der Analyse
- **Live-Logs:** Detaillierte Informationen zum Analyseverlauf
- **Statistiken:** Token-Verbrauch, Geschwindigkeit, Kosten
- **Stopp-Funktion:** Analyse bei Bedarf unterbrechen

### 9.6 Explorer-Tab

**[Screenshot-Platzhalter: Explorer mit Ergebnisübersicht]**

#### Ergebnisse durchsuchen

1. **Output-Dateien:** Liste aller Analyseergebnisse
2. **Datei-Vorschau:** Schnelle Übersicht der Inhalte
3. **Metadaten:** Datum, Größe, Analysemodus
4. **Download:** Direkte Download-Links

#### Visualisierungen konfigurieren

1. **Explorer-Config laden:** Konfiguration für Diagramme
2. **Diagrammtypen wählen:** Heatmaps, Netzwerke, Balkendiagramme
3. **Filter setzen:** Nach Kategorien, Attributen, Dokumenten
4. **Export:** Diagramme als PNG/PDF speichern

---

## 10. Output-Sheets und Ergebnisinterpretation

### 10.1 Struktur der Analyseergebnisse

QCA-AID erstellt eine umfassende Excel-Datei mit mehreren Arbeitsblättern:

**[Screenshot-Platzhalter: Excel-Datei mit Sheet-Übersicht]**

#### Hauptergebnisse (Sheet: "Codings")

**Spaltenstruktur:**
- **Dokument:** Quelldatei des Textsegments
- **Chunk_ID:** Eindeutige Segment-Nummer
- **Text:** Originaltext des kodierten Segments
- **Hauptkategorie:** Zugewiesene Hauptkategorie
- **Subkategorie:** Zugewiesene Subkategorie
- **Konfidenz:** Sicherheit der Kodierung (0.0-1.0)
- **Coder_ID:** Identifikation des Kodierers
- **Begründung:** Erklärung der Kodierentscheidung
- **Attribut_1/2/3:** Extrahierte Metadaten aus Dateinamen

**[Screenshot-Platzhalter: Codings-Sheet mit Beispieldaten]**

#### Häufigkeitsanalysen (Sheet: "Frequencies")

**Inhalte:**
- Absolute und relative Häufigkeiten pro Kategorie
- Verteilung nach Attributen (z.B. Hochschultyp, Position)
- Kreuztabellen zwischen Kategorien und Attributen
- Statistische Kennwerte (Mittelwerte, Standardabweichungen)

**[Screenshot-Platzhalter: Frequencies-Sheet mit Diagrammen]**

#### Intercoder-Reliabilität (Sheet: "Reliability")

**Metriken:**
- **Cohens Kappa:** Übereinstimmung zwischen Kodierern
- **Prozentuale Übereinstimmung:** Einfache Übereinstimmungsrate
- **Konfusionsmatrix:** Detaillierte Übereinstimmungsanalyse
- **Kategoriespezifische Reliabilität:** Reliabilität pro Kategorie

**Interpretation:**
- **κ > 0.8:** Sehr gute Übereinstimmung
- **κ 0.6-0.8:** Gute Übereinstimmung
- **κ 0.4-0.6:** Moderate Übereinstimmung
- **κ < 0.4:** Schlechte Übereinstimmung (Überarbeitung nötig)

### 10.2 Induktive Kategorien (Sheet: "Inductive_Categories")

**[Screenshot-Platzhalter: Induktive Kategorien mit Entwicklungshistorie]**

#### Neue Hauptkategorien
- **Name:** Automatisch generierter Kategorienname
- **Definition:** KI-generierte Definition
- **Häufigkeit:** Anzahl der Zuordnungen
- **Beispiele:** Repräsentative Textstellen
- **Qualitätsbewertung:** Automatische Bewertung der Kategorie

#### Neue Subkategorien
- **Hauptkategorie:** Zugehörige übergeordnete Kategorie
- **Subkategorie:** Name der neuen Subkategorie
- **Beschreibung:** Kurze Charakterisierung
- **Abgrenzung:** Unterscheidung zu bestehenden Subkategorien

### 10.3 Kategorienentwicklung (Sheet: "Category_Development")

**Dokumentation der Evolution:**
- **Iteration:** Analysedurchgang
- **Änderungstyp:** Neue Kategorie, Modifikation, Löschung
- **Begründung:** KI-generierte Erklärung
- **Auswirkung:** Anzahl betroffener Kodierungen

### 10.4 Qualitätsindikatoren interpretieren

#### Konfidenzwerte

**[Screenshot-Platzhalter: Konfidenzverteilung als Histogramm]**

- **Hoch (0.8-1.0):** Eindeutige Zuordnungen, hohe Sicherheit
- **Mittel (0.6-0.8):** Plausible Zuordnungen, moderate Sicherheit
- **Niedrig (0.4-0.6):** Unsichere Zuordnungen, manuelle Prüfung empfohlen
- **Sehr niedrig (<0.4):** Problematische Zuordnungen, Überarbeitung nötig

#### Konsistenz-Metriken

**Intra-Coder-Konsistenz:**
- Vergleich desselben Kodierers bei ähnlichen Textstellen
- Indikator für Regelklarheit und Kategorienqualität

**Inter-Coder-Konsistenz:**
- Übereinstimmung zwischen verschiedenen Kodierern
- Indikator für Objektivität und Nachvollziehbarkeit

### 10.5 Ergebnisvalidierung

#### Stichprobenprüfung

**Empfohlenes Vorgehen:**
1. **Zufallsstichprobe:** 10-20% der Kodierungen manuell prüfen
2. **Niedrige Konfidenz:** Alle Kodierungen <0.6 überprüfen
3. **Neue Kategorien:** Alle induktiven Kategorien validieren
4. **Grenzfälle:** Kodierungen an Kategoriengrenzen prüfen

#### Plausibilitätsprüfung

**Fragen zur Selbstreflexion:**
- Entsprechen die Häufigkeitsverteilungen den Erwartungen?
- Sind neue induktive Kategorien theoretisch sinnvoll?
- Gibt es unerwartete Muster in den Daten?
- Sind die Kodierungen nachvollziehbar begründet?

---
## 11. Optimaler Kodiermodus nach Forschungszielen

### 11.1 Entscheidungsmatrix für Kodiermodi

**[Screenshot-Platzhalter: Entscheidungsbaum für Modusauswahl]**

| Forschungsziel | Theoriestand | Datenmenge | Empfohlener Modus | Begründung |
|----------------|--------------|------------|-------------------|------------|
| **Theorieprüfung** | Etabliert | Groß | `deductive` | Maximale Vergleichbarkeit |
| **Theorieentwicklung** | Schwach | Mittel-Groß | `full` | Offenheit für Neues |
| **Theoriemodifikation** | Moderat | Mittel | `abductive` | Balance Struktur/Offenheit |
| **Exploration** | Minimal | Klein-Mittel | `grounded` | Datengetriebene Entwicklung |
| **Replikation** | Etabliert | Beliebig | `deductive` | Exakte Vergleichbarkeit |
| **Methodenvergleich** | Etabliert | Groß | `deductive` + `full` | Systematischer Vergleich |

### 11.2 Deduktiver Modus - Theorieprüfung

#### Anwendungsszenarien

**Ideal für:**
- Hypothesenprüfung mit etablierten Theorien
- Replikationsstudien
- Vergleichsstudien zwischen Gruppen/Zeitpunkten
- Standardisierte Inhaltsanalysen
- Evaluationsstudien mit festen Kriterien

**Beispiel-Forschungsfragen:**
- "Wie unterscheiden sich Digitalisierungsstrategien zwischen Universitäten und Fachhochschulen?"
- "Welche der theoretisch postulierten Barrieren zeigen sich empirisch?"
- "Haben sich die Herausforderungen seit 2020 verändert?"

#### Konfiguration

```json
{
  "ANALYSIS_MODE": "deductive",
  "CODER_SETTINGS": [
    {
      "temperature": 0.2,        // Niedrig für Konsistenz
      "coder_id": "deductive_1"
    },
    {
      "temperature": 0.3,        // Leicht variiert für Reliabilität
      "coder_id": "deductive_2"
    }
  ],
  "REVIEW_MODE": "consensus",    // Nur übereinstimmende Kodierungen
  "MULTIPLE_CODINGS": false     // Eine Kategorie pro Segment
}
```

#### Qualitätssicherung

- **Intercoder-Reliabilität:** Mindestens κ > 0.7
- **Vollständige Abdeckung:** Alle Textstellen sollten kodierbar sein
- **Kategorienbalance:** Keine stark über-/unterrepräsentierten Kategorien

### 11.3 Abduktiver Modus - Theoriemodifikation

#### Anwendungsszenarien

**Ideal für:**
- Verfeinerung bestehender Theorien
- Detaillierung bekannter Phänomene
- Anpassung an neue Kontexte
- Explorative Vertiefung etablierter Konzepte

**Beispiel-Forschungsfragen:**
- "Welche spezifischen Formen von Digitalisierungsstrategien lassen sich unterscheiden?"
- "Wie differenzieren sich die bekannten Herausforderungen im Detail aus?"
- "Welche Subtypen von Akteuren sind relevant?"

#### Konfiguration

```json
{
  "ANALYSIS_MODE": "abductive",
  "CODER_SETTINGS": [
    {
      "temperature": 0.4,        // Moderat für Balance
      "coder_id": "abductive_1"
    },
    {
      "temperature": 0.5,        // Etwas kreativer
      "coder_id": "abductive_2"
    }
  ],
  "REVIEW_MODE": "majority",     // Mehrheitsentscheidung
  "MULTIPLE_CODINGS": true      // Mehrfachkodierungen möglich
}
```

#### Besonderheiten

- **Subkategorien-Entwicklung:** Neue Subkategorien werden automatisch vorgeschlagen
- **Hauptkategorien bleiben:** Theoretische Struktur bleibt erhalten
- **Iterative Verfeinerung:** Mehrere Analysedurchgänge empfohlen

### 11.4 Induktiver Modus - Theorieentwicklung

#### Anwendungsszenarien

**Ideal für:**
- Entwicklung neuer Theorien
- Exploration unbekannter Phänomene
- Entdeckung unerwarteter Muster
- Grounded Theory-Ansätze mit Vorstrukturierung

**Beispiel-Forschungsfragen:**
- "Welche Phänomene zeigen sich bei der Digitalisierung von Hochschulen?"
- "Welche neuen Kategorien emergieren aus den Daten?"
- "Wie lässt sich das Phänomen X theoretisch strukturieren?"

#### Konfiguration

```json
{
  "ANALYSIS_MODE": "full",
  "CODER_SETTINGS": [
    {
      "temperature": 0.6,        // Höher für Kreativität
      "coder_id": "inductive_1"
    },
    {
      "temperature": 0.7,        // Noch kreativer
      "coder_id": "inductive_2"
    }
  ],
  "REVIEW_MODE": "manual",       // Manuelle Überprüfung nötig
  "MULTIPLE_CODINGS": true,      // Mehrfachkodierungen erwünscht
  "CODE_WITH_CONTEXT": true     // Kontext für bessere Kategorienbildung
}
```

#### Herausforderungen

- **Überstrukturierung:** Gefahr zu vieler neuer Kategorien
- **Qualitätskontrolle:** Intensive manuelle Nachbearbeitung nötig
- **Theoretische Integration:** Neue Kategorien müssen theoretisch eingeordnet werden

### 11.5 Grounded Theory Modus - Datengetriebene Entwicklung

#### Anwendungsszenarien

**Ideal für:**
- Reine Grounded Theory-Studien
- Explorative Vorstudien
- Theorieentwicklung ohne Vorannahmen
- Entdeckung emergenter Phänomene

**Beispiel-Forschungsfragen:**
- "Was passiert bei der Digitalisierung von Hochschulen?" (ohne Vorannahmen)
- "Welche Kategorien entwickeln sich aus den Daten?"
- "Wie strukturieren sich die Erfahrungen der Akteure?"

#### Konfiguration

```json
{
  "ANALYSIS_MODE": "grounded",
  "CODER_SETTINGS": [
    {
      "temperature": 0.8,        // Hoch für maximale Offenheit
      "coder_id": "grounded_1"
    }
  ],
  "REVIEW_MODE": "manual",       // Vollständige manuelle Kontrolle
  "MULTIPLE_CODINGS": true,
  "CODE_WITH_CONTEXT": true,
  "BATCH_SIZE": 3               // Kleinere Batches für Präzision
}
```

#### Besonderheiten

- **Schrittweise Entwicklung:** Codes werden zunächst gesammelt, später zu Hauptkategorien gruppiert
- **Iterative Analyse:** Mehrere Durchgänge mit Anpassung des Kategoriensystems
- **Theoretische Sättigung:** Analyse bis keine neuen Kategorien mehr entstehen

### 11.6 Materialspezifische Empfehlungen

#### Interview-Transkripte

**Charakteristika:**
- Dialogstruktur mit Fragen und Antworten
- Umgangssprache und Füllwörter
- Subjektive Perspektiven und Erfahrungen

**Empfohlene Konfiguration:**
```json
{
  "CHUNK_SIZE": 1000,           // Längere Chunks für Kontext
  "CHUNK_OVERLAP": 60,          // Mehr Überlappung für Dialogkontinuität
  "CODE_WITH_CONTEXT": true,    // Wichtig für Gesprächskontext
  "kodierregeln": {
    "exclusion": [
      "Interviewerfragen ohne inhaltlichen Bezug",
      "Füllwörter und Pausen",
      "Technische Unterbrechungen"
    ]
  }
}
```

#### Akademische Texte

**Charakteristika:**
- Formale Sprache und Fachterminologie
- Strukturierte Argumentation
- Literaturverweise und Zitate

**Empfohlene Konfiguration:**
```json
{
  "CHUNK_SIZE": 1200,           // Größere Chunks für komplexe Argumente
  "CHUNK_OVERLAP": 40,          // Weniger Überlappung bei klarer Struktur
  "CODE_WITH_CONTEXT": false,   // Weniger wichtig bei strukturierten Texten
  "kodierregeln": {
    "exclusion": [
      "Literaturverzeichnisse",
      "Methodische Beschreibungen",
      "Reine Zitate ohne Interpretation"
    ]
  }
}
```

#### Dokumente und Berichte

**Charakteristika:**
- Offizielle Sprache
- Strukturierte Gliederung
- Fakten und Empfehlungen

**Empfohlene Konfiguration:**
```json
{
  "CHUNK_SIZE": 800,            // Kleinere Chunks für präzise Fakten
  "CHUNK_OVERLAP": 30,          // Minimale Überlappung
  "MULTIPLE_CODINGS": false,    // Eindeutige Zuordnungen
  "kodierregeln": {
    "exclusion": [
      "Inhaltsverzeichnisse",
      "Tabellarische Auflistungen",
      "Formale Anhänge"
    ]
  }
}
```

#### Social Media und informelle Texte

**Charakteristika:**
- Kurze, fragmentierte Texte
- Umgangssprache und Slang
- Emotionale Ausdrücke

**Empfohlene Konfiguration:**
```json
{
  "CHUNK_SIZE": 500,            // Kleine Chunks für kurze Posts
  "CHUNK_OVERLAP": 20,          // Minimale Überlappung
  "BATCH_SIZE": 10,             // Mehr parallele Verarbeitung
  "CODER_SETTINGS": [
    {
      "temperature": 0.6,       // Höher für umgangssprachliche Nuancen
      "coder_id": "social_1"
    }
  ]
}
```

### 11.7 Kombinierte Ansätze

#### Sequenzielle Analyse

**Workflow:**
1. **Explorative Phase:** `grounded` Modus für erste Kategorienentwicklung
2. **Strukturierungsphase:** `abductive` Modus für Systematisierung
3. **Validierungsphase:** `deductive` Modus für finale Überprüfung

#### Parallele Analyse

**Vergleichende Kodierung:**
- Gleiche Daten mit verschiedenen Modi analysieren
- Systematischer Vergleich der Ergebnisse
- Triangulation für höhere Validität

**Beispiel-Konfiguration:**
```json
{
  "ANALYSIS_CONFIGS": [
    {
      "name": "deductive_analysis",
      "ANALYSIS_MODE": "deductive",
      "OUTPUT_DIR": "output/deductive"
    },
    {
      "name": "inductive_analysis", 
      "ANALYSIS_MODE": "full",
      "OUTPUT_DIR": "output/inductive"
    }
  ]
}
```

---

## 12. Best Practices und Qualitätssicherung

### 12.1 Vorbereitung der Datengrundlage

#### Textqualität sicherstellen

**Dokumentenvorbereitung:**
- **Bereinigung:** Entfernung von Literaturverzeichnissen, Fußnoten, Seitenzahlen
- **Formatierung:** Einheitliche Textformatierung, keine Sonderzeichen
- **Vollständigkeit:** Überprüfung auf fehlende Textpassagen (besonders bei PDFs)
- **Kodierung:** UTF-8 Encoding für Umlaute und Sonderzeichen

**[Screenshot-Platzhalter: Beispiel für bereinigte vs. unbereinigte Dokumente]**

#### Dateiorganisation

**Namenskonvention:**
```
Attribut1_Attribut2_Attribut3_Bezeichnung.txt

Beispiele:
Universität_Professor_Informatik_Interview_2024-01-15.txt
FH_Studierende_BWL_Fokusgruppe_2024-02-20.txt
Ministerium_Referent_Politik_Dokument_2024-03-10.txt
```

**Verzeichnisstruktur:**
```
projekt/
├── input/
│   ├── interviews/           # Nach Datentyp organisiert
│   ├── documents/
│   └── focus_groups/
├── output/
│   ├── 2024-01-15_analysis/  # Nach Datum organisiert
│   └── 2024-02-20_analysis/
└── codebooks/
    ├── v1.0_initial.json     # Versionierte Codebooks
    ├── v1.1_refined.json
    └── v2.0_final.json
```

### 12.2 Iterative Qualitätssicherung

#### Pilotphase (10-20% der Daten)

**Ziele:**
- Kategorienqualität testen
- Kodierregeln verfeinern
- Technische Parameter optimieren
- Erste Reliabilitätsprüfung

**Vorgehen:**
1. **Stichprobe ziehen:** Repräsentative Auswahl der Dokumente
2. **Erste Kodierung:** Mit vorläufigem Codebook
3. **Manuelle Überprüfung:** 100% der Pilotdaten manuell prüfen
4. **Anpassungen:** Kategorien und Regeln überarbeiten
5. **Wiederholung:** Bis zufriedenstellende Qualität erreicht

#### Hauptanalyse mit Stichprobenkontrolle

**Qualitätskontrolle während der Analyse:**
- **10% Stichprobe:** Zufällige Auswahl für manuelle Überprüfung
- **Niedrige Konfidenz:** Alle Kodierungen <0.6 prüfen
- **Neue Kategorien:** Alle induktiven Kategorien validieren
- **Grenzfälle:** Kodierungen an Kategoriengrenzen kontrollieren

**[Screenshot-Platzhalter: Qualitätskontroll-Dashboard in der Webapp]**

### 12.3 Intercoder-Reliabilität optimieren

#### Mehrere KI-Codierer konfigurieren

**Empfohlene Konfiguration:**
```json
{
  "CODER_SETTINGS": [
    {
      "temperature": 0.3,
      "coder_id": "conservative",
      "description": "Konservativer Kodierer für eindeutige Fälle"
    },
    {
      "temperature": 0.5,
      "coder_id": "balanced", 
      "description": "Ausgewogener Kodierer für Standardfälle"
    },
    {
      "temperature": 0.7,
      "coder_id": "creative",
      "description": "Kreativer Kodierer für Grenzfälle"
    }
  ]
}
```

#### Konsensbildung konfigurieren

**Review-Modi:**
- **`consensus`:** Nur übereinstimmende Kodierungen (höchste Qualität)
- **`majority`:** Mehrheitsentscheidung bei 3+ Kodierern
- **`weighted`:** Gewichtung nach Kodierer-Performance
- **`manual`:** Manuelle Entscheidung bei Konflikten

#### Reliabilitäts-Benchmarks

**Interpretationshilfen:**
- **κ > 0.8:** Exzellente Übereinstimmung → Analyse fortsetzen
- **κ 0.6-0.8:** Gute Übereinstimmung → Stichprobenkontrolle
- **κ 0.4-0.6:** Moderate Übereinstimmung → Kategorien überarbeiten
- **κ < 0.4:** Schlechte Übereinstimmung → Grundlegende Überarbeitung nötig

### 12.4 Kategorienqualität sicherstellen

#### Validierungscheckliste

**Für jede Kategorie prüfen:**
- [ ] **Definition:** Klar, abgrenzend, mindestens 15 Wörter
- [ ] **Operationalisierung:** Konkret anwendbare Regeln
- [ ] **Beispiele:** Mindestens 2, verschiedene Facetten zeigend
- [ ] **Abgrenzung:** Keine Überschneidungen mit anderen Kategorien
- [ ] **Vollständigkeit:** Alle relevanten Aspekte erfasst
- [ ] **Theoretische Fundierung:** Bezug zu Forschungsstand

#### Kategorienoptimierung

**Häufige Probleme und Lösungen:**

| Problem | Symptom | Lösung |
|---------|---------|--------|
| **Zu breite Kategorie** | >40% aller Kodierungen | Aufteilen in Subkategorien |
| **Zu enge Kategorie** | <2% aller Kodierungen | Mit ähnlicher Kategorie zusammenfassen |
| **Überschneidungen** | Niedrige Intercoder-Reliabilität | Abgrenzungskriterien schärfen |
| **Unklare Definition** | Inkonsistente Kodierungen | Definition präzisieren, Beispiele ergänzen |
| **Fehlende Kategorie** | Viele "Sonstige"-Kodierungen | Neue Kategorie entwickeln |

### 12.5 Technische Optimierung

#### Performance-Tuning

**Batch-Größe optimieren:**
```python
# Testlauf mit verschiedenen Batch-Größen
batch_sizes = [3, 5, 8, 10, 12]
for size in batch_sizes:
    # Zeitmessung und Qualitätsbewertung
    # Optimale Balance finden
```

**Chunk-Parameter anpassen:**
- **Zu kleine Chunks:** Kontextverlust, fragmentierte Kodierungen
- **Zu große Chunks:** Mehrfachkodierungen, unklare Zuordnungen
- **Optimale Größe:** 800-1200 Zeichen je nach Texttyp

#### Kostenoptimierung

**Token-Verbrauch reduzieren:**
- **Präzise Kategorien:** Weniger Nachfragen durch klarere Definitionen
- **Optimale Batch-Größe:** Weniger API-Calls durch größere Batches
- **Günstigere Modelle:** Für einfache Kodierungen ausreichend
- **Lokale Modelle:** Kostenlos für sensible oder große Datenmengen

**[Screenshot-Platzhalter: Token-Tracking und Kostenübersicht]**

### 12.6 Dokumentation und Nachvollziehbarkeit

#### Analysedokumentation

**Pflichtangaben:**
- **Codebook-Version:** Mit Datum und Änderungshistorie
- **Konfiguration:** Vollständige technische Parameter
- **Stichprobenkontrolle:** Umfang und Ergebnisse der manuellen Prüfung
- **Reliabilitätswerte:** Intercoder-Übereinstimmung pro Kategorie
- **Anpassungen:** Alle Änderungen am Kategoriensystem dokumentieren

#### Forschungstagebuch führen

**Empfohlene Einträge:**
```
Datum: 2024-01-15
Aktivität: Pilotanalyse Interview-Daten
Ergebnisse: κ = 0.65, Kategorie "Technologien" zu breit
Anpassungen: Aufgeteilt in "Hardware" und "Software"
Nächste Schritte: Wiederholung mit angepasstem Codebook

Datum: 2024-01-20
Aktivität: Hauptanalyse Batch 1-3
Ergebnisse: κ = 0.78, neue induktive Kategorie "KI-Tools"
Beobachtungen: Häufige Erwähnung von ChatGPT und ähnlichen Tools
Entscheidung: Kategorie ins Codebook aufnehmen
```

#### Reproduzierbarkeit sicherstellen

**Versionskontrolle:**
```bash
# Git-Repository für Projekt
git init
git add .
git commit -m "Initial codebook v1.0"

# Änderungen dokumentieren
git add QCA-AID-Codebook.json
git commit -m "Added AI-Tools subcategory to Technologies"

# Tags für wichtige Versionen
git tag -a v1.0 -m "Final codebook for main analysis"
```

**Konfiguration archivieren:**
- Vollständige Konfigurationsdateien speichern
- Screenshots der Webapp-Einstellungen
- Verwendete Modellversionen dokumentieren
- API-Parameter und Batch-Größen notieren

---
## 13. Häufige Probleme und Lösungen

### 13.1 Installation und Setup

#### Problem: Python-Versionskonflikte

**Symptom:** `ModuleNotFoundError` oder Kompatibilitätsfehler

**Ursache:** Python 3.13 oder inkompatible Versionen

**Lösung:**
```bash
# Python-Version prüfen
python --version

# Falls Python 3.13: Python 3.11 installieren
# Download von python.org/downloads/release/python-3110/

# Virtuelle Umgebung mit korrekter Version
python3.11 -m venv qca_aid_env
source qca_aid_env/bin/activate  # Linux/Mac
qca_aid_env\Scripts\activate     # Windows

# Abhängigkeiten neu installieren
pip install -r requirements.txt
```

#### Problem: spaCy-Installation fehlgeschlagen

**Symptom:** `OSError: [E050] Can't find model 'de_core_news_sm'`

**Lösung:**
```bash
# Deutsches Sprachmodell installieren
python -m spacy download de_core_news_sm

# Falls Fehler: Direkt von GitHub installieren
pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
```

#### Problem: Visual C++ Build Tools fehlen (Windows)

**Symptom:** `Microsoft Visual C++ 14.0 is required`

**Lösung:**
1. **Build Tools installieren:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/)
2. **C++ Build Tools** aktivieren
3. **MSVC** und **Windows SDK** auswählen
4. **Alternative:** Anaconda verwenden (enthält vorkompilierte Pakete)

### 13.2 API und Authentifizierung

#### Problem: API-Schlüssel nicht gefunden

**Symptom:** `OpenAI API key not found` oder `Authentication failed`

**Lösung:**
```bash
# .env-Datei im Projektverzeichnis erstellen
echo "OPENAI_API_KEY=sk-proj-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Oder Umgebungsvariable setzen (Windows)
setx OPENAI_API_KEY "sk-proj-..."

# Oder Umgebungsvariable setzen (Linux/Mac)
export OPENAI_API_KEY="sk-proj-..."
```

#### Problem: Rate Limit exceeded

**Symptom:** `Rate limit reached for requests`

**Lösung:**
```json
{
  "BATCH_SIZE": 3,              // Reduzieren für weniger parallele Anfragen
  "REQUEST_DELAY": 1.0,         // Pause zwischen Anfragen (Sekunden)
  "MAX_RETRIES": 5              // Mehr Wiederholungsversuche
}
```

#### Problem: Context length exceeded

**Symptom:** `This model's maximum context length is X tokens`

**Lösung:**
```json
{
  "CHUNK_SIZE": 800,            // Kleinere Chunks verwenden
  "CODE_WITH_CONTEXT": false,   // Kontext deaktivieren
  "BATCH_SIZE": 3               // Weniger Chunks pro Anfrage
}
```

### 13.3 Webapp-spezifische Probleme

#### Problem: Webapp startet nicht

**Symptom:** `ModuleNotFoundError: No module named 'streamlit'`

**Lösung:**
```bash
# Streamlit installieren
pip install streamlit

# Oder alle Abhängigkeiten neu installieren
pip install -r requirements.txt

# Webapp starten
cd QCA_AID_app
python start_webapp.py
```

#### Problem: Port bereits belegt

**Symptom:** `Port 8501 is already in use`

**Lösung:**
```bash
# Andere Streamlit-Instanzen beenden
pkill -f streamlit  # Linux/Mac
taskkill /f /im python.exe  # Windows (alle Python-Prozesse)

# Oder anderen Port verwenden
streamlit run webapp.py --server.port 8502
```

#### Problem: Datei-Browser öffnet nicht

**Symptom:** Klick auf 📁 zeigt keinen Dialog

**Lösung:**
```bash
# tkinter testen
python -m tkinter

# Falls Fehler (Linux):
sudo apt-get install python3-tk

# Falls Fehler (Mac):
# Python von python.org neu installieren

# Alternative: Pfade manuell eingeben
```

### 13.4 Konfiguration und Codebook

#### Problem: JSON-Syntax-Fehler

**Symptom:** `JSONDecodeError: Expecting ',' delimiter`

**Häufige Fehler:**
```json
// FALSCH: Trailing Comma
{
  "CHUNK_SIZE": 1000,
  "BATCH_SIZE": 5,  // ← Komma am Ende
}

// RICHTIG:
{
  "CHUNK_SIZE": 1000,
  "BATCH_SIZE": 5
}

// FALSCH: Einfache Anführungszeichen
{
  'MODEL_PROVIDER': 'OpenAI'  // ← Einfache Anführungszeichen
}

// RICHTIG:
{
  "MODEL_PROVIDER": "OpenAI"
}
```

**Lösung:**
- **Online-Validator:** [jsonlint.com](https://jsonlint.com/)
- **VS Code:** JSON-Syntax-Highlighting aktivieren
- **Python-Test:** `json.load()` zum Testen verwenden

#### Problem: Kategorien-Validierung fehlgeschlagen

**Symptom:** `Definition zu kurz` oder `Zu wenige Beispiele`

**Lösung:**
```json
{
  "Kategorie_Name": {
    "definition": "Mindestens 15 Wörter für eine vollständige und präzise Definition der Kategorie mit klarer Abgrenzung zu anderen Kategorien",
    "examples": [
      "Erstes konkretes Beispiel für die Kategorie",
      "Zweites Beispiel mit anderem Fokus",
      "Drittes Beispiel für Grenzfall"
    ],
    "subcategories": {
      "Sub_1": "Erste Subkategorie",
      "Sub_2": "Zweite Subkategorie"
    }
  }
}
```

### 13.5 Analyse-Probleme

#### Problem: Keine Eingabedateien gefunden

**Symptom:** `No input files found in directory`

**Lösung:**
```bash
# Verzeichnisstruktur prüfen
ls -la input/  # Linux/Mac
dir input\     # Windows

# Unterstützte Formate: .txt, .pdf, .docx
# Dateien in input/ Verzeichnis kopieren

# Pfad in Konfiguration prüfen
{
  "DATA_DIR": "input"  // Relativ zum Projektverzeichnis
}
```

#### Problem: PDF-Texte nicht lesbar

**Symptom:** Leere oder verstümmelte Texte aus PDF-Dateien

**Lösung:**
1. **PDF-Qualität prüfen:** Enthält die PDF Textebene oder nur Bilder?
2. **OCR verwenden:** Für gescannte PDFs externe OCR-Software nutzen
3. **Als Text exportieren:** PDF in Word öffnen und als .txt speichern
4. **Alternative Tools:** Adobe Acrobat, PDFtk, oder Online-Konverter

#### Problem: Analyse bricht ab

**Symptom:** `Analysis stopped unexpectedly` oder Timeout-Fehler

**Mögliche Ursachen und Lösungen:**

**Netzwerkprobleme:**
```json
{
  "MAX_RETRIES": 10,           // Mehr Wiederholungsversuche
  "RETRY_DELAY": 5,            // Längere Wartezeit zwischen Versuchen
  "TIMEOUT": 120               // Längerer Timeout (Sekunden)
}
```

**Speicherprobleme:**
```json
{
  "BATCH_SIZE": 3,             // Kleinere Batches
  "CHUNK_SIZE": 800,           // Kleinere Chunks
  "PARALLEL_WORKERS": 1        // Weniger parallele Prozesse
}
```

**API-Limits:**
```json
{
  "REQUEST_DELAY": 2.0,        // Längere Pausen zwischen Anfragen
  "BATCH_SIZE": 2              // Sehr kleine Batches
}
```

### 13.6 Ergebnis-Probleme

#### Problem: Niedrige Intercoder-Reliabilität

**Symptom:** κ < 0.6 zwischen Kodierern

**Diagnose und Lösungen:**

**Kategorien zu unscharf:**
```json
// Vorher: Unscharf
{
  "Technologie": {
    "definition": "Alles was mit Technik zu tun hat"
  }
}

// Nachher: Präzise
{
  "Technologie": {
    "definition": "Konkrete digitale Werkzeuge, Software und Hardware, die aktiv in Lehr- oder Verwaltungsprozessen eingesetzt werden",
    "rules": [
      "Codiere nur explizit genannte Technologien",
      "Unterscheide zwischen geplanter und tatsächlicher Nutzung"
    ]
  }
}
```

**Zu viele Grenzfälle:**
- Kategorien überarbeiten und schärfer abgrenzen
- Mehr Beispiele für typische und Grenzfälle
- Ausschlusskriterien definieren

#### Problem: Zu viele induktive Kategorien

**Symptom:** >20 neue Kategorien bei induktiver Analyse

**Lösung:**
```json
{
  "ANALYSIS_MODE": "abductive",     // Weniger offener Modus
  "CODER_SETTINGS": [
    {
      "temperature": 0.4,           // Weniger kreativ
      "min_frequency": 3            // Mindesthäufigkeit für neue Kategorien
    }
  ]
}
```

**Nachbearbeitung:**
- Ähnliche Kategorien zusammenfassen
- Seltene Kategorien (<2% der Kodierungen) prüfen
- Hierarchische Struktur entwickeln

#### Problem: Unplausible Kodierungen

**Symptom:** Kodierungen entsprechen nicht den Erwartungen

**Systematische Überprüfung:**
1. **Stichprobe ziehen:** 20-30 zufällige Kodierungen
2. **Manuell bewerten:** Sind die Zuordnungen nachvollziehbar?
3. **Muster identifizieren:** Welche Kategorien sind besonders problematisch?
4. **Ursachen analysieren:** Unklare Definitionen? Schlechte Beispiele?

**Häufige Ursachen:**
- **Zu abstrakte Kategorien:** Konkretere Definitionen entwickeln
- **Fehlende Beispiele:** Mehr und bessere Beispiele hinzufügen
- **Überschneidende Kategorien:** Abgrenzungskriterien schärfen
- **Ungeeignetes Modell:** Besseres/größeres Modell verwenden

### 13.7 Performance-Probleme

#### Problem: Sehr langsame Analyse

**Symptom:** <10 Chunks pro Minute verarbeitet

**Optimierungsmaßnahmen:**

**Batch-Größe erhöhen:**
```json
{
  "BATCH_SIZE": 12,             // Mehr parallele Verarbeitung
  "PARALLEL_WORKERS": 4         // Mehr Worker-Threads
}
```

**Modell wechseln:**
```json
{
  "MODEL_NAME": "gpt-4o-mini"   // Schnelleres Modell statt gpt-4o
}
```

**Lokale Modelle nutzen:**
```json
{
  "MODEL_PROVIDER": "local",
  "MODEL_NAME": "llama3.1:8b"  // Lokales Modell ohne API-Latenz
}
```

#### Problem: Hohe Kosten

**Symptom:** Unerwartete API-Kosten

**Kostenoptimierung:**
```json
{
  "MODEL_NAME": "gpt-4o-mini",      // Günstigeres Modell
  "BATCH_SIZE": 10,                 // Weniger API-Calls
  "CHUNK_SIZE": 800,                // Kleinere Chunks = weniger Tokens
  "CODE_WITH_CONTEXT": false        // Kontext spart Tokens
}
```

**Kostenkontrolle:**
- **Token-Tracking:** Verbrauch in Echtzeit überwachen
- **Budgetlimits:** API-Limits beim Anbieter setzen
- **Testläufe:** Kleine Stichproben vor Vollanalyse
- **Lokale Modelle:** Für große Projekte kostenlos

### 13.8 Debugging und Diagnose

#### Debug-Modus aktivieren

```json
{
  "DEBUG_MODE": true,
  "LOG_LEVEL": "DEBUG",
  "SAVE_INTERMEDIATE": true     // Zwischenergebnisse speichern
}
```

#### Log-Dateien analysieren

**Wichtige Log-Dateien:**
```bash
# QCA-AID Logs
cat .crush/logs/crush.log

# Webapp Logs
cat ~/.streamlit/logs/streamlit.log

# Python Fehler
python QCA-AID.py 2>&1 | tee debug.log
```

#### Systematische Fehlersuche

**Schritt-für-Schritt-Diagnose:**
1. **Minimalkonfiguration:** Einfachste Einstellungen testen
2. **Einzelne Datei:** Nur eine Eingabedatei verwenden
3. **Kleine Chunks:** CHUNK_SIZE auf 200 reduzieren
4. **Einzelner Coder:** Nur einen Kodierer verwenden
5. **Deduktiver Modus:** Komplexität reduzieren

**Isolierung von Problemen:**
```json
// Minimale Testkonfiguration
{
  "MODEL_PROVIDER": "OpenAI",
  "MODEL_NAME": "gpt-4o-mini",
  "CHUNK_SIZE": 200,
  "BATCH_SIZE": 1,
  "ANALYSIS_MODE": "deductive",
  "CODE_WITH_CONTEXT": false,
  "CODER_SETTINGS": [
    {
      "temperature": 0.3,
      "coder_id": "test"
    }
  ]
}
```

### 13.9 Notfall-Wiederherstellung

#### Analyse-Unterbrechung

**Automatische Wiederherstellung:**
- QCA-AID speichert Fortschritt automatisch
- Bei Neustart wird an letzter Position fortgesetzt
- Zwischenergebnisse in `output/temp/` verfügbar

**Manuelle Wiederherstellung:**
```bash
# Letzte Sicherung finden
ls -la output/temp/

# Fortschritt prüfen
grep "Progress:" output/temp/analysis_log.txt

# Analyse fortsetzen
python QCA-AID.py --resume
```

#### Korrupte Konfiguration

**Backup wiederherstellen:**
```bash
# Git-Versionen prüfen
git log --oneline QCA-AID-Codebook.json

# Letzte funktionierende Version wiederherstellen
git checkout HEAD~1 QCA-AID-Codebook.json
```

**Neu erstellen:**
1. **Beispielkonfiguration kopieren:** `examples/config-standard.json`
2. **Schrittweise anpassen:** Nur notwendige Änderungen
3. **Validierung:** Nach jeder Änderung testen

---

## 14. Anhang: Screenshots und Beispiele

### 14.1 Screenshot-Platzhalter

**Hinweis:** Die folgenden Bereiche sind für Screenshots vorgesehen, die Sie nach der Erstellung des Handbuchs einfügen können:

#### Installation und Setup
- [ ] **Python-Installation:** Download-Seite und Installationsoptionen
- [ ] **Verzeichnisstruktur:** Beispiel eines organisierten Projektordners
- [ ] **Erste Webapp-Ansicht:** Startbildschirm nach erfolgreicher Installation

#### Webapp-Bedienung
- [ ] **Hauptnavigation:** Übersicht der vier Haupttabs
- [ ] **Projekt-Dialog:** Auswahl des Projekt-Root-Verzeichnisses
- [ ] **Datei-Browser:** Native Dateiauswahl-Dialoge
- [ ] **Konfiguration-Tab:** Vollständige Ansicht aller Einstellungen
- [ ] **Modell-Auswahl:** Dropdown mit verfügbaren Modellen
- [ ] **Lokale Modelle:** Erkennungs-Dialog für LM Studio/Ollama

#### Codebook-Entwicklung
- [ ] **Codebook-Editor:** Kategorien-Eingabeformular
- [ ] **Validierung:** Echtzeit-Feedback bei Eingabefehlern
- [ ] **Import-Dialog:** Induktive Codes aus vorherigen Analysen
- [ ] **JSON-Vorschau:** Strukturansicht des Codebooks
- [ ] **Kategorien-Übersicht:** Liste aller definierten Kategorien

#### Analyse-Durchführung
- [ ] **Eingabedateien:** Liste mit Dateivorschau
- [ ] **Analyse-Start:** Konfigurationsprüfung und Start-Button
- [ ] **Fortschrittsanzeige:** Live-Updates während der Analyse
- [ ] **Log-Ausgabe:** Detaillierte Fortschrittsinformationen
- [ ] **Fehlerbehandlung:** Beispiele für Fehlermeldungen und Lösungen

#### Ergebnisse und Output
- [ ] **Excel-Übersicht:** Struktur der Ergebnisdatei mit Sheets
- [ ] **Codings-Sheet:** Beispieldaten mit Kodierungen
- [ ] **Häufigkeitsanalyse:** Diagramme und Statistiken
- [ ] **Reliabilitäts-Report:** Intercoder-Übereinstimmung
- [ ] **Induktive Kategorien:** Neu entwickelte Kategorien

#### Explorer und Visualisierung
- [ ] **Explorer-Übersicht:** Ergebnisdateien und Metadaten
- [ ] **Diagramm-Konfiguration:** Einstellungen für Visualisierungen
- [ ] **Netzwerk-Analyse:** Beispiel einer Akteurs-Netzwerk-Visualisierung
- [ ] **Heatmap:** Kategorie-Häufigkeiten nach Attributen
- [ ] **Export-Optionen:** Download und Sharing-Funktionen

### 14.2 Beispiel-Konfigurationen

#### Beispiel 1: Interview-Studie zur Hochschuldigitalisierung

**Forschungskontext:**
- 15 Experteninterviews mit Hochschulleitungen
- Deduktive Analyse mit etabliertem Kategoriensystem
- Fokus auf Strategien und Herausforderungen

**Konfiguration:**
```json
{
  "forschungsfrage": "Welche Digitalisierungsstrategien verfolgen deutsche Hochschulen und welche Herausforderungen identifizieren die Leitungen?",
  "config": {
    "MODEL_PROVIDER": "OpenAI",
    "MODEL_NAME": "gpt-4o-mini",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 50,
    "BATCH_SIZE": 5,
    "ANALYSIS_MODE": "deductive",
    "CODE_WITH_CONTEXT": true,
    "ATTRIBUTE_LABELS": {
      "attribut1": "Hochschultyp",
      "attribut2": "Bundesland",
      "attribut3": "Größe"
    }
  }
}
```

#### Beispiel 2: Explorative Dokumentenanalyse

**Forschungskontext:**
- Analyse von Strategiepapieren und Berichten
- Induktive Kategorienentwicklung
- Grounded Theory-Ansatz

**Konfiguration:**
```json
{
  "forschungsfrage": "Welche Themen und Muster zeigen sich in den Digitalisierungsstrategien deutscher Hochschulen?",
  "config": {
    "MODEL_PROVIDER": "local",
    "MODEL_NAME": "llama3.1:8b",
    "CHUNK_SIZE": 1200,
    "CHUNK_OVERLAP": 60,
    "BATCH_SIZE": 3,
    "ANALYSIS_MODE": "grounded",
    "CODE_WITH_CONTEXT": true,
    "CODER_SETTINGS": [
      {
        "temperature": 0.7,
        "coder_id": "explorative"
      }
    ]
  }
}
```

#### Beispiel 3: Vergleichsstudie mit Mehrfachkodierung

**Forschungskontext:**
- Vergleich zwischen Universitäten und Fachhochschulen
- Hohe Qualitätsanforderungen durch Mehrfachkodierung
- Fokus auf Intercoder-Reliabilität

**Konfiguration:**
```json
{
  "config": {
    "MODEL_PROVIDER": "Anthropic",
    "MODEL_NAME": "claude-3-5-sonnet-20241022",
    "ANALYSIS_MODE": "abductive",
    "REVIEW_MODE": "consensus",
    "CODER_SETTINGS": [
      {
        "temperature": 0.3,
        "coder_id": "conservative"
      },
      {
        "temperature": 0.4,
        "coder_id": "moderate"
      },
      {
        "temperature": 0.5,
        "coder_id": "liberal"
      }
    ]
  }
}
```

### 14.3 Musterdokumente

#### Beispiel-Codebook: Hochschuldigitalisierung

**Vollständiges Kategoriensystem:**
```json
{
  "deduktive_kategorien": {
    "Strategien": {
      "definition": "Geplante und systematische Ansätze zur Gestaltung der digitalen Transformation in Hochschulen, einschließlich Zielsetzungen, Maßnahmen und Umsetzungsplänen",
      "rules": [
        "Codiere sowohl explizite Strategiedokumente als auch implizite strategische Überlegungen",
        "Unterscheide zwischen Top-down und Bottom-up Strategien",
        "Berücksichtige zeitliche Dimensionen (kurz-, mittel-, langfristig)"
      ],
      "examples": [
        "Die Hochschule hat eine umfassende Digitalisierungsstrategie bis 2030 entwickelt",
        "Durch dezentrale Pilotprojekte sollen Best Practices identifiziert werden",
        "Die IT-Strategie sieht eine schrittweise Migration in die Cloud vor"
      ],
      "subcategories": {
        "Top_Down": "Von der Hochschulleitung initiierte und gesteuerte Strategien",
        "Bottom_Up": "Aus den Fakultäten und Bereichen entwickelte Ansätze",
        "Partizipativ": "Gemeinsam entwickelte Strategien mit breiter Beteiligung",
        "Adaptiv": "Flexible, sich anpassende Strategieansätze"
      }
    },
    "Technologien": {
      "definition": "Konkrete digitale Werkzeuge, Plattformen, Systeme und Infrastrukturen, die in Hochschulen eingesetzt werden oder deren Einsatz geplant ist",
      "rules": [
        "Codiere sowohl Hardware als auch Software",
        "Berücksichtige auch geplante oder diskutierte Technologien",
        "Unterscheide zwischen Kern-IT und fachspezifischen Tools"
      ],
      "examples": [
        "Das Learning Management System Moodle wird campusweit genutzt",
        "Neue Videokonferenz-Räume ermöglichen hybride Lehre",
        "KI-Tools wie ChatGPT werden in der Lehre erprobt"
      ],
      "subcategories": {
        "Lernplattformen": "LMS, E-Learning-Systeme, digitale Lernumgebungen",
        "Kommunikation": "Videokonferenz, Chat, Kollaborationstools",
        "Infrastruktur": "Server, Netzwerke, Cloud-Services, Hardware",
        "KI_Tools": "Künstliche Intelligenz und maschinelles Lernen"
      }
    }
  }
}
```

### 14.4 Checklisten und Vorlagen

#### Projekt-Setup Checkliste

**Vor der ersten Analyse:**
- [ ] Python 3.10/3.11 installiert und getestet
- [ ] QCA-AID heruntergeladen und Abhängigkeiten installiert
- [ ] API-Schlüssel konfiguriert (oder lokales Modell eingerichtet)
- [ ] Projektverzeichnis erstellt und strukturiert
- [ ] Eingabedateien vorbereitet und benannt
- [ ] Forschungsfrage formuliert
- [ ] Initiales Kategoriensystem entwickelt
- [ ] Kodierregeln definiert
- [ ] Konfiguration erstellt und validiert

#### Qualitätssicherung Checkliste

**Während der Analyse:**
- [ ] Pilotanalyse mit 10-20% der Daten durchgeführt
- [ ] Intercoder-Reliabilität >0.6 erreicht
- [ ] Stichprobenkontrolle (10% manuell geprüft)
- [ ] Kategorien bei Bedarf angepasst
- [ ] Fortschritt dokumentiert
- [ ] Zwischenergebnisse gesichert

**Nach der Analyse:**
- [ ] Vollständige Ergebnisse validiert
- [ ] Induktive Kategorien überprüft
- [ ] Häufigkeitsverteilungen plausibel
- [ ] Dokumentation vervollständigt
- [ ] Codebook finalisiert
- [ ] Ergebnisse exportiert und archiviert

#### Fehlerbehebung Checkliste

**Bei Problemen systematisch prüfen:**
- [ ] Python-Version korrekt (3.10 oder 3.11)
- [ ] Alle Abhängigkeiten installiert
- [ ] API-Schlüssel gültig und verfügbar
- [ ] Eingabedateien im korrekten Format
- [ ] Konfiguration syntaktisch korrekt
- [ ] Ausreichend Speicherplatz verfügbar
- [ ] Internetverbindung stabil (für Cloud-Modelle)
- [ ] Firewall-Einstellungen korrekt

---

## Fazit und Ausblick

QCA-AID bietet Sozialwissenschaftler:innen ein mächtiges Werkzeug zur KI-unterstützten qualitativen Inhaltsanalyse. Die Kombination aus bewährten methodischen Ansätzen und modernen KI-Technologien ermöglicht es, größere Datenmengen systematisch zu analysieren, ohne die Qualitätsstandards qualitativer Forschung zu vernachlässigen.

### Wichtige Erfolgsfaktoren

1. **Methodische Fundierung:** QCA-AID ersetzt nicht die methodische Expertise, sondern erweitert sie
2. **Qualitätskontrolle:** Regelmäßige manuelle Überprüfung bleibt essentiell
3. **Iterative Entwicklung:** Kategorien und Regeln sollten kontinuierlich verfeinert werden
4. **Transparenz:** Vollständige Dokumentation aller Entscheidungen und Parameter
5. **Kritische Reflexion:** KI-Ergebnisse müssen stets kritisch hinterfragt werden

### Weiterentwicklung

QCA-AID wird kontinuierlich weiterentwickelt. Aktuelle Entwicklungen und Updates finden Sie im [GitHub-Repository](https://github.com/JustusHenke/QCA-AID) und im [Changelog](CHANGELOG.md).

**Kontakt für Feedback und Fragen:**  
Justus Henke  
Institut für Hochschulforschung Halle-Wittenberg  
E-Mail: justus.henke@hof.uni-halle.de

---

**Viel Erfolg bei Ihrer qualitativen Forschung mit QCA-AID!** 🚀

# Konfiguration und Codebook - Anleitung

## ⚙️ Konfiguration

Die Konfiguration steuert alle technischen Parameter der QCA-AID Analyse.

### LLM-Einstellungen

- **Provider**: Wählen Sie den KI-Anbieter (OpenAI, Anthropic, Local, etc.)
  - OpenAI: Bewährte Modelle wie GPT-4
  - Anthropic: Claude-Modelle mit großem Kontextfenster
  - Local: Eigene Modelle via LM Studio oder Ollama
- **Modell**: Spezifisches Modell für die Analyse
  - Größere Modelle = bessere Qualität, aber höhere Kosten
  - Kleinere Modelle = schneller und günstiger

### Verzeichnisse

- **Eingabeverzeichnis**: Ordner mit Ihren Textdateien (.txt, .pdf, .docx)
  - Relativ zum Projektverzeichnis oder absoluter Pfad
  - Standard: `input`
- **Ausgabeverzeichnis**: Zielordner für Analyseergebnisse
  - Enthält Excel-Dateien mit Kodierungen
  - Standard: `output`

### Chunk-Einstellungen

- **Chunk-Größe**: Länge der Textabschnitte in Zeichen (1000-1500 empfohlen)
  - Kleinere Chunks = präzisere Kodierung, mehr API-Calls
  - Größere Chunks = schneller, aber weniger präzise
- **Chunk-Überlappung**: Überlappung zwischen Chunks (50-200 empfohlen)
  - Verhindert Verlust von Sinneinheiten an Chunk-Grenzen
  - Zu groß = redundante Kodierungen

### Analyse-Modi

- **deductive**: Nur vordefinierte Kategorien verwenden
- **inductive**: Neue Kategorien aus dem Material entwickeln
- **abductive**: Bestehende Kategorien um Subkategorien erweitern
- **grounded**: Codes sammeln, später Hauptkategorien generieren

#### Grounded Mode – Besonderheiten

Im **Grounded Mode** ist im Codebook **ausschließlich die Forschungsfrage verpflichtend**.
Deduktive (vordefinierte) Kategorien sind hier **optional**, da das Hauptkategorien-System
während der Analyse **emergent** aus dem Material entsteht.

- **Codebook**: Forschungsfrage definieren – Kategorien können leer bleiben
- **Analyse-Ablauf** (3 Phasen):
  1. **Phase 1 – Subcode-Sammlung**: Aus den Textsegmenten werden inkrementell Subcodes gesammelt.
  2. **Phase 2 – Hauptkategorien-Generierung**: Aus den gesammelten Subcodes werden vom LLM Hauptkategorien gebildet.
  3. **Phase 3 – Kodierung**: Alle Segmente werden mit den generierten Hauptkategorien kodiert.

#### Maximale Subkategorien (nur Grounded Mode)

In der **Konfiguration** erscheint im Grounded Mode zusätzlich das Feld
**„Maximale Subkategorien je Hauptkategorie"** (Default: **5**, Bereich 1–50).

- Bestimmt, auf wie viele Subkategorien die in Phase 1 gesammelten Subcodes
  pro Hauptkategorie **verdichtet** werden.
- Niedrigere Werte (3–5) → fokussiertere, leichter handhabbare Kategorien
- Höhere Werte (8–15) → differenziertere, feinkörnigere Kategorien
- Wird über `MAX_SUBCATEGORIES` in der Config persistiert und in Phase 2 als
  Verdichtungs-Instruktion an das LLM übergeben.

### Review-Modi

- **auto**: Automatische Übernahme aller Kodierungen
- **manual**: Manuelle Überprüfung jeder Kodierung
- **consensus**: Nur übereinstimmende Kodierungen mehrerer Coder
- **majority**: Mehrheitsentscheidung bei Uneinigkeit

### Erweiterte Einstellungen

- **Batch-Größe**: Parallele API-Anfragen (5-10 empfohlen)
  - Höher = schneller, aber höhere Serverlast
- **Mit Kontext kodieren**: Berücksichtigt umgebenden Text
  - Verbessert Verständnis, erhöht aber API-Calls
- **Mehrfachkodierungen**: Erlaubt mehrere Kategorien pro Segment
  - Schwellwert: Mindest-Konfidenz für zusätzliche Kategorien (0.65-0.85)
- **Relevanz-Schwellwert**: Filtert irrelevante Segmente (0.3 = Standard)
  - Höher = strenger, weniger Segmente kodiert
  - Niedriger = inklusiver, mehr Segmente kodiert

### Coder-Einstellungen

- **Temperatur**: Kreativität des Coders (0.0-2.0)
  - 0.0-0.3 = deterministisch, konsistent
  - 0.5-0.7 = ausgewogen
  - 1.0+ = kreativ, variabel
- **Coder-ID**: Eindeutige Kennung (z.B. `auto_1`, `auto_2`)
- Mehrere Coder ermöglichen Reliabilitätsprüfung

### Attribute

- Zusätzliche Metadaten für Segmente (z.B. Quelle, Jahr, Typ)
- **Schlüssel**: Interner Name (z.B. `attribut1`)
- **Label**: Anzeigename (z.B. `Quelle`)

---

## 📚 Codebook

Das Codebook definiert Ihr Kategoriensystem für die qualitative Inhaltsanalyse.

### Forschungsfrage

- Zentrale Fragestellung Ihrer Analyse
- Leitet die Kategorienentwicklung und Kodierung
- Sollte präzise und fokussiert formuliert sein
- **Im Grounded Mode** ist die Forschungsfrage das **einzige Pflichtfeld** im Codebook
  (Kategorien entstehen emergent)

### Kodierregeln

Drei Typen von Regeln strukturieren die Kodierung:

- **Allgemeine Regeln**: Grundprinzipien für alle Kategorien
  - Beispiel: "Kodiere nur explizite Aussagen"
  - Beispiel: "Berücksichtige den Kontext"
- **Formatregeln**: Technische Vorgaben
  - Beispiel: "Markiere relevante Textstellen"
  - Beispiel: "Gib Konfidenzwert an"
- **Ausschlussregeln**: Was nicht kodiert werden soll
  - Beispiel: "Literaturverzeichnisse ignorieren"
  - Beispiel: "Methodische Infoboxen ausschließen"

### Deduktive Kategorien

Vordefinierte Kategorien basierend auf Theorie oder Forschungsstand.

#### Kategorie-Struktur

- **Name**: Eindeutiger Bezeichner (z.B. `Akteure`)
- **Definition**: Präzise Beschreibung des Kategorieinhalts
  - Was wird erfasst?
  - Welche Aspekte gehören dazu?
- **Regeln**: Spezifische Kodieranweisungen
  - Wann wird diese Kategorie zugeordnet?
  - Abgrenzung zu anderen Kategorien
- **Beispiele**: Konkrete Textbeispiele (2-4 empfohlen)
  - Typische Formulierungen
  - Grenzfälle zur Verdeutlichung

#### Subkategorien

- Differenzierung innerhalb einer Hauptkategorie
- **Name**: Bezeichner der Subkategorie
- **Beschreibung**: Kurze Erläuterung des Fokus
- Ermöglichen feinere Analyse und Auswertung

### Kategorie-Management

- **Hinzufügen**: Neue Kategorien mit vollständiger Struktur erstellen
- **Bearbeiten**: Definitionen, Regeln und Beispiele anpassen
- **Löschen**: Nicht benötigte Kategorien entfernen
- **Validierung**: System prüft Vollständigkeit automatisch
  - Warnung bei fehlenden Definitionen
  - Hinweis bei zu wenigen Beispielen

### Best Practices

- **Trennschärfe**: Kategorien sollten sich klar unterscheiden
- **Vollständigkeit**: Alle relevanten Aspekte abdecken
- **Beispiele**: Mindestens 2-3 pro Kategorie für gute Kodierung
- **Subkategorien**: Für differenzierte Analyse nutzen (2-5 pro Kategorie)
- **Iterativ**: Kategorien während Pilotierung verfeinern

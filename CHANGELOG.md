# Changelog

## Versionen und Updates

Hier ist eine **deutlich kompaktere, inhaltlich vollständige** Version des Changelogs, strukturiert auf Kernpunkte reduziert und Redundanzen entfernt:

---

## Neu in 0.12.5 (2026-01-27)

### 🐛 Bugfixes

* **Webapp Projektwechsel**: Behebt AttributeError beim Wechsel des Projektordners
  * `render_config_info()` lädt jetzt automatisch `config_data` falls nicht vorhanden
  * Session State Variablen werden korrekt initialisiert nach Projektwechsel
  * Kein manuelles Neuladen der Seite mehr erforderlich

* **Config Excel Loader**: Verbesserte Extraktion von CODER_SETTINGS und ATTRIBUTE_LABELS
  * Coder IDs werden jetzt korrekt aus Excel geladen (nicht nur Anzahl)
  * Attribut-Namen werden jetzt korrekt aus Excel geladen (nicht nur Anzahl)
  * Verschachtelte Config-Strukturen werden korrekt extrahiert
  * Erweiterte Debug-Ausgaben für bessere Fehlerdiagnose

* **Tests Module Import**: Behebt ModuleNotFoundError beim Import von `QCA_AID_assets.optimization.tests`
  * Import von `record_api_call` ist jetzt optional mit try-except
  * Dummy-Funktion wird verwendet wenn tests-Modul nicht verfügbar ist
  * Code funktioniert jetzt auch ohne tests-Verzeichnis
  * `.gitignore` angepasst um tests-Verzeichnis nicht mehr global auszuschließen
---

## Neu in 0.12.4 (2026-01-14)

### 🎨 Visualisierungen

* **Statische Visualisierungen**: Sunburst und Treemap jetzt mit matplotlib statt Plotly
  * Sunburst: Kreisförmige hierarchische Darstellung mit konzentrischen Ringen
  * Treemap: Rechteckige Darstellung mit squarify-Layout + detaillierte Subkategorie-Ansicht
  * Ausgabe als hochauflösende PNG-Dateien (300 DPI)
  * Keine weißen/leeren HTML-Dateien mehr
  * Fallback auf Balkendiagramm wenn squarify nicht installiert
  * Neue Abhängigkeit: squarify>=0.4.3

* **Anpassbare Parameter für Sunburst**: figure_size, dpi, font_size, title_font_size, max_label_length, ring_width, color_scheme, show_values, label_alpha, label_bg_color (HEX), label_bg_alpha (0-1)

* **Anpassbare Parameter für Treemap**: figure_size, detail_figure_height, dpi, font_size, detail_font_size, title_font_size, color_scheme, detail_color_scheme, show_values, alpha

### 🐛 Bugfixes

* **Explorer Output-Verzeichnis**: Verwendet jetzt korrekt das in der Config-UI festgelegte Output-Verzeichnis
  * Explorer base_config wird automatisch mit output_dir aus Hauptkonfiguration synchronisiert
  * Änderungen am Output-Verzeichnis in der Config-UI werden sofort an Explorer weitergegeben
  * Kategorie-Loader verwendet das konfigurierte output_dir statt hardkodiertem "output"
  * Analyseergebnisse werden im konfigurierten Verzeichnis gespeichert

* **Filter-Ausgabe**: Spalten-Mapping wird nur noch angezeigt wenn generische Attribute verwendet werden
  * Reduziert unnötige Konsolenausgaben bei Visualisierungen
  * Mapping erscheint nur noch bei tatsächlicher Verwendung von Attribut_1, Attribut_2, etc.

* **Sunburst**: Schutz gegen Rekursionsfehler
  * Erkennt und verhindert echte zirkuläre Referenzen (nur im aktuellen Pfad)
  * Erlaubt denselben Knoten in verschiedenen Zweigen der Hierarchie
  * Maximale Hierarchietiefe von 10 Ebenen
  * Besseres Error-Handling mit aussagekräftigen Fehlermeldungen


### 🎨 UI-Verbesserungen

* **Explorer UI**: Parameter-Einstellungen für Sunburst und Treemap
  * Anpassbare Auflösung (DPI)
  * Schriftgrößen für Labels und Titel
  * Farbschemata (Set3, Pastel1, etc.)
  * Ring-Breite (Sunburst) und Detail-Höhe (Treemap)
  * Werte ein/ausblenden
  * Maximale Label-Länge (Sunburst)
  * **Label-Hintergrund (Sunburst)**: Farbe (HEX) und Transparenz (0-1)
  * Alle Parameter in ausklappbarem "Erweiterte Parameter" Bereich

* **Explorer UI**: Button "📂 Ordner öffnen" hinzugefügt
  * Nach erfolgreicher Analyse wird neben dem Ausgabepfad ein Button angezeigt
  * In der Dateiliste gibt es Buttons zum Öffnen des Ordners und Kopieren des Pfads
  * Im Explorer-View (Ergebnisansicht) wird eine Erfolgsmeldung mit Pfad und Button angezeigt
  * Plattformspezifische Implementierung (Windows/macOS/Linux)

* **Explorer UI**: Projekt-Verzeichnis-Wechsel verbessert
  * Beim Ändern des Projekt-Verzeichnisses werden jetzt alle Manager (Config, Codebook, Explorer) neu initialisiert
  * Explorer-UI zeigt nach Verzeichniswechsel den korrekten Pfad an
  * Eingabedateien werden im neuen Projekt-Verzeichnis korrekt erkannt

* **Explorer UI**: Analyse-Tabs optimiert
  * Button "Nach oben" entfernt (vereinfachtes Layout)
  * Tab-Häkchen (✅/⏸️) aktualisiert sich sofort beim Aktivieren/Deaktivieren einer Analyse
  * Bessere visuelle Rückmeldung beim Ändern des Analyse-Status

* **Explorer UI**: Filter-Verbesserungen
  * "nicht kodiert" wird automatisch aus allen Filter-Optionen ausgeschlossen
  * Betrifft Hauptkategorien, Subkategorien und alle Subkategorien-Listen
  * Case-insensitive Filterung (auch "Nicht Kodiert", "NICHT KODIERT" etc.)

* **Explorer UI**: Optionaler Ausschluss von "Nicht kodiert" in Analysen
  * Neue Checkbox "❌ 'Nicht kodiert' ausschließen" in allen Analyse-Parametern
  * **Visualisierungen** (Netzwerk, Heatmap, Sunburst, Treemap): Standardmäßig aktiviert (True)
  * **LLM-Analysen** (Summary/Paraphrase): Standardmäßig deaktiviert (False) - nicht-kodierte Texte können relevant sein
  * Benutzer können das Verhalten für jede Analyse individuell anpassen
  * Automatische Filterung von "Nicht kodiert" und "Kein Kodierkonsens" (case-insensitive)
  * Hinweismeldung zeigt an, wie viele Einträge ausgeschlossen wurden

* **Explorer-View**: Konfidenz-Verteilung verbessert
  * Lesbare Bin-Labels (z.B. "0.80-0.85" statt kryptischer Intervall-Notation)
  * Zusätzliche Statistiken: Durchschnitt, Median, Standardabweichung
  * Erklärungstext zur Bedeutung der Konfidenzwerte
  * Bessere Visualisierung mit 20 gleichmäßigen Bins von 0 bis 1

* **Explorer-View**: Scroll-Position optimiert
  * Ergebnisseite startet jetzt oben beim Titel
  * Dataframe und Charts in ausklappbaren Expandern (standardmäßig eingeklappt)
  * Analyse-Log in Expander verschoben
  * Statistiken prominent oben platziert

* **Filter-Dropdowns erweitert**
  * Dokument-Filter mit tatsächlichen Werten aus Excel
  * Attribut 1 & 2 Filter mit dynamischen Werten
  * Labels werden aus Konfiguration-Sheet gelesen (ATTRIBUT1_LABEL, ATTRIBUT2_LABEL)
  * Fallback auf Texteingabe wenn Werte nicht verfügbar

* **Heatmap-Parameter verbessert**
  * X/Y/Z-Achsen jetzt mit Dropdowns (verfügbare Spalten aus Excel)
  * Zahlenformat-Dropdown mit 8 vordefinierten Formaten
  * Expander mit 6 empfohlenen Heatmap-Analysen für QCA-Daten

* **Netzwerkanalyse-Parameter**
  * Tooltips mit größenabhängigen Empfehlungen (klein/mittel/groß)
  * Abschätzung von Knoten und Kanten basierend auf gefilterten Daten
  * Spezifische Empfehlungen basierend auf geschätzter Netzwerkgröße

* **Analysename vorausgefüllt**
  * Standardnamen basierend auf Analysetyp (z.B. "Netzwerkanalyse", "Heatmap-Analyse")
  * Gilt für alle Analysetypen inkl. Sunburst und Treemap

* **Tooltips verbessert**
  * Keyword-Harmonisierung: Detaillierte Erklärung mit Beispielen
  * Ähnlichkeitsschwelle: Konkrete Empfehlungen für verschiedene Szenarien

### 🐛 Bugfixes

* **Explorer UI**: Kategorienprüfung korrigiert
  * Kategorien werden jetzt aus der ausgewählten Analysedatei geladen
  * Bessere Fehlerbehandlung und Debugging-Ausgaben
  * Warnung wenn Kategorien-Sheet leer ist oder nicht gefunden wird

* **JSON-Konfiguration**: Laden/Speichern repariert
  * Entfernte fehlerhafte Prüfung die neue Konfigurationen ablehnte
  * `enabled_charts` und `color_scheme` sind Teil des neuen Formats
  * Konfigurationen können jetzt erfolgreich gespeichert und geladen werden

* **Sunburst & Treemap**: Datenvisualisierung korrigiert
  * Hierarchische Struktur wird jetzt korrekt aufgebaut
  * Werte werden während der Strukturerstellung gezählt (nicht nachträglich)
  * Duplikate werden vermieden durch Verwendung von Tupel-Keys
  * HTML-Ausgabe zeigt jetzt Daten korrekt an

### 🔧 Technische Verbesserungen

* **Versionsverwaltung konsolidiert**
  * `__version__.py` ist jetzt die einzige Quelle für Versionsnummern
  * `__init__.py` importiert aus `__version__.py`
  * Keine Duplikation mehr

* **CategoryLoader erweitert**
  * Lädt Werte aus Kodierungsergebnisse-Sheet
  * Liest Attribut-Labels aus Konfiguration-Sheet
  * Verwendet dynamische Spaltennamen basierend auf Labels

---

## Neu in 0.12.3 (2026-01-13)

### 🔧 Kritische Bugfixes (Mehrfachkodierung & Export)

* **API-Fehler behoben**: Vereinheitlichte Analyse über `UnifiedAnalyzer.analyze_batch()`, entfernte nicht-existente Methoden/Klassen, robuste Fallbacks.
* **Originaltext vollständig repariert**: Zuverlässige Text-Extraktion aus `result.text` inkl. Mehrfachkodierung, mehrstufige Fallbacks & Debugging.
* **Relevanzdaten im Export korrigiert**: Relevanz-Details werden korrekt gespeichert und gemappt (keine `N/A` mehr).
* **Begründungen für Nicht-Relevanz**: Segment-ID-Normalisierung behebt fehlende Justifications.
* **Intercoder-Reliabilität korrekt**: Vergleich nur zwischen Kodierern (Set-basiert), konsistente Alpha-Werte.
* **Datei- & Pfadprobleme behoben**: Einheitliche Zeitstempel/Namensschemata, korrektes Output-Verzeichnis, Indentation Error gefixt.

### 🚀 Mehrfachkodierung: neue Architektur

* **Batch-basierte Mehrfachkodierung** (2-phasig):

  1. Basiskodierung aller Segmente
  2. Fokussierte Batches je zusätzlicher Kategorie
* **Fokussierte API-Calls pro Kategorie** → korrekte Subkategorien, methodisch sauber (kein Cross-Mapping).
* **Alle Modi unterstützt**: deductive, inductive, abductive, grounded.
* **Sortierte Ergebnisse**: konsistente Reihenfolge inkl. Mehrfachkodierungs-Suffixe.
* **Realistischer Schwellenwert**: 0.85 → **0.65**, score-basiert und konsistent über alle Modi.

### ⚙️ User-Config & Methodik

* **User-Config vollständig integriert**: Forschungsfrage, Regeln & Kategorien priorisiert korrekt.
* **Modus-spezifisches Verhalten**:

  * Deduktiv → User-Kategorien
  * Induktiv/Grounded → emergente Codes (keine Vorgaben)
* **Grounded Theory korrekt umgesetzt** (keine vorgefassten Kategorien).

### 📊 Relevanzbewertung & Tracking (erweitert)

* **Neue Excel-Felder**: Relevanz-Stärke, Klassifikationskonfidenz, Forschungsaspekte, Relevanz-Aspekte.
* **Methodisch fundiert** (Mayring / Kuckartz / Gläser-Laudel).
* **Keine unnötigen Relevanz-Checks** nach Review → weniger API-Calls.

### 🧠 Kontext & Paraphrasen

* **Progressive Kontext-Paraphrasen** via Callback-System (batch-weise, dokument-spezifisch).
* **Doppelte Sammlung entfernt** → effizienter, keine redundanten Calls.

### 🛠️ Sonstiges

* **Optimiertes Logging**: klarer, kategorienbasiert, weniger Rauschen.
* **Dokumentation erweitert**: Methodische Grundlagen der Relevanzbestimmung ergänzt.



### Neu in 0.12.2  (2025-01-12)

BUGFIX: RELEVANZ-BEWERTUNG INKONSISTENZEN BEHOBEN
- **LLM-Prompt Klarstellung**: Explizite Konsistenz-Regel zwischen `is_relevant` und `reasoning` hinzugefügt
- **Intelligente Inkonsistenz-Erkennung**: Automatische Warnung bei widersprüchlichen LLM-Antworten
- **Verbesserte Begründungen**: Klarere Anweisungen für Relevanz-Begründungen (warum relevant/nicht relevant)
- **Doppelte Feldunterstützung**: Unterstützt sowohl `justification` als auch `reasoning` Felder aus LLM-Antworten

VERBESSERUNG: RELEVANZ-SCHWELLWERT SYSTEM
- **Neuer Standard-Wert**: `RELEVANCE_THRESHOLD` von 0.0 auf 0.3 geändert (entspricht LLM-Verhalten)
- **Intelligente Filterlogik**: 
  - Werte < 0.3: Inkludiert auch LLM-verworfene Segmente basierend auf Confidence
  - Werte ≥ 0.3: Standard LLM-Entscheidungen mit zusätzlicher Confidence-Filterung
- **Verbesserte UI-Erklärung**: Klare Beschreibung der drei Modi in der Konfiguration
- **Warnungen**: Automatische Hinweise bei niedrigen/hohen Schwellwerten

VERBESSERUNG: LOGGING OPTIMIERUNG
- **Vereinfachtes Relevanz-Logging**: Alle Segmente in einer übersichtlichen Liste statt doppelter Ausgabe
- **Klareres Review-Logging**: "Vervollständige Export" statt verwirrende "Relevanzprüfung für fehlende Segmente"
- **Entfernte Redundanz**: Keine doppelte Auflistung relevanter Segmente mehr
- **Fokussierte Ausgabe**: Nur wesentliche Informationen, weniger technische Details

CODE CLEANUP: UNGENUTZTE KOMPONENTEN ENTFERNT
- **Datei entfernt**: `fluent_components.py` (300+ Zeilen ungenutzter Code)
- **Bereinigte Imports**: Entfernung aller ungenutzten Fluent-Komponenten aus `__init__.py`
- **Reduzierte Komplexität**: Nur tatsächlich verwendete Fluent UI Teile bleiben erhalten
- **Sauberer Code**: Keine toten Code-Pfade mehr im webapp_components Modul

PERFORMANCE: WEBAPP STARTUP-OPTIMIERUNG
- **50-60% schnellerer Startup**: Von ~5-8s auf ~2-3s reduziert
- **Lazy Loading System**: Tab-Daten werden erst bei Bedarf geladen
- **Cached Icon Loading**: `@st.cache_data` für optimierte Icon-Darstellung
- **Minimale Session State**: Nur kritische Manager beim Start initialisiert
- **CSS Caching**: Einmaliges Laden der Fluent UI Styles
- **Performance Monitor**: Debug-Tool zur Messung der Startup-Zeiten in Sidebar

BUGFIX: FEHLENDER RELEVANCE_THRESHOLD IN CONFIG UI
- **Problem behoben**: `relevance_threshold` Parameter war im Model definiert aber nicht in der UI verfügbar
- **Neuer Slider**: "🎯 Relevanz-Schwellwert" in Analyse-Einstellungen hinzugefügt
- **Bereich**: 0.0 bis 1.0 in 0.05er Schritten konfigurierbar
- **Standard**: 0.0 (alle vom LLM als relevant markierten Segmente)
- **Dynamische Info**: Zeigt aktuelle Filterung basierend auf Schwellwert an
- **Vollständige Integration**: Korrekte Speicherung und Laden in JSON/XLSX

VERBESSERUNG: MEMORY MANAGEMENT
- **Reduzierter Memory-Footprint**: Nur aktive Tab-Daten im Speicher
- **Manager-Singleton Pattern**: Verhindert redundante Initialisierung
- **Optimierte Import-Strategie**: Lazy Loading für große Komponenten
- **Graceful Fallbacks**: Robuste Fehlerbehandlung bei Icon-Loading

### Neu in 0.12.1 (2025-01-08)

BUGFIX: KATEGORIEPRÄFERENZEN IM DEDUKTIVEN MODUS
- Robuste Typerkennung für verschiedene Kategorie-Datenstrukturen implementiert
- CategoryDefinition-Objekte (abduktive Analyse) korrekt verarbeitet
- Dictionary-Format (serialisiert) unterstützt
- String-Format (deduktive Analyse) korrekt behandelt
- "Keine starken Kategoriepräferenzen" Fehler im deduktiven Modus behoben
- Kategoriepräferenzen funktionieren jetzt in beiden Modi identisch

BUGFIX: ROBUSTE SPEICHERUNG FÜR CLOUD-SYNC
- Blockierende Speicherung verhindert Datenverlust bei Cloud-Sync-Problemen
- Erweiterte Retry-Logik (10 Versuche, bis 30s Wartezeit) für Windows/Dropbox
- Automatische Cloud-Prozess-Erkennung (Dropbox, OneDrive, etc.)
- Intelligente Benutzerwarnung bei Speicherproblemen mit Lösungsvorschlägen
- Proaktive Warnung beim Analysestart über mögliche Cloud-Sync-Konflikte
- Analyse stoppt automatisch bei dauerhaften Speicherproblemen

PERFORMANCE: BATCH-SPEICHERUNG FÜR CLOUD-SPEICHER
- **99.7% Reduktion der Festplattenzugriffe**: Statt einzelner Speichervorgänge nur noch 1 Batch-Speicherung
- **Keine Dateisperren mehr**: Kodierungen werden im Arbeitsspeicher gesammelt und erst am Ende gespeichert
- **Deutlich schnellere Analyse**: Besonders bei Cloud-Speichern (Dropbox, OneDrive) keine Wartezeiten mehr
- **Robuste Fehlerbehandlung**: Kodierungen werden auch bei Programmabbruch oder Fehlern gespeichert
- **Atomare Speicherung**: Alle Kodierungen werden zusammen gespeichert oder gar nicht
- **Rückwärtskompatibel**: Bestehende APIs und Konfigurationen bleiben unverändert

BUGFIX: VERLORENE KODIERUNGEN DURCH FALSCHEN RELEVANZ-THRESHOLD
- **Problem behoben**:  relevante Segmente wurden fälschlicherweise reduziert
- **Ursache**: Zusätzliche Threshold-Filterung nach bereits erfolgter Relevanzprüfung
- **Lösung**: Korrekte LLM-basierte Relevanzentscheidung implementiert
- **Neue Logik**: Nur Segmente mit `is_relevant: true` vom LLM werden kodiert
- **Konfigurierbar**: Neuer Parameter `RELEVANCE_THRESHOLD` (Standard: 0.0) für Mindest-Konfidenz
- **Verbesserte Logs**: "XX Segmente vom LLM als relevant identifiziert, darunter XX mit hinreichender Konfidenz"
- **Vollständige Analyse**: Keine verlorenen Kodierungen mehr bei optimierter Analyse

BUGFIX: FEHLENDE CATEGORYLOADER MODULE
- Fehlende `category_loader.py` Datei in `QCA_AID_assets/utils/config/` hinzugefügt
- Import-Fehler beim Start der Webapp behoben
- CategoryLoader-Funktionalität für Explorer-Interface wiederhergestellt

VERBESSERUNG: CONFIG UI DATEIAUSWAHL
- Dateiauswahl funktioniert jetzt mit benutzerdefinierten Dateinamen (z.B. "01_QCA-AID-Codebook.xlsx")
- Automatische Formaterkennung - Radio Buttons werden ausgeblendet wenn Format erkannt wird
- Korrekte Anzeige des ausgewählten Dateipfads im Textfeld
- Verbesserte Session State Synchronisation für File Browser

BUGFIX: CODEBOOK LADEN MIT BENUTZERDEFINIERTEN NAMEN
- ConfigLoader unterstützt jetzt Codebook-Dateien mit beliebigen Namen
- Korrekte Extraktion der CONFIG-Sheet Einstellungen
- Verbesserte Datentyp-Konvertierung (Boolean, Integer, Float)
- Priorität für aktuelle Werte aus CONFIG-Sheet vor Standard-Werten

UI VERBESSERUNG: ANALYSE BUTTONS
- "Neue Analyse" Button umbenannt zu "Erneut versuchen" für bessere Klarheit
- Intuitivere Benutzerführung bei Analyse-Fehlern und Abbrüchen

BUGFIX: DATEI-OPERATIONEN ROBUSTHEIT
- Retry-Mechanismus für Datei-Ersetzungs-Operationen implementiert
- Exponential Backoff mit Jitter löst Windows/Dropbox Sperr-Probleme
- Bis zu 5 automatische Wiederholungsversuche bei PermissionError
- Detailliertes Logging für bessere Fehlerdiagnose

VERBESSERUNG: BENUTZERDEFINIERTE CODEBOOK-NAMEN
- JSON-Dateien erhalten automatisch passende Namen zu XLSX-Dateien
- "01_QCA-AID-Codebook.xlsx" → "01_QCA-AID-Codebook.json"
- Session State Tracking des aktuell geladenen Dateinamens
- Konsistente Dateinamen für bessere Dateiverwaltung

### Neu in 0.12.0 (2025-12-20)

DYNAMIC CACHE SYSTEM: INTELLIGENTE MULTI-CODER CACHE-OPTIMIERUNG

**🚀 Besseres Cache-System für Multi-Coder Analysen:**
- **Deutliche API-Call Reduktion** bei Multi-Coder Szenarien durch intelligente Operation-Klassifikation
- **Shared vs. Coder-Specific Caching**: Relevanzprüfung und Kategorienentwicklung werden zwischen Kodierern geteilt, Kodierung bleibt kodierer-spezifisch
- **Methodologische Integrität**: Respektiert QCA-Prinzipien für alle Analysemodi (induktiv, abduktiv, grounded, deduktiv)
- **Automatische Strategie-Auswahl**: Wechselt dynamisch zwischen Single-Coder und Multi-Coder Strategien basierend auf Konfiguration

**💡 Intercoder-Reliabilität Integration:**
- **Automatische Datensammlung**: Alle Kodierungen (automatisch und manuell) werden für Reliabilitätsanalyse gespeichert
- **Manual-Coder Isolation**: Manuelle Kodierungen sind vollständig von automatischen Caches isoliert
- **Persistente Speicherung**: JSON-Datenbank für dauerhafte Reliabilitätsdaten
- **Built-in Reliabilitäts-Tests**: Integrierte Tests für Manual + Auto-Coder Kombinationen

**🔧 Weitere Features:**
- Cache-Plugin-System: Erweiterbare Cache-Strategien für spezielle Anwendungsfälle
- Debug-Tools für Cache: Umfassende Logging-Level, Cache-Dumps, Performance-Benchmarks
- Bessere Token-Stats und API-Call-Berechnung
- Smarte Filter in der Explorer Analyse fertiger Kodierungen
- Deutlich aufgeräumteres und informativeres Log-File


**📊 Cache-Strategien nach Analysemodus:**
- **Induktiv**: Kategorienentwicklung shared, Kodierung per-coder
- **Abduktiv**: Subkategorien-Entwicklung shared, Kodierung per-coder  
- **Grounded**: Subcode-Sammlung shared, keine Kodierung in Phase 1
- **Deduktiv**: Relevanzprüfung shared, Kodierung per-coder

**🛠️ Technische Implementierung:**
- `DynamicCacheManager`: Zentrale Cache-Verwaltung mit automatischer Strategie-Auswahl
- `CacheStrategy Pattern`: Modulare Strategien für verschiedene Szenarien
- `ReliabilityDatabase`: Persistente Speicherung aller Kodierungsergebnisse
- `Cache Migration System`: Automatische Migration bei Strategie-Wechseln
- `Performance Benchmarking`: Integrierte Tools für Leistungsmessung

**Legacy Features (weiterhin verfügbar):**
- Standard-Analyse nach bisherigem Verfahren
- dafür in der Config manuell "ENABLE_OPTIMIZATION" auf "False" setzen.
- falls es Problem mit der neuen optimierten Variante gibt

### Neu in 0.11.2 (2025-12-05)

**Bugfixes:**
- 🐛 **Kontext-Paraphrasen Log**: Klarere Log-Ausgabe für CODE_WITH_CONTEXT
  - Zeigt jetzt explizit, dass nur die letzten N Paraphrasen als Kontext verwendet werden
  - Verhindert Missverständnisse über die tatsächlich genutzte Kontextmenge
- 🐛 **Projekt- vs. Programmverzeichnis**: Korrekte Trennung zwischen Benutzer-Projektverzeichnis und QCA-AID-Installationsverzeichnis
  - Preisinformationen werden jetzt korrekt aus dem Installationsverzeichnis geladen
  - Eingabe- und Ausgabeverzeichnisse werden relativ zum Projektverzeichnis validiert
- 🐛 **Preisauskunft**: Token-Kosten werden jetzt korrekt für alle Provider angezeigt
  - Fuzzy-Matching für OpenRouter-Modelle (z.B. deepseek/deepseek-v3.2)
  - Korrekte Pfadauflösung zu den Preis-Konfigurationsdateien
  - Dezente Anzeige als Caption statt Info-Box
- 🐛 **Kategorie-Erstellung**: Fehler beim HinzuFügen neuer Kategorien behoben
  - Import-Konflikt bei CategoryData-Klasse gelöst
  - Automatische Konvertierung von Dictionary zu CategoryData-Instanzen
  - Typ-Check verwendet jetzt Klassennamen statt isinstance()
- 🐛 **Subcategories**: Korrekte Struktur für Subkategorien
  - Format geändert von "Schlüssel:Bezeichnung" zu "Name:Definition"
  - Definition ist jetzt optional
  - Unterstützung für Subkategorien ohne Definition

**Verbesserungen:**
- ✨ **Vereinfachtes Speichern**: Codebook-Tab speichert jetzt direkt in die aktuelle Datei
  - Kein zusätzlicher Datei-Dialog mehr im Codebook-Tab
  - Dateiauswahl erfolgt nur noch im Config-Tab
  - Ein-Klick-Speichern für schnelleren Workflow
- ✨ **Neue Visualisierungen**: Sunburst und Treemap für hierarchische Daten
  - Interaktive HTML-Visualisierungen
  - Zwei Versionen: Standard und mit Werten in Beschriftungen
  - Automatische Keyword-Harmonisierung
- ✨ **Verbesserte Verzeichnisvalidierung**: 
  - Visuelle Rückmeldung (✅/⚠️/❌) für Verzeichnispfade
  - Anzeige des vollständigen aufgelösten Pfads
  - Unterstützung für relative und absolute Pfade

### Neu in 0.11.1 (2025-12-01)

**Bugfixes:**
- 🐛 **Setup.bat**: Desktop-Icon wird nun korrekt erstellt
- 🐛 **Local LLM**: Response-Format wird jetzt korrekt erkannt
- 🐛 **TokenTracker**: Kostenberechnung wurde korrigiert (Preise waren um Faktor 10 zu hoch)
- 🐛 **Projektordner**: Manuell gesetzter Projektordner wird nun korrekt in der Analyse übernommen (nicht nur in der App)

**Verbesserungen:**
- ✨ **Automatisches Config-Update**: LLM-Provider-Configs werden automatisch aktualisiert, wenn sie älter als 7 Tage sind
  - Neue Modelle werden automatisch erkannt
  - Preise bleiben aktuell
  - Fallback auf lokale Configs bei Netzwerkproblemen

### Neu in 0.11.0 (2025-11-30)

QCA-AID WEBAPP: VOLLSTÄNDIGE WEBBASIERTE BENUTZEROBERFLÄCHE

**WICHTIG: Lokale Modelle für Datenschutz**
- ✨ **Vollständige Integration lokaler LLM-Modelle**
  - LM Studio und Ollama Unterstützung in der Webapp
  - Automatische Erkennung laufender lokaler Server
  - 100% Datenschutz - Alle Daten bleiben auf Ihrem Computer
  - Kostenlos - Keine API-Gebühren
  - DSGVO-konform - Ideal für sensible Forschungsdaten
  - Einfache Bedienung: "Local (LM Studio/Ollama)" auswählen und auf "Erkennen" klicken
  - Siehe [LOCAL_MODELS_GUIDE.md](LOCAL_MODELS_GUIDE.md) für detaillierte Anleitung

Webapp-Features:
- ✨ **Vollständige Weboberfläche** für QCA-AID
  - Intuitive grafische Benutzeroberfläche für alle Funktionen
  - Keine Kommandozeilen-Kenntnisse erforderlich
  - Lokale Ausführung - alle Daten bleiben auf Ihrem Computer
  - Streamlit-basierte moderne Web-UI
- ✨ **Grafischer Konfigurationseditor**
  - Visuelle Bearbeitung aller CONFIG-Parameter
  - Dropdown-Menüs für Modellauswahl mit Live-Updates
  - Inline-Validierung mit sofortigen Fehlermeldungen
  - Automatische Synchronisation mit Excel/JSON-Codebook
- ✨ **Visueller Codebook-Editor**
  - Strukturierte Bearbeitung von Kategorien und Subkategorien
  - Drag-and-Drop für Beispiele und Regeln
  - Live-Vorschau der Kategorienhierarchie
  - Import/Export von Kategoriensystemen
- ✨ **Integrierte Analyse-Steuerung**
  - Analysen direkt aus der Webapp starten
  - Echtzeit-Fortschrittsanzeige mit Prozentangaben
  - Live-Log-Ausgabe während der Analyse
  - Abbruch-Funktion für laufende Analysen
- ✨ **Dateimanagement**
  - Übersicht aller Input-Dateien mit Metadaten
  - Upload-Funktion für neue Dokumente
  - Vorschau von Textinhalten
  - Batch-Upload für mehrere Dateien
- ✨ **Ergebnisvisualisierung**
  - Interaktive Tabellen mit Kodierungsergebnissen
  - Filterfunktionen nach Kategorien und Attributen
  - Export-Funktionen für verschiedene Formate
  - Statistik-Dashboard mit Diagrammen
- ✨ **Explorer-Integration**
  - QCA-AID-Explorer direkt in der Webapp
  - Konfiguration von Analysetypen über GUI
  - Visualisierungen (Netzwerk, Heatmap, Sentiment)
  - Export von Explorer-Ergebnissen

Technische Verbesserungen:
- ✨ **Modulare Webapp-Architektur**
  - Komponenten-basierte Struktur in `QCA_AID_app/`
  - Wiederverwendbare UI-Komponenten
  - Klare Trennung von UI und Logik
  - Erweiterbar für neue Features
- ✨ **Session-Management**
  - Persistente Einstellungen über Sessions
  - Automatische Wiederherstellung bei Neustart
  - Multi-User-fähig (verschiedene Browser-Tabs)
- ✨ **Robuste Fehlerbehandlung**
  - Benutzerfreundliche Fehlermeldungen
  - Automatische Wiederherstellung bei Problemen
  - Detaillierte Logs für Debugging
- ✨ **Performance-Optimierung**
  - Caching für schnellere Ladezeiten
  - Asynchrone Verarbeitung für UI-Responsiveness
  - Effiziente Datenübertragung

Benutzerfreundlichkeit:
- 📚 **Beispielkonfigurationen**
  - Vorkonfigurierte Templates in `QCA_AID_assets/examples/`
  - Best-Practice-Beispiele für verschiedene Szenarien
  - Schritt-für-Schritt-Tutorials
- 🚀 **Ein-Klick-Setup**
  - Windows: `setup.bat`
  - Richtet Python und benötigte Pakete ein
  - Erstellt Desktop Icon
- ✨ **Modellkosten-Anzeige**
  - Dezente Anzeige der Input/Output-Token-Kosten bei Modellauswahl
  - Automatische Anzeige für alle kommerziellen Modelle
  - "Kostenlos"-Hinweis für lokale Modelle
  - Hilft bei kostenbasierter Modellauswahl

Datenschutz und Sicherheit:
- 🔒 **Lokale Modelle für maximalen Datenschutz**
  - Vollständige Integration von LM Studio und Ollama
  - Keine Datenübermittlung an externe Server
  - DSGVO-konform für sensible Forschungsdaten
  - Automatische Erkennung und Filterung von Chat-Modellen
  - Embedding-Modelle werden automatisch ausgeblendet

Bugfixes:
- 🐛 Console-Logging verbessert
  - Line-Buffering für vollständige Log-Erfassung
  - Korrekte Zeitstempel für alle Ausgaben
  - Keine verlorenen Log-Einträge mehr
  - Robuste Flush-Mechanismen
- 🐛 Doppelte Kostenanzeige bei lokalen Modellen behoben
- 🐛 LaTeX-Rendering von Dollar-Zeichen in Preisanzeige behoben

Code Quality:
- 📦 Neue Module: `webapp.py`, `start_webapp.py`, `webapp_components/`, `webapp_logic/`, `webapp_models/`
- Vollständige Integration mit bestehendem QCA-AID-System
- 📚 Umfassende Inline-Dokumentation
- Keine Breaking Changes - CLI bleibt vollständig funktional

### Neu in 0.10.4 (2025-11-30)

ERWEITERTE LLM-PROVIDER-UNTERSTÜTZUNG

Multi-Provider-System:
- ✨ Unterstützung für mehrere LLM-Provider
  - **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo und weitere Modelle
  - **Anthropic**: Claude Sonnet 4.5, Claude 3.5 Sonnet, Claude 3 Opus
  - **Mistral**: Mistral Large, Mistral Medium, Mistral Small
  - **OpenRouter**: Zugriff auf Modelle verschiedener Anbieter über eine API
  - **Lokale Modelle**: LM Studio und Ollama Integration
- ✨ Dynamisches Modell-Management
  - Automatisches Laden von Modell-Metadaten von GitHub (Catwalk)
  - Lokale Fallback-Konfigurationen für Offline-Betrieb
  - 24-Stunden Cache für schnellere Ladezeiten
  - Einheitliches Format für alle Provider (Normalisierung)
- ✨ Erweiterte Modell-Informationen
  - Context Window (Token-Limits)
  - Kosten pro 1M Input/Output-Tokens
  - Modell-Capabilities (Reasoning, Attachments, etc.)
  - Anpassbare Pricing-Overrides via `pricing_overrides.json`

Webapp-Integration:
- ✨ Dynamische Modellauswahl in der Webapp
  - Dropdown-Menüs zeigen alle verfügbaren Provider
  - Modellauswahl passt sich automatisch an gewählten Provider an
  - Anzeige aktueller Modelle aus allen Providern
  - Nahtlose Integration in bestehende Konfiguration

Technische Verbesserungen:
- ✨ LLMProviderManager für zentrale Verwaltung
  - Automatische Provider-Erkennung und -Initialisierung
  - Filter-Funktionen (nach Provider, Kosten, Context Window)
  - Robuste Fehlerbehandlung mit Fallback-Mechanismen
  - Erweiterbar für neue Provider ohne Code-Änderungen
- ✨ Lokale Modell-Erkennung
  - Automatische Erkennung von LM Studio (Port 1234)
  - Automatische Erkennung von Ollama (Port 11434)
  - Graceful Degradation wenn lokale Server offline sind

API-Key-Verwaltung:
- ℹ️ API-Keys werden über Umgebungsvariablen verwaltet
  - `OPENAI_API_KEY` für OpenAI-Modelle
  - `ANTHROPIC_API_KEY` für Anthropic-Modelle
  - `MISTRAL_API_KEY` für Mistral-Modelle
  - `OPENROUTER_API_KEY` für OpenRouter-Modelle
- ℹ️ Empfohlene Speicherung in `.env` Datei im Projektverzeichnis
- ℹ️ Siehe README.md für detaillierte Anleitung

### Neu in 0.10.3 (2025-11-28)

QCA-AID JSON-KONFIGURATION: VOLLSTÄNDIGE INTEGRATION

JSON-Konfigurationsunterstützung:
- ✨ Vollständige JSON-Unterstützung für QCA-AID-Codebook
  - Neue Datei `QCA-AID-Codebook.json` als alternatives Konfigurationsformat
  - Excel-Konfiguration (`QCA-AID-Codebook.xlsx`) weiterhin vollständig unterstützt
  - Automatische bidirektionale Synchronisation zwischen Excel und JSON
  - Intelligente Dateierkennung: System wählt automatisch neuere Datei basierend auf Zeitstempel
  - Automatische Erstellung fehlender Dateien (JSON oder Excel) beim ersten Start
- ✨ Round-Trip Konvertierung ohne Datenverlust
  - Vollständige Übertragung aller Elemente: Forschungsfrage, Kodierregeln, Kategorien, CONFIG
  - Erhalt aller Datentypen (Boolean, Integer, Float, String, Listen, Dictionaries)
  - Korrekte Verarbeitung verschachtelter Strukturen (CODER_SETTINGS, ATTRIBUTE_LABELS)
  - Hierarchische Kategorien mit Definition, Regeln, Beispielen und Unterkategorien
- ✨ UTF-8 Encoding und Formatierung
  - Korrekte Darstellung deutscher Umlaute (ä, ö, ü, ß)
  - Menschenlesbare JSON-Struktur mit 2-Leerzeichen-Einrückung
  - ensure_ascii=False für native Unicode-Zeichen
  - Logische Struktur mit klar benannten Schlüsseln

Validierung und Fehlerbehandlung:
- ✨ Umfassende numerische Parametervalidierung
  - CHUNK_SIZE: Prüfung >= 1, automatische Standardwerte bei ungültigen Werten
  - CHUNK_OVERLAP: Prüfung < CHUNK_SIZE, automatische Korrektur bei Konflikten
  - BATCH_SIZE: Prüfung zwischen 1-20, Warnung bei Performance-kritischen Werten
  - Float-Thresholds: Validierung zwischen 0.0-1.0 für alle Schwellenwerte
  - Detaillierte Warnmeldungen mit Standardwerten bei Validierungsfehlern
- ✨ Enum-Parametervalidierung
  - ANALYSIS_MODE: Strikte Validierung gegen {full, abductive, deductive, inductive, grounded}
  - REVIEW_MODE: Strikte Validierung gegen {auto, manual, consensus, majority}
  - Automatische Fallback-Werte bei ungültigen Eingaben
  - Klare Fehlermeldungen mit Liste gültiger Werte
- ✨ Intelligente Pfadverwaltung
  - Automatische Unterscheidung zwischen relativen und absoluten Pfaden
  - Relative Pfade werden relativ zum Projektverzeichnis aufgelöst
  - Absolute Pfade werden direkt verwendet
  - Automatische Erstellung nicht-existierender Verzeichnisse
  - Robuste Fehlerbehandlung bei Pfadproblemen
- ✨ Robuste Fehlerbehandlung
  - Graceful Fallback bei Synchronisationsfehlern
  - Detaillierte Fehlermeldungen bei ungültigen Konfigurationen
  - Automatische Verwendung von Standardwerten bei fehlenden Parametern
  - Warnung bei Encoding-Problemen mit automatischer Korrektur

Dokumentation und Beispiele:
- 📚 Vollständige Beispiel-JSON-Datei (`QCA-AID-Codebook-Example.json`)
- 📚 Detaillierte Dokumentation (`QCA-AID-Codebook-Example-Documentation.md`)
- 📚 Migration Guide (`MIGRATION_GUIDE.md`) mit Schritt-für-Schritt-Anleitungen
- 📚 Aktualisierte README mit JSON-Konfigurationshinweisen
- 📚 Beispiele für alle Datentypen und Strukturen

Bugfixes:
- �  Token-Tracking korrigiert: Singleton-Pattern implementiert
  - Problem: Mehrere separate TokenTracker-Instanzen in verschiedenen Modulen führten zu inkonsistenten Statistiken
  - Lösung: Globale `get_global_token_counter()` Funktion stellt sicher, dass alle Module dieselbe Instanz verwenden
  - Alle Token-Statistiken werden jetzt korrekt aggregiert und angezeigt
  - Session- und Daily-Statistiken zeigen nun akkurate Werte
  - Betrifft: `analysis_manager.py`, `deductive_coding.py`, `inductive_coding.py`, `relevance_checker.py`, `openai_provider.py`

Code Quality:
- 📦 Erweiterte Module: `config/loader.py`, `config/converter.py`, `config/synchronizer.py`
- 📦 Verbessertes Token-Tracking: `tracking/token_tracker.py` mit Singleton-Pattern
- Vollständige Implementierung aller 10 Requirements mit 60+ Acceptance Criteria
- Umfassende Systemtests bestätigen korrekte Funktionalität
- 📚 Detaillierte Inline-Dokumentation mit Requirement-Referenzen
- Vollständige Abwärtskompatibilität - keine Breaking Changes

Vorteile der JSON-Konfiguration:
- 🚀 Schnelleres Laden (JSON-Parsing ~10x schneller als Excel)
- 📝 Versionskontrollfreundlich (Git-Diffs lesbar und nachvollziehbar)
- 🔧 Programmatische Konfigurationsänderungen möglich
- 🌍 Bessere Portabilität zwischen Systemen
- 👥 Einfachere Zusammenarbeit durch Textformat

### Neu in 0.10.2 (2025-11-27)

QCA-AID-EXPLORER REFACTORING: MODULARE ARCHITEKTUR & JSON-KONFIGURATION

Explorer Verbesserungen:
- ✨ Vollständiges Refactoring in modulare Struktur innerhalb von `QCA_AID_assets`
  - Minimales Launcher-Skript `QCA-AID-Explorer.py` (< 50 Zeilen)
  - Alle Funktionalitäten in logische Module organisiert
  - Neue Module: `explorer.py`, `analysis/qca_analyzer.py`, `utils/config/loader.py`, `utils/config/converter.py`, `utils/config/synchronizer.py`, `utils/visualization/layout.py`
- ✨ JSON-Konfigurationsunterstützung
  - Neue Datei `QCA-AID-Explorer-Config.json` als alternatives Konfigurationsformat
  - Excel-Konfiguration (`QCA-AID-Explorer-Config.xlsx`) weiterhin vollständig unterstützt
  - Automatische bidirektionale Synchronisation zwischen Excel und JSON
  - Konfliktauflösung bei Differenzen mit Benutzerabfrage
  - Automatische Migration beim ersten Start
- 🔧 Verbesserte Wartbarkeit und Testbarkeit
  - Einzelne Komponenten können isoliert getestet werden
  - Module können in anderen Projekten wiederverwendet werden
  - Vollständige Dokumentation mit Docstrings
  - JSON-Schema-basierte Validierung mit detaillierten Fehlermeldungen
- 🔧 Performance und Versionskontrolle
  - JSON-Laden schneller als Excel-Parsing
  - Versionskontrollfreundlich (Git-Diffs lesbar)
  - Programmatische Konfigurationsänderungen möglich
- 🔧 Vereinheitlichte LLM Provider
  - Nutzt ausgereiften LLM Provider aus QCA-AID mit Model Capability Detection
  - Robuste Retry-Logik und Fehlerbehandlung
- 🔧 Robuste Spaltennamenerkennung
  - Automatische Normalisierung von Spaltennamen mit Encoding-Problemen
  - Verbesserte Fehlerbehandlung bei leeren Graphen und fehlenden Daten

Code Quality:
- 📦 Neue Module: `config_loader.py`, `config_synchronizer.py`, `config_converter.py`
- Umfassende Test-Suite für Konfigurationsmanagement
- 📚 Aktualisierte Dokumentation in `qca-aid-explorer-readme.md`
- Funktionalität bleibt vollständig erhalten - keine Breaking Changes

### Neu in 0.10.1

PARAPHRASEN-BASIERTER BATCH-KONTEXT & BUGFIXES

Neue Features:
- ✨ Paraphrasen-basierter Batch-Context für intelligenteres Kodieren
  - Nutzt bereits generierte Paraphrasen aus vorherigen Batches als Kontext
  - Verbessert das Verständnis impliziter Bezüge im Text
  - Minimaler Performance-Overhead (<5%)
  - Konfigurierbar: `CODE_WITH_CONTEXT` Flag und `CONTEXT_PARAPHRASE_COUNT` Anzahl
- ✨ Neue Excel-Spalte "Kontext_verwendet" in Kodierungsergebnisse
  - Zeigt an, ob Kontextparaphrasen bei der Kodierung verwendet wurden

Verbesserungen:
- 🔧 Begründungen bei nicht-relevanten Segmenten
  - RelevanceChecker-Begründungen werden korrekt in Export-Tabelle übernommen
  - Mit "[Relevanzprüfung]" Präfix gekennzeichnet
  - Intelligente Fallback-Begründungen bei fehlenden Details
- 🔧 Unified Timeout-Animation im UI
  - "Analysemodus ändern?" und "Gespeichertes Codesystem verwenden?" zeigen Countdown inline animiert
- 🔧 Dokument-isolierte Paraphrasen-Batches
  - Batches enthalten IMMER nur Segmente aus EINEM Dokument
  - Keine Paraphrasen-Vermischung zwischen Dokumenten
  - Segmente automatisch nach Dokument sortiert (reproducible Reihenfolge)

Bugfixes:
- 🐛 RelevanceChecker: Entfernt dupliziertes `justification` Feld
  - Nur noch `reasoning` Feld für Begründungen
  - Reduziert Code-Duplikation in results_exporter.py um ~99 Zeilen
- 🐛 Inductive Coding Mode: Missing `datetime` Import behoben
  - Fehler: `name 'datetime' is not defined` → Behoben
- 🐛 Inductive Coding Mode: CategoryDefinition mit None definition
  - Fehler: `AttributeError: 'NoneType' object has no attribute 'definition'` → Behoben
  - Sichere Filterung ungültiger Kandidaten in `_validate_and_integrate_strict()`
  - Robuste None-Checks in `_meets_quality_standards()`
- 🐛 Export-Tabelle: Duplizierung bei Begründungs-Logik aufgelöst
  - Vorher: ~50 Zeilen Debug-Code mit mehrfachen Checks
  - Nachher: Single-Pass Logik mit klarer Priorität

Code Quality:
- 📉 Entfernt: 904 Zeilen obsoleter Code (alte progressive_context Methoden)
- 📉 Refactored: 1,089 Zeilen Duplikats-Code aus analysis_manager, deductive_coding, results_exporter
- Alle Dateien syntaktisch korrekt verifiziert

### Neu in 0.10.0

MASSIVES REFACTORING: KOMPLETTE MODULARISIERUNG DES GESAMTSYSTEMS
- Transformation der monolithischen Codebase in modulare Mikroservice-ähnliche Architektur
- Auflösung von QCA_Utils.py (3954 Zeilen) und Ausgliedern von Code aus main.py in spezialisierte Module
- Neue modulare Struktur mit 8 Fachmodulen:
  - `utils/llm/` - LLM-Abstraktionsschicht (OpenAI, Mistral mit Factory-Pattern)
  - `utils/config/` - Konfigurationsladung und Validation
  - `utils/tracking/` - Token-Tracking und Kostenberechnung für alle API-Calls
  - `utils/dialog/` - Tkinter GUI-Komponenten für manuelles Kodieren
  - `utils/export/` - Export-Formatierung, PDF-Annotation, Excel-Generierung
  - `utils/io/` - Dokumentenladung (.txt, .pdf, .docx) und Datei-I/O
  - `utils/analysis/` - Hilfsreiches für Kodierungslogik (Kategorien, Konsensus)
  - `core/`, `analysis/`, `preprocessing/`, `quality/`, `export/`, `management/` - Spezialisierte Subdomain-Module

Architektur-Verbesserungen:
- Reduzierte zirkuläre Abhängigkeiten durch klare Modul-Grenzen
- Verbesserte Code-Wartbarkeit mit fokussierten, testbaren Komponenten
- Erweiterte Testbarkeit: Isolierte Module ermöglichen Unit-Testing ohne API-Dependencies
- Bessere Skalierbarkeit: Neue Provider, Export-Formate oder Analysemodi können leicht hinzugefügt werden
- Windows Unicode-Kodierungsfixes: Robuste Verarbeitung von Sonderzeichen und Umlauten
- Vereinfachtes Onboarding: Klare Verantwortlichkeiten pro Modul

UI/UX Verbesserungen:
|- Verbesserte Analyse-Konfiguration beim Start mit übersichtlicher Darstellung
|- Konfigurationsparameter-Übersicht: Zeigt alle wichtigen Einstellungen beim Programmstart
|- Interaktive Analysemodus-Auswahl mit 10s Timeout (inductive/abductive/deductive/grounded)
|- Intelligente Codebook-Verwaltung: Erkennt gespeicherte induktive Codesysteme automatisch
|- Optionale manuelle Kodierung mit informativen Hinweisen zum Workflow
|- Zusammenfassung der Konfigurationsentscheidungen vor Analysestart
|- Robust gestaltete Excel-Tabellenerstellung mit Fallback auf AutoFilter bei Fehlern

Bugfixes:
|- Import-Fehler in category_revision.py behoben (fehlende openpyxl-Imports)
|- token_counter nicht definiert in main.py behoben (Import hinzugefügt)
|- PDF-Annotation nicht verfügbar - fuzzywuzzy und python-Levenshtein installiert
|- Tuple-Import in pdf_annotator.py ergänzt
|- DocumentToPDFConverter.convert_document_to_pdf() -> convert() Methode korrigiert
|- Robustere Excel-Tabellenerstellung mit Validierung und Fallback-Mechanismen
|- `re` Import in pdf_annotator.py hinzugefügt
|- cleanup_temp_pdfs() Methode in DocumentToPDFConverter implementiert
|- Platform-Import in manual_coding.py hinzugefügt
|- Threading-Event für manuelle Kodierung synchronisiert (Fenster warten auf Schließung)
|- ESC-Taste Handling für manuelles Kodieren verbessert (Doppel-ESC zum Abbruch)
|- Doppelte Abfrage zur manuellen Kodierung entfernt
|- CodingResult zu Dictionary Konvertierung in manueller Kodierung robuster gemacht
|- Annotierte PDFs werden jetzt in `output/Annotated/` Unterordner gespeichert
|- Benutzerdefinierte INPUT_DIR/OUTPUT_DIR Ordnernamen werden konsistent respektiert

Manuelle Kodierung Verbesserungen:
|- Threading-basierte Synchronisation für sequenzielle Fenster-Verarbeitung
|- ESC-Taste drücken und nochmal ESC zum bestätigen für Abbruch
|- Mehrfachkodierung mit CodingResult Objekten jetzt unterstützt
|- Robustes Tkinter-Fenster-Management mit korrektem Thread-Handling


### Neu in 0.9.18 (2025-07-07)

KATEGORIE-KONSISTENZ: Deduktiver Modus mit Hauptkategorie-Vorauswahl (1-3 wahrscheinlichste), 40-60% weniger Token, keine inkompatiblen Subkategorie-Zuordnungen
SUBKATEGORIE-VALIDIERUNG: Strikte Konsistenzprüfung mit automatischer Entfernung fremder Subkategorien, zweistufige Validierung, detailliertes Tracking
PERFORMANCE-OPTIMIERUNG: Fokussierte AI-Kodierung nur mit relevanten Kategorien, verbesserte Qualität durch kategorie-spezifischen Fokus, kompatibel mit allen Features
PYMUPDF-FIX: fitz.open() durch fitz.Document() ersetzt, robuste Fehlerbehandlung für PDF-Laden/-Speichern
CONFIDENCE-SCALES: Zentrale Klasse mit 5 spezialisierten Skalen (0.6+ definitiv, 0.8+ eindeutig), einheitliche textbelegte Konfidenz-Bewertungen in allen Prompts
EXPORT-FIX: Begründungen bei Nichtkodierung werden nun korrekt exportiert

### Neu in 0.9.17 (2025-06-22)
- Input dateien können jetzt als annotierte Version exportiert werden
- PDF werden direkt annotiert, TXT und DOCX werden in PDF umgewandelt und annotiert. 
- kann über 'EXPORT_ANNOTATED_PDFS': True (default) bzw. mit False deaktiviert werden.

### Neu in 0.9.16.2 (2025-06-11)

Bugfixes und Verbesserungen
Verbessertes Kodierungsergebnisse Sheet: Optimierte Darstellung und Formatierung der Kodierungsergebnisse im Excel-Export Grounded Mode Optimierung: Entfernung deduktiver Kategorien bei der Kodierung im Grounded Mode für reinere induktive Kategorienentwicklung Neuer Token-Counter: Präziserer Token-Counter basierend auf tatsächlichen Tokens beim API Provider für genauere Kostenberechnung

### Neu in 0.9.16.1

Bugfixes und Verbesserungen

Überarbeitete Intercoder-Berechnung: Verbesserte Intercoder-Reliabilitätsberechnung um der Mehrfachkodierung gerecht zu werden, nach Krippendorf 2011 mittels Sets Export-Layout überarbeitet: Komplett überarbeiteter Aufbau und Layout des Excel-Exports für bessere Übersichtlichkeit
Neu in 0.9.16

Erweiterte manuelle Kodierung mit Mehrfachkodierung-Support

Mehrfachkategorien-Auswahl: Benutzer können nun mehrere Kategorien gleichzeitig auswählen (Strg+Klick, Shift+Klick) Intelligente Validierung: Automatische Validierung verhindert inkonsistente Mehrfachauswahlen Separate Kodierungsinstanzen: Automatische Erstellung separater Kodierungsinstanzen bei verschiedenen Hauptkategorien Verbesserte GUI: Erweiterte Benutzeroberfläche mit Auswahlinfo und speziellem Mehrfachkodierungs-Dialog Nahtlose Integration: Konsistente Integration mit dem bestehenden Mehrfachkodierungs-System
### 
Neu in 0.9.15 (2025-06-02)

    COMPLETE RESTRUCTURING OF INDUCTIVE MODE: Vollständige Neustrukturierung des induktiven Modus • Vereinfachte und robustere Kategorienentwicklung mit verbesserter Konsistenz • Optimierte Sättigungsprüfung und stabilere Kategorienvalidierung • Reduzierte Komplexität bei gleichzeitig erhöhter Methodentreue
    IMPROVED ABDUCTIVE MODE: Verbesserungen beim abduktiven Modus • Präzisere Subkategorien-Entwicklung zu bestehenden Hauptkategorien • Bessere Integration neuer Subkategorien in das bestehende System
    GRACEFUL ANALYSIS INTERRUPTION: Analyse kann mit ESC-Taste abgebrochen werden • Zwischenergebnisse werden automatisch gespeichert bei Benutzerabbruch • Wiederaufnahme der Analyse ab dem letzten Checkpoint möglich • Vollständige Datenintegrität auch bei vorzeitigem Abbruch
    MASSIVE PERFORMANCE BOOST: 4x Beschleunigung durch Parallelisierung • Parallele Verarbeitung aller Segmente eines Batches gleichzeitig • Optimierte API-Calls durch intelligente Bündelung • Dramatisch reduzierte Analysezeiten bei großen Datenmengen
    Enhanced error handling and stability improvements
    Improved progress monitoring and user feedback
    Optimized memory usage for large document sets

### Neu in 0.9.14 (2025-05-28)

    Implementierung der Mehrfachkodierung von Textsegmenten für mehrere Hauptkategorien
    Neue CONFIG-Parameter: MULTIPLE_CODINGS (default: True) und MULTIPLE_CODING_THRESHOLD (default: 0.7)
    Erweiterte Relevanzprüfung erkennt Segmente mit Bezug zu mehreren Hauptkategorien (>=70% Relevanz)
    Fokussierte Kodierung: Segmente werden gezielt für jede relevante Hauptkategorie kodiert
    Export-Erweiterung: Mehrfach kodierte Segmente erscheinen pro Hauptkategorie separat in der Outputtabelle
    Neue Export-Felder: Mehrfachkodierung_Instanz, Kategorie_Fokus, Fokus_verwendet
    Eindeutige Chunk-IDs mit Instanz-Suffix bei Mehrfachkodierung (z.B. "DOC-5-1", "DOC-5-2")
    Effiziente Batch-Verarbeitung und Caching für Mehrfachkodierungs-Prüfungen
    Konfigurierbare Deaktivierung der Mehrfachkodierung für traditionelle Einzelkodierung

### Neu in 0.9.13 (2025-05-15)

    Vollständige Implementierung des 'majority' Review-Modus mit einfacher Mehrheitsentscheidung
    Neue 'manual_priority' Option bevorzugt manuelle vor automatischen Kodierungen
    Korrigierte Review-Logik: REVIEW_MODE wird jetzt korrekt respektiert, unabhängig von Kodierer-Typ
    Konsistente Behandlung der REVIEW_MODE Konfiguration mit einheitlichem Standard 'consensus'
    Verbesserte Tie-Breaking-Mechanismen bei Gleichstand zwischen Kodierungen
    Erweiterte Dokumentation der Review-Modi im consensus_info Export-Feld

QCA-AID-Explorer Verbesserungen:
- 🔧 Robuste Filter-Logik mit automatischem Mapping von Attribut_1-3 zu tatsächlichen Spaltennamen
- 🔧 Selektive Keyword-Harmonisierung nur für Analysetypen, die sie benötigen
- 🔧 Verbesserte Fehlerbehandlung: Filter für nicht existierende Spalten werden übersprungen
- 🔧 Performance-Optimierung: Unnötige Keyword-Verarbeitung vermieden
- 📊 Detaillierte Debug-Ausgaben über angewendete Filter und Spalten-Mappings

### Neu in 0.9.12 (2025-05-10)

    Verbesserter manueller Kodierungsworkflow mit korrekter Handhabung des letzten Segments
    Verbesserte Funktionalität der Schaltflächen "Kodieren & Abschließen" für eine intuitivere Vervollständigung der Kodierung
    Robustes manuelles Code-Review-System zur Behebung von Unstimmigkeiten zwischen den Codierern hinzugefügt
    Die Tkinter-Ressourcenverwaltung wurde verbessert, um Fehler beim Schließen von Fenstern zu vermeiden
    Verbesserte Fehlerbehandlung für den Export von Überprüfungsentscheidungen
    Allgemeine Stabilitätsverbesserungen für die Schnittstelle zur manuellen Kodierung
    Neue Funktion zur automatischen Sicherung des Kodierfortschritts
    Verbesserte Benutzerführung im manuellen Kodierungsmodus
    Optimierte Darstellung der Kodierhistorie

### Neu in 0.9.11 (2025-04-12)

    Neuer 'grounded' Analysemodus hinzugefügt, inspiriert von Grounded Theory und Kuckartz
    Im 'grounded' Modus werden die Subcodes schrittweise gesammelt, ohne sie den Hauptkategorien zuzuordnen
    Die gesammelten Subcodes werden vom deduktiven Kodierer direkt zur Kodierung verwendet
    Nach der Verarbeitung aller Segmente werden aus den Subcodes anhand von Schlüsselwörtern Hauptkategorien generiert
    Die Subcodes werden im endgültigen Export mit den generierten Hauptkategorien abgeglichen
    Die Ausgabe wird im Codebuch und in den Exporten als "grounded" (nicht "induktiv") gekennzeichnet
    Verbesserte Fortschrittsvisualisierung während der Subcode-Erfassung
    Verbesserte Handhabung von Schlüsselwörtern mit direkter Verbindung zu Subcodes

QCA-AID-Explorer Verbesserungen:
- ✨ Neue Schlüsselwort-basierte Sentiment-Analyse
  - Visualisiert wichtigste Begriffe aus Textsegmenten als Bubbles
  - Eingefärbt nach Sentiment (positiv/negativ oder benutzerdefinierte Kategorien)
  - Flexible Konfiguration: Anpassbare Sentiment-Kategorien, Farbschemata und Prompts
  - Umfassende Ergebnisexporte: Excel-Tabellen mit Sentiment-Verteilungen, Kreuztabellen, Keyword-Rankings
- 📊 Excel-basierte Konfiguration (QCA-AID-Explorer-Config.xlsx)
- 📊 Heatmap-Visualisierung von Codes entlang von Dokumentattributen
- 📊 Mehrere Analysetypen konfigurierbar (Netzwerk, Heatmap, Zusammenfassungen)
- 📊 Anpassbare Parameter für jede Analyse
- 🔧 Eindeutige Segment-IDs mit Präfix zur Chunk-Nummer
- 🔧 Prägnantere progressive Zusammenfassungen mit weniger Informationsverlust

### Neu in 0.9.9

    Abduktivmodus: induktive Codierung nur für Subcodes ohne HinzuFügen von Hauptcodes
    kann entweder beim starten des Skripts ausgewählt oder im Codebook konfiguriert
    leicht verschärfte Relevanzprüfung für Textsegmente (aus Interviews)
    Kodierkonsens: Segmente ohne Konsens als "kein Kodierkonsens" markieren; wenn kein Konsens besteht, wird die Kodierung mit höherem Konfidenzwert gewählt, sonst "kein Kodierkonsens"

### Weitere Hinweise zur Version (0.9.8)

    Progressive Dokumentenzusammenfassung als Kodierungskontext (max. 80 Wörter)
    Aktivieren durch Setzen des CONFIG-Wertes CODE_WITH_CONTEXT im Codebook auf 'true' (Standard: false)
    Eignet sich insbesondere bei deduktivem Kodieren. Es kann Einfluss auf die Kodierung nehmen, daher testen, ob die Funktion zu besseren Ergebnissen führt. Den Kontext beizuFügen, erleichtert es dem Sprachmodell einzuschätzen, ob die Inhalte im größeren Zusammenhang des Textes bedeutsam sind. Damit wird gewissermaßen ein Gedächtnis des bisherigen Textes in die Verarbeitung des Textsegments integriert.

### Weitere Hinweise zur Version (0.9.7)

    NEU: Mistral Support! Es kann jetzt auch die Mistral API genutzt werden. Umschalten zwischen OpenAI und Mistral mit CONFIG-Parameter 'MODEL_PROVIDER'. Standardmodell für OpenAI ist 'GPT-4o-mini', für Mistral 'mistral-small'.
    NEU: Ausschlusskriterien während der Relevanzprüfung in 'KODIERREGELN' definieren (z.B. Literaturverzeichnis)
    NEU: HinzuFügen von Ausschlusskriterien für die Relevanzprüfung in Codebuch-Kodierregeln
    NEU: Export von Begründungen für nicht relevante Textsegmente
    Verbesserte Relevanzprüfung, Rechtfertigung und Aufforderung zur Kodierung von Segmenten
    NEU: Erstellen von Zusammenfassungen und Diagrammen aus Ihren kodierten Daten mit 'QCA-AID-Explorer.py'.

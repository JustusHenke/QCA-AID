# Changelog

## Versionen und Updates

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
  - Mit "[RelevanzprÜfung]" Präfix gekennzeichnet
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
  - Fehler: `name 'datetime' is not defined` → ✅ Behoben
- 🐛 Inductive Coding Mode: CategoryDefinition mit None definition
  - Fehler: `AttributeError: 'NoneType' object has no attribute 'definition'` → ✅ Behoben
  - Sichere Filterung ungültiger Kandidaten in `_validate_and_integrate_strict()`
  - Robuste None-Checks in `_meets_quality_standards()`
- 🐛 Export-Tabelle: Duplizierung bei Begründungs-Logik aufgelöst
  - Vorher: ~50 Zeilen Debug-Code mit mehrfachen Checks
  - Nachher: Single-Pass Logik mit klarer Priorität

Code Quality:
- 📉 Entfernt: 904 Zeilen obsoleter Code (alte progressive_context Methoden)
- 📉 Refactored: 1,089 Zeilen Duplikats-Code aus analysis_manager, deductive_coding, results_exporter
- ✅ Alle Dateien syntaktisch korrekt verifiziert

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


QCA-AID-Explorer.py Enhancements:
- Excel-basierte Konfiguration (QCA-AID-Explorer-Config.xlsx)
- Heatmap-Visualisierung von Codes entlang von Dokumentattributen
- Mehrere Analysetypen konfigurierbar (Netzwerk, Heatmap, Zusammenfassungen)
- Anpassbare Parameter für jede Analyse
- Eindeutige Segment-IDs mit Präfix zur Chunk-Nummer
- Prägnantere progressive Zusammenfassungen mit weniger Informationsverlust

### Neu in 0.9.9

    Abduktivmodus: induktive Codierung nur für Subcodes ohne Hinzufügen von Hauptcodes
    kann entweder beim starten des Skripts ausgewählt oder im Codebook konfiguriert
    leicht verschärfte Relevanzprüfung für Textsegmente (aus Interviews)
    Kodierkonsens: Segmente ohne Konsens als "kein Kodierkonsens" markieren; wenn kein Konsens besteht, wird die Kodierung mit höherem Konfidenzwert gewählt, sonst "kein Kodierkonsens"

### Weitere Hinweise zur Version (0.9.8)

    Progressive Dokumentenzusammenfassung als Kodierungskontext (max. 80 Wörter)
    Aktivieren durch Setzen des CONFIG-Wertes CODE_WITH_CONTEXT im Codebook auf 'true' (Standard: false)
    Eignet sich insbesondere bei deduktivem Kodieren. Es kann Einfluss auf die Kodierung nehmen, daher testen, ob die Funktion zu besseren Ergebnissen führt. Den Kontext beizufügen, erleichtert es dem Sprachmodell einzuschätzen, ob die Inhalte im größeren Zusammenhang des Textes bedeutsam sind. Damit wird gewissermaßen ein Gedächtnis des bisherigen Textes in die Verarbeitung des Textsegments integriert.

### Weitere Hinweise zur Version (0.9.7)

    NEU: Mistral Support! Es kann jetzt auch die Mistral API genutzt werden. Umschalten zwischen OpenAI und Mistral mit CONFIG-Parameter 'MODEL_PROVIDER'. Standardmodell für OpenAI ist 'GPT-4o-mini', für Mistral 'mistral-small'.
    NEU: Ausschlusskriterien während der Relevanzprüfung in 'KODIERREGELN' definieren (z.B. Literaturverzeichnis)
    NEU: Hinzufügen von Ausschlusskriterien für die Relevanzprüfung in Codebuch-Kodierregeln
    NEU: Export von Begründungen für nicht relevante Textsegmente
    Verbesserte Relevanzprüfung, Rechtfertigung und Aufforderung zur Kodierung von Segmenten
    NEU: Erstellen von Zusammenfassungen und Diagrammen aus Ihren kodierten Daten mit 'QCA-AID-Explorer.py'.

# QCA-AID Webapp - Dokumentation

![QCA-AID](../banner-qca-aid.png)

Die QCA-AID Webapp ist eine lokale, webbasierte Benutzeroberfläche zur Verwaltung von QCA-AID Analysen mit intuitiver Konfiguration, Codebook-Editor und Analyse-Steuerung.

## Schnellstart

### Webapp starten
**Windows:** Doppelklick auf `Start-QCA-AID-Webapp.bat`  
**Manuell:** `python start_webapp.py` oder `streamlit run webapp.py`

Die Webapp öffnet sich unter: `http://127.0.0.1:8501`

### Erste Schritte (5 Minuten)
1. **Projekt-Root festlegen**: Wähle dein Projektverzeichnis (optional, Standard: aktuelles Verzeichnis)
2. **Config-Tab**: Konfiguration laden (`../examples/config-standard.json`)
3. **Codebook-Tab**: Forschungsfrage und Kategorien anpassen
4. **Analyse-Tab**: Eingabedateien prüfen, Analyse starten
5. **Explorer-Tab**: Ergebnisse ansehen

## Installation

**Voraussetzungen:** Python 3.10/3.11, 4 GB RAM, moderner Browser

```bash
pip install -r ../requirements.txt
```

Die Batch-Datei prüft und installiert Abhängigkeiten automatisch.

## Benutzeroberfläche

### 0. Projekt-Management

**Projekt-Root-Verzeichnis:**
Die Webapp unterstützt projektbasiertes Arbeiten. Das Projekt-Root-Verzeichnis ist der zentrale Ordner, in dem alle Konfigurationsdateien, Input- und Output-Ordner organisiert werden.

**Projekt-Root festlegen:**
1. Klicke auf "📁 Projekt-Verzeichnis ändern" im Config- oder Codebook-Tab
2. Wähle dein Projektverzeichnis im Dialog
3. Die Webapp speichert diese Einstellung automatisch in `.qca-aid-project.json`

**Vorteile:**
- Alle Pfade werden relativ zum Projekt-Root aufgelöst
- Einfacher Wechsel zwischen verschiedenen Projekten
- Automatisches Laden der letzten Projekt-Einstellungen beim Start
- Bessere Organisation von Konfigurationen und Daten

**Standard-Verhalten:**
- Beim ersten Start: Aktuelles Arbeitsverzeichnis als Projekt-Root
- Bei erneutem Start: Letztes verwendetes Projekt-Root wird geladen
- Projekt-Einstellungen werden in `.qca-aid-project.json` gespeichert

### 1. Konfiguration (Config-Tab)

**Datei-Browser:**
- Klicke auf das 📁-Symbol neben dem Pfad-Eingabefeld
- Wähle eine Konfigurationsdatei (.json oder .xlsx)
- Der vollständige Pfad wird automatisch eingetragen
- Format wird automatisch erkannt (JSON/XLSX)

**Modell-Einstellungen:**
- MODEL_PROVIDER: OpenAI, Mistral
- MODEL_NAME: z.B. gpt-4o-mini

**Chunk-Einstellungen:**
- CHUNK_SIZE: 800-1500 (empfohlen: 1000)
- CHUNK_OVERLAP: 30-100 (empfohlen: 50)
- BATCH_SIZE: 5-8 (empfohlen: 5)

**Analyse-Optionen:**
- ANALYSIS_MODE: deductive, abductive, full, grounded
- CODE_WITH_CONTEXT: Kontextuelle Kodierung
- MULTIPLE_CODINGS: Mehrfachkodierungen

**Coder-Einstellungen:**
- temperature: 0.0-1.0 (Kreativität)
- coder_id: Eindeutige Bezeichnung

### 2. Codebook (Codebook-Tab)

**Datei-Browser:**
- Klicke auf das 📁-Symbol zum Laden oder Speichern
- Beim Speichern wird ein vorgeschlagener Dateiname angeboten
- Unterstützt JSON und XLSX Formate

**Codebook-Elemente:**
- **Forschungsfrage**: Zentrale Forschungsfrage
- **Kodierregeln**: Allgemeine, Format- und Ausschlussregeln
- **Kategorien**: Name, Definition, Regeln, Beispiele, Subkategorien
- **JSON-Vorschau**: Echtzeit-Strukturansicht

**Induktive Codes importieren:**
Wenn vorherige Analysen induktive Codes entwickelt haben, kannst du diese importieren:

1. **Automatische Erkennung**: Beim Öffnen des Codebook-Tabs scannt die Webapp den Output-Ordner
2. **Benachrichtigung**: Falls induktive Codes gefunden werden, erscheint eine Info-Meldung
3. **Import-Button**: Klicke auf "Induktive Codes importieren"
4. **Datei auswählen**: Wähle die Analyse-Datei mit den gewünschten Codes
5. **Vorschau**: Sieh dir die zu importierenden Codes an
6. **Konflikt-Behandlung**: Bei Namenskonflikten bietet die Webapp Umbenennungsoptionen
7. **Separate Anzeige**: Importierte Codes erscheinen in einer eigenen Sektion
8. **Quellenangabe**: Jeder importierte Code zeigt seine Herkunftsdatei

**Vorteile des Imports:**
- Nutze automatisch entwickelte Kategorien aus vorherigen Analysen
- Kombiniere deduktive Basis-Codes mit induktiven Erkenntnissen
- Iterative Verfeinerung des Kategoriensystems
- Vollständige Bearbeitbarkeit importierter Codes

### 3. Analyse (Analyse-Tab)

- **Eingabedateien**: Liste mit Vorschau (.txt, .pdf, .docx)
- **Analyse-Steuerung**: Start/Stop, Fortschritt, Echtzeit-Logs
- **Ergebnisse**: Output-Links, Statistiken

### 4. Explorer (Explorer-Tab)

- **Output-Dateien**: XLSX-Ergebnisse mit Vorschau
- **Explorer-Config**: Diagrammtypen, Visualisierungseinstellungen

## Projekt-Management Workflows

### Neues Projekt erstellen

1. **Projektordner vorbereiten:**
   ```
   mein-projekt/
   ├── input/          # Eingabedateien
   ├── output/         # Analyseergebnisse
   ├── config/         # Konfigurationen (optional)
   └── codebooks/      # Codebook-Dateien (optional)
   ```

2. **Projekt-Root setzen:**
   - Webapp starten
   - Im Config-Tab auf "📁 Projekt-Verzeichnis ändern" klicken
   - `mein-projekt/` Ordner auswählen

3. **Konfiguration erstellen:**
   - Beispielkonfiguration laden oder neue erstellen
   - Mit Datei-Browser im gewünschten Ordner speichern

4. **Codebook erstellen:**
   - Im Codebook-Tab Kategorien definieren
   - Mit Datei-Browser speichern

### Zwischen Projekten wechseln

1. Klicke auf "📁 Projekt-Verzeichnis ändern"
2. Wähle das neue Projektverzeichnis
3. Die Webapp lädt automatisch:
   - Gespeicherte Projekt-Einstellungen
   - Letzte verwendete Konfiguration
   - Letzte verwendetes Codebook

### Projekt-Einstellungen verwalten

**Automatische Speicherung:**
- Projekt-Root wird in `.qca-aid-project.json` gespeichert
- Letzte verwendete Dateipfade werden gespeichert
- Einstellungen werden beim nächsten Start geladen

**Manuelle Verwaltung:**
- `.qca-aid-project.json` im Projekt-Root bearbeiten
- Datei löschen, um Einstellungen zurückzusetzen

### Induktive Codes aus Analysen nutzen

**Workflow:**

1. **Erste Analyse durchführen:**
   - Deduktives oder abduktives Codebook erstellen
   - Analyse mit `ANALYSIS_MODE: abductive` oder `grounded` durchführen
   - System entwickelt induktive Codes aus den Daten

2. **Induktive Codes importieren:**
   - Codebook-Tab öffnen
   - Benachrichtigung über verfügbare induktive Codes beachten
   - "Induktive Codes importieren" klicken
   - Analyse-Datei auswählen (z.B. `analysis_2024-11-29.xlsx`)

3. **Codes überprüfen und anpassen:**
   - Importierte Codes in separater Sektion ansehen
   - Definitionen und Regeln verfeinern
   - Bei Bedarf umbenennen oder löschen

4. **Erweitertes Codebook speichern:**
   - Kombiniertes Codebook (deduktiv + induktiv) speichern
   - Für weitere Analysen verwenden

5. **Iterative Verfeinerung:**
   - Weitere Analysen mit erweitertem Codebook durchführen
   - Neue induktive Codes importieren
   - Kategoriensystem kontinuierlich verbessern

**Beispiel-Szenario:**

```
Iteration 1:
- 5 deduktive Basis-Kategorien
- Analyse entwickelt 3 neue induktive Kategorien
- Import → 8 Kategorien gesamt

Iteration 2:
- 8 Kategorien als Basis
- Analyse entwickelt 2 weitere induktive Kategorien
- Import → 10 Kategorien gesamt

Iteration 3:
- 10 Kategorien als Basis
- Keine neuen induktiven Kategorien → Sättigung erreicht
```

## Fehlerbehebung

### Datei-Browser öffnet nicht

**Problem:** Datei-Browser-Dialog erscheint nicht beim Klick auf 📁  
**Ursache:** tkinter nicht installiert oder GUI-Backend fehlt

**Lösung Windows:**
```bash
# tkinter ist normalerweise in Python enthalten
python -m tkinter  # Test ob tkinter funktioniert
```

**Lösung Linux:**
```bash
# tkinter installieren
sudo apt-get install python3-tk  # Debian/Ubuntu
sudo yum install python3-tkinter  # RedHat/CentOS
```

**Lösung macOS:**
```bash
# tkinter ist normalerweise in Python enthalten
# Falls nicht: Python von python.org neu installieren
```

**Alternative:** Pfade manuell eingeben, wenn Datei-Browser nicht funktioniert

### Datei-Browser hinter Webapp-Fenster

**Problem:** Dialog öffnet sich, ist aber nicht sichtbar  
**Lösung:**
- Alt+Tab (Windows/Linux) oder Cmd+Tab (macOS) zum Wechseln
- Webapp-Fenster minimieren
- Dialog sollte im Vordergrund erscheinen

### Pfad-Validierung zeigt Fehler

**Problem:** "Ungültiger Pfad" obwohl Pfad korrekt erscheint  
**Lösungen:**

1. **Relative vs. absolute Pfade:**
   - Absolute Pfade verwenden: `C:\Users\...\config.json`
   - Oder relativ zum Projekt-Root: `config/standard.json`

2. **Sonderzeichen im Pfad:**
   - Keine Umlaute in Ordnernamen
   - Keine Leerzeichen (oder in Anführungszeichen)
   - Backslashes escapen: `C:\\Users\\...`

3. **Schreibrechte prüfen:**
   - Rechtsklick auf Ordner → Eigenschaften → Sicherheit
   - Schreibrechte für aktuellen Benutzer aktivieren

4. **Verzeichnis existiert nicht:**
   - Webapp bietet an, Verzeichnis zu erstellen
   - Oder manuell erstellen vor dem Speichern

### Induktive Codes werden nicht erkannt

**Problem:** Keine Benachrichtigung über verfügbare induktive Codes  
**Lösungen:**

1. **Output-Ordner prüfen:**
   - Sind XLSX-Dateien im Output-Ordner?
   - Wurden Analysen mit `abductive` oder `grounded` Modus durchgeführt?

2. **Dateiformat prüfen:**
   - Nur XLSX-Dateien werden gescannt
   - Dateinamen sollten `analysis_` enthalten

3. **Analyse-Modus prüfen:**
   - `deductive` Modus entwickelt keine induktiven Codes
   - `abductive`, `full` oder `grounded` verwenden

4. **Manuell prüfen:**
   - XLSX-Datei in Excel/LibreOffice öffnen
   - Nach Sheet "Inductive Categories" suchen
   - Falls vorhanden: Datei sollte erkannt werden

### Import-Konflikt bei Kategorien

**Problem:** "Kategorie existiert bereits" beim Import  
**Lösung:**

1. **Umbenennen akzeptieren:**
   - Webapp schlägt automatisch neue Namen vor
   - Z.B. "Technologie" → "Technologie_imported"

2. **Manuell umbenennen:**
   - Vor Import: Bestehende Kategorie umbenennen
   - Oder: Nach Import die importierte Kategorie umbenennen

3. **Zusammenführen:**
   - Definitionen vergleichen
   - Wenn identisch: Eine Kategorie löschen
   - Wenn unterschiedlich: Beide behalten mit klaren Namen

### Projekt-Einstellungen gehen verloren

**Problem:** Projekt-Root muss bei jedem Start neu gesetzt werden  
**Lösungen:**

1. **Schreibrechte prüfen:**
   - `.qca-aid-project.json` muss im Projekt-Root erstellt werden können
   - Ordner-Schreibrechte prüfen

2. **Datei manuell prüfen:**
   ```bash
   # Im Projekt-Root-Verzeichnis
   cat .qca-aid-project.json  # Linux/Mac
   type .qca-aid-project.json  # Windows
   ```

3. **Versteckte Dateien anzeigen:**
   - Windows: Ansicht → Versteckte Elemente aktivieren
   - macOS: Cmd+Shift+. im Finder
   - Linux: Ctrl+H im Dateimanager

4. **Neu erstellen:**
   - `.qca-aid-project.json` löschen
   - Webapp neu starten
   - Projekt-Root neu setzen

### Webapp startet nicht
**Problem:** `ModuleNotFoundError: No module named 'streamlit'`  
**Lösung:** `pip install -r ../requirements.txt` oder `Start-QCA-AID-Webapp.bat` nutzen

### Port bereits belegt
**Problem:** `Port 8501 is already in use`  
**Lösung:** Streamlit wählt automatisch Port 8502, 8503, etc. oder `Start-QCA-AID-Webapp.bat` beendet alte Instanzen

### API-Schlüssel fehlt
**Problem:** `OpenAI API key not found`  
**Lösung:**
1. Erstelle `.environ.env` im Home-Verzeichnis
2. Füge hinzu: `OPENAI_API_KEY=sk-...`
3. Webapp neu starten

### Konfiguration lädt nicht
**Lösung:**
1. Dateiformat prüfen (XLSX oder JSON)
2. Beispielkonfiguration aus `../examples/` nutzen
3. JSON-Syntax validieren

### Analyse startet nicht
**Problem:** Keine Eingabedateien  
**Lösung:** Dateien in `../input/` Verzeichnis kopieren (.txt, .pdf, .docx)

### Performance-Probleme
**Lösung:**
1. Browser-Cache leeren (Strg+Shift+Entf)
2. Streamlit-Cache löschen: `rm -rf ~/.streamlit/cache`
3. Webapp neu starten

### Lange Pfade werden abgeschnitten

**Problem:** Vollständiger Pfad nicht sichtbar im Eingabefeld  
**Lösung:**
- Mit Maus über Pfad fahren → Tooltip zeigt vollständigen Pfad
- Pfad wird mit Ellipsis (...) gekürzt für bessere Lesbarkeit
- Vollständiger Pfad wird intern korrekt verwendet

### Datei-Browser startet im falschen Ordner

**Problem:** Dialog öffnet sich nicht im Projekt-Verzeichnis  
**Lösung:**
- Projekt-Root korrekt setzen
- Webapp neu starten
- Dialog sollte nun im Projekt-Root starten

### Weitere Probleme

**Rate limit exceeded:**
- Warten, BATCH_SIZE reduzieren, API-Guthaben prüfen

**Context length exceeded:**
- CHUNK_SIZE reduzieren (z.B. 1500 → 1000)
- CODE_WITH_CONTEXT deaktivieren

**Langsame Analyse:**
- BATCH_SIZE erhöhen (10-12)
- Schnelleres Modell wählen (gpt-4o-mini)

**PDF nicht lesbar:**
- Textebene prüfen (nicht nur Bilder)
- Als .txt exportieren

**Sonderzeichen in Vorschau:**
- Datei mit UTF-8 Encoding speichern

## API für Entwickler

### Architektur
```
UI Components (webapp_components/)
         ↓
Business Logic (webapp_logic/)
         ↓
Data Models (webapp_models/)
```

### Data Models

**ConfigData:**
```python
from webapp_models.config_data import ConfigData

config = ConfigData(
    model_provider="OpenAI",
    model_name="gpt-4o-mini",
    chunk_size=1000,
    # ...
)
config_dict = config.to_dict()
is_valid, errors = config.validate()
```

**CodebookData:**
```python
from webapp_models.codebook_data import CodebookData

codebook = CodebookData(
    forschungsfrage="...",
    kodierregeln={...},
    deduktive_kategorien={...}
)
```

**ProjectSettings:**
```python
from webapp_models.project_data import ProjectSettings

settings = ProjectSettings(
    project_root=Path("/path/to/project"),
    last_config_file=Path("config.json"),
    last_codebook_file=Path("codebook.json")
)
settings.save(Path(".qca-aid-project.json"))
```

**InductiveCodeData:**
```python
from webapp_models.inductive_code_data import InductiveCodeData

inductive_code = InductiveCodeData(
    name="Emergent_Theme",
    definition="...",
    source_file="analysis_2024-11-29.xlsx",
    is_inductive=True
)
```

### Business Logic

**ProjectManager:**
```python
from webapp_logic.project_manager import ProjectManager

manager = ProjectManager()
manager.set_root_directory(Path("/path/to/project"))
config_path = manager.get_config_path()
manager.save_settings()
```

**FileBrowserService:**
```python
from webapp_logic.file_browser_service import FileBrowserService

# Datei auswählen
file_path = FileBrowserService.open_file_dialog(
    title="Select Config",
    file_types=[("JSON files", "*.json"), ("Excel files", "*.xlsx")]
)

# Verzeichnis auswählen
dir_path = FileBrowserService.open_directory_dialog(
    title="Select Project Root"
)

# Datei speichern
save_path = FileBrowserService.save_file_dialog(
    title="Save Codebook",
    default_filename="codebook.json"
)
```

**InductiveCodeExtractor:**
```python
from webapp_logic.inductive_code_extractor import InductiveCodeExtractor

extractor = InductiveCodeExtractor(output_dir=Path("output"))
analysis_files = extractor.find_analysis_files()
has_codes = extractor.has_inductive_codes(analysis_files[0])
codes = extractor.extract_inductive_codes(analysis_files[0])
```

**CodeMerger:**
```python
from webapp_logic.code_merger import CodeMerger

merger = CodeMerger()
merged_codes, warnings = merger.merge_codes(
    deductive_codes=existing_codes,
    inductive_codes=imported_codes,
    source_file="analysis_2024-11-29.xlsx"
)
conflicts = merger.detect_conflicts(deductive_codes, inductive_codes)
```

**ConfigManager:**
```python
from webapp_logic.config_manager import ConfigManager

manager = ConfigManager()
config = manager.load_config("config.json", format="json")
manager.save_config(config, "output.json", format="json")
```

**CodebookManager:**
```python
from webapp_logic.codebook_manager import CodebookManager

manager = CodebookManager()
manager.add_category(name="...", definition="...", ...)
```

**FileManager:**
```python
from webapp_logic.file_manager import FileManager

manager = FileManager()
files = manager.list_files("../input", extensions=[".txt", ".pdf"])
```

**AnalysisRunner:**
```python
from webapp_logic.analysis_runner import AnalysisRunner

runner = AnalysisRunner()
runner.start_analysis(config_dict)
progress, status = runner.get_progress()
```

### Eigene UI-Komponente

```python
import streamlit as st

def render_custom_tab():
    st.header("Meine Erweiterung")
    
    if 'custom_data' not in st.session_state:
        st.session_state.custom_data = {}
    
    user_input = st.text_input("Eingabe")
    
    if st.button("Verarbeiten"):
        result = process_data(user_input)
        st.success(f"Ergebnis: {result}")
```

### Neuen Analysemodus hinzufügen

1. Erweitere `VALID_ANALYSIS_MODES` in `webapp_models/config_data.py`
2. Implementiere Logik in `../QCA_AID_assets/analysis/`
3. Füge Option in `webapp_components/config_ui.py` hinzu

## Tipps und Best Practices

### Performance-Optimierung

**Chunk-Größe:**
- Kurze Dokumente (< 5 Seiten): 800
- Mittlere Dokumente (5-20 Seiten): 1000
- Lange Dokumente (> 20 Seiten): 1500

**Batch-Größe:**
- Hohe Präzision: 3-4
- Standard: 5-8
- Hohe Geschwindigkeit: 10-12

### Workflow-Empfehlungen

1. Mit 2-3 Testdokumenten starten
2. Kodierqualität überprüfen
3. Kategorien und Regeln anpassen
4. Vollständige Analyse durchführen

### Dateiorganisation

**Gute Benennung:**
```
FH_Professor_2024-01-15.txt
Uni_Doktorand_2024-02-20.txt
```

**Vermeiden:**
```
interview1.txt
data.txt
final_version_v3_neu.txt
```

### Kategorien-Best Practices

**Gute Definition:**
```
Name: Technologieeinsatz_Lehre
Definition: Erfasst alle Aussagen über den konkreten Einsatz 
digitaler Technologien in Lehrveranstaltungen.

Regeln:
- Codiere nur explizite Aussagen
- Unterscheide geplante vs. tatsächliche Nutzung
- Berücksichtige didaktischen Kontext

Beispiele:
- "Wir nutzen Moodle für Lernmaterialien"
- "Studierende arbeiten mit Padlet"
```

**Vermeiden:**
- Vage Definitionen
- Zu breite Kategorien
- Überlappende Kategorien
- Fehlende Beispiele

## Debug und Erweiterte Fehlerbehebung

### Debug-Modus
```bash
streamlit run webapp.py --logger.level=debug
```

### Log-Dateien
```bash
cat ~/.streamlit/logs/streamlit.log
cat ../.crush/logs/crush.log
```

### Abhängigkeiten prüfen
```bash
pip list
pip show streamlit pandas openpyxl
pip install --force-reinstall streamlit
```

### Virtuelle Umgebung neu erstellen
```bash
deactivate
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r ../requirements.txt
```

## FAQ

**Kann ich die Webapp von einem anderen Computer erreichen?**  
Nein, nur localhost aus Sicherheitsgründen.

**Warum ist die Webapp langsamer als CLI?**  
Zusätzliche Features wie Echtzeit-Updates verursachen Overhead.

**Mehrere Analysen gleichzeitig?**  
Nein, nur eine Analyse pro Webapp-Instanz.

**Werden Daten in der Cloud gespeichert?**  
Nein, alle Daten bleiben lokal. Nur API-Anfragen gehen ins Internet.

## Weitere Ressourcen

- [Beispielkonfigurationen](../examples/) - Vorkonfigurierte Templates
- [Hauptdokumentation](../README.md) - QCA-AID Gesamtdokumentation
- [Changelog](../CHANGELOG.md) - Versionshistorie

## Support

Bei Fragen oder Problemen:
- GitHub Issues: https://github.com/JustusHenke/QCA-AID/issues
- E-Mail: justus.henke@hof.uni-halle.de

## Neue Features in Version 0.11.x

### Projekt-Management
- Projekt-Root-Verzeichnis festlegen und verwalten
- Automatische Speicherung und Wiederherstellung von Projekt-Einstellungen
- Einfacher Wechsel zwischen verschiedenen Projekten
- Relative Pfadauflösung für bessere Portabilität

### Datei-Browser Integration
- Native Datei-Browser-Dialoge für alle Dateioperationen
- Automatische Format-Erkennung (JSON/XLSX)
- Visuelle Pfad-Validierung mit Echtzeit-Feedback
- Intelligente Fehlerbehandlung und Vorschläge

### Induktive Code-Import
- Automatische Erkennung induktiver Codes aus Analysen
- Import-Dialog mit Datei-Vorschau und Metadaten
- Konflikt-Erkennung und Umbenennungsoptionen
- Separate Anzeige mit Quellenangabe
- Vollständige Integration in Codebook-Workflow

### UI-Verbesserungen
- Datei-Browser-Buttons (📁) neben allen Pfad-Eingaben
- Tooltips mit vollständigen Pfaden bei langen Dateinamen
- Erfolgsbestätigungen mit Dateiinformationen
- Verbesserte Fehlerbehandlung und Benutzerführung

**Version:** 0.11.1  
**Letzte Aktualisierung:** Dezember 2024

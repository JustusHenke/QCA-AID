# QCA-AID Explorer

## Überblick
QCA-AID Explorer ist ein leistungsstarkes Tool zur Analyse qualitativer Kodierungsdaten. Es ermöglicht die Visualisierung von Kodiernetzwerken mit Hauptkategorien, Subkategorien und Schlüsselwörtern sowie die automatisierte Zusammenfassung von kodierten Textsegmenten mithilfe von Large Language Models (LLMs).

**Version:** 0.4 (2025-04-07)

**Neuerungen in Version 0.4:**
- Konfiguration über Excel-Datei "QCA-AID-Explorer-Config.xlsx"
- Heatmap-Visualisierung von Codes entlang Dokumentattributen
- Mehrere Analysetypen konfigurierbar (Netzwerk, Heatmap, verschiedene Zusammenfassungen)
- Anpassbare Parameter für jede Analyse
- Verbessertes ForceAtlas2-Layout für Netzwerkvisualisierungen
- SVG-Export für bessere Bearbeitbarkeit

## Installation

### Voraussetzungen
- Python 3.8 oder höher
- Pip (Python Package Manager)

### Benötigte Pakete installieren
Führen Sie den folgenden Befehl aus, um alle erforderlichen Abhängigkeiten zu installieren:

```bash
pip install networkx reportlab scikit-learn pandas openpyxl matplotlib seaborn httpx python-dotenv openai mistralai python-docx numpy
```

### API-Schlüssel konfigurieren
Das Tool unterstützt folgende LLM-Provider:
- OpenAI
- Mistral AI

Erstellen Sie eine Datei namens `.environ.env` in Ihrem Benutzerverzeichnis mit folgendem Inhalt:

```
OPENAI_API_KEY=ihr_openai_api_schlüssel
MISTRAL_API_KEY=ihr_mistral_api_schlüssel
```

## Einrichtung

1. **Legen Sie den Skriptordner an:**
   - Erstellen Sie einen Ordner für das QCA-AID Explorer Skript
   - Erstellen Sie einen Unterordner namens "output" für die Ausgabedateien

2. **Konfigurationsdatei vorbereiten:**
   - Erstellen Sie eine Excel-Datei namens "QCA-AID-Explorer-Config.xlsx" im Skriptordner
   - Fügen Sie ein Basis-Sheet und mindestens ein Analysesheet hinzu (siehe unten)

3. **Datendatei platzieren:**
   - Legen Sie Ihre Kodierungsdaten-Excel-Datei im "output"-Ordner ab
   - Die Datei sollte ein Tabellenblatt namens "Kodierte_Segmente" mit Ihren qualitativen Daten enthalten

## Konfiguration

Die Konfiguration erfolgt vollständig über die Excel-Datei "QCA-AID-Explorer-Config.xlsx", die aus folgenden Teilen besteht:

### 1. Basis-Sheet
Das Blatt "Basis" enthält globale Parameter für alle Analysen:

| Parameter | Beschreibung | Beispielwert |
|-----------|--------------|--------------|
| provider | Name des LLM-Providers (openai oder mistral) | openai |
| model | Name des zu verwendenden Modells | gpt-4o-mini |
| temperature | Kreativitätsparameter (0.0 - 1.0) | 0.7 |
| script_dir | Pfad zum Skriptverzeichnis | C:/Projekte/QCA-Explorer |
| output_dir | Name des Ausgabeverzeichnisses | output |
| explore_file | Name der zu analysierenden Excel-Datei | Meine_Analyse_20250407.xlsx |
| clean_keywords | Harmonisierung von Schlüsselwörtern aktivieren | True |
| similarity_threshold | Schwellenwert für Ähnlichkeit bei Harmonisierung | 0.7 |

### 2. Analyseblätter
Jedes weitere Blatt definiert eine eigene Analyse und muss mindestens diese Parameter enthalten:

| Parameter | Beschreibung |
|-----------|--------------|
| analysis_type | Art der Analyse (netzwerk, heatmap, summary_paraphrase, summary_reasoning, custom_summary) |
| filter_* | Filter für die Datenauswahl (z.B. filter_Hauptkategorie, filter_Dokument) |

### Analysetypen und spezifische Parameter

#### Netzwerkanalyse (`analysis_type: netzwerk`)
Erstellt eine Netzwerkvisualisierung der Codes.

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| node_size_factor | Faktor für die Knotengröße | 1.0 |
| layout_iterations | Anzahl der Layout-Iterationen | 100 |
| gravity | Anziehungskraft zum Zentrum | 0.05 |
| scaling | Skalierungsfaktor für Abstände | 2.0 |

#### Heatmap (`analysis_type: heatmap`)
Erstellt eine Heatmap der Codehäufigkeiten.

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| x_attribute | Spalte für die X-Achse | Dokument |
| y_attribute | Spalte für die Y-Achse | Hauptkategorie |
| z_attribute | Werttyp (count oder percentage) | count |
| cmap | Farbpalette für die Heatmap | YlGnBu |
| figsize | Größe der Abbildung (Format: Breite,Höhe) | 14,10 |
| annot | Zeige Werte in der Heatmap | True |
| fmt | Formatierung der Zahlen | .0f |

#### Zusammenfassungen
Es gibt drei Arten von Zusammenfassungen:

1. **Paraphrasenzusammenfassung** (`analysis_type: summary_paraphrase`)
2. **Begründungszusammenfassung** (`analysis_type: summary_reasoning`)
3. **Benutzerdefinierte Zusammenfassung** (`analysis_type: custom_summary`)

Zusätzlicher Parameter für alle Zusammenfassungen:
| Parameter | Beschreibung |
|-----------|--------------|
| prompt_template | Angepasster Prompt für die KI (optional, Standard-Prompts werden verwendet, wenn nicht angegeben) |

## Ausführung

1. **Skript starten:**
   ```
   python qca-aid-explorer.py
   ```

2. **Ausgabedateien:**
   - Alle Ausgaben werden im "output"-Ordner gespeichert
   - Netzwerkvisualisierungen: PDF und SVG
   - Heatmaps: PDF und PNG
   - Netzwerk- und Heatmap-Daten: Excel-Dateien
   - Zusammenfassungen: Textdateien

## Tipps und Tricks

### Analysen überspringen
Es gibt drei Möglichkeiten, bestimmte Analysen zu überspringen:

1. **Sheets umbenennen:** Fügen Sie einen Unterstrich am Anfang des Sheet-Namens hinzu (z.B. "_Netzwerk1")
2. **Sheets ausblenden:** Blenden Sie Sheets aus, die nicht verarbeitet werden sollen
3. **Mehrere Konfigurationsdateien:** Erstellen Sie verschiedene Konfigurationsdateien für unterschiedliche Analysesätze

### Keyword-Harmonisierung
Die Keyword-Harmonisierung erkennt ähnliche Schlüsselwörter und vereinheitlicht sie:

- Aktivieren Sie diese Funktion mit `clean_keywords: True` im Basis-Sheet
- Passen Sie die Erkennungsschwelle mit `similarity_threshold` an (0.7 bedeutet 70% Ähnlichkeit)
- Die harmonisierten Begriffe werden in der Konsole ausgegeben

### Anpassung der Visualisierungen
- Experimentieren Sie mit verschiedenen Parametern für Netzwerke und Heatmaps
- Für Netzwerke: Erhöhen Sie `iterations` für bessere Layouts, passen Sie `gravity` für kompaktere/lockerere Netzwerke an
- Für Heatmaps: Probieren Sie verschiedene Farbpaletten mit `cmap` (z.B. "YlOrRd", "Blues", "viridis")

### Erstellen eigener Prompts
Bei benutzerdefinierten Zusammenfassungen können Sie Ihre eigenen Prompts erstellen:
- Verwenden Sie `{text}` als Platzhalter für die zu analysierenden Texte
- Verwenden Sie `{filters}` als Platzhalter für die aktiven Filter
- Strukturieren Sie den Prompt klar, um spezifischere Analysen zu erhalten

## Fehlerbehandlung

### Häufige Probleme

1. **API-Schlüssel nicht gefunden:**
   - Stellen Sie sicher, dass die Datei `.environ.env` im Benutzerverzeichnis existiert
   - Überprüfen Sie, ob die API-Schlüssel korrekt sind

2. **Excel-Datei nicht gefunden:**
   - Überprüfen Sie, ob der Pfad zur Konfigurationsdatei korrekt ist
   - Stellen Sie sicher, dass die Analyse-Datei im "output"-Ordner liegt

3. **Fehler bei der Keyword-Harmonisierung:**
   - Reduzieren Sie den Wert für `similarity_threshold`
   - Deaktivieren Sie die Harmonisierung mit `clean_keywords: False`

4. **Layoutprobleme bei Netzwerken:**
   - Versuchen Sie, `iterations` zu erhöhen (z.B. auf 200)
   - Passen Sie `gravity` und `scaling` an
   - Bei sehr großen Netzwerken: Filtern Sie die Daten stärker

## Beispiel für eine Konfigurationsdatei

### Basis-Sheet
| Parameter | Wert |
|-----------|------|
| provider | openai |
| model | gpt-4o-mini |
| temperature | 0.7 |
| script_dir | C:/MeinProjekt |
| output_dir | output |
| explore_file | QCA-Analyse_2025.xlsx |
| clean_keywords | True |
| similarity_threshold | 0.7 |

### Netzwerkanalyse-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Hauptkategorie | Herausforderungen |
| filter_Dokument | Interview_01 |
| analysis_type | netzwerk |
| node_size_factor | 1.2 |
| gravity | 0.08 |

### Heatmap-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Hauptkategorie |  |
| analysis_type | heatmap |
| x_attribute | Dokument |
| y_attribute | Hauptkategorie |
| cmap | YlOrRd |

### Zusammenfassung-Sheet
| Parameter | Wert |
|-----------|------|
| filter_Dokument | Interview_03 |
| analysis_type | summary_paraphrase |
| prompt_template | Bitte analysieren Sie die folgenden Texte und identifizieren Sie die Hauptthemen: {text} |

## Kontakt und Unterstützung

Bei Fragen oder Problemen wenden Sie sich bitte an den Entwickler oder öffnen Sie ein Issue im Quell-Repository.

---

**Hinweis:** Stellen Sie sicher, dass Sie über die erforderlichen Rechte für die verwendeten API-Schlüssel und Daten verfügen. QCA-AID Explorer ist ein Werkzeug zur qualitativen Analyse und übernimmt keine Verantwortung für die Ergebnisse oder den Inhalt der generierten Analysen.

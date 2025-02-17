![QCA-AID](banner-qca-aid.png)

# QCA-AID: Qualitative Content Analysis with AI Support - Deductive Coding

Dieses Python-Skript implementiert Mayrings Methode der deduktiven Qualitativen Inhaltsanalyse mit induktiver Erweiterung mit KI-Unterstützung durch die OpenAI API. Es kombiniert traditionelle qualitative Forschungsmethoden mit modernen KI-Fähigkeiten, um Forschende bei der Analyse von Dokumenten- und Interviewdaten zu unterstützen. Das Ziel dieses Tools ist nicht, die menschliche Arbeit der Inhaltsanalyse zu ersetzen, sondern neue Möglichkeiten zu eröffnen, mehr Zeit für die Analyse und Reflexion bereits vorstrukturierter Textdaten zu gewinnen. 

Chancen:
- Es ermöglicht mehr Dokumente in einer Untersuchung zu berücksichtigen als in herkömmlichen Verfahren, bei denen Personalkapazitäten stark begrenzt sind.    
- Es ermöglicht die Umsetzung von Intercoder-Vergleichen mittels zugeschalteten KI-Coder, wo sonst nur ein menschlicher Coder pro Dokument arbeiten würde, und kann damit zur Qualitätsverbesserung beitragen
- QCA-AID kann auch ganz ohne KI-Coder genutzt werden, als Alternative zu kostenpflichtigen Programmen.
- Es ermöglicht zusätzliche explorative Dokumentenanalysen, die sonst aus pragmatischen Gründen mit einfacheren Verfahren umgesetzt würden

Zu beachten ist aber:
- Gefahr der Überkonfidenz in eine automatisiert ermittelte Struktur der Daten 
- Bei geringer Anzahl von Dokumenten überwiegen weiterhin die Vorteile menschlicher Kodierung (Close-reading, Kontextverständnis, Erfahrung)

## Merkmale von QCA-AID

- Automatisierte Textvorverarbeitung und Chunking
- Deduktive Anwendung von Kategorien
- Induktive Kategorienentwicklung (kann übersprungen werden)
- Multi-Coder-Unterstützung (AI und Mensch)
- Fähigkeiten zur Zusammenführung und Aufteilung von Kategorien
- Berechnung der Intercoder-Zuverlässigkeit
- Überarbeitung und Optimierung des Kategoriesystems
- Iterativer Analyseprozess mit Sättigungsprüfungen
- Umfassender Analyseexport
- Export des erweiterten Codebooks
- Attributbasierte Analyse für demografische oder kontextbezogene Faktoren
- Konfigurierbare Analyseparameter und Schwellenwerte
- Detaillierte Dokumentation des Kodierungsprozesses
- Schätzung der verwendeten Input-/Output-Tokens der API-Aufrufe

__ACHTUNG!__
Bitte beachten Sie, dass sich dieses Skript noch in der Entwicklung befindet und noch nicht alle Funktionen verfügbar sind oder zuverlässig arbeiten. Es wird aktuell eine Nutzung zu Testzwecken empfohlen. 
Beachten Sie auch, dass KI-Ergebnisse nicht perfekt sind und die Ergebnisse von der Qualität der Eingabedaten abhängen.
Verwenden Sie das Skript auf eigene Gefahr! 

--> Feedback ist willkommen! <--
Kontakt: justus.henke@hof.uni-halle.de

## Hinweis zum Datenschutz

Die KI-gestützte Datenverarbeitung nutzt die Schnittstelle von OpenAI. Auch wenn diese Anfragen offiziell nicht für das Training von Modellen genutzt werden, stellt diese eine Verarbeitung durch Dritte dar. Prüfen Sie, ob Ihre Dokumente dafür freigegeben sind und entfernen Sie ggf. sensible Informationen. Eine Nutzung mit hochsensiblen Daten wird ausdrücklich nicht empfohlen. 

## Weitere Hinweise zur aktuellen Version 

- Sollte die induktive Kodierung zu großzügig sein und zu viele Subcodes erstellen, kann können Sie den CONFIG-Wert `Temperature` herunterregeln (z.B. auf '0.1'), dann wird konservativer kodiert. 
- Momentan wird nur Konsensentscheidung der Kodierer zugelassen, Mehrheitsvoting (bei n>2 Kodierern) oder Manuelles Review bei unterschiedlichen Kodierungen für ein Segment  ist noch nicht implementiert. 
- Die Konsensbildung erfolgt in einem mehrstufigen Prozess: Zunächst wird die Hauptkategorie mit der höchsten Übereinstimmung unter den Kodierern bestimmt, wobei bei Gleichstand die Kategorie mit der höchsten durchschnittlichen Konfidenz gewählt wird. Anschließend werden Subkategorien identifiziert, die von mindestens 50 % der Kodierer genutzt wurden, und die finale Konsens-Kodierung basiert auf der qualitativ besten Einzelskodierung mit den ermittelten Konsens-Subkategorien.

## Installation

1. Klonen Sie dieses Repository
2. Installieren Sie die benötigten Pakete:
   ```bash
   pip install -r requirements.txt
   ```

## Speichern des API-Schlüssels

Um den API-Schlüssel sicher zu speichern und zu verwenden, folgen Sie diesen Schritten:

1. **Erstellen Sie eine .environ.env Datei**:
   - Die Datei sollte im Home-Verzeichnis Ihres Benutzers erstellt werden.
   - Unter Windows ist dies typischerweise: `C:\Users\IhrBenutzername\`
   - Unter macOS und Linux: `/home/IhrBenutzername/`

2. **Dateiinhalt**:
   - Öffnen Sie die .environ.env Datei mit einem Texteditor.
   - Fügen Sie folgende Zeile hinzu, ersetzen Sie dabei `IhrAPISchlüssel` mit Ihrem tatsächlichen API-Schlüssel:
     ```
     OPENAI_API_KEY=IhrAPISchlüssel
     ```

3. **Speichern der Datei**:
   - Speichern Sie die Datei und schließen Sie den Texteditor.

4. **Sicherheitshinweis**:
   - Stellen Sie sicher, dass die .environ.env Datei nicht in öffentliche Repositories hochgeladen wird.
   - Fügen Sie `.environ.env` zu Ihrer .gitignore Datei hinzu, wenn Sie Git verwenden.

5. **Verwendung im Code**:
   - Das Programm liest den API-Schlüssel automatisch aus dieser Datei.
   - Der Pfad zur Datei wird wie folgt definiert:
     ```python
     env_path = os.path.join(os.path.expanduser("~"), '.environ.env')
     ```

6. **Überprüfung**:
   - Stellen Sie sicher, dass die Umgebungsvariable korrekt geladen wird, indem Sie das Programm starten.
   - Falls Probleme auftreten, überprüfen Sie den Dateipfad und den Inhalt der .environ.env Datei.

Durch diese Methode wird Ihr API-Schlüssel sicher gespeichert und ist für das Programm zugänglich, ohne dass er direkt im Quellcode erscheint.

## Unterstützte Eingabedateien

Das Programm kann bestimmte Dateitypen im Eingabeverzeichnis (DATA_DIR) verarbeiten. Folgende Dateiformate werden derzeit unterstützt:

1. **Textdateien**:
   - .txt (Plain Text)

2. **Dokumentformate**:
   - .pdf (Portable Document Format)

3. **Tabellenkalkulationen**:
   - .xlsx (Microsoft Excel)

Hinweise zur Verwendung:
- Stellen Sie sicher, dass Ihre Eingabedateien in einem der oben genannten Formate vorliegen.
- Das Programm liest alle unterstützten Dateien im Eingabeverzeichnis automatisch ein.
- Bei der Verwendung von PDF-Dateien wird der Text extrahiert; komplexe Formatierungen oder eingebettete Bilder werden dabei nicht berücksichtigt.
- Excel-Dateien (.xlsx) sollten strukturierte Daten enthalten, wobei jede Zeile als separater Textabschnitt behandelt wird.

Für optimale Ergebnisse empfehlen wir die Verwendung von einfachen Textformaten (.txt), insbesondere für längere Textpassagen oder Transkripte.

**Wichtig**: 
- Stellen Sie sicher, dass alle Dateien im Eingabeverzeichnis für die Analyse relevant sind, da das Programm versuchen wird, jede unterstützte Datei zu verarbeiten.
- Andere Dateiformate wie .docx, .csv, .md, .srt oder .vtt werden derzeit nicht unterstützt. Konvertieren Sie diese gegebenenfalls in eines der unterstützten Formate.


## Verwendung der Codebook.xlsx

Die `Codebook.xlsx` Datei ist ein zentrales Element für die Konfiguration und Anpassung der qualitativen Inhaltsanalyse. Sie enthält wichtige Informationen wie die Forschungsfrage, Kodierregeln und deduktive Kategorien. Folgen Sie diesen Schritten, um die Datei effektiv zu nutzen:

1. **Dateistruktur**: Die Excel-Datei besteht aus mehreren Blättern:
   - FORSCHUNGSFRAGE
   - KODIERREGELN
   - DEDUKTIVE_KATEGORIEN
   - CONFIG

2. **FORSCHUNGSFRAGE**: 
   - Tragen Sie Ihre Forschungsfrage in Zelle A2 ein.

3. **KODIERREGELN**:
   - Spalte A: Allgemeine Kodierregeln
   - Spalte B: Formatregeln
   - Fügen Sie pro Zeile eine Regel hinzu.

4. **DEDUKTIVE_KATEGORIEN**:
   - Spalte A (Key): Hauptkategoriename
   - Spalte B (Sub-Key): Art der Information (definition, rules, examples, subcategories)
   - Spalte C (Sub-Sub-Key): Nur für Subkategorien relevant
   - Spalte D (Value): Inhalt der jeweiligen Information
   - Beispiel:
     ```
     | Key       | Sub-Key     | Sub-Sub-Key | Value                        |
     |-----------|-------------|-------------|------------------------------|
     | Akteure   | definition  |             | Erfasst alle handelnden...   |
     | Akteure   | rules       |             | Codiere Aussagen zu: Indi... |
     | Akteure   | examples    | [0]         | Die Arbeitsgruppe trifft...  |
     | Akteure   | subcategories | Individuelle_Akteure | Einzelpersonen und deren... |
     ```

5. **CONFIG**:
   - Hier können Sie verschiedene Konfigurationsparameter einstellen:
     - MODEL_NAME: Name des zu verwendenden Sprachmodells
     - DATA_DIR: Verzeichnis für Eingabedaten
     - OUTPUT_DIR: Verzeichnis für Ausgabedaten
     - CHUNK_SIZE: Größe der Textabschnitte für die Analyse
     - CHUNK_OVERLAP: Überlappung zwischen Textabschnitten
     - ATTRIBUTE_LABELS: Bezeichnungen für Attribute
     - CODER_SETTINGS: Einstellungen für automatische Kodierer

6. **Aktualisierung**: 
   - Speichern Sie Ihre Änderungen in der Excel-Datei.
   - Das Programm liest die aktualisierten Informationen beim nächsten Start automatisch ein.

7. **Validierung**:
   - Das System überprüft die Vollständigkeit und Konsistenz der eingegebenen Daten.
   - Achten Sie auf Warnmeldungen und Hinweise zur Verbesserung des Kategoriensystems.

Durch sorgfältige Pflege und Aktualisierung der `Codebook.xlsx` können Sie die Analyse optimal an Ihre Forschungsfragen und -methoden anpassen.

## Ausgabedateien

Das Programm speichert verschiedene Ausgabedateien im konfigurierten Ausgabeverzeichnis (OUTPUT_DIR). Hier finden Sie eine Übersicht der erstellten Dateien und deren Inhalte:

1. **Ergebnisse der Analyse (Excel-Mappe)**:
   - Dateiname: `QCA_Results_[DATUM]_[ZEIT].xlsx`
   - Inhalt: Eine umfassende Excel-Mappe mit mehreren Tabellenblättern, die verschiedene Aspekte der Analyse abdecken:
     - Kodierergebnisse
     - Kategoriensystem
     - Zusammenfassende Statistiken
     - Visualisierungen (falls vorhanden)
   - Diese Datei dient als Hauptquelle für die Analyse und Interpretation der Ergebnisse.

2. **Kategorie-Revisionen**:
   - Dateiname: `category_revisions_[DATUM]_[ZEIT].json`
   - Inhalt: Ein detailliertes Protokoll aller Änderungen am Kategoriensystem während des Analyseprozesses. Dies umfasst:
     - Hinzufügungen neuer Kategorien
     - Modifikationen bestehender Kategorien
     - Löschungen von Kategorien
     - Zeitstempel und Begründungen für jede Änderung

Alle Dateien sind mit Datum und Uhrzeit versehen, um mehrere Analysedurchläufe unterscheiden zu können. Diese Ausgaben bieten eine umfassende Dokumentation des Analyseprozesses und ermöglichen eine detaillierte Nachverfolgung und Auswertung der Ergebnisse.

**Hinweis**: Stellen Sie sicher, dass Sie über ausreichend Speicherplatz im Ausgabeverzeichnis verfügen, insbesondere bei der Analyse großer Datenmengen oder bei mehreren Durchläufen.


# QCA-Mayring: Qualitative Content Analysis with AI Support

This Python script implements Mayring's Qualitative Content Analysis methodology with AI support through the OpenAI API. It combines traditional qualitative research methods with modern AI capabilities to assist researchers in analyzing interview data.

__ATTENTION!__
Please note that this script is still under development and not all functions are available yet. Please also note
Please also note that AI-Assistance is not perfect and the results depend on the quality of the input data.
Use the script at your own risk!
Feedback is welcome!

## Features

- Automated text preprocessing and chunking
- Deductive category application
- Inductive category development
- Multi-coder support (AI and human)
- Category merging and splitting capabilities
- Intercoder reliability calculation
- Category system revision and optimization
- Iterative analysis process with saturation checks
- Comprehensive analysis export
- Attribute-based analysis for demographic or contextual factors
- Configurable analysis parameters and thresholds
- Detailed documentation of the coding process

## Note on data protection

AI-supported data processing uses the OpenAI interface. Even if these requests are not officially used for training models, this constitutes processing by third parties. Check whether your documents are approved for this and remove sensitive information if necessary. Use with highly sensitive data is expressly not recommended.

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```



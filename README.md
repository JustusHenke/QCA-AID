![QCA-AID](banner-qca-aid.png)

# QCA-AID: Qualitative Content Analysis with AI Support - Deductive Coding

Dieses Python-Skript implementiert Mayrings Methode der deduktiven Qualitativen Inhaltsanalyse mit induktiver Erweiterung mit KI-Unterstützung durch die OpenAI API. Es kombiniert traditionelle qualitative Forschungsmethoden mit modernen KI-Fähigkeiten, um Forschende bei der Analyse von Dokumenten- und Interviewdaten zu unterstützen. Das Ziel dieses Tools ist nicht, die menschliche Arbeit der Inhaltsanalyse zu ersetzen, sondern neue Möglichkeiten zu eröffnen, mehr Zeit für die Analyse und Reflexion bereits vorstrukturierter Textdaten zu gewinnen. 

Anwendungsmöglichkeiten:
- Es ermöglicht mehr Dokumente in einer Untersuchung zu berücksichtigen als in herkömmlichen Verfahren, bei denen Personalkapazitäten stark begrenzt sind.    
- Es ermöglicht die Umsetzung von Intercoder-Vergleichen mittels zugeschalteten KI-Coder, wo sonst nur ein menschlicher Coder pro Dokument arbeiten würde, und kann damit zur Qualitätsverbesserung beitragen
- QCA-AID kann auch ganz ohne KI-Coder genutzt werden, als Alternative zu kostenpflichtigen Programmen.
- Es ermöglicht zusätzliche explorative Dokumentenanalysen, die sonst aus pragmatischen Gründen mit einfacheren Verfahren umgesetzt würden

Zu beachten ist aber:
- Gefahr der Überkonfidenz in eine automatisiert ermittelte Struktur der Daten 
- Bei geringer Anzahl von Dokumenten überwiegen weiterhin die Vorteile menschlicher Kodierung (Close-reading, Kontextverständnis, Erfahrung)

## Merkmale von QCA-AID

- Automatisierte Textvorverarbeitung und Chunking
- Relevanzprüfung der Textsegmente vor der Kodierung
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
Bitte beachten Sie, dass sich dieses Skript noch in der Entwicklung befindet und noch nicht alle Funktionen verfügbar sind oder zuverlässig arbeiten. Es wird aktuell eine Nutzung zu Testzwecken empfohlen. Prüfen Sie regelmäßig, ob eine neue Version hier bereitgestellt ist.
Beachten Sie auch, dass KI-Ergebnisse nicht perfekt sind und die Ergebnisse von der Qualität der Eingabedaten abhängen.
Verwenden Sie das Skript auf eigene Gefahr! 

--> Feedback ist willkommen! <--
Kontakt: justus.henke@hof.uni-halle.de

## Hinweis zum Datenschutz

Die KI-gestützte Datenverarbeitung nutzt die Schnittstelle von OpenAI bzw. Mistral. Auch wenn diese Anfragen offiziell nicht für das Training von Modellen genutzt werden, stellt diese eine Verarbeitung durch Dritte dar. Prüfen Sie, ob Ihre Dokumente dafür freigegeben sind und entfernen Sie ggf. sensible Informationen. Eine Nutzung mit hochsensiblen Daten wird ausdrücklich nicht empfohlen. 

Prinzipiell ist die Verarbeitung der Daten per LLM auch auf einem lokalen Rechner möglich Dafür kann OLLAMA oder LMSTUDIO genutzt werden und das Setup im Client muss etwas angepasst werden mehr dazu hier: https://ollama.com/blog/openai-compatibility oder https://lmstudio.ai/docs/api/endpoints/openai

## Weitere Hinweise zur aktuellen Version (0.9.7)

- NEU: Mistral Support! Es kann jetzt auch die Mistral API genutzt werden. Umschalten zwischen OpenAI und Mistral mit CONFIG-Parameter 'MODEL_PROVIDER'. Standardmodell für OpenAI ist 'GPT-4o-mini', für Mistral 'mistral-small'.
- NEU: Ausschlusskriterien während der Relevanzprüfung in 'KODIERREGELN' definieren (z.B. Literaturverzeichnis)
- NEU: Hinzufügen von Ausschlusskriterien für die Relevanzprüfung in Codebuch-Kodierregeln
- NEU: Export von Begründungen für nicht relevante Textsegmente
- Verbesserte Relevanzprüfung, Rechtfertigung und Aufforderung zur Kodierung von Segmenten
- NEU: Erstellen von Zusammenfassungen und Diagrammen aus Ihren kodierten Daten mit 'QCA-AID-Explorer.py'.

- Bei größeren Mengen an Texten kann es im induktiven Modus immer wieder mal zu übermäßigen Vergaben von Subkategorien kommen. Das entsprechende Prompting, das diese Ergebnisse produziert wird noch weiter verfeinert. 
- Sollte die induktive Kodierung zu großzügig sein und zu viele Subcodes erstellen, kann können Sie den CONFIG-Wert `Temperature` herunterregeln (z.B. auf '0.1'), dann wird konservativer kodiert. 
- Beachten Sie, dass die Forschungsfrage am besten alle Aspekte der Hauptkategorien abdeckt bzw. letztere sich aus der Frage ableiten lassen. Damit ist eine zuverlässigere Kodierung möglich, da die Forschungsfrage zentral ist, um ein Textsegment als relevant vorauszuwählen. Die Forschungsfrage sollte die Aspekte der Hauptkategorien möglichst ausgewogen adressieren und nicht bereits eine Hauptkategorie bevorzugen (es sei denn, das ist beabsichtigt).
- Während der Bearbeitung werden mehrere API-Calls durchgeführt (Relevanzprüfung, Code-Entwicklung, Sättigungsprüfung), die Verarbeitung von Texten ist also relativ langsam: Ca. 400 Textsegmente à 1.000 Zeichen je Stunde, also ca. 200-250 Seiten je Stunde.  
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
     MISTRAL_API_KEY=IhrAPISchhlüssel
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
   - .docx (Microsoft Word)

Hinweise zur Verwendung:
- Stellen Sie sicher, dass Ihre Eingabedateien in einem der oben genannten Formate vorliegen.
- Das Programm liest alle unterstützten Dateien im Eingabeverzeichnis automatisch ein.
- Bei der Verwendung von PDF-Dateien wird der Text extrahiert; komplexe Formatierungen oder eingebettete Bilder werden dabei nicht berücksichtigt.

Für optimale Ergebnisse  wird die Verwendung von einfachen Textformaten (.txt) empfohlen, insbesondere für längere Textpassagen oder Transkripte. Entfernen Sie Literaturverzeichnisse und andere Textteile, die nicht kodiert werden sollen.

**Wichtig**: 
- Stellen Sie sicher, dass alle Dateien im Eingabeverzeichnis für die Analyse relevant sind, da das Programm versuchen wird, jede unterstützte Datei zu verarbeiten.
- Andere Dateiformate wie .csv, .md, .srt oder .vtt werden derzeit nicht unterstützt. Konvertieren Sie diese gegebenenfalls in eines der unterstützten Formate.

## QCA-AID: Konfiguration und Nutzung

QCA-AID unterstützt die qualitative Inhaltsanalyse nach Mayring mit KI-Unterstützung. Diese Anleitung erklärt die wichtigsten Schritte zur Konfiguration und Nutzung.

### Codebook.xlsx

Die Excel-Datei `Codebook.xlsx` ist zentral für die Konfiguration der Analyse und enthält:

#### Tabellenblätter
- **FORSCHUNGSFRAGE**: Tragen Sie Ihre Forschungsfrage in Zelle B1 ein
- **KODIERREGELN**: Allgemeine Kodierregeln (Spalte A), Formatregeln (Spalte B), Ausschlusskriterien für die Relevanzprüfung (Spalte C)
- **DEDUKTIVE_KATEGORIEN**: Hauptkategorien mit Definition, Regeln, Beispielen und Subkategorien
- **CONFIG**: Technische Einstellungen wie Modell, Verzeichnisse und Chunk-Größen

#### Struktur der DEDUKTIVE_KATEGORIEN

     | Key       | Sub-Key     | Sub-Sub-Key | Value                        |
     |-----------|-------------|-------------|------------------------------|
     | Akteure   | definition  |             | Erfasst alle handelnden...   |
     | Akteure   | rules       |             | Codiere Aussagen zu: Indi... |
     | Akteure   | examples    | [0]         | Die Arbeitsgruppe trifft...  |
     | Akteure   | subcategories | Individuelle_Akteure | Einzelpersonen und deren... |

#### Struktur der CONFIG
Hier können Sie verschiedene Konfigurationsparameter einstellen:
- MODEL_PROVIDER: Name des LLM-Anbieters ('OpenAI' oder 'Mistral')
- MODEL_NAME: Name des zu verwendenden Sprachmodells
- DATA_DIR: Verzeichnis für Eingabedaten
- OUTPUT_DIR: Verzeichnis für Ausgabedaten
- CHUNK_SIZE: Größe der Textabschnitte für die Analyse
- CHUNK_OVERLAP: Überlappung zwischen Textabschnitten
- ATTRIBUTE_LABELS: Bezeichnungen für Attribute, die aus dem Dateinamen extrahiert werden (z.B. "Part1_Part2_Restname.txt")
- CODER_SETTINGS: Einstellungen für automatische Kodierer

### Verzeichnisstruktur

#### Eingabeverzeichnis (input)
- Standardpfad: `input/` im Skriptverzeichnis
- Unterstützte Formate:
  - .txt (Textdateien)
  - .pdf (PDF-Dokumente)
  - .docx (Word-Dokumente)
- Namenskonvention: `attribut1_attribut2_weiteres.extension`
  - Beispiel: `university-type_position_2024-01-01.txt`
  - Die Attribute werden für spätere Analysen genutzt

#### Ausgabeverzeichnis (output)
- Standardpfad: `output/` im Skriptverzeichnis
- Erzeugte Dateien:
  - `QCA-AID_Analysis_[DATUM].xlsx`: Hauptergebnisdatei mit Kodierungen und Analysen
  - `category_revisions.json`: Protokoll der Kategorienentwicklung
  - `codebook_inductive.json`: Erweitertes Kategoriensystem nach induktiver Phase

### Wichtige Hinweise
- Entfernen Sie am besten Literaturverzeichnisse und nicht zu kodierende Textteile aus den Eingabedokumenten
- Prüfen Sie bei PDF-Dokumenten die korrekte Textextraktion
- Sichern Sie regelmäßig die QCA-AID-Codebook.xlsx
- Die Verzeichnispfade können in der CONFIG angepasst werden


Durch sorgfältige Pflege und Aktualisierung der `QCA-AID-Codebook.xlsx` können Sie die Analyse optimal an Ihre Forschungsfragen und -methoden anpassen. Insbesondere die DEDUKTIVEN_KATEGORIEN sollten gründlich mit Definitionen und Beispielen versorgt werden, um das LLM nötigen Kontext mitzuliefern. 

### CODE_WITH_CONTEXT
----------------
Wenn CONFIG-Parameter `CODE_WITH_CONTEXT` aktiviert ist (True), nutzt QCA-AID einen progressiven Dokumentkontext für die Kodierung.
Dabei wird für jedes Dokument ein fortlaufend aktualisiertes Summary erstellt, das bei
der Kodierung der nachfolgenden Chunks als Kontext verwendet wird.

Vorteile:
- Bessere Kontextsicherheit durch Berücksichtigung vorheriger Dokumentinhalte
- Verbesserte Kodierqualität bei kontextabhängigen Kategorien (z.B. "dominante Akteure")
- Mehr Konsistenz in der Kodierung eines Dokuments

Nachteile:
- Dokumente müssen sequentiell verarbeitet werden
- Geringer erhöhter Tokenverbrauch
- Mögliche Fehlerfortpflanzung bei falsch interpretierten frühen Abschnitten

Empfehlung:
- Für Analysen mit hierarchischen oder relationalen Kategorien aktivieren
- Für einfache thematische Kategorisierungen kann ohne Kontext gearbeitet werden


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



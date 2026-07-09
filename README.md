![QCA-AID](banner-qca-aid.png)

# QCA-AID: Qualitative Content Analysis – AI-supported Discovery

**KI-unterstützte Qualitative Inhaltsanalyse nach Mayring**  
Für Sozialwissenschaftler:innen, die größere Textmengen systematisch auswerten wollen – ohne Programmierkenntnisse.

> **Ziel:** Nicht die menschliche Analyse ersetzen, sondern mehr Zeit für Reflexion und Interpretation gewinnen, indem KI die Vorstrukturierung übernimmt.

---

## 📋 Inhaltsverzeichnis

1. [Schnellstart (2 Minuten)](#schnellstart-2-minuten)
2. [Webapp-Bedienung – alle Einstellungen erklärt](#webapp-bedienung--alle-einstellungen-erklärt)
3. [Analyse-Modi im Überblick](#analyse-modi-im-überblick)
4. [Dokumente vorbereiten](#dokumente-vorbereiten)
5. [Ergebnisse nutzen](#ergebnisse-nutzen)
6. [Best Practices & Qualitätssicherung](#best-practices--qualitätssicherung)
7. [Wichtige Hinweise](#wichtige-hinweise)
8. [Häufige Probleme & Lösungen](#häufige-probleme--lösungen)

> **📖 Ausführliches Handbuch:** [`QCA-AID-Nutzerhandbuch.md`](QCA-AID-Nutzerhandbuch.md)  
> **📦 Neueste Änderungen:** [`CHANGELOG.md`](CHANGELOG.md)

---

## ⚡ Schnellstart (2 Minuten)

### 1. Python installieren

**Wichtig:** Nur **Python 3.10 oder 3.11** – Python 3.12/3.13 werden nicht unterstützt!

| Plattform | Vorgehen |
|-----------|----------|
| **Windows** | [python.org](https://www.python.org/downloads/release/python-3110/) → `python-3.11.0-amd64.exe` herunterladen → **☑ "Add Python to PATH"** aktivieren → Installieren |
| **macOS** | `brew install python@3.11` (mit [Homebrew](https://brew.sh)) oder Installer von [python.org](https://www.python.org/downloads/release/python-3110/) |
| **Linux** | `sudo apt install python3.11 python3.11-venv` (Debian/Ubuntu) oder `sudo dnf install python3.11` (Fedora) |

Prüfen: `python --version` → `Python 3.11.x`

### 2. QCA-AID herunterladen

```bash
# Empfohlen: Mit Git (Terminal / Eingabeaufforderung)
git clone https://github.com/JustusHenke/QCA-AID.git
cd QCA-AID
```

> **Ohne Git:** Auf der [GitHub-Seite](https://github.com/JustusHenke/QCA-AID) auf "Code" → "Download ZIP" → entpacken → ins Verzeichnis wechseln.

### 3. Installation (automatisch)

| Plattform | Befehl |
|-----------|--------|
| **Windows** | Doppelklick auf `setup.bat` **oder** im Terminal: `setup.bat` |
| **macOS / Linux** | Terminal: `pip3 install -r requirements.txt && python3 -m spacy download de_core_news_sm` |

> 🔧 Bei Fehlern unter **Windows**: [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/) installieren.  
> 🔧 **setup.bat schließt sofort?** → `setup_debug.bat` nutzen oder manuell installieren (siehe [Nutzerhandbuch](QCA-AID-Nutzerhandbuch.md#13-häufige-probleme-und-lösungen)).

### 4. API-Schlüssel (nur für Cloud-Modelle)

Im QCA-AID-Ordner eine `.env`-Datei erstellen und API-Keys eintragen:

```bash
OPENAI_API_KEY=sk-proj-...
# ANTHROPIC_API_KEY=sk-ant-...     # optional
# MISTRAL_API_KEY=...               # optional
# OPENROUTER_API_KEY=sk-or-...      # optional
```

> **Kein API-Key nötig** bei Nutzung lokaler Modelle (LM Studio / Ollama) – 100% datenschutzkonform und offline-fähig.  
> **Seit v0.12.7.4:** `.env` wird automatisch geladen – keine manuelle Konfiguration nötig.

### 5. Webapp starten 🚀

```bash
python start_webapp.py
# Browser öffnet automatisch: http://127.0.0.1:8501
```

**Fertig!** Sie sehen nun die QCA-AID Webapp – alle weiteren Schritte direkt im Browser.

> **🐍 CLI-Alternative:** QCA-AID kann auch per Kommandozeile genutzt werden.  
> Siehe [Nutzerhandbuch § 5.4](QCA-AID-Nutzerhandbuch.md#54-qca-aid-per-kommandozeile-cli-nutzen) oder:
> ```bash
> pip install -e . --no-build-isolation   # einmalig
> qcaaid --help                            # Hauptanalyse
> qcaaid-explorer --help                   # Explorer
> qcaaid-webapp                            # Webapp
> ```

---

## 🖥️ Webapp-Bedienung – alle Einstellungen erklärt

Die Webapp hat **4 Hauptbereiche** (Tabs): Konfiguration → Codebook → Analyse → Explorer.

### 📌 Tab: Konfiguration (Modell & technische Parameter)

Hier stellen Sie ein, **welche KI** Ihre Texte analysiert und **wie** die Verarbeitung abläuft.

#### LLM-Anbieter & Modell

| Einstellung | Erklärung | Empfehlung |
|------------|-----------|------------|
| **Anbieter (Provider)** | Wer liefert die KI? | **Anfänger:** OpenAI ⭐ · **Datenschutz:** Local (LM Studio/Ollama) |
| **Modell** | Welches KI-Modell? | Einsteiger: `gpt-4o-mini` · Lokal: `llama3.1:8b` |
| **🔄 Lokale Modelle erkennen** | Sucht nach laufenden LM Studio/Ollama-Servern | Einmal klicken nach LM-Studio-Start |
| **🔧 Custom API Base URL** | Für eigene KI-Endpunkte (z.B. Uni-Server GWDG, Azure). **Sichtbar bei ALLEN Providern** – sobald gesetzt, wird das Modell-Feld zum Freitext-Eingabefeld | Nur wenn von Ihrer Institution bereitgestellt |
| **API-Key Variable** | Name der Umgebungsvariable für den API-Key (z.B. `GWDG_API_KEY` statt `OPENAI_API_KEY`) | Nur bei Custom-Endpoints nötig |

**So richten Sie lokale Modelle ein (datenschutzkonform):**

| Schritt | LM Studio (einfach) | Ollama (fortgeschritten) |
|---------|-------------------|--------------------------|
| Download | [lmstudio.ai](https://lmstudio.ai/) | [ollama.ai](https://ollama.ai/) |
| Modell laden | "Discover" → Modell suchen → Download | `ollama pull llama3.1:8b` |
| Server starten | "Local Server" → "Start Server" | `ollama serve` (automatisch) |
| In Webapp | Local wählen → "🔄 Erkennen" | Local wählen → "🔄 Erkennen" |

> **Detail-Anleitung:** [`QCA_AID_assets/docs/user_doc/LOCAL_MODELS_GUIDE.md`](QCA_AID_assets/docs/user_doc/LOCAL_MODELS_GUIDE.md)

#### Chunk-Einstellungen (Wie wird Ihr Text zerteilt?)

Die KI kann nur Textabschnitte einer bestimmten Länge verarbeiten. Deshalb werden Ihre Dokumente in "Chunks" aufgeteilt.

| Einstellung | Was bedeutet das? | Empfehlung |
|------------|-------------------|------------|
| **Chunk-Größe** | Maximale Zeichen pro Textabschnitt | **1000** (Standard) · Interviews: 1000 · Lange Texte: 1500 · Kurze Dok.: 800 |
| **Überlappung** | Wie viele Zeichen überlappen sich benachbarte Abschnitte? | **50** (Standard) · Verhindert, dass relevante Stellen am Abschnittsende zerschnitten werden |
| **Batch-Größe** | Wie viele Abschnitte gleichzeitig analysieren? | **5** (Standard) · Höher = schneller, aber ungenauer. 3-4 = höchste Präzision, 10-12 = schnell |

#### Relevanz-Schwellwert (Was wird analysiert?)

Steuert, welche Textstellen die KI als "relevant für Ihre Forschungsfrage" einstuft.

| Wert | Bedeutung | Wann sinnvoll? |
|------|-----------|---------------|
| **0,3** ⭐ | Standard – KI entscheidet wie gewohnt | Für die meisten Analysen **empfohlen** |
| 0,0–0,2 | Weniger streng – auch zweifelhafte Stellen werden einbezogen | Bei explorativen Studien oder Angst vor Datenverlust |
| 0,4–0,6 | Strenger – nur klar relevante Stellen | Bei großen Datenmengen oder sehr klarer Forschungsfrage |
| 0,7–1,0 | Sehr streng – nur hochrelevante Stellen | Nur für spezifische Teilanalysen |

> **Faustregel:** Mit 0,3 starten → Ergebnisse prüfen → bei Bedarf anpassen.

#### Coder-Einstellungen (KI-"Persönlichkeiten")

Sie können mehrere KI-Kodierer parallel arbeiten lassen (für Qualitätsvergleiche).

| Einstellung | Erklärung | Empfehlung |
|------------|-----------|------------|
| **Temperatur** | Wie "kreativ" kodiert die KI? | 0,2–0,3 für deduktiv (präzise) · 0,4–0,6 für abduktiv · 0,7+ für induktiv/explorativ |
| **Coder-ID** | Name des KI-Kodierers | Eindeutigen Namen vergeben (z.B. "auto_konservativ") |

> **Mehrere Coder:** Im Codebook unter CONFIG → `CODER_SETTINGS` als JSON-Array konfigurieren (siehe Nutzerhandbuch).

#### Qualitätssicherung

| Einstellung | Erklärung |
|------------|-----------|
| **Review-Modus** | `consensus` = Nur Übereinstimmungen · `majority` = Mehrheit · `manual` = Sie entscheiden |
| **Auto-Save** | Wie oft (Minuten) automatisch gespeichert wird |
| **Attribut-Labels** | Metadaten aus Dateinamen (z.B. `attribut1` = "Hochschultyp") – siehe Abschnitt Dokumente |

---

### 📋 Tab: Codebook (Ihr Kategoriensystem)

Hier definieren Sie, **nach welchen Kategorien** die Texte durchsucht werden sollen.

#### Aufbau eines Codebooks

| Bereich | Inhalt | Beispiel |
|---------|--------|---------|
| **Forschungsfrage** | Ihre zentrale Forschungsfrage | "Welche Digitalisierungsstrategien verfolgen Hochschulen?" |
| **Kodierregeln** | Allgemeine Regeln, Formatregeln, Ausschlusskriterien | "Literaturverzeichnisse nicht kodieren" |
| **Deduktive Kategorien** | Ihre vorab definierten Kategorien | "Strategien", "Technologien", "Herausforderungen" |
| **CONFIG** | Technische Einstellungen (s.o.) | Modell, Chunk-Größe, Batch-Größe |

#### Kategorien richtig definieren

Jede Kategorie braucht:

```json
{
  "Kategorienname": {
    "definition": "Präzise Definition (min. 15 Wörter) mit klarer Abgrenzung",
    "rules": ["Konkrete Kodierregeln für diese Kategorie"],
    "examples": ["Mindestens 2 Beispiele aus dem Textmaterial"],
    "subcategories": {
      "Sub_1": "Beschreibung",
      "Sub_2": "Beschreibung"
    }
  }
}
```

> ⚠️ **Wichtig:** Unschärfe Definitionen führen zu freizügiger Kodierung. Textnahe Codes sind besser als abstrakte!  
> 💡 **Tipp:** Begrenzen Sie Hauptkategorien auf **5–7** und stellen Sie sicher, dass sie sich gegenseitig ausschließen.

#### Codebook speichern

| Format | Vorteil | Datei |
|--------|---------|-------|
| **Excel (.xlsx)** | Vertraute Tabellen-Oberfläche – ideal für Einsteiger | `QCA-AID-Codebook.xlsx` |
| **JSON (.json)** | 10× schneller, git-freundlich – ideal für Fortgeschrittene | `QCA-AID-Codebook.json` |

> Die Formate synchronisieren sich automatisch – Änderungen in einem werden ins andere übernommen.

---

### ▶️ Tab: Analyse (Durchführung)

| Bereich | Was Sie sehen/tun |
|---------|------------------|
| **Eingabedateien** | Liste aller Dateien im Input-Ordner – mit Vorschau und Attributen |
| **Konfiguration prüfen** | ✅ Grüner Haken = alles bereit |
| **Codebook validieren** | ✅ Grüner Haken = Kategorien sind gültig |
| **🚀 Analyse starten** | Startet die Kodierung – nach einem Klick läuft alles automatisch |
| **Fortschritt** | Echtzeit-Balken + Live-Logs während der Analyse |
| **Stopp-Funktion** | Analyse bei Bedarf unterbrechen |

---

### 📊 Tab: Explorer (Ergebnisse)

| Funktion | Beschreibung |
|----------|-------------|
| **Output-Dateien** | Alle bisher erstellten Analyseergebnisse |
| **Vorschau** | Schnelle Inhaltsübersicht pro Datei |
| **Visualisierungen** | Heatmaps, Balkendiagramme, Netzwerke (konfigurierbar) |
| **Download** | Diagramme als PNG/PDF speichern |

---

## 🎯 Analyse-Modi im Überblick

| Modus | Neue Kategorien? | Wofür? |
|-------|-----------------|--------|
| **`deductive`** | ❌ Keine | Theorieprüfung, Replikationsstudien ⭐ Standard |
| **`abductive`** | 🔹 Nur Subkategorien | Theorie verfeinern, Nuancen entdecken |
| **`inductive`** | ✅ Haupt- & Subkategorien | Exploration neuer Phänomene |
| **`grounded`** | 🧱 Schrittweise | Datengetriebene Theorieentwicklung |

---

## 📄 Dokumente vorbereiten

### Unterstützte Formate
`*.txt` (bevorzugt) · `*.pdf` · `*.docx`

### Namenskonvention (für automatische Metadaten)
```
attribut1_attribut2_attribut3_name.txt
```
**Beispiel:** `Universität_Professor_Informatik_Interview-2024.txt`

Die Attribute werden automatisch extrahiert und in der Konfiguration als Labels definiert (z.B. `attribut1` = "Hochschultyp").

### Empfohlene Ordnerstruktur
```
mein-projekt/
├── input/          ← Ihre Dokumente hier (.txt, .pdf, .docx)
├── output/         ← Analyseergebnisse (wird automatisch erstellt)
└── config/         ← Konfigurationen (optional)
```

> **Projekt wechseln?** In der Webapp: "📁 Projekt-Verzeichnis ändern" → Ordner auswählen.  
> Oder CLI: `export QCA_AID_PROJECT_ROOT=/pfad/zum/projekt`

---

## 📈 Ergebnisse nutzen

Die Analyse erzeugt eine Excel-Datei `QCA-AID_Analysis_[DATUM].xlsx` mit diesen Arbeitsblättern:

| Sheet | Inhalt |
|-------|--------|
| **Codings** | Alle kodierten Textstellen mit Kategorie, Konfidenz, Begründung |
| **Frequencies** | Häufigkeitsverteilungen pro Kategorie (auch nach Attributen) |
| **Reliability** | Intercoder-Übereinstimmung (Cohens Kappa) |
| **Inductive_Categories** | Neu entdeckte Kategorien (bei induktiven Modi) |
| **Category_Development** | Änderungshistorie des Kategoriensystems |

Außerdem: `category_revisions.json` (Kategorienentwicklung) und `codebook_inductive.json` (erweitertes Codebook).

---

## 💡 Best Practices & Qualitätssicherung

### 🔄 Empfohlener Workflow
1. **Pilotphase (10–20% der Daten):** Testanalyse → manuelle Prüfung → Kategorien anpassen
2. **Hauptanalyse:** Mit optimiertem Codebook alle Dokumente analysieren
3. **Stichprobenkontrolle:** 10% der Kodierungen manuell prüfen
4. **Intercoder-Vergleich:** Mehrere KI-Coder (versch. Temperaturen) laufen lassen → Kappa-Wert prüfen

### 📊 Qualitäts-Benchmarks (Cohens Kappa)
| Wert | Bedeutung |
|------|-----------|
| κ > 0,8 | 🟢 Exzellent – Analyse fortsetzen |
| κ 0,6–0,8 | 🟡 Gut – Stichprobenkontrolle |
| κ 0,4–0,6 | 🟠 Moderat – Kategorien überarbeiten |
| κ < 0,4 | 🔴 Schlecht – Grundlegende Überarbeitung |

### 🎯 3 Erfolgsregeln
1. **Präzise Kategorien** → klare Definitionen + Beispiele + Abgrenzung
2. **Manuelle Kontrolle** → immer einen Teil der Ergebnisse selbst prüfen
3. **Iterativ arbeiten** → erst piloten, dann optimieren, dann skalieren

---

## ⚠️ Wichtige Hinweise

### Entwicklungsstand
- **Aktive Entwicklung** (Version 0.13.0) – nicht alle Funktionen arbeiten optimal
- Nutzung zu **Testzwecken empfohlen** mit manueller Validierung
- Prüfen Sie regelmäßig auf [Updates](CHANGELOG.md)

### Datenschutz
| Modell-Typ | Daten bleiben lokal | Internet nötig |
|-----------|-------------------|---------------|
| **Lokal** (LM Studio/Ollama) | ✅ **100%** | ❌ Nein – offline-fähig |
| **Cloud** (OpenAI, Anthropic, Mistral) | ❌ Daten an Dritte | ✅ Ja |
| **Custom Endpoint** (z.B. GWDG) | ⚠️ Im institutionellen Rechenzentrum | ✅ Ja |

> **Für sensible Daten:** Ausschließlich lokale Modelle verwenden!

### Haftungsausschluss
- KI-Ergebnisse sind **nicht perfekt** und hängen von der Qualität der Eingabedaten ab
- Sie verwenden das Tool **auf eigene Verantwortung**, ohne jegliche Gewährleistung
- Gefahr der **Überkonfidenz** in automatisierte Ergebnisse – bitte kritisch bleiben!

---

## ❓ Häufige Probleme & Lösungen

| Problem | Lösung |
|---------|--------|
| **Python 3.13?** | Python 3.11 installieren (s.o.) – 3.13 wird nicht unterstützt |
| **setup.bat schließt sofort** | `setup_debug.bat` nutzen oder manuell installieren |
| **API-Key nicht gefunden** | `.env`-Datei im QCA-AID-Ordner prüfen – wird seit v0.12.7.4 automatisch geladen |
| **Port 8501 belegt** | Andere Streamlit-Instanz schließen: `taskkill /f /im python.exe` (Windows) / `pkill -f streamlit` (Mac/Linux) |
| **spaCy-Fehler** | `python -m spacy download de_core_news_sm` ausführen |
| **PDF nicht lesbar** | Gescannte PDFs: Vorher mit OCR bearbeiten oder als .txt speichern |
| **Analyse zu langsam** | Batch-Größe erhöhen (max. 12) oder schnelleres Modell wählen |
| **Hohe Kosten** | Lokales Modell nutzen oder auf `gpt-4o-mini` wechseln |
| **Analyse bricht ab** | Kleinere Chunks (800), kleinere Batch-Größe (3), Timeout erhöhen |
| **Zu viele neue Kategorien** | `abductive` statt `inductive` Modus wählen |

> **Ausführliche Fehlerbehebung:** Kapitel 13 im [Nutzerhandbuch](QCA-AID-Nutzerhandbuch.md#13-häufige-probleme-und-lösungen)

---

## 📚 Weiterführende Dokumentation

| Dokument | Inhalt |
|----------|--------|
| **[QCA-AID-Nutzerhandbuch.md](QCA-AID-Nutzerhandbuch.md)** | Vollständige Anleitung – Installation, Webapp, Codebook, Ergebnisse |
| **[LOCAL_MODELS_GUIDE.md](QCA_AID_assets/docs/user_doc/LOCAL_MODELS_GUIDE.md)** | Lokale Modelle einrichten (LM Studio, Ollama) – Schritt für Schritt |
| **[CUSTOM_PROVIDER_GUIDE.md](QCA_AID_assets/docs/user_doc/CUSTOM_PROVIDER_GUIDE.md)** | Eigene KI-Endpoints konfigurieren (GWDG, Azure, ...) |
| **[CHANGELOG.md](CHANGELOG.md)** | Alle Neuerungen und Bugfixes pro Version |

---

## 📄 Zitiervorschlag

```
Henke, J. (2026). QCA-AID: Qualitative Content Analysis with AI-supported Discovery 
(Version 0.13.0) [Software]. Institut für Hochschulforschung Halle-Wittenberg. 
https://github.com/JustusHenke/QCA-AID
```

---

## 📧 Kontakt & Feedback

**Feedback ist willkommen!**  
Justus Henke · justus.henke@hof.uni-halle.de  
Institut für Hochschulforschung Halle-Wittenberg

---

## 📜 Lizenz

Siehe [LICENSE](LICENSE).
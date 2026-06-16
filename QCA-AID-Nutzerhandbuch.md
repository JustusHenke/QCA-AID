# QCA-AID Nutzerhandbuch
## Qualitative Inhaltsanalyse mit KI-Unterstützung

![QCA-AID Banner](banner-qca-aid.png)

**Version:** 0.12.7.4  
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

### Methodische Grundlagen der Relevanzbestimmung

Die Relevanzbestimmung der Textsegmente in QCA-AID erfolgt **forschungsfragengeleitet** und **informationslogisch**. Ausgangspunkt ist die Annahme, dass Relevanz nicht textimmanent, sondern ausschließlich in Bezug auf die jeweilige Forschungsfrage bestimmt werden kann. Entsprechend werden Textsegmente nicht danach bewertet, ob sie für sich genommen einen hohen Erkenntniswert oder eine besondere inhaltliche Tiefe aufweisen, sondern danach, ob sie relevante Informationen zur Beantwortung der Forschungsfrage liefern.

**Theoretische Fundierung:**
In Anlehnung an die qualitative Inhaltsanalyse nach **Mayring** sowie an thematische und informationslogische Auswertungsansätze nach **Kuckartz** sowie **Gläser und Laudel** wird Relevanz definiert als der inhaltliche Bezug eines Textsegments zu mindestens einem zentralen Aspekt der Forschungsfrage. Dabei werden auch kurze, beiläufige oder bestätigende Aussagen als relevant betrachtet, sofern sie einen nachvollziehbaren Informationsbeitrag zur Forschungsfrage leisten. Umfang, Ausführlichkeit oder Neuheitsgrad einer Aussage stellen kein Ausschlusskriterium dar.

**Praktisches Vorgehen:**
Die Forschungsfrage wird zu diesem Zweck in ihre inhaltlichen Aspekte zerlegt (z. B. zentrale Konzepte, Akteure, Prozesse oder Bedingungen), die als Referenzrahmen für die Relevanzprüfung dienen. Ein Textsegment wird als relevant eingestuft, wenn es explizite oder implizite Informationen zu mindestens einem dieser Aspekte enthält. Segmente, die ausschließlich allgemeinen Kontext, organisatorische Informationen oder thematisch verwandte, jedoch nicht forschungsfragenbezogene Inhalte aufweisen, werden als nicht relevant klassifiziert.

**Methodische Einordnung:**
Mit diesem Vorgehen wird einerseits eine systematische und regelgeleitete Materialreduktion ermöglicht, andererseits bleibt die Analyse offen für unterschiedliche Ausprägungen, Intensitäten und Formen relevanter Aussagen. Die Relevanzprüfung ist damit sowohl mit kategoriengeleiteten Ansätzen (Mayring) als auch mit thematisch-explorativen und informationslogischen Verfahren (Kuckartz; Gläser/Laudel) vereinbar.

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

### 3.3 Induktiver Modus (`inductive`)

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

| Komponente | Mindestens | Empfohlen |
|-----------|-----------|-----------|
| **Arbeitsspeicher** | 4 GB RAM | 8 GB RAM (bei lokalen KI-Modellen 16 GB) |
| **Festplatte** | 2 GB frei | 10 GB frei (für lokale KI-Modelle) |
| **Betriebssystem** | Windows 10 · macOS 11+ · Linux | jeweils aktuell |
| **Python** | **3.10 oder 3.11** – nicht 3.12/3.13! | Python 3.11.11 |
| **Browser** | Firefox, Chrome, Edge | aktuellste Version |
| **Internet** | Für Cloud-KI-Modelle nötig | – |

> ⚠️ **Python 3.12 und 3.13 werden nicht unterstützt.** Bitte Python 3.11 installieren.

### 5.2 Schritt-für-Schritt Installation

#### Schritt 1: Python installieren

| Plattform | Vorgehen |
|-----------|----------|
| **Windows** | [python.org](https://www.python.org/downloads/release/python-3110/) → `python-3.11.0-amd64.exe` herunterladen → **☑ "Add Python to PATH"** aktivieren → Installieren → **Neu starten!** |
| **macOS** | Mit [Homebrew](https://brew.sh): `brew install python@3.11` – **oder** Installer von [python.org](https://www.python.org/downloads/release/python-3110/) |
| **Linux** (Debian/Ubuntu) | `sudo apt update && sudo apt install python3.11 python3.11-venv` |
| **Linux** (Fedora) | `sudo dnf install python3.11` |

**Nach der Installation prüfen:**
```bash
python --version
# Ausgabe: Python 3.11.x
```
> ❓ Falls `python` nicht gefunden wird: Prüfen Sie, ob Sie "Add Python to PATH" aktiviert haben (Windows) bzw. ob Python über den Paketmanager installiert ist (Linux). Ggf. `python3 --version` versuchen.

#### Schritt 2: QCA-AID herunterladen

**Variante A: Mit Git (empfohlen für Updates)**

Öffnen Sie ein Terminal (Eingabeaufforderung unter Windows) und geben Sie ein:

```bash
git clone https://github.com/JustusHenke/QCA-AID.git
cd QCA-AID
```

> **Git installieren:** [git-scm.com](https://git-scm.com/) – Vorteil: Sie können später mit `git pull` einfach Updates einspielen.

**Variante B: Ohne Git (ZIP-Download)**
1. [github.com/JustusHenke/QCA-AID](https://github.com/JustusHenke/QCA-AID) öffnen
2. Auf grünen Button **"Code"** klicken → **"Download ZIP"**
3. ZIP-Datei entpacken
4. In den Ordner `QCA-AID` wechseln

#### Schritt 3: Abhängigkeiten installieren

**Einfachste Methode (Windows):**
Einfach auf `setup.bat` doppelklicken – das Skript installiert alles automatisch.

**Falls `setup.bat` sofort schließt:** Stattdessen `setup_debug.bat` doppelklicken (bleibt offen und zeigt Fehler).

**Manuelle Installation (alle Plattformen):**

Öffnen Sie ein Terminal im QCA-AID-Ordner und führen Sie aus:

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Deutsches Sprachmodell laden
python -m spacy download de_core_news_sm
```

| Bei Fehlern… | Lösung |
|-------------|--------|
| **Windows: "Microsoft Visual C++ 14.0 is required"** | [Visual Studio Build Tools](https://visualstudio.microsoft.com/de/visual-cpp-build-tools/) installieren – dort "C++ Build Tools" + MSVC + Windows SDK aktivieren |
| **macOS/Linux: "externally-managed-environment"** | Virtuelle Umgebung erstellen: `python3.11 -m venv venv && source venv/bin/activate` (Mac) oder `source venv/bin/activate` (Linux) – danach `pip install -r requirements.txt` wiederholen |
| **"pip not found"** | Python-Paketmanager nachinstallieren: `python -m ensurepip --upgrade` |

#### Schritt 4: API-Schlüssel einrichten (nur für Cloud-KI nötig)

Für Cloud-Modelle (OpenAI, Anthropic, Mistral) benötigen Sie einen API-Schlüssel. Erstellen Sie eine Datei namens `.env` im QCA-AID-Ordner:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxx  # nur wenn nötig
# MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx  # nur wenn nötig
```

> 🔐 **Seit Version 0.12.7.4** wird die `.env`-Datei **automatisch geladen** – kein manuelles Laden nötig.
> 🔐 **Kein API-Key nötig** bei lokalen Modellen (LM Studio / Ollama) – 100% datenschutzkonform.
> ⚠️ `.env` nie öffentlich teilen und zur `.gitignore` hinzufügen!

**Wo bekomme ich einen API-Key?**
- **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys) – nach Registrierung Guthaben aufladen (ca. 5–10 € reichen für eine Studie)
- **Anthropic:** [console.anthropic.com](https://console.anthropic.com/)
- **Mistral:** [console.mistral.ai](https://console.mistral.ai/)
- **OpenRouter** (Zugang zu vielen Modellen): [openrouter.ai/keys](https://openrouter.ai/keys)

#### Schritt 5: Installation testen

```bash
python start_webapp.py
```

Daraufhin öffnet sich automatisch Ihr Browser mit der QCA-AID Webapp unter `http://127.0.0.1:8501`. **Fertig!** 🎉

> **Webapp startet nicht?** → Siehe Abschnitt [13.3 Webapp-spezifische Probleme](#133-webapp-spezifische-probleme)

### 5.3 Projektverzeichnis einrichten

QCA-AID arbeitet projektbasiert. Die empfohlene Struktur:

```
mein-forschungsprojekt/
├── input/          ← Ihre Textdateien hier ablegen (.txt, .pdf, .docx)
├── output/         ← Analyseergebnisse (wird automatisch erstellt)
└── config/         ← Konfigurationsdateien (optional)
```

**In der Webapp:** Klicken Sie auf "📁 Projekt-Verzeichnis ändern" und wählen Sie Ihren Projektordner aus. Die Einstellung wird gespeichert.

**Oder per Umgebungsvariable (für Fortgeschrittene):**
```bash
# Windows (Eingabeaufforderung)
set QCA_AID_PROJECT_ROOT=C:\Pfad\zu\meinem\Projekt

# macOS / Linux
export QCA_AID_PROJECT_ROOT=/pfad/zu/meinem/projekt
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
| **OpenRouter** | ⚠️ Cloud | 💰💰 variabel | ⭐⭐⭐⭐⭐ viele Modelle | ⭐⭐⭐⭐ Einfach |
| **Custom Endpoint** 🎓 | ⚠️ In Uni-RZ | 💰 Oft kostenlos | ⭐⭐⭐⭐ variiert | ⭐⭐ Mittel |

### 6.2 Lokale Modelle (Empfohlen für sensible Daten)

**Vorteile auf einen Blick:**
- ✅ **100% Datenschutz** – Ihre Daten verlassen nie Ihren Rechner
- ✅ **Kostenlos** – Keine API-Gebühren
- ✅ **DSGVO-konform** – Ideal für personenbezogene Forschungsdaten
- ✅ **Offline-fähig** – Keine Internetverbindung nötig

**Zwei Wege zu lokalen Modellen:**

| Kriterium | LM Studio | Ollama |
|-----------|-----------|--------|
| **Zielgruppe** | Einsteiger, grafische Oberfläche | Fortgeschrittene, Kommandozeile |
| **Installation** | [lmstudio.ai](https://lmstudio.ai/) herunterladen & installieren | [ollama.ai](https://ollama.ai/) herunterladen & installieren |
| **Modell laden** | "Discover" → Modell suchen → Download | Terminal: `ollama pull llama3.1:8b` |
| **Server starten** | "Local Server" → "Start Server" (Port 1234) | Automatisch (Port 11434) |
| **In QCA-AID** | "Local" wählen → "🔄 Erkennen" | "Local" wählen → "🔄 Erkennen" |

**Empfohlene lokale Modelle (Richtwerte für deutschsprachige Textanalyse):**

| Modell | Größe | RAM nötig | Geschwindigkeit | Qualität |
|--------|-------|-----------|----------------|----------|
| **Llama 3.1 8B** | 4,7 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Mistral 7B** | 4,1 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Qwen 2.5 14B** | 8,5 GB | 16 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| **Llama 3.1 70B** | 40 GB | 32–48 GB | ⚡ | ⭐⭐⭐⭐⭐ |

> **Detail-Anleitung:** [`QCA_AID_assets/docs/user_doc/LOCAL_MODELS_GUIDE.md`](QCA_AID_assets/docs/user_doc/LOCAL_MODELS_GUIDE.md)

### 6.3 Cloud-Modelle

#### OpenAI (empfohlen für Einsteiger)
| Modell | Kosten | Geschwindigkeit | Qualität |
|--------|--------|----------------|----------|
| `gpt-4o-mini` ⭐ | 💰 Niedrig | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `gpt-4o` | 💰💰 Mittel | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| `gpt-5.4` (neu) | 💰💰💰 Höher | ⚡⚡ | ⭐⭐⭐⭐⭐ |

#### Anthropic (Claude)
| Modell | Kosten | Geschwindigkeit | Qualität |
|--------|--------|----------------|----------|
| `claude-3-5-sonnet` | 💰💰 Mittel | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `claude-opus-4-7` (neu) | 💰💰💰 Höher | ⚡ | ⭐⭐⭐⭐⭐ |

#### Mistral
| Modell | Kosten | Geschwindigkeit | Qualität |
|--------|--------|----------------|----------|
| `mistral-small-latest` | 💰 Niedrig | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `mistral-large-latest` | 💰💰 Mittel | ⚡⚡ | ⭐⭐⭐⭐⭐ |

#### OpenRouter (Zugang zu vielen Modellen über eine API)
OpenRouter bündelt zahlreiche Modelle – nützlich, wenn Sie verschiedene Anbieter vergleichen möchten:

```bash
# In der .env-Datei
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

**In der Webapp:** Einfach "OpenRouter" als Anbieter wählen und ein Modell aus dem Dropdown auswählen.

> **Neu in 0.12.7.3:** Bei aktivierter Custom API Base URL wird das Modell-Feld zum Freitext-Eingabefeld – Sie können dann jedes beliebige Modell eingeben.

### 6.4 Custom API Endpoints 🎓 (Institutionelle KI-Nutzung)

**Wofür?** Viele Hochschulen und Forschungseinrichtungen betreiben eigene KI-Server auf Basis OpenAI-kompatibler APIs. Beispiele:
- **GWDG Academic Cloud** (Göttingen): `https://chat-ai.academiccloud.de/v1`
- **Azure OpenAI**: `https://ihre-resource.openai.azure.com/`
- Eigene lokale oder institutionelle Server

**So konfigurieren Sie einen Custom Endpoint in der Webapp:**

1. **Anbieter wählen:** Z.B. "OpenAI" – der Custom Endpoint ist **bei allen Providern** sichtbar
2. **🔧 Erweiterte Einstellungen ausklappen:** Custom API Base URL eingeben (muss mit `http://` oder `https://` beginnen)
3. **Modell eingeben:** Sobald eine Base URL gesetzt ist, wird das Modell-Feld zum Freitext (z.B. `openai-gpt-oss-120b`)
4. **API-Key Variable:** Wenn Ihr Endpoint einen anderen Umgebungsvariablen-Namen verwendet (z.B. `GWDG_API_KEY` statt `OPENAI_API_KEY`), tragen Sie ihn hier ein

**Beispiel für `.env` bei GWDG:**
```
GWDG_API_KEY=xxxxxxxxxxxxxxxxxxxxx
```

**In der Konfigurationsdatei (JSON):**
```json
{
  "model_provider": "OpenAI",
  "model_name": "openai-gpt-oss-120b",
  "api_base_url": "https://chat-ai.academiccloud.de/v1",
  "api_key_env": "GWDG_API_KEY"
}
```

> **Detail-Anleitung:** [`QCA_AID_assets/docs/user_doc/CUSTOM_PROVIDER_GUIDE.md`](QCA_AID_assets/docs/user_doc/CUSTOM_PROVIDER_GUIDE.md)

### 6.5 Modellauswahl-Empfehlungen

| Ihre Situation | Empfohlenes Modell | Begründung |
|---------------|-------------------|------------|
| **Erste Schritte, kleine Studie** ⭐ | OpenAI `gpt-4o-mini` | Günstig, schnell, sehr gute Qualität |
| **Sensible Forschungsdaten** | LM Studio + Llama 3.1 8B | 100% datenschutzkonform |
| **Große Studie (50+ Dokumente)** | OpenAI `gpt-4o` oder `gpt-5.4` | Höchste Qualität für viele Daten |
| **Sehr knappes Budget** | Lokales Modell (Mistral 7B) | Kostenlos, ausreichende Qualität |
| **Hochschul-eigene KI-Infrastruktur** | Custom Endpoint (GWDG, Azure, …) | Oft kostenlos, datenschutzkonform |
| **Methodenvergleich** | OpenRouter (versch. Modelle) | Ein API-Key, viele Modelle

---

## 7. Konfigurationseinstellungen (Webapp-Tab: Konfiguration)

Alle Einstellungen nehmen Sie am besten direkt in der Webapp vor – dort gibt es für jede Option ein Eingabefeld oder einen Schieberegler mit Erklärungstext. Dieses Kapitel erklärt die Bedeutung jeder Einstellung.

### 7.1 Die Einstellungen im Überblick

#### LLM-Anbieter & Modell (Abschnitt "Modell-Einstellungen")

| Webapp-Feld | Erklärung | Tipp |
|------------|-----------|------|
| **Provider** | Wer liefert die KI? | OpenAI ⭐ für Einsteiger · "Local" für Datenschutz · Custom für Uni-Server |
| **Modell** | Welches KI-Modell? | Dropdown zeigt passende Modelle. **Neu:** Wird zum Freitext-Feld, sobald eine Custom API Base URL gesetzt ist |
| **🔄 Erkennen** (Local) | Sucht nach LM Studio oder Ollama auf Ihrem Rechner | Einfach klicken nach Start des lokalen Servers |
| **🔧 Custom API Base URL** (erweiterter Bereich) | Eigene KI-Endpunkt-Adresse | Seit v0.12.7.3 **bei allen Providern sichtbar**. Format: `https://mein-server.de/v1` |
| **API-Key Variable** | Name der Umgebungsvariable | Nur bei Custom-Endpoints: z.B. `GWDG_API_KEY` statt `OPENAI_API_KEY` |

> 💡 **API-Key-Prüfung in Echtzeit:** Die Webapp zeigt direkt an, ob der konfigurierte API-Key gefunden wird.

#### Chunk-Einstellungen (Abschnitt "Chunk-Einstellungen")

Ihre Dokumente werden in Textabschnitte ("Chunks") zerlegt, da KI-Modelle nur eine begrenzte Textmenge auf einmal verarbeiten können.

| Einstellung | Beschreibung | Empfehlung |
|------------|-------------|------------|
| **Chunk-Größe** | Maximale Zeichen pro Textabschnitt | **1000** (Standard). Je nach Textart: 800 (kurze Dokumente), 1200 (wissenschaftliche Texte), 1500 (sehr lange Texte) |
| **Überlappung** | Überlappung benachbarter Abschnitte in Zeichen | **50** (Standard). Verhindert, dass relevante Stellen genau an der Schnittkante zerschnitten werden |
| **Batch-Größe** | Wie viele Abschnitte gleichzeitig kodiert werden | **5** (Standard). Niedriger = präziser (3-4), höher = schneller (10-12) |

> **Faustregel Chunk-Größe:** Kleine Chunks = präzise Kodierung, aber weniger Kontext. Große Chunks = mehr Kontext, aber riskieren Mehrfachkodierungen.

#### Relevanz-Schwellwert (Abschnitt "Relevanz-Schwellwert")

Steuert, wie streng die KI Textstellen als relevant für Ihre Forschungsfrage einstuft. Der Wert ist ein Schieberegler in der Webapp (0,0 bis 1,0).

| Position | Wirkung | Wann sinnvoll? |
|----------|---------|---------------|
| **0,3** ⭐ | KI entscheidet wie gewohnt – Standard | Für **die meisten Analysen empfohlen** |
| 0,0–0,2 | Weniger streng – auch zweifelhafte Stellen werden einbezogen | Bei explorativen Studien oder Sorge vor verlorenen Daten |
| 0,4–0,6 | Strenger – nur klar relevante Stellen | Bei großen Datenmengen oder sehr fokussierter Forschungsfrage |
| 0,7–1,0 | Sehr streng – nur hochrelevante Stellen | Nur für spezielle Teilanalysen oder Nachkodierungen |

**Empfohlener Workflow:** Mit 0,3 starten, Ergebnisse prüfen, bei Bedarf anpassen.

#### Coder-Einstellungen (Abschnitt "Coder-Einstellungen")

Sie können mehrere KI-Kodierer mit unterschiedlicher Kreativität parallel laufen lassen, um die Qualität zu vergleichen (Intercoder-Reliabilität).

| Einstellung | Beschreibung | Empfehlung |
|------------|-------------|------------|
| **Temperatur** (0,0–1,0) | Wie kreativ/flexibel kodiert die KI | **0,2–0,3** für präzise deduktive Kodierung · **0,4–0,6** für abduktive Analyse · **0,7+** für explorative/induktive Analyse |
| **Coder-ID** | Name des Kodierers (für Nachvollziehbarkeit) | Einen eindeutigen Namen vergeben, z.B. `auto_präzise` |

> **Mehrere Coder im Codebook:** Im CONFIG-Bereich des Codebooks können Sie `CODER_SETTINGS` als JSON-Array hinterlegen, um z.B. drei Coder mit 0,3 / 0,5 / 0,7 Temperatur gleichzeitig laufen zu lassen.

#### Qualitätssicherung & Speicherung

| Einstellung | Erklärung |
|------------|-----------|
| **Review-Modus** | `consensus` = Nur Übereinstimmungen mehrerer Coder · `majority` = Mehrheit entscheidet · `manual` = Sie prüfen jeden Konflikt selbst |
| **Auto-Save** | Automatische Sicherung in Minuten – hilft bei langen Analysen |
| **Kontextuelle Kodierung** | ✅ Aktiviert = KI fasst den bisherigen Dokumentverlauf zusammen → konsistentere Kodierung, aber langsamer |
| **Mehrfachkodierungen** | ✅ Aktiviert = ein Textsegment kann mehreren Kategorien zugeordnet werden |

### 7.2 Attribut-Extraktion aus Dateinamen

QCA-AID kann automatisch Metadaten aus den Dateinamen extrahieren. Das ermöglicht spätere Auswertungen nach Gruppen (z.B. "Alle Kodierungen von Universitäten vs. Fachhochschulen").

**Namenskonvention:**
```
Attribut1_Attribut2_Attribut3_FreierName.txt
```

**In der Webapp einrichten (Tab Konfiguration):**
1. Attribut-Labels definieren, z.B.:
   - `attribut1` = **Hochschultyp**
   - `attribut2` = **Position**
   - `attribut3` = **Fachbereich**
2. Dateien entsprechend benennen, z.B.:
   - `Universität_Professor_Informatik_Interview-2024.txt`
   - `FH_Studierende_BWL_Fokusgruppe.txt`

**Im Analyse-Output** erscheinen dann separate Spalten für Hochschultyp, Position und Fachbereich – mit denen Sie später filtern und vergleichen können.

### 7.3 Konfiguration speichern: Excel oder JSON

QCA-AID speichert alle Einstellungen zusammen mit dem Codebook. Dabei werden **beide Formate automatisch synchronisiert** – egal, ob Sie Excel oder JSON bearbeiten, die jeweils andere Datei wird mit aktualisiert.

| Format | Vorteile | Für wen? |
|--------|---------|----------|
| **Excel** (`.xlsx`) | Vertraute Tabellen-Oberfläche, einfache Bedienung | Einsteiger |
| **JSON** (`.json`) | 10× schneller beim Laden, ideal für Git-Versionierung | Fortgeschrittene |

**Seit Version 0.12.7:** Codebook wird automatisch mit der Konfiguration geladen – kein separater "Codebook laden"-Schritt mehr nötig.

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

In der Webapp unter **Codebook-Tab → Abschnitt "Induktive Codes"** wird automatisch angezeigt, wenn neue Codes aus einer früheren Analyse verfügbar sind.

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

### 8.5 Codebook speichern und verwalten (Webapp)

**Speichern in der Webapp:**
- Ein Klick auf **"Speichern"** schreibt das Codebook sowohl als `QCA-AID-Codebook.xlsx` als auch als `QCA-AID-Codebook.json`
- **Seit Version 0.12.7:** Das Codebook wird automatisch mit der Konfiguration geladen – kein separater "Codebook laden"-Dialog mehr nötig
- Der zuletzt verwendete Speicherpfad wird für künftige Speichervorgänge automatisch übernommen

**Versionskontrolle mit Git (empfohlen für Projekte):**
```bash
git add QCA-AID-Codebook.json     # JSON ist besser für Git geeignet
git commit -m "Kategorien erweitert: KI-Tools hinzugefügt"
git tag -a v1.0 -m "Finales Codebook Hauptanalyse"
```

### 8.6 Validierung und Qualitätskontrolle

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

Nach dem Start (`python start_webapp.py`) öffnet sich Ihr Browser mit der QCA-AID Webapp. Sie hat **vier Haupt-Tabs** (oben in der Navigation):

| Tab | Symbol | Wofür? |
|-----|--------|--------|
| **Konfiguration** | ⚙️ | KI-Modell wählen, Chunk-Größe, Relevanz-Schwellwert und alle technischen Parameter |
| **Codebook** | 📋 | Kategorien definieren, bearbeiten und aus vorherigen Analysen importieren |
| **Analyse** | ▶️ | Analyse starten, Fortschritt verfolgen, Ergebnisse live einsehen |
| **Explorer** | 📊 | Ergebnisse visualisieren, Diagramme erstellen und exportieren |

Die Webapp läuft **nur auf Ihrem Rechner** (localhost) – niemand sonst hat Zugriff auf Ihre Daten.

### 9.2 Projekt-Management

Zu Beginn wählen Sie Ihr Projektverzeichnis aus:

1. **Oben in der Seitenleiste** klicken Sie auf "📁 Projekt-Verzeichnis ändern"
2. **Ordner auswählen:** Navigieren Sie zu Ihrem Projektordner (der die `input/`- und `output/`-Unterordner enthält)
3. **Automatische Speicherung:** Ihre Wahl wird in `.qca-aid-project.json` gespeichert und beim nächsten Start wieder verwendet

**Empfohlene Projektstruktur:**
```
mein-forschungsprojekt/
├── input/              ← Ihre Textdateien hier ablegen (.txt, .pdf, .docx)
├── output/             ← Analyseergebnisse (wird automatisch erstellt)
├── config/             ← Konfigurationsdateien (optional)
└── codebooks/          ← Codebook-Versionen (optional)
```

> **Projekt wechseln?** Die Webapp merkt sich Ihr Projekt – Sie können aber jederzeit ein anderes auswählen.

### 9.3 Konfiguration-Tab (⚙️)

Dieser Tab ist in mehrere Abschnitte unterteilt, die nacheinander durchgehen.

#### 1. Konfigurationsdatei laden

Oben sehen Sie ein Pfad-Eingabefeld mit 📁-Button. Hier laden Sie eine bestehende `.json`- oder `.xlsx`-Konfigurationsdatei.  
**Seit Version 0.12.7:** Das Codebook wird automatisch mitgeladen – kein separater Schritt nötig.

#### 2. Modell-Einstellungen

**Cloud-Modell einrichten (empfohlen für Einsteiger):**
1. **Provider wählen:** "OpenAI", "Anthropic", "Mistral" oder "OpenRouter"
2. **Modell aus Dropdown wählen** – die Liste zeigt passende Modelle
3. Der **API-Key wird automatisch geprüft**: entweder aus der `.env`-Datei (seit v0.12.7.4 automatisch geladen) oder aus einer Umgebungsvariable

**Lokales Modell einrichten (für Datenschutz):**
1. **Provider** auf "Local (LM Studio/Ollama)" stellen
2. **"🔄 Lokale Modelle erkennen" klicken** – die Webapp sucht nach laufenden LM Studio- oder Ollama-Servern auf Ihrem Rechner
3. Aus den erkannten Modellen auswählen

**Custom API Endpoint (für Hochschul-KI-Server):**
1. **Ausklappbaren Bereich "🔧 Erweiterte Einstellungen: Custom API Base URL" öffnen**
2. **Adresse eintragen:** z.B. `https://chat-ai.academiccloud.de/v1` (GWDG)
3. **Hinweis:** Sobald eine Base URL gesetzt ist, wird das Modell-Feld zum Freitext – Sie können dann jedes beliebige Modell eingeben
4. **API-Key Variable:** Wenn Ihr Endpoint eine eigene Umgebungsvariable nutzt (z.B. `GWDG_API_KEY`), tragen Sie den Namen hier ein

#### 3. Chunk-Einstellungen (Schieberegler)

| Regler | Bereich | Standard | Erklärung |
|-------|---------|---------|-----------|
| **Chunk-Größe** | 400–2000 | 1000 | Maximale Zeichen pro Textabschnitt |
| **Überlappung** | 0–200 | 50 | Überlappung benachbarter Abschnitte in Zeichen |
| **Batch-Größe** | 1–20 | 5 | Wie viele Abschnitte gleichzeitig kodiert werden |

#### 4. Relevanz-Schwellwert (Schieberegler)

Der Schieberegler hat drei Zonen, die farblich markiert sind:

| Zone | Bereich | Anzeige | Wofür? |
|------|---------|---------|--------|
| 🟢 **Standard** | 0,3 | "Verwendet LLM-Entscheidungen wie sie sind" | Normalbetrieb |
| 🔵 **Streng** | 0,4–1,0 | "Nur hochrelevante Segmente" | Weniger Rauschen |
| 🟠 **Inklusiv** | 0,0–0,2 | "Inkludiert LLM-verworfene Segmente" | Maximale Vollständigkeit |

#### 5. Weitere Einstellungen

- **Kontextuelle Kodierung:** ✅ = aktiviert – die KI fasst den bisherigen Dokumentverlauf zusammen und kodiert konsistenter (aber langsamer)
- **Mehrfachkodierungen:** ✅ = erlaubt – ein Text kann mehreren Kategorien zugeordnet werden
- **Attribut-Labels:** Hier definieren Sie die Metadaten aus Dateinamen (siehe Abschnitt 7.2)

### 9.4 Codebook-Tab (📋)

Hier entwickeln und pflegen Sie Ihr Kategoriensystem.

#### Kategorien bearbeiten

Der Editor zeigt für jede Kategorie ein Formular:

1. **Kategorie hinzufügen:** Klick auf "➕ Kategorie hinzufügen"
2. **Felder ausfüllen:** Name (ohne Leerzeichen), Definition, Regeln, Beispiele, Subkategorien
3. **Echtzeit-Validierung:** Ungültige Eingaben werden sofort markiert (z.B. zu kurze Definition)
4. **Speichern:** Änderungen werden in Excel- und JSON-Datei synchronisiert

**Darauf achtet die Validierung:**
- Definition mindestens 15 Wörter lang
- Mindestens 2 Beispiele pro Kategorie
- Mindestens 2 Subkategorien pro Kategorie
- Keine doppelten Kategoriennamen

#### Induktive Codes importieren

Nach einer Analyse im `full`- oder `abductive`-Modus können neu entdeckte Kategorien zurück ins Codebook übernommen werden:

1. **Hinweis abwarten:** Die Webapp zeigt automatisch an, wenn neue induktive Codes verfügbar sind
2. **"Induktive Codes importieren" klicken**
3. **Analyse-Datei auswählen:** Wählen Sie die entsprechende Excel-Datei aus dem Output-Ordner
4. **Vorschau prüfen:** Sie sehen, welche Codes entdeckt wurden
5. **Konflikte lösen:** Falls ein Name bereits existiert, können Sie umbenennen
6. **Import bestätigen:** Neue Codes erscheinen im Codebook

#### Speichern

Das Codebook wird automatisch sowohl als Excel (`.xlsx`) als auch als JSON (`.json`) gespeichert – beide Versionen sind identisch. Excel ist für die manuelle Bearbeitung gedacht, JSON ist schneller und git-freundlich.

### 9.5 Analyse-Tab (▶️)

Hier starten und überwachen Sie die Analyse.

#### Vorbereitung prüfen

Vor dem Start sehen Sie zwei Statusanzeigen:

- **✅ Konfiguration geprüft** – alle technischen Einstellungen sind gültig
- **✅ Codebook validiert** – alle Kategorien erfüllen die Mindestanforderungen

Erst wenn beide grün sind, wird der Start-Button aktiv.

#### Analyse starten

1. Klick auf **"🚀 Analyse starten"**
2. Die Verarbeitung läuft automatisch durch alle Dokumente
3. Sie sehen:
   - **Fortschrittsbalken** – wie viele Chunks bereits kodiert sind
   - **Live-Logs** – detaillierte Meldungen zum aktuellen Schritt
   - **Token-Verbrauch** – wie viele KI-Token bereits verwendet wurden

#### Analyse unterbrechen

- **"Analyse stoppen"** – bricht die laufende Analyse ab. Bereits kodierte Ergebnisse bleiben erhalten.
- **Automatische Sicherung:** In regelmäßigen Abständen (einstellbar) wird der Zwischenstand gespeichert.

### 9.6 Explorer-Tab (📊)

Nach der Analyse können Sie die Ergebnisse durchstöbern und visualisieren.

#### Datei-Übersicht

- **Alle Output-Dateien** des aktuellen Projekts werden aufgelistet
- **Vorschau:** Per Klick auf eine Datei sehen Sie den Inhalt
- **Metadaten:** Datum, Analysemodus, verwendetes Modell

#### Visualisierungen

1. **Explorer-Config laden:** Wählen Sie eine Analyse-Ergebnisdatei aus
2. **Diagrammtyp wählen:** Heatmaps (Kategorien × Attribute), Balkendiagramme, Kreisdiagramme, Netzwerk-Graphen
3. **Filter setzen:** Nach Kategorien, Attributen oder Dokumenten eingrenzen
4. **Export:** Diagramme als PNG oder PDF speichern

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

QCA-AID erstellt eine umfassende Excel-Datei (`QCA-AID_Analysis_[DATUM].xlsx`) mit mehreren Arbeitsblättern. Die genaue Anzahl hängt vom gewählten Analysemodus ab.

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

> **Lesebeispiel:** In der Excel-Tabelle sehen Sie für jede kodierte Textstelle eine Zeile. Sortieren Sie nach "Konfidenz" (aufsteigend), um die unsichersten Kodierungen zuerst zu prüfen.

#### Häufigkeitsanalysen (Sheet: "Frequencies")

**Inhalte:**
- Absolute und relative Häufigkeiten pro Kategorie
- Verteilung nach Attributen (z.B. Hochschultyp, Position)
- Kreuztabellen zwischen Kategorien und Attributen
- Statistische Kennwerte (Mittelwerte, Standardabweichungen)

> Dieses Sheet ist besonders nützlich, um zu sehen, welche Kategorien besonders häufig vorkommen und wie sie sich über verschiedene Gruppen (z.B. Hochschultypen) verteilen.

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

Nur sichtbar bei Analysen im `full`, `abductive` oder `grounded`-Modus. Hier stehen alle neu entdeckten Kategorien, die die KI während der Analyse identifiziert hat.

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

Jede Kodierung erhält einen Konfidenzwert zwischen 0,0 (unsicher) und 1,0 (sehr sicher). Verteilen Sie die Kodierungen nach diesem Wert, um Prioritäten für die manuelle Prüfung zu setzen.

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

Die folgende Tabelle hilft bei der Auswahl des passenden Analysemodus:

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

Die optimale Konfiguration hängt vom Textmaterial ab. Hier eine Kurzübersicht:

| Material | Chunk-Größe | Überlappung | Kontext-Kodierung | Besonderheit |
|----------|------------|-------------|-------------------|-------------|
| **Interview-Transkripte** | 1000 | 60 | ✅ Aktivieren | Interviewerfragen ausschließen |
| **Wissenschaftliche Texte** | 1200 | 40 | ❌ Deaktivieren | Literaturverzeichnisse ausschließen |
| **Dokumente, Berichte** | 800 | 30 | ❌ Deaktivieren | Inhaltsverzeichnisse ausschließen |
| **Social Media, Foren** | 500 | 20 | ❌ Deaktivieren | Höhere Temperatur (0,6) für Nuancen |

**In der Webapp einstellen:**
1. **Konfiguration-Tab** → Chunk-Einstellungen als Schieberegler
2. **Kontextuelle Kodierung** als An/Aus-Schalter
3. **Ausschlusskriterien** im Codebook unter "Kodierregeln" → "Ausschlusskriterien"

**Tipp zu Ausschlusskriterien:**
Im Codebook unter `KODIERREGELN` → `exclusion` können Sie festlegen, was nicht kodiert werden soll:
- Bei Interviews: "Interviewerfragen ohne inhaltlichen Bezug"
- Bei wissenschaftlichen Texten: "Literaturverzeichnisse", "Reine Zitate ohne Interpretation"
- Bei Dokumenten: "Inhaltsverzeichnisse", "Tabellarische Auflistungen"
```

### 11.7 Kombinierte Ansätze

**Sequenziell (empfohlen):**
1. **Erkunden:** `grounded` oder `full` → neue Kategorien entdecken
2. **Strukturieren:** `abductive` → Kategorien verfeinern und systematisieren
3. **Validieren:** `deductive` → Mit endgültigem Codebook alle Dokumente kodieren

**Parallel (für Methodenvergleich):**
Führen Sie die Analyse zweimal mit unterschiedlichen Modi durch (z.B. `deductive` und `full`) und vergleichen Sie die Ergebnisse. Die Outputs landen in verschiedenen Ordnern und können im Explorer-Tab nebeneinander betrachtet werden.

---

## 12. Best Practices und Qualitätssicherung

### 12.1 Vorbereitung der Datengrundlage

#### Textqualität sicherstellen

**Dokumentenvorbereitung:**
- **Bereinigung:** Entfernung von Literaturverzeichnissen, Fußnoten, Seitenzahlen
- **Formatierung:** Einheitliche Textformatierung, keine Sonderzeichen
- **Vollständigkeit:** Überprüfung auf fehlende Textpassagen (besonders bei PDFs)
- **Kodierung:** UTF-8 Encoding für Umlaute und Sonderzeichen
- **Tipp:** Speichern Sie Textdokumente im `.txt`-Format – das ist am robustesten für die Verarbeitung

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

> **Praxistipp:** Öffnen Sie die Ergebnis-Excel und filtern Sie nach "Konfidenz < 0,6". Diese Kodierungen sollten Sie prioritär manuell prüfen.

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
Je nach verwendetem Modell und Dokumenttyp kann eine andere Batch-Größe optimal sein. Testen Sie mit einer kleinen Stichprobe:

- **Hohe Präzision:** Batch-Größe 3–4 (langsamer, genauer)
- **Ausgewogen:** Batch-Größe 5–8 ⭐ Empfohlen
- **Hohe Geschwindigkeit:** Batch-Größe 10–12 (schneller, weniger präzise)

**Chunk-Parameter anpassen:**
- **Zu kleine Chunks (< 800):** Kontextverlust, fragmentierte Kodierungen
- **Zu große Chunks (> 1500):** Mehrfachkodierungen, unklare Zuordnungen
- **Optimale Größe:** 800–1200 Zeichen je nach Texttyp

#### Kostenoptimierung

**Token-Verbrauch reduzieren:**
- **Präzise Kategorien:** Weniger Nachfragen durch klarere Definitionen
- **Optimale Batch-Größe:** Weniger API-Calls durch größere Batches
- **Günstigeres Modell:** z.B. `gpt-4o-mini` statt `gpt-4o`
- **Lokale Modelle:** Kostenlos und unbegrenzt nutzbar

> In der Webapp sehen Sie während der Analyse den geschätzten Token-Verbrauch – so behalten Sie die Kosten im Blick.

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

#### Problem: setup.bat schließt sofort (Windows)

**Symptom:** Das Setup-Fenster erscheint kurz und schließt sofort wieder, ohne dass Sie die Fehlermeldung lesen können.

**Lösung 1 (Empfohlen): `setup_debug.bat` verwenden**
Diese Datei wurde speziell für diesen Fall ergänzt: Sie hält das Fenster nach Fehlern offen, sodass Sie die Meldung lesen können. Einfach doppelklicken.

**Lösung 2: Manuell in der Eingabeaufforderung ausführen**
```cmd
# 1. Windows + R → "cmd" eingeben → Enter
# 2. Zum QCA-AID Ordner navigieren:
cd "C:\Pfad\zu\QCA-AID"
# 3. Setup ausführen:
setup.bat
```

**Lösung 3: Datei entsperren**
1. Rechtsklick auf `setup.bat` → "Eigenschaften"
2. Unten bei "Sicherheit": **"Zulassen"** oder **"Entsperren"** anklicken
3. "OK" → erneut versuchen

**Lösung 4: Als Administrator ausführen**
1. Rechtsklick auf `setup.bat`
2. "Als Administrator ausführen" wählen

**Lösung 5: Manuelle Installation**
```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download de_core_news_sm
python start_webapp.py
```

### 13.2 API und Authentifizierung

#### Problem: API-Schlüssel nicht gefunden

**Symptom:** `API key not found` oder `Authentication failed` oder der API-Key-Status in der Webapp zeigt rot

**Mögliche Ursachen und Lösungen:**

**1. `.env`-Datei fehlt oder falsch benannt**
```bash
# Prüfen, ob die .env-Datei existiert:
# Windows (Eingabeaufforderung):
dir .env
# macOS / Linux:
ls -la .env

# Falls nicht: .env-Datei im QCA-AID-Ordner erstellen:
# Inhalt: OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx
```

> **Seit Version 0.12.7.4** wird `.env` automatisch geladen – Sie müssen nichts weiter tun. Die Webapp sucht an drei Stellen: aktuelles Verzeichnis → Repository-Root → `~/.environ.env`.

**2. Falscher Umgebungsvariablen-Name bei Custom Endpoints**
Wenn Sie einen Custom API Endpoint nutzen (z.B. GWDG), geben Sie in der Webapp unter **"API-Key Variable"** den exakten Namen an, den Sie in der `.env`-Datei verwendet haben, z.B. `GWDG_API_KEY`.

**3. Umgebungsvariable manuell setzen**
```bash
# Windows (Eingabeaufforderung, temporär für diese Sitzung):
set OPENAI_API_KEY=sk-proj-...

# Windows (dauerhaft):
setx OPENAI_API_KEY "sk-proj-..."

# macOS / Linux (temporär):
export OPENAI_API_KEY="sk-proj-..."

# macOS / Linux (dauerhaft in ~/.bashrc oder ~/.zshrc):
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.zshrc
```

**4. Proxy / Firewall blockiert**
Wenn Sie sich in einem Uni-Netzwerk befinden, kann ein Proxy die Verbindung blockieren. Kontaktieren Sie in dem Fall Ihre IT-Abteilung.

#### Problem: Leere oder unvollständige Antworten bei lokalen Modellen (LM Studio / Ollama)

**Symptom:** Die Analyse bleibt hängen oder liefert leere JSON-Antworten, besonders beim Start der ersten Analyse.

**Ursache:** Lokale Modelle brauchen manchmal einen "Warm-up"-Durchlauf. Cold-Start-Probleme sind bekannt.

**Lösung:**
- **Seit Version 0.12.7.2** wird automatisch ein zweiter Versuch mit 2s Pause unternommen, bevor ein Fehler gemeldet wird
- Falls das Problem weiter besteht: Einfach die Analyse neu starten – nach dem ersten Batch läuft es meist stabil
- Hilft auch: Ein kleiner Test-Prompt im LM Studio Chatfenster vor dem Start der Analyse

#### Problem: Custom API Key Environment Variable wird nicht erkannt

**Symptom:** Bei Nutzung eines Custom Endpoints (z.B. GWDG) erscheint "API-Key nicht gefunden", obwohl die Variable in der `.env`-Datei steht.

**Ursache vor 0.12.7.4:** Die `.env`-Datei wurde nirgendwo automatisch geladen.

**Lösung:**
- **Seit Version 0.12.7.4** wird `.env` automatisch geladen – einfach aktualisieren
- Stellen Sie sicher, dass der **Name der Umgebungsvariable** in der Webapp unter "API-Key Variable" exakt dem Eintrag in der `.env`-Datei entspricht
- Beispiel: In `.env` steht `GWDG_API_KEY=xyz` → in der Webapp "API-Key Variable" = `GWDG_API_KEY`

#### Problem: Analyse bricht mit JSON-Fehler ab

**Symptom:** `JSONDecodeError` oder "Ungültige JSON-Antwort" während der Analyse.

**Lösung:**
- **Seit Version 0.12.7.1/0.12.7.2** werden JSON-Fehler automatisch abgefangen:
  - Bei leeren Antworten: automatischer Wiederholungsversuch mit Pause
  - Bei fehlerhaften JSON: Reparaturversuch (Klammern schließen, Kommas entfernen)
  - Bei anhaltenden Fehlern: Überspringen des betroffenen Batches (kein Datenverlust)
- Falls die Analyse dennoch abbricht: Einfach neu starten – der Fortschritt wurde automatisch gespeichert

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

## 14. Anhang: Beispielkonfigurationen und Vorlagen

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

#### Beispiel 4: Sensible Daten mit Custom Endpoint (GWDG)

**Forschungskontext:**
- Personenbezogene Interviews, keine Cloud-Dienste erlaubt
- Hochschule stellt eigenen KI-Endpoint via GWDG Academic Cloud
- Datenschutzkonforme Verarbeitung im Rechenzentrum der Hochschule

**Konfiguration:**
```json
{
  "config": {
    "MODEL_PROVIDER": "OpenAI",
    "MODEL_NAME": "openai-gpt-oss-120b",
    "API_BASE_URL": "https://chat-ai.academiccloud.de/v1",
    "API_KEY_ENV": "GWDG_API_KEY",
    "ANALYSIS_MODE": "deductive",
    "CHUNK_SIZE": 1000,
    "BATCH_SIZE": 5,
    "CODE_WITH_CONTEXT": true
  }
}
```

**Dazugehörige `.env`-Datei:**
```
GWDG_API_KEY=xxxxxxxxxxxxxxxxxxxxx
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

QCA-AID wird kontinuierlich weiterentwickelt. Die aktuellste Version ist **0.12.7.4** (Juni 2026). Aktuelle Entwicklungen und Updates finden Sie im [GitHub-Repository](https://github.com/JustusHenke/QCA-AID) und im [Changelog](CHANGELOG.md).

**Kontakt für Feedback und Fragen:**  
Justus Henke  
Institut für Hochschulforschung Halle-Wittenberg  
E-Mail: justus.henke@hof.uni-halle.de

---

**Viel Erfolg bei Ihrer qualitativen Forschung mit QCA-AID!** 🚀

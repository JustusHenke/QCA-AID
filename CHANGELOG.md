# Changelog

## Versionen und Updates

---

## Neu in 0.12.9.3 (2026-07-01)

### ⚡ Parallele Relevanzprüfung und Kategoriepräferenzen

**Problem:** Die Relevanzprüfung (Schritt 1) und Kategoriepräferenzen (Schritt 2) verarbeiteten Batches sequenziell – obwohl jeder Batch unabhängig ist und keine Abhängigkeiten zwischen Batches bestehen.

**Lösung:** `asyncio.gather()` für Relevanz- und Kategoriepräferenz-Batches:

- **`analyze_relevance_simple`** (`unified_relevance_analyzer.py`):
  - Neue Helper-Methode `_process_single_relevance_batch()` mit vollständiger Retry/Fallback-Logik
  - Hauptmethode erstellt Tasks für alle Batches und führt sie mit `asyncio.gather(*, return_exceptions=True)` parallel aus
  - Ergebnisse werden nach Abschluss in Reihenfolge zusammengeführt

- **`analyze_category_preferences`** (`unified_relevance_analyzer.py`):
  - Analoge Refaktorisierung mit `_process_single_category_preferences_batch()`
  - Parallele Verarbeitung aller Kategoriepräferenz-Batches

**Gesamte Pipeline jetzt parallel:**

```
Schritt 1: Relevanz-Batches        → 🚀 PARALLEL (asyncio.gather)
Schritt 2: Kategoriepräferenzen     → 🚀 PARALLEL (asyncio.gather)
Schritt 3: auto_1 + auto_2 Kodierung → 🚀 PARALLEL (asyncio.gather, seit 0.12.9)
```

**Erwarteter Effekt:**
- Schritt 1: −60-70% (14 Batches parallel statt sequenziell)
- Schritt 2: −60-70% (10 Batches parallel statt sequenziell)
- Gesamtlaufzeit bei 105 Segmenten: von ~1h auf ~20-30 Min geschätzt

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.9.3 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.9.2 (2026-07-01)

### 🐛 Bugfix: LLM gibt "SEGMENT N" als segment_id zurück statt des korrekten Dokumentnamens

**Problem:** In Batches mit 8 Segmenten gab das LLM gelegentlich "SEGMENT 1", "SEGMENT 2" etc. als `segment_id` in der JSON-Antwort zurück, obwohl der Prompt den korrekten Namen enthielt (z.B. `keinDS_na_privat_HS395.docx_chunk_5`). Der Grund: Der Prompt formatiert Segmente als `SEGMENT {i+1} (ID: {seg['segment_id']})`, und das LLM wählte vereinfachend das Präfix statt der ID in Klammern.

**Symptom:** Die Export-Tabelle zeigte 18 "Unbekanntes_Dokument"-Zeilen mit `segment_id` wie "SEGMENT 1-1", "SEGMENT 2-2" etc., die nicht zu den echten segment_ids passten.

**Fix** in `unified_relevance_analyzer.py._parse_batch_result()`:
- Validiert jede LLM-Antwort-`segment_id` gegen die Menge der Original-segment_ids
- Erkennt "SEGMENT N"-Pattern und ersetzt es durch die korrekte `segment_id` via positionalem Mapping (idx-th Result → idx-th Segment)
- Bei allen nicht-passenden IDs: Fallback auf positionales Mapping

**Gleichzeitig** (0.12.9.1): Text-Fix in `controller.py` – `result["text"]` wird jetzt immer gesetzt, nicht nur bei Mehrfachkodierung.

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.9.2 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.9.1 (2026-07-01)

### 🐛 Bugfix: Segment-Texte fehlten im Export bei normaler Kodierung

**Problem:** Die Export-Tabelle zeigte für die meisten Segmente "Unbekanntes_Dokument" und "[Original-Text nicht verfügbar]" statt der echten Dokumentnamen und Transkript-Texte. Die Coding-Matrix war korrekt (Dokument-Spalten zeigten richtige Namen), aber die Segment-Text-Tabelle war leer.

**Root Cause:** In `_batch_analyze_deductive()` (und analog in inductive/abductive) wurde der Text aus dem Segment nur bei **Mehrfachkodierung** (`first_result["result"]["text"] = original_text`) in das Ergebnis geschrieben. Bei **normaler Kodierung** wurde `_format_single_coding_result()` aufgerufen, das kein `text`-Feld im Result-Dictionary setzte. Der Export versuchte dann vergeblich, den Text über `segment_id` im Cache zu finden – aber der Cache war nur aus den fehlerhaften Coding-Ergebnissen befüllt.

**Fix (4 Stellen):**
- `_batch_analyze_deductive` (deductive): `formatted_res["result"]["text"] = original_text` nach `_format_single_coding_result()`
- `_batch_analyze_deductive` (inductive): `ind_format_res["result"]["text"] = original_text` nach `_format_single_coding_result_inductive()`
- `_batch_analyze_deductive` (abductive): `abd_format_res["result"]["text"] = original_text` nach `_format_single_coding_result_abductive()`
- Parallele Closures (deductive/abductive): `formatted_res["result"]["text"] = segment["text"]` bei Einzelsegment-Verarbeitung

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.9.1 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.9 (2026-07-01)

### ⚡ Parallele Multi-Kodierung — API-Laufzeit halbiert

**Problem:** Im Multi-Kodierer-Modus (2 Kodierer mit unterschiedlichen Temperaturen für Intercoder-Reliability) wurden beide Kodierer **strikt sequenziell** verarbeitet – auto_1 musste alle Batches abschließen, bevor auto_2 starten konnte. Bei 105 Segmenten mit 13 Batches und Mehrfachkodierung resultierte das in einer Kodierungszeit von über 2 Stunden (statt erwarteter ~30 Minuten). Die `Geschwindigkeit`-Anzeige sank kontinuierlich, da der Durchschnittsberechner `total_segments / elapsed_time` den steigenden Zeitverbrauch widerspiegelte.

**Ursachenanalyse:**
- `asyncio` war bereits importiert und `AsyncOpenAI` unterstützt parallele Requests
- Der `OptimizationController` rief jedoch `await self._batch_analyze_deductive(...)` sequenziell im `for coder_config in coder_settings:`-Loop auf
- Jeder `await`-Aufruf blockierte den Event-Loop bis zum Abschluss des gesamten Batches inkl. aller fokussierten Mehrfachkodierungs-Calls
- Der `DynamicCacheManager` war für die Kodierungs-Performance irrelevant (0% Hit Rate, Cache ist ein No-Op bei frischen Analysen)

**Lösung:** `asyncio.gather()` für parallele Kodierung beider Kodierer in allen Analysemodi:

- **Deductive Mode** (`_analyze_deductive`):
  - Statt `for coder_config: await _batch_analyze_deductive()` → `_coder_task_deductive()` async-Funktion pro Kodierer + `asyncio.gather(*tasks, return_exceptions=True)`
  - Jeder Kodierer verarbeitet seinen gesamten Batch inkl. fokussierter Mehrfachkodierung unabhängig
  - Ergebnisse werden erst nach `gather` gemeinsam in `all_results` eingefügt

- **Inductive Mode** (`_analyze_inductive`):
  - Statt `for coder_config: await _analyze_inductive_coding_only()` → direkte Task-Erstellung + `asyncio.gather()`
  - `_store_results_for_reliability()` wird erst nach Abschluss aller Kodierer aufgerufen (Thread-sicher)

- **Abductive Mode** (`_analyze_abductive`):
  - Gleiches Muster wie Deductive: async Closure-Funktion `_coder_task_abductive()` pro Kodierer + `asyncio.gather()`
  - Funktioniert sowohl für Batch- als auch für Einzelsegment-Verarbeitung

**Sicherheitsgarantien:**
- `return_exceptions=True` → Ein fehlgeschlagener Kodierer killt den anderen nicht
- AsyncIO ist single-threaded (Cooperative Multitasking) → Keine Race Conditions im Cache oder in der Reliability-DB
- Closure-Variablen (`_cid`, `_ctemp`) werden als Parameter übergeben (kein Late-Binding-Fehler)
- `Grounded Mode` bleibt unverändert (nur Single-Coder)

**Erwarteter Effekt:**
- Kodierungszeit: **−50%** (beide Coder laufen parallel statt sequenziell)
- Relevanzprüfung und Kategoriepräferenzen waren bereits shared (1× API-Call für alle Kodierer)
- Gesamtlaufzeit bei deduktiver Analyse mit 105 Segmenten: von ~2h auf ~1h geschätzt

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.9 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.8.4 (2026-07-01)

### 🛡️ API-Fehler-Resilienz — Drei-Stufen-Schutz gegen vorzeitigen Abbruch

**Problem:** Ein transientes API-Fehler (z.B. HTTP 500) während der Kodierung löste einen Kaskadeneffekt aus: Die Exception propagierte ungedämpft durch alle Schichten, verlor dabei alle bisherigen Teilergebnisse, und der Fallback auf die Standard-Analyse fand keine Segmente mehr vor (da `processed_segments` bereits alle als „verarbeitet" markiert hatte). Das Ergebnis: 0 verarbeitete Batches, komplett verlorene Analyse.

**Lösung:** Drei unabhängige Schutzmechanismen:

1. **Retry mit exponential backoff** (`unified_relevance_analyzer.py`):
   - Transiente API-Fehler (500, 502, 503, 429, 529) werden erkannt und bis zu 2 Mal mit steigender Wartezeit (2s → 3s) wiederholt
   - Nicht-transiente Fehler werden weiterhin sofort geworfen
   - Betroffene Methode: `analyze_batch()` im `UnifiedRelevanceAnalyzer`

2. **Fehlerisolierte Kodierer-Verarbeitung** (`controller.py`):
   - `try/except` pro Kodierer-Konfiguration in der deduktiven Batch-Schleife
   - Fehler in einem Kodierer (z.B. `auto_2`) verlieren nicht die Ergebnisse anderer Kodierer (z.B. `auto_1`)
   - Betroffene Methode: `_analyze_deductive()` im `OptimizationController`

3. **Funktionierender Standard-Analyse-Fallback** (`analysis_manager.py`):
   - Beim Fallback auf die Standard-Analyse werden die `processed_segments` für die betroffenen Segmente zurückgesetzt
   - Die Hauptschleife (`_get_next_batch`) kann die Segmente jetzt korrekt erneut aufgreifen
   - Betroffene Stelle: `except`-Block in `_analyze_normal_modes()`

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.8.4 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.8.3 (2026-07-01)

### 🔧 ConfigLoader — Codebook-Pfad wird jetzt korrekt aufgelöst

**Problem:** Wenn der Benutzer ein Codebook mit abweichendem Dateinamen (z.B. `QCA-AID-Codebook_deductive.json`) geladen hatte, ignorierte der ConfigLoader diesen Pfad und fiel auf die hartcodierte Standarddatei `QCA-AID-Codebook.json` zurück. Dies führte dazu, dass vordefinierte Kategorien nicht geladen wurden (`0 deduktive Kategorien geladen`) und der gesamte deduktive Workflow mit einem `AttributeError: 'NoneType' object has no attribute 'items'` abbrach.

**Lösung:** Dreistufige Pfad-Auflösung im ConfigLoader (Reihenfolge: Env-Var → Projekt-Settings → Fallback):

- **`ConfigLoader._resolve_last_codebook_path()`** — Neue Methode:
  1. Prüft Env-Var `QCA_AID_CODEBOOK_PATH` (höchste Priorität, gesetzt vom Webapp-Subprocess)
  2. Liest `last_codebook_file` aus `.qca-aid-project.json` (Projekt-Settings)
  3. Fallback auf `QCA-AID-Codebook.json` im Projekt-Root
- **`AnalysisRunner`** — Übergibt `CODEBOOK_PATH` als Env-Var `QCA_AID_CODEBOOK_PATH` an den Subprocess
- **`analysis_ui.py`** — `CODEBOOK_PATH` wird aus `session_state.current_config_filepath` ins config_dict übernommen
- **`main.py`** — Zeigt `📄 Config-Datei: <Pfad>` prominent im Log an (nach ConsoleLogger-Start)
- **`config_ui.py`** — `📁 Aktuelle Konfiguration:` zeigt jetzt den tatsächlich geladenen Dateinamen statt den hartcodierten Standard

### 🐛 NoneType-Fehler bei leeren Kategorie-Definitionen

**Problem:** `cat_defs if cat_defs else None` in `analysis_manager.py` konvertierte ein leeres Dict `{}` (falsy in Python) zu `None`, was in `controller.py._serialize_category_definitions()` zu `AttributeError: 'NoneType' object has no attribute 'items'` führte.

**Fix:**
- **`analysis_manager.py`** — `cat_defs if cat_defs else None` → `cat_defs` (leeres Dict wird korrekt weitergegeben)
- **`controller.py`** — Früher Guard in `_serialize_category_definitions()`: Bei `None` oder `{}` wird sofort `{}` zurückgegeben

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.8.3 gehoben (Datum 2026-07-01).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.8.2 (2026-06-23)

### 🧬 Grounded Mode — Phase 1.5: Konzeptuelle Subcode-Verdichtung via LLM

Nach dem ersten Fix in 0.12.8.1 (strikte ID-basierte Subcode-Zuordnung) wurde ein zweites, tieferliegendes Architektur-Problem sichtbar: Die in Phase 1 vom LLM gesammelten Subcodes sind **deskriptive Einzelbeobachtungen auf gleicher Abstraktionsebene** – oft Paraphrasen oder Detailvarianten voneinander. In einem realen Lauf mit 182 Roh-Subcodes führte das harte `max_subcategories=5`-Limit pro Hauptkategorie dazu, dass **188 von 213 Subcodes (88%) verworfen** wurden, weil 36-49 Subcodes je Hauptkategorie konkurrierten.

**Fix:** Neuer Zwischenschritt **Phase 1.5** zwischen Subcode-Sammlung und Hauptkategorien-Generierung.

- **Neue Methode** `OptimizationController._consolidate_subcodes_with_llm(subcodes, research_question, target_count=50)`:
  - Fordert das LLM auf, ähnliche Roh-Subcodes konzeptuell zu abstrahierteren Subcodes zusammenzufassen
  - Strikt vertikale Verdichtung (Subcode-Hierarchie) – **keine** Hauptkategorien-Bildung
  - Jeder verdichtete Subcode enthält `merged_from_ids` (vollständige Provenienz) und `merge_reason` (Begründung)
  - Ziel-Anzahl dynamisch: `max(30, min(50, subcodes_count // 2))`
  - **Vorsichtsprinzip**: Originale, die das LLM nicht zuordnet, bleiben zusätzlich erhalten (lieber behalten als versehentlich verlieren)
  - **Sättigungs-Einschätzung**: Das LLM gibt zusätzlich eine Einschätzung zur thematischen Sättigung zurück
  - **Robuster Fallback**: Bei API-Fehler werden Original-Subcodes unverändert verwendet – die Analyse läuft weiter

- **Integration in `analysis_manager.py`** zwischen Phase 1 und Phase 2:
  - IDs werden nach Verdichtung neu vergeben (1, 2, 3, ...) für saubere Phase-2-Referenzierung
  - Verwendet dieselbe LLM-Provider-Instanz wie der Rest der Pipeline

**Drei-Stufen-Architektur jetzt:**

```
Phase 1   Roh-Subcodes (LLM, deskriptiv, 100+ Einträge)
Phase 1.5 Pre-Clustering (LLM, konzeptuell → ca. 50 Einträge)        ← NEU
Phase 2   Hauptkategorien (LLM, thematische Zuordnung)
          + deterministisches Ranking auf max_subcategories
Phase 3   Kodierung (LLM, gegen Hauptkategorien + Subkategorien)
```

**Erwarteter Effekt:** Statt 88% Informationsverlust durch das harte Ranking sollten in der finalen Tabelle jetzt deutlich mehr abstrahierte, aber inhaltlich valide Subkategorien sichtbar sein. Das LLM bekommt in Phase 2 außerdem ein **deutlich saubereres Subcode-Set** – weniger Drifts und Paraphrasen-Rauschen.

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.8.2 gehoben (Datum 2026-06-23).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.8.1 (2026-06-23)

### 🐛 Grounded Mode — Subkategorien wurden verschluckt

Im Grounded Mode konnten nach Phase 2 (Hauptkategorien-Generierung) nur noch **1 von 182** Subcodes tatsächlich als Subkategorie zugeordnet werden – die Export-Tabelle zeigte daher trotz großer Materialbasis praktisch leere Subkategorie-Zeilen.

**Root Cause** (drei kaskadierende Bugs):

1. **Phase-2-Prompt war überlastet**: Das LLM sollte gleichzeitig 182 Subcodes gruppieren, paraphrasierte Namen liefern (`related_subcodes`) und eigenständig auf `max_subcategories=5` verdichten. Bei Modellen wie `qwen3.6-35b-a3b` resultierte das in Namens-Drift – das LLM formulierte Subcode-Namen um, und der strikte `==`-Lookup in der Folgeverarbeitung schlug fehl.
2. **Striktes Namens-Matching** in `_parse_main_categories_from_grounded`: Bereits ein abweichender Buchstabe führte zum Verlust der Subkategorie.
3. **Phase 3 ohne Subkategorien**: Da Phase 2 bereits leere `subcategories` zurückgab, konnte das LLM in Phase 3 keine Subkategorien vergeben.

**Fix (in `QCA_AID_assets/optimization/controller.py`):**

- **Striktes ID-basiertes Mapping**: Gesammelte Subcodes erhalten beim Anlegen eine deterministische `id` (`1, 2, 3, ...`). Der Phase-2-Prompt präsentiert sie als `[ID 42] Subcode-Name` und fordert ausschließlich numerische IDs in `related_subcode_ids` zurück – **kein** Freitext, **keine** Paraphrasierung.
- **`_parse_main_categories_from_grounded` nutzt strikten Lookup** auf `id` bzw. Original-Name (case-insensitive). Unbekannte Referenzen werden mit Warnung geloggt, statt unbemerkt verloren zu gehen.
- **Verdichtung wandert aus dem LLM-Prompt**: Die `max_subcategories`-Regel steht nicht mehr im Prompt (verhindert Drift), sondern wird durch die neue Methode **`_enforce_max_subcategories_limit`** deterministisch umgesetzt.
- **Ranking-Kriterien** der neuen Methode (in Reihenfolge der Priorität): Confidence des Subcodes → Anzahl Textbelege → Anzahl unterschiedlicher Keywords → kürzerer Name.
- **Diagnostik**: Nicht aufgelöste Subcode-IDs werden im Log mit `⚠️ X Subcode-Referenzen aus Phase 2 konnten nicht eindeutig zugeordnet werden und wurden verworfen` ausgewiesen, statt unbemerkt zu verschwinden.
- **Abwärtskompatibilität**: Die alten Feldnamen (`related_subcodes`, `subcodes`, `assigned_subcodes`) werden weiterhin als Fallback akzeptiert.

**Erwartetes Verhalten beim nächsten Lauf:** alle Hauptkategorien zeigen jetzt `└─ X Subkategorien`-Zeilen, und die Export-Tabelle enthält korrekt zugeordnete Subkategorien pro Hauptkategorie (begrenzt durch `MAX_SUBCATEGORIES` aus der Config). HINWEIS: Bei hoher Subcode-Anzahl (>50) wird das harte 5er-Limit in Phase 2 trotzdem viele Subkategorien verwerfen – Abhilfe in Version 0.12.8.2 (Phase 1.5 Pre-Clustering).

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.8.1 gehoben (Datum 2026-06-23).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.8 (2026-06-18)

### 🧪 Grounded Mode — Forschungsfrage als einziges Pflichtfeld

Im **Grounded Mode** (`ANALYSIS_MODE=grounded`) ist im Codebook ab sofort **ausschließlich die Forschungsfrage** verpflichtend. Deduktive (vordefinierte) Kategorien sind **optional**, da das Hauptkategorien-System während der Analyse **emergent** aus dem Material entsteht (3-Phasen-Workflow: Subcode-Sammlung → Hauptkategorien-Generierung → Kodierung).

- **`CodebookData.validate(analysis_mode=...)`** akzeptiert jetzt einen optionalen `analysis_mode`. Im Grounded Mode wird die Kategorie-Pflicht übersprungen – nur die Forschungsfrage wird validiert.
- **Codebook-UI**: Modus-aware Status-Banner. Im Grounded Mode erscheint ein eigener Hinweis („Hauptkategorien werden emergent gebildet – Forschungsfrage reicht"), statt der bisherigen „Aktion erforderlich"-Warnung.
- **Analyse-UI**: Der Readiness-Check blockiert im Grounded Mode nicht mehr wegen eines leeren Codebooks. Stattdessen wird eine Warnung angezeigt, dass die emergent-generierten Kategorien noch fehlen – das ist erwartetes Verhalten.
- **Rückwärtskompatibilität**: Alle anderen Modi (`deductive`, `inductive`, `abductive`) verhalten sich unverändert – das leere Codebook bleibt dort ein harter Fehler.

### 🎛️ Neue Konfigurationsoption: Maximale Subkategorien je Hauptkategorie

- **Neues Feld `max_subcategories`** in `ConfigData` (Default: **5**, Bereich 1–50). Wird in JSON als `max_subcategories` und in `CONFIG` als `MAX_SUBCATEGORIES` persistiert.
- **UI-Feld in der Konfiguration** erscheint **nur im Grounded Mode** (direkt unter dem Analyse-Modus-Dropdown) als Number-Input inklusive Tipp („3–7 ist für die meisten Studien ein guter Ausgangspunkt") und Sanity-Hinweis bei Werten >15.
- **Validierung**: Nur im Grounded Mode geprüft; in anderen Modi ignoriert.
- **Config-Manager** transportiert das Feld round-trip über JSON und XLSX (`get_default_config()`, `_load_from_xlsx()`).

### 🧠 Verdichtungs-Logik in Phase 2 (LLM-gestützt)

- **`OptimizationController.generate_grounded_main_categories(max_subcategories=N)`** akzeptiert den Parameter und reicht ihn an den Prompt weiter.
- **`_build_main_categories_generation_prompt(max_subcategories=N)`** blendet eine **Verdichtungs-Regel** in den LLM-Prompt ein, **wenn** die Anzahl gesammelter Subcodes den Maximalwert übersteigt. Die Regel lautet sinngemäß:
  - Maximal N Subkategorien je Hauptkategorie
  - Synonyme / sehr ähnliche Subcodes zu einer Subkategorie zusammenfassen
  - 3–12 Hauptkategorien insgesamt
- Bei kleinen Datenmengen (Subcodes ≤ max) **kein** Verdichtungs-Hinweis im Prompt → minimaler Prompt-Overhead.
- **Sicherheits-Clamp** auf `[1, 50]` in beiden Methoden.
- **`analysis_manager.py`** reicht `CONFIG['MAX_SUBCATEGORIES']` an den Controller durch.

### 📚 Dokumentation

- **`KONFIGURATION_ANLEITUNG.md`**: Neuer Abschnitt „Grounded Mode – Besonderheiten" erläutert Pflicht-/Optional-Felder, 3-Phasen-Ablauf und das neue `max_subcategories`-Feld mit Empfehlungen.
- **`QCA-AID-Nutzerhandbuch.md`**:
  - Abschnitt 3.4 (Grounded Theory Modus) um Hinweis-Box zur 0.12.8-Erweiterung ergänzt.
  - Abschnitt 11.5 komplett überarbeitet: Konfigurationsbeispiel mit `MAX_SUBCATEGORIES`, neuer Block „Codebook im Grounded Mode (ab 0.12.8)", Tabelle mit Werte-Empfehlungen und interner Erläuterung der 3-Phasen-Logik.
- **`examples/config-grounded.json`**: Enthält jetzt `"MAX_SUBCATEGORIES": 5`.

### 🧹 Sonstiges

- QCA-AID Version auf 0.12.8 gehoben (Datum 2026-06-18).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.12.7.4 (2026-06-05)

### 🔑 .env-Autoload für API-Keys

- **Automatisches Laden von `.env` aus dem Projektverzeichnis** in `start_QCA-AID-app.py` und `QCA_AID_app/start_webapp.py`
- Durchsucht 3 Orte in Prioritätsreihenfolge: aktuelles Arbeitsverzeichnis → Repository-Root → `~/.environ.env`
- Bevorzugt `python-dotenv` (falls installiert), sonst manuelles Fallback
- Lädt **vor** venv-Reexecution und **vor** Unicode-Fix, damit API-Keys von Beginn an in `os.environ` verfügbar sind
- Behebt: Bei Custom API Endpoints mit benutzerdefiniertem Env-Var-Name (z.B. `GWDG_API_KEY`) wurde der Key aus einer `.env`-Datei **nicht** automatisch gelesen, da nirgendwo `load_dotenv()` aufgerufen wurde.

### 🧹 Sonstiges

- QCA-AID Assets Version auf 0.12.7.4 gehoben
- CHANGELOG.md aktualisiert

---

## Neu in 0.12.7.3 (2026-06-04)

### 🌐 Custom API Provider — Erweiterte Flexibilität

- **Custom API Base URL für ALLE Provider sichtbar** (zuvor nur OpenAI/Local)
- **Modell-Feld wird Freitext-Input** sobald eine Custom API Base URL gesetzt ist — kein Dropdown-Zwang mehr, beliebige Modelleingabe möglich
- **Benutzerdefinierter Env-Var-Name für API-Key** (z.B. `GWDG_API_KEY`, `CUSTOM_API_KEY`) statt festem `OPENAI_API_KEY`
- Echtzeit-API-Key-Prüfung in der UI zum konfigurierten Env-Var-Namen
- Full-Stack-Integration: Factory → OpenAIProvider → alle Analyse-Module (deduktiv, induktiv, Relevanz, Explorer)
- API_KEY_ENV wird als Umgebungsvariable an den Analyse-Subprocess übergeben
- Defensive `getattr`-Zugriffe für Kompatibilität mit alten Session-State-Objekten

---

## Neu in 0.12.7.2 (2026-04-22)

### 🐛 Bugfixes / Robustheit

- **Retry-Logik bei leeren LLM-Antworten (lokale Modelle):**
  - Automatischer Retry mit 2s Pause bei leerer/ungültiger JSON-Antwort in der Relevanzprüfung
  - Hilft besonders bei Cold-Start-Problemen lokaler LLMs (LM Studio, Ollama)
  - Fallback greift erst nach fehlgeschlagenem Retry

- **Robustere LLMResponse-Verarbeitung:**
  - `None`-Content von LLM-Antworten wird zu leerem String normalisiert
  - Verhindert AttributeError bei leeren Antworten lokaler Server

---

## Neu in 0.12.7.1 (2026-04-22)

### 🐛 Bugfixes / Robustheit

- **Robustere JSON-Fehlerbehandlung im UnifiedRelevanceAnalyzer:**
  - Alle `json.loads()`-Aufrufe nach LLM-Antworten mit try/except abgesichert
  - Fallback-Strategie bei JSON-Parsing-Fehlern: Segmente werden als relevant markiert (statt verworfen), damit keine Daten verloren gehen
  - Betrifft: Standard-Relevanzprüfung, Kategorie-Präselektion, umfassende Analyse, induktive/abduktive/grounded Batches
  - Diagnostische Ausgabe (Auszug der LLM-Antwort) bei Parsing-Fehlern für einfacheres Debugging

- **Leisere Logging-Ausgabe im EscapeHandler:**
  - Fehlende `keyboard`-Modul-Warnung von `print()` auf `logging.debug()` umgestellt
  - Vermeidet störende Konsolenausgaben bei normalem Betrieb ohne installiertes `keyboard`-Modul

---

## Neu in 0.12.7 (2026-04-22)

### 🔧 Verbesserungen

- **Vereinheitlichtes Laden von Config und Codebook:**
  - Codebook wird jetzt automatisch beim Laden der Konfiguration mitgeladen (Config UI)
  - Separater "Codebook laden"-Dialog im Codebook-Tab entfernt (Redundanz beseitigt)
  - Gespeicherter Dateipfad wird für zukünftige Speicher-Operationen korrekt weitergegeben

- **Robustere Pfadauflösung:**
  - Relative Pfade werden in ConfigManager und CodebookManager korrekt gegen das Projektverzeichnis aufgelöst
  - Verhindert Fehler bei Pfadangaben ohne absoluten Pfad

- **Robustere Codebook-Datenverarbeitung:**
  - `CategoryData.from_dict()` behandelt None-Werte und unerwartete Typen graceful
  - `CodebookData.from_dict()` mit robuster Kodierregeln-Verarbeitung
  - Validierung gibt success=True zurück wenn Kategorien trotz Warnungen nutzbar sind

- **Session-State-basiertes Codebook-Management:**
  - Analyse-Tab und Webapp nutzen Session State statt wiederholtes Laden von Festplatte
  - Verbesserte Statusanzeige im Codebook-Tab (unterscheidet "geladen" vs. "keine Kategorien")

### 🔧 Aktualisierte LLM-Modellkonfigurationen

- **Anthropic:** Claude Opus 4.7 hinzugefügt (mit Reasoning-Levels)
- **OpenRouter:** 
  - Neue Modelle: Claude Opus 4.7, Kimi K2.6, inclusionAI Ling-2.6-flash (free)
  - Entfernte veraltete Modelle: Mercury, Mercury Coder, Llama 4 Maverick, GPT-4o extended, GPT-5 Image/Image Mini, Meituan LongCat Flash
  - Aktualisierte Preise und Parameter für zahlreiche Modelle (DeepSeek, Qwen, MiniMax, Mistral, xAI, Z.ai u.a.)

---

## Neu in 0.12.6.3 (2026-04-18)

### 🐛 Bugfixes / Robustheit

- **Robustere JSON-Reparatur bei abgeschnittenen LLM-Antworten:**
  - Neue `_close_brackets()`-Methode mit korrekter Stack-basierter Klammeranalyse (berücksichtigt Strings)
  - Erkennung und Schließung unterminated Strings
  - Entfernung von trailing Commas vor schließenden Klammern
  - Fallback-Strategie: Zeichenweises Abschneiden bis valides JSON entsteht
  - Ersetzt die bisherige naive Klammerzählung

- **Retry-Logik für Batch-Analyse bei JSON-Fehlern:**
  - Bis zu 2 Wiederholungsversuche bei `JSONDecodeError` in der Batch-Kodierung
  - Graceful Skip: Bei persistentem Fehler wird der Batch übersprungen statt die gesamte Analyse abzubrechen
  - Nicht-JSON-Fehler werden weiterhin sofort geworfen

### 🔧 Verbesserungen

- **Aktualisierte LLM-Modellkonfigurationen:**
  - OpenAI: Neue Modelle (GPT-5.4, GPT-5.4 Pro, GPT-5.4 Nano) hinzugefügt
  - Anthropic: Aktualisierte Modellpreise und -konfigurationen
  - OpenRouter: Erweiterte Modellliste mit aktuellen Preisen

---

## Neu in 0.12.6 (2026-03-05)

### 🌐 Custom Provider Integration

**Custom API Base URL Support für OpenAI-kompatible Endpoints:**
- ✨ **GWDG Academic Cloud Integration**: Vollständige Unterstützung für institutionelle OpenAI-kompatible APIs
  - Neues optionales Konfigurationsfeld `api_base_url` in ConfigData
  - UI-Integration im Konfigurationsreiter mit Validierung und Beispielen
  - Automatische Übergabe der Base URL an alle Analyse-Module
  - Beispiel: `https://chat-ai.academiccloud.de/v1`

- ✨ **Erweiterte Provider-Unterstützung**:
  - Azure OpenAI: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
  - Lokale OpenAI-kompatible Server (LM Studio, Ollama, Text Generation WebUI)
  - Beliebige OpenAI-kompatible Endpoints

- ✨ **Backend-Implementierung**:
  - `OpenAIProvider` unterstützt jetzt `base_url` Parameter im Constructor
  - `LLMProviderFactory.create_provider()` akzeptiert `base_url` Parameter
  - Alle Analyse-Module übergeben `base_url` an Provider (deductive_coding, inductive_coding, relevance_checker, explorer)
  - CONFIG Dictionary erweitert um `API_BASE_URL` Feld

- ✨ **UI-Features**:
  - Neuer Expander "🔧 Erweiterte Einstellungen: Custom API Base URL"
  - Eingabefeld mit Echtzeit-Validierung (muss mit http:// oder https:// beginnen)
  - Detaillierte Anleitung speziell für GWDG Academic Cloud
  - Beispielkonfigurationen für verschiedene Anwendungsfälle
  - Wird nur bei OpenAI und Local Providern angezeigt

- 📚 **Dokumentation**:
  - `CUSTOM_PROVIDER_GUIDE.md`: Vollständige technische Anleitung
  - `GWDG_INTEGRATION_ANLEITUNG.md`: Kurzanleitung speziell für GWDG
  - README.md aktualisiert mit neuem Abschnitt "Custom OpenAI-kompatible Endpoints"
  - Beispielkonfigurationen für JSON und Excel

**Technische Details:**
- Base URL wird aus der Konfiguration gelesen und an den OpenAI Client übergeben
- Validierung der Base URL (muss mit http:// oder https:// beginnen)
- Backward-kompatibel: Wenn keine Base URL angegeben ist, wird die Standard-OpenAI-URL verwendet
- Funktioniert mit allen Analysemodi (deductive, inductive, abductive, grounded)

**Anwendungsbeispiel GWDG:**
```json
{
  "model_provider": "OpenAI",
  "model_name": "openai-gpt-oss-120b",
  "api_base_url": "https://chat-ai.academiccloud.de/v1"
}
```

---

## Neu in 0.12.5 (2026-01-27)
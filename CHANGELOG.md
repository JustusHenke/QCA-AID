# Changelog

## Versionen und Updates

---

## Neu in 0.16.0 (2026-11-24)

### Textanker: Wortlautanker fuer den Pruefen-Tab

#### Details
- **Neue Spalte `Textanker` in `Kodierungsergebnisse`:** Jede kodierte Zeile führt die maßgeblichen, zusammenhängenden, wortwörtlichen Textausschnitte des Chunks - per LLM erfragt (Feld `Konkrete Textstellen` in allen Kodierpfaden), gegen den Chunktext verifiziert, ohne Zeilenumbrüche `" | "`-getrennt (max. 6 je max. 1000 Zeichen). Nicht kodierte Zeilen lassen die Zelle leer; Paraphrasen und erfundene Stellen werden verworfen.
- **Präzisere Markierungen im Prüfen-Tab:** Der Viewer löst die Textanker einzeln im PDF auf - mehrere Stellen je Kodierung, auch seitenübergreifend. Scheitert ein einzelner Anker, bleiben die übrigen unberührt; scheitern **alle**, greift wie bisher die gestaffelte Chunk-Markierung (1200/800/500/240), danach "ohne Anker". Die Garantie "nie falsch platzieren" gilt unverändert.
- **Panel mit Anker-Regionen:** Mehrfach verankerte Kodierungen zeigen je Region eine klickbare Zeile "S. N · ‹Zitatvorschau›" (aktive Region mit ▶, übrige Regionen derselben Kodierung auf 35 % abgeschwächt). Geteilte Stellen mehrerer Kodierungen weiter mit Schraffur und Randstreifen; Hinweistext erklärt die Anker als KI-generierte, menschlich prüfbare Wortlautstellen.
- **Rezensionspaket + Writeback konsistent:** Das Blatt `Kodierungen` unter `output/review/` führt die Spalte `Textanker` mit; beim Zurückschreiben in die Analyse-XLSX bleibt sie unangetastet (additiv wie alle Bestandsspalten). Alte XLSX ohne die Spalte laden weiter wie bisher.
- **Tests:** Export-Formatierung (24), Anker-Kette mit Voll-Fallback (22), Mehrfach-Regionen im Overlay und Panel (15) sowie neue Roundtrip- und End-to-End-Tests; Suite vollständig grün.

### 🧹 Sonstiges

## Neu in 0.15.4 (2026-11-24)

### Prüfen-Tab: Entscheidungen in die Analyse-XLSX übernehmen

#### Details
- **Ein Ziel, additiv geschrieben:** Der Knopf „In Analyse-XLSX übernehmen“ (Export-Bereich des Prüfen-Tabs) ergänzt die zentrale Arbeitsmappe um fünf Spalten in `Kodierungsergebnisse` — `Review_Status` (bestätigt/verworfen/offen), `Hauptkategorie_Review`, `Subkategorien_Review`, `Review_Änderung` und `Review_Stand`. Bestandsspalten, Häufigkeitsanalysen, Intercoder-Bericht und Kategorien bleiben unverändert; die Statuszelle wird eingefärbt (grün/rot/gelb), Änderungen orange.
- **Neues Blatt `Review_Entscheidungen`:** eine Zeile je Kodierung mit `Hauptkategorie_vorher`/`_nachher`, `Subkategorien_vorher`/`_nachher`, Status, geänderten Feldern und Stand — direkt hinter `Kodierungsergebnisse`, mit Filter und fixierter Kopfzeile.
- **Sicherung vor jedem Schreiben:** `<output>/backup/<datei>_vor-review_<stempel>.xlsx`. Der Stempel entspricht dem der Excel-Ausgabe. Meldet Excel die Datei als gesperrt, bricht die Übernahme mit Hinweis ab statt halb zu schreiben.
- **Rückweg beim Laden:** Liegt `Hauptkategorie_Review` vor, gilt sie als aktueller Code; `Review_Status` bestimmt bestätigt/verworfen. Damit sind die Entscheidungen über Sitzungen hinweg in der zentralen Datei sichtbar und der Prüfen-Tab öffnet mit dem letzten Stand. Ein geänderter Code markiert die Kodierung als menschliche Entscheidung (`created_by = human`).
- **Headless und getestet:** `webapp_logic/review_writeback.py` (`write_review_into_xlsx`, `change_summary`, `WritebackError`) plus 16 Tests (`test_review_writeback.py`) und Verdrahtungstests (`test_review_tab_wiring.py`), inklusive Round-Trip Schreiben → Laden.

## Neu in 0.15.3 (2026-11-24)

### PDF-Export: annotierte PDFs in output/Annotated/ mit Namensstempel, keine unannotierten Zwischen-PDFs

#### Details
- **Ein Unterordner, ein Stempel:** Die Analysis-Pipeline legt annotierte PDFs unter `output/Annotated/` ab und versieht sie mit demselben Namensstempel wie die Excel-Ausgabe: `<dokument>_QCA_annotiert_<YYYYMMDD_HHMMSS>.pdf`. Der Stempel wird einmal pro Lauf erzeugt, alle PDFs eines Exports gehören damit sichtbar zusammen.
- **Keine unannotierten PDFs mehr:** DOCX/TXT werden für die Annotation nur noch in einem temporären Arbeitsverzeichnis (`qca_aid_convert_*` im System-Temp) nach PDF konvertiert und nach dem Annotieren gelöscht. Im Ausgabeverzeichnis bleibt ausschließlich das annotierte PDF; die frühere Zwischendatei `<dokument>.pdf` entfällt.
- **Neue Pfadregel als Modul:** `QCA_AID_assets/utils/export/paths.py` bündelt `ANNOTATED_SUBDIR`, `export_stamp()`, `annotated_pdf_path()` und `conversion_workspace()`. `DocumentToPDFConverter.convert(..., target_dir=...)` akzeptiert ein Zielverzeichnis, `PDFAnnotator.annotate_pdf_with_codings()` nutzt ohne `output_path` dieselbe Regel (Ablage neben der Quelle in `Annotated/`).
- **Tests:** `QCA_AID_app/tests/test_annotated_pdf_export.py` (Pfadregel, Stempelformat, Aufräumen des Arbeitsverzeichnisses, Konverter-Zielverzeichnis, beide Exporter, ein Stempel für mehrere Dateien).

### Änderungen aus dem Prüfen-Tab bleiben im Rezensionspaket

- Kodierungsänderungen im Viewer (bestätigen, verwerfen, anderen Code zuweisen) wirken auf den Sitzungszustand und werden **nicht** in die Analyse-XLSX zurückgeschrieben. Sie werden als eigenes, prüfbares Paket nach `output/review/` exportiert: annotiertes PDF, `<dokument>_review_<stamp>.xlsx` (Blatt `Kodierungen`, `Aenderungen`, `Meta`) und `<dokument>_review_<stamp>.json`. Die Analyse-Ausgabe bleibt unangetastet – der Vergleich „vorher/nachher“ steht im Blatt `Aenderungen`.

## Neu in 0.15.2 (2026-11-24)

### Prüfen-Tab: Schraffur für Mehrfachkodierungen, Auswahl hervorheben, „Abbrechen“ im Ändern-Dialog

- **Schraffur statt Übermalen:** Im Textmarker-Modus lag die zweite Kodierung derselben Textstelle nur als dünner Randstreifen an – optisch kaum zu sehen. Jetzt bekommt die Basiskodierung den flächigen Textmarker, jede weitere Kodierung eine diagonale Schraffur in ihrer Kategoriefarbe über exakt dieselben Quads (plus weiterhin den Randstreifen zum Anklicken). SVG-`<pattern>` je Kodierung, abschaltbar über `RenderOptions.hatch_shared`.
- **Klick auf den Code wählt die Kodierung:** Der Kategoriename in der Liste ist jetzt ein Button. Ein Klick setzt die Kodierung aktiv („▶“ und blauer Hinweis), springt bei Bedarf auf ihre Seite und hebt sie im Viewer hervor: die gewählte Markierung bleibt voll deckend, alle anderen werden auf 35 % abgeschwächt.
- **„Ändern“ ist inline und abbrechbar:** Statt eines Popovers öffnet der Knopf „✎ Ändern“ einen Bereich mit „Diesen Code übernehmen“, „Anlegen und übernehmen“ und **„Abbrechen“**; Abbrechen schließt ohne Änderung. Segment-ID, Konfidenz, Begründung und Inferenz-Hinweis bleiben im Popover „Details“.

#### Technische Hinweise

- `QCA_AID_assets/annotation/renderer.py`: neu `_add_hatch` (Pattern-Defs je Kodierung), `_emphasis` (Deckkraft der Auswahl), `HATCH_SIZE`, `DIM_OPACITY`; `_add_rect` kann ein `opacity`-Attribut setzen.
- `QCA_AID_assets/annotation/model.py`: `RenderOptions.hatch_shared` (Default `True`, in `to_dict`/`from_dict`).
- `QCA_AID_app/viewer/coding_panel.py`: Parameter `on_select`, Editier-Zustand in `session_state[f"{key_prefix}_editing"]`, Inline-Editor mit „Abbrechen“.
- `QCA_AID_app/webapp_components/review_ui.py`: `_select_coding` setzt `review_active_segment` (und die Seite) und rerunnt; das Overlay wird über `active_coding_id` neu gebaut, ohne den Viewer neu zu laden.
- Tests: AppTest-Smoke für Öffnen/Abbrechen des Dialogs und Klick auf den Code; Renderer-Tests für Schraffur, Auswahl-Deckkraft und `hatch_shared=False`. Gesamt 487 Tests.

### 🧹 Sonstiges

## Neu in 0.15.1 (2026-11-24)

### Prüfen-Tab: Mehrfachkodierung sichtbar, seitenbezogener Panel-Kopf

Anlass: Von 7 Kodierungen in `text_demo_wiss.pdf` waren im Viewer nur wenige Markierungen zu sehen.

- **Eine Textstelle, ein Highlight, eigene Stripes je Kodierung:** Wird derselbe Chunk mehrfach kodiert (z. B. `Legitimationsstrategien` *und* `Strukturelle_Rahmenbedingungen`), sind die Quads beider Kodierungen identisch – die zweite Markierung lag exakt auf der ersten und hat sie unsichtbar gemacht. Das Overlay gruppiert jetzt nach Textstelle (Seite + Quad-Geometrie) statt nach Segment-ID: Die Textstelle bekommt ein In-Text-Highlight, jede weitere Kodierung einen eigenen, klickbaren Margin-Stripe in ihrer Kategoriefarbe. Aus wenigen sichtbaren Flächen werden alle 7 Kodierungen unterscheidbar.
- **Panel-Kopf zählt die aktuelle Seite:** „Kodierungen auf S. 1 — 7 · 4 Textstellen · 7 Vorschläge · 0 bestätigt · 0 verworfen · 0 ohne Anker“. Hat die Seite weniger Kodierungen als das Dokument, steht die Gesamtzahl als Hinweis darunter.
- **Ganze Chunks markiert:** Die erste Ankerstufe reicht jetzt bis 1200 Zeichen (vorher 800). Chunks bis zu dieser Laenge werden vollstaendig markiert statt nur im Kopf - im Demo-Dokument 100 % jedes Chunks. Scheitert die lange Stufe, greifen unveraendert 800 / 500 / 240 Zeichen.
- **Zitatvorschau je Zeile:** Jede Zeile zeigt die ersten ~110 Zeichen der markierten Textstelle.
- **Hinweis auf geteilte Textstelle:** „↔ gleiche Textstelle wie 1 weitere Kodierung“.
- **Hinweis auf verteilte Markierungen:** „⇅ Markierung verteilt über n Abschnitte“, wenn der Chunk Text aus mehreren Spalten/Absätzen enthält (etwa Fließtext + Abstract) und die Markierung deshalb über die Seite springt.

#### Technische Hinweise

- `QCA_AID_assets/annotation/renderer.py`: Gruppierung `_group_by_region` mit `_region_key` (gerundete Quad-Geometrie); Margin-Stripes gestapelt je weiterer Kodierung.
- `QCA_AID_app/viewer/coding_panel.py`: neu `panel_header`, `panel_regions`, `_quote_preview`, `_spread`; `CodingRow` um `quote`, `same_region`, `blocks` erweitert.
- `QCA_AID_app/webapp_logic/review_service.py`: `QUOTE_STAGES = (1200, 800, 500, 240)`; `normalize_doc_name` als Raw-Docstring (Escape-Warnung).
- Tests: `test_overlay_renderer.py` (gleiche Textstelle wird nicht übermalt, aktive Kodierung gewinnt das Highlight), `test_coding_panel.py` (Seitenkopf, Textstellenzählung, Zitatvorschau, Shared-Region-Flags), `test_review_service.py` (Overlay-Map).
- Die Markierung enthält **keinen Chunk-Offset**: Chunk 0 beginnt mit dem Seitenkopf des Artikels (`32 NUMBER 109_Winter 2022 THE PUBLIC-PRIVATE DEBATE | …`) – das ist die gedruckte Kopfzeile in der linken Randspalte, nicht die Chunk-ID.

### 🧹 Sonstiges

## Neu in 0.15.0 (2026-11-23)

### Prüfen-Tab: Dokumentvorschau mit Annotation und Kodierprüfung

Neuer Arbeitsplatz „Prüfen" zwischen Analyse und Explorer:

- **Dokumentvorschau (Canonical PDF):** Eingabedokumente (PDF, DOCX, DOC, HTML, MD, TXT) werden einmalig in ein kanonisches PDF konvertiert, gecacht und mit Textindex abgelegt. Analyse-Vorschau und Prüfen-Tab nutzen dieselbe Geometrie.
- **Kodierungen laden und prüfen:** Kodierungen aus der Analyse-XLSX laden, an der Vorschau verankern (Textmarker, Unterstreichen, Randnotiz, Klammer), bestätigen oder verwerfen.
- **Umcodieren:** Pro Kodierung einen anderen Code vergeben – Auswahl aus Codebook-Kategorien und Subcodes (plus den bereits vergebenen Codes) oder Freitext im Format `Kategorie › Subcode`. Korrekturen werden als menschliche Entscheidung markiert (`created_by=human`, Status bestätigt).
- **Rückkanal im Viewer:** Klick auf eine markierte Stelle im PDF selektiert die Kodierung im Panel und merkt sich die Seite (eigener Streamlit-Component-Host; ohne Rückkanal läuft die Vorschau unverändert weiter).
- **Gestufte Ankerlaengen:** Chunks werden moeglichst weit markiert (bis zu 800 Zeichen, dann 500, dann der konservative 240er-Kopf). Scheitert eine Stufe im PDF, greift die naechste – längere Markierungen, ohne die Garantie „nie falsch platziert" anzutasten.
- **Export:** annotiertes PDF (Highlights mit /QuadPoints, Legende, Marginalien) und Kodierungen als XLSX.

#### Technische Hinweise

- **PyMuPDF ist jetzt Kernabhängigkeit** (Seitengeometrie, Textindex, PDF-Export, Text-Fallbacks für DOCX/MD/TXT). Weasyprint, docling, LibreOffice und Pandoc bleiben optional – fehlende Konverter werden in der Vorschau mit konkreter Installationshilfe gemeldet; Analyse und Export laufen weiter.
- Neue Module: `webapp_logic/preview_service.py`, `webapp_logic/review_service.py`, `viewer/code_options.py` sowie Component-Host `viewer/_component_host/`.
- Erneuerte `requirements.txt`/Extras: `pip install qca-aid[converters]` für optionale Konverter-Fallbacks.
- Konvertierungsfehler bleiben nicht dauerhaft: nach Erfolg wird der Eintrag im Fehlerregister (`documents/_not_convertible.json`) gelöscht.

### 🧹 Sonstiges

## Neu in 0.14.1 (2026-11-19)

### Fix: Intercoder-Reliabilität rechnet jetzt echtes Krippendorff's Alpha

**Problem:** `ReliabilityCalculator` lieferte Werte, die nur *wie* Alpha aussahen:

- `expected_agreement` war hartkodiert (`0.25` für Subkategorien/Overall, `0.20` für Hauptkategorien). Krippendorff's Alpha verlangt die erwartete Übereinstimmung aus den Randverteilungen der Koinzidenzmatrix. Die ausgegebenen Zahlen waren damit mit veröffentlichten Alpha-Koeffizienten nicht vergleichbar.
- `max(0.0, alpha)` kappte negative Werte. Negative Alpha sind aber ein valider Befund (systematische Nicht-Übereinstimmung) und wurden bisher als "0" verdeckt.
- Die Konsistenzprüfung verlangte, dass Overall-Alpha zwischen Haupt- und Sub-Alpha liegt, und meldete sonst "❌ Mathematische Inkonsistenz". Das ist keine mathematische Notwendigkeit – bei unterschiedlichen De-Werten der Ebenen ist eine Abweichung normal.
- Die beobachtete "Übereinstimmung" war ein ungewichteter Mittelwert paarweiser Jaccard-Koeffizienten, nicht die Koinzidenz von Alpha.

**Lösung:** `QCA_AID_assets/quality/reliability.py` neu implementiert (Koinzidenzmatrix-Verfahren nach Krippendorff 2004/2011; Hayes & Krippendorff 2007):

- **Kern:** `alpha = 1 - Do/De` aus einer Koinzidenzmatrix über alle Einheiten und Kodierungen. Fehlende Werte (nicht jedes Segment von jedem Kodierer kodiert), beliebig viele Kodierer, ungleiche Kodiererzahlen pro Einheit und Mehrfachkodierungen (`MULTIPLE_CODINGS`) sind damit ohne Sonderbehandlung zulässig. Nur Segmente mit Kodierungen von mindestens zwei *verschiedenen* Kodierern gehen ein.
- **Distanzmetriken:** Hauptkategorien nominal; Subkategorien Jaccard-Distanz auf den Kategorienmengen `d = (1 - |S1∩S2| / |S1∪S2|)^2`; Overall als Mittel beider Ebenen `d = (d_main + d_sub)/2` – Alpha über die vollständige Kodierentscheidung, ohne dass eine Ebene die andere dominiert.
- **Keine Kappung mehr:** Alpha bleibt im Wertebereich [-1, 1]; negative Werte werden berichtet und in der Bewertung als "systematische Nicht-Übereinstimmung" ausgewiesen.
- **Diagnose:** `alpha_details` enthält je Ebene Do, De, Anzahl Einheiten/Werte/Kategorien und ein `degenerate`-Flag (keine Varianz in den Randverteilungen → Alpha nicht interpretierbar). Neu außerdem `category_alphas`: binäres Alpha je Hauptkategorie, um Problemkategorien zu identifizieren.
- **Konsistenzprüfung korrigiert:** Statt des falschen "Overall muss zwischen den Ebenen liegen" gibt es jetzt Hinweise auf zu wenige vergleichbare Segmente (< 10), nicht interpretierbare Ebenen und – falls zutreffend – eine Erläuterung, warum Overall außerhalb der Ebenen liegen kann.
- **API stabil:** `calculate_comprehensive_reliability()` (Dict mit `overall_alpha`, `main_categories_alpha`, `subcategories_alpha`, `agreement_analysis`, `statistics`), `calculate_reliability()` und die bisherigen privaten Einzelmethoden bleiben verfügbar; `_calculate_detailed_agreement_analysis()` und `_group_by_original_segments()` bleiben unverändert nutzbar (Export).

**Robustheitsvariante (binäre Hamming-Distanz):** Für mengenwertige Kodierungen (mehrere Subkategorien pro Segment) gibt es keine kanonische Alpha-Variante – die Wahl der Mengenmetrik ist eine Modellentscheidung (vgl. MASI, Passonneau 2006, ebenfalls Jaccard-basiert). Damit sie nicht bloß behauptet werden muss, berechnet der Report zusätzlich eine Variante auf **binären Indikatorvektoren**:

- Distanz `d^2 = |S1 Δ S2| / K` über das Kategorienuniversum K – der Intervall-Fall aus Krippendorff (2004), angewandt auf Indikatorvektoren.
- Die Normierung auf K ist kosmetisch: Do und De skalieren gemeinsam, Alpha ist invariant gegen K (im Test abgesichert); sie hält Do/De nur in [0, 1].
- Neue Reportfelder `overall_alpha_binary` und `subcategories_alpha_binary` sowie `alpha_details['overall_binary']` / `['subcategories_binary']`.
- Konsole und Excel weisen die Differenz aus: `<= 0.05` gilt als „Metrikwahl unkritisch", darüber als Begründungsbedarf im Methodenteil.

**Methodische Hinweise im Excel-Export:** Das Blatt `IntercoderBericht` enthält jetzt einen Abschnitt „2. Methodik, Formeln und Interpretation" mit Formel, Do/De je Ebene (inkl. binärer Variante), Wertebereich, Schwellen, Distanzmetriken, Behandlung fehlender Werte und Mehrfachkodierungen, Stichprobenumfang, Kategoriendiagnostik, der Grenze der Set-Kodierung (mit Verweis auf `irr::kripp.alpha` / `irrCAC` / `krippendorff`) und den Referenzen. Das Blatt `Intercoder_Übersicht` enthält eine kompakte Fassung (Formel, Do/De, Schwellen, Metrik-Robustheit, Referenz). Damit ist der Bericht ohne Rückgriff auf die Dokumentation lesbar und zitierfähig.

**Validierung:** Die Berechnung stimmt auf 1e-9 mit der Referenzimplementierung des Pakets `krippendorff` (0.9.0) überein (Beispieldatensatz aus Krippendorff 2011: α = 0.765957…, 4 Kodierer, 12 Einheiten, fehlende Werte). Neue Tests: `QCA_AID_assets/quality/tests/test_reliability_alpha.py` (23 Tests: Referenzwerte, Handrechnung, negative Alpha, Randverteilungs-Abhängigkeit, fehlende Werte, Mehrfachkodierung, Set-Distanzen, Hamming-Variante inkl. K-Invarianz, Berichtsstruktur, Kategorie-Diagnostik).

**Dokumentation:** `README.md` und `QCA-AID-Nutzerhandbuch.md` sprachen von "Cohens Kappa" und κ-Schwellen, berechnet wurde aber Alpha – jetzt einheitlich Krippendorff's Alpha mit den Schwellen .800/.667 (Krippendorff 2004, S. 241f.). Im Excel-Export (`IntercoderBericht`) sind die Zeilen entsprechend umbenannt; die deskriptive Set-Tabelle ist als solche gekennzeichnet. Ein toter Kommentarblock ("KORRIGIERTE KRIPPENDORFF'S ALPHA BERECHNUNG") am Ende von `review_manager.py` wurde entfernt.

### 🧹 Sonstiges

- **Zirkulärer Import behoben:** `analysis_manager` importierte `ResultsExporter` auf Modulebene, `results_exporter` importierte `InductiveCoder` aus `analysis`. Folge: `import QCA_AID_assets.export.results_exporter` schlug mit `ImportError: partially initialized module` fehl, solange nicht vorher ein analysis-Modul geladen war (im App-Flow unsichtbar, für Skripte und Einzeltests fatal). Der Export-Import in `analysis_manager` ist jetzt lokal an der einzigen Verwendungsstelle, die `InductiveCoder`-Annotation in `results_exporter` läuft über `TYPE_CHECKING`. Regressionsschutz: `QCA_AID_app/tests/test_module_import_order.py` (frischer Interpreter je Modul).
- Version auf 0.14.1 gehoben (Patch: Reliabilitätsberechnung, Doku-Korrektur).

## Neu in 0.14.0 (2026-11-19)

### Fix: TypeError in calculate_multiple_coding_stats bei None-Kategorien

**Problem:** Ein induktiver Lauf (v0.13.6) beendete sich *nach* erfolgreichem Export mit

```
Fehler in der Hauptausführung: '<' not supported between instances of 'NoneType' and 'str'
  File "...utils/analysis.py", line 63, in calculate_multiple_coding_stats
    category_combinations.append(' + '.join(sorted(unique_categories)))
TypeError: '<' not supported between instances of 'NoneType' and 'str'
```

Ursache: `ReviewManager._create_category_specific_segments()` setzt für Segmente ohne gültige Kategorie (alle Kodierungen `None` oder `"Nicht kodiert"`) `target_category = None`; der Consensus-Pfad schrieb dieses `None` in `_get_consensus_for_category_segment()` ungeprüft als `category` in die finalen Kodierungen. Die Statistik-Funktion sortierte anschließend eine Menge aus `str` und `None`. Da der Fehler im Block „finale Statistiken" auftrat, liefen Export und Review bereits durch – nur Token-/Relevanzstatistiken und der Abschluss gingen verloren, und der Lauf meldete fälschlich einen Fehler.

**Lösung (drei Ebenen):**

1. *Ursache:* `ReviewManager._get_consensus_for_category_segment()` fällt bei fehlendem `target_category` auf die Kategorie der besten Kodierung bzw. `"Nicht kodiert"` zurück. `_log_review_decision()` verträgt jetzt zusätzlich `justification = None` (vorher `TypeError: object of type 'NoneType' has no len()`).
2. *Speicherung:* `ExtendedCodingResult.__post_init__()` normalisiert `category` (`None`/leer → `"Nicht kodiert"`, sonst `strip()`), damit auch die Reliability-Database-Werte für spätere Läufe sauber sind.
3. *Statistik:* `calculate_multiple_coding_stats()` normalisiert `segment_id`, `coder_id` und `category` über einen Helper (`None` → `""`), sortiert nur noch Strings, zeigt kategorielose Kodierungen als `(ohne Kategorie)` und liest `total_coding_instances` robust als Integer. `main.py` prüft `coder_id` vor `.startswith("auto")` gegen `None`.
4. *Anzeige-Block:* Die Statistikausgabe in `main()` läuft jetzt in der neuen Funktion `_report_multiple_coding_stats()` mit eigenem `try/except`. Ein Fehler dort wird als `WARNING` inkl. Traceback geloggt (landet über den Tee-Writer im Console-Log) und auf der Konsole als `[WARNUNG]`/`[HINWEIS]` gemeldet – der Lauf endet nicht mehr mit „Fehler in der Hauptausführung“, da Export und Review zu diesem Zeitpunkt abgeschlossen und gültig sind.

**Tests:** neu `QCA_AID_assets/utils/tests/test_analysis_multiple_coding_stats.py` (None-Kategorie, ausschließlich None-Kategorien, None-Coder-ID, String-Instanzzähler, None-Segment-ID, Fokus-Adhärenz, Leerfälle), `QCA_AID_assets/quality/tests/test_review_manager_category_fallback.py` (Fallback bei `target_category = None`, Erhalt gültiger Kategorien) und `QCA_AID_app/tests/test_multiple_coding_stats_reporting.py` (Fehler wird geloggt, aber nicht geworfen; Erfolgsfall warnt nicht; echte None-Kategorien lösen keine Warnung mehr aus). `python -m pytest QCA_AID_app/tests QCA_AID_assets/utils/tests QCA_AID_assets/quality/tests` → alle grün.

### Neu: temperature-Unterstützung wird genau einmal geprüft (vor der Relevanzprüfung)

**Problem:** Ob das Modell den Parameter `temperature` akzeptiert, wurde bisher *pro API-Call* neu „ausprobiert“: Solange `token_counter.model_capabilities` keinen Eintrag hatte, schickte jeder Request `temperature` mit, scheiterte bei Reasoning-Modellen (o1/o3, GPT-5-Familie o. ä.) mit `400 Unsupported parameter: 'temperature'` und wurde ohne `temperature` wiederholt – doppelte Latenz, doppelte Fehler-Logs, und die Entscheidung fiel zufällig beim ersten echten Fach-Call (Relevanzprüfung).

**Lösung:** Neues Modul `QCA_AID_assets/utils/llm/capabilities.py` mit `ensure_temperature_capability()`:

- `IntegratedAnalysisManager.analyze_material()` ruft die Prüfung **einmalig vor der Relevanzprüfung** auf (also vor dem ersten fachlichen LLM-Call, für alle Modi inkl. `grounded`) – `_ensure_temperature_capability()`.
- Der Test ist ein einzelner, minimaler Completion-Aufruf (`"ping"`, `temperature=0.0`); das Ergebnis (True/False) landet im globalen Capability-Cache `token_counter.model_capabilities`.
- Ab dann laufen **alle** Requests deterministisch mit oder ohne `temperature` – die Provider (OpenAI, Anthropic, Mistral, OpenRouter, Local) lesen nur noch den Cache und probieren nicht erneut.
- Ein `_PROBED_MODELS`-Marker verhindert eine zweite Prüfung im selben Prozess, auch wenn die Prüfung ohne eindeutiges Ergebnis endet (dann Fallback „mit temperature“, geloggt als WARNING).
- Konsolen-/Log-Ausgabe: `🌡️ Prüfe EINMALIG, ob '<model>' den Parameter 'temperature' akzeptiert (vor der Relevanzprüfung)...` plus `✅ Modell akzeptiert 'temperature' …` bzw. `⚠️ Modell akzeptiert 'temperature' NICHT - alle weiteren Calls laufen ohne temperature.` Bei bereits geprüftem Modell (z. B. zweiter Lauf in derselben Session) nur eine kurze Zeile ohne API-Call.

**Tests:** neu `QCA_AID_app/tests/test_temperature_capability_probe.py` – Prüfung läuft genau einmal (zweiter Aufruf ohne API-Call), Ablehnung wird erkannt und gecacht, Cache verhindert Prüfung komplett, unklarer Fehler → Fallback „mit temperature“ (einmalig), fehlender Provider → `None` mit Warnung, `force=True` erlaubt erneute Prüfung; zusätzlich ein Test gegen den **echten** `OpenAIProvider`-Codepfad (Probe-Call mit `temperature`, Retry ohne, danach gehen alle Calls ohne `temperature` raus).

### Neu: Custom API Base URL nur mit Häkchen (OpenAI / Custom API Provider)

**Problem:** In der Konfigurations-UI war die Sektion „Custom API Base URL“ für alle Provider sichtbar und jede Eingabe wurde sofort in die Konfiguration geschrieben (`API_BASE_URL`/`API_KEY_ENV`). Ein einmal eingetragener Custom-Endpoint blieb damit auch dann aktiv, wenn der Nutzer den Standard-Cloud-Provider von OpenAI verwenden wollte – inklusive Freitext-Modellfeld und Vorrangprüfung gegen den Custom-Env-Var.

**Lösung:** Die Custom-API-Sektion ist jetzt an Provider und Häkchen gebunden (`webapp_components/config_ui.py`):

- Bei **„OpenAI / Custom API Provider“** gibt es das Häkchen **„Custom API Base URL verwenden“** (Default: angehakt, wenn bereits eine URL gespeichert ist).
  - **Häkchen gesetzt:** Base URL + Env-Var-Name werden übernommen, Modell-Name bleibt Freitext.
  - **Häkchen nicht gesetzt:** Es gilt der Standard-Cloud-Provider von OpenAI; `api_base_url` und `custom_api_key_env` werden aus der Konfiguration **entfernt** (Meldung + Log `[config] Custom API-Einstellungen zurückgesetzt …`), das Modell-Feld fällt auf das Standard-Dropdown zurück.
- Bei **„Local (LM Studio/Ollama)“** bleibt die URL ein normales Eingabefeld (der Endpoint ist dort funktional erforderlich, kein Häkchen).
- Bei **Anthropic/Mistral/OpenRouter** wird die Sektion ausgeblendet – die Provider-Factory reicht `base_url` ohnehin nur an OpenAI- und Local-Provider weiter – mit Hinweis-Caption.
- Die API-Key-Prüfung läuft jetzt **nach** der Custom-API-Sektion, damit ein gerade zurückgesetzter Endpoint sofort korrekt geprüft wird.

Neue, Streamlit-freie Helfer in `config_ui.py`: `provider_supports_custom_endpoint()`, `normalize_api_base_url()`, `is_valid_api_base_url()`, `reset_custom_api_settings()`.

**Tests:** neu `QCA_AID_app/tests/test_config_custom_api_toggle.py` – Provider-Gating (nur `openai`/`gpt`/`local`), URL-Normalisierung/Validierung, Reset entfernt URL und Env-Var und persistiert `None` (Standard-Cloud), Idempotenz des Resets, Kompatibilität mit alten `SimpleNamespace`-Session-Objekten. Zusätzlich `QCA_AID_app/tests/test_config_ui_custom_api_streamlit.py` (headless über `streamlit.testing.v1.AppTest`): Häkchen ist bei gespeicherter Custom-URL angehakt, Entfernen setzt URL + Env-Var zurück und markiert die Config als geändert, ohne Custom-URL ist es nicht angehakt, und bei Anthropic/Mistral/OpenRouter erscheint die Sektion gar nicht (gespeicherte Werte bleiben unangetastet). Wird das URL-Feld geleert, verschwindet auch die Custom-Env-Variable.

### 🧹 Sonstiges

- Version auf 0.14.0 gehoben (Minor: zwei neue Funktionen – temperature-Capability-Prüfung und Custom-API-Häkchen – plus Bugfixes bei den Mehrfachkodierungs-Statistiken).

## Neu in 0.13.6 (2026-11-18)

### Bugfix: NameError batch_callback im induktiven Modus

**Problem:** `OptimizationController._analyze_inductive()` nutzte `batch_callback` (Phasen-Label „Phase 1: Relevanzprüfung"), ohne den Namen als Parameter zu deklarieren. Er kam per `**kwargs` aus `analyze_segments()` herein, wurde aber nie an einen lokalen Namen gebunden – jeder induktive Lauf (`_analyze_normal_modes()` → `analyze_segments()`) brach daher sofort mit `NameError: name 'batch_callback' is not defined` ab und fiel still auf die Standard-Analyse zurück (inkl. Verlust des Fortschritts: „1062 Segmente aus processed_segments entfernt"). `_analyze_grounded()` hatte den Parameter bereits korrekt deklariert.

**Lösung:** `batch_callback: Optional[callable] = None` in der Signatur von `_analyze_inductive()` ergänzt (`QCA_AID_assets/optimization/controller.py`), analog zu `_analyze_grounded()`.

### Gleichartige latente NameErrors (per pyflakes gefunden und mitbehoben)

- `analysis/analysis_manager.py`: `import statistics` fehlte – `get_progress_report()` wäre beim ersten Batch-Zeitmittel abgestürzt.
- `analysis/inductive_coding.py`: `import traceback` fehlte – im `except`-Zweig der Grounded-Analyse hätte der fehlende Import den Originalfehler maskiert.
- `export/results_exporter.py`: `import re` und `from ..core.validators import CategoryValidator` fehlten – die Subkategorie-Validierung beim Export und die Segment-ID-Helfer (`_extract_segment_number`, `_extract_first_letters`) wären mit NameError abgebrochen.
- `utils/io/document_reader.py`: `except PyPDF2.errors.PdfReadError` → `except pypdf.errors.PdfReadError` (die Abhängigkeit ist `pypdf`; bei PDF-Lesefehlern drohte NameError statt des vorgesehenen OCR-Fallbacks).

**Tests:** neu `QCA_AID_app/tests/test_optimization_controller_callbacks.py` (Signatur-Check für `_analyze_inductive`/`_analyze_grounded` plus Smoke-Test der Relevanzphase mit und ohne Callback). Gesamtsuite: `python -m pytest QCA_AID_app/tests` → 106 passed. `pyflakes QCA_AID_assets` meldet keine echten undefined names mehr (verbleibende Treffer sind reine String-Annotationen bzw. eine Test-Datei).

### 🧹 Sonstiges

- Version auf 0.13.6 gehoben (`scripts/version_guardrail.py bump`).

## Neu in 0.13.5 (2026-11-18)

### Induktiver Modus: nur Forschungsfrage verpflichtend (mode-aware Readiness/Validierung)

### 🐛 Workflow verlangte im induktiven Modus Kategorien, die die Engine ignoriert

**Problem:** Die Analyse-Engine behandelt den induktiven Modus seit jeher wie einen emergenten Modus: `_analyze_normal_modes()` setzt `current_categories = {}` und `deductive_categories = {}`, das Kategoriensystem entsteht batchweise aus dem Material – Orientierung ist ausschließlich die Forschungsfrage (`InductiveCoder._create_inductive_mode_prompt`). Der Workflow (Readiness-Layer, Codebook-Validierung, Codebook-Tab) kannte diese Ausnahme aber nur für `grounded`. Ergebnis: Ein methodisch korrekt aufgesetztes induktives Projekt (nur Forschungsfrage) zeigte im Analyse-Tab den Blocker „Codebook enthält keine Kategorien“, der Codebook-Schritt blieb unvollständig und `CodebookData.validate()` meldete „At least one category is required“.

**Lösung:** `grounded` und `inductive` werden jetzt einheitlich als *emergente Modi* behandelt:

- `webapp_logic/readiness.py`: neue Konstante `EMERGENT_MODES` + `requires_predefined_categories()`; `codebook_ready()` und `_codebook_issues()` verlangen in beiden Modi nur die Forschungsfrage. Die Meldung ist modusabhängig („im induktiven Modus OK – das Kategoriensystem entsteht aus dem Material“) und bleibt eine Warnung, kein Blocker.
- `webapp_models/codebook_data.py` (`CodebookData.validate`): Early-Return für `{"grounded", "inductive"}`; Forschungsfrage bleibt in **allen** Modi Pflicht.
- `webapp_components/codebook_ui.py`: Status-Banner und Validierungs-Hinweis unterscheiden jetzt Grounded Mode und induktiven Modus (inkl. Hinweis, dass vorhandene Kategorien im induktiven Modus nicht als Startpunkt verwendet werden).

**Nicht geändert:** `deductive`, `abductive` und `full` verlangen weiterhin vorab definierte Kategorien – abduktiv erweitert bestehende Hauptkategorien, deduktiv kodiert ausschließlich dagegen.

**Tests:** `tests/test_webapp_readiness.py` (induktiv = Warning statt Blocker, Workflow-Step vollständig, fehlende Forschungsfrage bleibt Blocker, abduktiv weiterhin Kategorie-Pflicht) und neu `tests/test_webapp_codebook_validation.py` (mode-aware `CodebookData.validate`). Gesamtsuite: 103 passed.

### 🧹 Sonstiges

## Neu in 0.13.3 (2026-11-12)

### Bugfix-Release: Projekt-Codebook hat Vorrang, Material-Icons, LLM-Katalog-Update

### 🐛 Projektverzeichnis-Config hat immer Vorrang

**Problem:** Der Analyse-Subprozess löste den Codebook-Pfad über den Zeiger `last_codebook_file` aus der `.qca-aid-project.json` auf — bevor er die Standard-Datei im Projektverzeichnis prüfte. Stand in diesem Zeiger noch eine Codebook-Kopie aus einer alten Session (z. B. mit `model_provider: local`), lief die Analyse plötzlich gegen den lokalen LM-Studio-Server, obwohl in der Webapp und im Projekt-Codebook OpenRouter eingestellt war. Der Lauf brach dann still mit `400 - No models loaded` ab, während das stille Fallback-Verhalten den Eindruck eines erfolgreichen Laufs erweckte (Alpha = 1.000 bei 0 Kodierungen).

**Lösung:** Neue Prioritätsreihenfolge bei der Codebook-Auflösung:
1. Env-Var `QCA_AID_CODEBOOK_PATH` (explizite Benutzer-Auswahl in der laufenden Session)
2. **Standard-Datei im Projektverzeichnis** — gewinnt immer über den gespeicherten Zeiger
3. Fallback `last_codebook_file` — nur noch, wenn im Projektverzeichnis kein Codebook liegt (mit deutlicher Log-Warnung)

Umgesetzt an beiden Stellen: `QCA_AID_assets/utils/config/loader.py` (`_resolve_last_codebook_path`) und `QCA_AID_app/webapp_logic/project_manager.py` (`get_config_path`/`get_codebook_path`). Zusätzlich wurde der `.qca-aid-project.json`-Zeiger des Test-Projekts auf das korrekte Codebook zurückgesetzt.

### 🧹 Sonstiges

- LLM-Modellkataloge aktualisiert: OpenRouter, OpenAI und Anthropic (`QCA_AID_assets/utils/llm/configs/*.json`), u. a. mit Z.ai GLM-5.3-Modellfamilie.
- Minimale Streamlit-Version auf >=1.40 angehoben (`pyproject.toml`, `requirements.txt`) — Voraussetzung fuer den `:material/...:`-Icon-Shorthand.
- Fix: `:material/...:`-Shorthand wird in rohen HTML-Bloecken (`unsafe_allow_html`) nicht geparst und erschien als Rohtext. Betroffen waren die Status-Banner der Explorer-Analysen (jetzt natives `st.success`/`st.info`) und der App-Header-Fallback ohne PNG-Icon (jetzt natives Markdown). Alle anderen Kontexte (Alerts, Button-/Expander-/Tab-/Widget-Labels, Spinner, Toast, Dialog-Titel) rendern den Shorthand korrekt — empirisch verifiziert via Playwright-Screenshots unter Streamlit 1.51.

## Neu in 0.13.2 (2026-11-12)

### 🧭 Workflow-Navigation — klickbare Schritte und Weiter-Buttons

**Problem:** Die vier Arbeitsschritte (Konfiguration → Codebook → Analyse → Explorer) waren nur über gleichwertige Reiter erreichbar. Es gab keinen Hinweis darauf, welcher Schritt als nächstes dran ist, und Reiter ohne Eingaben sahen identisch zu fortgeschrittenen aus.

**Lösung:** Die Sidebar zeigt den Workflow jetzt als klickbare Schritte mit Status (erledigt ✅ / aktuell / offen). Am Ende jedes Tabs gibt es einen „Weiter zu …“-Button zum nächsten Schritt. Nach jedem Tab-Wechsel beginnt die Seite oben — bisher sprang Streamlit auf die alte Scroll-Position zurück und neue Tabs starteten in der Mitte.

### 📊 Analyse → Explorer in einem Klick

Nach abgeschlossener Analyse öffnet der Button „📊 Ergebnisse im Explorer öffnen“ den Explorer-Tab mit der neuesten Ergebnisdatei bereits vorausgewählt — kein erneutes Suchen der Datei.

### 💾 Ungespeicherte Änderungen sichtbar machen

**Problem:** Änderungen an Konfiguration, Codebook und Explorer-Einstellungen gingen verloren, wenn man den Tab wechselte, ohne zu wissen, dass noch etwas ungespeichert war.

**Lösung:** Bei ungespeicherten Änderungen erscheint eine Leiste mit „💾 Speichern“ und „🗑 Verwerfen“. Die betroffenen Tabs tragen ein ⚠️-Badge, bis gespeichert oder verworfen wurde. „Verwerfen“ fragt vorher per Dialog nach; Schlägt das Speichern fehl (z. B. Datei nicht mehr vorhanden), bleibt der Zustand erhalten und der Fehler wird angezeigt.

### 🚀 Explorer-Kompaktmodus mit Start-Bestätigung

Die Explorer-Analyse-Seite startet in einem kompakten Standardbereich: Datei-Auswahl, Zusammenfassung der wichtigsten Einstellungen und Start-Button. Vor dem (kostenpflichtigen) Start erscheint ein Bestätigungsdialog mit der vollständigen Zusammenfassung. Alle erweiterten Einstellungen (Modell, Kategorien, Optionen) sind weiterhin über Expander erreichbar.

### 🔗 Probleme direkt beheben

Die Readiness-Prüfung meldet Blocker pro Tab — jeder Blocker hat jetzt einen „→ Beheben“-Button, der genau zur betroffenen Einstellung springt und dort per Info-Banner markiert, was zu tun ist. Die Konfiguration ist außerdem einspaltig gruppiert und Abschnitte mit Validierungsfehlern öffnen sich automatisch.

#### Behobene Probleme

- **Navigation:** Klick auf Workflow-Schritte oder Weiter-Buttons landete fälschlich im Konfiguration-Tab statt im Ziel-Tab.
- **Scroll-Position:** Nach einem Tab-Wechsel begann die Seite in der Mitte statt oben.
- **Entfernen ohne Rückfrage:** Kategorien im Codebook und Analysen im Explorer wurden sofort gelöscht; jetzt bestätigt ein Dialog vorher.

### 🧹 Sonstiges

- Paginierung der Dateilisten als eigene Komponente (`st.pagination` existiert in Streamlit 1.51 noch nicht).
- System-Emojis in der Webapp-UI durch Material-Symbols-Icons ersetzt (`:material/...:`-Shorthand, von Streamlit lokal gebuendelt -- offline, themefest, einheitliches Rendering). CLI- und Log-Ausgaben behalten bewusst Emojis. Neue Icon-Registry: `webapp_components/icons.py`; neuer Lint-Test `test_no_emoji_in_webapp.py` verhindert Rueckfall.
- Statusabhaengige Icons heben sich farblich von statischen ab: erledigt gruen (:green[:material/check_circle:]), Warnung orange, Blocker rot, Hinweis blau (Streamlit-Farb-Markdown, Dark-Mode-sicher); offen/neutral bleibt ungefaerbt. Tab-Badges, Sidebar-Workflow und Codebook-Import-Vorschau sind betroffen. Versteckte \u-Escape-Emojis (z. B. 🧩 in der Erweiterte-Einstellungen-Sektion) aufgedoest; Lint-Test prueft jetzt auch \u-Escapes.
- Neues Schutznetz gegen vergessene Releases: Ein Guardrail prüft bei jedem Testlauf, dass `pyproject.toml`, `__version__.py` und `CHANGELOG.md` konsistent sind. `python scripts/version_guardrail.py bump <version> --summary "..."` führt Versionsprünge synchron über alle Stellen aus.

## Neu in 0.13.1 (2026-07-24)

### 📊 Fortschrittsanzeige für Grounded Mode komplett überarbeitet

**Problem:** Die Fortschrittsanzeige im Grounded Mode war seit mehreren Versionen kaputt. Bei einer Analyse mit 423 Segmenten zeigte das Log über 5½ Stunden hinweg ununterbrochen `Verarbeitet: 423 Segmente` — ohne dass sich die Zahl jemals änderte. Die Geschwindigkeitsanzeige produzierte absurde Werte (z.B. 304.073.482 Segmente/Stunde in den ersten Sekunden). Es gab keine Information darüber, in welcher Phase sich die Analyse befand, wie viele Batches bereits verarbeitet waren oder welche Kodierungen bereits vorlagen. Beim Drücken von ESC stand stets `Erstellte Kodierungen: 0`, obwohl bereits hunderte API-Calls gelaufen waren.

**Ursache:** In `_analyze_grounded_mode` (`analysis_manager.py`) wurden alle 423 Segmente sofort zu `processed_segments` hinzugefügt, bevor die eigentliche Analyse begann. Die Fortschrittsüberwachung (`monitor_progress` in `main.py`) las dann einfach `len(processed_segments)` und zeigte immer 423 — unabhängig vom tatsächlichen Verarbeitungsstand.

#### Behobene Probleme

| # | Problem | Fix |
|---|---------|-----|
| 1 | `processed_segments` wurde vorab mit allen Segmenten gefüllt → Fortschritt immer 423/423 | Vorab-Befüllung entfernt; Segmente werden erst nach Abschluss von Phase 1 hinzugefügt |
| 2 | Keine Phasen-Information (`"Optimierte Analyse läuft..."` über Stunden) | Phasen-Tracking mit `current_phase`, `current_phase_batch`, `current_phase_total_batches`, `current_coder_id` in allen Phasen (1, 1.5, 2, 3) |
| 3 | Kein Batch-Zähler in Phase 3 (Kodierung) — nur sporadische "Keine Kontext-Paraphrasen" Meldungen | `analyze_batch` loggt jetzt `📦 Kodier-Batch N/M: 8 Segmente (auto_1)` für jeden Batch |
| 4 | Geschwindigkeitsberechnung kaputt (304M Segmente/Stunde am Anfang) | Berechnung nur wenn `elapsed_time > 5s` und `processed_segments > 0` |
| 5 | `Erstellte Kodierungen: 0` bei jedem ESC-Dialog | `result_callback` speichert Kodierungen progressiv in `self.coding_results` nach jedem Coder-Durchlauf |

#### Neue Ausgabe der Fortschrittsanzeige

**Vorher:**
```
--- Analysefortschritt ---
Verarbeitet: 423 Segmente
Geschwindigkeit: 172.6 Segmente/Stunde
Status: Optimierte Analyse läuft...
------------------------
```

**Nachher:**
```
--- Analysefortschritt ---
Phase: Phase 3: Kodierung
  Batch 7/53 (auto_1)
Verarbeitet: 56/423 Segmente (13.2%)
Geschwindigkeit: 345.2 Segmente/Stunde
Status: Phase 3: Kodierung
Kodierungen: 48
------------------------
```

#### Geänderte Dateien

- **`QCA_AID_assets/analysis/analysis_manager.py`**:
  - Neue Instanzvariablen `current_phase`, `current_phase_batch`, `current_phase_total_batches`, `current_coder_id`
  - `processed_segments` wird nicht mehr vorab gefüllt
  - `get_progress_report()` erweitert: Phase, Batch-Info, Coder, `total_segments`, `progress_percent`, `phase_batch_info`
  - Geschwindigkeitsberechnung mit `elapsed_time > 5` Guard
  - Phasen-Tracking in `_analyze_grounded_mode` (optimiert) und `_analyze_grounded_mode_standard` (Fallback)
  - `result_callback` für progressive Kodierungs-Speicherung in Phase 3
  - `progress_callback` für Batch-Updates an `code_with_grounded_categories`

- **`QCA_AID_assets/main.py`**:
  - `monitor_progress()` komplett überarbeitet: Zeigt Phase, Batch-Info, Coder, Fortschritt mit Prozent, Geschwindigkeit, Status und Kodierungen

- **`QCA_AID_assets/optimization/controller.py`**:
  - `code_with_grounded_categories()`: Neuer Parameter `progress_callback` und `result_callback`
  - `analyze_segments()`: `batch_callback` wird aus `**kwargs` extrahiert und an `_analyze_grounded` weitergegeben
  - `_analyze_grounded()`: Neuer Parameter `batch_callback`; Callback-Aufruf nach Batch-Verarbeitung
  - Multi-Coder und Fallback-Pfad: `progress_callback` und `coder_id` werden an `analyze_batch` übergeben
  - `result_callback` wird nach jedem Coder-Durchlauf aufgerufen

- **`QCA_AID_assets/optimization/unified_relevance_analyzer.py`**:
  - `analyze_batch()`: Neue Parameter `progress_callback` und `coder_id`
  - Batch-Logging: `📦 Kodier-Batch N/M: X Segmente (coder_id)` für jeden Batch
  - `progress_callback` wird zu Beginn und pro Batch aufgerufen

### 🧹 Sonstiges

- QCA-AID Version auf 0.13.1 gehoben (Datum 2026-07-24).
- CHANGELOG.md aktualisiert.

---

## Neu in 0.13.0 (2026-07-09)

### 🖥️ CLI-Modus — QCA-AID als Kommandozeilen-Tool

QCA-AID kann ab sofort vollständig über die Kommandozeile bedient werden — ohne Webapp-Browser. Drei neue globale Befehle stehen nach Installation zur Verfügung:

| Befehl | Zweck |
|--------|-------|
| `qcaaid` | Hauptanalyse (Kodierung) — äquivalent zum Analyse-Tab der Webapp |
| `qcaaid-explorer` | Explorer (Visualisierung) — äquivalent zum Explorer-Tab |
| `qcaaid-webapp` | Startet die Streamlit-Webapp |

#### Installation

```bash
# Im QCA-AID-Ordner einmalig ausführen:
pip install -e . --no-build-isolation
```

Danach sind `qcaaid`, `qcaaid-explorer` und `qcaaid-webapp` von **jedem Verzeichnis** aus als Terminal-Befehl verfügbar.

#### Neue CLI-Optionen

**`qcaaid` (Hauptanalyse):**

| Option | Beschreibung |
|--------|-------------|
| `--version` | Version ausgeben |
| `--project-root PATH` | Projekt-Verzeichnis festlegen |
| `--mode MODE` | Analysemodus: `deductive`, `abductive`, `inductive`, `grounded` |
| `--config PATH` | Pfad zu einer spezifischen Codebook-JSON-Datei |
| `-n`, `--non-interactive` | Keine interaktiven Eingaben — für Skripte/Pipelines |
| `--no-manual` | Manuelles Kodieren deaktivieren |
| `--no-pdf` | PDF-Annotation deaktivieren |
| `--use-saved-codebook` | Gespeichertes induktives Codebook laden |

**`qcaaid-explorer` (Explorer):**

| Option | Beschreibung |
|--------|-------------|
| `--version` | Version ausgeben |
| `--config PATH` | Pfad zu einer Explorer-Config-JSON-Datei |
| `-n`, `--non-interactive` | Keine interaktiven Eingaben |
| `--output-dir PATH` | Output-Verzeichnis festlegen |

#### Implementierung

- **`pyproject.toml`**: Neues Python-Paket-Setup mit `[project.scripts]` für Console-Scripts
- **`QCA_AID_assets/cli.py`**: CLI-Einstiegspunkt für `qcaaid` (argparse + asyncio)
- **`QCA_AID_assets/explorer_cli.py`**: CLI-Einstiegspunkt für `qcaaid-explorer`
- **`QCA_AID_assets/webapp_cli.py`**: CLI-Einstiegspunkt für `qcaaid-webapp`
- **`QCA-AID.py`** und **`QCA-AID-Explorer.py`**: Um argparse erweitert, CLI-Args werden an `main()` bzw. `explorer.main()` durchgereicht
- **`main.py`**: `main()` akzeptiert `cli_args` Dictionary; `--non-interactive` überspringt alle interaktiven Eingaben (analog zum Webapp-Modus); `--mode` überschreibt den Analysemodus aus dem Codebook
- **`explorer.py`**: `main()` akzeptiert `cli_args` Dictionary; `--config` lädt eine spezifische Explorer-Config; `--non-interactive` überspringt interaktive Synchronisations-Abfragen

#### Bugfixes

- **Explorer-Config**: Analyse-Name `"Heatmaü"` → `"Heatmap"` korrigiert
- **Explorer-Config**: Analyse-Name `"zusamm"` → `"Zusammenfassung_Paraphrasen"` korrigiert

#### Dokumentation

- **`QCA-AID-Nutzerhandbuch.md`**: Neuer Abschnitt §5.4 "QCA-AID per Kommandozeile (CLI) nutzen" mit Installationsanleitung, alle Optionen, Beispiele und typische CLI-Workflows
- **`README.md`**: Schnellstart-Abschnitt um CLI-Hinweis mit Beispielen ergänzt

### 🧹 Sonstiges

- QCA-AID Version auf 0.13.0 gehoben (Datum 2026-07-09).
- `pyproject.toml` erstellt (ersetzt `setup.py`/`setup.cfg` als moderner Python-Packaging-Standard).
- CHANGELOG.md aktualisiert.

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

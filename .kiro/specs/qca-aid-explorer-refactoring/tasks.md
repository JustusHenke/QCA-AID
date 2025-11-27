# Implementation Plan

- [x] 1. Vorbereitung und Backup





  - Erstelle Backup der aktuellen QCA-AID-Explorer.py
  - Erstelle neue Verzeichnisstruktur für Module
  - Führe bestehende QCA-AID Tests durch (Baseline)
  - _Requirements: 1.1, 1.3_

- [x] 2. Erstelle neue Module und __init__.py Dateien





  - Erstelle QCA_AID_assets/analysis/__init__.py
  - Erstelle QCA_AID_assets/utils/visualization/__init__.py
  - Erstelle QCA_AID_assets/utils/prompts.py
  - Aktualisiere bestehende __init__.py Dateien für neue Exports
  - _Requirements: 2.5_

- [x] 3. Verschiebe LLMResponse Klasse





  - Verschiebe LLMResponse aus QCA-AID-Explorer.py nach QCA_AID_assets/utils/llm/response.py
  - Füge Docstring hinzu
  - Aktualisiere __init__.py für Export
  - _Requirements: 2.3, 7.2_

- [x] 3.1 Schreibe Property-Test für LLMResponse Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 2.3**

- [x] 4. Verschiebe ConfigLoader Klasse





  - Verschiebe ConfigLoader aus QCA-AID-Explorer.py nach QCA_AID_assets/utils/config/loader.py
  - Füge Docstring hinzu
  - Aktualisiere __init__.py für Export
  - _Requirements: 2.1, 7.2_

- [x] 4.1 Schreibe Property-Test für ConfigLoader Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 2.1**

- [x] 5. Verschiebe QCAAnalyzer Klasse





  - Erstelle QCA_AID_assets/analysis/qca_analyzer.py
  - Verschiebe QCAAnalyzer Klasse mit allen Methoden
  - Füge Docstrings für alle öffentlichen Methoden hinzu
  - Aktualisiere Imports innerhalb der Klasse
  - Aktualisiere __init__.py für Export
  - _Requirements: 2.2, 7.2_

- [x] 5.1 Schreibe Property-Test für QCAAnalyzer Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 2.2**

- [x] 6. Verschiebe Visualisierungsfunktionen





  - Erstelle QCA_AID_assets/utils/visualization/layout.py
  - Verschiebe create_forceatlas_like_layout Funktion
  - Füge Docstring hinzu
  - Aktualisiere __init__.py für Export
  - _Requirements: 3.3, 7.2_

- [x] 6.1 Schreibe Property-Test für Visualisierungsfunktionen Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 3.3**

- [x] 7. Verschiebe Hilfsfunktionen





  - Erstelle QCA_AID_assets/utils/prompts.py mit get_default_prompts
  - Füge create_filter_string zu QCA_AID_assets/utils/common.py hinzu
  - Füge Docstrings hinzu
  - Aktualisiere __init__.py Dateien für Exports
  - _Requirements: 4.2, 4.3, 7.2_

- [x] 7.1 Schreibe Property-Test für Hilfsfunktionen Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 4.2, 4.3**

- [x] 8. Verschiebe main Funktion





  - Erstelle QCA_AID_assets/explorer.py (NICHT main.py - das ist für QCA-AID)
  - Verschiebe main Funktion aus QCA-AID-Explorer.py
  - Aktualisiere alle Imports in der main Funktion
  - Füge Docstring hinzu
  - Stelle sicher, dass QCA_AID_assets/main.py (QCA-AID) unverändert bleibt
  - _Requirements: 4.1, 5.1, 5.2, 7.2_

- [x] 8.1 Schreibe Property-Test für main Funktion Position






  - **Property 3: Alle Module sind an den richtigen Orten**
  - **Validates: Requirements 4.1**

- [x] 9. Erstelle minimales Launcher-Skript





  - Erstelle neues QCA-AID-Explorer.py mit maximal 50 Zeilen
  - Implementiere Import von QCA_AID_assets.explorer.main (NICHT QCA_AID_assets.main)
  - Implementiere Aufruf von main
  - Implementiere Windows Event Loop Policy
  - Implementiere Fehlerbehandlung
  - Stelle sicher, dass QCA-AID.py unverändert bleibt
  - _Requirements: 1.1, 1.2, 1.5_

- [ ]* 9.1 Schreibe Property-Test für Launcher-Skript Größe
  - **Property 1: Launcher-Skript ist minimal**
  - **Validates: Requirements 1.1**

- [x] 10. Aktualisiere alle Imports





  - Konvertiere relative Imports zu absoluten Imports
  - Aktualisiere Import-Statements in allen Modulen
  - Stelle sicher, dass externe Abhängigkeiten am Anfang importiert werden
  - _Requirements: 5.1, 5.2, 5.4_

- [ ]* 10.1 Schreibe Property-Test für Import-Korrektheit
  - **Property 5: Imports sind korrekt**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 11. Checkpoint - Stelle sicher, dass alle Tests bestehen





  - Stelle sicher, dass alle Tests bestehen, frage den Benutzer bei Fragen

- [x] 12. Teste Funktionalitätserhaltung





  - Führe das refaktorierte System mit Testkonfiguration aus
  - Vergleiche Ausgabedateien mit Original
  - Prüfe alle Analysetypen (Netzwerk, Heatmap, Summary, Sentiment)
  - _Requirements: 1.4, 3.5, 6.1, 6.3_

- [ ]* 12.1 Schreibe Property-Test für Funktionalitätserhaltung
  - **Property 6: Funktionalität bleibt erhalten**
  - **Validates: Requirements 1.4, 3.5, 6.1**

- [ ]* 12.2 Schreibe Property-Test für Konfigurationsverarbeitung
  - **Property 7: Alle Konfigurationen werden korrekt verarbeitet**
  - **Validates: Requirements 4.4, 4.5, 6.2, 6.3, 6.4, 6.5**

- [x] 13. Prüfe auf Code-Duplikate




  - Suche nach duplizierten Klassen und Funktionen
  - Entferne alte Implementierungen aus QCA-AID-Explorer.py
  - _Requirements: 1.3_

- [ ]* 13.1 Schreibe Property-Test für Code-Duplikate
  - **Property 2: Keine Code-Duplikate**
  - **Validates: Requirements 1.3**

- [x] 14. Validiere __init__.py Exports





  - Prüfe, dass alle Klassen und Funktionen in __init__.py exportiert werden
  - Teste Imports aus den Modulen
  - _Requirements: 2.5_

- [ ]* 14.1 Schreibe Property-Test für __init__.py Exports
  - **Property 4: __init__.py Dateien exportieren korrekt**
  - **Validates: Requirements 2.5**

- [x] 15. Validiere Docstrings





  - Prüfe, dass alle öffentlichen Klassen Docstrings haben
  - Prüfe, dass alle öffentlichen Funktionen Docstrings haben
  - _Requirements: 7.2_

- [ ]* 15.1 Schreibe Property-Test für Docstrings
  - **Property 8: Docstrings sind vorhanden**
  - **Validates: Requirements 7.2**

- [x] 16. Aktualisiere Dokumentation





  - Aktualisiere qca-aid-explorer-readme.md mit neuer Struktur
  - Füge Beispiele für die Verwendung der neuen Module hinzu
  - Beschreibe die Änderungen im Vergleich zur alten Struktur
  - _Requirements: 7.1, 7.5_

- [x] 17. Validiere QCA-AID Kompatibilität





  - Führe alle QCA-AID Tests erneut durch
  - Vergleiche mit Baseline aus Task 1
  - Stelle sicher, dass QCA-AID.py weiterhin funktioniert
  - Stelle sicher, dass keine QCA-AID Funktionalität beeinträchtigt wurde
  - _Requirements: 1.4, 6.1_

- [x] 18. Finaler Checkpoint - Stelle sicher, dass alle Tests bestehen





  - Stelle sicher, dass alle Tests bestehen, frage den Benutzer bei Fragen

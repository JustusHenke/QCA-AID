# Implementation Plan

- [x] 1. Implementiere ConfigConverter Klasse





  - Erstelle neue Datei `QCA_AID_assets/utils/config/converter.py`
  - Implementiere `xlsx_to_json()` Methode die Excel-Datei liest und in Dictionary konvertiert
  - Implementiere `json_to_xlsx()` Methode die Dictionary in Excel-Datei schreibt
  - Implementiere `save_json()` mit UTF-8 und Einrückung
  - Implementiere `load_json()` mit Fehlerbehandlung
  - _Requirements: 1.2, 1.3, 1.4, 4.1, 4.2, 4.3_

- [x] 1.1 Schreibe Property-Test für Round-Trip Konvertierung


  - **Property 1: Konvertierung erhält Datenstruktur (Round-Trip)**
  - **Validates: Requirements 1.3**

- [-] 2. Implementiere ConfigSynchronizer Klasse



  - Erstelle neue Datei `QCA_AID_assets/utils/config/synchronizer.py`
  - Implementiere `__init__()` mit Pfad-Initialisierung
  - Implementiere `sync()` Hauptmethode die Synchronisationslogik orchestriert
  - Implementiere `_detect_differences()` für Differenzerkennung
  - Implementiere `_prompt_user_choice()` für Benutzerabfrage
  - Implementiere `_update_from_xlsx()` und `_update_from_json()`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.1 Schreibe Property-Test für Differenzerkennung






  - **Property 2: Differenzerkennung ist vollständig**
  - **Validates: Requirements 2.1, 2.2**

- [x] 2.2 Schreibe Property-Test für Synchronisationskonsistenz













  - **Property 3: Synchronisation stellt Konsistenz her**
  - **Validates: Requirements 2.4, 2.5**

- [x] 3. Erweitere ConfigLoader für JSON-Support





  - Modifiziere `QCA-AID-Explorer.py` ConfigLoader Klasse
  - Füge `_sync_configs()` Methode hinzu die ConfigSynchronizer aufruft
  - Füge `_load_from_json()` Methode hinzu
  - Modifiziere `_load_config()` um JSON zu bevorzugen wenn vorhanden
  - Implementiere Fallback-Logik bei JSON-Fehlern
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.1, 5.2_

- [x] 3.1 Schreibe Property-Test für Lade-Äquivalenz


  - **Property 4: JSON-Laden entspricht XLSX-Laden**
  - **Validates: Requirements 3.2**

- [x] 3.2 Schreibe Unit-Test für JSON-Präferenz


  - Teste dass JSON geladen wird wenn beide Dateien existieren
  - _Requirements: 3.1_

- [x] 3.3 Schreibe Unit-Test für Fallback-Verhalten


  - Teste Fallback auf XLSX bei korrupter JSON
  - _Requirements: 3.3_

- [x] 4. Implementiere JSON-Format-Validierung




  - Erstelle Validierungsfunktion in `QCA_AID_assets/core/validators.py`
  - Prüfe auf erforderliche Keys: "base_config", "analysis_configs"
  - Prüfe Datentypen der Konfigurationswerte
  - Integriere Validierung in ConfigLoader
  - _Requirements: 4.3_

- [x] 4.1 Schreibe Property-Test für JSON-Formatierung



  - **Property 5: JSON-Formatierung ist korrekt**
  - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 5. Integriere Synchronisation in main()





  - Modifiziere `main()` Funktion in `QCA-AID-Explorer.py`
  - Stelle sicher dass Synchronisation vor ConfigLoader-Initialisierung läuft
  - Füge Logging für Synchronisationsprozess hinzu
  - _Requirements: 5.1, 5.2, 5.3, 5.4_


- [x] 5.1 Schreibe Unit-Test für automatische Synchronisation

  - Teste dass Synchronisation ohne Differenzen automatisch abläuft
  - _Requirements: 5.3_

- [x] 6. Checkpoint - Stelle sicher dass alle Tests bestehen





  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Erstelle Dokumentation und Beispiele
  - Füge Kommentare zu neuen Klassen und Methoden hinzu
  - Erstelle Beispiel JSON-Konfigurationsdatei
  - Aktualisiere README mit JSON-Informationen
  - _Requirements: Alle_

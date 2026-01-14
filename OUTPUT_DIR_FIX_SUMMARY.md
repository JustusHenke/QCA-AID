# Explorer Output-Verzeichnis Fix - Zusammenfassung

## Problem
Der Explorer verwendete ein hardkodiertes "output"-Verzeichnis statt des in der Config-UI festgelegten Output-Verzeichnisses.

## Ursache
Die `base_config` des Explorers wurde beim Laden nicht mit dem `output_dir` aus der Hauptkonfiguration synchronisiert.

## Lösung

### 1. Synchronisation beim Laden (webapp.py)
Wenn der Explorer-Tab initialisiert wird, wird jetzt das `output_dir` aus der Hauptkonfiguration in die Explorer `base_config` übernommen:

```python
# Sync output_dir from main config if available
if 'config_data' in st.session_state and hasattr(st.session_state.config_data, 'output_dir'):
    config_data.base_config['output_dir'] = st.session_state.config_data.output_dir
```

Dies geschieht sowohl beim Laden einer bestehenden Explorer-Konfiguration als auch beim Erstellen einer neuen Default-Konfiguration.

### 2. Live-Synchronisation bei Änderungen (config_ui.py)
Wenn der Benutzer das Output-Verzeichnis in der Config-UI ändert, wird die Änderung sofort an die Explorer-Konfiguration weitergegeben:

```python
if new_output_dir != config.output_dir:
    config.output_dir = new_output_dir
    st.session_state.config_modified = True
    
    # Sync output_dir to Explorer config if it exists
    if 'explorer_config_data' in st.session_state:
        st.session_state.explorer_config_data.base_config['output_dir'] = new_output_dir
```

## Datenfluss

```
Config UI (output_dir)
    ↓
st.session_state.config_data.output_dir
    ↓
st.session_state.explorer_config_data.base_config['output_dir']
    ↓
ExplorerAnalysisRunner (config_data)
    ↓
QCAAnalyzer (base_config)
    ↓
self.base_output_dir = Path(script_dir) / output_dir
```

## Betroffene Dateien
- `QCA_AID_app/webapp.py` - Synchronisation beim Tab-Wechsel
- `QCA_AID_app/webapp_components/config_ui.py` - Live-Synchronisation bei Änderungen
- `CHANGELOG.md` - Dokumentation des Fixes

## Test-Anleitung

1. **Webapp starten**
   ```bash
   python QCA_AID_app/start_webapp.py
   ```

2. **Output-Verzeichnis in Config-UI ändern**
   - Gehe zum "Konfiguration"-Tab
   - Ändere das "Ausgabeverzeichnis" (z.B. von "output" zu "my_results")
   - Speichere die Konfiguration

3. **Explorer-Analyse durchführen**
   - Wechsle zum "Explorer"-Tab
   - Lade eine Analysedatei (QCA-AID_Analysis_*.xlsx)
   - Konfiguriere eine Analyse (z.B. Sunburst oder Treemap)
   - Führe die Analyse aus

4. **Ergebnis prüfen**
   - Die Meldung "📁 Ergebnisse gespeichert in: ..." sollte das konfigurierte Verzeichnis anzeigen
   - Die Dateien sollten im konfigurierten Verzeichnis gespeichert werden (nicht in "output")

## Commit
- Commit: 26993b5
- Message: "fix: Explorer nutzt jetzt konfiguriertes Output-Verzeichnis"
- Branch: main
- Pushed: ✅

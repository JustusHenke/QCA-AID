# QCA-AID Webapp

Streamlit-basierte Webanwendung zur Verwaltung von QCA-AID Analysen.

## Schnellstart

### Windows
Doppelklick auf `Start-QCA-AID-Webapp.bat`

### Manuell
```bash
cd QCA_AID_app
python start_webapp.py
```

### Oder direkt mit Streamlit
```bash
cd QCA_AID_app
streamlit run webapp.py
```

## Struktur

```
QCA_AID_app/
├── webapp.py                 # Hauptanwendung
├── start_webapp.py           # Startup-Script
├── Start-QCA-AID-Webapp.bat  # Windows One-Click-Starter
├── WEBAPP.md                 # Vollständige Dokumentation
├── webapp_components/        # UI-Komponenten
│   ├── config_ui.py
│   ├── codebook_ui.py
│   ├── analysis_ui.py
│   └── explorer_ui.py
├── webapp_logic/             # Geschäftslogik
│   ├── config_manager.py
│   ├── codebook_manager.py
│   ├── file_manager.py
│   ├── analysis_runner.py
│   └── validators.py
├── webapp_models/            # Datenmodelle
│   ├── config_data.py
│   ├── codebook_data.py
│   └── file_info.py
└── .streamlit/               # Streamlit-Konfiguration
    └── config.toml
```

## Dokumentation

Siehe [WEBAPP.md](WEBAPP.md) für die vollständige Dokumentation.

## Tests

Tests befinden sich im `../tests/` Verzeichnis und können mit pytest ausgeführt werden:

```bash
cd ..
pytest tests/
```

## Support

- GitHub Issues: https://github.com/JustusHenke/QCA-AID/issues
- E-Mail: justus.henke@hof.uni-halle.de

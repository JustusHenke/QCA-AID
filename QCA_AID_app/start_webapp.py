#!/usr/bin/env python3
"""
Startup script für QCA-AID Webapp
Prüft Abhängigkeiten und startet Streamlit
"""
import sys
import subprocess
from pathlib import Path

# Fix für Unicode-Encoding auf Windows-Konsolen
if sys.platform == 'win32':
    try:
        # Setze stdout und stderr auf UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # Fallback für ältere Python-Versionen
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def check_dependencies():
    """Prüft ob erforderliche Pakete installiert sind"""
    required = ['streamlit', 'pandas', 'openpyxl']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Fehlende Abhängigkeiten: {', '.join(missing)}")
        print("Installiere mit: pip install -r requirements.txt")
        return False
    return True


def ensure_config():
    """Erstellt .streamlit/config.toml falls nicht vorhanden"""
    config_dir = Path('.streamlit')
    config_file = config_dir / 'config.toml'
    
    if not config_file.exists():
        config_dir.mkdir(exist_ok=True)
        config_content = '''[server]
# Bind only to localhost for security
address = "127.0.0.1"
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
# Disable usage statistics collection
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
'''
        config_file.write_text(config_content)
        print("Streamlit-Konfiguration erstellt")


def ensure_directories():
    """Erstellt erforderliche Verzeichnisse falls nicht vorhanden"""
    # Webapp directories (relative to QCA_AID_app)
    webapp_directories = [
        'webapp_components',
        'webapp_logic',
        'webapp_models'
    ]
    
    # Project directories (relative to parent directory)
    project_directories = [
        '../input',
        '../output'
    ]
    
    for directory in webapp_directories:
        Path(directory).mkdir(exist_ok=True)
    
    for directory in project_directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Verzeichnisstruktur überprüft")


def main():
    """Haupteinstiegspunkt"""
    print("QCA-AID Webapp wird gestartet...")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    ensure_config()
    ensure_directories()
    
    print("=" * 50)
    print("Starte Streamlit auf http://127.0.0.1:8501")
    print("Drücke Ctrl+C zum Beenden")
    print("=" * 50)
    
    # Starte Streamlit
    try:
        subprocess.run(['streamlit', 'run', 'webapp.py'])
    except KeyboardInterrupt:
        print("\nWebapp beendet")
    except FileNotFoundError:
        print("\nFehler: webapp.py nicht gefunden")
        print("Stelle sicher, dass du dich im QCA_AID_app Verzeichnis befindest")
        sys.exit(1)


if __name__ == '__main__':
    main()

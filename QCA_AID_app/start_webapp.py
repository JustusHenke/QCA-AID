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
    """Prüft ob erforderliche Pakete installiert sind und gibt Diagnose-Hinweise"""
    required = ['streamlit', 'pandas', 'openpyxl']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        python_exe = sys.executable
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        print(f"\nFehlende Abhängigkeiten: {', '.join(missing)}")
        print(f"\n  Python-Interpreter:  {python_exe}")
        print(f"  Python-Version:      {sys.version.split()[0]}")
        print(f"  Virtuelle Umgebung:  {'Ja' if in_venv else 'Nein'}")
        
        print(f"\nInstalliere mit diesem Interpreter:")
        print(f"  {python_exe} -m pip install -r requirements.txt")
        
        if not in_venv:
            print(f"\nHinweis: Es ist keine virtuelle Umgebung aktiv.")
            print(f"  Falls du ein venv nutzt, aktiviere es zuerst:")
            if sys.platform == 'win32':
                print(f"    .\\venv\\Scripts\\activate")
            else:
                print(f"    source venv/bin/activate")
            print(f"  Dann starte dieses Script erneut.")
        
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
enableCORS = true
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
    webapp_path = Path(__file__).parent / 'webapp.py'
    if not webapp_path.exists():
        print(f"\nFehler: webapp.py nicht gefunden unter {webapp_path}")
        print("Stelle sicher, dass die Datei QCA_AID_app/webapp.py existiert")
        sys.exit(1)
    
    try:
        # Versuche zuerst streamlit als Python-Modul (robuster als CLI)
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(webapp_path)])
    except KeyboardInterrupt:
        print("\nWebapp beendet")
    except Exception as e:
        print(f"\nFehler beim Starten: {e}")
        print("Stelle sicher, dass streamlit installiert ist: pip install streamlit")
        sys.exit(1)


if __name__ == '__main__':
    main()

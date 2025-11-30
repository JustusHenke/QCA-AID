@echo off
REM ============================================
REM QCA-AID Webapp - One-Click Starter
REM ============================================

echo.
echo ========================================
echo   QCA-AID Webapp wird gestartet...
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python wurde nicht gefunden!
    echo.
    echo Bitte installieren Sie Python 3.10 oder 3.11:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo   WARNUNG: Streamlit nicht installiert
    echo ========================================
    echo.
    echo Streamlit und andere Abhaengigkeiten muessen installiert werden.
    echo.
    echo Moechten Sie die Installation jetzt durchfuehren?
    echo.
    echo Druecken Sie J fuer Ja oder N fuer Nein
    echo.
    set /p install="Ihre Wahl (J/N): "
    
    if /i "%install%"=="J" (
        echo.
        echo ========================================
        echo   Installiere Abhaengigkeiten...
        echo ========================================
        echo.
        python -m pip install -r ..\requirements.txt
        if errorlevel 1 (
            echo.
            echo ========================================
            echo   FEHLER: Installation fehlgeschlagen!
            echo ========================================
            echo.
            pause
            exit /b 1
        )
        echo.
        echo ========================================
        echo   Installation erfolgreich!
        echo ========================================
        echo.
        timeout /t 2 >nul
    ) else (
        echo.
        echo ========================================
        echo   Installation abgebrochen
        echo ========================================
        echo.
        echo Bitte installieren Sie manuell mit:
        echo   pip install -r ..\requirements.txt
        echo.
        pause
        exit /b 1
    )
)

REM Start the webapp
echo Pruefe auf laufende Instanzen...
echo.

REM Kill any running streamlit processes silently
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo Starte QCA-AID Webapp...
echo.
echo Die Webapp oeffnet sich automatisch im Browser unter:
echo   http://127.0.0.1:8501
echo.
echo Zum Beenden druecken Sie Strg+C in diesem Fenster.
echo.
echo ========================================
echo.

python -m streamlit run webapp.py

REM Streamlit was closed or exited
echo.
echo ========================================
echo   Webapp wurde beendet
echo ========================================
echo.

@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo QCA-AID Setup für Windows
echo ========================================
echo.

REM Prüfe ob Python installiert ist
echo [1/4] Prüfe Python-Installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [FEHLER] Python ist nicht installiert oder nicht im PATH!
    echo.
    echo Bitte installieren Sie Python von: https://www.python.org/downloads/
    echo.
    echo WICHTIG: Wählen Sie Python Version 3.8 bis 3.12 ^(max. 3.12 wegen spaCy^)
    echo          Aktivieren Sie bei der Installation "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Hole Python-Version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python gefunden: Version %PYTHON_VERSION%

REM Prüfe Python-Version (muss zwischen 3.8 und 3.12 sein)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo.
    echo [WARNUNG] Python Version zu alt! Mindestens Python 3.8 erforderlich.
    echo           Bitte aktualisieren Sie Python auf Version 3.8-3.12
    echo.
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% LSS 8 (
    echo.
    echo [WARNUNG] Python Version zu alt! Mindestens Python 3.8 erforderlich.
    echo           Bitte aktualisieren Sie Python auf Version 3.8-3.12
    echo.
    pause
    exit /b 1
)

if %MAJOR% EQU 3 if %MINOR% GTR 12 (
    echo.
    echo [WARNUNG] Python Version zu neu! Maximal Python 3.12 empfohlen ^(wegen spaCy^).
    echo           Aktuelle Version: %PYTHON_VERSION%
    echo.
    echo Möchten Sie trotzdem fortfahren? ^(J/N^)
    set /p CONTINUE=
    if /i not "!CONTINUE!"=="J" (
        echo Setup abgebrochen.
        pause
        exit /b 1
    )
)

echo Python-Version OK: %PYTHON_VERSION%
echo.

REM Prüfe pip
echo [2/4] Prüfe pip-Installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [FEHLER] pip ist nicht verfügbar!
    echo Versuche pip zu installieren...
    python -m ensurepip --default-pip
    if errorlevel 1 (
        echo pip-Installation fehlgeschlagen. Bitte manuell installieren.
        pause
        exit /b 1
    )
)
echo pip gefunden und bereit.
echo.

REM Upgrade pip
echo Aktualisiere pip...
python -m pip install --upgrade pip --quiet
echo.

REM Installiere Requirements
echo [3/4] Installiere Python-Pakete aus requirements.txt...
echo Dies kann einige Minuten dauern...
echo.

if not exist requirements.txt (
    echo [FEHLER] requirements.txt nicht gefunden!
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [FEHLER] Installation der Pakete fehlgeschlagen!
    echo Bitte prüfen Sie die Fehlermeldungen oben.
    echo.
    pause
    exit /b 1
)

echo.
echo Alle Pakete erfolgreich installiert!
echo.

REM Installiere spaCy Sprachmodell (optional, aber empfohlen)
echo Möchten Sie das deutsche spaCy-Sprachmodell installieren? ^(empfohlen^) ^(J/N^)
set /p INSTALL_SPACY=
if /i "!INSTALL_SPACY!"=="J" (
    echo Installiere spaCy Deutsch-Modell...
    python -m spacy download de_core_news_sm
    echo.
)

REM Erstelle Desktop-Verknüpfung
echo [4/4] Erstelle Desktop-Verknüpfung...

set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%start_QCA-AID-app.py
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT_PATH=%DESKTOP%\QCA-AID.lnk

REM Erstelle VBScript für Verknüpfung
set VBS_PATH=%TEMP%\create_shortcut.vbs
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBS_PATH%"
echo sLinkFile = "%SHORTCUT_PATH%" >> "%VBS_PATH%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%VBS_PATH%"
echo oLink.TargetPath = "pythonw.exe" >> "%VBS_PATH%"
echo oLink.Arguments = """%SCRIPT_PATH%""" >> "%VBS_PATH%"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%VBS_PATH%"
echo oLink.Description = "QCA-AID Webapp starten" >> "%VBS_PATH%"
echo oLink.Save >> "%VBS_PATH%"

cscript //nologo "%VBS_PATH%"
del "%VBS_PATH%"

if exist "%SHORTCUT_PATH%" (
    echo Desktop-Verknüpfung erstellt: %SHORTCUT_PATH%
) else (
    echo [WARNUNG] Desktop-Verknüpfung konnte nicht erstellt werden.
    echo Sie können die Anwendung manuell starten mit: python start_QCA-AID-app.py
)

echo.
echo ========================================
echo Setup erfolgreich abgeschlossen!
echo ========================================
echo.
echo Sie können QCA-AID jetzt starten:
echo   - Doppelklick auf die Desktop-Verknüpfung "QCA-AID"
echo   - Oder: python start_QCA-AID-app.py
echo.
echo Viel Erfolg mit QCA-AID!
echo.
pause

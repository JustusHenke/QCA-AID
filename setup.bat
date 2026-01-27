@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo QCA-AID Setup für Windows
echo ========================================
echo.

REM --- 1. Python-Installation prüfen ---
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
    goto :error
)

REM --- 2. Python-Version prüfen ---
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python gefunden: Version %PYTHON_VERSION%

REM Python-Version parsen
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

REM Version prüfen
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
    goto :error
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
        goto :error
    )
)
echo Python-Version OK: %PYTHON_VERSION%
echo.

REM --- 3. pip prüfen und aktualisieren ---
echo [2/4] Prüfe pip-Installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [FEHLER] pip ist nicht verfügbar!
    echo Versuche pip zu installieren...
    python -m ensurepip --default-pip
    if errorlevel 1 (
        echo pip-Installation fehlgeschlagen. Bitte manuell installieren.
        goto :error
    )
)
echo pip gefunden und bereit.
echo.
echo Aktualisiere pip...
python -m pip install --upgrade pip --quiet
echo.

REM --- 4. Python-Pfad ermitteln ---
REM Versuche, den Python-Pfad aus der Registry zu lesen
for /f "tokens=2,*" %%a in ('reg query "HKLM\SOFTWARE\Python\PythonCore\%MAJOR%.%MINOR%\InstallPath" /ve 2^>nul') do set PYTHON_PATH=%%b\python.exe
REM Falls nicht gefunden, versuche es mit 'where'
if not exist "!PYTHON_PATH!" (
    for /f "delims=" %%i in ('where python 2^>nul') do set PYTHON_PATH=%%i
)
if "!PYTHON_PATH!"=="" (
    echo [FEHLER] Python-Pfad konnte nicht ermittelt werden.
    goto :error
)
echo Python-Pfad: !PYTHON_PATH!
echo.

echo [3/4] Installiere Python-Pakete aus requirements.txt...
echo Dies kann einige Minuten dauern...
echo.

if not exist requirements.txt (
    echo [FEHLER] requirements.txt nicht gefunden!
    goto :error
)

REM Prüfe, ob requirements.txt gültige Pakete enthält
findstr /r /c:"^[^#][^ ]" "requirements.txt" >nul
if errorlevel 1 (
    echo [WARNUNG] requirements.txt ist leer oder enthält keine gültigen Pakete.
    echo           Bitte stellen Sie sicher, dass die Datei Paketnamen enthält ^(z. B. "numpy==1.21.0"^).
    goto :error
)

echo Starte Installation der Pakete...
"!PYTHON_PATH!" -m pip install --user --no-warn-script-location -r requirements.txt

if errorlevel 1 (
    echo.
    echo [FEHLER] Installation der Pakete fehlgeschlagen!
    echo Bitte prüfen Sie die Fehlermeldungen oben.
    echo.
    goto :error
) else (
    echo.
    echo [ERFOLG] Alle Pakete erfolgreich installiert!
    echo.
)


REM --- 6. spaCy-Modell installieren (optional) ---
echo.
echo Möchten Sie das deutsche spaCy-Sprachmodell installieren? ^(empfohlen^) ^(J/N^)
set /p INSTALL_SPACY=
if /i "!INSTALL_SPACY!"=="J" (
    echo Installiere spaCy Deutsch-Modell...
    "!PYTHON_PATH!" -m spacy download de_core_news_sm
    if errorlevel 1 (
        echo [WARNUNG] Installation des spaCy-Modells fehlgeschlagen.
        echo           Sie können es später manuell installieren mit:
        echo             python -m spacy download de_core_news_sm
        echo.
        echo Das Setup wird trotzdem fortgesetzt...
    ) else (
        echo ✅ spaCy Deutsch-Modell erfolgreich installiert!
    )
    echo.
) else (
    echo spaCy-Installation übersprungen.
    echo.
)

REM --- 7. Desktop-Verknüpfung mit Icon erstellen ---
echo [4/4] Erstelle Desktop-Verknüpfung...
set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%start_QCA-AID-app.py
set PNG_ICON=%SCRIPT_DIR%qca_aid_icon.png
set ICO_ICON=%SCRIPT_DIR%qca_aid_icon.ico

REM Prüfe beide mögliche Desktop-Pfade (lokal und OneDrive)
set DESKTOP1=%USERPROFILE%\Desktop
set DESKTOP2=%USERPROFILE%\OneDrive\Desktop

REM Prüfe, welcher Desktop-Pfad existiert
if exist "!DESKTOP2!" (
    set DESKTOP=!DESKTOP2!
) else if exist "!DESKTOP1!" (
    set DESKTOP=!DESKTOP1!
) else (
    echo [FEHLER] Kein Desktop-Pfad gefunden ^(weder %USERPROFILE%\Desktop noch %USERPROFILE%\OneDrive\Desktop^).
    goto :error
)

set SHORTCUT_PATH=!DESKTOP!\QCA-AID.lnk


REM Erstelle VBScript für Verknüpfung - OHNE Delayed Expansion in den Werten
set VBS_PATH=%TEMP%\create_shortcut.vbs
(
    echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
    echo sLinkFile = "%DESKTOP%\QCA-AID.lnk"
    echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
    echo oLink.TargetPath = "!PYTHON_PATH!"
    echo oLink.Arguments = """!SCRIPT_PATH!"""
    echo oLink.WorkingDirectory = "!SCRIPT_DIR!"
    echo oLink.Description = "QCA-AID Webapp starten"
    if exist "!ICO_ICON!" (
        echo oLink.IconLocation = "!ICO_ICON!,0"
    )
    echo oLink.Save
) > "!VBS_PATH!"

REM Führe das VBScript aus
cscript //nologo "!VBS_PATH!"
del "!VBS_PATH!"

if exist "!SHORTCUT_PATH!" (
    echo Desktop-Verknüpfung erstellt: !SHORTCUT_PATH!
    if exist "!ICO_ICON!" (
        echo Mit benutzerdefiniertem Icon: !ICO_ICON!
    )
) else (
    echo [WARNUNG] Desktop-Verknüpfung konnte nicht erstellt werden.
    echo Sie können die Anwendung manuell starten mit: "!PYTHON_PATH!" "!SCRIPT_PATH!"
)
echo.

REM --- 8. Abschluss ---
echo ========================================
echo Setup erfolgreich abgeschlossen!
echo ========================================
echo.
echo Sie können QCA-AID jetzt starten:
echo   - Doppelklick auf die Desktop-Verknüpfung "QCA-AID"
echo   - Oder: "!PYTHON_PATH!" "!SCRIPT_PATH!"
echo.
echo Viel Erfolg mit QCA-AID!
echo.
goto :end

:error
echo.
echo ========================================
echo Setup wurde mit Fehlern beendet!
echo ========================================
echo.
echo Bitte prüfen Sie die Fehlermeldungen oben und versuchen Sie es erneut.
echo.
pause

:end
echo.
echo Drücken Sie eine beliebige Taste zum Beenden...
pause
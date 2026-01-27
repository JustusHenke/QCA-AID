@echo off
REM Diese Datei startet setup.bat in einem Fenster, das nicht automatisch schließt
REM Nützlich für Debugging und bei restriktiven Windows-Einstellungen

echo Starte QCA-AID Setup im Debug-Modus...
echo.
cmd /k "setup.bat"

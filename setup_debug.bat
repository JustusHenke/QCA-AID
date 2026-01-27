@echo off
REM Diese Datei startet setup.bat in einem Fenster, das nicht automatisch schließt
REM Nützlich für Debugging und bei restriktiven Windows-Einstellungen

echo Starte QCA-AID Setup im Debug-Modus...
echo.
call setup.bat
echo.
echo Setup-Prozess abgeschlossen.
echo Fenster bleibt offen fuer Fehleranalyse.
pause

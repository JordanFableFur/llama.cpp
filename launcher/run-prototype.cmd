@echo off
REM Double-click me. Launches the WPF prototype (zero install). Uses Windows PowerShell 5.1 (STA) for WPF.
powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0FiresideLauncher.prototype.ps1"

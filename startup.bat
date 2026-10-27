@echo off
cd /d %~dp0
:: Start Photool at login, in the background with a tray icon:
::   startup.bat install    start it at every login
::   startup.bat remove     stop starting it at login
::   startup.bat            say whether it is set up
set "PY=%USERPROFILE%\photool-env\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python venv not found. Run setup.bat first.
    exit /b 1
)
"%PY%" backend\login_item.py %*

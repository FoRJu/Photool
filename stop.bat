@echo off
cd /d %~dp0

echo ====================================
echo  Photool - Stopping...
echo ====================================
echo.

echo [1/2] Stopping the server...
:: By what it is running, not by the port: the venv's python.exe is a launcher, so
:: the process on the port has a parent that would otherwise keep running, and a
:: server started in the background has no window to close.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'serve\.py|uvicorn.+main:app' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000 "') do (
    taskkill /pid %%a /t /f >nul 2>&1
)

echo [2/2] Stopping frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":5173 "') do (
    taskkill /pid %%a /t /f >nul 2>&1
)

echo.
echo Done.

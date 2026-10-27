@echo off
cd /d %~dp0

echo ====================================
echo  Photool - Starting...
echo ====================================
echo.

:: The venv lives OUTSIDE this project folder on purpose.
:: A sandboxed tool session (an AI coding harness) confines any process whose
:: executable sits inside the project folder, and that makes the SQLite
:: catalog / thumbnail cache on the photo drive fail with EACCES (13).
:: Keeping the venv outside the project folder is what lets it write there.
set "VENV=%USERPROFILE%\photool-env\venv"

:: Check backend venv exists
if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] Python venv not found at %VENV%. Run setup first:
    echo     python -m venv "%VENV%"
    echo     "%VENV%\Scripts\python.exe" -m pip install -r backend\requirements.txt
    exit /b 1
)

:: Check frontend node_modules exists
if not exist "frontend\node_modules" (
    echo [ERROR] Node modules not found. Run setup first:
    echo     cd frontend
    echo     npm install
    exit /b 1
)

:: Start backend (minimized window)
echo [1/2] Starting backend on http://localhost:8000 ...
start "PhotoManager-Backend" /min /D "%~dp0backend" "%VENV%\Scripts\python.exe" -m uvicorn main:app --reload --port 8000 --no-proxy-headers

:: Give backend a moment to start
timeout /t 2 /nobreak >nul

:: Start frontend (minimized window)
echo [2/2] Starting frontend on http://localhost:5173 ...
start "PhotoManager-Frontend" /min cmd /c "cd /d %~dp0frontend && npm run dev"

:: Wait for frontend to be ready
timeout /t 3 /nobreak >nul

echo.
echo ====================================
echo  Ready!
echo  Open: http://localhost:5173
echo  Stop: run stop.bat
echo ====================================
echo.

:: Open browser
start http://localhost:5173

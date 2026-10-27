@echo off
cd /d %~dp0

echo ====================================
echo  Photool - First Setup
echo ====================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    exit /b 1
)
echo [OK] Python found

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from nodejs.org
    exit /b 1
)
echo [OK] Node.js found

:: Check exiftool
exiftool -ver >nul 2>&1
if errorlevel 1 (
    echo [WARN] exiftool not found. Required for fast CR3 thumbnail extraction.
    echo        Installing via winget...
    winget install OliverBetz.ExifTool
    echo        If winget failed, download from https://exiftool.org
    echo        and add to PATH.
)
if not errorlevel 1 echo [OK] exiftool found

echo.

:: Backend setup
:: The venv lives OUTSIDE this project folder on purpose - see dev.bat.
echo [1/3] Setting up Python backend...
set "VENV=%USERPROFILE%\photool-env\venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo     Creating venv at %VENV% ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Could not create the venv at %VENV%
        exit /b 1
    )
)
cd backend
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed
    exit /b 1
)
echo [OK] Backend ready (%VENV%)
cd ..

:: Frontend setup
echo.
echo [2/3] Setting up React frontend...
cd frontend
call npm install --silent
echo [OK] Frontend ready
cd ..

:: Config
echo.
if not exist "backend\config.yaml" (
    echo [3/3] Creating config.yaml...
    copy backend\config.example.yaml backend\config.yaml >nul
    echo [ACTION REQUIRED] Edit backend\config.yaml with your library path.
    echo        Open it now? 
    pause
    notepad backend\config.yaml
) else (
    echo [3/3] config.yaml already exists, skipping.
)

echo.
echo ====================================
echo  Setup complete! Run dev.bat to start.
echo ====================================

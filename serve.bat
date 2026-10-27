@echo off
cd /d %~dp0

echo ====================================
echo  Photool - serving the album to other devices
echo ====================================
echo.

:: The same venv dev.bat uses, outside the project folder on purpose (see dev.bat).
set "VENV=%USERPROFILE%\photool-env\venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found at %VENV%. Run setup.bat first.
    pause
    exit /b 1
)

:: Packages: installs anything requirements.txt lists that the venv is missing
:: (the first run after an update), and is quick when nothing is.
"%PY%" -c "import uvicorn, fastapi, rawpy" >nul 2>&1
if errorlevel 1 (
    echo [1/3] Installing Python packages...
    "%PY%" -m pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Python packages OK
)

:: The page. Built when missing, or when asked: serve.bat rebuild (serve.py also
:: rebuilds it by itself when the app's source is newer than the built page)
if /i "%~1"=="rebuild" if exist "frontend\dist" rmdir /s /q "frontend\dist"
if not exist "frontend\dist\index.html" (
    echo [2/3] Building the page...
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Node.js / npm not found. Install it from nodejs.org.
        pause
        exit /b 1
    )
    pushd frontend
    if not exist "node_modules" call npm install
    call npm run build
    popd
    if not exist "frontend\dist\index.html" (
        echo [ERROR] The page did not build.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Page already built ^(run "serve.bat rebuild" after updating the code^)
)

cd backend

:: In the background, with no window: what Windows starts at login. The log is
:: serve.log beside the catalog; stop.bat stops it.
if /i "%~1"=="background" (
    echo [3/3] Starting in the background. stop.bat stops it.
    start "" "%VENV%\Scripts\pythonw.exe" serve.py --background
    exit /b 0
)

echo [3/3] Starting. Leave this window open; close it to stop.
echo        The first time, Windows Firewall asks: allow Python on PRIVATE networks only.
echo.
"%PY%" serve.py
pause

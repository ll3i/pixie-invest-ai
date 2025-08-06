@echo off
echo =====================================
echo Starting Pixie Web Server
echo =====================================
echo.

cd /d "%~dp0\web"

echo Current directory: %cd%
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Starting Flask server...
echo Access the service at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
@echo off
echo =====================================
echo Pixie Project Setup and Run
echo =====================================
echo.

cd /d "%~dp0"

echo [1/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists!
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt -q

echo.
echo [4/4] Creating .env file from example...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo .env file created! Please edit it with your API keys.
    )
)

echo.
echo =====================================
echo Setup complete!
echo =====================================
echo.
echo Starting Pixie Web Server...
echo.

cd web
python app.py

pause
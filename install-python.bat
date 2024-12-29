@echo off
REM Check if Python is installed
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Installing Python...
    
    REM Download and install Python silently
    REM This assumes you have downloaded the Python installer (e.g., python-3.9.13-amd64.exe) and placed it in the app's folder
    REM Replace with the correct path to the Python installer
    set PYTHON_INSTALLER=python-3.9.13-amd64.exe

    IF NOT EXIST %PYTHON_INSTALLER% (
        echo Python installer not found. Please include the installer in your app directory.
        exit /b 1
    )
    
    REM Run the Python installer silently
    start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1
    
    REM Check if Python was installed successfully
    where python >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to install Python. Exiting...
        exit /b 1
    )
    echo Python installed successfully.
)

REM Create virtual environment
python -m venv python-env

REM Activate virtual environment
call python-env\Scripts\activate

REM Install necessary Python packages
pip install -r requirements.txt

REM Run the AI server
python ai_server.py

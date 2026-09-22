@echo off
:: =======================================================================
:: run_daily_auto.bat — Automated Daily Trading Bot Launcher
:: Designed for Windows Task Scheduler or unattended execution.
:: =======================================================================

:: Navigate to project root directory
cd /d "%~dp0"

:: Ensure logs directory exists
if not exist "logs" mkdir "logs"

:: Detect Python virtual environment if one exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [%date% %time%] Starting autonomous daily trading loop... >> logs\daily_scheduler.log

:: Execute the full 3-stage daily pipeline:
:: 1. Pre-Market Scan & Cumulative Optuna Tuning
:: 2. Market Hours Paper/Live Trading Session
:: 3. Post-Market Trade Audit & Performance Scorecard
python main.py daily-loop --trials 25 >> logs\daily_scheduler.log 2>&1

echo [%date% %time%] Daily trading pipeline completed. >> logs\daily_scheduler.log

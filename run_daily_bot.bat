@echo off
title IBKR AI Trading Bot - Master Controller
color 0A

echo ================================================================
echo           IBKR AI QUANT TRADING SYSTEM - 24H PIPELINE
echo ================================================================
echo.
echo  [1] Pre-Market: Scan Trending Pennies, Tune Trials ^& Build Trade Plan
echo  [2] Market Hours: Start Live Paper Trading (IBKR)
echo  [3] Post-Market: Run Trade Audit ^& Generate Performance Scorecard
echo  [4] Full 24-Hour Loop: Auto-Plan -^> Paper Trade -^> Post-Market Review
echo  [5] Launch Web Dashboard (http://localhost:5000)
echo  [6] Scan Top Trending Penny Stocks Today
echo  [7] Run Strategy Backtest (PDSB - 5 Days)
echo  [8] Exit
echo.
echo ================================================================
set /p choice="Select an option (1-8): "

if "%choice%"=="1" (
    echo.
    echo Running Pre-Market Scan, Tuning, and Plan Generation...
    python main.py daily-plan --trials 25
    pause
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting Paper Trading Session...
    python main.py paper-trade
    pause
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Running Post-Market Trade Audit...
    python main.py daily-review
    pause
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Starting Autonomous 24-Hour Loop...
    python main.py daily-loop --trials 25
    pause
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Launching Web Dashboard...
    python main.py dashboard
    pause
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Scanning Trending Penny Stocks...
    python main.py scan-pennies --top 5
    pause
    goto end
)

if "%choice%"=="7" (
    echo.
    echo Running Backtest on PDSB...
    python main.py backtest --symbol PDSB --days 5 --interval 15m --capital 1000
    pause
    goto end
)

if "%choice%"=="8" (
    exit
)

:end

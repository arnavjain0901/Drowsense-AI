@echo off
title Drowsense AI - Driver Monitoring System

cd /d "%~dp0"

echo ==========================================
echo        DROWSENSE AI
echo   Driver Monitoring System
echo ==========================================
echo.
echo Starting AI system...
echo.

call venv\Scripts\activate.bat

python app.py

echo.
echo Drowsense AI has stopped.
pause
@echo off
title Lanzador de Traductor LSM
echo Iniciando el Traductor de Lenguaje de Senas Mexicano (LSM)...
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en "%~dp0venv". 
    echo Por favor, asegurate de que la instalacion se completo correctamente.
    pause
    exit /b
)
venv\Scripts\python.exe main.py
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] La aplicacion se cerro con un codigo de error: %errorlevel%.
    pause
)

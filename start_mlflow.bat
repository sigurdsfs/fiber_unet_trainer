@echo off
setlocal

rem Always run from the directory this script lives in, regardless of machine/path.
cd /d "%~dp0"

rem Edit this if your conda environment name differs on this machine.
set CONDA_ENV_NAME=cnn_test

call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment "%CONDA_ENV_NAME%".
    echo Make sure conda is initialized for cmd.exe (run "conda init cmd.exe" once, then reopen this terminal^).
    pause
    exit /b 1
)

rem Newer MLflow versions (3.x) put the filesystem-based ./mlruns store into
rem maintenance mode and refuse to start without this opt-out.
set MLFLOW_ALLOW_FILE_STORE=true

start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:5000"

rem mlflow ui defaults to 4 worker processes; multi-worker uvicorn socket
rem sharing is broken on Windows (OSError WinError 10022), so force 1 worker.
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000 --workers 1

pause
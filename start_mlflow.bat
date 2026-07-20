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

rem Poll the port instead of a fixed delay: with a large ./mlruns store the
rem server can take well over 3 seconds to come up, which was opening the
rem browser too early and showing "this site can't be reached". Uses a
rem hidden PowerShell helper (no batch && / || nesting, no curl dependency)
rem that retries for up to 60 seconds, then opens the browser once the
rem server actually answers.
start "" powershell -NoProfile -WindowStyle Hidden -Command "$ok=$false; for ($i=0; $i -lt 60 -and -not $ok; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5000' -TimeoutSec 1 | Out-Null; $ok=$true } catch { Start-Sleep -Seconds 1 } }; if ($ok) { Start-Process 'http://127.0.0.1:5000' } else { Write-Host 'MLflow did not respond on http://127.0.0.1:5000 within 60s.' }"

rem mlflow ui defaults to 4 worker processes; multi-worker uvicorn socket
rem sharing is broken on Windows (OSError WinError 10022), so force 1 worker.
rem Use "python -m mlflow" rather than mlflow.exe directly: the generated
rem launcher exe in Scripts\ bakes in the python.exe path at install time,
rem which breaks ("Fatal error in launcher") if the conda env is ever
rem renamed/recreated. python -m always uses the currently active env.
python -m mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000 --workers 1

pause
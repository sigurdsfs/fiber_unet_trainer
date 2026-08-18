@echo off
setlocal

cd /d "%~dp0"

call conda activate cnn_test

if errorlevel 1 (
    echo Failed to activate the cnn_test Conda environment.
    pause
    exit /b 1
)

rem Training logs to MLflow immediately and errors out if nothing is listening on
rem 127.0.0.1:5000, so make sure the tracking server is up before starting - launch
rem it in its own window (left running afterward, same as running start_mlflow.bat
rem by hand) unless it's already up from an earlier session.
set "MLFLOW_UP=no"
for /f %%A in ('powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5000' -TimeoutSec 1 | Out-Null; 'yes' } catch { 'no' }"') do set "MLFLOW_UP=%%A"

if /i "%MLFLOW_UP%"=="yes" (
    echo MLflow is already running at http://127.0.0.1:5000
) else (
    echo Starting MLflow tracking server in a new window...
    start "MLflow Server" "%~dp0start_mlflow.bat"

    echo Waiting for MLflow to come up...
    powershell -NoProfile -Command "$ok=$false; for ($i=0; $i -lt 60 -and -not $ok; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5000' -TimeoutSec 1 | Out-Null; $ok=$true } catch { Start-Sleep -Seconds 1 } }; if (-not $ok) { Write-Host 'WARNING: MLflow did not respond within 60s - continuing anyway.' }"
)

python -m fiberseg.train --config ".\configs\micronet\unetPlus_encoder_sweep_improved.yaml"

echo.
echo Training command finished.
pause
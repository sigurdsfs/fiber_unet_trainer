@echo off
setlocal

cd /d "%~dp0"

call conda activate cnn_test

if errorlevel 1 (
    echo Failed to activate the cnn_test Conda environment.
    pause
    exit /b 1
)

python -m fiberseg.train --config ".\configs\micronet\unet_resnet50.yaml"

echo.
echo Training command finished.
pause
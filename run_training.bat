@echo off
setlocal

cd /d "C:\Users\lababr\Desktop\fiber_unet_trainer_v2 - Sigurd\fiber_unet_trainer"

call conda activate cnn_test

if errorlevel 1 (
    echo Failed to activate the cnn_test Conda environment.
    pause
    exit /b 1
)

python -m fiberseg.train --config ".\configs\unetPlus_resnet50.yaml"

echo.
echo Training command finished.
pause
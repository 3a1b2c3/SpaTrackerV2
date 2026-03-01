:: filepath: c:\workspace\world\Infinite-World\infer_all_examples.bat
@echo off
setlocal enabledelayedexpansion
:: Infinite World - Run inference over all example_case subfolders
:: Usage: infer_all_examples.bat [num_gpus] [checkpoint_dir] [config_yaml] [prompts_yaml] [num_chunks] [low_memory]
:: Example: infer_all_examples.bat 1
:: Example: infer_all_examples.bat 1 .\checkpoints

set NUM_GPUS=%1
if "%NUM_GPUS%"=="" set NUM_GPUS=1
set CHECKPOINT_DIR=%2
if "%CHECKPOINT_DIR%"=="" set CHECKPOINT_DIR=.\checkpoints
set CONFIG_YAML=%3
set PROMPTS_YAML=%4
set NUM_CHUNKS=%5
set LOW_MEMORY=%6

echo ==============================================
echo Infinite World - Batch Inference over all examples
echo ==============================================
echo GPUs: %NUM_GPUS%   Checkpoint: %CHECKPOINT_DIR%

for /d %%D in (.\assets\example_case\*) do (
    echo.
    echo ==============================================
    echo Example: %%~nxD
    echo ==============================================

    :: Find image file (.jpg or .png)
    set IMG=
    for %%I in ("%%D\*.jpg" "%%D\*.png") do (
        if not defined IMG set IMG=%%I
    )

    if defined IMG (
        :: Action path
        set ACTION=%%D\0001.json

        :: Output path
        set OUT=.\out\%%~nxD_output.mp4

        :: Read prompt from prompt.txt if present
        set PROMPT=
        if exist "%%D\prompt.txt" (
            set /p PROMPT=<"%%D\prompt.txt"
        )

        echo Image:  !IMG!
        echo Action: !ACTION!
        echo Output: !OUT!
        echo Prompt: !PROMPT!

        call infer_local.bat %NUM_GPUS% %CHECKPOINT_DIR% "!IMG!" "!ACTION!" "!OUT!" "!PROMPT!" %CONFIG_YAML% %PROMPTS_YAML% %NUM_CHUNKS% %LOW_MEMORY%
    ) else (
        echo No image found in %%D, skipping.
    )
)

echo.
echo ==============================================
echo All examples done.
echo ==============================================

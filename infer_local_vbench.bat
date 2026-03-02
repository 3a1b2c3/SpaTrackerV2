@echo off
:: Infinite World - VBench Crop Dataset Batch Inference
:: Runs inference on all images in the VBench crop dataset and writes timing stats to txt.
:: Usage: infer_local_vbench.bat [checkpoint_dir] [action_path] [output_base] [config_yaml] [num_chunks] [low_memory] [max_aspects] [image_types]
:: Runs inference 5x per image (VBench requires 5 videos/prompt), writes {prompt}-{0..4}.mp4 to output_base\videos\
:: image_types: comma-separated filter e.g. "background,scenery,abstract" (default: all types)
:: Example: infer_local_vbench.bat .\checkpoints .\assets\example_case\0001.json .\out\vbench "" 2 1 1 "background,scenery"

setlocal enabledelayedexpansion

:: Parameters
set CHECKPOINT_DIR=%1
if "%CHECKPOINT_DIR%"=="" set CHECKPOINT_DIR=.\checkpoints
set ACTION_PATH=%2
if "%ACTION_PATH%"=="" set ACTION_PATH=.\assets\example_case\0001.json
set OUTPUT_BASE=%3
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=.\out\vbench
set CONFIG_YAML=%4
set NUM_CHUNKS=%5
if "%NUM_CHUNKS%"=="" set NUM_CHUNKS=2
set LOW_MEMORY=%6
set MAX_ASPECTS=%7
if "%MAX_ASPECTS%"=="" set MAX_ASPECTS=1
set IMAGE_TYPES=%~8

set CROP_DIR=C:\workspace\world\VBench\vbench2_beta_i2v\vbench2_beta_i2v\data\crop
set PROMPTS_YAML=%OUTPUT_BASE%\vbench_prompts.yaml
set VBENCH_OUTPUT_DIR=%OUTPUT_BASE%\videos
set STATS_FILE=%OUTPUT_BASE%\vbench_stats.txt
set LOG_FILE=%OUTPUT_BASE%\vbench_run.log

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

echo ==============================================
echo Infinite World - VBench Crop Batch Inference
echo ==============================================
echo Crop dir:    %CROP_DIR%
echo Action path: %ACTION_PATH%
echo Output base: %OUTPUT_BASE%
echo Num chunks:  %NUM_CHUNKS%
if not "%IMAGE_TYPES%"=="" echo Image types: %IMAGE_TYPES%
if "%LOW_MEMORY%"=="1" echo Low memory:  ENABLED

:: -------------------------------------------------------
:: Phase 1: Generate prompts YAML from all crop images
:: -------------------------------------------------------
echo.
echo Generating prompts YAML...

python scripts\gen_vbench_prompts.py "%CROP_DIR%" "%ACTION_PATH%" "%PROMPTS_YAML%" %MAX_ASPECTS% "%IMAGE_TYPES%" > "%OUTPUT_BASE%\img_count.tmp" 2>&1
set /p NUM_IMAGES=<"%OUTPUT_BASE%\img_count.tmp"
del "%OUTPUT_BASE%\img_count.tmp"

if not defined NUM_IMAGES set NUM_IMAGES=0
echo Generated %NUM_IMAGES% image prompts -^> %PROMPTS_YAML%

:: -------------------------------------------------------
:: Phase 2: Run inference 5x for VBench with timing
:: -------------------------------------------------------

:: Snapshot GPU before
set GPU_INFO_BEFORE=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_BEFORE=%%a  used=%%b  total=%%c
    )
)

:: Record start time
set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a START_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

:: Build optional args
set OPTIONAL_ARGS=--prompts "%PROMPTS_YAML%" --num_chunks %NUM_CHUNKS%
if not "%CONFIG_YAML%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --config "%CONFIG_YAML%"
if "%LOW_MEMORY%"=="1" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --low_memory

:: Truncate log file
type nul > "%LOG_FILE%"

set LAST_EXIT=0
for /l %%i in (0,1,4) do (
    echo.
    echo [VBench] Run %%i/4 starting at !TIME!...
    python scripts\infworld_inference.py %OPTIONAL_ARGS% --seed %%i --vbench_index %%i --vbench_output_dir "%VBENCH_OUTPUT_DIR%" 2>&1 | powershell -Command "$input | Tee-Object -Append -FilePath '%LOG_FILE%'"
    set LAST_EXIT=!ERRORLEVEL!
    echo [VBench] Run %%i/4 done. Exit: !LAST_EXIT!
)
set EXIT_CODE=%LAST_EXIT%

:: Record end time
set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do set /a END_S=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400
set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

:: Avg seconds per video (5 runs x NUM_IMAGES)
set AVG_S=N/A
set /a TOTAL_VIDEOS=NUM_IMAGES*5
if %TOTAL_VIDEOS% gtr 0 if %ELAPSED% gtr 0 set /a AVG_S=ELAPSED/TOTAL_VIDEOS

:: Snapshot GPU after
set GPU_INFO_AFTER=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_AFTER=%%a  used=%%b  total=%%c
    )
)

echo ==============================================
echo Done. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit: %EXIT_CODE%
echo Stats: %STATS_FILE%
echo ==============================================

:: -------------------------------------------------------
:: Write stats txt
:: -------------------------------------------------------
(
    echo VBench Crop Inference Stats
    echo ===========================
    echo Date:           %DATE%
    echo Start:          %START_TIME%
    echo End:            %END_TIME%
    echo Elapsed:        %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s ^(%ELAPSED%s^)
    echo Exit code:      %EXIT_CODE%
    echo.
    echo === Input ===
    echo Crop dir:       %CROP_DIR%
    echo Images:         %NUM_IMAGES%
    echo Action path:    %ACTION_PATH%
    echo Prompts YAML:   %PROMPTS_YAML%
    echo.
    echo === Settings ===
    echo Checkpoint:     %CHECKPOINT_DIR%
    echo Num chunks:     %NUM_CHUNKS%
    echo Low memory:     %LOW_MEMORY%
    echo.
    echo === Performance ===
    echo Total elapsed:  %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s ^(%ELAPSED%s^)
    echo Total videos:   %TOTAL_VIDEOS% ^(%NUM_IMAGES% images x 5 runs^)
    echo Avg per video:  %AVG_S%s
    echo.
    echo === GPU ===
    echo GPU before:     %GPU_INFO_BEFORE%
    echo GPU after:      %GPU_INFO_AFTER%
    echo.
    echo === Output ===
    echo Output base:    %OUTPUT_BASE%
    echo VBench videos:  %VBENCH_OUTPUT_DIR%
    echo Log:            %LOG_FILE%
) > "%STATS_FILE%"

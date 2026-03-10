@echo off
:: Infinite World - VBench Crop Dataset Batch Inference
:: Runs inference on all images in the VBench crop dataset and writes timing stats to txt.
:: Usage: infer_local_vbench.bat [checkpoint_dir] [action_path] [output_base] [config_yaml] [num_chunks] [low_memory] [max_aspects] [image_types] [num_samples] [base_seed] [dimension]
:: Runs inference num_samples times per image (VBench requires 5), writes {caption}-{0..N-1}.mp4 to output_base\videos\
:: image_types: comma-separated filter e.g. "background,scenery,abstract" (default: all types)
:: Skips already-generated videos automatically.
:: Example: infer_local_vbench.bat .\checkpoints .\assets\example_case\0001.json .\out\vbench "" 2 1 1 "background,scenery" 5 42

setlocal enabledelayedexpansion

:: --help
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h"     goto :help
if /i "%~1"=="/?"     goto :help
goto :run

:help
echo.
echo Infinite World - VBench Crop Batch Inference
echo.
echo Usage:
echo   infer_local_vbench.bat [checkpoint_dir] [action_path] [output_base] [config_yaml]
echo                          [num_chunks] [low_memory] [max_aspects] [image_types] [num_samples]
echo.
echo Arguments (positional, all optional):
echo   1  checkpoint_dir   Path to checkpoint directory          (default: .\checkpoints)
echo   2  action_path      Path to action JSON file              (default: .\assets\example_case\0001.json)
echo   3  output_base      Base output directory                 (default: .\out\vbench)
echo   4  config_yaml      Path to config YAML                   (default: from config)
echo   5  num_chunks       Number of video chunks per sample     (default: 2)
echo   6  low_memory       Enable low-memory mode: 0 or 1        (default: 0)
echo   7  max_aspects      Max aspect ratio variants             (default: 1)
echo   8  image_types      Comma-separated type filter           (default: all)
echo                         e.g. "background,scenery,abstract"
echo   9  num_samples      Videos to generate per prompt         (default: 5)
echo   10 base_seed        Base random seed; incremented per     (default: 42)
echo                         sample (base_seed+0 .. base_seed+N-1)
echo   11 dimension         VBench image type filter passed to    (default: all)
echo                         infworld_inference --type
echo                         e.g. "scenery,indoor" or "all" for no filter
echo.
echo Notes:
echo   - VBench requires 5 samples per prompt (num_samples=5)
echo   - Already-generated videos are skipped automatically
echo   - Outputs: {output_base}\videos_{base_seed}\{caption}-{0..N-1}.mp4
echo             e.g. a table and chairs in a room with sunlight coming through the window-4.mp4
echo   - Log:     {output_base}\vbench_run.log
echo   - Stats:   {output_base}\vbench_stats.txt
echo.
echo Example:
echo   infer_local_vbench.bat .\checkpoints .\assets\example_case\0001.json .\out\vbench "" 2 1 1 "background,scenery" 5
echo.
exit /b 0

:run
:: Parameters
set CHECKPOINT_DIR=%1
if "%CHECKPOINT_DIR%"=="" set CHECKPOINT_DIR=.\checkpoints
set ACTION_PATH=%2
if "%ACTION_PATH%"=="" set ACTION_PATH=.\assets\example_case\0001.json
set OUTPUT_BASE=%3
if "%OUTPUT_BASE%"=="" set OUTPUT_BASE=.\out\vbench
set CONFIG_YAML=%~4
if defined CONFIG_YAML if not exist "%CONFIG_YAML%" set CONFIG_YAML=
set NUM_CHUNKS=%5
if "%NUM_CHUNKS%"=="" set NUM_CHUNKS=2
set LOW_MEMORY=%6
set MAX_ASPECTS=%7
if "%MAX_ASPECTS%"=="" set MAX_ASPECTS=1
set IMAGE_TYPES=%~8
set NUM_SAMPLES=%9
if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=5
:: Args 10+ cannot be read via %~10/%~11 in Windows batch (%~10 = %~1 + "0")
:: BASE_SEED is always randomised; pass --type via IMAGE_TYPES (arg 8) instead
set BASE_SEED=42

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
set /a _VF=1+NUM_CHUNKS*80
echo Num chunks:  %NUM_CHUNKS%  ^(= %_VF% video frames: 1 + %NUM_CHUNKS%x80^)
if not "%IMAGE_TYPES%"=="" echo Image types: %IMAGE_TYPES%
if "%LOW_MEMORY%"=="1" echo Low memory:  ENABLED

:: -------------------------------------------------------
:: Phase 1: Generate prompts YAML from all crop images
:: -------------------------------------------------------
echo.
echo Generating prompts YAML...

python "%~dp0scripts\gen_vbench_prompts.py" "%CROP_DIR%" "%ACTION_PATH%" "%PROMPTS_YAML%" "%IMAGE_TYPES%" > "%OUTPUT_BASE%\img_count.tmp" 2>&1
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
set OPTIONAL_ARGS=--prompts "%PROMPTS_YAML%" --num_chunks %NUM_CHUNKS% --num_samples %NUM_SAMPLES% --seed %BASE_SEED%
if not "%IMAGE_TYPES%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --type "%IMAGE_TYPES%"
if not "%CONFIG_YAML%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --config "%CONFIG_YAML%"
if "%LOW_MEMORY%"=="1" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --low_memory

:: Truncate log file
type nul > "%LOG_FILE%"

echo.
echo [VBench] Generating %NUM_SAMPLES% samples per prompt...
python "%~dp0scripts\infworld_inference.py" %OPTIONAL_ARGS% --vbench_output_dir "%VBENCH_OUTPUT_DIR%" 2>&1 | powershell -Command "$input | Tee-Object -Append -FilePath '%LOG_FILE%'"
set EXIT_CODE=%ERRORLEVEL%
echo [VBench] Done. Exit: %EXIT_CODE%

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
set /a TOTAL_VIDEOS=NUM_IMAGES*NUM_SAMPLES
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
    echo Dimension:      %DIMENSION%
    echo.
    echo === Performance ===
    echo Num samples:    %NUM_SAMPLES%
    echo Total videos:   %TOTAL_VIDEOS% ^(%NUM_IMAGES% images x %NUM_SAMPLES% samples^)
    echo Avg per video:  %AVG_S%s
    echo.
    echo === GPU ===
    echo GPU before:     %GPU_INFO_BEFORE%
    echo GPU after:      %GPU_INFO_AFTER%
    echo.
    echo === Output ===
    echo Output base:    %OUTPUT_BASE%
    echo Seed:           %BASE_SEED%
    echo VBench videos:  %VBENCH_OUTPUT_DIR%
    echo Log:            %LOG_FILE%
) > "%STATS_FILE%"

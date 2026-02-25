:: filepath: c:\workspace\world\Infinite-World\infer_local.bat
@echo off
:: Infinite World - Local Inference Script (Single/Multi GPU)
:: Usage: infer_local.bat [num_gpus] [checkpoint_dir] [image_path] [action_path] [output_path] [prompt] [config_yaml] [prompts_yaml] [num_chunks] [low_memory]
:: Example: infer_local.bat 1 .\checkpoints ".\assets\example_case\3th person\ski.jpg" ".\assets\example_case\3th person\0001.json" .\out\ski_output.mp4 "Imagine yourself cross country skiing uphill through a bright winter landscape on a sunny day."
:: Example: infer_local.bat 1 .\checkpoints .\assets\example_case\nature\forest.jpg .\assets\example_case\nature\0001.json .\out\nature_output.mp4 "Imagine yourself walking through a sunlit forest, where golden rays of light pierce through the canopy above in long, dramatic shafts."
:: Example: infer_local.bat 1
:: Example: infer_local.bat 1 "" "" "" "" "" "" "" "" 1
:: Example: infer_local.bat 1 .\checkpoints .\assets\example_case\racer\Screenshot.png examples\00 .\out\racer_output.mp4 "Your prompt here"
:: Example: infer_local.bat 1 .\checkpoints .\assets\example_case\racer\Screenshot.png examples\00 .\out\racer_output.mp4 "Your prompt" .\configs\custom_config.yaml .\prompts\mine.yaml 2 1

:: Default values
set NUM_GPUS=%1
if "%NUM_GPUS%"=="" set NUM_GPUS=1
set CHECKPOINT_DIR=%2
if "%CHECKPOINT_DIR%"=="" set CHECKPOINT_DIR=.\checkpoints
set IMAGE_PATH=%3
if "%IMAGE_PATH%"=="" set IMAGE_PATH=.\assets\example_case\racer\Screenshot.png
set ACTION_PATH=%4
if "%ACTION_PATH%"=="" set ACTION_PATH=examples\00
set OUTPUT_PATH=%5
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=.\out\racer_output.mp4
set PROMPT=%6
if "%PROMPT%"=="" set PROMPT=A fast-paced RC car race through a suburban street on a sunny day, with the miniature vehicle zipping past houses, driveways, and mailboxes, accompanied by the hum of its motor and the cheerful sounds of children playing in the background.
set CONFIG_YAML=%7
set PROMPTS_YAML=%8
if "%PROMPTS_YAML%"=="" set PROMPTS_YAML=.\prompts\examples.yaml
set NUM_CHUNKS=%9
shift
set LOW_MEMORY=%9

echo ==============================================
echo Infinite World - Local Inference
echo ==============================================
echo Using %NUM_GPUS% GPU(s)
echo Working directory: %cd%
echo Checkpoint directory: %CHECKPOINT_DIR%
echo Image: %IMAGE_PATH%
echo Action path: %ACTION_PATH%
echo Output: %OUTPUT_PATH%
echo Prompt: %PROMPT%
if not "%CONFIG_YAML%"=="" echo Config YAML: %CONFIG_YAML%
if not "%PROMPTS_YAML%"=="" echo Prompts YAML: %PROMPTS_YAML%
if not "%NUM_CHUNKS%"=="" echo Number of chunks: %NUM_CHUNKS%
if "%LOW_MEMORY%"=="1" echo Low Memory Mode: ENABLED

:: Display GPU information
echo ==============================================
echo GPU Information:
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    nvidia-smi --query-gpu=name,memory.total --format=csv
) else (
    echo nvidia-smi not found. Ensure NVIDIA drivers are installed.
)
echo ==============================================

:: Build optional arguments
set OPTIONAL_ARGS=
if not "%CONFIG_YAML%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --config %CONFIG_YAML%
if not "%PROMPTS_YAML%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --prompts %PROMPTS_YAML%
if not "%NUM_CHUNKS%"=="" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --num_chunks %NUM_CHUNKS%
if "%LOW_MEMORY%"=="1" set OPTIONAL_ARGS=%OPTIONAL_ARGS% --low_memory

:: Snapshot GPU state before inference
set GPU_INFO_BEFORE=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_BEFORE=%%a  used=%%b  total=%%c
    )
)

:: Determine stats/log file paths alongside output
for %%F in ("%OUTPUT_PATH%") do set STATS_FILE=%%~dpnF_stats.txt
for %%F in ("%OUTPUT_PATH%") do set LOG_FILE=%%~dpnF_run.log

:: Ensure output directory exists
for %%F in ("%OUTPUT_PATH%") do if not exist "%%~dpF" mkdir "%%~dpF"

:: Record start time
set START_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do (
    set /a START_S=%%a*3600+%%b*60+%%c
)

if "%NUM_GPUS%"=="1" (
    :: Single GPU: run directly to avoid torchrun port (EADDRINUSE)
    python scripts\infworld_inference.py %OPTIONAL_ARGS% 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"
) else (
    set MASTER_PORT=%MASTER_PORT%
    if "%MASTER_PORT%"=="" set MASTER_PORT=29400
    echo MASTER_PORT: %MASTER_PORT%
    torchrun --nnodes=1 --nproc_per_node=%NUM_GPUS% ^
        --rdzv_id=100 --rdzv_backend=c10d ^
        --rdzv_endpoint=localhost:%MASTER_PORT% ^
        scripts\infworld_inference.py %OPTIONAL_ARGS% 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"
)
set EXIT_CODE=%ERRORLEVEL%

:: Record end time and compute elapsed seconds
set END_TIME=%TIME%
for /f "tokens=1-4 delims=:., " %%a in ("%TIME: =0%") do (
    set /a END_S=%%a*3600+%%b*60+%%c
)
set /a ELAPSED=END_S-START_S
if %ELAPSED% lss 0 set /a ELAPSED+=86400

set /a ELAPSED_H=ELAPSED/3600
set /a ELAPSED_M=(ELAPSED%%3600)/60
set /a ELAPSED_SS=ELAPSED%%60

:: Extract FPS lines logged by the inference script
set FPS_LINES=
for /f "delims=" %%L in ('findstr /i "fps\|frames per second\|frame/s\|it/s" "%LOG_FILE%" 2^>nul') do (
    set FPS_LINES=%%L
)

:: Snapshot GPU state after inference
set GPU_INFO_AFTER=N/A
where nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "skip=1 tokens=1,2,3 delims=, " %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total --format^=csv^,noheader') do (
        set GPU_INFO_AFTER=%%a  used=%%b  total=%%c
    )
)

:: Extract video properties from output file using ffprobe
set VIDEO_FPS=N/A
set VIDEO_FRAMES=N/A
set VIDEO_DURATION=N/A
set VIDEO_RESOLUTION=N/A
where ffprobe >nul 2>&1
if %ERRORLEVEL%==0 (
    if exist "%OUTPUT_PATH%" (
        for /f "tokens=1" %%V in ('ffprobe -v error -select_streams v:0 -show_entries stream^=r_frame_rate -of csv^=p^=0 "%OUTPUT_PATH%" 2^>nul') do set VIDEO_FPS_RAW=%%V
        for /f "tokens=1" %%V in ('ffprobe -v error -select_streams v:0 -show_entries stream^=nb_frames -of csv^=p^=0 "%OUTPUT_PATH%" 2^>nul') do set VIDEO_FRAMES=%%V
        for /f "tokens=1" %%V in ('ffprobe -v error -select_streams v:0 -show_entries format^=duration -of csv^=p^=0 "%OUTPUT_PATH%" 2^>nul') do set VIDEO_DURATION=%%Vs
        for /f "tokens=1" %%V in ('ffprobe -v error -select_streams v:0 -show_entries stream^=width^,height -of csv^=p^=0^:s^=x "%OUTPUT_PATH%" 2^>nul') do set VIDEO_RESOLUTION=%%V
        :: Compute numeric FPS from fraction (e.g. 30000/1001 -> ~29.97)
        for /f "tokens=1,2 delims=/" %%a in ("%VIDEO_FPS_RAW%") do (
            set /a FPS_NUM=%%a
            set /a FPS_DEN=%%b
        )
        if defined FPS_DEN if %FPS_DEN% gtr 0 (
            set /a VIDEO_FPS=FPS_NUM/FPS_DEN
        ) else (
            set VIDEO_FPS=%VIDEO_FPS_RAW%
        )
    )
)

:: Compute inference FPS: total frames / elapsed seconds
set INFER_FPS=N/A
if defined VIDEO_FRAMES if %ELAPSED% gtr 0 (
    set /a INFER_FPS=VIDEO_FRAMES/ELAPSED
)

echo ==============================================
echo Job complete. Elapsed: %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  Exit code: %EXIT_CODE%
echo Video: %VIDEO_RESOLUTION%  %VIDEO_FPS% fps  %VIDEO_FRAMES% frames  %VIDEO_DURATION%
echo Inference speed: %INFER_FPS% frames/s
echo Stats written to: %STATS_FILE%
echo ==============================================

(
    echo Infinite World Inference Stats
    echo ==============================
    echo Date:              %DATE%
    echo Start time:        %START_TIME%
    echo End time:          %END_TIME%
    echo Elapsed:           %ELAPSED_H%h %ELAPSED_M%m %ELAPSED_SS%s  ^(%ELAPSED%s^)
    echo Exit code:         %EXIT_CODE%
    echo.
    echo === Settings ===
    echo GPUs:              %NUM_GPUS%
    echo Checkpoint:        %CHECKPOINT_DIR%
    echo Image:             %IMAGE_PATH%
    echo Action path:       %ACTION_PATH%
    echo Output:            %OUTPUT_PATH%
    echo Prompt:            %PROMPT%
    if not "%CONFIG_YAML%"=="" echo Config YAML:       %CONFIG_YAML%
    if not "%PROMPTS_YAML%"=="" echo Prompts YAML:      %PROMPTS_YAML%
    if not "%NUM_CHUNKS%"=="" echo Num chunks:        %NUM_CHUNKS%
    echo Low memory:        %LOW_MEMORY%
    echo.
    echo === Performance ===
    echo Inference speed:   %INFER_FPS% frames/s
    echo Video FPS:         %VIDEO_FPS%
    echo Video frames:      %VIDEO_FRAMES%
    echo Video duration:    %VIDEO_DURATION%
    echo Video resolution:  %VIDEO_RESOLUTION%
    if defined FPS_LINES echo Script FPS log:    %FPS_LINES%
    echo.
    echo === GPU ===
    echo GPU before:        %GPU_INFO_BEFORE%
    echo GPU after:         %GPU_INFO_AFTER%
    echo.
    echo === Log ===
    echo Run log:           %LOG_FILE%
) > "%STATS_FILE%"

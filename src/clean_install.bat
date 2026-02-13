@echo off
echo Cleaning corrupted pycolmap packages...

cd /d "C:\Users\kschmid\AppData\Local\Programs\Python\Python312\Lib\site-packages"

echo Removing corrupted directories...
if exist "~ycolmap" rmdir /s /q "~ycolmap" && echo Removed ~ycolmap
if exist "~~colmap" rmdir /s /q "~~colmap" && echo Removed ~~colmap
if exist "~ycolmap.dist-info" rmdir /s /q "~ycolmap.dist-info" && echo Removed ~ycolmap.dist-info
if exist "~~colmap.dist-info" rmdir /s /q "~~colmap.dist-info" && echo Removed ~~colmap.dist-info

echo.
echo Clearing pip cache...
pip cache purge

echo.
echo Installing pycolmap from PyPI only (not NVIDIA index)...
pip install --index-url https://pypi.org/simple pycolmap==3.13.0

echo.
echo Testing...
python -c "import pycolmap; print('SUCCESS:', pycolmap.__version__)"

pause

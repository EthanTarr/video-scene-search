@echo off
echo Setting OpenMP environment variable to resolve runtime conflicts...
set KMP_DUPLICATE_LIB_OK=TRUE

echo Launching Video Scene Search GUI...
echo.
echo Note: This GUI combines video processing and search functionality.
echo - Process videos to extract scenes and embeddings
echo - Search through processed videos using text queries
echo - Manage your video database
echo.

cd /d "%~dp0"
"%USERPROFILE%\anaconda3\python.exe" scripts/search_gui.py

pause

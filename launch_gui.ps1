# PowerShell script to launch Video Scene Search GUI
# This script sets the OpenMP environment variable to resolve runtime conflicts

Write-Host "Setting OpenMP environment variable to resolve runtime conflicts..." -ForegroundColor Green
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "Launching Video Scene Search GUI..." -ForegroundColor Green
Write-Host ""
Write-Host "Note: This GUI combines video processing and search functionality:" -ForegroundColor Yellow
Write-Host "  - Process videos to extract scenes and embeddings" -ForegroundColor Yellow
Write-Host "  - Search through processed videos using text queries" -ForegroundColor Yellow
Write-Host "  - Manage your video database" -ForegroundColor Yellow
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Launch the GUI
& "$env:USERPROFILE\anaconda3\python.exe" "scripts/search_gui.py"

Write-Host "GUI closed." -ForegroundColor Green

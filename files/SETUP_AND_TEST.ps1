# Tennis Tagger - Setup and Test Script (PowerShell)
# Run this to set up environment and test all improvements

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "TENNIS TAGGER - SETUP AND TEST" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Set environment variable to disable online downloads
Write-Host "Setting environment for offline model loading..." -ForegroundColor Yellow
$env:DISABLE_ONLINE_MODEL_DOWNLOADS = "1"
[System.Environment]::SetEnvironmentVariable('DISABLE_ONLINE_MODEL_DOWNLOADS', '1', 'User')
Write-Host "  DISABLE_ONLINE_MODEL_DOWNLOADS = $env:DISABLE_ONLINE_MODEL_DOWNLOADS" -ForegroundColor Green
Write-Host ""

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Install/upgrade dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes)" -ForegroundColor Gray
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed" -ForegroundColor Green
Write-Host ""

# Verify psutil is installed (for benchmark)
Write-Host "Installing benchmark dependencies..." -ForegroundColor Yellow
pip install psutil -q
Write-Host "  Benchmark tools ready" -ForegroundColor Green
Write-Host ""

# Check models directory
Write-Host "Checking models..." -ForegroundColor Yellow
$modelsDir = "models"
if (Test-Path $modelsDir) {
    $modelFiles = Get-ChildItem -Path $modelsDir -Filter "*.pt"
    Write-Host "  Found $($modelFiles.Count) model file(s):" -ForegroundColor Green
    foreach ($model in $modelFiles) {
        $sizeMB = [math]::Round($model.Length / 1MB, 1)
        Write-Host "    - $($model.Name) ($sizeMB MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "  Warning: models/ directory not found" -ForegroundColor Red
}
Write-Host ""

# Check config
Write-Host "Checking configuration..." -ForegroundColor Yellow
$configPath = "config\config.yaml"
if (Test-Path $configPath) {
    Write-Host "  Configuration file found" -ForegroundColor Green
    # Check if models_root is set
    $configContent = Get-Content $configPath -Raw
    if ($configContent -match 'models_root') {
        Write-Host "  models_root configured" -ForegroundColor Green
    } else {
        Write-Host "  Warning: models_root not found in config" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Error: Config file not found" -ForegroundColor Red
}
Write-Host ""

Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Test with a short video (< 30 seconds):" -ForegroundColor White
Write-Host "   python scripts\quick_benchmark.py --video `"C:\path\to\short_clip.mp4`" --batch 8" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Process a full video with checkpointing:" -ForegroundColor White
Write-Host "   python main.py --video `"C:\path\to\video.mp4`" --output results.csv --batch 16 --checkpoint-interval 1000" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Resume after interruption:" -ForegroundColor White
Write-Host "   python main.py --video `"C:\path\to\video.mp4`" --output results.csv --batch 16 --resume" -ForegroundColor Gray
Write-Host ""

Read-Host "Press Enter to exit"

# ============================================
# TENNIS TAGGER - COMPLETE AUTOMATED SETUP
# ============================================
# This script will:
# 1. Install ALL missing dependencies
# 2. Fix ALL import mismatches
# 3. Verify the setup
# 4. Launch the app

param(
    [switch]$SkipInstall,
    [switch]$SkipFix,
    [switch]$SkipRun
)

$ErrorActionPreference = "Continue"

# Colors
function Write-Header($text) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host " $text" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($text) {
    Write-Host "➤ $text" -ForegroundColor Green
}

function Write-Error2($text) {
    Write-Host "✗ $text" -ForegroundColor Red
}

function Write-Success($text) {
    Write-Host "✓ $text" -ForegroundColor Green
}

# Check we're in the right directory
if (!(Test-Path "main.py")) {
    Write-Error2 "main.py not found! Please run this from your project directory."
    Write-Host "Example: cd C:\Users\liamp\Downloads\files" -ForegroundColor Yellow
    exit 1
}

Write-Header "TENNIS TAGGER - COMPLETE SETUP"
Write-Host "Project Directory: $PWD" -ForegroundColor Cyan
Write-Host ""

# ============================================
# STEP 1: Install Dependencies
# ============================================
if (!$SkipInstall) {
    Write-Header "STEP 1: Installing All Dependencies"
    
    $packages = @(
        "gradio",
        "easyocr", 
        "filterpy",
        "ffmpeg-python",
        "imageio",
        "imageio-ffmpeg",
        "opencv-contrib-python",
        "PyQt5"
    )
    
    Write-Step "Installing missing packages..."
    Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($pkg in $packages) {
        Write-Host "  Installing $pkg..." -NoNewline
        pip install $pkg --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ⚠️" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Success "Dependency installation complete!"
} else {
    Write-Host "⏭️  Skipping dependency installation (--SkipInstall)" -ForegroundColor Yellow
}

# ============================================
# STEP 2: Fix Import Errors
# ============================================
if (!$SkipFix) {
    Write-Header "STEP 2: Fixing Import Mismatches"
    
    # Backup
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = "main.py.backup_$timestamp"
    Copy-Item "main.py" $backupFile
    Write-Success "Backup created: $backupFile"
    Write-Host ""
    
    # Read file
    $content = Get-Content "main.py" -Raw
    
    # All fixes
    $fixes = @(
        @{
            Old = "from detection.tracker import ObjectTracker"
            New = "from detection.tracker import MultiObjectTracker as ObjectTracker"
            Name = "ObjectTracker -> MultiObjectTracker"
        },
        @{
            Old = "from analysis.csv_generator import CSVGenerator"
            New = "from analysis.csv_generator import DartfishCSVGenerator as CSVGenerator"
            Name = "CSVGenerator -> DartfishCSVGenerator"
        },
        @{
            Old = "from analysis.comparator import TagComparator"
            New = "from analysis.comparator import CSVComparator as TagComparator"
            Name = "TagComparator -> CSVComparator"
        },
        @{
            Old = "from analysis.qc_feedback import FeedbackLoop"
            New = "from analysis.comparator import FeedbackLoop"
            Name = "FeedbackLoop location fix"
        }
    )
    
    $fixCount = 0
    foreach ($fix in $fixes) {
        if ($content -match [regex]::Escape($fix.Old)) {
            $content = $content -replace [regex]::Escape($fix.Old), $fix.New
            Write-Success "Fixed: $($fix.Name)"
            $fixCount++
        }
    }
    
    # Save
    if ($fixCount -gt 0) {
        $content | Set-Content "main.py" -NoNewline
        Write-Host ""
        Write-Success "Applied $fixCount fix(es) to main.py"
    } else {
        Write-Host "  All imports already correct!" -ForegroundColor Cyan
    }
} else {
    Write-Host "⏭️  Skipping import fixes (--SkipFix)" -ForegroundColor Yellow
}

# ============================================
# STEP 3: Verify Setup
# ============================================
Write-Header "STEP 3: Verifying Setup"

Write-Step "Checking critical packages..."
$criticalPackages = @("torch", "cv2", "gradio", "easyocr", "filterpy", "mediapipe")
$allGood = $true

foreach ($pkg in $criticalPackages) {
    Write-Host "  Checking $pkg..." -NoNewline
    $result = python -c "import $pkg; print('OK')" 2>&1
    if ($result -match "OK") {
        Write-Host " ✓" -ForegroundColor Green
    } else {
        Write-Host " ✗" -ForegroundColor Red
        $allGood = $false
    }
}

Write-Host ""
if ($allGood) {
    Write-Success "All packages verified!"
} else {
    Write-Error2 "Some packages are missing. Try running: pip install -r requirements_complete.txt"
}

# ============================================
# STEP 4: Set Python Path
# ============================================
Write-Header "STEP 4: Setting Python Path"
$env:PYTHONPATH = $PWD
Write-Success "PYTHONPATH = $PWD"

# ============================================
# STEP 5: Launch App
# ============================================
if (!$SkipRun -and $allGood) {
    Write-Header "STEP 5: Launching Tennis Tagger"
    Write-Host "Starting app.py..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    
    python app.py
} elseif ($SkipRun) {
    Write-Host ""
    Write-Host "⏭️  Skipping app launch (--SkipRun)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To launch manually:" -ForegroundColor Yellow
    Write-Host "  `$env:PYTHONPATH = `$PWD" -ForegroundColor Cyan
    Write-Host "  python app.py" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Error2 "Setup incomplete. Fix errors above before running."
}

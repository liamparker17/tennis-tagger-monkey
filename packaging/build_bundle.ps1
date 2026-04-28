# Build a self-contained Tennis Tagger bundle in dist/TennisTagger/
#
# Usage:
#   .\packaging\build_bundle.ps1                    # CPU-only torch (smaller, ~600MB)
#   .\packaging\build_bundle.ps1 -Gpu               # CUDA 12.1 torch (~2.5GB)
#   .\packaging\build_bundle.ps1 -SkipPyDeps        # reuse existing python/ dir
#   .\packaging\build_bundle.ps1 -SentryDsn 'https://abc@o123.ingest.sentry.io/456'
#
# The Sentry DSN can also come from the TENNIS_TAGGER_SENTRY_DSN env var
# (handy for CI). When set, a sentry.dsn file is dropped into the bundle so
# the installed app reports crashes back to Sentry without each end user
# having to set anything.
#
# Run from project root.

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$SkipPyDeps,
    [string]$PythonVersion = "3.11.9",
    [string]$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [string]$SentryDsn = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

$Root    = Resolve-Path "$PSScriptRoot\.."
$Dist    = Join-Path $Root "dist\TennisTagger"
$Cache   = Join-Path $Root "dist\.cache"
$PyDir   = Join-Path $Dist "python"
$FfDir   = Join-Path $Dist "ffmpeg"

Write-Host "==> Bundle root: $Dist" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Dist, $Cache | Out-Null

# ---------- 1. Build Go binary ----------
Write-Host "==> Building tagger.exe" -ForegroundColor Cyan
Push-Location $Root
try {
    $env:CGO_ENABLED = "0"
    & go build -ldflags "-s -w" -o (Join-Path $Dist "tagger.exe") .\cmd\tagger
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }
} finally {
    Pop-Location
}

# ---------- 2. Embedded Python ----------
$PyZip = Join-Path $Cache "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $PyZip)) {
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    Write-Host "==> Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $PyZip
}

if (-not $SkipPyDeps) {
    if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
    Write-Host "==> Extracting embedded Python -> $PyDir"
    Expand-Archive -Path $PyZip -DestinationPath $PyDir -Force

    # Enable site-packages in embedded python (uncomment 'import site' in ._pth)
    $pthFile = Get-ChildItem $PyDir -Filter "python*._pth" | Select-Object -First 1
    if ($pthFile) {
        $content = Get-Content $pthFile.FullName
        $content = $content -replace '#\s*import site', 'import site'
        $content += "`r`n..\ml`r`n.."
        Set-Content -Path $pthFile.FullName -Value $content -Encoding ASCII
    }

    # Bootstrap pip
    $getPip = Join-Path $Cache "get-pip.py"
    if (-not (Test-Path $getPip)) {
        Write-Host "==> Downloading get-pip.py"
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
    }
    $pyExe = Join-Path $PyDir "python.exe"
    & $pyExe $getPip --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "get-pip failed" }

    # ---------- 3. Python deps ----------
    Write-Host "==> Installing PyTorch ($(if ($Gpu) {'CUDA 12.1'} else {'CPU'}))" -ForegroundColor Cyan
    $torchIndex = if ($Gpu) { "https://download.pytorch.org/whl/cu121" } else { "https://download.pytorch.org/whl/cpu" }
    & $pyExe -m pip install --no-warn-script-location --index-url $torchIndex torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

    Write-Host "==> Installing project requirements" -ForegroundColor Cyan
    $req = Join-Path $Root "ml\requirements.txt"
    # Skip torch/torchvision/onnxruntime-gpu lines — torch already installed; onnxruntime-gpu only if -Gpu
    $tmpReq = Join-Path $Cache "requirements-bundle.txt"
    Get-Content $req | Where-Object {
        $_ -notmatch '^\s*torch(\s|>=|==|$)' -and
        $_ -notmatch '^\s*torchvision' -and
        ($Gpu -or $_ -notmatch '^\s*onnxruntime-gpu')
    } | Set-Content $tmpReq
    if (-not $Gpu) { Add-Content $tmpReq "onnxruntime>=1.19" }

    & $pyExe -m pip install --no-warn-script-location -r $tmpReq
    if ($LASTEXITCODE -ne 0) { throw "pip install -r failed" }
}

# ---------- 4. ffmpeg ----------
$FfZip = Join-Path $Cache "ffmpeg.zip"
if (-not (Test-Path $FfZip)) {
    Write-Host "==> Downloading ffmpeg"
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $FfZip
}
if (Test-Path $FfDir) { Remove-Item -Recurse -Force $FfDir }
New-Item -ItemType Directory -Path $FfDir | Out-Null
$tmpFf = Join-Path $Cache "ffmpeg_extract"
if (Test-Path $tmpFf) { Remove-Item -Recurse -Force $tmpFf }
Expand-Archive -Path $FfZip -DestinationPath $tmpFf -Force
$ffBin = Get-ChildItem $tmpFf -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
$fpBin = Get-ChildItem $tmpFf -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
if (-not $ffBin) { throw "ffmpeg.exe not found in archive" }
Copy-Item $ffBin.FullName  (Join-Path $FfDir "ffmpeg.exe")
Copy-Item $fpBin.FullName  (Join-Path $FfDir "ffprobe.exe")

# ---------- 5. App files ----------
Write-Host "==> Copying ml/, models/, UI" -ForegroundColor Cyan
$dirs = @("ml", "models")
foreach ($d in $dirs) {
    $src = Join-Path $Root $d
    $dst = Join-Path $Dist $d
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    # robocopy for speed, exclude caches
    & robocopy $src $dst /E /XD "__pycache__" ".pytest_cache" "tests" "corrections" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Null
}
Copy-Item (Join-Path $Root "tagger_ui.py")     $Dist
Copy-Item (Join-Path $Root "preflight.py")     $Dist
Copy-Item (Join-Path $Root "README.md")        $Dist

# ---------- 6. Launcher ----------
Copy-Item (Join-Path $PSScriptRoot "launcher.bat") (Join-Path $Dist "TennisTagger.bat") -Force

# ---------- 6b. Sentry DSN ----------
# Bake the DSN into a sentry.dsn file alongside the binary so the installed
# app reports crashes without each end user having to set env vars. If the
# DSN is empty, Sentry stays a no-op.
$dsnFile = Join-Path $Dist "sentry.dsn"
$dsn = $SentryDsn
if (-not $dsn) { $dsn = $env:TENNIS_TAGGER_SENTRY_DSN }
if ($dsn) {
    Write-Host "==> Writing sentry.dsn (telemetry enabled)" -ForegroundColor Cyan
    Set-Content -Path $dsnFile -Value $dsn.Trim() -Encoding ASCII -NoNewline
} elseif (Test-Path $dsnFile) {
    # Stale DSN from a previous build — remove so an opt-out build doesn't
    # silently re-enable telemetry.
    Remove-Item -Force $dsnFile
}

# ---------- 7. Sanity check ----------
# Smoke-test the bundled binary. We don't redirect stderr (`2>&1`) and we
# wrap the call so the non-zero exit from `--help` (Go's flag package
# returns 2) doesn't trip $ErrorActionPreference=Stop on the host.
Write-Host "==> Smoke test: tagger.exe --help" -ForegroundColor Cyan
$taggerExe = Join-Path $Dist "tagger.exe"
if (-not (Test-Path $taggerExe)) { throw "tagger.exe missing from bundle: $taggerExe" }
$prevPref = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $taggerExe --help | Select-Object -First 5
    # Exit code 0 or 2 (--help convention) is fine; anything else is a real error.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) {
        throw "tagger.exe smoke test returned unexpected exit code $LASTEXITCODE"
    }
    $global:LASTEXITCODE = 0
} finally {
    $ErrorActionPreference = $prevPref
}

$size = (Get-ChildItem $Dist -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("==> Bundle ready: {0:N0} MB at {1}" -f $size, $Dist) -ForegroundColor Green
Write-Host "    Next: compile packaging\installer.iss with Inno Setup 6 to produce the installer .exe"

# Packaging Tennis Tagger

Builds a Windows installer (.exe) that ships the Go binary, an embedded Python
runtime with all ML dependencies, ffmpeg, and the pre-trained models. The end
user just runs the installer — no Python, no pip, no PATH setup.

## One-time setup

1. **Inno Setup 6** — https://jrsoftware.org/isdl.php (`ISCC.exe` must be on PATH or invoke by full path).
2. **Go 1.26+** and a working `go build` for `./cmd/tagger`.
3. PowerShell 5.1+ (default on Windows 10/11).

## Build

From the project root (PowerShell — execution policy may need a one-time bypass):

```powershell
# If you hit "running scripts is disabled":
powershell -ExecutionPolicy Bypass -File .\packaging\build_bundle.ps1

# CPU-only torch (~600 MB installer, runs anywhere)
.\packaging\build_bundle.ps1

# CUDA 12.1 torch (~2.5 GB installer, requires NVIDIA GPU on user's machine)
.\packaging\build_bundle.ps1 -Gpu

# Iterating on the installer without re-running pip
.\packaging\build_bundle.ps1 -SkipPyDeps

# Bake a Sentry DSN into the bundle so crashes from end users are reported
.\packaging\build_bundle.ps1 -SentryDsn 'https://abc@o123.ingest.sentry.io/456'
# (or set $env:TENNIS_TAGGER_SENTRY_DSN before invoking — CI-friendly)
```

This produces `dist\TennisTagger\` with everything needed to run the app.
You can sanity-check by double-clicking `dist\TennisTagger\TennisTagger.bat`.

Then compile the installer:

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

Output: `dist\TennisTagger-Setup-0.1.0.exe`.

## What the installer does

- Lets the user pick an install directory (default `Program Files\TennisTagger`, falls back to `%LocalAppData%` if not admin).
- Copies the bundle.
- Creates Start Menu shortcut + optional desktop icon.
- Registers an uninstaller in Add/Remove Programs.
- Offers to launch the app on finish.

## Layout after install

```
TennisTagger\
  TennisTagger.bat       <- launcher (Start Menu / desktop shortcut points here)
  tagger.exe             <- Go pipeline binary
  tagger_ui.py           <- Tk GUI entry point
  preflight.py
  python\                <- embedded CPython 3.11 + site-packages
  ml\                    <- ML modules (bridge_server, tracknet, trajectory, ...)
  models\                <- *.pt weights + manifest.json
  ffmpeg\                <- ffmpeg.exe + ffprobe.exe
```

The launcher prepends `ffmpeg\` to PATH and points `PYTHONPATH` at the bundled
`ml\` dir, so the app is fully self-contained and can't conflict with any
system Python.

## CI builds (GitHub Actions)

`.github/workflows/release.yml` builds the installer automatically:

- **On every `v*` tag push** — produces a draft GitHub Release with the
  installer + SHA256 attached.
- **Manually** via Actions → "Build Windows Installer" → Run workflow
  (with optional `gpu` checkbox).

To cut a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then publish the draft release in the GitHub UI.

### Model weights in CI

Model `.pt` files are gitignored. The CI job needs them via one of:

1. **Git LFS** — migrate weights with `git lfs migrate import --include="models/*.pt"`,
   then push. The workflow auto-pulls LFS on checkout.
2. **`MODELS_BUNDLE_URL` repo secret** — set it to a URL serving a zip whose
   top-level layout matches `models/*.pt`. The workflow downloads and unpacks
   it before building. Good options: private GitHub Release asset, S3
   pre-signed URL, Backblaze B2.

Without one of these the workflow fails with a descriptive error.

## Bumping the version

Edit `MyAppVersion` in `installer.iss` and re-run the two build commands.

## Trimming the installer

The two biggest payloads are PyTorch (~200 MB CPU / ~2 GB CUDA) and the
`models\` directory (~200 MB). To slim the build:

- Drop unused weights from `models\` before running `build_bundle.ps1` — only
  `yastrebksv_tracknet.pt` and `yolov8s.pt` are required for the default run
  per `CLAUDE.md`.
- Strip pip caches: `python\Lib\site-packages\**\__pycache__` is already
  cleaned by the uninstaller; you can also delete them after the bundle step
  before compiling the installer.

## Code signing (optional)

Unsigned installers trip Windows SmartScreen. If you have a code-signing cert,
add to `[Setup]`:

```ini
SignTool=signtool sign /f "$qC:\path\cert.pfx$q" /p $qpassword$q /tr http://timestamp.digicert.com /td sha256 /fd sha256 $f
```

and register the tool with `ISCC /Ssigntool=...`.

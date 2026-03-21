# 🎾 Tennis Tagger - Desktop Mode Guide

## Overview

You now have **TWO modes** to run Tennis Tagger:

### 1. **Browser Mode** (Original)
- Opens in your web browser (Chrome, Edge, etc.)
- Runs on localhost

### 2. **Desktop Mode** (NEW!)
- Opens in a **native desktop window**
- No browser required
- Looks and feels like a regular desktop application
- Can be compiled to standalone EXE files

---

## 🚀 Quick Start - Desktop Mode

### Option A: Unified Launcher (Control Center)
**Double-click:** `start_desktop.bat`

This opens a **desktop window** with:
- ✅ Control center to start/stop services
- ✅ Training System button
- ✅ Tagging System button
- ✅ Live logs viewer
- ✅ One window controls everything!

**Port:** 7862

---

### Option B: Training Only (Desktop)
**Double-click:** `start_training_desktop.bat`

Opens **just the training system** in a desktop window:
- Train models
- Manage datasets
- Batch QC corrections
- Live visualization

**Port:** 7861

---

## 📦 Building Standalone EXE Files

Want to run on **any Windows PC without Python installed**?

### Step 1: Build the EXE
**Double-click:** `build_exe.bat`

This will:
- Install PyInstaller (if needed)
- Build 3 EXE files (takes 5-10 minutes)
- Save them to `dist/` folder

### Step 2: What Gets Built

You'll get:
1. **TennisTagger_Launcher.exe** - Control center (recommended!)
2. **TennisTagger_Training.exe** - Training system only
3. **TennisTagger_Tagging.exe** - Video tagging only

### Step 3: Deploy to Another PC

Copy these folders to the new PC:
```
📁 TennisTagger/
├── TennisTagger_Launcher.exe
├── TennisTagger_Training.exe
├── TennisTagger_Tagging.exe
├── 📁 config/
├── 📁 models/
└── 📁 data/
```

**Double-click the EXE** - no Python needed!

---

## 🎯 What's the Difference?

### Browser Mode vs Desktop Mode

| Feature | Browser Mode | Desktop Mode |
|---------|--------------|--------------|
| **Window Type** | Browser tab | Native desktop window |
| **Installation** | Python + packages | Python + pywebview |
| **Portable** | No | Yes (as EXE) |
| **Look & Feel** | Web interface | Desktop application |
| **Multiple Windows** | Multiple browser tabs | Multiple desktop windows |

**Both modes have the EXACT SAME features!** Desktop mode just wraps the interface in a native window.

---

## 🔧 Technical Details

### How It Works
1. **Backend**: Gradio server runs in background
2. **Frontend**: PyWebView embeds the web interface in a native window
3. **Communication**: Same as browser mode (HTTP to localhost)

### Dependencies
- **pywebview** - Creates native desktop windows
- **gradio** - Backend interface framework
- **PyInstaller** - Packages Python apps as EXE files

All installed automatically by the batch files!

---

## 🎮 Recommended Setup

### For Daily Use:
**Use:** `start_desktop.bat` (Unified Launcher)
- Opens one desktop window
- Control everything from there
- Cleanest experience

### For Distribution:
1. Run `build_exe.bat` once
2. Copy `dist/TennisTagger_Launcher.exe` + folders to USB drive
3. Run on any Windows PC!

---

## 📊 Comparison Table

### All Available Launchers

| Launcher | Mode | Window Type | What It Opens |
|----------|------|-------------|---------------|
| `start.bat` | Browser | Browser tab | Unified launcher (7862) |
| `start_training.bat` | Browser | Browser tab | Training only (7861) |
| `start_all_venv.bat` | Browser | 2 browser tabs | Training + Tagging |
| `start_desktop.bat` | Desktop | Desktop window | Unified launcher (7862) |
| `start_training_desktop.bat` | Desktop | Desktop window | Training only (7861) |

### EXE Files (After Building)

| EXE File | What It Is |
|----------|------------|
| `TennisTagger_Launcher.exe` | Control center (start/stop services) |
| `TennisTagger_Training.exe` | Training system v3.1 |
| `TennisTagger_Tagging.exe` | Video tagging system |

---

## 🐛 Troubleshooting

### "PyWebView not installed"
The batch files auto-install it. If manual install needed:
```bash
pip install pywebview
```

### "Window is blank/white"
Wait 5-10 seconds for the server to start. The window opens before the server is ready.

### "EXE build failed"
Make sure you have:
- Python 3.9+
- All packages installed (`pip install -r requirements.txt`)
- At least 2GB free disk space

### "EXE is huge (500MB+)"
Yes, that's normal. PyInstaller bundles Python + all libraries.

To reduce size:
- Use `--onefile` (already used)
- Don't include unnecessary packages
- Use UPX compression (advanced)

---

## ✅ Advantages of Desktop Mode

### Why Use Desktop Mode?

1. **Professional Look**: Native window, not a browser tab
2. **Better Experience**: Feels like a real application
3. **Portable**: Build once, run anywhere (as EXE)
4. **No Browser Clutter**: Separate from your web browsing
5. **Easier Distribution**: Give users an EXE file

### Why Use Browser Mode?

1. **Lighter**: No PyWebView dependency
2. **Debugging**: Browser dev tools available
3. **Familiar**: Everyone knows how to use browsers
4. **Cross-platform**: Works on Mac/Linux too

---

## 🎓 Summary

**Quick Start for You:**
```bash
# Desktop mode (recommended!)
start_desktop.bat

# Or just training
start_training_desktop.bat
```

**Deploy to Other PCs:**
```bash
# Build once
build_exe.bat

# Copy dist/ folder to other PCs
# Run TennisTagger_Launcher.exe
```

**That's it!** Both modes work identically - desktop mode just looks more professional! 🎾

# 🎾 Tennis Tagger - Quick Start Guide

## 🚀 Super Simple - Just Run These!

### **Want Both Apps?**
**Double-click:** `START_HERE.bat`

This opens **2 separate desktop windows**:
1. Training System v3.1 (port 7861)
2. Video Tagging System (port 7860)

Each runs independently - close either window anytime!

---

### **Want Only Training?**
**Double-click:** `start_training_desktop.bat`

Opens only the Training System in a desktop window.

---

### **Want Only Tagging?**
**Double-click:** `start_tagging_desktop.bat`

Opens only the Video Tagging System in a desktop window.

---

## 📦 Building Standalone EXE Files

Want to run on **any Windows PC without Python**?

**Double-click:** `build_exe.bat`

This creates 2 EXE files in the `dist/` folder:
- `TennisTagger_Training.exe`
- `TennisTagger_Tagging.exe`

Copy these + the `config/`, `models/`, and `data/` folders to any PC!

---

## 🎯 What You Get

### Training System v3.1
- ✅ Train all 3 tasks simultaneously (Stroke, Serve, Placement)
- ✅ Model versioning (v1→v2→v3) - no overwriting!
- ✅ Incremental learning with low learning rate
- ✅ Batch QC with accuracy grading
- ✅ Dataset merging from multiple PCs
- ✅ Live training visualization
- ✅ Training time estimates

### Video Tagging System
- ✅ Process tennis match videos
- ✅ Auto-detect serves, strokes, placements
- ✅ Generate Dartfish-compatible CSVs
- ✅ QC correction tools
- ✅ Batch processing

---

## 💡 Desktop vs Browser Mode

Both modes work identically, but desktop mode:
- ✅ Opens in a **native desktop window** (not browser)
- ✅ Looks like a real application
- ✅ Can be packaged as EXE
- ✅ Each app runs independently

---

## 🔧 First Time Setup

The batch files will automatically:
1. Activate your Python virtual environment
2. Install PyWebView (one-time)
3. Start the server in background
4. Open the desktop window

Just double-click and wait! Takes ~5-10 seconds first time.

---

## ❓ Troubleshooting

### "Port already in use"
Another instance is running. Close it first or use a different port.

### "Window is blank"
Wait 5-10 seconds for the server to start.

### "PyWebView error"
The batch file will auto-install it. If manual install needed:
```bash
pip install pywebview
```

---

## 📁 File Structure

```
📁 TennisTagger/
├── START_HERE.bat                    ⭐ Run this! (Both apps)
├── start_training_desktop.bat        (Training only)
├── start_tagging_desktop.bat         (Tagging only)
├── build_exe.bat                     (Build standalone EXE)
│
├── training_desktop.py               (Training app wrapper)
├── tagging_desktop.py                (Tagging app wrapper)
├── training_interface_production.py  (Training v3.1 backend)
│
├── 📁 models/                        (Your trained models)
│   └── 📁 versions/                  (Model backups)
├── 📁 data/
│   ├── 📁 datasets/                  (Training datasets)
│   └── 📁 qc_corrections/            (QC queue)
└── 📁 config/                        (Configuration files)
```

---

## ✅ That's It!

**Quick start:** Double-click `START_HERE.bat`

**Build EXE:** Double-click `build_exe.bat`

**Done!** 🎾

# Directory Creation Fix

## The Problem

When processing videos, you got an error:
```
Error: [Errno 2] No such file or directory: 'C:\Users\liamp\Downloads\files\logs\tennis_tagger.log'
```

This happened because the `logs` directory didn't exist when the system tried to create the log file.

## The Solution

I've implemented automatic directory creation in multiple places:

### 1. **Logging Setup** (`main.py:45-49`)
```python
# Ensure log directory exists
log_file = log_config.get('log_file', 'tennis_tagger.log')
log_path = Path(log_file)
if log_path.parent != Path('.'):
    log_path.parent.mkdir(parents=True, exist_ok=True)
```

Now the logging system automatically creates its directory before trying to write logs.

### 2. **Desktop App Startup** (`training_desktop.py:44-53`)
```python
# Ensure all required directories exist
print("Checking required directories...")
directories = [
    "logs", "data", "data/output", "data/training_pairs",
    "data/datasets", "data/feature_extraction", "cache",
    "models", "models/versions"
]
for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
print("✓ All directories ready")
```

The desktop app now creates all required directories on startup before anything else runs.

### 3. **Manual Setup Script** (`setup_directories.py`)
```bash
python setup_directories.py
```

If you ever encounter directory errors, run this script to create everything at once.

## Required Directories

The system needs these directories:

```
C:\Users\liamp\Downloads\files\
├── logs/                       # Log files
├── data/
│   ├── output/                 # Processed video outputs
│   ├── training_pairs/         # Video+CSV pairs for training
│   ├── datasets/               # Training datasets
│   └── feature_extraction/     # Extracted features
├── cache/                      # Temporary cache files
├── models/                     # Model files
│   └── versions/              # Model version backups
└── config/                     # Configuration files (already exists)
```

## What Changed

### Before:
```
User processes video
    ↓
System tries to create log file
    ↓
❌ ERROR: logs/ directory doesn't exist
    ↓
Process crashes
```

### After:
```
User starts desktop app
    ↓
System creates all directories (< 1 second)
    ↓
User processes video
    ↓
✅ Logs written successfully
    ↓
Processing continues
```

## How to Use

### Option 1: Normal Startup (Automatic)
Just run the desktop app as usual:
```bash
python training_desktop.py
```

All directories are created automatically during startup!

### Option 2: Manual Setup (if needed)
If you encounter directory errors:
```bash
python setup_directories.py
```

This manually creates all required directories.

## Testing

After these fixes:
1. ✅ Start `training_desktop.py`
2. ✅ Browse for your video
3. ✅ Upload matching CSV
4. ✅ Click "Submit Video+CSV Pair" or "Process Video"
5. ✅ Processing should start without directory errors

## Summary

**Problem:** Missing `logs` directory caused crashes
**Solution:** Automatic directory creation on startup
**Result:** No more directory-related errors!

All directories are now created automatically - you'll never see "No such file or directory" errors again.

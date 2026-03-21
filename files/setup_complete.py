#!/usr/bin/env python3
"""
Tennis Tagger - Complete Automated Setup
Fixes all import errors and installs all dependencies
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{YELLOW}{text}{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_step(text):
    print(f"{GREEN}➤ {text}{RESET}")

def check_file_exists(filepath):
    """Check if critical file exists"""
    if not os.path.exists(filepath):
        print_error(f"{filepath} not found!")
        print(f"{YELLOW}Please run this from your project directory.{RESET}")
        sys.exit(1)

def install_package(package):
    """Install a Python package"""
    print(f"  Installing {package}...", end=' ')
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"{GREEN}✓{RESET}")
        return True
    except:
        print(f"{YELLOW}⚠{RESET}")
        return False

def verify_package(package):
    """Verify a package is installed"""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def fix_main_py():
    """Fix all import mismatches in main.py"""
    print_header("STEP 2: Fixing Import Mismatches")
    
    main_py = "main.py"
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{main_py}.backup_{timestamp}"
    shutil.copy2(main_py, backup)
    print_success(f"Backup created: {backup}")
    print()
    
    # Read file
    with open(main_py, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # All fixes
    fixes = [
        {
            'old': "from detection.tracker import ObjectTracker",
            'new': "from detection.tracker import MultiObjectTracker as ObjectTracker",
            'name': "ObjectTracker -> MultiObjectTracker"
        },
        {
            'old': "from analysis.csv_generator import CSVGenerator",
            'new': "from analysis.csv_generator import DartfishCSVGenerator as CSVGenerator",
            'name': "CSVGenerator -> DartfishCSVGenerator"
        },
        {
            'old': "from analysis.comparator import TagComparator",
            'new': "from analysis.comparator import CSVComparator as TagComparator",
            'name': "TagComparator -> CSVComparator"
        },
        {
            'old': "from analysis.qc_feedback import FeedbackLoop",
            'new': "from analysis.comparator import FeedbackLoop",
            'name': "FeedbackLoop location fix"
        }
    ]
    
    fix_count = 0
    for fix in fixes:
        if fix['old'] in content:
            content = content.replace(fix['old'], fix['new'])
            print_success(f"Fixed: {fix['name']}")
            fix_count += 1
    
    # Save
    if fix_count > 0:
        with open(main_py, 'w', encoding='utf-8') as f:
            f.write(content)
        print()
        print_success(f"Applied {fix_count} fix(es) to main.py")
    else:
        print(f"{CYAN}  All imports already correct!{RESET}")
    
    return True

def main():
    print_header("TENNIS TAGGER - COMPLETE AUTOMATED SETUP")
    print(f"{CYAN}Project Directory: {os.getcwd()}{RESET}\n")
    
    # Check we're in the right place
    check_file_exists("main.py")
    check_file_exists("app.py")
    
    # ==========================================
    # STEP 1: Install Dependencies
    # ==========================================
    print_header("STEP 1: Installing All Dependencies")
    
    packages = [
        "gradio",
        "easyocr",
        "filterpy",
        "ffmpeg-python",
        "imageio",
        "imageio-ffmpeg",
        "opencv-contrib-python",
        "PyQt5"
    ]
    
    print_step("Installing missing packages...")
    print(f"{YELLOW}This may take 5-10 minutes...{RESET}\n")
    
    for pkg in packages:
        install_package(pkg)
    
    print()
    print_success("Dependency installation complete!")
    
    # ==========================================
    # STEP 2: Fix Imports
    # ==========================================
    fix_main_py()
    
    # ==========================================
    # STEP 3: Verify Setup
    # ==========================================
    print_header("STEP 3: Verifying Setup")
    
    print_step("Checking critical packages...")
    critical = ['torch', 'cv2', 'gradio', 'easyocr', 'filterpy', 'mediapipe']
    all_good = True
    
    for pkg in critical:
        print(f"  Checking {pkg}...", end=' ')
        if verify_package(pkg):
            print(f"{GREEN}✓{RESET}")
        else:
            print(f"{RED}✗{RESET}")
            all_good = False
    
    print()
    if all_good:
        print_success("All packages verified!")
    else:
        print_error("Some packages are missing. Install manually if needed.")
    
    # ==========================================
    # STEP 4: Set Python Path
    # ==========================================
    print_header("STEP 4: Setting Python Path")
    os.environ['PYTHONPATH'] = os.getcwd()
    print_success(f"PYTHONPATH = {os.getcwd()}")
    
    # ==========================================
    # STEP 5: Ready to Launch
    # ==========================================
    print_header("SETUP COMPLETE!")
    
    if all_good:
        print(f"{GREEN}✓ All checks passed!{RESET}")
        print()
        print(f"{YELLOW}To launch the app:{RESET}")
        print(f"{CYAN}  python app.py{RESET}")
        print()
        
        response = input("Launch now? (y/n): ").strip().lower()
        if response == 'y':
            print()
            print_header("Launching Tennis Tagger")
            print(f"{YELLOW}Press Ctrl+C to stop{RESET}\n")
            print(f"{CYAN}{'=' * 60}{RESET}\n")
            
            subprocess.call([sys.executable, "app.py"])
    else:
        print_error("Setup incomplete. Fix errors above before running.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Setup cancelled by user.{RESET}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

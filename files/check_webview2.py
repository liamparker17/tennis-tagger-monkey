"""
WebView2 Runtime Checker
Checks if Microsoft Edge WebView2 Runtime is installed
"""

import os
import sys
import platform
import subprocess

print("="*60)
print("  WEBVIEW2 RUNTIME CHECKER")
print("="*60)
print()

if platform.system() != 'Windows':
    print("This script is only for Windows systems.")
    sys.exit(0)

print("Checking for WebView2 Runtime installation...")
print()

# Check registry for WebView2
webview2_found = False
webview2_version = None

try:
    import winreg

    # Check the main installation location
    reg_paths = [
        r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    ]

    for reg_path in reg_paths:
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
            version, _ = winreg.QueryValueEx(key, "pv")
            winreg.CloseKey(key)
            webview2_found = True
            webview2_version = version
            break
        except:
            pass

except ImportError:
    print("⚠️  Warning: winreg module not available")
    print()

# Check if Edge is installed (comes with WebView2)
edge_found = False
edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

for path in edge_paths:
    if os.path.exists(path):
        edge_found = True
        print(f"✓ Microsoft Edge found at: {path}")
        break

print()

if webview2_found:
    print("="*60)
    print("✓ WEBVIEW2 RUNTIME IS INSTALLED")
    print("="*60)
    print()
    if webview2_version:
        print(f"Version: {webview2_version}")
    print()
    print("Your system should work with PyWebView!")
    print()

elif edge_found:
    print("="*60)
    print("⚠️  WEBVIEW2 STATUS UNCLEAR")
    print("="*60)
    print()
    print("Microsoft Edge is installed, which usually includes WebView2.")
    print("If PyWebView still doesn't work, you may need to install")
    print("the standalone WebView2 Runtime.")
    print()

else:
    print("="*60)
    print("❌ WEBVIEW2 RUNTIME NOT FOUND")
    print("="*60)
    print()
    print("WebView2 Runtime is required for PyWebView to work properly")
    print("on Windows with the EdgeChromium backend.")
    print()
    print("SOLUTION:")
    print()
    print("1. Download the WebView2 Runtime Installer:")
    print()
    print("   https://go.microsoft.com/fwlink/p/?LinkId=2124703")
    print()
    print("2. Run the installer (it's a small download, ~100MB)")
    print()
    print("3. After installation, your PyWebView apps should work!")
    print()

    try:
        import webbrowser
        response = input("Would you like to open the download page now? (y/n): ")
        if response.lower() == 'y':
            webbrowser.open('https://go.microsoft.com/fwlink/p/?LinkId=2124703')
            print()
            print("Opening browser... Download and install WebView2 Runtime.")
    except:
        pass

print()

# Additional diagnostic info
print("-"*60)
print("ADDITIONAL SYSTEM INFO")
print("-"*60)
print(f"OS: {platform.system()} {platform.version()}")
print(f"Architecture: {platform.architecture()[0]}")
print(f"Python: {sys.version.split()[0]}")

# Check if pywebview is installed
try:
    import webview
    print(f"PyWebView: Installed (version: {getattr(webview, '__version__', 'unknown')})")
except ImportError:
    print("PyWebView: NOT INSTALLED")

print()
print("="*60)

input("\nPress Enter to exit...")

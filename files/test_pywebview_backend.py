"""
PyWebView Backend Tester
This script tests which backend PyWebView will use and if it works
"""

import sys
import platform

print("="*60)
print("  PYWEBVIEW BACKEND DIAGNOSTICS")
print("="*60)
print()

# System info
print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.version()}")
print(f"Architecture: {platform.architecture()}")
print()

# Test pywebview import
try:
    import webview
    print("✓ PyWebView imported successfully")
    print(f"  PyWebView version: {webview.__version__ if hasattr(webview, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ Failed to import PyWebView: {e}")
    sys.exit(1)

print()

# Test which backend will be used
print("-"*60)
print("Testing available backends...")
print("-"*60)

# On Windows, pywebview tries these in order:
# 1. edgechromium (best - modern Chromium-based Edge)
# 2. edgehtml (older Edge)
# 3. mshtml (Internet Explorer - worst)

if platform.system() == 'Windows':
    print("\nWindows detected. Testing backends:")
    print()

    # Test EdgeChromium
    try:
        from webview.platforms.winforms import EdgeChromium
        print("✓ EdgeChromium (modern Edge) - AVAILABLE")
        print("  This is the best backend for Windows 10/11")
    except Exception as e:
        print(f"✗ EdgeChromium - NOT AVAILABLE: {e}")

    # Test EdgeHTML
    try:
        from webview.platforms.winforms import EdgeHTML
        print("✓ EdgeHTML (legacy Edge) - AVAILABLE")
    except Exception as e:
        print(f"✗ EdgeHTML - NOT AVAILABLE: {e}")

    # Test MSHTML
    try:
        from webview.platforms.winforms import MSHTML
        print("✓ MSHTML (Internet Explorer) - AVAILABLE")
        print("  Warning: This is the worst backend, may have issues")
    except Exception as e:
        print(f"✗ MSHTML - NOT AVAILABLE: {e}")

print()
print("-"*60)
print("Creating test window...")
print("-"*60)
print()

# Create a simple test window
try:
    print("Opening a test window with a simple webpage...")
    print("Close the window to continue.")
    print()

    window = webview.create_window(
        'PyWebView Backend Test',
        html='''
        <!DOCTYPE html>
        <html>
        <head>
            <title>PyWebView Test</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                }
                h1 { margin: 0 0 20px 0; }
                p { font-size: 18px; line-height: 1.6; }
                .success { color: #4ade80; font-size: 48px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success">✓</div>
                <h1>PyWebView is Working!</h1>
                <p>If you can see this window with styling,<br>
                   your PyWebView backend is working correctly.</p>
                <p style="margin-top: 20px; font-size: 14px; opacity: 0.8;">
                    Close this window to continue the test.
                </p>
            </div>
        </body>
        </html>
        ''',
        width=600,
        height=400,
        resizable=True
    )

    # Start with debug enabled
    webview.start(debug=True)

    print()
    print("="*60)
    print("✓ TEST SUCCESSFUL!")
    print("="*60)
    print()
    print("If you saw a styled window, PyWebView is working correctly.")
    print("You should now be able to run your desktop apps.")
    print()

except Exception as e:
    print()
    print("="*60)
    print("❌ TEST FAILED!")
    print("="*60)
    print()
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Possible solutions:")
    print("1. Install WebView2 Runtime:")
    print("   https://go.microsoft.com/fwlink/p/?LinkId=2124703")
    print()
    print("2. Try installing pywebview with winforms backend:")
    print("   pip install pywebview[winforms]")
    print()
    print("3. Update pywebview:")
    print("   pip install --upgrade pywebview")
    print()

input("Press Enter to exit...")

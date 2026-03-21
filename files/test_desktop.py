"""
Simple test to see if PyWebView works
"""

import sys
import time

print("="*60)
print("TESTING DESKTOP MODE")
print("="*60)
print()

# Test 1: Import webview
print("[1/4] Testing PyWebView import...")
try:
    import webview
    print("✓ PyWebView imported successfully")
    if hasattr(webview, '__version__'):
        print(f"  Version: {webview.__version__}")
except Exception as e:
    print(f"✗ Failed to import PyWebView: {e}")
    print("\nInstalling PyWebView...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pywebview"])
    import webview
    print("✓ PyWebView installed and imported")
print()

# Test 2: Import gradio
print("[2/4] Testing Gradio import...")
try:
    import gradio as gr
    print(f"✓ Gradio imported successfully (v{gr.__version__})")
except Exception as e:
    print(f"✗ Failed to import Gradio: {e}")
    sys.exit(1)
print()

# Test 3: Create simple Gradio app
print("[3/4] Creating test Gradio app...")
try:
    def greet(name):
        return f"Hello {name}!"

    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    print("✓ Gradio app created")
except Exception as e:
    print(f"✗ Failed to create Gradio app: {e}")
    sys.exit(1)
print()

# Test 4: Launch in desktop window
print("[4/4] Testing desktop window...")
print("A desktop window should open in 3 seconds...")
print("Close the window to complete the test.")
print()

def start_server():
    """Start Gradio server"""
    demo.launch(
        server_name="127.0.0.1",
        server_port=7999,
        share=False,
        prevent_thread_lock=False,
        inbrowser=False,
        quiet=True
    )

# Start server in background
import threading
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait for server
time.sleep(3)

try:
    # Create window
    window = webview.create_window(
        'Test Desktop Window',
        'http://127.0.0.1:7999',
        width=800,
        height=600
    )

    print("✓ Desktop window created")
    print("\n" + "="*60)
    print("If you see a window, desktop mode works!")
    print("Close the window to finish the test.")
    print("="*60)

    # Start GUI (blocking)
    webview.start()

    print("\n✓ Test completed successfully!")
    print("Desktop mode is working!")

except Exception as e:
    print(f"\n✗ Desktop window failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nPress Enter to exit...")
    input()

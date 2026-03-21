"""
Tennis Tagger - Training Desktop App
Native desktop window for training interface - Runs independently!
"""

import webview
import threading
import time
import sys
import socket
import requests


def is_port_available(port):
    """Check if port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result != 0
    except:
        return True


def wait_for_server(url, timeout=30):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def start_training_server():
    """Start training interface server"""
    from pathlib import Path

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

    from training_interface_production import create_training_interface

    print("Starting training interface server on port 7861...")

    app = create_training_interface()

    # Launch server (will run in this thread)
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        prevent_thread_lock=False,
        inbrowser=False,
        quiet=False  # Show startup messages
    )


def main():
    """Launch training system in desktop window"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║   🎾 TRAINING SYSTEM v3.1 - STANDALONE DESKTOP APP           ║
║                                                              ║
║   Multi-Task • Incremental • Versioning • Batch QC          ║
╚══════════════════════════════════════════════════════════════╝

Starting training system...

Features:
✓ Train all 3 tasks simultaneously
✓ Model versioning (v1→v2→v3)
✓ Incremental learning (no overwriting!)
✓ Batch QC with accuracy tracking
✓ Dataset merging from multiple PCs
✓ Live training visualization
    """)

    # Check if port is already in use
    if not is_port_available(7861):
        print("\n⚠️  WARNING: Port 7861 is already in use!")
        print("Another instance may be running.")
        print("\nConnecting to existing instance...")
        time.sleep(2)
    else:
        print("\n[*] Port 7861 is available")

    # Check if pywebview is installed
    try:
        import webview
    except ImportError:
        print("\n❌ PyWebView not installed!")
        print("\nInstalling pywebview...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywebview"])
        print("✅ PyWebView installed! Please restart the application.")
        input("\nPress Enter to exit...")
        return

    print("[*] Starting server in background...")

    # Start server in background thread
    server_thread = threading.Thread(target=start_training_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    print("[*] Waiting for server to initialize...")
    server_url = 'http://127.0.0.1:7861'

    if wait_for_server(server_url):
        print("[OK] Server is ready!")
    else:
        print("\n❌ ERROR: Server failed to start within 30 seconds")
        print("Check if there are any errors above.")
        input("\nPress Enter to exit...")
        return

    print("[*] Opening desktop window...")

    # Create desktop window with proper API backend for Windows
    try:
        # Detect best available backend
        print("[*] Detecting PyWebView backend...")

        window = webview.create_window(
            title='🎾 Tennis Tagger - Training System v3.1',
            url=server_url,
            width=1600,
            height=1000,
            resizable=True,
            fullscreen=False,
            min_size=(1200, 800),
            confirm_close=True
        )

        print("[OK] Desktop window created!")
        print("\n" + "="*60)
        print("  TRAINING SYSTEM IS NOW RUNNING")
        print("="*60)
        print("\nClose the window to stop the application.")
        print("")

        # Start the GUI with debug enabled (blocking call)
        # On Windows, this will try: edgechromium -> edgehtml -> mshtml
        webview.start(debug=True)

        print("\n[*] Window closed. Shutting down...")

    except KeyboardInterrupt:
        print("\n\n[*] Interrupted. Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()

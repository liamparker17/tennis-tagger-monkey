"""
Tennis Tagger - Tagging Desktop App
Native desktop window for video tagging interface - Runs independently!
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


def start_tagging_server():
    """Start tagging interface server"""
    try:
        print("Starting tagging interface server on port 7860...")

        # Import and create the Gradio interface
        from app import TennisGUI

        gui = TennisGUI()
        interface = gui.create_interface()

        # Launch server
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            prevent_thread_lock=False,
            inbrowser=False,
            quiet=False
        )

    except Exception as e:
        print(f"Error starting tagging server: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Launch tagging system in desktop window"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║   🎾 VIDEO TAGGING SYSTEM - STANDALONE DESKTOP APP           ║
║                                                              ║
║   Auto-tag tennis matches with AI                           ║
╚══════════════════════════════════════════════════════════════╝

Starting video tagging system...

Features:
✓ Process tennis match videos
✓ Auto-detect serves, strokes, placements
✓ Generate Dartfish-compatible CSVs
✓ QC correction tools
✓ Batch processing
    """)

    # Check if port is already in use
    if not is_port_available(7860):
        print("\n⚠️  WARNING: Port 7860 is already in use!")
        print("Another instance may be running.")
        print("\nConnecting to existing instance...")
        time.sleep(2)
    else:
        print("\n[*] Port 7860 is available")

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
    server_thread = threading.Thread(target=start_tagging_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    print("[*] Waiting for server to initialize...")
    server_url = 'http://127.0.0.1:7860'

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
            title='🎾 Tennis Tagger - Video Tagging System',
            url=server_url,
            width=1400,
            height=900,
            resizable=True,
            fullscreen=False,
            min_size=(1000, 700),
            confirm_close=True
        )

        print("[OK] Desktop window created!")
        print("\n" + "="*60)
        print("  VIDEO TAGGING SYSTEM IS NOW RUNNING")
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

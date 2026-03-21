"""
Tennis Tagger - Unified Launcher
Manages all services from one clean interface
"""

import gradio as gr
import subprocess
import sys
import time
import psutil
import signal
from pathlib import Path
from datetime import datetime
import threading


class ServiceManager:
    """Manage Tennis Tagger services"""
    
    def __init__(self):
        self.processes = {}
        self.logs = {
            'main': [],
            'training': []
        }
    
    def start_service(self, name, script, port):
        """Start a service"""
        if name in self.processes and self.processes[name].poll() is None:
            return f"⚠️ {name} already running"
        
        try:
            # Start process with output capture
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes[name] = process
            
            # Start log monitoring thread
            threading.Thread(
                target=self._monitor_logs,
                args=(name, process),
                daemon=True
            ).start()
            
            return f"✅ {name} started on port {port}"
            
        except Exception as e:
            return f"❌ Failed to start {name}: {str(e)}"
    
    def stop_service(self, name):
        """Stop a service"""
        if name not in self.processes:
            return f"⚠️ {name} not running"
        
        process = self.processes[name]
        
        if process.poll() is not None:
            return f"⚠️ {name} already stopped"
        
        try:
            # Graceful shutdown
            process.terminate()
            process.wait(timeout=5)
            return f"✅ {name} stopped"
        except subprocess.TimeoutExpired:
            # Force kill if needed
            process.kill()
            return f"✅ {name} force stopped"
    
    def get_status(self, name):
        """Get service status"""
        if name not in self.processes:
            return "⚫ Not started"
        
        if self.processes[name].poll() is None:
            return "🟢 Running"
        else:
            return "🔴 Stopped"
    
    def get_logs(self, name):
        """Get service logs"""
        return "\n".join(self.logs.get(name, ["No logs yet"]))
    
    def _monitor_logs(self, name, process):
        """Monitor process logs"""
        for line in process.stdout:
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_line = f"[{timestamp}] {line.rstrip()}"
            self.logs.setdefault(name, []).append(log_line)
            
            # Keep only last 100 lines
            if len(self.logs[name]) > 100:
                self.logs[name] = self.logs[name][-100:]


def create_launcher():
    """Create unified launcher interface"""
    
    manager = ServiceManager()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Tennis Tagger - Launcher") as app:
        
        gr.Markdown("""
        # 🎾 Tennis Tagger - System Launcher
        ## Control all services from one place
        """)
        
        with gr.Tabs():
            
            # ==================== TAB 1: SERVICES ====================
            with gr.Tab("🚀 Services") as services_tab:
                
                gr.Markdown("### Service Control")
                
                # Main App
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🎬 Main App (Video Processing)")
                        main_status = gr.Markdown("⚫ Not started")
                        gr.Markdown("*Process videos, generate CSVs, QC corrections*")
                    
                    with gr.Column(scale=1):
                        main_start_btn = gr.Button("▶️ Start", variant="primary")
                        main_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                        main_open_btn = gr.Button("🌐 Open (7860)", variant="secondary")
                
                gr.Markdown("---")
                
                # Training App
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🎓 Training System")
                        training_status = gr.Markdown("⚫ Not started")
                        gr.Markdown("*Train models, manage datasets, batch QC*")
                    
                    with gr.Column(scale=1):
                        training_start_btn = gr.Button("▶️ Start", variant="primary")
                        training_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                        training_open_btn = gr.Button("🌐 Open (7861)", variant="secondary")
                
                gr.Markdown("---")
                
                # Quick Actions
                gr.Markdown("### Quick Actions")
                
                with gr.Row():
                    start_all_btn = gr.Button("▶️ Start All Services", variant="primary", size="lg")
                    stop_all_btn = gr.Button("⏹️ Stop All Services", variant="secondary", size="lg")
                
                system_status = gr.Markdown("System idle")
            
            # ==================== TAB 2: LOGS ====================
            with gr.Tab("📋 Logs") as logs_tab:
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Main App Logs")
                        main_logs = gr.Textbox(
                            label="Main App Output",
                            lines=20,
                            interactive=False,
                            max_lines=20
                        )
                        refresh_main_logs_btn = gr.Button("🔄 Refresh")
                    
                    with gr.Column():
                        gr.Markdown("### Training System Logs")
                        training_logs = gr.Textbox(
                            label="Training Output",
                            lines=20,
                            interactive=False,
                            max_lines=20
                        )
                        refresh_training_logs_btn = gr.Button("🔄 Refresh")
            
            # ==================== TAB 3: SETTINGS ====================
            with gr.Tab("⚙️ Settings") as settings_tab:
                
                gr.Markdown("### System Settings")
                
                gr.Markdown("#### Service Ports")
                main_port = gr.Textbox(label="Main App Port", value="7860", interactive=False)
                training_port = gr.Textbox(label="Training Port", value="7861", interactive=False)
                
                gr.Markdown("#### Auto-Start")
                auto_start_main = gr.Checkbox(label="Auto-start Main App on launch", value=False)
                auto_start_training = gr.Checkbox(label="Auto-start Training on launch", value=False)
                
                gr.Markdown("#### Terminal Access")
                gr.Markdown("""
                View service logs in the **Logs** tab.
                Terminals run in background - no popup windows!
                """)
                
                gr.Markdown("---")
                
                gr.Markdown("### System Information")
                
                system_info = gr.Markdown(f"""
**Python**: {sys.version.split()[0]}
**Working Directory**: {Path.cwd()}
**Services**: Main App, Training System
                """)
        
        # ==================== FUNCTIONS ====================
        
        def start_main():
            result = manager.start_service('main', 'app.py', 7860)
            time.sleep(2)
            status = manager.get_status('main')
            return result, status
        
        def stop_main():
            result = manager.stop_service('main')
            status = manager.get_status('main')
            return result, status
        
        def start_training():
            result = manager.start_service('training', 'training_interface_production.py', 7861)
            time.sleep(2)
            status = manager.get_status('training')
            return result, status
        
        def stop_training():
            result = manager.stop_service('training')
            status = manager.get_status('training')
            return result, status
        
        def start_all():
            main_result = manager.start_service('main', 'app.py', 7860)
            training_result = manager.start_service('training', 'training_interface_production.py', 7861)
            time.sleep(2)
            main_st = manager.get_status('main')
            training_st = manager.get_status('training')
            
            status = f"{main_result}\n{training_result}"
            return status, main_st, training_st
        
        def stop_all():
            main_result = manager.stop_service('main')
            training_result = manager.stop_service('training')
            main_st = manager.get_status('main')
            training_st = manager.get_status('training')
            
            status = f"{main_result}\n{training_result}"
            return status, main_st, training_st
        
        def open_main():
            import webbrowser
            webbrowser.open('http://localhost:7860')
            return "Opened Main App in browser"
        
        def open_training():
            import webbrowser
            webbrowser.open('http://localhost:7861')
            return "Opened Training System in browser"
        
        def refresh_main_logs():
            return manager.get_logs('main')
        
        def refresh_training_logs():
            return manager.get_logs('training')
        
        # ==================== CONNECT EVENTS ====================
        
        main_start_btn.click(
            start_main,
            outputs=[system_status, main_status]
        )
        
        main_stop_btn.click(
            stop_main,
            outputs=[system_status, main_status]
        )
        
        main_open_btn.click(
            open_main,
            outputs=[system_status]
        )
        
        training_start_btn.click(
            start_training,
            outputs=[system_status, training_status]
        )
        
        training_stop_btn.click(
            stop_training,
            outputs=[system_status, training_status]
        )
        
        training_open_btn.click(
            open_training,
            outputs=[system_status]
        )
        
        start_all_btn.click(
            start_all,
            outputs=[system_status, main_status, training_status]
        )
        
        stop_all_btn.click(
            stop_all,
            outputs=[system_status, main_status, training_status]
        )
        
        refresh_main_logs_btn.click(
            refresh_main_logs,
            outputs=[main_logs]
        )
        
        refresh_training_logs_btn.click(
            refresh_training_logs,
            outputs=[training_logs]
        )
    
    return app


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        🎾 TENNIS TAGGER - UNIFIED LAUNCHER                   ║
║                                                              ║
║    Control all services from one clean interface            ║
╚══════════════════════════════════════════════════════════════╝

Starting launcher...

Open your browser to: http://localhost:7862

From there you can:
- Start/stop services
- View logs
- Open apps in new tabs

No more terminal spam! 🎉
    """)
    
    app = create_launcher()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        prevent_thread_lock=False
    )

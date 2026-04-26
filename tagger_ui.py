"""Tennis Tagger — walkthrough launcher.

One question per screen. The user never sees more than one or two
choices at a time. Power-user tools (training, label generation) are
tucked behind a small "Advanced" link on the home screen.

Run with:
    pythonw tagger_ui.py
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import END, filedialog, messagebox, simpledialog, ttk


# Prefer console-attached python.exe over pythonw.exe for child processes so
# stdout/stderr is captured reliably by the log pipe. Also force unbuffered
# I/O via -u — pythonw-launched children otherwise block-buffer stdout and
# their output (including tracebacks) vanishes when they crash.
def _child_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.is_file():
            return str(sibling)
    return sys.executable


PYEXE = _child_python()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _estimate_total_points(pairs_root: Path) -> int:
    """Rough total: body rows across every Dartfish CSV under each match."""
    total = 0
    if not pairs_root.is_dir():
        return 0
    for match in pairs_root.iterdir():
        if not match.is_dir():
            continue
        for csv in match.glob("*.csv"):
            try:
                with csv.open("r", encoding="utf-8", errors="replace") as f:
                    n = sum(1 for _ in f)
                total += max(0, n - 1)  # minus header
            except OSError:
                pass
    return total


def _points_in_match(match_dir: Path) -> int:
    for csv in match_dir.glob("*.csv"):
        try:
            with csv.open("r", encoding="utf-8", errors="replace") as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            continue
    return 0


def _count_clips(clips_root: Path) -> int:
    if not clips_root.is_dir():
        return 0
    return sum(1 for _ in clips_root.glob("*/p_*.mp4"))


_EPOCH_RE = __import__("re").compile(r"^ep (\d+)", __import__("re").MULTILINE)

def _last_epoch_from_log(text: str) -> int:
    last = -1
    for m in _EPOCH_RE.finditer(text):
        try:
            last = max(last, int(m.group(1)))
        except ValueError:
            pass
    return last

REPO = Path(__file__).resolve().parent
TAGGER_EXE = REPO / "tagger.exe"
PREFLIGHT_SCRIPT = REPO / "preflight.py"
PSEUDO_SCRIPT = REPO / "ml" / "generate_pseudo_labels.py"
DARTFISH_SCRIPT = REPO / "ml" / "dartfish_to_yolo.py"
TRAIN_SCRIPT = REPO / "ml" / "train_yolo_ball.py"
TRACKNET_WEIGHTS = REPO / "models" / "tracknetv2_tennis_wasb.pt"
TRAINING_PAIRS_DIR = REPO / "files" / "data" / "training_pairs"

# ---- Palette ----
BG       = "#0f1419"
PANEL    = "#1a2028"
PANEL_HI = "#242c37"
ACCENT   = "#c8ff3e"
ACCENT_D = "#a8dc22"
TEXT     = "#e7edf2"
TEXT_DIM = "#8b95a1"
DANGER   = "#ff5b5b"
OK       = "#55d38a"
BORDER   = "#2a3240"

FONT_H1       = ("Segoe UI Semibold", 26)
FONT_QUESTION = ("Segoe UI Semibold", 22)
FONT_OPTION   = ("Segoe UI Semibold", 15)
FONT_SUB      = ("Segoe UI", 12)
FONT_BODY     = ("Segoe UI", 11)
FONT_LOG      = ("Consolas", 10)


class OptionButton(tk.Frame):
    """A big clickable card used as a wizard option."""

    def __init__(self, parent, title: str, subtitle: str, command, primary: bool = False) -> None:
        super().__init__(parent, bg=PANEL, bd=0, highlightthickness=2,
                         highlightbackground=ACCENT if primary else BORDER,
                         cursor="hand2")
        self._cmd = command
        self._primary = primary

        inner = tk.Frame(self, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=24, pady=20)

        self._title = tk.Label(inner, text=title, font=FONT_OPTION,
                               fg=TEXT, bg=PANEL, anchor="w")
        self._title.pack(fill="x")
        self._sub = tk.Label(inner, text=subtitle, font=FONT_SUB,
                             fg=TEXT_DIM, bg=PANEL, anchor="w",
                             wraplength=720, justify="left")
        self._sub.pack(fill="x", pady=(6, 0))

        for w in (self, inner, self._title, self._sub):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

    def _on_click(self, _e):
        self._cmd()

    def _on_enter(self, _e):
        self._paint(PANEL_HI, ACCENT)

    def _on_leave(self, _e):
        self._paint(PANEL, ACCENT if self._primary else BORDER)

    def _paint(self, bg, border):
        self.config(bg=bg, highlightbackground=border)
        for w in self.winfo_children():
            self._recolor(w, bg)

    def _recolor(self, widget, color):
        try:
            widget.config(bg=color)
        except tk.TclError:
            pass
        for c in widget.winfo_children():
            self._recolor(c, color)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Tennis Tagger")
        root.geometry("1050x820")
        root.minsize(900, 720)
        root.configure(bg=BG)

        self.proc: subprocess.Popen | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        # Wizard state (cleared when returning home)
        self.state: dict = {}

        self._build_style()
        self._build_chrome()

        self.show_home()
        self.root.after(100, self._pump_log)

    # ------------------------------------------------------------------
    # Static chrome (header + body + footer)
    # ------------------------------------------------------------------

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton",
                        font=("Segoe UI Semibold", 11),
                        foreground=BG, background=ACCENT,
                        borderwidth=0, focusthickness=0, padding=(18, 9))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_D), ("disabled", "#444")],
                  foreground=[("disabled", "#888")])
        style.configure("Ghost.TButton",
                        font=("Segoe UI", 11),
                        foreground=TEXT, background=BG,
                        borderwidth=0, focusthickness=0, padding=(14, 7))
        style.map("Ghost.TButton",
                  background=[("active", PANEL), ("disabled", BG)],
                  foreground=[("disabled", TEXT_DIM)])
        style.configure("Horizontal.TProgressbar",
                        background=ACCENT, troughcolor=PANEL,
                        borderwidth=0, lightcolor=ACCENT, darkcolor=ACCENT)

    def _build_chrome(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=36, pady=(24, 6))
        tk.Label(header, text="🎾  Tennis Tagger", font=FONT_H1,
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(header, text="v1 • beta", font=FONT_SUB,
                 fg=TEXT_DIM, bg=BG).pack(side="left", padx=(12, 0), pady=(10, 0))

        # Body frame — cleared & rebuilt by each show_* method
        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(fill="both", expand=True, padx=36, pady=(6, 0))

        # Footer
        self.footer = tk.Frame(self.root, bg=BG)
        self.footer.pack(fill="x", padx=36, pady=(6, 20))
        self.back_btn = ttk.Button(self.footer, text="← Back",
                                   style="Ghost.TButton",
                                   command=self._on_back)
        self.step_lbl = tk.Label(self.footer, text="", font=FONT_SUB,
                                 fg=TEXT_DIM, bg=BG)
        # (layout is set per-screen)

    def _clear_body(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()
        for w in self.footer.winfo_children():
            w.pack_forget()

    def _show_back(self, step_text: str = "") -> None:
        self.back_btn.pack(in_=self.footer, side="left")
        if step_text:
            self.step_lbl.config(text=step_text)
            self.step_lbl.pack(in_=self.footer, side="right", pady=(6, 0))

    def _step_bar(self, parent: tk.Widget, labels: list[str], current: int) -> None:
        """Draws a horizontal progress bar with one pill per step.

        `current` is 1-based. 0 = 'about to start', len(labels)+1 = 'done'.
        Completed steps get an accent fill; the current one gets a white
        border; upcoming ones are dim.
        """
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(0, 22))

        for i, label in enumerate(labels, start=1):
            cell = tk.Frame(bar, bg=BG)
            cell.pack(side="left", fill="x", expand=True)

            # Pill (dot + line)
            pill = tk.Frame(cell, bg=BG)
            pill.pack(fill="x")

            # Dot
            dot = tk.Canvas(pill, width=28, height=28, bg=BG, highlightthickness=0)
            dot.pack(side="left")
            if i < current:
                dot.create_oval(4, 4, 24, 24, fill=ACCENT, outline="")
                dot.create_text(14, 14, text="✓", fill=BG,
                                font=("Segoe UI Semibold", 11))
            elif i == current:
                dot.create_oval(4, 4, 24, 24, fill=ACCENT, outline=TEXT, width=2)
                dot.create_text(14, 14, text=str(i), fill=BG,
                                font=("Segoe UI Semibold", 11))
            else:
                dot.create_oval(4, 4, 24, 24, fill=PANEL, outline=BORDER, width=1)
                dot.create_text(14, 14, text=str(i), fill=TEXT_DIM,
                                font=("Segoe UI Semibold", 11))

            # Connector line to the next pill (not after the last)
            if i < len(labels):
                line = tk.Frame(pill, height=2,
                                bg=(ACCENT if i < current else BORDER))
                line.pack(side="left", fill="x", expand=True, padx=6, pady=13)

            # Label beneath the dot
            color = TEXT if i == current else (ACCENT if i < current else TEXT_DIM)
            tk.Label(cell, text=label, font=FONT_SUB, fg=color, bg=BG,
                     anchor="w").pack(fill="x", pady=(4, 0))

    def _next_up_hint(self, parent: tk.Widget, text: str) -> None:
        """Small muted line that tells the user what's coming next."""
        tk.Label(parent, text=f"↳ {text}", font=FONT_SUB,
                 fg=TEXT_DIM, bg=BG, anchor="w", wraplength=900, justify="left"
                 ).pack(fill="x", pady=(24, 0))

    def _on_back(self) -> None:
        # Stop anything running, then home.
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Stop?", "Something is still running. Stop it and go back?"):
                return
            try:
                self.proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        self.state.clear()
        self.show_home()

    # ------------------------------------------------------------------
    # Screens
    # ------------------------------------------------------------------

    def show_home(self) -> None:
        self._clear_body()
        self.state.clear()
        self.proc = None

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="What would you like to do?",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap, text="Pick one.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(0, 24))

        OptionButton(
            wrap,
            title="Tag a new match",
            subtitle="Pick a match video and let Tennis Tagger watch it. "
                     "It will produce a Dartfish-style CSV next to the video.",
            command=self.show_tag_pick_video, primary=True,
        ).pack(fill="x", pady=8)

        OptionButton(
            wrap,
            title="Add a match I've already tagged by hand",
            subtitle="Add a video together with its existing Dartfish CSV. "
                     "Used to grow the training set so the tagger gets smarter over time.",
            command=self.show_add_intro,
        ).pack(fill="x", pady=8)

        last_ckpt = REPO / "files" / "models" / "point_model" / "current" / "last.pt"
        n_matches = sum(1 for p in TRAINING_PAIRS_DIR.iterdir() if p.is_dir()) \
            if TRAINING_PAIRS_DIR.is_dir() else 0
        if n_matches > 0:
            if last_ckpt.exists():
                title = "Resume point-model training"
                subtitle = (f"Picks up from the last saved epoch. "
                            f"{n_matches} match{'es' if n_matches != 1 else ''} "
                            f"in the training folder.")
            else:
                title = "Train the point model"
                subtitle = (f"Cuts clips, extracts features, and trains from "
                            f"{n_matches} match{'es' if n_matches != 1 else ''} "
                            f"already in the training folder. Takes hours — "
                            f"good to start before you step away.")
            OptionButton(
                wrap, title=title, subtitle=subtitle,
                command=self._begin_point_model_training,
            ).pack(fill="x", pady=8)

        # Small 'Advanced' link in the footer area
        adv = tk.Frame(wrap, bg=BG)
        adv.pack(fill="x", pady=(24, 0))
        tk.Label(adv, text="Advanced:", font=FONT_SUB, fg=TEXT_DIM, bg=BG
                 ).pack(side="left")
        ttk.Button(adv, text="Build labels + train ball finder", style="Ghost.TButton",
                   command=self.show_adv_labels_then_train).pack(side="left", padx=(8, 0))
        ttk.Button(adv, text="Manage training set", style="Ghost.TButton",
                   command=self.show_manage_training).pack(side="left", padx=(4, 0))
        ttk.Button(adv, text="Share with a friend (USB)", style="Ghost.TButton",
                   command=self.show_share_hub).pack(side="left", padx=(4, 0))

    # ---- Path A: Tag a new match ----

    def show_tag_pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick the match video to tag",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if not path:
            return  # stay on home
        self.state["video"] = path
        sidecar = Path(path + ".setup.json")
        if sidecar.exists():
            self.show_tag_running()
        else:
            self.show_tag_needs_setup()

    def show_tag_needs_setup(self) -> None:
        self._clear_body()
        self._show_back(step_text="Step 1 of 2 — Set up the match")
        video = self.state["video"]

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="First — set this match up",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text=f"{Path(video).name}\n\n"
                      "This video has never been set up. Setup takes about 30 seconds: "
                      "click the 4 corners of the court and type the two players' names. "
                      "It makes the tagging much more accurate.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 20))

        OptionButton(
            wrap,
            title="Open the setup window",
            subtitle="A separate window opens. When you're done, come back here and press continue.",
            command=self._on_open_setup, primary=True,
        ).pack(fill="x", pady=6)

        OptionButton(
            wrap,
            title="Skip setup (less accurate)",
            subtitle="Tag the match anyway. The court and players will be guessed automatically.",
            command=self.show_tag_running,
        ).pack(fill="x", pady=6)

        # After setup is done in the other window, this button advances.
        self._setup_continue_btn = ttk.Button(
            wrap, text="I finished setup — continue",
            style="Accent.TButton",
            command=self._on_setup_continue, state="disabled",
        )
        self._setup_continue_btn.pack(anchor="w", pady=(20, 0))

    def _on_open_setup(self) -> None:
        video = self.state["video"]
        if not PREFLIGHT_SCRIPT.exists():
            messagebox.showerror("Missing file", "preflight.py is missing.")
            return
        try:
            subprocess.Popen([sys.executable, str(PREFLIGHT_SCRIPT), video], cwd=str(REPO))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Couldn't open setup", str(exc))
            return
        # Enable the continue button and start polling for the sidecar
        self._setup_continue_btn.config(state="normal")
        self._poll_sidecar()

    def _poll_sidecar(self) -> None:
        video = self.state.get("video")
        if not video:
            return
        if Path(video + ".setup.json").exists():
            self._setup_continue_btn.config(
                text="✓ Setup saved — continue",
                style="Accent.TButton",
            )
            return
        self.root.after(1000, self._poll_sidecar)

    def _on_setup_continue(self) -> None:
        if not Path(self.state["video"] + ".setup.json").exists():
            messagebox.showwarning(
                "Not saved yet",
                "I can't find the setup file. Finish the setup window and press Save there first.",
            )
            return
        self.show_tag_running()

    def show_tag_running(self) -> None:
        if not TAGGER_EXE.exists():
            messagebox.showerror("Can't start", "tagger.exe is missing.")
            self.show_home()
            return
        video = self.state["video"]
        self._run_screen(
            title="Tagging the match…",
            subtitle=f"{Path(video).name}\n\nThis can take a minute or two for a short clip, "
                     "longer for a full match. You can switch away — it keeps running.",
            argv=[str(TAGGER_EXE), video],
            on_success=lambda: self.show_tag_done(video + "_output.csv"),
            step_text="Step 2 of 2 — Tagging",
        )

    def show_tag_done(self, csv_path: str) -> None:
        self._clear_body()
        self._show_back()
        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="Done — the match is tagged",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap, text=f"Saved to:\n{csv_path}",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 20))

        OptionButton(
            wrap, title="Open the CSV",
            subtitle="Opens the file in whatever you usually open spreadsheets with.",
            command=lambda: self._open_path(csv_path), primary=True,
        ).pack(fill="x", pady=6)

        OptionButton(
            wrap, title="Tag another match",
            subtitle="Back to the home screen.",
            command=self.show_home,
        ).pack(fill="x", pady=6)

    # ---- Path B: Add a pre-tagged match ----

    ADD_STEPS = ["Video", "Tags", "Confirm"]

    def show_add_intro(self) -> None:
        """Preview all three steps up front so the tagger knows what's coming."""
        self._clear_body()
        self._show_back(step_text="Getting started")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        # step 0 = intro, so nothing is "current" yet — highlight step 1.
        self._step_bar(wrap, self.ADD_STEPS, current=0)

        tk.Label(wrap, text="Here's what we'll do together",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text="Three quick steps. You'll add BOTH the video and the tags file — "
                      "don't worry, the tags file comes right after the video.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 24))

        steps = [
            ("1", "Pick the match video",
             "The .mp4 of the match."),
            ("2", "Pick the Dartfish tags",
             "The .csv you exported from Dartfish for this same match."),
            ("3", "Confirm and add",
             "We copy both files into the training folder."),
        ]
        for num, title, sub in steps:
            self._step_preview_row(wrap, num, title, sub)

        ttk.Button(wrap, text="Start →", style="Accent.TButton",
                   command=self.show_add_pick_video
                   ).pack(anchor="w", pady=(24, 0))

    def _step_preview_row(self, parent, num: str, title: str, sub: str) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=num, font=("Segoe UI Black", 18),
                 fg=ACCENT, bg=BG, width=2
                 ).pack(side="left", padx=(0, 14))
        col = tk.Frame(row, bg=BG)
        col.pack(side="left", fill="x", expand=True)
        tk.Label(col, text=title, font=FONT_OPTION, fg=TEXT, bg=BG,
                 anchor="w").pack(fill="x")
        tk.Label(col, text=sub, font=FONT_SUB, fg=TEXT_DIM, bg=BG,
                 anchor="w", wraplength=800, justify="left"
                 ).pack(fill="x", pady=(1, 0))

    def show_add_pick_video(self) -> None:
        """Step 1: pick the video. Only one thing to do on this screen."""
        self._clear_body()
        self._show_back(step_text="Step 1 of 3 — The video")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        self._step_bar(wrap, self.ADD_STEPS, current=1)

        tk.Label(wrap, text="Step 1 — Pick the match video",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text="The match file itself (.mp4).",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 18))

        ttk.Button(wrap, text="Pick the video…", style="Accent.TButton",
                   command=self._on_add_pick_video).pack(anchor="w")

        self._next_up_hint(wrap, "Next: Step 2 — pick the Dartfish tags (.csv).")

    def _on_add_pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick the match video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if not path:
            return
        self.state["video"] = path
        self.show_add_pick_csv()

    def show_add_pick_csv(self) -> None:
        """Step 2: pick the CSV. Only one thing to do on this screen."""
        self._clear_body()
        self._show_back(step_text="Step 2 of 3 — The tags file")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        self._step_bar(wrap, self.ADD_STEPS, current=2)

        tk.Label(wrap, text="Step 2 — Now pick the Dartfish tags",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text="The spreadsheet (.csv) that was exported from Dartfish when this "
                      "match was tagged. It goes together with the video you just picked.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 18))

        ttk.Button(wrap, text="Pick the tags file…", style="Accent.TButton",
                   command=self._on_add_pick_csv).pack(anchor="w")

        self._next_up_hint(wrap, "Next: Step 3 — confirm and save.")

    def _on_add_pick_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Pick the Dartfish CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.state["csv"] = path
        self.show_add_confirm()

    def show_add_confirm(self) -> None:
        self._clear_body()
        self._show_back(step_text="Step 3 of 3 — Confirm")
        video = self.state["video"]
        csv_path = self.state["csv"]

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        self._step_bar(wrap, self.ADD_STEPS, current=3)

        tk.Label(wrap, text="Does this look right?",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text=f"Video:  {Path(video).name}\nTags:    {Path(csv_path).name}",
                 font=FONT_BODY, fg=TEXT, bg=BG, anchor="w", justify="left"
                 ).pack(fill="x", pady=(0, 20))
        tk.Label(wrap,
                 text=f"If you press Add, both files will be copied into the training folder:\n"
                      f"{TRAINING_PAIRS_DIR}",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 24))

        OptionButton(
            wrap, title="Continue to setup →",
            subtitle="Next: mark the court and players so the model learns from "
                     "consistent coordinates.",
            command=self.show_add_setup, primary=True,
        ).pack(fill="x", pady=6)

    def show_add_setup(self) -> None:
        self._clear_body()
        self._show_back(step_text="Setup the match")
        video = self.state["video"]

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="Mark the court and name the players",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text=f"{Path(video).name}\n\n"
                      "Click the 4 corners of the court and type the two players' names. "
                      "This anchors every clip in the same coordinates so the model can "
                      "learn player identity, court positions, and side-swap timing from "
                      "consistent labels.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 20))

        OptionButton(
            wrap, title="Open the setup window",
            subtitle="A separate window opens. When you're done, come back here and press continue.",
            command=self._on_add_open_setup, primary=True,
        ).pack(fill="x", pady=6)

        self._add_setup_continue_btn = ttk.Button(
            wrap, text="I finished setup — copy into training set",
            style="Accent.TButton",
            command=self._on_add_setup_continue, state="disabled",
        )
        self._add_setup_continue_btn.pack(anchor="w", pady=(20, 0))

    def _on_add_open_setup(self) -> None:
        video = self.state["video"]
        if not PREFLIGHT_SCRIPT.exists():
            messagebox.showerror("Missing file", "preflight.py is missing.")
            return
        try:
            subprocess.Popen([sys.executable, str(PREFLIGHT_SCRIPT), video],
                             cwd=str(REPO))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Couldn't open setup", str(exc))
            return
        self._add_setup_continue_btn.config(state="normal")
        self._poll_add_sidecar()

    def _poll_add_sidecar(self) -> None:
        video = self.state.get("video")
        if not video:
            return
        if Path(video + ".setup.json").exists():
            self._add_setup_continue_btn.config(
                text="✓ Setup saved — copy into training set")
            return
        self.root.after(1000, self._poll_add_sidecar)

    def _on_add_setup_continue(self) -> None:
        if not Path(self.state["video"] + ".setup.json").exists():
            messagebox.showwarning(
                "Not saved yet",
                "I can't find the setup file. Finish the setup window and press Save there first.")
            return
        self._do_add()

    def _do_add(self) -> None:
        video = Path(self.state["video"])
        csv_path = Path(self.state["csv"])
        sidecar = Path(self.state["video"] + ".setup.json")
        stamp = video.stem
        dest = TRAINING_PAIRS_DIR / stamp
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, dest / video.name)
            shutil.copy2(csv_path, dest / csv_path.name)
            if sidecar.exists():
                shutil.copy2(sidecar, dest / (video.name + ".setup.json"))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Couldn't copy files", str(exc))
            return
        self.show_add_done(dest)

    def show_add_done(self, dest: Path) -> None:
        self._clear_body()
        self._show_back()
        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        n_matches = sum(1 for p in TRAINING_PAIRS_DIR.iterdir() if p.is_dir()) \
            if TRAINING_PAIRS_DIR.is_dir() else 0

        tk.Label(wrap, text="Added to the training set",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text=f"Saved under:\n{dest}\n\n"
                      f"Training folder now has {n_matches} match{'es' if n_matches != 1 else ''}.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 20))

        OptionButton(
            wrap, title="Add another match",
            subtitle="Keep growing the training set.",
            command=self.show_add_pick_video, primary=True,
        ).pack(fill="x", pady=6)

        run_dir = REPO / "files" / "models" / "point_model" / "current"
        resume_hint = " (resumes if a checkpoint exists)" if (run_dir / "last.pt").exists() else ""

        OptionButton(
            wrap, title="Begin training the point model" + resume_hint,
            subtitle="Cuts clips, extracts features, and trains the multi-task "
                     "point model over every match in the training folder. "
                     "Takes hours — good to start before you step away.",
            command=self._begin_point_model_training,
        ).pack(fill="x", pady=6)

        if (run_dir / "last.pt").exists():
            OptionButton(
                wrap, title="Reset training (start fresh next time)",
                subtitle="Deletes the active checkpoint. Keeps clips and features.",
                command=self._reset_point_model_training,
            ).pack(fill="x", pady=6)

        OptionButton(
            wrap, title="Go home",
            subtitle="Done for now.",
            command=self.show_home,
        ).pack(fill="x", pady=6)

    def _begin_point_model_training(self) -> None:
        if not TRAINING_PAIRS_DIR.is_dir():
            messagebox.showerror("No training folder",
                                 "The training folder doesn't exist yet. "
                                 "Add a pre-tagged match first.")
            return
        self._show_pick_matches_to_train()

    def _show_pick_matches_to_train(self) -> None:
        self._clear_body()
        self._show_back(step_text="Pick matches")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="Which matches should we train on?",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text="All selected by default. Uncheck any you want to leave out — "
                      "their clips and features stay on disk and can be added back later.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w",
                 wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 16))

        matches = sorted((d for d in TRAINING_PAIRS_DIR.iterdir() if d.is_dir()),
                         key=lambda d: d.name)
        if not matches:
            tk.Label(wrap, text="(no matches in the training folder)",
                     font=FONT_SUB, fg=TEXT_DIM, bg=BG
                     ).pack(anchor="w", pady=(20, 0))
            return

        box = tk.Frame(wrap, bg=PANEL, highlightthickness=1,
                       highlightbackground=BORDER)
        box.pack(fill="both", expand=True, pady=(0, 12))
        canvas = tk.Canvas(box, bg=PANEL, bd=0, highlightthickness=0)
        sb = tk.Scrollbar(box, orient="vertical", command=canvas.yview,
                          bg=PANEL, troughcolor=BG, bd=0, activebackground=PANEL_HI)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        vars_: list[tuple[Path, tk.BooleanVar]] = []
        for m in matches:
            v = tk.BooleanVar(value=True)
            vars_.append((m, v))
            csvs = list(m.glob("*.csv"))
            n_points = 0
            if csvs:
                try:
                    with csvs[0].open("r", encoding="utf-8", errors="replace") as f:
                        n_points = max(0, sum(1 for _ in f) - 1)
                except OSError:
                    pass
            row = tk.Frame(inner, bg=PANEL)
            row.pack(fill="x", padx=12, pady=4)
            cb = tk.Checkbutton(row, variable=v, bg=PANEL, fg=TEXT,
                                activebackground=PANEL, activeforeground=TEXT,
                                selectcolor=PANEL_HI, bd=0, highlightthickness=0)
            cb.pack(side="left")
            tk.Label(row, text=m.name, font=FONT_OPTION, fg=TEXT, bg=PANEL,
                     anchor="w").pack(side="left", padx=(8, 12))
            tk.Label(row, text=f"{n_points} points", font=FONT_SUB,
                     fg=TEXT_DIM, bg=PANEL, anchor="e"
                     ).pack(side="right")

        btns = tk.Frame(wrap, bg=BG)
        btns.pack(fill="x")
        ttk.Button(btns, text="Continue →", style="Accent.TButton",
                   command=lambda: self._continue_with_selection(vars_)
                   ).pack(side="right")
        ttk.Button(btns, text="Select all", style="Ghost.TButton",
                   command=lambda: [v.set(True) for _, v in vars_]
                   ).pack(side="left")
        ttk.Button(btns, text="Clear", style="Ghost.TButton",
                   command=lambda: [v.set(False) for _, v in vars_]
                   ).pack(side="left", padx=(6, 0))

    def _continue_with_selection(self,
                                 vars_: list[tuple[Path, tk.BooleanVar]]) -> None:
        selected = [m for m, v in vars_ if v.get()]
        if not selected:
            messagebox.showerror("No matches selected",
                                 "Pick at least one match to train on.")
            return
        self._launch_point_model_chain(selected)

    def _launch_point_model_chain(self, selected: list[Path]) -> None:
        clips_dir = REPO / "files" / "data" / "clips"
        features_dir = REPO / "files" / "data" / "features"
        run_dir = REPO / "files" / "models" / "point_model" / "current"
        last_ckpt = run_dir / "last.pt"

        subtitle = ("Stages run back-to-back: (1) fine-tune the ball detector "
                    "from your preflight labels — one-off, skipped once the "
                    "weights exist; (2) cut clips from every Dartfish CSV; "
                    "(3) extract per-frame features; (4) train the multi-task "
                    "point model. Already-cut clips and already-extracted "
                    "features are skipped automatically. Takes hours on a real "
                    "dataset — leave the window open overnight.")

        train_cmd = [PYEXE, "-u", "-m", "ml.point_model", "train",
                     "--clips", str(clips_dir),
                     "--features", str(features_dir),
                     "--out", str(run_dir)]
        if last_ckpt.exists():
            train_cmd += ["--resume", str(last_ckpt)]
            subtitle = (f"Resuming from {last_ckpt.name} in "
                        f"{run_dir.relative_to(REPO)}.\n\n" + subtitle)

        total_points = sum(_points_in_match(m) for m in selected)
        selected_names = [m.name for m in selected]
        epochs_target = 20  # matches ml.point_model default

        # Chain: [ball-yolo prep + train if weights missing], then
        # clips(match1), features(match1), clips(match2), features(match2), …, train
        argvs: list[list[str]] = []
        ball_weights = REPO / "files" / "models" / "yolo_ball" / "best.pt"
        ball_yolo_out = REPO / "files" / "data" / "yolo_ball"
        n_ball_stages = 0
        if not ball_weights.exists():
            argvs.append([PYEXE, "-u", "-m", "ml.ball_labels_to_yolo",
                          "--pairs-dir", str(TRAINING_PAIRS_DIR),
                          "--output-dir", str(ball_yolo_out)])
            argvs.append([PYEXE, "-u", "-m", "ml.train_yolo",
                          "--data", str(ball_yolo_out / "_shared" / "data.yaml"),
                          "--base", str(REPO / "models" / "yolov8s.pt"),
                          "--out", str(REPO / "files" / "models" / "yolo_ball"),
                          "--imgsz", "640", "--batch", "4"])
            n_ball_stages = 2
        for m in selected:
            argvs.append([PYEXE, "-u", "-m", "ml.dartfish_to_clips",
                          str(TRAINING_PAIRS_DIR), "--out", str(clips_dir),
                          "--only", m.name])
            argvs.append([PYEXE, "-u", "-m", "ml.feature_extractor",
                          str(clips_dir), "--out", str(features_dir),
                          "--only", m.name])
        argvs.append(train_cmd)
        n_prep_stages = n_ball_stages + len(selected) * 2  # all non-train stages

        def _count_selected_clips() -> int:
            return sum(len(list((clips_dir / n).glob("p_*.mp4")))
                       for n in selected_names)

        def _count_selected_features() -> int:
            return sum(len(list((features_dir / n).glob("*.npz")))
                       for n in selected_names)

        def progress_fn(stage_idx, n_stages, log_text):
            if stage_idx >= n_prep_stages:
                done = _last_epoch_from_log(log_text)
                sub = min(1.0, (done + 1) / epochs_target) if done >= 0 else 0.0
                label = (f"training — epoch {done + 1}/{epochs_target}"
                         if done >= 0 else "training — starting…")
                overall = (n_prep_stages + sub) / n_stages
                return overall, label
            # One-off ball-YOLO fine-tune (runs only when weights missing).
            if stage_idx < n_ball_stages:
                label = ("preparing ball-detector training data"
                         if stage_idx == 0
                         else "training ball detector (one-off, ~15–60 min)")
                return stage_idx / n_stages, label
            # pre-train: weight clips and features each as half of prep work
            match_idx = stage_idx - n_ball_stages
            clips_n = _count_selected_clips()
            feat_n = _count_selected_features()
            prep_done = (clips_n + feat_n) / max(total_points * 2, 1)
            prep_frac = (n_ball_stages + min(1.0, prep_done) *
                         (len(selected) * 2)) / n_stages
            phase = ("cutting clips" if match_idx % 2 == 0
                     else "extracting features")
            label = (f"{phase} — clips {clips_n}/{total_points}, "
                     f"features {feat_n}/{total_points} "
                     f"(match {match_idx // 2 + 1}/{len(selected)})")
            return prep_frac, label

        self._run_screen(
            title="Training the point model…",
            subtitle=subtitle + f"\n\nSelected {len(selected)} match"
                                 f"{'es' if len(selected) != 1 else ''} "
                                 f"({total_points} points).",
            argvs=argvs,
            on_success=self.show_home,
            step_text="Training point model",
            progress_fn=progress_fn,
        )

    def _reset_point_model_training(self) -> None:
        run_dir = REPO / "files" / "models" / "point_model" / "current"
        if not run_dir.exists():
            messagebox.showinfo("Nothing to reset",
                                "No active training run yet.")
            return
        if not messagebox.askyesno(
                "Reset training?",
                f"This deletes:\n{run_dir}\n\n"
                "The next training run will start from epoch 0. "
                "Cut clips and extracted features are kept. Continue?"):
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        messagebox.showinfo("Reset", "Training checkpoint cleared.")

    # ---- Manage training set ----

    def show_manage_training(self) -> None:
        self._clear_body()
        self._show_back(step_text="Training set")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="Training set",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))

        if not TRAINING_PAIRS_DIR.is_dir():
            tk.Label(wrap,
                     text="No matches imported yet. Add one from the home screen.",
                     font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w"
                     ).pack(fill="x", pady=(0, 20))
            return

        matches = sorted((d for d in TRAINING_PAIRS_DIR.iterdir() if d.is_dir()),
                         key=lambda d: d.name)
        tk.Label(wrap,
                 text=f"{len(matches)} match{'es' if len(matches) != 1 else ''} "
                      f"under {TRAINING_PAIRS_DIR}",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(0, 16))

        clips_root = REPO / "files" / "data" / "clips"
        features_root = REPO / "files" / "data" / "features"

        box = tk.Frame(wrap, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="both", expand=True)

        canvas = tk.Canvas(box, bg=PANEL, bd=0, highlightthickness=0)
        sb = tk.Scrollbar(box, orient="vertical", command=canvas.yview,
                          bg=PANEL, troughcolor=BG, bd=0, activebackground=PANEL_HI)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        if not matches:
            tk.Label(inner, text="(empty)", font=FONT_SUB, fg=TEXT_DIM, bg=PANEL,
                     padx=16, pady=16).pack(anchor="w")
            return

        for m in matches:
            vids = list(m.glob("*.mp4"))
            csvs = list(m.glob("*.csv"))
            clips_count = sum(1 for _ in (clips_root / m.name).glob("p_*.mp4")) \
                if (clips_root / m.name).is_dir() else 0
            feats_count = sum(1 for _ in (features_root / m.name).glob("*.npz")) \
                if (features_root / m.name).is_dir() else 0
            sidecar = (m / (vids[0].name + ".setup.json")) if vids else None
            has_setup = bool(sidecar and sidecar.exists())

            row = tk.Frame(inner, bg=PANEL)
            row.pack(fill="x", padx=12, pady=6)

            col = tk.Frame(row, bg=PANEL)
            col.pack(side="left", fill="x", expand=True)
            tk.Label(col, text=m.name, font=FONT_OPTION, fg=TEXT, bg=PANEL,
                     anchor="w").pack(fill="x")
            setup_label = "setup ✓" if has_setup else "setup ✗"
            detail = (f"{vids[0].name if vids else '(no video)'} • "
                      f"{csvs[0].name if csvs else '(no csv)'} • "
                      f"clips: {clips_count} • features: {feats_count} • "
                      f"{setup_label}")
            tk.Label(col, text=detail, font=FONT_SUB, fg=TEXT_DIM, bg=PANEL,
                     anchor="w").pack(fill="x")

            ttk.Button(row, text="Remove", style="Ghost.TButton",
                       command=lambda md=m: self._remove_training_match(md)
                       ).pack(side="right", padx=(12, 0))
            if vids:
                setup_text = "Redo setup" if has_setup else "Run setup"
                ttk.Button(row, text=setup_text, style="Ghost.TButton",
                           command=lambda md=m, vp=vids[0]: self._run_setup_for_pair(md, vp)
                           ).pack(side="right", padx=(12, 0))

    def _run_setup_for_pair(self, match_dir: Path, video_in_pair: Path) -> None:
        if not PREFLIGHT_SCRIPT.exists():
            messagebox.showerror("Missing file", "preflight.py is missing.")
            return
        try:
            subprocess.Popen([sys.executable, str(PREFLIGHT_SCRIPT),
                              str(video_in_pair)], cwd=str(REPO))
        except Exception as exc:
            messagebox.showerror("Couldn't open setup", str(exc))
            return
        messagebox.showinfo(
            "Setup launched",
            "The preflight window has opened. After you click the court corners and "
            "save, the sidecar will be written next to the video in the training pair. "
            "Reopen 'Manage training set' to see the ✓ update.",
        )

    def _remove_training_match(self, match_dir: Path) -> None:
        if not messagebox.askyesno(
                "Remove match?",
                f"Delete:\n\n{match_dir}\n\n"
                "Also removes any cut clips and extracted features for this match. "
                "Does not touch the trained model. Continue?"):
            return
        clips_dir = REPO / "files" / "data" / "clips" / match_dir.name
        features_dir = REPO / "files" / "data" / "features" / match_dir.name
        for d in (match_dir, clips_dir, features_dir):
            shutil.rmtree(d, ignore_errors=True)
        self.show_manage_training()

    # ---- Advanced: Share with a friend (USB) ----
    #
    # Walkthrough: Hub -> Send (pick folder -> done) | Receive (pick folder -> review -> action -> done)

    def show_share_hub(self) -> None:
        self._clear_body()
        self._show_back(step_text="Share with a friend")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="Share with a friend",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap,
                 text="Plug a USB stick or external drive into your machine. "
                      "Tennis Tagger uses it to swap trained models with a friend.",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w", justify="left", wraplength=720
                 ).pack(fill="x", pady=(0, 16))

        local_pt = REPO / "files" / "models" / "point_model" / "current" / "best.pt"
        has_pt = local_pt.exists()

        OptionButton(
            wrap,
            title="Send my model to my friend",
            subtitle=("Save your trained model to a folder on the USB drive. "
                      "Eject the drive and hand it over." if has_pt else
                      "You haven't trained a model yet — train one first to share it."),
            command=self._share_send_step1 if has_pt else (lambda: messagebox.showinfo(
                "Nothing to share yet",
                "Train a point model first. The home screen has a 'Train the point model' option once you've added matches.")),
            primary=has_pt,
        ).pack(fill="x", pady=8)

        OptionButton(
            wrap,
            title="Use my friend's model",
            subtitle="Plug in the drive your friend gave you. Pick the folder they sent. "
                     "Tennis Tagger walks you through what to do with it.",
            command=self._share_recv_step1,
        ).pack(fill="x", pady=8)

    # ---- Send walkthrough ----

    def _share_send_step1(self) -> None:
        """Step 1 of 2: pick destination folder."""
        out_dir = filedialog.askdirectory(
            title="Pick an empty folder on your USB drive",
            mustexist=False,
        )
        if not out_dir:
            return
        out_path = Path(out_dir)
        if out_path.exists() and any(out_path.iterdir()):
            messagebox.showerror(
                "Folder not empty",
                f"{out_path} already has files in it.\n\n"
                "Pick a brand-new folder (or make one) so we don't overwrite anything.",
            )
            return
        self._share_send_step2(out_path)

    def _share_send_step2(self, out_path: Path) -> None:
        """Step 2 of 2: ask for name, run export, show success."""
        author = simpledialog.askstring(
            "Your name",
            "Your name (so your friend knows whose model this is):",
            parent=self.root,
        )
        if author is None:
            return
        out_path.mkdir(parents=True, exist_ok=True)

        cmd = [str(TAGGER_EXE), "model", "export",
               "--author", (author or "anonymous"),
               str(out_path)]
        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=60)
        except Exception as e:
            messagebox.showerror("Send failed", str(e))
            return
        if r.returncode != 0:
            messagebox.showerror("Send failed", (r.stderr or r.stdout or "Unknown error").strip())
            return
        messagebox.showinfo(
            "Model saved to USB",
            f"Done!\n\n{r.stdout.strip()}\n\n"
            f"Eject the drive and give it to your friend. Tell them to pick this folder:\n"
            f"{out_path}",
        )
        self.show_home()

    # ---- Receive walkthrough ----

    def _share_recv_step1(self) -> None:
        """Step 1 of 3: pick the bundle folder."""
        if not self._require(TAGGER_EXE, "tagger.exe is missing"):
            return
        in_dir = filedialog.askdirectory(
            title="Pick the folder on your friend's USB drive",
        )
        if not in_dir:
            return
        bundle = Path(in_dir)
        manifest_path = bundle / "manifest.json"
        if not manifest_path.exists():
            messagebox.showerror(
                "That doesn't look like a model bundle",
                f"{manifest_path} was not found.\n\n"
                "Pick the folder that has manifest.json and weights.pt inside it.",
            )
            return

        try:
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Couldn't read the bundle", str(e))
            return

        self._share_recv_step2(bundle, manifest)

    def _share_recv_step2(self, bundle: Path, manifest: dict) -> None:
        """Step 2 of 3: show what we found, pick action."""
        self._clear_body()
        self._show_back(step_text="Use my friend's model — Step 2 of 3")

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        author = manifest.get("author", "unknown")
        created = manifest.get("created_at", "unknown date")
        size_mb = manifest.get("size_bytes", 0) / (1024 * 1024)
        notes = manifest.get("notes", "")

        tk.Label(wrap, text=f"{author}'s trained model",
                 font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap, text=f"Made on {created}    •    {size_mb:.1f} MB",
                 font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(0, 6))
        if notes:
            tk.Label(wrap, text=f"Note: {notes}",
                     font=FONT_BODY, fg=TEXT_DIM, bg=BG, anchor="w", wraplength=720, justify="left"
                     ).pack(fill="x", pady=(0, 12))

        local_pt = REPO / "files" / "models" / "point_model" / "current" / "best.pt"
        has_local = local_pt.exists()

        tk.Label(wrap, text="What would you like to do with it?",
                 font=FONT_OPTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(8, 12))

        OptionButton(
            wrap,
            title="Merge with my model",
            subtitle="Average their weights with yours. This is the recommended weekly flow — "
                     "you both end up with a model that learned from both sets of footage. "
                     "Your current model is backed up first.",
            command=lambda: self._share_recv_step3(bundle, manifest, action="merge"),
            primary=has_local,
        ).pack(fill="x", pady=8)

        OptionButton(
            wrap,
            title="Replace my model with theirs",
            subtitle="Use their model directly instead of yours. Good for trying out their model "
                     "as-is. Your current model is backed up first so you can switch back.",
            command=lambda: self._share_recv_step3(bundle, manifest, action="replace"),
        ).pack(fill="x", pady=8)

        if not has_local:
            tk.Label(wrap, text="(You don't have a trained model yet, so 'Merge' isn't available — "
                                "Replace will install theirs.)",
                     font=FONT_SUB, fg=TEXT_DIM, bg=BG, anchor="w", wraplength=720, justify="left"
                     ).pack(fill="x", pady=(8, 0))

    def _share_recv_step3(self, bundle: Path, manifest: dict, action: str) -> None:
        """Step 3 of 3: do the thing, show result."""
        local_pt = REPO / "files" / "models" / "point_model" / "current" / "best.pt"

        # Always back up the current model before changing anything.
        backup = None
        if local_pt.exists():
            backup = local_pt.with_name(f"best.before-{action}-{_timestamp()}.pt")
            try:
                shutil.copy2(local_pt, backup)
            except OSError as e:
                messagebox.showerror("Backup failed", str(e))
                return

        if action == "merge" and not local_pt.exists():
            # Fall through to replace — nothing to merge with.
            action = "replace"

        if action == "merge":
            cmd = [str(TAGGER_EXE), "model", "merge", "--python", PYEXE, str(bundle)]
            verb = "Merged"
        else:
            cmd = [str(TAGGER_EXE), "model", "import", str(bundle)]
            verb = "Replaced"

        try:
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            messagebox.showerror("Timed out", "Operation took too long (>10 min).")
            return
        except Exception as e:
            messagebox.showerror(f"{verb} failed", str(e))
            return
        if r.returncode != 0:
            messagebox.showerror(f"{verb.replace('ed', '')} failed",
                                 (r.stderr or r.stdout or "Unknown error").strip())
            return

        backup_msg = f"\n\nYour previous model was backed up to:\n{backup.name}" if backup else ""
        messagebox.showinfo(
            f"{verb} successfully",
            f"{verb} {manifest.get('author','your friend')}'s model into yours."
            f"{backup_msg}\n\nRe-tag a known match to sanity-check the result.",
        )
        self.show_home()

    # ---- Advanced: YOLO ball-finder pipeline (labels → train) ----

    def show_adv_labels_then_train(self) -> None:
        if not self._require(DARTFISH_SCRIPT, "dartfish_to_yolo.py is missing"):
            return
        if not self._require(TRAIN_SCRIPT, "train_yolo_ball.py is missing"):
            return
        if not TRAINING_PAIRS_DIR.is_dir():
            messagebox.showerror("No training folder",
                                 "The training folder doesn't exist yet. "
                                 "Add a pre-tagged match first.")
            return
        output_dir = filedialog.askdirectory(
            title="Where should the training labels be saved?")
        if not output_dir:
            return
        yaml_path = Path(output_dir) / "dataset.yaml"
        self._run_screen(
            title="Building labels, then training the ball finder…",
            subtitle="Stage 1 projects every human-marked bounce into video frames. "
                     "Stage 2 trains YOLO on those frames. Runs end-to-end — leave "
                     "it overnight.",
            argvs=[
                [sys.executable, str(DARTFISH_SCRIPT),
                 "--pairs-dir", str(TRAINING_PAIRS_DIR),
                 "--output-dir", output_dir],
                [sys.executable, str(TRAIN_SCRIPT), "--dataset", str(yaml_path)],
            ],
            on_success=self.show_home,
            step_text="Advanced — Labels + training",
        )

    # ------------------------------------------------------------------
    # Generic "running" screen
    # ------------------------------------------------------------------

    def _run_screen(self, *, title: str, subtitle: str,
                    argv: list[str] | None = None,
                    argvs: list[list[str]] | None = None,
                    on_success, step_text: str = "",
                    progress_fn=None) -> None:
        chain = argvs if argvs is not None else ([argv] if argv is not None else [])
        if not chain:
            raise ValueError("_run_screen requires argv or argvs")

        self._clear_body()
        self._show_back(step_text=step_text)

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text=title, font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap, text=subtitle, font=FONT_SUB, fg=TEXT_DIM, bg=BG,
                 anchor="w", wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 16))

        top = tk.Frame(wrap, bg=BG)
        top.pack(fill="x", pady=(0, 4))
        self.control_btn = ttk.Button(top, text="▶  Play", style="Accent.TButton",
                                      command=self._toggle_chain)
        self.control_btn.pack(side="right")

        mode = "determinate" if progress_fn is not None else "indeterminate"
        self.progress = ttk.Progressbar(top, mode=mode, maximum=100,
                                        style="Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._progress_mode = mode
        if mode == "determinate":
            self.progress_label = tk.Label(wrap, text="Ready — press Play to start.",
                                           font=FONT_SUB, fg=TEXT_DIM, bg=BG,
                                           anchor="w")
            self.progress_label.pack(fill="x", pady=(0, 8))
        else:
            self.progress_label = None
        self._progress_fn = progress_fn

        # Log panel
        box = tk.Frame(wrap, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="both", expand=True)
        sb = tk.Scrollbar(box, bg=PANEL, troughcolor=BG, bd=0,
                          activebackground=PANEL_HI)
        sb.pack(side="right", fill="y")
        self.log = tk.Text(box, font=FONT_LOG, wrap="word",
                           yscrollcommand=sb.set,
                           bg=PANEL, fg=TEXT, insertbackground=TEXT,
                           bd=0, padx=12, pady=10, relief="flat", height=12)
        self.log.pack(side="left", fill="both", expand=True)
        sb.config(command=self.log.yview)

        self._on_success = on_success
        self._chain_template = list(chain)
        self._chain = []
        self._chain_idx = 0
        self._write_log("[ready — press Play to start]\n\n")

    def _tick_progress(self) -> None:
        if getattr(self, "proc", None) is None and \
                self._chain_idx >= len(getattr(self, "_chain", [])):
            return
        try:
            frac, label = self._progress_fn(self._chain_idx,
                                            len(self._chain),
                                            self._log_text())
        except Exception as e:  # noqa: BLE001
            frac, label = None, f"(progress error: {e})"
        if frac is not None and self.progress_label is not None:
            pct = max(0.0, min(1.0, frac)) * 100.0
            try:
                self.progress.configure(value=pct)
                self.progress_label.configure(text=f"{pct:5.1f}%  —  {label}")
            except tk.TclError:
                return
        self.root.after(1000, self._tick_progress)

    def _log_text(self) -> str:
        try:
            return self.log.get("1.0", END)
        except (tk.TclError, AttributeError):
            return ""

    def _toggle_chain(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            self._stop_chain()
        else:
            self._play_chain()

    def _play_chain(self) -> None:
        template = getattr(self, "_chain_template", [])
        if not template:
            return
        self._chain = list(template)
        self._chain_idx = 0
        try:
            self.control_btn.configure(text="■  Stop")
        except tk.TclError:
            return
        if self._progress_mode == "indeterminate":
            try: self.progress.start(12)
            except tk.TclError: pass
        self._start_next_in_chain()
        if self._progress_fn is not None:
            self._tick_progress()

    def _stop_chain(self) -> None:
        self._chain = []
        self._chain_idx = 0
        proc = getattr(self, "proc", None)
        try:
            self.control_btn.configure(text="▶  Play")
        except tk.TclError:
            pass
        if proc is None or proc.poll() is not None:
            self._write_log("\n[stop requested — nothing running]\n")
            return
        self._write_log("\n[stopping…]\n")
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                proc.terminate()
        except Exception as exc:  # noqa: BLE001
            self._write_log(f"[stop failed: {exc}]\n")

    def _start_next_in_chain(self) -> None:
        if self._chain_idx >= len(self._chain):
            self._on_proc_done(0)
            return
        argv = self._chain[self._chain_idx]
        stage = f"[stage {self._chain_idx + 1}/{len(self._chain)}] " \
                f"{' '.join(str(a) for a in argv)}\n"
        self._write_log(stage)
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(REPO),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self._write_log(f"ERROR: failed to start: {exc}\n")
            if hasattr(self, "progress"):
                try: self.progress.stop()
                except tk.TclError: pass
            return
        threading.Thread(target=self._reader_thread, args=(self.proc,),
                         daemon=True).start()

    def _reader_thread(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(line)
        rc = proc.wait()
        self.log_queue.put(f"__DONE__:{rc}\n")

    def _pump_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line.startswith("__DONE__"):
                    rc = int(line.split(":", 1)[1].strip() or "0")
                    self._on_proc_done(rc)
                else:
                    self._write_log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._pump_log)

    def _on_proc_done(self, rc: int) -> None:
        self.proc = None
        chain = getattr(self, "_chain", [])
        if rc == 0 and getattr(self, "_chain_idx", 0) < len(chain) - 1:
            self._chain_idx += 1
            self._write_log(f"\n[stage {self._chain_idx}/{len(chain)} finished ok]\n\n")
            self._start_next_in_chain()
            return

        if hasattr(self, "progress"):
            try:
                self.progress.stop()
            except tk.TclError:
                pass
        try:
            self.control_btn.configure(text="▶  Play")
        except (tk.TclError, AttributeError):
            pass
        if rc == 0:
            self._write_log("\n[finished successfully]\n")
            cb = getattr(self, "_on_success", None)
            if cb:
                self.root.after(400, cb)
        else:
            self._write_log(f"\n[stopped with exit code {rc}]\n")

    def _write_log(self, text: str) -> None:
        # Always persist to disk first so crashes/fast window-close don't eat
        # the output. The file is created on the first write of each run.
        try:
            if getattr(self, "_log_path", None) is None:
                log_dir = REPO / "files" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._log_path = log_dir / f"run-{_timestamp()}.log"
            with open(self._log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception:
            pass
        if not hasattr(self, "log"):
            return
        try:
            self.log.insert(END, text)
            self.log.see(END)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _require(self, path: Path, msg: str) -> bool:
        if path.exists():
            return True
        messagebox.showerror("Missing file", msg)
        return False

    def _open_path(self, path: str) -> None:
        try:
            if sys.platform.startswith("win"):
                import os
                os.startfile(path)  # noqa: S606 — user-initiated
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Couldn't open", str(exc))


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

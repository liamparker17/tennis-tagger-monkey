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
from pathlib import Path
from tkinter import END, filedialog, messagebox, ttk

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

        # Small 'Advanced' link in the footer area
        adv = tk.Frame(wrap, bg=BG)
        adv.pack(fill="x", pady=(24, 0))
        tk.Label(adv, text="Advanced:", font=FONT_SUB, fg=TEXT_DIM, bg=BG
                 ).pack(side="left")
        ttk.Button(adv, text="Build training labels", style="Ghost.TButton",
                   command=self.show_adv_labels).pack(side="left", padx=(8, 0))
        ttk.Button(adv, text="Train the ball finder", style="Ghost.TButton",
                   command=self.show_adv_train).pack(side="left", padx=(4, 0))

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
            wrap, title="Add it to the training set",
            subtitle="Copies the two files so the model can learn from this match later.",
            command=self._do_add, primary=True,
        ).pack(fill="x", pady=6)

    def _do_add(self) -> None:
        video = Path(self.state["video"])
        csv_path = Path(self.state["csv"])
        stamp = video.stem
        dest = TRAINING_PAIRS_DIR / stamp
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, dest / video.name)
            shutil.copy2(csv_path, dest / csv_path.name)
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

        OptionButton(
            wrap, title="Build training labels from the whole folder",
            subtitle="Runs the label builder over every match in the training folder. "
                     "Takes hours — good to start before you step away.",
            command=self._build_labels_from_training_dir,
        ).pack(fill="x", pady=6)

        OptionButton(
            wrap, title="Go home",
            subtitle="Done for now.",
            command=self.show_home,
        ).pack(fill="x", pady=6)

    def _build_labels_from_training_dir(self) -> None:
        if not self._require(DARTFISH_SCRIPT, "dartfish_to_yolo.py is missing"):
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
        self._run_screen(
            title="Building training labels from your Dartfish tags…",
            subtitle="Reading every Dartfish CSV and projecting the human-marked "
                     "bounces into the video frames. Runs for a while — leave it "
                     "overnight and the ball finder will learn from your own tags.",
            argv=[sys.executable, str(DARTFISH_SCRIPT),
                  "--pairs-dir", str(TRAINING_PAIRS_DIR),
                  "--output-dir", output_dir],
            on_success=self.show_home,
            step_text="Building labels",
        )

    # ---- Advanced ----

    def show_adv_labels(self) -> None:
        if not self._require(PSEUDO_SCRIPT, "generate_pseudo_labels.py is missing"):
            return
        if not self._require(TRACKNET_WEIGHTS, f"Missing weights: {TRACKNET_WEIGHTS.name}"):
            return
        input_dir = filedialog.askdirectory(title="Folder with your tagged videos")
        if not input_dir:
            return
        output_dir = filedialog.askdirectory(title="Where should the training labels go?")
        if not output_dir:
            return
        self._run_screen(
            title="Building training labels…",
            subtitle="This runs TrackNet across your footage. It can take hours. "
                     "Leave the window open and come back later.",
            argv=[sys.executable, str(PSEUDO_SCRIPT),
                  "--input-dir", input_dir, "--output-dir", output_dir,
                  "--tracknet-weights", str(TRACKNET_WEIGHTS)],
            on_success=self.show_home,
            step_text="Advanced — Making labels",
        )

    def show_adv_train(self) -> None:
        if not self._require(TRAIN_SCRIPT, "train_yolo_ball.py is missing"):
            return
        dataset_dir = filedialog.askdirectory(title="Training labels folder (where dataset.yaml lives)")
        if not dataset_dir:
            return
        yaml_path = Path(dataset_dir) / "dataset.yaml"
        if not yaml_path.is_file():
            messagebox.showerror("Not ready",
                                 f"No dataset.yaml found inside:\n{dataset_dir}")
            return
        self._run_screen(
            title="Training the ball finder…",
            subtitle="Overnight job. The window will keep updating as it learns.",
            argv=[sys.executable, str(TRAIN_SCRIPT), "--dataset", str(yaml_path)],
            on_success=self.show_home,
            step_text="Advanced — Training",
        )

    # ------------------------------------------------------------------
    # Generic "running" screen
    # ------------------------------------------------------------------

    def _run_screen(self, *, title: str, subtitle: str, argv: list[str],
                    on_success, step_text: str = "") -> None:
        self._clear_body()
        self._show_back(step_text=step_text)

        wrap = tk.Frame(self.body, bg=BG)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text=title, font=FONT_QUESTION, fg=TEXT, bg=BG, anchor="w"
                 ).pack(fill="x", pady=(20, 6))
        tk.Label(wrap, text=subtitle, font=FONT_SUB, fg=TEXT_DIM, bg=BG,
                 anchor="w", wraplength=900, justify="left"
                 ).pack(fill="x", pady=(0, 16))

        self.progress = ttk.Progressbar(wrap, mode="indeterminate",
                                        style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 12))
        self.progress.start(12)

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

        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(REPO),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self._write_log(f"ERROR: failed to start: {exc}\n")
            self.progress.stop()
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
        if hasattr(self, "progress"):
            try:
                self.progress.stop()
            except tk.TclError:
                pass
        self.proc = None
        if rc == 0:
            self._write_log("\n[finished successfully]\n")
            cb = getattr(self, "_on_success", None)
            if cb:
                self.root.after(400, cb)
        else:
            self._write_log(f"\n[stopped with exit code {rc}]\n")

    def _write_log(self, text: str) -> None:
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

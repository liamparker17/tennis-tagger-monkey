"""Pre-flight setup UI — grab a frame from a match video, let the user
click the 4 court corners and name the near/far players, then save a
sidecar JSON file next to the video.

Writes:   <video>.setup.json

The Go pipeline will (in a follow-up patch) load this sidecar and use
the provided corners + player names instead of guessing.

Run with:
    python preflight.py <video.mp4>
    python preflight.py                      # opens a file picker
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    BOTH, DISABLED, END, HORIZONTAL, LEFT, NORMAL, RIGHT, TOP, Button, Canvas,
    Entry, Frame, Label, Listbox, PhotoImage, Scale, Scrollbar, StringVar, Tk,
    Toplevel, filedialog, messagebox,
)

import cv2
import numpy as np

HUGE = ("Segoe UI", 20, "bold")
BIG = ("Segoe UI", 14)
MED = ("Segoe UI", 12)

# Corners, in the order the user is asked to click them.
# Near = closer to camera (bottom of frame), Far = far baseline (top of frame).
CORNER_STEPS = [
    ("Near-LEFT baseline corner",
     "Closest corner to the camera, on the LEFT side of the frame.",
     "#ff3b30"),
    ("Near-RIGHT baseline corner",
     "Closest corner to the camera, on the RIGHT side of the frame.",
     "#ff9500"),
    ("Far-RIGHT baseline corner",
     "Far baseline (top of frame), on the RIGHT.",
     "#34c759"),
    ("Far-LEFT baseline corner",
     "Far baseline (top of frame), on the LEFT.",
     "#007aff"),
]

DEFAULT_SEEK_FRACTION = 0.10  # 10% into the video
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 720

# Step 6: player-position labeling.
LABEL_NEXT_FRAME_JUMP = 60     # "Next frame" button jumps ~2s at 30 fps
LABEL_MIN_BOX_PX = 6           # drags shorter than this are treated as clicks, not boxes
BALL_NEXT_FRAME_JUMP = 15      # ball moves fast — jump ~0.5s at 30 fps
BALL_MIN_BOX_PX = 4            # ball box can be tiny; lower threshold than players


class Preflight:
    def __init__(self, root: Tk, video_path: Path) -> None:
        self.root = root
        self.video_path = video_path
        root.title(f"Set up match — {video_path.name}")
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = min(1400, screen_w - 40)
        win_h = min(900, screen_h - 80)
        root.geometry(f"{win_w}x{win_h}")
        root.minsize(min(1100, win_w), min(600, win_h))
        # Cap the image canvas height so huge videos don't dominate the
        # viewport. With the scrollable wrapper in place we no longer need
        # to squeeze the canvas down to keep the save button visible.
        self.max_canvas_h = max(600, min(win_h - 200, 900))

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            messagebox.showerror("Can't open video", f"Could not open:\n{video_path}")
            root.destroy()
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Current frame shown on the canvas
        self.current_frame_idx = int(self.total_frames * DEFAULT_SEEK_FRACTION)
        self.scale = 1.0
        self.temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.temp_png.close()

        # Dedicated capture + temp png for scrubber hover previews, so that
        # seeking for the preview never disturbs the main canvas state.
        self.preview_cap = cv2.VideoCapture(str(video_path))
        self.preview_temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.preview_temp_png.close()
        self.preview_win: Toplevel | None = None
        self.preview_img: PhotoImage | None = None
        self.preview_last_idx: int = -1
        self.preview_pending: str | None = None  # after() id

        # State
        self.corners: list[tuple[float, float]] = []   # in ORIGINAL pixel coords
        self.step = 0                                   # 0..3 = corners, 4 = players+swaps, 5 = label players, 6 = label ball
        # Frames where the players switch ends. Populated only during step 4.
        self.side_swaps: list[int] = []
        # Step 6 state: user-drawn player boxes, keyed by frame index.
        # Boxes are stored in ORIGINAL (unscaled) pixel coords as (x1, y1, x2, y2).
        self.player_labels: dict[int, list[tuple[float, float, float, float]]] = {}
        # Step 7 state: one ball bbox per frame, original pixel coords.
        self.ball_labels: dict[int, tuple[float, float, float, float]] = {}
        # Step 5 (players) state: up to 2 Lab triples per player. Collected
        # by clicking on the canvas while in step 4. Empty lists mean the
        # user skipped the color picker — feature extraction falls back to
        # sort-by-y identity assignment.
        self.player_colors: dict[str, list[tuple[float, float, float]]] = {"near": [], "far": []}
        # Tk widgets populated later so we can redraw swatch backgrounds.
        self._color_swatches: dict[str, list] = {"near": [], "far": []}
        # Order in which colors were picked, used for undo (each entry is
        # ("near"|"far", slot_index)).
        self._color_pick_order: list[tuple[str, int]] = []
        self._drag_start: tuple[int, int] | None = None
        self._drag_rect_id: int | None = None
        self._label_box_ids: list[int] = []
        self._ball_box_ids: list[int] = []
        self._selected_box_idx: int | None = None

        # ---- Scrollable wrapper so the preflight always fits any monitor.
        # On short screens or with Windows display scaling the controls +
        # image canvas would otherwise push Save off the bottom of the window.
        # Everything below is packed into `body` (the inner frame), not root. ----
        scroll_canvas = Canvas(root, highlightthickness=0)
        scrollbar = Scrollbar(root, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        scroll_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        body = Frame(scroll_canvas)
        body_id = scroll_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda _e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>",
                           lambda e: scroll_canvas.itemconfigure(body_id, width=e.width))

        def _on_mousewheel(event):
            # Windows delivers ±120 per notch; divide to get ±1 line units.
            scroll_canvas.yview_scroll(int(-event.delta / 120), "units")
        root.bind_all("<MouseWheel>", _on_mousewheel)

        # ---- Layout ----
        self.title_var = StringVar()
        self.help_var = StringVar()

        top = Frame(body, pady=10)
        top.pack(fill="x")
        Label(top, textvariable=self.title_var, font=HUGE).pack()
        Label(top, textvariable=self.help_var, font=BIG, fg="#444").pack(pady=(4, 0))

        # ---- Navigation: earlier / later frame ----
        nav = Frame(body, pady=4)
        nav.pack(fill="x")
        Label(nav, text="Is this a good frame? If the players aren't visible yet:", font=MED).pack(side=LEFT, padx=20)
        Button(nav, text="< 10 seconds earlier", font=MED, command=lambda: self.jump_seconds(-10)).pack(side=LEFT, padx=4)
        Button(nav, text="10 seconds later >", font=MED, command=lambda: self.jump_seconds(10)).pack(side=LEFT, padx=4)

        # ---- Bottom buttons (inside the scrollable body — user scrolls to
        # reach Save when content overflows the viewport) ----
        btns = Frame(body, pady=10)
        btns.pack(side="bottom", fill="x")
        self.undo_btn = Button(btns, text="Undo last click", font=BIG, command=self.undo, state=DISABLED)
        self.undo_btn.pack(side=LEFT, padx=20)
        self.save_btn = Button(btns, text="Save setup", font=BIG, command=self.save, state=DISABLED)
        self.save_btn.pack(side=RIGHT, padx=20)

        # ---- Player-name + side-swap section (hidden until corners done).
        # Packed at bottom so it sits just above the button bar and stays on-screen. ----
        self.players_frame = Frame(body, pady=6)
        self._players_frame_packed = False

        # ---- Player-position labeling section (step 6, hidden until corners + names done). ----
        self.label_frame = Frame(body, pady=6)

        # ---- Ball-position labeling section (step 7, hidden until player labels done). ----
        self.ball_frame = Frame(body, pady=6)

        # ---- Canvas (packed last so it takes only leftover space) ----
        self.canvas = Canvas(body, bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True, padx=20, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # ---- Player-name + color-pick + side-swap widgets (compact) ----
        # Row 0/1: name entry + 2 color swatches + Clear button per player.
        # Clicking on the frame fills the next empty swatch in order:
        # near[0] → near[1] → far[0] → far[1].
        for row, key, caption in ((0, "near", "BOTTOM player (near):"),
                                  (1, "far",  "TOP player (far):")):
            Label(self.players_frame, text=caption, font=MED).grid(
                row=row, column=0, sticky="w", padx=10, pady=2)
            entry = Entry(self.players_frame, font=MED, width=24)
            entry.grid(row=row, column=1, padx=4, pady=2, sticky="w")
            if key == "near": self.near_entry = entry
            else: self.far_entry = entry
            sw_frame = Frame(self.players_frame)
            sw_frame.grid(row=row, column=2, sticky="w", padx=4)
            for _slot in range(2):
                sw = Label(sw_frame, text="  ?  ", font=MED, bg="#dddddd",
                           fg="#666666", width=4, relief="solid", bd=1)
                sw.pack(side=LEFT, padx=2)
                self._color_swatches[key].append(sw)
            Button(self.players_frame, text="Clear", font=MED,
                   command=lambda k=key: self._clear_player_colors(k)).grid(
                row=row, column=3, sticky="w", padx=4)

        # One-line helper
        Label(self.players_frame,
              text="Type names, then click each player twice on the frame "
                   "(shirt + shorts) to lock their color ID through side swaps.",
              font=MED, fg="#555").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(6, 2))

        # Side-swap controls (row 3: slider + time, row 4: buttons + list, row 5: continue)
        self.swap_slider = Scale(self.players_frame, from_=0, to=max(self.total_frames - 1, 1),
                                 orient=HORIZONTAL, length=600, showvalue=False,
                                 command=self._on_slider_move)
        self.swap_slider.set(self.current_frame_idx)
        self.swap_slider.grid(row=3, column=0, columnspan=3, padx=10, pady=4, sticky="we")
        self.swap_slider.bind("<Motion>", self._on_slider_hover)
        self.swap_slider.bind("<Leave>", lambda _e: self._hide_preview())
        self.swap_slider.bind("<ButtonPress-1>", self._on_slider_hover)
        self.swap_slider.bind("<B1-Motion>", self._on_slider_hover)
        self.swap_slider.bind("<ButtonRelease-1>", lambda _e: self._hide_preview())
        self.swap_time_var = StringVar(value="0:00")
        Label(self.players_frame, textvariable=self.swap_time_var, font=MED,
              width=10, anchor="w").grid(row=3, column=3, sticky="w", padx=4)

        swap_btns = Frame(self.players_frame)
        swap_btns.grid(row=4, column=0, columnspan=4, sticky="w", padx=10, pady=2)
        Button(swap_btns, text="Mark swap here", font=MED,
               command=self._mark_swap).pack(side=LEFT, padx=(0, 4))
        Button(swap_btns, text="Remove selected", font=MED,
               command=self._remove_swap).pack(side=LEFT, padx=4)
        Label(swap_btns, text="  Swaps:", font=MED).pack(side=LEFT, padx=(10, 2))
        self.swap_list = Listbox(swap_btns, font=MED, height=2, width=22)
        self.swap_list.pack(side=LEFT, padx=4)

        Button(self.players_frame, text="Continue to player labeling ▶", font=BIG,
               command=self._enter_label_step).grid(
            row=5, column=0, columnspan=4, sticky="e", padx=10, pady=(6, 4))

        # ---- Player-position labeling widgets (step 6) ----
        self.label_slider = Scale(self.label_frame, from_=0, to=max(self.total_frames - 1, 1),
                                  orient=HORIZONTAL, length=600, showvalue=False,
                                  command=self._on_label_slider_move)
        self.label_slider.grid(row=0, column=0, columnspan=3, padx=10, pady=4, sticky="we")
        self.label_slider.bind("<Motion>", self._on_slider_hover)
        self.label_slider.bind("<Leave>", lambda _e: self._hide_preview())
        self.label_slider.bind("<ButtonPress-1>", self._on_slider_hover)
        self.label_slider.bind("<B1-Motion>", self._on_slider_hover)
        self.label_slider.bind("<ButtonRelease-1>", lambda _e: self._hide_preview())

        self.label_time_var = StringVar(value="0:00")
        Label(self.label_frame, textvariable=self.label_time_var, font=MED,
              width=10, anchor="w").grid(row=0, column=3, sticky="w", padx=4)

        Button(self.label_frame, text="Save boxes on this frame", font=MED,
               command=self._save_frame_labels).grid(row=1, column=0, sticky="w", padx=10, pady=4)
        Button(self.label_frame, text="Clear boxes on this frame", font=MED,
               command=self._clear_frame_labels).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        Button(self.label_frame, text=f"Next frame (+{LABEL_NEXT_FRAME_JUMP}) ▶", font=MED,
               command=self._jump_next_label_frame).grid(row=1, column=2, sticky="w", padx=4, pady=4)

        self.label_count_var = StringVar(value="0 frames labeled")
        Label(self.label_frame, textvariable=self.label_count_var, font=MED, fg="#444").grid(
            row=1, column=3, sticky="w", padx=10)

        Label(self.label_frame, text="Labeled frames (click to revisit):", font=MED).grid(
            row=2, column=0, sticky="nw", padx=10, pady=(6, 2))
        self.label_list = Listbox(self.label_frame, font=MED, height=4, width=40)
        self.label_list.grid(row=2, column=1, columnspan=3, padx=10, pady=(6, 4), sticky="w")
        self.label_list.bind("<<ListboxSelect>>", self._on_label_list_select)

        Button(self.label_frame, text="Continue to ball labeling ▶", font=BIG,
               command=self._enter_ball_step).grid(
            row=3, column=0, columnspan=4, sticky="e", padx=10, pady=(10, 4))

        # ---- Ball-position labeling widgets (step 7) ----
        self.ball_slider = Scale(self.ball_frame, from_=0, to=max(self.total_frames - 1, 1),
                                 orient=HORIZONTAL, length=600, showvalue=False,
                                 command=self._on_ball_slider_move)
        self.ball_slider.grid(row=0, column=0, columnspan=3, padx=10, pady=4, sticky="we")
        self.ball_slider.bind("<Motion>", self._on_slider_hover)
        self.ball_slider.bind("<Leave>", lambda _e: self._hide_preview())
        self.ball_slider.bind("<ButtonPress-1>", self._on_slider_hover)
        self.ball_slider.bind("<B1-Motion>", self._on_slider_hover)
        self.ball_slider.bind("<ButtonRelease-1>", lambda _e: self._hide_preview())

        self.ball_time_var = StringVar(value="0:00")
        Label(self.ball_frame, textvariable=self.ball_time_var, font=MED,
              width=10, anchor="w").grid(row=0, column=3, sticky="w", padx=4)

        Button(self.ball_frame, text="Clear ball box on this frame", font=MED,
               command=self._clear_ball_label).grid(row=1, column=0, sticky="w", padx=10, pady=4)
        Button(self.ball_frame, text=f"Next frame (+{BALL_NEXT_FRAME_JUMP}) ▶", font=MED,
               command=self._jump_next_ball_frame).grid(row=1, column=1, sticky="w", padx=4, pady=4)

        self.ball_count_var = StringVar(value="0 ball boxes")
        Label(self.ball_frame, textvariable=self.ball_count_var, font=MED, fg="#444").grid(
            row=1, column=2, sticky="w", padx=10)

        Label(self.ball_frame, text="Labeled ball frames (click to revisit):", font=MED).grid(
            row=2, column=0, sticky="nw", padx=10, pady=(6, 2))
        self.ball_list = Listbox(self.ball_frame, font=MED, height=4, width=40)
        self.ball_list.grid(row=2, column=1, columnspan=3, padx=10, pady=(6, 4), sticky="w")
        self.ball_list.bind("<<ListboxSelect>>", self._on_ball_list_select)

        self.load_frame()
        self.update_prompt()

    # ------------------------------------------------------------------
    # Frame loading / display
    # ------------------------------------------------------------------

    def jump_seconds(self, delta: int) -> None:
        new_idx = self.current_frame_idx + int(delta * self.fps)
        new_idx = max(0, min(new_idx, self.total_frames - 1))
        if new_idx == self.current_frame_idx:
            return
        self.current_frame_idx = new_idx
        # Only invalidate corners if we're still in the corners step.
        # Once corners are placed, the user may scrub around to find swap points.
        if self.step < 4 and self.corners:
            self.corners.clear()
            self.step = 0
            self.update_prompt()
        if self.step == 5:
            self.label_slider.set(new_idx)
            self.label_time_var.set(self._format_time(new_idx))
        elif self.step == 6:
            self.ball_slider.set(new_idx)
            self.ball_time_var.set(self._format_time(new_idx))
        self.load_frame()

    def _format_time(self, frame_idx: int) -> str:
        total_s = frame_idx / max(self.fps, 1.0)
        m, s = divmod(int(total_s), 60)
        return f"{m:d}:{s:02d}"

    def _on_slider_move(self, value) -> None:
        if self.step < 4:
            return
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        if idx == self.current_frame_idx:
            return
        self.current_frame_idx = idx
        self.swap_time_var.set(self._format_time(idx))
        self.load_frame()

    # ------------------------------------------------------------------
    # Scrubber hover preview (YouTube / WMP style)
    # ------------------------------------------------------------------

    _PREVIEW_W = 280
    _PREVIEW_DEBOUNCE_MS = 40

    def _frame_idx_from_event(self, event) -> int:
        width = max(self.swap_slider.winfo_width(), 1)
        # Tk Scale leaves a small trough margin; approximate with a 10px pad.
        pad = 10
        usable = max(width - 2 * pad, 1)
        x = max(0, min(event.x - pad, usable))
        total = max(self.total_frames - 1, 1)
        return int(round(x / usable * total))

    def _on_slider_hover(self, event) -> None:
        if self.step < 4:
            return
        idx = self._frame_idx_from_event(event)
        # Position the preview just above the cursor, clamped to screen.
        root_x = self.swap_slider.winfo_rootx() + event.x
        root_y = self.swap_slider.winfo_rooty()
        self._show_preview(idx, root_x, root_y)

    def _show_preview(self, frame_idx: int, root_x: int, root_y: int) -> None:
        if self.preview_win is None:
            win = Toplevel(self.root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.configure(bg="#111", bd=1, relief="solid")
            self._preview_img_label = Label(win, bg="#111")
            self._preview_img_label.pack()
            self._preview_time_label = Label(win, bg="#111", fg="white",
                                             font=MED, pady=2)
            self._preview_time_label.pack(fill="x")
            self.preview_win = win

        # Move the window so its bottom edge sits just above the slider row,
        # and it's horizontally centred on the cursor.
        self.preview_win.update_idletasks()
        pw = self.preview_win.winfo_reqwidth() or self._PREVIEW_W
        ph = self.preview_win.winfo_reqheight() or 180
        x = max(0, root_x - pw // 2)
        y = max(0, root_y - ph - 8)
        self.preview_win.geometry(f"+{x}+{y}")
        self.preview_win.deiconify()

        # Always refresh the time label so it tracks the cursor even between
        # debounced frame reads.
        self._preview_time_label.config(
            text=f"{self._format_time(frame_idx)}  (frame {frame_idx})")

        if frame_idx == self.preview_last_idx:
            return
        # Debounce the actual seek/decode so fast hover doesn't thrash ffmpeg.
        if self.preview_pending is not None:
            try:
                self.root.after_cancel(self.preview_pending)
            except Exception:
                pass
        self.preview_pending = self.root.after(
            self._PREVIEW_DEBOUNCE_MS,
            lambda fi=frame_idx: self._render_preview_frame(fi))

    def _render_preview_frame(self, frame_idx: int) -> None:
        self.preview_pending = None
        if self.preview_win is None or not self.preview_cap.isOpened():
            return
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.preview_cap.read()
        if not ok or frame is None:
            return
        h, w = frame.shape[:2]
        scale = self._PREVIEW_W / float(w)
        disp_w = self._PREVIEW_W
        disp_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (disp_w, disp_h))
        cv2.imwrite(self.preview_temp_png.name, resized)
        self.preview_img = PhotoImage(file=self.preview_temp_png.name)
        self._preview_img_label.config(image=self.preview_img)
        self.preview_last_idx = frame_idx

    def _hide_preview(self) -> None:
        if self.preview_pending is not None:
            try:
                self.root.after_cancel(self.preview_pending)
            except Exception:
                pass
            self.preview_pending = None
        if self.preview_win is not None:
            try:
                self.preview_win.withdraw()
            except Exception:
                pass

    def _mark_swap(self) -> None:
        if self.step < 4:
            return
        idx = self.current_frame_idx
        if idx in self.side_swaps:
            return
        self.side_swaps.append(idx)
        self.side_swaps.sort()
        self._refresh_swap_list()

    def _remove_swap(self) -> None:
        sel = self.swap_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.side_swaps):
            self.side_swaps.pop(idx)
            self._refresh_swap_list()

    def _refresh_swap_list(self) -> None:
        self.swap_list.delete(0, END)
        for i, frame_idx in enumerate(self.side_swaps, start=1):
            self.swap_list.insert(END, f"{i}.  {self._format_time(frame_idx)}  (frame {frame_idx})")

    def load_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            messagebox.showerror("Read failed", f"Could not read frame {self.current_frame_idx}")
            return

        # Scale to fit canvas bounds, preserving aspect ratio
        scale_w = MAX_CANVAS_W / self.frame_w
        scale_h = self.max_canvas_h / self.frame_h
        self.scale = min(scale_w, scale_h, 1.0)
        disp_w = int(self.frame_w * self.scale)
        disp_h = int(self.frame_h * self.scale)
        resized = cv2.resize(frame, (disp_w, disp_h)) if self.scale < 1.0 else frame

        cv2.imwrite(self.temp_png.name, resized)
        self.tk_image = PhotoImage(file=self.temp_png.name)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self._redraw_corners()
        self._redraw_labels()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def on_canvas_click(self, event) -> None:
        if self.step in (5, 6):
            # Start a rubber-band drag to draw a player or ball box.
            self._drag_start = (event.x, event.y)
            if self._drag_rect_id is not None:
                self.canvas.delete(self._drag_rect_id)
            outline = "#ffff00" if self.step == 5 else "#ff66ff"
            self._drag_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=outline, width=2, tags="label_draft")
            return
        if self.step == 4:
            # Color-pick mode: sample a 3x3 median around the click and
            # store it in the next empty near/far slot.
            self._capture_color_click(event)
            return
        if self.step >= 5:
            return  # other post-corners steps use their own handlers
        # Convert display coords back to original pixel coords
        orig_x = event.x / self.scale
        orig_y = event.y / self.scale
        self.corners.append((orig_x, orig_y))
        self.step += 1
        self._redraw_corners()
        self.update_prompt()

    def _on_canvas_drag(self, event) -> None:
        if self.step not in (5, 6) or self._drag_start is None or self._drag_rect_id is None:
            return
        x0, y0 = self._drag_start
        self.canvas.coords(self._drag_rect_id, x0, y0, event.x, event.y)

    def _on_canvas_release(self, event) -> None:
        if self.step not in (5, 6) or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        lx, rx = sorted((x0, x1))
        ty, by = sorted((y0, y1))
        min_px = LABEL_MIN_BOX_PX if self.step == 5 else BALL_MIN_BOX_PX
        if (rx - lx) < min_px or (by - ty) < min_px:
            return
        ox1 = lx / self.scale
        oy1 = ty / self.scale
        ox2 = rx / self.scale
        oy2 = by / self.scale
        frame = self.current_frame_idx
        if self.step == 5:
            self.player_labels.setdefault(frame, []).append((ox1, oy1, ox2, oy2))
            self._redraw_labels()
            self._refresh_label_list()
        else:
            # One ball per frame — overwrite any previous box for this frame.
            self.ball_labels[frame] = (ox1, oy1, ox2, oy2)
            self._redraw_ball_labels()
            self._refresh_ball_list()

    # ------------------------------------------------------------------
    # Step 5 (players_frame) — color picking
    # ------------------------------------------------------------------
    def _next_color_slot(self) -> tuple[str, int] | None:
        """Pick order: near[0] → near[1] → far[0] → far[1]. Returns None
        when all four are filled."""
        for key in ("near", "far"):
            for slot in range(2):
                if slot >= len(self.player_colors[key]):
                    return (key, slot)
        return None

    def _capture_color_click(self, event) -> None:
        slot = self._next_color_slot()
        if slot is None:
            return  # all 4 already picked; Clear to redo
        key, idx = slot
        # Read a 3x3 patch in display coords, convert to original-pixel
        # median for robustness, then BGR→Lab.
        ox = int(event.x / self.scale); oy = int(event.y / self.scale)
        r = 1  # 3x3 window
        # Re-read the current frame from the capture to avoid Tk compositing.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        h, w = frame.shape[:2]
        x1 = max(0, ox - r); y1 = max(0, oy - r)
        x2 = min(w, ox + r + 1); y2 = min(h, oy + r + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return
        bgr_median = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        lab = cv2.cvtColor(bgr_median.reshape(1, 1, 3),
                           cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        self.player_colors[key].append((float(lab[0]), float(lab[1]), float(lab[2])))
        self._color_pick_order.append((key, idx))
        # Update swatch background to picked RGB so user sees what was captured.
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(bgr_median[2]), int(bgr_median[1]), int(bgr_median[0]))
        sw = self._color_swatches[key][idx]
        sw.config(bg=hex_color, text="    ", fg=hex_color)

    def _clear_player_colors(self, key: str) -> None:
        self.player_colors[key] = []
        self._color_pick_order = [p for p in self._color_pick_order if p[0] != key]
        for sw in self._color_swatches[key]:
            sw.config(bg="#dddddd", text="  ?  ", fg="#666666")

    def _undo_last_color(self) -> bool:
        """Remove the most recently picked color. Returns True if one was
        removed."""
        if not self._color_pick_order:
            return False
        key, _slot = self._color_pick_order.pop()
        if self.player_colors[key]:
            self.player_colors[key].pop()
        # Redraw this player's swatches from remaining state.
        for i, sw in enumerate(self._color_swatches[key]):
            if i < len(self.player_colors[key]):
                lab = self.player_colors[key][i]
                bgr = cv2.cvtColor(np.array([[[lab[0], lab[1], lab[2]]]],
                                            dtype=np.uint8),
                                   cv2.COLOR_LAB2BGR)[0, 0]
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(bgr[2]), int(bgr[1]), int(bgr[0]))
                sw.config(bg=hex_color, text="    ", fg=hex_color)
            else:
                sw.config(bg="#dddddd", text="  ?  ", fg="#666666")
        return True

    def undo(self) -> None:
        if self.step == 6:
            # Back out of ball-labeling into player-labeling.
            self.ball_frame.pack_forget()
            self.step = 5
            self._clear_ball_canvas_markers()
            self.update_prompt()
        elif self.step == 5:
            # Back out of player-labeling into the names/swaps step.
            self.label_frame.pack_forget()
            self.step = 4
            self._clear_label_canvas_markers()
            self.update_prompt()
        elif self.step == 4:
            # Prefer undoing the last color pick if any exist; only back out
            # to corner 4 once the color picker is empty.
            if self._undo_last_color():
                return
            self.players_frame.pack_forget()
            self.step = 3
            self.corners.pop()
            self._redraw_corners()
            self.update_prompt()
        elif self.corners:
            self.corners.pop()
            self.step -= 1
            self._redraw_corners()
            self.update_prompt()

    # ------------------------------------------------------------------
    # Step 6: player-position labeling
    # ------------------------------------------------------------------

    def _enter_label_step(self) -> None:
        near = self.near_entry.get().strip()
        far = self.far_entry.get().strip()
        if not near or not far:
            messagebox.showwarning("Missing names",
                                   "Please type both player names before labeling.")
            return
        self.step = 5
        self.update_prompt()

    def _on_label_slider_move(self, value) -> None:
        if self.step != 5:
            return
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        if idx == self.current_frame_idx:
            return
        self.current_frame_idx = idx
        self.label_time_var.set(self._format_time(idx))
        self.load_frame()

    def _save_frame_labels(self) -> None:
        # Boxes are already stored on drag-release, so this is a user-visible
        # confirmation plus a guard against accidentally leaving zero boxes.
        frame = self.current_frame_idx
        boxes = self.player_labels.get(frame, [])
        if not boxes:
            messagebox.showwarning("No boxes",
                                   "Draw at least one box on this frame first.")
            return
        self._refresh_label_list()
        messagebox.showinfo("Saved",
                            f"Frame {frame}: {len(boxes)} box(es) saved.")

    def _clear_frame_labels(self) -> None:
        frame = self.current_frame_idx
        self.player_labels.pop(frame, None)
        self._redraw_labels()
        self._refresh_label_list()

    def _jump_next_label_frame(self) -> None:
        if self.step != 5:
            return
        new_idx = min(self.current_frame_idx + LABEL_NEXT_FRAME_JUMP,
                      self.total_frames - 1)
        if new_idx == self.current_frame_idx:
            return
        self.current_frame_idx = new_idx
        self.label_slider.set(new_idx)
        self.label_time_var.set(self._format_time(new_idx))
        self.load_frame()

    def _on_label_list_select(self, _event) -> None:
        sel = self.label_list.curselection()
        if not sel:
            return
        frames = sorted(self.player_labels.keys())
        if sel[0] >= len(frames):
            return
        target = frames[sel[0]]
        self.current_frame_idx = target
        self.label_slider.set(target)
        self.label_time_var.set(self._format_time(target))
        self.load_frame()

    def _refresh_label_list(self) -> None:
        self.label_list.delete(0, END)
        for frame in sorted(self.player_labels.keys()):
            n = len(self.player_labels[frame])
            self.label_list.insert(END,
                f"{self._format_time(frame)}  (frame {frame}) — {n} box(es)")
        total = len(self.player_labels)
        self.label_count_var.set(f"{total} frame(s) labeled")

    def _clear_label_canvas_markers(self) -> None:
        self.canvas.delete("label_box")
        self._label_box_ids = []
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _redraw_labels(self) -> None:
        """Draw stored player boxes for the current frame onto the canvas."""
        self._clear_label_canvas_markers()
        if self.step != 5:
            return
        boxes = self.player_labels.get(self.current_frame_idx, [])
        for i, (ox1, oy1, ox2, oy2) in enumerate(boxes):
            dx1 = ox1 * self.scale
            dy1 = oy1 * self.scale
            dx2 = ox2 * self.scale
            dy2 = oy2 * self.scale
            rid = self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2,
                outline="#00ffcc", width=2, tags="label_box")
            self.canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                                    text=f"P{i+1}", fill="#00ffcc", font=BIG,
                                    tags="label_box")
            self._label_box_ids.append(rid)

    # ------------------------------------------------------------------
    # Step 7: ball-position labeling
    # ------------------------------------------------------------------

    def _enter_ball_step(self) -> None:
        if not self.player_labels:
            if not messagebox.askyesno(
                "No player boxes",
                "You haven't drawn any player boxes. Continue to ball labeling anyway?"):
                return
        self.step = 6
        self.update_prompt()

    def _on_ball_slider_move(self, value) -> None:
        if self.step != 6:
            return
        try:
            idx = int(float(value))
        except (TypeError, ValueError):
            return
        if idx == self.current_frame_idx:
            return
        self.current_frame_idx = idx
        self.ball_time_var.set(self._format_time(idx))
        self.load_frame()

    def _clear_ball_label(self) -> None:
        self.ball_labels.pop(self.current_frame_idx, None)
        self._redraw_ball_labels()
        self._refresh_ball_list()

    def _jump_next_ball_frame(self) -> None:
        if self.step != 6:
            return
        new_idx = min(self.current_frame_idx + BALL_NEXT_FRAME_JUMP,
                      self.total_frames - 1)
        if new_idx == self.current_frame_idx:
            return
        self.current_frame_idx = new_idx
        self.ball_slider.set(new_idx)
        self.ball_time_var.set(self._format_time(new_idx))
        self.load_frame()

    def _on_ball_list_select(self, _event) -> None:
        sel = self.ball_list.curselection()
        if not sel:
            return
        frames = sorted(self.ball_labels.keys())
        if sel[0] >= len(frames):
            return
        target = frames[sel[0]]
        self.current_frame_idx = target
        self.ball_slider.set(target)
        self.ball_time_var.set(self._format_time(target))
        self.load_frame()

    def _refresh_ball_list(self) -> None:
        self.ball_list.delete(0, END)
        for frame in sorted(self.ball_labels.keys()):
            self.ball_list.insert(END,
                f"{self._format_time(frame)}  (frame {frame})")
        self.ball_count_var.set(f"{len(self.ball_labels)} ball box(es)")

    def _clear_ball_canvas_markers(self) -> None:
        self.canvas.delete("ball_box")
        self._ball_box_ids = []
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _redraw_ball_labels(self) -> None:
        self._clear_ball_canvas_markers()
        if self.step != 6:
            return
        box = self.ball_labels.get(self.current_frame_idx)
        if box is None:
            return
        ox1, oy1, ox2, oy2 = box
        dx1 = ox1 * self.scale; dy1 = oy1 * self.scale
        dx2 = ox2 * self.scale; dy2 = oy2 * self.scale
        rid = self.canvas.create_rectangle(
            dx1, dy1, dx2, dy2,
            outline="#ff66ff", width=2, tags="ball_box")
        self.canvas.create_text(dx1 + 4, dy1 + 4, anchor="nw",
                                text="ball", fill="#ff66ff", font=BIG,
                                tags="ball_box")
        self._ball_box_ids.append(rid)

    def _redraw_corners(self) -> None:
        # Remove old markers (tagged)
        self.canvas.delete("marker")
        for i, (ox, oy) in enumerate(self.corners):
            dx = ox * self.scale
            dy = oy * self.scale
            color = CORNER_STEPS[i][2]
            r = 10
            self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r,
                                    outline="white", fill=color, width=2, tags="marker")
            self.canvas.create_text(dx + 14, dy, anchor="w",
                                    text=f"{i+1}", fill="white", font=HUGE, tags="marker")
        # Draw connecting lines between placed corners
        if len(self.corners) >= 2:
            pts = [(ox * self.scale, oy * self.scale) for ox, oy in self.corners]
            for i in range(len(pts) - 1):
                self.canvas.create_line(*pts[i], *pts[i+1],
                                        fill="#ffffcc", width=2, dash=(4, 2), tags="marker")
            if len(pts) == 4:
                self.canvas.create_line(*pts[3], *pts[0],
                                        fill="#ffffcc", width=2, dash=(4, 2), tags="marker")

    def update_prompt(self) -> None:
        self.undo_btn.config(state=NORMAL if (self.corners or self.step >= 4) else DISABLED)

        if self.step < 4:
            title, helptext, _ = CORNER_STEPS[self.step]
            self.title_var.set(f"Step {self.step + 1} of 7 — Click the {title}")
            self.help_var.set(helptext)
            self.players_frame.pack_forget()
            self.label_frame.pack_forget()
            self.ball_frame.pack_forget()
            self.canvas.config(cursor="crosshair")
            self.save_btn.config(state=DISABLED)
            self._clear_label_canvas_markers()
            self._clear_ball_canvas_markers()
        elif self.step == 4:
            self.title_var.set("Step 5 of 7 — Players, colors, and side-swaps")
            self.help_var.set(
                "Name both players, then click twice on each to capture their "
                "shirt and shorts colors for automatic identity tracking. "
                "Mark any side-swaps on the slider. Then continue."
            )
            self.players_frame.pack(side="top", fill="x", padx=20, before=self.canvas)
            self.label_frame.pack_forget()
            self.ball_frame.pack_forget()
            self.canvas.config(cursor="crosshair")
            self.save_btn.config(state=NORMAL)
            self._clear_label_canvas_markers()
            self._clear_ball_canvas_markers()
        elif self.step == 5:
            self.title_var.set("Step 6 of 7 — Label both players on many frames")
            self.help_var.set(
                "Scrub to any moment, then click-and-drag on the frame to draw a "
                "tight box around each player. Aim for ~50+ varied frames across "
                "the match. Use Next frame to jump ahead ~2s. When done, press "
                "Continue to ball labeling."
            )
            self.players_frame.pack_forget()
            self.label_frame.pack(side="top", fill="x", padx=20, before=self.canvas)
            self.ball_frame.pack_forget()
            self.label_slider.set(self.current_frame_idx)
            self.label_time_var.set(self._format_time(self.current_frame_idx))
            self.canvas.config(cursor="tcross")
            self.save_btn.config(state=NORMAL)
            self._clear_ball_canvas_markers()
            self._refresh_label_list()
            self._redraw_labels()
        else:  # step == 6
            self.title_var.set("Step 7 of 7 — Label the ball on many frames")
            self.help_var.set(
                "Scrub to moments where the ball is clearly visible (contact, "
                "mid-flight, bounce), then drag a small tight box around the "
                "ball. One box per frame. Aim for ~30+ varied frames. "
                "Press Save setup when done."
            )
            self.players_frame.pack_forget()
            self.label_frame.pack_forget()
            self.ball_frame.pack(side="top", fill="x", padx=20, before=self.canvas)
            self.ball_slider.set(self.current_frame_idx)
            self.ball_time_var.set(self._format_time(self.current_frame_idx))
            self.canvas.config(cursor="tcross")
            self.save_btn.config(state=NORMAL)
            self._clear_label_canvas_markers()
            self._refresh_ball_list()
            self._redraw_ball_labels()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> None:
        near = self.near_entry.get().strip()
        far = self.far_entry.get().strip()
        if not near or not far:
            messagebox.showwarning("Missing names", "Please type both player names first.")
            return

        sidecar = {
            "video": str(self.video_path.resolve()),
            "created": datetime.now(timezone.utc).isoformat(),
            "frame_width": self.frame_w,
            "frame_height": self.frame_h,
            "reference_frame_index": self.current_frame_idx,
            "reference_frame_ms": int(1000 * self.current_frame_idx / self.fps),
            "court_corners_pixel": [
                {"label": "near_left",  "x": self.corners[0][0], "y": self.corners[0][1]},
                {"label": "near_right", "x": self.corners[1][0], "y": self.corners[1][1]},
                {"label": "far_right",  "x": self.corners[2][0], "y": self.corners[2][1]},
                {"label": "far_left",   "x": self.corners[3][0], "y": self.corners[3][1]},
            ],
            "players": {"near": near, "far": far},
            "side_swaps": list(self.side_swaps),
            "player_labels": [
                {
                    "frame": frame,
                    "time_ms": int(1000 * frame / self.fps),
                    "boxes": [
                        {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                        for b in boxes
                    ],
                }
                for frame, boxes in sorted(self.player_labels.items())
                if boxes
            ],
            "ball_labels": [
                {
                    "frame": frame,
                    "time_ms": int(1000 * frame / self.fps),
                    "bbox": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                }
                for frame, b in sorted(self.ball_labels.items())
            ],
            # Only emit when both players have at least one picked color.
            # Keys are `a` (= P1 = the NEAR player at preflight time, whose
            # name is in `players.near`) and `b` (= P2 = FAR at preflight).
            # These are HUMAN-identity keys — they do not flip on side swaps.
            **({"player_colors": {
                "a": [list(c) for c in self.player_colors["near"]],
                "b": [list(c) for c in self.player_colors["far"]],
            }} if self.player_colors["near"] and self.player_colors["far"] else {}),
        }

        out_path = self.video_path.with_suffix(self.video_path.suffix + ".setup.json")
        out_path.write_text(json.dumps(sidecar, indent=2))
        messagebox.showinfo("Saved", f"Setup saved to:\n{out_path.name}\n\nYou can now tag the video.")
        self.cleanup()
        self.root.destroy()

    def cleanup(self) -> None:
        self._hide_preview()
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.preview_cap.release()
        except Exception:
            pass
        try:
            os.unlink(self.temp_png.name)
        except Exception:
            pass
        try:
            os.unlink(self.preview_temp_png.name)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", help="Path to match video")
    args = parser.parse_args()

    video_path: str | None = args.video
    if not video_path:
        root = Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Pick a tennis match video to set up",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        root.destroy()
        if not video_path:
            return 0

    path = Path(video_path)
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    root = Tk()
    Preflight(root, path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

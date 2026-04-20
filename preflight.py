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
    Entry, Frame, Label, Listbox, PhotoImage, Scale, StringVar, Tk,
    filedialog, messagebox,
)

import cv2

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


class Preflight:
    def __init__(self, root: Tk, video_path: Path) -> None:
        self.root = root
        self.video_path = video_path
        root.title(f"Set up match — {video_path.name}")
        root.geometry("1400x900")
        root.minsize(1100, 800)

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

        # State
        self.corners: list[tuple[float, float]] = []   # in ORIGINAL pixel coords
        self.step = 0                                   # 0..3 = corners, 4 = players+swaps
        # Frames where the players switch ends. Populated only during step 4.
        self.side_swaps: list[int] = []

        # ---- Layout ----
        self.title_var = StringVar()
        self.help_var = StringVar()

        top = Frame(root, pady=10)
        top.pack(fill="x")
        Label(top, textvariable=self.title_var, font=HUGE).pack()
        Label(top, textvariable=self.help_var, font=BIG, fg="#444").pack(pady=(4, 0))

        # ---- Navigation: earlier / later frame ----
        nav = Frame(root, pady=4)
        nav.pack(fill="x")
        Label(nav, text="Is this a good frame? If the players aren't visible yet:", font=MED).pack(side=LEFT, padx=20)
        Button(nav, text="< 10 seconds earlier", font=MED, command=lambda: self.jump_seconds(-10)).pack(side=LEFT, padx=4)
        Button(nav, text="10 seconds later >", font=MED, command=lambda: self.jump_seconds(10)).pack(side=LEFT, padx=4)

        # ---- Canvas ----
        self.canvas = Canvas(root, bg="#222", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True, padx=20, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # ---- Player-name + side-swap section (hidden until corners done) ----
        self.players_frame = Frame(root, pady=6)
        Label(self.players_frame, text="Player at the BOTTOM of the screen (near camera):", font=BIG).grid(row=0, column=0, sticky="w", padx=10, pady=4)
        self.near_entry = Entry(self.players_frame, font=BIG, width=40)
        self.near_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=4, sticky="w")
        Label(self.players_frame, text="Player at the TOP of the screen (far side):", font=BIG).grid(row=1, column=0, sticky="w", padx=10, pady=4)
        self.far_entry = Entry(self.players_frame, font=BIG, width=40)
        self.far_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=4, sticky="w")

        # Side-swap controls
        Label(self.players_frame,
              text="When do the players switch ends? (optional — drag the slider, then press 'Mark swap here')",
              font=MED, fg="#555").grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 2))

        self.swap_slider = Scale(self.players_frame, from_=0, to=max(self.total_frames - 1, 1),
                                 orient=HORIZONTAL, length=600, showvalue=False,
                                 command=self._on_slider_move)
        self.swap_slider.set(self.current_frame_idx)
        self.swap_slider.grid(row=3, column=0, columnspan=2, padx=10, pady=4, sticky="we")
        self.swap_time_var = StringVar(value="0:00")
        Label(self.players_frame, textvariable=self.swap_time_var, font=MED,
              width=10, anchor="w").grid(row=3, column=2, sticky="w", padx=4)

        Button(self.players_frame, text="Mark swap here", font=MED,
               command=self._mark_swap).grid(row=4, column=0, sticky="w", padx=10, pady=4)
        Button(self.players_frame, text="Remove selected", font=MED,
               command=self._remove_swap).grid(row=4, column=1, sticky="w", padx=4, pady=4)

        Label(self.players_frame, text="Swaps marked:", font=MED).grid(
            row=5, column=0, sticky="nw", padx=10, pady=(6, 2))
        self.swap_list = Listbox(self.players_frame, font=MED, height=4, width=30)
        self.swap_list.grid(row=5, column=1, columnspan=2, padx=10, pady=(6, 4), sticky="w")

        # ---- Bottom buttons ----
        btns = Frame(root, pady=10)
        btns.pack(fill="x")
        self.undo_btn = Button(btns, text="Undo last click", font=BIG, command=self.undo, state=DISABLED)
        self.undo_btn.pack(side=LEFT, padx=20)
        self.save_btn = Button(btns, text="Save setup", font=BIG, command=self.save, state=DISABLED)
        self.save_btn.pack(side=RIGHT, padx=20)

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
        scale_h = MAX_CANVAS_H / self.frame_h
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

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def on_canvas_click(self, event) -> None:
        if self.step >= 4:
            return  # corners done; player entry is via keyboard
        # Convert display coords back to original pixel coords
        orig_x = event.x / self.scale
        orig_y = event.y / self.scale
        self.corners.append((orig_x, orig_y))
        self.step += 1
        self._redraw_corners()
        self.update_prompt()

    def undo(self) -> None:
        if self.step == 4:
            # User was in player-name step; go back to corner 4
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
        self.undo_btn.config(state=NORMAL if self.corners or self.step == 4 else DISABLED)

        if self.step < 4:
            title, helptext, _ = CORNER_STEPS[self.step]
            self.title_var.set(f"Step {self.step + 1} of 5 — Click the {title}")
            self.help_var.set(helptext)
            self.players_frame.pack_forget()
            self.save_btn.config(state=DISABLED)
        else:
            self.title_var.set("Step 5 of 5 — Players and side-swaps")
            self.help_var.set(
                "Type both player names. If the players switch ends during "
                "the match, scrub to each swap and press Mark swap here. "
                "Skip the scrubber if there are no swaps. Then press Save."
            )
            self.players_frame.pack(fill="x", padx=20)
            self.save_btn.config(state=NORMAL)

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
        }

        out_path = self.video_path.with_suffix(self.video_path.suffix + ".setup.json")
        out_path.write_text(json.dumps(sidecar, indent=2))
        messagebox.showinfo("Saved", f"Setup saved to:\n{out_path.name}\n\nYou can now tag the video.")
        self.cleanup()
        self.root.destroy()

    def cleanup(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            os.unlink(self.temp_png.name)
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

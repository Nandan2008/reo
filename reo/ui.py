"""reo's face: a reactor on a dark screen, with the conversation beside it.

Tkinter insists on owning the main thread, so the voice loop runs in a worker and
talks to the window through a queue. Nothing here touches Tk from that thread.
"""
import math
import queue
import threading
import tkinter as tk
from datetime import datetime

BACKGROUND = "#05080c"
GRID = "#0e1a26"
EDGE = "#16293a"
TEXT = "#b9cfe2"
MUTED = "#4a6176"

# Colour and how fast the rings turn, per state.
STATES = {
    "idle":      ("#1d7fa8", 0.35),
    "listening": ("#19d2e6", 2.60),
    "thinking":  ("#d8a12e", 1.80),
    "speaking":  ("#2fd389", 1.10),
}

# Each ring: radius as a fraction of the reactor, line width, arc span, direction.
RINGS = (
    (1.00, 2, 110, +1.00),
    (0.86, 1, 40,  -1.60),
    (0.72, 3, 200, +0.70),
    (0.56, 1, 70,  -2.30),
    (0.40, 2, 150, +1.30),
)

TRANSCRIPT = 14          # lines kept on screen
FRAME_MS = 33


class Display:
    def __init__(self, title="REO", subtitle=""):
        self._events = queue.Queue()
        self._state = "idle"
        self._spin = 0.0
        self._pulse = 0.0
        self._lines = []
        self._subtitle = subtitle
        self.closed = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1180x720")
        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _: self._toggle_fullscreen())
        self.root.bind("<F11>", lambda _: self._toggle_fullscreen())

        self.canvas = tk.Canvas(self.root, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._tick()
        self._drain()

    # ---- called from the voice thread -------------------------------------

    def set_state(self, name: str) -> None:
        self._events.put(("state", name, None))

    def line(self, who: str, text: str) -> None:
        self._events.put(("line", who, text))

    def run(self, worker) -> None:
        threading.Thread(target=worker, daemon=True).start()
        self.root.mainloop()

    # ---- main thread only --------------------------------------------------

    def _toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    def _drain(self) -> None:
        try:
            while True:
                kind, a, b = self._events.get_nowait()
                if kind == "state":
                    self._state = a if a in STATES else "idle"
                else:
                    self._lines.append((a, b))
                    del self._lines[:-TRANSCRIPT]
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(40, self._drain)

    # ---- drawing -----------------------------------------------------------

    def _brackets(self, w, h):
        """Corner marks. They do the work of a frame without boxing anything in."""
        arm, inset = 34, 22
        for cx, cy, dx, dy in ((inset, inset, 1, 1), (w - inset, inset, -1, 1),
                               (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1)):
            self.canvas.create_line(cx, cy, cx + arm * dx, cy, fill=EDGE)
            self.canvas.create_line(cx, cy, cx, cy + arm * dy, fill=EDGE)

    def _reactor(self, cx, cy, size, colour):
        c = self.canvas
        for fraction, width, span, rate in RINGS:
            r = size * fraction
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=GRID, width=1)
            start = (self._spin * rate) % 360
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=start, extent=span,
                         style="arc", outline=colour, width=width)

        # tick marks around the outside, a few brighter than the rest
        outer = size * 1.12
        for i in range(60):
            angle = math.radians(i * 6 + self._spin * 0.25)
            length = 12 if i % 5 == 0 else 5
            fill = colour if i % 15 == 0 else EDGE
            x1, y1 = cx + outer * math.cos(angle), cy + outer * math.sin(angle)
            x2 = cx + (outer + length) * math.cos(angle)
            y2 = cy + (outer + length) * math.sin(angle)
            c.create_line(x1, y1, x2, y2, fill=fill)

        core = size * 0.18 * (1 + 0.10 * math.sin(self._pulse))
        for scale, width in ((1.0, 2), (0.62, 1)):
            r = core * scale
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=colour, width=width)
        r = core * 0.30
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=colour, outline="")

    def _transcript(self, x, y, width, colour):
        c = self.canvas
        c.create_text(x, y, text="TRANSCRIPT", anchor="nw", fill=MUTED,
                      font=("Consolas", 9, "bold"))
        c.create_line(x, y + 18, x + width, y + 18, fill=EDGE)

        row = y + 32
        for who, text in self._lines[-TRANSCRIPT:]:
            label = {"you": "YOU", "reo": "REO"}.get(who, "")
            fill = {"you": TEXT, "reo": colour}.get(who, MUTED)
            if label:
                c.create_text(x, row, text=label, anchor="nw", fill=MUTED,
                              font=("Consolas", 9))
            # wrap by hand so the label column stays clean
            c.create_text(x + 38, row, text=text[:120], anchor="nw", fill=fill,
                          font=("Consolas", 10), width=width - 44)
            row += 20 + 14 * (len(text) // 70)

    def _tick(self) -> None:
        colour, speed = STATES[self._state]
        self._spin = (self._spin + speed * 1.6) % 3600
        self._pulse = (self._pulse + speed * 0.05) % (2 * math.pi)

        c = self.canvas
        c.delete("all")
        w = self.canvas.winfo_width() or 1180
        h = self.canvas.winfo_height() or 720

        self._brackets(w, h)

        # header
        c.create_text(46, 40, text="R  E  O", anchor="w", fill=TEXT,
                      font=("Consolas", 22, "bold"))
        if self._subtitle:
            c.create_text(46, 68, text=self._subtitle.upper(), anchor="w", fill=MUTED,
                          font=("Consolas", 9))
        c.create_text(w - 46, 40, text=datetime.now().strftime("%H:%M:%S"), anchor="e",
                      fill=MUTED, font=("Consolas", 16))
        c.create_text(w - 46, 64, text=datetime.now().strftime("%a %d %b %Y").upper(),
                      anchor="e", fill=MUTED, font=("Consolas", 9))
        c.create_line(46, 88, w - 46, 88, fill=EDGE)

        # reactor on the left, conversation on the right
        panel = min(520, max(360, w * 0.42))
        size = min(panel * 0.36, (h - 220) * 0.42)
        self._reactor(46 + panel / 2, h / 2 - 10, size, colour)

        c.create_text(46 + panel / 2, h / 2 + size * 1.5, text=self._state.upper(),
                      fill=colour, font=("Consolas", 14, "bold"))

        self._transcript(46 + panel, 120, w - panel - 92, colour)

        c.create_text(46, h - 40, text="ESC  FULLSCREEN        SAY GOODBYE TO QUIT",
                      anchor="w", fill=MUTED, font=("Consolas", 8))

        if not self.closed:
            self.root.after(FRAME_MS, self._tick)

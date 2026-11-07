"""reo's face: a reactor on a dark screen, with the conversation beside it.

Tkinter insists on owning the main thread, so the voice loop runs in a worker and
talks to the window through a queue. Nothing here touches Tk from that thread.

Only the reactor is redrawn each frame. Redrawing the whole canvas instead cost
6% of a core and made transcription 30% slower: Python's GIL means the drawing
competes with the voice loop for the same interpreter.
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

COILS = 10               # the windings around the core
GLOW_LAYERS = 7          # Tk has no alpha, so a halo is rings fading to the background


def _blend(first: str, second: str, amount: float) -> str:
    """Mix two #rrggbb colours. Tk has no alpha, so glow is mixed by hand."""
    amount = max(0.0, min(1.0, amount))
    a = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))

TRANSCRIPT = 14          # lines kept on screen
FRAME_MS = 40            # 25fps is plenty for a turning ring
SPIN = "spin"            # the only items redrawn every frame
LIVE = "live"            # clock and system readout, redrawn once a second


class Display:
    def __init__(self, title="REO", subtitle=""):
        self._events = queue.Queue()
        self._state = "idle"
        self._spin = 0.0
        self._pulse = 0.0
        self._lines = []
        self._subtitle = subtitle
        self._size = (0, 0)
        self._stale = True        # the fixed layer needs drawing
        self._second = None
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
        self._stale = True

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
                self._stale = True     # both change what the fixed layer shows
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(40, self._drain)

    # ---- the layer that only changes on an event ---------------------------

    def _frame(self, w, h, colour) -> None:
        """Angled corners and edge rails, the furniture of a heads-up display."""
        c = self.canvas
        arm, inset, cut = 56, 20, 14
        for cx, cy, dx, dy in ((inset, inset, 1, 1), (w - inset, inset, -1, 1),
                               (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1)):
            c.create_line(cx + cut * dx, cy, cx + arm * dx, cy, fill=colour)
            c.create_line(cx, cy + cut * dy, cx, cy + arm * dy, fill=colour)
            c.create_line(cx + cut * dx, cy, cx, cy + cut * dy, fill=colour)  # the mitre

        # rails down both sides, ticking every 40px
        for y in range(140, h - 120, 40):
            long = (y // 40) % 4 == 0
            c.create_line(inset, y, inset + (12 if long else 6), y,
                          fill=EDGE if not long else GRID)
            c.create_line(w - inset, y, w - inset - (12 if long else 6), y,
                          fill=EDGE if not long else GRID)

    def _draw_fixed(self, w, h, colour) -> None:
        c = self.canvas
        c.delete("all")

        self._frame(w, h, EDGE)

        c.create_text(46, 40, text="R  E  O", anchor="w", fill=TEXT,
                      font=("Consolas", 22, "bold"))
        c.create_text(141, 40, text="//", anchor="w", fill=colour,
                      font=("Consolas", 22, "bold"))
        if self._subtitle:
            c.create_text(46, 68, text=self._subtitle.upper(), anchor="w", fill=MUTED,
                          font=("Consolas", 9))
        c.create_line(46, 88, w - 46, 88, fill=EDGE)
        c.create_line(46, 88, 200, 88, fill=colour, width=2)

        panel, size = self._reactor_box(w, h)
        mid = 46 + panel / 2
        c.create_text(mid, h / 2 + size * 1.5, text=self._state.upper(),
                      fill=colour, font=("Consolas", 14, "bold"))
        c.create_line(mid - 60, h / 2 + size * 1.5 + 16, mid + 60, h / 2 + size * 1.5 + 16,
                      fill=EDGE)

        self._transcript(46 + panel, 120, w - panel - 92, colour)

        c.create_text(46, h - 40, text="ESC  FULLSCREEN        SAY GOODBYE TO QUIT",
                      anchor="w", fill=MUTED, font=("Consolas", 8))

    def _transcript(self, x, y, width, colour) -> None:
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
            c.create_text(x + 38, row, text=text[:160], anchor="nw", fill=fill,
                          font=("Consolas", 10), width=width - 44)
            row += 20 + 14 * (len(text) // 70)

    # ---- the layer that moves ----------------------------------------------

    def _readout(self, w, h, colour) -> None:
        """Live system figures under the reactor. Once a second, not every frame."""
        c = self.canvas
        panel, size = self._reactor_box(w, h)
        x = 46
        y = h / 2 + size * 1.5 + 44

        try:
            import psutil
            battery = psutil.sensors_battery()
            rows = [
                ("CPU", f"{psutil.cpu_percent():.0f}%"),
                ("MEM", f"{psutil.virtual_memory().percent:.0f}%"),
                ("PWR", f"{battery.percent:.0f}%" if battery else "AC"),
            ]
        except Exception:
            rows = [("CPU", "--"), ("MEM", "--"), ("PWR", "--")]

        width = panel - 40
        for i, (label, value) in enumerate(rows):
            row = y + i * 22
            c.create_text(x, row, text=label, anchor="w", fill=MUTED,
                          font=("Consolas", 9), tags=LIVE)
            c.create_text(x + width, row, text=value, anchor="e", fill=colour,
                          font=("Consolas", 10, "bold"), tags=LIVE)
            # a thin bar for the ones that are percentages
            if value.endswith("%"):
                filled = int(value[:-1]) / 100
                bar_x, bar_w = x + 44, width - 90
                c.create_line(bar_x, row, bar_x + bar_w, row, fill=GRID, tags=LIVE)
                c.create_line(bar_x, row, bar_x + bar_w * filled, row,
                              fill=colour, width=2, tags=LIVE)

    @staticmethod
    def _reactor_box(w, h):
        panel = min(520, max(360, w * 0.42))
        return panel, min(panel * 0.36, (h - 220) * 0.42)

    def _ring(self, cx, cy, r, **kw):
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, tags=SPIN, **kw)

    def _reactor(self, cx, cy, size, colour) -> None:
        """An arc reactor: a glowing core inside a ring of coils.

        Five thin circles was not it. The light has to come from the middle, so
        the core is layered from near-white outwards and a halo fades into the
        background underneath everything else.
        """
        c = self.canvas
        pulse = 1 + 0.07 * math.sin(self._pulse)

        # halo, drawn first so everything sits on top of it
        for i in range(GLOW_LAYERS, 0, -1):
            fraction = i / GLOW_LAYERS
            r = size * (0.62 + 0.55 * fraction) * pulse
            self._ring(cx, cy, r, outline=_blend(BACKGROUND, colour, 0.16 * (1 - fraction)),
                       width=max(2, int(size * 0.10 * fraction)))

        # housing
        self._ring(cx, cy, size, outline=_blend(BACKGROUND, colour, 0.55), width=3)
        self._ring(cx, cy, size * 0.95, outline=GRID, width=1)

        # the coils: thick segments with gaps, turning slowly
        gap, span = 36.0 / COILS, 324.0 / COILS
        for i in range(COILS):
            start = (self._spin * 0.25 + i * (span + gap)) % 360
            lit = (i % 3 == 0)
            for r, width in ((size * 0.80, max(3, size * 0.11)),):
                c.create_arc(cx - r, cy - r, cx + r, cy + r, start=start, extent=span,
                             style="arc", width=width, tags=SPIN,
                             outline=colour if lit else _blend(BACKGROUND, colour, 0.45))

        # inner chamber and its spokes
        self._ring(cx, cy, size * 0.60, outline=_blend(BACKGROUND, colour, 0.70), width=2)
        for i in range(6):
            angle = math.radians(i * 60 - self._spin * 0.6)
            cos, sin = math.cos(angle), math.sin(angle)
            c.create_line(cx + size * 0.40 * cos, cy + size * 0.40 * sin,
                          cx + size * 0.58 * cos, cy + size * 0.58 * sin,
                          fill=_blend(BACKGROUND, colour, 0.8), width=2, tags=SPIN)

        # the core, bright in the middle and falling off outwards
        for step, (scale, mix) in enumerate(((0.40, 0.35), (0.30, 0.60),
                                             (0.22, 0.85), (0.15, 1.00))):
            r = size * scale * pulse
            self._ring(cx, cy, r, fill=_blend(colour, "#ffffff", mix * 0.55),
                       outline="")
        r = size * 0.07 * pulse
        self._ring(cx, cy, r, fill="#ffffff", outline="")

        # a sweep around the outside, so it reads as turning
        for rate, extent, width in ((1.0, 70, 2), (-1.7, 28, 2)):
            r = size * 1.10
            c.create_arc(cx - r, cy - r, cx + r, cy + r, start=(self._spin * rate) % 360,
                         extent=extent, style="arc", outline=colour, width=width, tags=SPIN)

        outer = size * 1.20
        for i in range(0, 60, 3):
            angle = math.radians(i * 6 + self._spin * 0.25)
            length = 10 if i % 5 == 0 else 4
            cos, sin = math.cos(angle), math.sin(angle)
            c.create_line(cx + outer * cos, cy + outer * sin,
                          cx + (outer + length) * cos, cy + (outer + length) * sin,
                          fill=colour if i % 15 == 0 else EDGE, tags=SPIN)

    def _tick(self) -> None:
        colour, speed = STATES[self._state]
        self._spin = (self._spin + speed * 2.0) % 3600
        self._pulse = (self._pulse + speed * 0.06) % (2 * math.pi)

        c = self.canvas
        w = c.winfo_width() or 1180
        h = c.winfo_height() or 720
        if (w, h) != self._size:
            self._size, self._stale = (w, h), True
        if self._stale:
            self._stale = False
            self._draw_fixed(w, h, colour)
            self._second = None        # the clock went with everything else

        now = datetime.now()
        if now.second != self._second:
            self._second = now.second
            c.delete(LIVE)
            c.create_text(w - 46, 40, text=now.strftime("%H:%M:%S"), anchor="e",
                          fill=TEXT, font=("Consolas", 18), tags=LIVE)
            c.create_text(w - 46, 64, text=now.strftime("%a %d %b %Y").upper(),
                          anchor="e", fill=MUTED, font=("Consolas", 9), tags=LIVE)
            self._readout(w, h, colour)

        c.delete(SPIN)
        panel, size = self._reactor_box(w, h)
        self._reactor(46 + panel / 2, h / 2 - 10, size, colour)

        if not self.closed:
            self.root.after(FRAME_MS, self._tick)

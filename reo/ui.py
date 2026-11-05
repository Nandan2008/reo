"""A window for reo: a reactor that shows what it is doing, and the conversation.

Tkinter insists on owning the main thread, so the voice loop runs in a worker and
talks to the window through a queue. Nothing here touches Tk from that thread.
"""
import queue
import threading
import tkinter as tk

BACKGROUND = "#070b10"
PANEL = "#0c131b"
DIM = "#2a3a4a"
TEXT = "#c8d6e4"
MUTED = "#55687c"

# What the reactor looks like in each state: colour, and how fast it turns.
STATES = {
    "idle":      ("#2d6a8f", 0.6),
    "listening": ("#19c3d6", 3.0),
    "thinking":  ("#c9922e", 2.0),
    "speaking":  ("#32c57a", 1.2),
}

SIZE = 260          # reactor canvas, square
RINGS = ((108, 5), (84, 3), (60, 2))


class Display:
    def __init__(self, title="REO", subtitle=""):
        self._events = queue.Queue()
        self._state = "idle"
        self._angle = 0.0
        self._pulse = 0.0
        self.closed = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("520x680")
        self.root.minsize(420, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(self.root, text="R E O", bg=BACKGROUND, fg=TEXT,
                 font=("Consolas", 20, "bold")).pack(pady=(18, 0))
        self.subtitle = tk.Label(self.root, text=subtitle, bg=BACKGROUND, fg=MUTED,
                                 font=("Consolas", 9))
        self.subtitle.pack()

        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE, bg=BACKGROUND,
                                highlightthickness=0)
        self.canvas.pack(pady=6)

        self.status = tk.Label(self.root, text="IDLE", bg=BACKGROUND, fg=MUTED,
                               font=("Consolas", 11, "bold"))
        self.status.pack()

        frame = tk.Frame(self.root, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=16, pady=(12, 16))
        self.log = tk.Text(frame, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                           font=("Consolas", 10), wrap="word", padx=12, pady=10,
                           state="disabled", spacing1=2, spacing3=6)
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("you", foreground="#7fa8c9")
        self.log.tag_configure("reo", foreground="#19c3d6")
        self.log.tag_configure("note", foreground=MUTED)

        self._tick()
        self._drain()

    # ---- called from the voice thread -------------------------------------

    def set_state(self, name: str) -> None:
        self._events.put(("state", name, None))

    def line(self, who: str, text: str) -> None:
        self._events.put(("line", who, text))

    def run(self, worker) -> None:
        """Run `worker` in a thread and hand this thread to Tk."""
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.root.mainloop()

    # ---- main thread only --------------------------------------------------

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    def _drain(self) -> None:
        try:
            while True:
                kind, a, b = self._events.get_nowait()
                if kind == "state":
                    self._state = a if a in STATES else "idle"
                    self.status.config(text=self._state.upper(),
                                       fg=STATES[self._state][0])
                else:
                    self._append(a, b)
        except queue.Empty:
            pass
        if not self.closed:
            self.root.after(40, self._drain)

    def _append(self, who: str, text: str) -> None:
        tag = {"you": "you", "reo": "reo"}.get(who, "note")
        label = {"you": "YOU  ", "reo": "REO  "}.get(who, "     ")
        self.log.config(state="normal")
        self.log.insert("end", f"{label}{text}\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _tick(self) -> None:
        import math

        colour, speed = STATES[self._state]
        self._angle = (self._angle + speed * 4) % 360
        self._pulse = (self._pulse + speed * 0.06) % (2 * math.pi)

        c = self.canvas
        c.delete("all")
        mid = SIZE / 2

        for radius, width in RINGS:
            c.create_oval(mid - radius, mid - radius, mid + radius, mid + radius,
                          outline=DIM, width=1)
            # a bright arc sweeping around each ring, each at its own offset
            start = self._angle * (radius / 90.0)
            c.create_arc(mid - radius, mid - radius, mid + radius, mid + radius,
                         start=start, extent=70, style="arc", outline=colour, width=width)

        core = 26 + 5 * math.sin(self._pulse)
        c.create_oval(mid - core, mid - core, mid + core, mid + core,
                      outline=colour, width=2)
        c.create_oval(mid - core / 2.4, mid - core / 2.4, mid + core / 2.4, mid + core / 2.4,
                      fill=colour, outline="")

        if not self.closed:
            self.root.after(33, self._tick)

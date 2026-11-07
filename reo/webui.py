"""reo's window, as a web page in a native frame.

Tkinter has no alpha, no gradients and no blur, so the reactor was concentric
circles pretending to glow. A WebView does have those, and the assistants that
look good use exactly this trick - a local HTML page hosted in a desktop window.

Same interface as the Tk Display, so main() does not care which is in use.
"""
import json
import queue
import threading
import time
from pathlib import Path

import webview

from . import config
from .stats import session as stats

PAGE = Path(__file__).resolve().parent / "hud" / "index.html"
_window = None          # kept out of the exposed object on purpose


class Bridge:
    """What the page may ask for. The page pulls; python never pushes.

    WebView2 only allows calls from its own UI thread, and evaluate_js from a
    worker filled the console with COM errors. Letting the page poll turns that
    round: every call arrives on the thread WebView2 wants.
    """

    def __init__(self):
        # Deliberately no window attribute: pywebview walks every attribute of
        # the object it exposes, and a window here sent it recursing through
        # the native widget tree until the stack ran out.
        self._pending = queue.Queue()
        self._subtitle = ""

    # --- called from the page ---------------------------------------------

    def poll(self):
        events = []
        try:
            while True:
                events.append(self._pending.get_nowait())
        except queue.Empty:
            pass
        return {"events": events, "session": stats.rows(), "device": _device(),
                "subtitle": self._subtitle}

    def toggle_fullscreen(self):
        if _window is not None:
            _window.toggle_fullscreen()

    def quit(self):
        if _window is not None:
            _window.destroy()


class Display:
    def __init__(self, title="REO", subtitle=""):
        self.closed = False
        self.bridge = Bridge()
        self.bridge._subtitle = subtitle
        self.window = webview.create_window(
            title, str(PAGE), width=1280, height=780,
            background_color="#04070b", fullscreen=True, js_api=self.bridge,
        )
        global _window
        _window = self.window
        self.window.events.closed += self._on_closed

    # ---- called from the voice thread -------------------------------------

    def set_state(self, name: str) -> None:
        self.bridge._pending.put(["state", name])

    def line(self, who: str, text: str) -> None:
        self.bridge._pending.put(["line", who, text])

    def run(self, worker) -> None:
        threading.Thread(target=worker, daemon=True).start()
        # http_server serves the page from 127.0.0.1 instead of file://.
        # Over file:// WebView2 never injected window.pywebview, so the page
        # rendered but could never ask python for anything.
        webview.start(http_server=True)   # blocks until the window closes
        self.closed = True

    def _on_closed(self) -> None:
        self.closed = True


def _device():
    """CPU, memory and power, with a percentage for the bar."""
    try:
        import psutil
        battery = psutil.sensors_battery()
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        rows = [["CPU", f"{cpu:.0f}%", cpu], ["MEM", f"{memory:.0f}%", memory]]
        if battery:
            rows.append(["PWR", f"{battery.percent:.0f}%", battery.percent])
        else:
            rows.append(["PWR", "AC"])
        rows.append(["MODEL", config.LLM_MODEL.split("/")[-1]])
        return rows
    except Exception:
        return [["CPU", "--"], ["MEM", "--"], ["PWR", "--"]]

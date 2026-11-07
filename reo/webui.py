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
PUMP_MS = 0.08          # how often queued updates are pushed to the page


class Display:
    def __init__(self, title="REO", subtitle=""):
        self._events = queue.Queue()
        self._subtitle = subtitle
        self._ready = False
        self.closed = False
        self.window = webview.create_window(
            title, str(PAGE), width=1280, height=780,
            background_color="#04070b", fullscreen=True,
        )
        self.window.events.closed += self._on_closed

    # ---- called from the voice thread -------------------------------------

    def set_state(self, name: str) -> None:
        self._events.put(("state", (name,)))

    def line(self, who: str, text: str) -> None:
        self._events.put(("line", (who, text)))

    def run(self, worker) -> None:
        threading.Thread(target=worker, daemon=True).start()
        threading.Thread(target=self._pump, daemon=True).start()
        webview.start()          # blocks until the window closes
        self.closed = True

    # ---- the pump ----------------------------------------------------------

    def _on_closed(self) -> None:
        self.closed = True

    def _call(self, function: str, *args) -> None:
        payload = ", ".join(json.dumps(a) for a in args)
        try:
            self.window.evaluate_js(f"window.reo && reo.{function}({payload})")
        except Exception:
            pass                 # the window went away mid-update

    def _pump(self) -> None:
        """Push queued updates and refresh the figures, off the UI thread."""
        time.sleep(1.2)          # let the page load before talking to it
        if self._subtitle:
            self._call("subtitle", self._subtitle)
        last_stats = 0.0

        while not self.closed:
            try:
                while True:
                    name, args = self._events.get_nowait()
                    self._call(name, *args)
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last_stats > 1.0:
                last_stats = now
                self._call("stats", stats.rows(), _device())

            time.sleep(PUMP_MS)


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

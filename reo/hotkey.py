"""A global key combination that turns listening on and off.

The listener runs on its own thread, so the combination works while any window
has focus - reo does not need to be in front.
"""
import threading

from pynput import keyboard

from . import config


class Toggle:
    """Press once to start listening, press again to stop."""

    def __init__(self, on_change=None):
        self._on = threading.Event()
        self._woke = threading.Event()
        self._on_change = on_change
        self._listener = keyboard.GlobalHotKeys({config.HOTKEY: self._flip})
        self._listener.daemon = True

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    @property
    def listening(self) -> bool:
        return self._on.is_set()

    def _flip(self) -> None:
        if self._on.is_set():
            self._on.clear()
        else:
            self._on.set()
        self._woke.set()          # release anyone waiting, in either direction
        if self._on_change:
            self._on_change(self.listening)

    def wait_until_listening(self) -> None:
        """Block until the combination is pressed to start."""
        while not self._on.is_set():
            self._woke.clear()
            self._woke.wait(timeout=0.5)


def describe(combination: str = None) -> str:
    """Turn pynput's '<ctrl>+<alt>+r' into something worth printing."""
    combination = combination or config.HOTKEY
    return "+".join(part.strip("<>").capitalize() for part in combination.split("+"))

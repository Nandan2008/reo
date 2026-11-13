"""The controls a laptop has that are not keys you can type.

SendKeys cannot press the volume or media keys - it only speaks in characters
and a handful of named keys - so these go through keybd_event with the virtual
key codes Windows actually listens for. Window commands and the lock screen
come from user32 directly for the same reason.

Everything here is reversible. Nothing deletes, nothing spends, nothing closes
a document without the program it belongs to getting a say.
"""
import ctypes

# Virtual key codes. Windows moves the volume two notches per press, so a
# spoken "turn it up" has to repeat before anyone hears a difference.
VK = {
    "mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "next_track": 0xB0, "previous_track": 0xB1,
    "stop": 0xB2, "play_pause": 0xB3,
}
KEYUP = 0x0002

# ShowWindow commands.
SHOW = {"minimise": 6, "minimize": 6, "maximise": 3, "maximize": 3, "restore": 9}

REPEATS = {"volume_up": 5, "volume_down": 5}     # ~10% per spoken request


def _user32():
    return ctypes.windll.user32


def _tap(code: int, times: int = 1) -> None:
    user32 = _user32()
    for _ in range(max(1, times)):
        user32.keybd_event(code, 0, 0, 0)
        user32.keybd_event(code, 0, KEYUP, 0)


def _foreground_title() -> str:
    user32 = _user32()
    handle = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(handle)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value or "that window"


def control(action: str, amount: int = 0) -> str:
    """One spoken command about the machine itself."""
    action = (action or "").strip().lower().replace(" ", "_").replace("-", "_")

    if action in VK:
        times = amount if amount > 0 else REPEATS.get(action, 1)
        _tap(VK[action], times)
        if action == "mute":
            return "Toggled mute."
        if action.startswith("volume"):
            return f"Volume {action.split('_')[1]}, {times} notches."
        return f"Sent {action.replace('_', ' ')}."

    if action in SHOW:
        user32 = _user32()
        handle = user32.GetForegroundWindow()
        if not handle:
            return "No window is in front."
        title = _foreground_title()
        user32.ShowWindow(handle, SHOW[action])
        return f"{action.replace('ise', 'ised').replace('ize', 'ized').capitalize()} {title}."

    if action == "close_window":
        user32 = _user32()
        handle = user32.GetForegroundWindow()
        if not handle:
            return "No window is in front."
        title = _foreground_title()
        # WM_CLOSE asks the program to close, which lets it prompt about
        # unsaved work. Killing the process would not.
        user32.PostMessageW(handle, 0x0010, 0, 0)
        return f"Asked {title} to close."

    if action == "show_desktop":
        user32 = _user32()
        user32.keybd_event(0x5B, 0, 0, 0)        # left Windows key
        _tap(0x44)                                # D
        user32.keybd_event(0x5B, 0, KEYUP, 0)
        return "Showed the desktop."

    if action == "lock":
        _user32().LockWorkStation()
        return "Locked."

    return (f"No such control: {action}. Try volume_up, volume_down, mute, "
            "play_pause, next_track, previous_track, minimise, maximise, "
            "restore, close_window, show_desktop or lock.")

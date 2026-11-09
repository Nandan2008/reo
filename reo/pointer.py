"""The mouse, and an honest account of what reo can see while using it.

reo has no eyes. There is no vision model on this account - the provider was
asked and offered none - so it cannot look at the screen and find a button.
What it can do is move the pointer where it is told and click there, which is
useful when you know the coordinates and useless when you do not.

That limit is the reason play_youtube exists in tools.py: "open the channel
and click the first video" is not a clicking problem once you can work out the
video's address, and guessing at pixels would have been the worse answer.
"""
import ctypes

from . import config


def screen() -> tuple[int, int]:
    """Logical screen size, with Windows' display scaling taken into account.

    Without SetProcessDPIAware a 2560-wide display reports 1707 at 150%
    scaling, and every coordinate reo was given would land short and to the
    left of what the user meant.
    """
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 0, 0


def _controller():
    from pynput.mouse import Controller
    return Controller()


def _check(x: int, y: int) -> str:
    """Refuse a coordinate that is not on the screen rather than clamping it.

    Clamping turns a misheard number into a click somewhere plausible, which
    is worse than not clicking: the corners hold Close and Start.
    """
    width, height = screen()
    if width and not (0 <= x < width and 0 <= y < height):
        return f"({x}, {y}) is off the screen, which is {width} by {height}."
    return ""


def allowed() -> str:
    if not config.ALLOW_MOUSE:
        return "Mouse control is off. Set REO_ALLOW_MOUSE=1 in .env to turn it on."
    return ""


def where() -> str:
    refused = allowed()
    if refused:
        return refused
    width, height = screen()
    x, y = _controller().position
    return f"The pointer is at ({x}, {y}) on a {width} by {height} screen."


def move(x: int, y: int) -> str:
    refused = allowed() or _check(int(x), int(y))
    if refused:
        return refused
    _controller().position = (int(x), int(y))
    return f"Pointer moved to ({int(x)}, {int(y)})."


def click(x=None, y=None, button: str = "left", double: bool = False) -> str:
    from pynput.mouse import Button

    refused = allowed()
    if refused:
        return refused
    pointer = _controller()
    if x is not None and y is not None:
        refused = _check(int(x), int(y))
        if refused:
            return refused
        pointer.position = (int(x), int(y))
    which = {"left": Button.left, "right": Button.right, "middle": Button.middle}
    if button not in which:
        return f"No such mouse button: {button}."
    pointer.click(which[button], 2 if double else 1)
    at = pointer.position
    return f"{'Double-clicked' if double else 'Clicked'} {button} at {at}."


def scroll(amount: int = -3) -> str:
    """Negative scrolls down, the way a wheel turns towards you."""
    refused = allowed()
    if refused:
        return refused
    _controller().scroll(0, int(amount))
    return f"Scrolled {'down' if int(amount) < 0 else 'up'} {abs(int(amount))}."

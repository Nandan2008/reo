"""Putting back the last thing reo did.

Reo is driven by speech, and speech mishears. "Turn it down" comes back as
"turn it up", "minimise this" lands on the wrong window, a file gets written
over the one he wanted. Every one of those is a small thing that is annoying
to put right by hand and trivial to put right here.

Only what can honestly be reversed is recorded. Volume and mute, the media
keys, a window that was minimised or maximised, showing the desktop, and a
file that was written over. Locking the screen and closing a window are not
on the list and never will be: reo cannot unlock anything, and a program
that has closed has already decided what to do about unsaved work.

Anything reo does that cannot be undone clears the record rather than leaving
the one before it. Otherwise "lock the screen" followed by "undo that" would
quietly put the volume back to where it was five minutes earlier, which is
not what anybody meant.
"""

_last = None        # (what it was, how to put it back)
_putting_back = False   # true while an undo is running


def remember(what: str, putting_back) -> None:
    """Record how to reverse what just happened."""
    global _last
    # Undoing is not itself something to undo. Putting the volume back calls
    # control() like anything else, so without this it recorded itself and
    # "undo that" twice turned the volume up again instead of saying there
    # was nothing left to do.
    if _putting_back:
        return
    _last = (what, putting_back)


def forget() -> None:
    """Something happened that cannot be undone."""
    global _last
    _last = None


def waiting() -> str:
    """What "undo that" would put back, or an empty string."""
    return _last[0] if _last else ""


def undo() -> str:
    """Put the last thing back."""
    global _last, _putting_back
    if _last is None:
        return "There is nothing of mine to undo."
    what, putting_back = _last
    # One undo, not a loop: saying it twice must not put it back and then
    # take it away again.
    _last = None
    _putting_back = True
    try:
        said = putting_back()
    except Exception as e:
        return f"Could not undo {what}: {e}"
    finally:
        _putting_back = False
    return said or f"Put {what} back."

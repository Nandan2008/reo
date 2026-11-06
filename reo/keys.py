"""Typing into other programs.

SendKeys goes to whichever window has focus. reo's first attempt sent text into
the void because nothing brought the target window forward first - the user kept
saying "I cannot see it". So focusing is part of the job, not the caller's problem.
"""
import re
import subprocess

TIMEOUT = 20

# SendKeys reads these as instructions, so they have to be wrapped to be typed.
SPECIAL = "+^%~(){}[]"

PS_LIST = r"""
Get-Process | Where-Object { $_.MainWindowTitle } |
  Select-Object -ExpandProperty MainWindowTitle
"""

PS_TYPE = r"""
Add-Type -AssemblyName System.Windows.Forms
$all = @(Get-Process | Where-Object { $_.MainWindowTitle -like $env:REO_WINDOW })
if ($all.Count -eq 0) { Write-Output 'NOWINDOW'; exit }
# An exact title wins outright. Otherwise refuse to choose between several:
# guessing once typed into the wrong Notepad tab, and once into an audio panel.
$exact = @($all | Where-Object { $_.MainWindowTitle -eq $env:REO_EXACT })
$target = if ($exact.Count -eq 1) { $exact[0] }
          elseif ($all.Count -eq 1) { $all[0] }
          else { $null }
if (-not $target) {
  Write-Output ('MANY:' + (($all | ForEach-Object { $_.MainWindowTitle }) -join '|'))
  exit
}
$shell = New-Object -ComObject WScript.Shell
$null = $shell.AppActivate($target.Id)
Start-Sleep -Milliseconds 400
[System.Windows.Forms.SendKeys]::SendWait($env:REO_TEXT)
Write-Output ('TYPED:' + $target.MainWindowTitle)
"""


def _powershell(script: str, env: dict = None) -> str:
    import os

    environment = dict(os.environ)
    if env:
        environment.update(env)
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=TIMEOUT, env=environment,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    return (done.stdout or done.stderr or "").strip()


def escape(text: str) -> str:
    """Wrap the characters SendKeys would otherwise act on."""
    return "".join("{" + c + "}" if c in SPECIAL else c for c in text)


def list_windows() -> str:
    """Titles of the windows currently open, so a target can be named."""
    titles = [t for t in _powershell(PS_LIST).splitlines() if t.strip()]
    return "\n".join(titles) if titles else "No windows with titles were found."


# SendKeys names for keys that are not characters, plus the modifier prefixes.
KEYS = {
    "enter": "{ENTER}", "return": "{ENTER}", "send": "{ENTER}",
    "tab": "{TAB}", "escape": "{ESC}", "esc": "{ESC}",
    "backspace": "{BACKSPACE}", "delete": "{DEL}", "del": "{DEL}",
    "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
    "home": "{HOME}", "end": "{END}", "pageup": "{PGUP}", "pagedown": "{PGDN}",
    "space": " ",
}
MODIFIERS = {"ctrl": "^", "control": "^", "alt": "%", "shift": "+"}


def keystroke(key: str) -> str:
    """Turn "enter" or "ctrl+a" into what SendKeys expects, or "" if unknown."""
    parts = [p.strip().lower() for p in key.split("+") if p.strip()]
    if not parts:
        return ""
    prefix = "".join(MODIFIERS[p] for p in parts[:-1] if p in MODIFIERS)
    if len(prefix) != len(parts) - 1:
        return ""                        # an unknown modifier
    last = parts[-1]
    if last in KEYS:
        return prefix + KEYS[last]
    if len(last) == 1 and last.isalnum():
        return prefix + last
    if re.fullmatch(r"f\d{1,2}", last):
        return prefix + "{" + last.upper() + "}"
    return ""


def press_key(key: str, window: str = None) -> str:
    """Press a key such as enter, tab or ctrl+a, in `window`."""
    stroke = keystroke(key)
    if not stroke:
        return (f"{key!r} is not a key reo knows. Try enter, tab, escape, up, down, "
                "delete, or a combination like ctrl+a.")
    return _send(stroke, window, f"Pressed {key} in")


def type_text(text: str, window: str = None, enter: bool = False) -> str:
    """Bring `window` to the front, type `text`, and optionally press enter."""
    if not text:
        return "Nothing to type."
    doing = "Typed it and pressed enter in" if enter else "Typed it into"
    return _send(escape(text) + ("{ENTER}" if enter else ""), window, doing)


def _send(keys: str, window: str, doing: str) -> str:
    if not window:
        return "Say which window. Open windows are:\n" + list_windows()

    result = _powershell(PS_TYPE, {"REO_WINDOW": f"*{window}*",
                                   "REO_EXACT": window,
                                   "REO_TEXT": keys})

    if result == "TIMEOUT":
        return "That timed out."
    if result == "NOWINDOW":
        return f"No open window matches {window!r}. Open windows are:\n{list_windows()}"
    if result.startswith("MANY:"):
        return ("Several windows match that, so nothing was sent. Ask the user which:\n"
                + "\n".join(result[5:].split("|")))
    if result.startswith("TYPED:"):
        return f'{doing} "{result[6:]}".'
    return f"It may have failed: {result[:200]}"

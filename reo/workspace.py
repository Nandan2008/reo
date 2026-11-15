"""A folder reo is allowed to write into.

Asked to write a program, reo could do everything except the actual writing:
it had no way to put anything on disk. It could only say the code out loud,
one sentence at a time, which is no use to anybody.

So it gets a desk. One folder, in plain sight on the Desktop, that it can
write files into, read them back and open them. Everything is kept inside
that folder: a name that would land anywhere else is refused rather than
tidied up, because the names come from speech and speech mishears.

Nothing here runs anything. Writing a .py file and running a .py file are
very different powers, and the second one lives behind REO_ALLOW_SHELL.
"""
import os
import re
from pathlib import Path

from . import config

# What a file may be called. Letters, digits, dot, dash, underscore, and a
# forward slash for a subfolder. Not a colon (drive letters), not a backslash,
# and never two dots in a row.
NAME = re.compile(r"[A-Za-z0-9._/-]+")
BIGGEST = 400_000        # a quarter of a megabyte of text is a very long file


def desk() -> Path:
    """The folder, made if it is not there yet."""
    place = Path(config.WORKSPACE).expanduser()
    place.mkdir(parents=True, exist_ok=True)
    return place


def _inside(name: str) -> Path:
    """Where that name lands, or a refusal if it lands outside the folder."""
    wanted = (name or "").strip().replace("\\", "/")
    if not wanted:
        raise ValueError("Say what the file should be called.")
    # A leading slash is refused rather than stripped. Stripping it turned
    # "/etc/passwd" into a file called etc/passwd on the desk, which stays
    # inside the folder but is not what anybody meant, and quietly doing
    # something else with a name is how a misunderstanding becomes a file.
    if wanted.startswith("/") or ":" in wanted:
        raise ValueError(f"{name!r} is a path, not a name for a file on the desk.")
    if not NAME.fullmatch(wanted) or ".." in wanted:
        raise ValueError(f"{name!r} is not a usable file name.")
    home = desk().resolve()
    landed = (home / wanted).resolve()
    # The real check. Everything above is a courtesy; this is the one that
    # matters, and it is done after resolving so links cannot step around it.
    if landed != home and home not in landed.parents:
        raise ValueError(f"{name!r} would end up outside {home}.")
    return landed


def write(name: str, content: str) -> str:
    """Put a file on the desk, keeping one copy of whatever was there."""
    try:
        path = _inside(name)
    except ValueError as e:
        return str(e)
    if content is None:
        content = ""
    if len(content) > BIGGEST:
        return f"That is {len(content):,} characters; {BIGGEST:,} is the limit."

    replaced = path.exists()
    if replaced:
        # Speech mishears, and a misheard name should not cost him an hour's
        # work. One copy back is enough to undo it.
        try:
            path.with_suffix(path.suffix + ".last").write_bytes(path.read_bytes())
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Could not write it: {e}"

    lines = content.count("\n") + 1 if content else 0
    where = path.relative_to(desk().resolve())
    return (f"{'Replaced' if replaced else 'Wrote'} {where} "
            f"({lines} lines) in {desk()}."
            + (" The previous one is beside it, ending .last." if replaced else ""))


def read(name: str) -> str:
    try:
        path = _inside(name)
    except ValueError as e:
        return str(e)
    if not path.is_file():
        return f"There is no {name} on the desk."
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Could not read it: {e}"
    if len(text) > 40_000:
        return text[:40_000] + "\n... (cut off here)"
    return text or "(that file is empty)"


def listing() -> str:
    home = desk()
    found = sorted(p for p in home.rglob("*") if p.is_file())
    if not found:
        return f"The desk is empty. It is at {home}."
    rows = [f"{p.relative_to(home)}  ({p.stat().st_size:,} bytes)" for p in found[:60]]
    more = f"\n... and {len(found) - 60} more" if len(found) > 60 else ""
    return f"In {home}:\n" + "\n".join(rows) + more


def show(name: str) -> str:
    """Open it with whatever Windows uses for that kind of file."""
    try:
        path = _inside(name)
    except ValueError as e:
        return str(e)
    if not path.is_file():
        return f"There is no {name} on the desk."
    try:
        os.startfile(str(path))      # noqa: S606 - a file reo itself just wrote
    except OSError as e:
        return f"Could not open it: {e}"
    return f"Opened {path.name}."

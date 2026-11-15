"""Answering the easy questions without waking the model at all.

Asked the time, reo was making a round trip to Groq, a tool call, and a second
round trip - about two seconds for something the machine knows instantly. The
pattern comes from other assistants: route deterministically first, and only pay
for the model when the phrasing is actually fuzzy.

Every pattern here matches the WHOLE utterance. A keyword search would hijack
"what time is it in Tokyo" or "how is my battery life compared to last year",
which genuinely need the model. Anything not matched falls through untouched.
"""
import ast
import datetime
import operator
import re

from . import undo as undoing
from .text import normalize

# Spoken arithmetic, before the expression is parsed.
WORD_OPERATORS = [
    (r"\bplus\b|\badd(ed to)?\b", "+"),
    (r"\bminus\b|\bsubtract(ed from)?\b|\btake away\b", "-"),
    (r"\btimes\b|\bmultiplied by\b|\bmultiply by\b|\bx\b", "*"),
    (r"\bdivided by\b|\bdivide by\b|\bover\b", "/"),
    (r"\bto the power of\b", "**"),
]

SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _evaluate(node):
    """Arithmetic only. No names, no calls, no attributes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("not arithmetic")


def _tidy(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, float):
        value = round(value, 4)
    return f"{value:,}"


# --- the answers ------------------------------------------------------------

def _time(_):
    return datetime.datetime.now().strftime("It is %I:%M %p.").replace(" 0", " ")


def _date(_):
    return datetime.datetime.now().strftime("It is %A, %d %B %Y.")


def _battery(_):
    import psutil
    battery = psutil.sensors_battery()
    if battery is None:
        return "This machine has no battery."
    state = "charging" if battery.power_plugged else "on battery"
    return f"{battery.percent:.0f} percent, {state}."


def _memory(_):
    import psutil
    used = psutil.virtual_memory()
    return f"{used.percent:.0f} percent of memory in use, {used.available / 2**30:.1f} gigabytes free."


def _disk(_):
    import psutil
    free = psutil.disk_usage("C:\\").free / 2**30
    return f"{free:.0f} gigabytes free on the C drive."


def _uptime(_):
    import psutil
    since = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    hours, minutes = divmod(int(since.total_seconds()) // 60, 60)
    if hours:
        return f"Up for {hours} hours and {minutes} minutes."
    return f"Up for {minutes} minutes."


def _percent_of(match):
    return f"{_tidy(float(match['part']) / 100 * float(match['whole']))}."


def _square_root(match):
    return f"{_tidy(float(match['n']) ** 0.5)}."


def _arithmetic(match):
    expression = match["expr"]
    for pattern, symbol in WORD_OPERATORS:
        expression = re.sub(pattern, symbol, expression)
    if not re.fullmatch(r"[\d\s+\-*/%.()]+", expression):
        return None
    try:
        return f"{_tidy(_evaluate(ast.parse(expression, mode='eval').body))}."
    except Exception:
        return None



# --- the things that do something -------------------------------------------
#
# "Volume up" was costing a round trip to the model, a tool call, and a second
# round trip - about two and a half seconds to press a key this machine can
# press itself, and the tokens of a 1200 token prompt against an 8000 token
# minute. The same for "open chrome" and "lock the laptop", which are among
# the things said most often.
#
# Deliberately limited. Everything here is reversible: volume, media, windows,
# the lock screen, a program, a website. Nothing deletes, spends, types or
# moves the mouse - if it is not on this list the model decides, as before.
# Anything that does not come off cleanly returns None, so the model still
# gets its turn rather than reo reading out a failure.
#
# Remember normalize() has already been through the words: there is no
# punctuation left, so "youtube.com" arrives as "youtube com".

# Words that mean the thing after "open" is not a program. Without these,
# "open the first result" tried to launch a program called "the first result".
NOT_AN_APP = {"the", "a", "an", "my", "this", "that", "it", "them", "up", "all",
              "first", "second", "third", "last", "next", "new", "another",
              "one", "two", "again", "and", "some", "any", "me"}

# Endings worth treating as a web address. Not "in", "co", "me", "ai" or "app":
# they are ordinary words, and "open task manager in" is not a website.
TLD = "com|org|net|io|tv|gov|edu"


def _control(action: str, amount: int = 0):
    from . import system
    return system.control(action, amount)


def _by(match) -> int:
    """How many notches he asked for, if he said."""
    asked = match.groupdict().get("by")
    return int(asked) if asked else 0


def _louder(match):
    return _control("volume_up", _by(match))


def _quieter(match):
    return _control("volume_down", _by(match))


def _either_way(match):
    return (_louder if match["way"] == "up" else _quieter)(match)


def _window(match):
    return _control(match["what"].replace("z", "s"))


def _open_app(match):
    """Open a program, or hand it back to the model if the name was no good."""
    name = match["app"].strip()
    if not name or name.split()[0] in NOT_AN_APP:
        return None
    from .apps import open_app
    said = open_app(name)
    if said.startswith("Opened ") or " was already open" in said:
        return said
    # Not installed, several matches, or a timeout. The model can look at the
    # list and work out what he meant; reading the failure out loud is the
    # worst of both.
    return None


def _open_site(match):
    from .tools import open_website
    host = match["site"].replace(" ", "")
    return open_website(f"{host}.{match['tld']}")


def _youtube(match):
    found = match.groupdict()
    what = (found.get("what") or found.get("what2") or "").strip()
    if not what or what in NOT_AN_APP:
        return None
    from .tools import play_youtube
    said = play_youtube(what)
    return said if said.startswith(("Playing", "Opened")) else None


# Ordered, and every one matches the whole utterance or not at all.
DOES = [
    # the laptop's own controls
    (r"(turn|put|make) (it|this|that|the volume|the sound) (?P<way>up|down)"
     r"( a bit| a little| slightly)?( by (?P<by>\d+)( notches| steps| percent)?)?",
     _either_way),
    (r"(turn )?(the )?(volume|sound) (?P<way>up|down)"
     r"( a bit| a little)?( by (?P<by>\d+)( notches| steps| percent)?)?",
     _either_way),
    (r"(turn |make )?(it |that |the volume )?louder( by (?P<by>\d+))?", _louder),
    (r"(turn |make )?(it |that |the volume )?(quieter|softer)( by (?P<by>\d+))?",
     _quieter),
    (r"(un)?mute( it| the (volume|sound|audio))?", lambda m: _control("mute")),
    (r"(play|pause|resume)( it| that| the (song|music|video|track))?|play pause",
     lambda m: _control("play_pause")),
    (r"(skip|next)( it| this)?( (track|song|video|one))?|skip (forward|ahead)",
     lambda m: _control("next_track")),
    (r"(previous|last|go back a) (track|song|video|one)",
     lambda m: _control("previous_track")),
    (r"(?P<what>minimi[sz]e|maximi[sz]e|restore)( (it|this|that|the))?( window)?",
     _window),
    (r"show (me )?(the )?desktop|minimi[sz]e everything|hide everything",
     lambda m: _control("show_desktop")),
    (r"close (it|this|that|the)( window)?|close (the |this )?window",
     lambda m: _control("close_window")),
    (r"lock (it|up|the (laptop|screen|computer|pc|machine)"
     r"|my (laptop|screen|computer|pc))", lambda m: _control("lock")),
    # putting the last thing back
    (r"undo( that| it| the last( thing| one)?)?"
     r"|(can you |please )?(undo|reverse) (that|it|the last (thing|one))"
     r"|put (that|it) back|take (that|it) back|reverse that",
     lambda m: undoing.undo()),
    # opening things
    (r"(open|go to|visit|bring up) (?P<site>[a-z0-9][a-z0-9 -]*) (?P<tld>" + TLD + r")",
     _open_site),
    (r"(open|launch|start|run|bring up) (?P<app>[a-z0-9][a-z0-9 ]*)", _open_app),
    (r"(play|put on) (?P<what>.+) on youtube"
     r"|(search|look up|find) (?P<what2>.+) on youtube and play"
     r"( the (first|1st)( video| one)?)?", _youtube),
]


# Ordered; the first whole-utterance match wins.
ROUTES = [
    (r"(what(s| is)? (the )?)?time( is it)?( now| right now| please)?", _time),
    (r"(what(s| is)? (the |todays? )?)?(date|day)( is it)?( today)?", _date),
    (r"(what(s| is)? (my |the )?)?battery( level| percentage| status)?"
     r"|how much battery( do i have| is left)?|am i charging", _battery),
    (r"(how much |whats my |what is my )?(ram|memory)( am i using| usage| is used)?", _memory),
    (r"(how much )?(disk|storage|free) space( do i have| is left| on c)?"
     r"|how much space( is left| do i have)?", _disk),
    (r"(whats |what is |how long is )?(my |the )?uptime"
     r"|how long (has|have) (my |this )?(computer|laptop|pc) been (on|running|up)", _uptime),
    (r"(whats |what is |calculate )?(?P<part>[\d.]+) ?(percent|%) of (?P<whole>[\d.]+)", _percent_of),
    (r"(whats |what is )?(the )?square root of (?P<n>[\d.]+)", _square_root),
    (r"(whats |what is |calculate )(?P<expr>[\d.]+(?:\s+(?:plus|minus|times|multiplied by|"
     r"divided by|over|x|\+|-|\*|/)\s+[\d.]+)+)", _arithmetic),
]

# Questions first, then the things that act: a question about the volume
# should not change it.
COMPILED = [(re.compile(pattern + r"$"), answer)
            for pattern, answer in ROUTES + DOES]


def answer(text: str):
    """A spoken answer if this is a question the machine can answer itself,
    or None to let the model have it."""
    asked = normalize(text)
    if not asked:
        return None
    for pattern, respond in COMPILED:
        match = pattern.fullmatch(asked)
        if match:
            try:
                return respond(match)
            except Exception:
                return None          # fall through to the model rather than fail
    return None

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

COMPILED = [(re.compile(pattern + r"$"), answer) for pattern, answer in ROUTES]


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

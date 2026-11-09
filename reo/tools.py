"""Actions reo can take. Only whitelisted, non-destructive actions for now."""
import datetime
import json
import re
import subprocess
import urllib.parse
import webbrowser

import psutil

from . import pointer

# Spoken name -> Windows command. Only these can be launched.
APPS = {
    "notepad": "notepad",
    "calculator": "calc",
    "file explorer": "explorer",
    "explorer": "explorer",
    "chrome": "chrome",
    "edge": "msedge",
    "vs code": "code",
    "vscode": "code",
    "settings": "ms-settings:",
    "task manager": "taskmgr",
    "paint": "mspaint",
}


def open_app(name: str) -> str:
    from .apps import open_app as _open
    return _open(name)


def list_apps() -> str:
    from .apps import list_apps as _list
    return _list()


def open_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}."


_searcher = None
_search_cache = {}

# Wikipedia refuses a request that will not say who it is, and reo was getting
# a flat 403 from the best reference on the web. Their policy asks for a
# descriptive agent, so give one rather than dressing up as somebody's browser.
AGENT = "reo-voice-assistant/1.0 (open-source personal voice assistant)"


def _client(timeout: int = 15):
    import primp
    return primp.Client(timeout=timeout, headers={"User-Agent": AGENT})


def warm_up() -> None:
    """Build the search client before anyone asks for it.

    The first search of a session took 5.3s against 1.4s for the next, and all
    of that difference was construction. Called on a background thread at
    startup, it is paid while the speech model is still loading.
    """
    global _searcher
    try:
        from ddgs import DDGS
        _searcher = DDGS()
    except Exception as e:
        print(f"  [search] warm-up skipped: {e}")


def search_web(query: str) -> str:
    """Look something up and hand the text back, so reo can answer out loud."""
    global _searcher

    key = query.strip().lower()
    if key in _search_cache:      # the model rewords and retries; don't pay twice
        return _search_cache[key]

    try:
        if _searcher is None:
            from ddgs import DDGS
            _searcher = DDGS()    # building one per search cost about 1.3s each time
        results = _searcher.text(query, max_results=4)
    except Exception as e:
        return f"Could not search: {e}"

    if not results:
        return f"No results for {query}."
    answer = "\n".join(f"{r['title']}: {r['body']}" for r in results)
    _search_cache[key] = answer
    return answer


PAGE_LIMIT = 3000   # a voice answer needs the gist, not the whole page


def read_page(url: str) -> str:
    """Fetch a page and return its text, for when a search snippet is not enough."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        response = _client().get(url)
        if response.status_code >= 400:
            return f"{url} returned HTTP {response.status_code}."
        text = " ".join((response.text_plain or "").split())
    except Exception as e:
        return f"Could not read {url}: {e}"
    if not text:
        return f"{url} had no readable text."
    return text[:PAGE_LIMIT] + (" ...(truncated)" if len(text) > PAGE_LIMIT else "")


WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_SEARCH = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
               "&format=json&srlimit=4&srsearch=")


def look_up(topic: str) -> str:
    """Wikipedia, through its own API rather than by scraping the page.

    Faster than a search and a page read together - about 0.3s - and it comes
    back as a written summary instead of navigation furniture.
    """
    import json as _json

    def summary(title: str):
        response = _client(10).get(WIKI_SUMMARY + urllib.parse.quote(title.replace(" ", "_")))
        if response.status_code != 200:
            return None
        return _json.loads(response.text)

    page = summary(topic)
    # No exact title, or a title that only lists the possibilities: ask
    # Wikipedia which article was meant. The first hit for an ambiguous word is
    # usually the disambiguation page itself, so skip anything that leads back
    # where we started - "Mercury" has to reach the planet, not the list.
    if page is None or page.get("type") == "disambiguation":
        rejected = (page or {}).get("title", "").lower()
        try:
            found = _json.loads(_client(10).get(WIKI_SEARCH + urllib.parse.quote(topic)).text)
            hits = found.get("query", {}).get("search", [])
        except Exception:
            hits = []
        for hit in hits:
            if hit["title"].lower() == rejected:
                continue
            candidate = summary(hit["title"])
            if candidate and candidate.get("type") != "disambiguation"                     and candidate.get("extract"):
                page = candidate
                break
    if not page or not page.get("extract") or page.get("type") == "disambiguation":
        return f"Wikipedia has no single article on {topic}."
    return f"{page.get('title', topic)}: {page['extract']}"


def remember(fact: str) -> str:
    from .memory import memory
    return memory.remember(fact)


def forget(about: str) -> str:
    from .memory import memory
    return memory.forget(about)


def recall(about: str = "") -> str:
    from .memory import memory
    return memory.recall(about)


YOUTUBE_SEARCH = "https://www.youtube.com/results?search_query="
WATCH = "https://www.youtube.com/watch?v="
_VIDEO_ID = re.compile(r'"videoId":"([\w-]{11})"')
_TITLE = re.compile(r'"title":\{"runs":\[\{"text":"(.*?)"\}')


def play_youtube(query: str, position: int = 1) -> str:
    """Open a video from a YouTube search, by its address rather than a click.

    "Search for the channel and click the first video" looks like it needs a
    mouse and eyes, and reo has neither. It does not: the search page carries
    every result's id, so the nth video has a knowable address and can simply
    be opened. Deterministic, and it cannot click the wrong thing.
    """
    try:
        page = _client().get(YOUTUBE_SEARCH + urllib.parse.quote_plus(query))
    except Exception as e:
        return f"Could not reach YouTube: {e}"
    if page.status_code != 200:
        return f"YouTube returned HTTP {page.status_code}."

    found = list(dict.fromkeys(_VIDEO_ID.findall(page.text)))
    if not found:
        return f"No videos found for {query}."
    wanted = max(1, int(position or 1))
    if wanted > len(found):
        return f"Only found {len(found)} videos for {query}."
    video = found[wanted - 1]

    name = ""
    titles = _TITLE.findall(page.text)
    if len(titles) >= wanted:
        try:    # the page stores titles JSON-escaped
            name = json.loads(f'"{titles[wanted - 1]}"')
        except ValueError:
            name = titles[wanted - 1]

    webbrowser.open(WATCH + video)
    which = "first" if wanted == 1 else f"number {wanted}"
    return f"Playing the {which} result{f': {name}' if name else ''}."


def open_search(query: str) -> str:
    """Put the results on screen when the user wants to look themselves."""
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(query))
    return f"Opened a search for {query}."


def get_datetime() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def get_battery() -> str:
    b = psutil.sensors_battery()
    if b is None:
        return "No battery info available."
    return f"{b.percent:.0f}% ({'charging' if b.power_plugged else 'on battery'})"


def type_text(text: str, window: str = None, enter: bool = False) -> str:
    from .keys import type_text as _type
    return _type(text, window, enter)


def press_key(key: str, window: str = None) -> str:
    from .keys import press_key as _press
    return _press(key, window)


def list_windows() -> str:
    from .keys import list_windows as _list
    return _list()


def run_command(command: str, confirmed: bool = False) -> str:
    from .shell import run_command as _run
    return _run(command, confirmed)


FUNCTIONS = {
    "open_app": open_app,
    "list_apps": list_apps,
    "open_website": open_website,
    "search_web": search_web,
    "look_up": look_up,
    "play_youtube": play_youtube,
    "move_cursor": lambda x, y: pointer.move(x, y),
    "click_mouse": lambda x=None, y=None, button="left", double=False: pointer.click(
        x, y, button, double),
    "scroll_mouse": lambda amount=-3: pointer.scroll(amount),
    "cursor_position": lambda: pointer.where(),
    "remember": remember,
    "forget": forget,
    "recall": recall,
    "open_search": open_search,
    "read_page": read_page,
    "run_command": run_command,
    "type_text": type_text,
    "press_key": press_key,
    "list_windows": list_windows,
    "get_datetime": get_datetime,
    "get_battery": get_battery,
}

TOOLS = [
    {
        "name": "open_app",
        "description": ("Open a program, or bring it forward if already running. Use this "
                        "rather than hunting for an .exe, which misses Store apps."),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "e.g. 'notepad', 'claude'"}},
            "required": ["name"],
        },
    },
    {
        "name": "list_apps",
        "description": "List the programs Windows can launch, to find the right name.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "open_website",
        "description": ("Show the user a website in their browser. To read a page yourself, "
                        "use read_page."),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "e.g. youtube.com"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": ("Look up current information on the web: news, prices, who holds an "
                        "office now."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "type_text",
        "description": ("Type into another program, bringing that window to the front first. "
                        "Name the window loosely: 'claude', 'notepad'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to type"},
                "window": {"type": "string",
                           "description": "Part of the target window's title"},
                "enter": {"type": "boolean",
                          "description": ("True to press enter after typing, which "
                                          "sends the message in a chat or search box. "
                                          "Use it when the user says send, or enter it.")},
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": ("Press a key in another window: enter, tab, escape, arrows, ctrl+a. "
                        "Use it to send something already typed."),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "e.g. 'enter', 'ctrl+a'"},
                "window": {"type": "string",
                           "description": "Part of the target window's title"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "list_windows",
        "description": "List the titles of open windows, to find what to type into.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_command",
        "description": ("Run a PowerShell command: files, settings, programs, system info. "
                        "Reads run; destructive ones come back asking to be confirmed."),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The PowerShell command"},
                "confirmed": {"type": "boolean",
                              "description": "True only after the user said yes out loud"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_page",
        "description": ("Read a web page's text, when search snippets lack the answer."),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to read"}},
            "required": ["url"],
        },
    },
    {
        "name": "play_youtube",
        "description": "Search YouTube and open a result. Use instead of trying to click.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "position": {"type": "integer", "description": "1 = first"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cursor_position",
        "description": "Pointer position and screen size.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "move_cursor",
        "description": "Move the pointer to a coordinate.",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
    {
        "name": "click_mouse",
        "description": "Click, moving there first. Only coordinates you were given; never guess.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"}, "y": {"type": "integer"},
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "double": {"type": "boolean"},
            },
        },
    },
    {
        "name": "scroll_mouse",
        "description": "Scroll the wheel. Negative is down.",
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
        },
    },
    {
        "name": "look_up",
        "description": "Wikipedia. Prefer over search_web for facts that do not change.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "remember",
        "description": "Store a fact about the user for next time.",
        "input_schema": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    },
    {
        "name": "recall",
        "description": ("Only for a fact NOT in the remembered list you were already given. Never for one that is."),
        "input_schema": {
            "type": "object",
            "properties": {"about": {"type": "string"}},
            "required": ["about"],
        },
    },
    {
        "name": "forget",
        "description": "Drop remembered facts matching this.",
        "input_schema": {
            "type": "object",
            "properties": {"about": {"type": "string"}},
            "required": ["about"],
        },
    },
    {
        "name": "open_search",
        "description": "Open a web search in the browser for the user to read themselves.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_battery",
        "description": "Get the laptop battery level and charging status.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, args: dict) -> str:
    fn = FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool {name}."
    try:
        return fn(**args)
    except Exception as e:  # report failures to the model instead of crashing
        return f"Tool {name} failed: {e}"

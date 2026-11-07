"""Actions reo can take. Only whitelisted, non-destructive actions for now."""
import datetime
import subprocess
import urllib.parse
import webbrowser

import psutil

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
        import primp
        response = primp.Client(timeout=15).get(url)
        if response.status_code >= 400:
            return f"{url} returned HTTP {response.status_code}."
        text = " ".join((response.text_plain or "").split())
    except Exception as e:
        return f"Could not read {url}: {e}"
    if not text:
        return f"{url} had no readable text."
    return text[:PAGE_LIMIT] + (" ...(truncated)" if len(text) > PAGE_LIMIT else "")


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

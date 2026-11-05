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
    target = APPS.get(name.lower().strip())
    if not target:
        return f"Not allowed or unknown app '{name}'. Allowed: {', '.join(sorted(APPS))}"
    subprocess.Popen(f'start "" {target}', shell=True)  # target comes only from the whitelist
    return f"Opened {name}."


def open_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}."


def search_web(query: str) -> str:
    """Look something up and hand the text back, so reo can answer out loud."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Could not search: {e}"
    if not results:
        return f"No results for {query}."
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)


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


def run_command(command: str, confirmed: bool = False) -> str:
    from .shell import run_command as _run
    return _run(command, confirmed)


FUNCTIONS = {
    "open_app": open_app,
    "open_website": open_website,
    "search_web": search_web,
    "open_search": open_search,
    "read_page": read_page,
    "run_command": run_command,
    "get_datetime": get_datetime,
    "get_battery": get_battery,
}

TOOLS = [
    {
        "name": "open_app",
        "description": "Open an application on the laptop. Allowed: " + ", ".join(sorted(APPS)),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "App name, e.g. 'notepad'"}},
            "required": ["name"],
        },
    },
    {
        "name": "open_website",
        "description": ("Show a website to the user in their browser. This puts a window on "
                        "their screen, so only use it when they asked to see something. "
                        "To read a page yourself, use read_page instead."),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "e.g. youtube.com"}},
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": ("Look up current information on the web and read the results. "
                        "Use this for anything you are unsure of or that may have changed: "
                        "news, prices, scores, dates, who holds an office now."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_command",
        "description": ("Run a PowerShell command on this Windows machine. Use it for "
                        "anything the other tools do not cover: files, folders, settings, "
                        "programs, system information. Prefer a specific tool when one "
                        "fits. Some commands are refused outright and some must be "
                        "confirmed by the user first; the result will tell you which."),
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
        "description": ("Read the text of a web page. Use it after search_web when the "
                        "snippets do not contain the answer, such as a live price or a "
                        "detail buried in an article."),
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
